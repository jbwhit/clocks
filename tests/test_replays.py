"""The committed replay archives: invariants on the files, and the gate.

The invariants read only the committed files. The gate re-records each demo
and requires its inputs to match the archive: the true parameters and (for
model comparison) model orders and prior exactly, the computed clock positions,
rates and observations at ``rtol=1e-12, atol=1e-12``. For the six filter
demos it also requires the posterior summaries after observations 1, 8 and the
last to match within ``BOUNDS`` (``DENSITY_BOUNDS`` for density); for
``demo_model_comparison`` it checks the three conditions on the model
probabilities instead.

After an intended change to a demo's inputs or inference, re-record its
archive with ``uv run python scripts/record_replays.py --out assets/replays
--overwrite <demo>`` and commit the file.
"""

import functools
import importlib
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.special import logsumexp

from clocks._demos.demo_echolocation_3d import close_view
from clocks._replay import DEMOS, PARAM_NAMES, Replay, Scenario, load_replay
from clocks._scenarios import contrast_matrix
from clocks.physics import clock_rates_batch_multi
from clocks.types import ClockArray

ROOT = Path(__file__).parents[1]
ARCHIVES = ROOT / "assets/replays"

FILTER_DEMOS = tuple(name for name in DEMOS if name != "demo_model_comparison")
PREDICTING_DEMOS = ("demo_1d", "demo_multi_mass", "demo_echolocation_3d")
SLOW_DEMOS = {"demo_multi_mass_2d", "demo_density", "demo_echolocation_3d"}

# (mean positions, mean masses, sd positions, sd masses) after observation 1, 8, last
BOUNDS = {
    "demo_1d": (
        (0.095, 0.0025, 0.063, 0.0027),
        (0.024, 0.00091, 0.015, 0.00079),
        (0.0085, 0.00034, 0.0045, 0.00026),
    ),
    "demo_2d": (
        (0.48, 0.0054, 0.28, 0.0020),
        (0.11, 0.00059, 0.062, 0.00052),
        (0.025, 0.00025, 0.018, 0.00014),
    ),
    "demo_multi_mass": (
        (0.87, 0.0095, 0.19, 0.0065),
        (0.060, 0.0013, 0.063, 0.00079),
        (0.0079, 0.00015, 0.0058, 0.00012),
    ),
    "demo_multi_mass_2d": (
        (1.6, 0.011, 0.85, 0.0071),
        (4.2, 0.026, 2.9, 0.0096),
        (0.46, 0.0027, 0.47, 0.0014),
    ),
    "demo_echolocation_3d": (
        (0.28, 0.021, 0.043, 0.0029),
        (0.012, 0.00077, 0.0097, 0.00059),
        (0.0073, 0.00047, 0.0044, 0.00027),
    ),
}
# density: (mean μ, σ, A), (sd μ, σ, A) after observation 1, 8, last
DENSITY_BOUNDS = (
    ((0.49, 0.38, 0.00050), (0.38, 0.15, 0.00069)),
    ((0.067, 0.22, 0.0016), (0.039, 0.20, 0.0031)),
    ((0.019, 0.027, 0.00011), (0.014, 0.021, 0.000083)),
)

# The dots' predicted readings against the recorded ones, per row and clock: the
# mean within this many predicted s.d. / sqrt(m), the s.d. within a ratio range.
# Measured on the committed archives, worst over every row and clock: mean 3.39
# (echolocation row 34), 2.82 (demo_1d row 7), 2.47 (demo_multi_mass row 40);
# ratio 0.892 to 1.109 in every row but one.
PREDICTION_MEAN_BOUND = 6.0
PREDICTION_SPREAD = (0.6, 1.5)
# Echolocation's prior row is not checked at all, and its dots are not passed
# to the forward model. The row is not drawn: the observed-against-fitted panel
# is empty at k = 0. One prior particle, inside the head, carries 99.8% of the
# centre clock's predicted variance, and whether a display sample holds it is
# chance, so neither the spread (0.019 here; 2.82 and 0.008 in the reviewer's
# recordings with 5,999 and 6,001 particles) nor the mean (1.49 here; 6.59 in
# one with 23,088 particles) describes the archive there. Skipping the row also
# avoids an edge: a valid float64 particle can cross the weak-field limit once
# rounded to float32, which would make the forward model raise at that dot. The
# committed dots after row 0 have margin; a re-recording that hits it must
# account for the rounding in the evaluation here. demo_1d and demo_multi_mass
# keep row 0: their panels draw the prior's predicted bars.
FIRST_CHECKED_ROW = {"demo_echolocation_3d": 1}
# The demos whose panels draw the prior's predicted bars, so row 0 is checked.
PRIOR_CHECKED_DEMOS = ("demo_1d", "demo_multi_mass")
# Where predicted_std is 0, as in a one-particle run, the dots' mean reading must
# be within this of predicted_mean and their s.d. at most this. Rounding a dot to
# float32 moves its reading by 6.4e-11 (demo_1d) and 1.4e-9 (demo_multi_mass) in
# one-particle recordings.
ZERO_SPREAD_ATOL = 1e-7


def _params(names: Sequence[str]) -> list[object]:
    """One pytest parameter per demo, the slow ones marked, named by the demo."""
    return [
        pytest.param(
            name, id=name, marks=[pytest.mark.slow] if name in SLOW_DEMOS else []
        )
        for name in names
    ]


@functools.lru_cache(maxsize=None)
def _archive(name: str) -> Replay:
    return load_replay(ARCHIVES / f"{name}.npz")


@functools.lru_cache(maxsize=None)
def _fresh(name: str) -> Replay:
    """The demo's packaged run, recorded once per test session."""
    return importlib.import_module(f"clocks._demos.{name}").record()


def _scenario(name: str) -> Scenario:
    return importlib.import_module(f"clocks._demos.{name}").scenario()


# --- helpers that take plain arrays ------------------------------------------


def _inside_prior(cloud: NDArray, prior_ranges: Sequence[Sequence[float]]) -> bool:
    """Whether every dot lies inside the prior, the endpoints rounded to float32."""
    lo = np.asarray([r[0] for r in prior_ranges], dtype=np.float32)
    hi = np.asarray([r[1] for r in prior_ranges], dtype=np.float32)
    return bool(np.all((cloud >= lo) & (cloud <= hi)))


def _posterior_from_evidence(
    model_prior: NDArray, model_log_evidence: NDArray
) -> NDArray[np.float64]:
    """Each row's model posterior from its evidence, normalised in log space."""
    log_joint = np.log(model_prior) + model_log_evidence
    return np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))


def _assert_follows_evidence(
    model_posterior: NDArray, model_prior: NDArray, model_log_evidence: NDArray
) -> None:
    """Assert each posterior row is its evidence's, at the one tolerance for this."""
    np.testing.assert_allclose(
        model_posterior,
        _posterior_from_evidence(model_prior, model_log_evidence),
        rtol=1e-9,
        atol=1e-12,
    )


def _checkpoints(n_observations: int) -> tuple[int, ...]:
    """The rows the gate compares: after observation 1, 8 and the last."""
    return tuple(sorted({1, min(8, n_observations), n_observations}))


def _sd(cov: NDArray) -> NDArray[np.float64]:
    return np.sqrt(np.diagonal(cov, axis1=-2, axis2=-1))


def _assert_view_is_the_close_view(
    view_bounds: NDArray, true_params: NDArray, mean: NDArray, cov: NDArray
) -> None:
    """Assert the cube is the one the rows k >= 1 and the truth define."""
    expected = close_view(true_params[:3], mean[1:, :3], _sd(cov)[1:, :3])
    assert tuple(view_bounds.tolist()) == expected, (
        f"view_bounds {view_bounds.tolist()}, close view of the archive {expected}"
    )


def _assert_weight_matches_dots(
    weight_in_view: NDArray, cloud: NDArray, view_bounds: NDArray
) -> None:
    """Assert each row's weight in view is near the fraction of dots in view.

    The dots are an equal-weight sample of the weighted cloud, so their
    unweighted fraction p inside the cube estimates the weight inside it, with
    sampling error sqrt(p (1 - p) / m); the bound is six of those, or 0.01
    where p is near 0 or 1. Measured on the committed archive, the worst gap is
    0.03 of its bound (row 0: 0.0005 against 0.0169).
    """
    low, high = view_bounds
    positions = cloud[:, :, :3]
    inside = np.all((positions >= low) & (positions <= high), axis=2)
    p = inside.mean(axis=1)
    bound = np.maximum(0.01, 6 * np.sqrt(p * (1 - p) / cloud.shape[1]))
    gap = np.abs(weight_in_view - p)
    worst = int(np.argmax(gap / bound))
    assert np.all(gap <= bound), (
        f"row {worst}: weight_in_view differs from the dots by {gap[worst]:.3g}, "
        f"bound {bound[worst]:.3g}"
    )


def _readings_at(
    params: NDArray, clock_positions: NDArray, track_offset: float, centred: bool
) -> NDArray[np.float64]:
    """The forward model's reading at every clock, one row per parameter vector.

    ``params`` (n, P) is laid out as the library lays it out: positions
    row-major, then masses. ``centred`` removes each row's mean over clocks, as
    echolocation's recorder does.
    """
    dimensions = clock_positions.shape[1]
    n_masses = params.shape[1] // (dimensions + 1)
    split = n_masses * dimensions
    rates = clock_rates_batch_multi(
        params[:, :split].reshape(-1, n_masses, dimensions),
        params[:, split:],
        ClockArray(clock_positions, track_offset=track_offset),
    )
    return rates - rates.mean(axis=1, keepdims=True) if centred else rates


def _readings_of_dots(
    cloud: NDArray,
    clock_positions: NDArray,
    track_offset: float,
    centred: bool,
    first_row: int,
) -> NDArray[np.float64]:
    """The forward model at every displayed dot of rows ``first_row`` on.

    Returns (rows - first_row, m, C); earlier rows' dots are never evaluated.
    """
    return np.stack(
        [
            _readings_at(dots, clock_positions, track_offset, centred)
            for dots in cloud[first_row:].astype(np.float64)
        ]
    )


def _assert_predictions_match_dots(
    predicted_mean: NDArray,
    predicted_std: NDArray,
    dot_readings: NDArray,
    first_row: int,
) -> None:
    """Assert, row by row, the recorded predictions summarise the dots' readings.

    ``dot_readings`` (rows, m, C) is the forward model at each displayed dot of
    rows ``first_row`` on. Per clock, the dots' mean is within
    ``PREDICTION_MEAN_BOUND`` predicted s.d. / sqrt(m) of ``predicted_mean``,
    and their s.d. is ``PREDICTION_SPREAD`` times ``predicted_std``.
    """
    m = dot_readings.shape[1]
    low, high = PREDICTION_SPREAD
    for row, readings in enumerate(dot_readings, start=first_row):
        mean_gap = np.abs(readings.mean(axis=0) - predicted_mean[row])
        dots_sd = readings.std(axis=0)
        zero = predicted_std[row] == 0
        # Clocks with no predicted spread: compared in absolute terms.
        for clock in np.flatnonzero(zero & ~(mean_gap <= ZERO_SPREAD_ATOL)):
            raise AssertionError(
                f"row {row}, clock {clock}: the dots' mean reading differs from "
                f"predicted_mean by {mean_gap[clock]:.3g} where predicted_std is 0"
            )
        for clock in np.flatnonzero(zero & ~(dots_sd <= ZERO_SPREAD_ATOL)):
            raise AssertionError(
                f"row {row}, clock {clock}: the dots' s.d. is {dots_sd[clock]:.3g} "
                "where predicted_std is 0"
            )
        positive = np.flatnonzero(~zero)
        if positive.size == 0:
            continue
        sd = predicted_std[row][positive]
        deviation = mean_gap[positive] / sd
        worst = int(np.argmax(deviation))
        assert np.all(deviation <= PREDICTION_MEAN_BOUND / np.sqrt(m)), (
            f"row {row}, clock {positive[worst]}: the dots' mean reading is "
            f"{deviation[worst] * np.sqrt(m):.3g} s.d./sqrt(m) from predicted_mean"
        )
        ratio = dots_sd[positive] / sd
        outside = int(np.argmax((ratio < low) | (ratio > high)))
        assert np.all((low <= ratio) & (ratio <= high)), (
            f"row {row}, clock {positive[outside]}: the dots' s.d. is "
            f"{ratio[outside]:.3g} of predicted_std, expected {low} to {high}"
        )


def _assert_prediction_matches(
    predicted_mean: NDArray, predicted_std: NDArray, model: NDArray
) -> None:
    """Assert one row's predicted readings are within half an s.d. of ``model``."""
    gap = np.abs(predicted_mean - model) / predicted_std
    assert np.all(gap <= 0.5), (
        f"predicted_mean differs from the model at the mean by {gap.max():.3g} "
        f"predicted s.d. at clock {int(np.argmax(gap))}"
    )


def _assert_dots_share_orientation(cloud: NDArray, cov: NDArray) -> None:
    """Assert, in every row, the dots' correlations are near the covariance's."""
    off_diagonal = ~np.eye(cov.shape[-1], dtype=bool)
    for row, (dots, covariance) in enumerate(zip(cloud, cov)):
        from_dots = np.corrcoef(dots.astype(np.float64), rowvar=False)
        sd = _sd(covariance)
        recorded = covariance / np.outer(sd, sd)
        gap = np.abs(from_dots - recorded)[off_diagonal]
        assert np.all(gap <= 0.4), (
            f"row {row}: dots' correlation differs from cov's by {gap.max():.3g}"
        )


# --- invariants on the committed archives ------------------------------------


def test_archives_are_exactly_the_seven() -> None:
    assert {path.stem for path in ARCHIVES.glob("*.npz")} == set(DEMOS)


@pytest.mark.parametrize("name", DEMOS)
def test_archive_matches_its_scenario(name: str) -> None:
    meta, arrays = _archive(name).meta, _archive(name).arrays
    scenario = _scenario(name)

    assert meta["demo"] == scenario.demo
    assert meta["seed"] == scenario.seed
    assert meta["noise_std"] == scenario.noise_std
    assert meta["n_observations"] == scenario.n_observations
    assert meta["n_particles"] == scenario.n_particles
    assert meta["track_offset"] == scenario.clock_array.track_offset
    assert meta["param_names"] == list(scenario.param_names)
    assert meta["prior_ranges"] == [list(pair) for pair in scenario.prior_ranges]
    assert_computed_input_matches(
        arrays["clock_positions"], scenario.clock_array.positions, "clock_positions"
    )
    assert np.array_equal(arrays["true_params"], scenario.true_params)
    np.testing.assert_allclose(
        arrays["true_rates"], scenario.true_rates, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("name", DEMOS)
def test_scenario_names_are_the_documented_ones(name: str) -> None:
    assert _scenario(name).param_names == PARAM_NAMES[name]


@pytest.mark.parametrize("name", DEMOS)
def test_archive_truth_is_physical(name: str) -> None:
    arrays = _archive(name).arrays
    true_params = arrays["true_params"]

    assert np.max(np.abs(arrays["true_rates"] ** 2 - 1.0)) <= 0.08
    if name in ("demo_multi_mass", "demo_model_comparison"):
        assert true_params[0] < true_params[1]
    if name == "demo_multi_mass_2d":
        assert true_params[0] < true_params[2]


@pytest.mark.parametrize("name", FILTER_DEMOS)
def test_archive_sentinels_and_ranges(name: str) -> None:
    meta, arrays = _archive(name).meta, _archive(name).arrays
    n_particles = meta["n_particles"]
    ess = arrays["ess"]

    assert arrays["log_evidence"][0] == 0.0
    assert ess[0] == n_particles
    assert np.all(ess > 0)
    assert np.all(ess <= n_particles * (1 + 1e-12))
    for row, cov in enumerate(arrays["cov"]):
        np.testing.assert_allclose(
            cov, cov.T, rtol=0, atol=1e-12 * np.abs(cov).max(), err_msg=f"row {row}"
        )
        assert np.linalg.eigvalsh(cov).min() >= -1e-12 * np.trace(cov), f"row {row}"
    assert np.all(arrays["mh_acceptances"] <= arrays["mh_proposals"])
    assert np.all(arrays["tempering_stages"] >= 1)


@pytest.mark.parametrize("name", FILTER_DEMOS)
def test_archive_dots_agree_with_summaries(name: str) -> None:
    meta, arrays = _archive(name).meta, _archive(name).arrays
    cloud = arrays["cloud"]
    moments = cloud.astype(np.float64)
    m = cloud.shape[1]
    sd = _sd(arrays["cov"])

    assert _inside_prior(cloud, meta["prior_ranges"])
    deviation = np.abs(moments.mean(axis=1) - arrays["mean"])
    assert np.all(deviation <= 6 * sd / np.sqrt(m))
    ratio = moments.std(axis=1) / sd
    assert np.all((0.6 <= ratio) & (ratio <= 1.5))


@pytest.mark.parametrize(
    ("value", "inside"),
    [
        pytest.param(np.float32(np.nextafter(0.15, 0.0)), True, id="just_inside"),
        pytest.param(
            np.nextafter(np.float32(0.15), np.float32(1)), False, id="one_step_above"
        ),
        pytest.param(
            np.nextafter(np.float32(0.005), np.float32(0)), False, id="one_step_below"
        ),
    ],
)
def test_float32_endpoints(value: np.float32, inside: bool) -> None:
    cloud = np.array([[[value]]], dtype=np.float32)

    assert _inside_prior(cloud, [[0.005, 0.15]]) is inside


def test_echolocation_archive_channels() -> None:
    arrays = _archive("demo_echolocation_3d").arrays
    observations = arrays["observations"]
    view_bounds = arrays["view_bounds"]
    weight_in_view = arrays["weight_in_view"]

    np.testing.assert_allclose(observations.sum(axis=1), 0.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        arrays["filter_observations"],
        observations @ contrast_matrix(27).T,
        rtol=1e-12,
        atol=1e-12,
    )
    assert view_bounds[0] <= -1.0
    assert view_bounds[1] >= max(1.0, arrays["true_params"][:3].max())
    assert np.all((0 <= weight_in_view) & (weight_in_view <= 1 + 1e-12))


def test_echolocation_archive_view_is_the_close_view() -> None:
    arrays = _archive("demo_echolocation_3d").arrays

    _assert_view_is_the_close_view(
        arrays["view_bounds"], arrays["true_params"], arrays["mean"], arrays["cov"]
    )


def test_a_view_that_is_not_the_close_view_is_caught() -> None:
    arrays = _archive("demo_echolocation_3d").arrays

    with pytest.raises(AssertionError):
        _assert_view_is_the_close_view(
            np.array([-2.0, 6.0]), arrays["true_params"], arrays["mean"], arrays["cov"]
        )


def test_echolocation_archive_weight_matches_the_dots() -> None:
    arrays = _archive("demo_echolocation_3d").arrays

    _assert_weight_matches_dots(
        arrays["weight_in_view"], arrays["cloud"], arrays["view_bounds"]
    )


def test_a_weight_in_view_the_dots_contradict_is_caught() -> None:
    arrays = _archive("demo_echolocation_3d").arrays
    corrupted = arrays["weight_in_view"].copy()
    corrupted[-1] = 0.5

    with pytest.raises(AssertionError):
        _assert_weight_matches_dots(corrupted, arrays["cloud"], arrays["view_bounds"])


def test_a_weight_in_view_off_by_its_sampling_error_is_caught() -> None:
    # The row with the fewest dots in view (row 0 today, 0.4%), its weight set
    # 0.036 above the dots' fraction: inside the old fixed 0.05, outside six
    # sampling errors.
    OFFSET = 0.036
    OLD_TOLERANCE = 0.05  # the weight check's fixed tolerance, before the bound
    arrays = _archive("demo_echolocation_3d").arrays
    cloud, low_high = arrays["cloud"], arrays["view_bounds"]
    low, high = low_high
    positions = cloud[:, :, :3]
    inside = np.all((positions >= low) & (positions <= high), axis=2)
    p = inside.mean(axis=1)
    row = int(np.argmin(p))
    bound = max(0.01, 6 * np.sqrt(p[row] * (1 - p[row]) / cloud.shape[1]))
    assert bound < OFFSET
    assert OFFSET < OLD_TOLERANCE  # why 0.036: inside the old fixed tolerance
    corrupted = arrays["weight_in_view"].copy()
    corrupted[row] = p[row] + OFFSET

    with pytest.raises(AssertionError, match=f"row {row}:"):
        _assert_weight_matches_dots(corrupted, cloud, low_high)


def _first_row(name: str) -> int:
    return FIRST_CHECKED_ROW.get(name, 0)


@functools.lru_cache(maxsize=None)
def _dot_readings(name: str) -> NDArray[np.float64]:
    """The forward model at the displayed dots of every checked row."""
    meta, arrays = _archive(name).meta, _archive(name).arrays
    return _readings_of_dots(
        arrays["cloud"],
        arrays["clock_positions"],
        meta["track_offset"],
        centred=name == "demo_echolocation_3d",
        first_row=_first_row(name),
    )


@pytest.mark.parametrize("name", PREDICTING_DEMOS)
def test_archive_predictions_match_the_dots(name: str) -> None:
    arrays = _archive(name).arrays

    _assert_predictions_match_dots(
        arrays["predicted_mean"],
        arrays["predicted_std"],
        _dot_readings(name),
        _first_row(name),
    )


@pytest.mark.parametrize("name", ["demo_1d", "demo_multi_mass"])
def test_a_one_particle_recording_passes_the_prediction_check(name: str) -> None:
    module = importlib.import_module(f"clocks._demos.{name}")
    replay = module.record(n_observations=2, n_particles=1)
    arrays = replay.arrays

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _assert_predictions_match_dots(
            arrays["predicted_mean"],
            arrays["predicted_std"],
            _readings_of_dots(
                arrays["cloud"],
                arrays["clock_positions"],
                replay.meta["track_offset"],
                centred=False,
                first_row=0,
            ),
            0,
        )


def test_a_one_particle_prediction_off_the_dot_is_caught() -> None:
    module = importlib.import_module("clocks._demos.demo_1d")
    replay = module.record(n_observations=2, n_particles=1)
    arrays = replay.arrays
    readings = _readings_of_dots(
        arrays["cloud"],
        arrays["clock_positions"],
        replay.meta["track_offset"],
        centred=False,
        first_row=0,
    )
    corrupted = arrays["predicted_mean"].copy()
    corrupted[0, 0] += 3e-7  # 3 x the tolerance, written out so loosening it fails

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(
            AssertionError,
            match="row 0, clock 0: the dots' mean reading differs from "
            "predicted_mean by .* where predicted_std is 0",
        ):
            _assert_predictions_match_dots(
                corrupted, arrays["predicted_std"], readings, 0
            )


def test_a_spread_where_none_is_predicted_is_caught() -> None:
    with pytest.raises(
        AssertionError,
        match="row 0, clock 0: the dots' s.d. is .* where predicted_std is 0",
    ):
        _assert_predictions_match_dots(
            np.array([[1.0]]),
            np.array([[0.0]]),
            np.array([[[1.0 - 3e-7], [1.0 + 3e-7]]]),  # s.d. 3e-7, 3 x the tolerance
            0,
        )


def test_a_nan_prediction_where_no_spread_is_predicted_is_caught() -> None:
    with pytest.raises(AssertionError, match="row 0, clock 0:"):
        _assert_predictions_match_dots(
            np.array([[np.nan]]),
            np.array([[0.0]]),
            np.array([[[1.0], [1.0]]]),
            0,
        )


def test_a_nan_dot_where_no_spread_is_predicted_is_caught() -> None:
    with pytest.raises(AssertionError, match="row 0, clock 0:"):
        _assert_predictions_match_dots(
            np.array([[1.0]]),
            np.array([[0.0]]),
            np.array([[[np.nan], [1.0]]]),
            0,
        )


def test_a_mixed_row_names_the_original_clock() -> None:
    # Clock 0 has no spread and agrees; clock 1 is 10 predicted s.d. off, 14.1
    # in units of s.d./sqrt(m) against 6. A helper that renumbered the
    # positive-spread clocks would say clock 0.
    with pytest.raises(
        AssertionError, match="row 0, clock 1: the dots' mean reading is"
    ):
        _assert_predictions_match_dots(
            np.array([[1.0, 1.0]]),
            np.array([[0.0, 0.01]]),
            np.array([[[1.0, 1.09], [1.0, 1.11]]]),
            0,
        )


def test_a_mixed_row_names_the_original_clock_for_a_spread() -> None:
    # Clock 0 has a spread and agrees (mean 1.0, s.d. 0.01 = predicted_std);
    # clock 1 has none, its mean matches, but its dots spread by 3e-7.
    with pytest.raises(
        AssertionError, match="row 0, clock 1: the dots' s.d. is 3e-07 where"
    ):
        _assert_predictions_match_dots(
            np.array([[1.0, 2.0]]),
            np.array([[0.01, 0.0]]),
            np.array([[[0.99, 2.0 - 3e-7], [1.01, 2.0 + 3e-7]]]),
            0,
        )


def test_prior_cases_and_the_skip_map_partition_the_predicting_demos() -> None:
    assert set(PRIOR_CHECKED_DEMOS).isdisjoint(FIRST_CHECKED_ROW)
    assert set(PRIOR_CHECKED_DEMOS) | set(FIRST_CHECKED_ROW) == set(PREDICTING_DEMOS)


def test_echolocation_prior_dots_are_not_evaluated() -> None:
    # A prior dot placed on a clock makes the forward model raise (a singular
    # distance: echolocation's track offset is 0), as a dot rounded across the
    # weak-field limit would; skipping row 0 must never reach it.
    meta, arrays = (
        _archive("demo_echolocation_3d").meta,
        _archive("demo_echolocation_3d").arrays,
    )
    cloud = arrays["cloud"].copy()
    cloud[0, 0, :3] = arrays["clock_positions"][13]

    readings = _readings_of_dots(
        cloud,
        arrays["clock_positions"],
        meta["track_offset"],
        centred=True,
        first_row=_first_row("demo_echolocation_3d"),
    )

    assert readings.shape[0] == len(cloud) - 1


_SHIFTS = [
    pytest.param(name, row, shift, id=f"{label}-{name}")
    for label, row, shift in [
        ("prior", 0, 0.02),
        ("first_update", 1, 0.02),
        ("final", -1, 0.002),
    ]
    for name in (PRIOR_CHECKED_DEMOS if label == "prior" else PREDICTING_DEMOS)
]


@pytest.mark.parametrize(("name", "row", "shift"), _SHIFTS)
def test_a_shifted_prediction_is_caught(name: str, row: int, shift: float) -> None:
    arrays = _archive(name).arrays
    corrupted = arrays["predicted_mean"].copy()
    clock = int(np.argmax(arrays["predicted_std"][row]))  # where a shift shows least
    corrupted[row, clock] += shift
    shown_row = row % len(corrupted)

    with pytest.raises(
        AssertionError, match=f"row {shown_row}, clock {clock}: .* from predicted_mean"
    ):
        _assert_predictions_match_dots(
            corrupted, arrays["predicted_std"], _dot_readings(name), _first_row(name)
        )


@pytest.mark.parametrize("name", PREDICTING_DEMOS)
def test_widened_final_prediction_bars_are_caught(name: str) -> None:
    arrays = _archive(name).arrays
    corrupted = arrays["predicted_std"].copy()
    corrupted[-1] *= 10
    last = len(corrupted) - 1

    with pytest.raises(AssertionError, match=f"row {last}, clock .* of predicted_std"):
        _assert_predictions_match_dots(
            arrays["predicted_mean"], corrupted, _dot_readings(name), _first_row(name)
        )


def _model_at_final_mean(name: str, mean: NDArray) -> NDArray[np.float64]:
    """The forward model at the last row of ``mean``, as the demo predicts.

    Echolocation predicts centred readings, so the mean over clocks is removed.
    This ties the predictions to the displayed estimate, which the dots check
    cannot: the dots and the mean are each within their own tolerance.
    """
    meta, arrays = _archive(name).meta, _archive(name).arrays
    return _readings_at(
        mean[-1][None, :],
        arrays["clock_positions"],
        meta["track_offset"],
        centred=name == "demo_echolocation_3d",
    )[0]


@pytest.mark.parametrize("name", PREDICTING_DEMOS)
def test_archive_predictions_match_the_final_estimate(name: str) -> None:
    arrays = _archive(name).arrays

    _assert_prediction_matches(
        arrays["predicted_mean"][-1],
        arrays["predicted_std"][-1],
        _model_at_final_mean(name, arrays["mean"]),
    )


def test_a_final_prediction_off_the_estimate_is_caught() -> None:
    arrays = _archive("demo_1d").arrays

    with pytest.raises(AssertionError):
        _assert_prediction_matches(
            arrays["predicted_mean"][-1] + 0.002,
            arrays["predicted_std"][-1],
            _model_at_final_mean("demo_1d", arrays["mean"]),
        )


def test_a_final_estimate_off_the_predictions_is_caught() -> None:
    # The reviewer's edit: the final mean mass lowered by 0.00045. The dots and
    # every other invariant pass it; on the committed archive it is 1.19
    # predicted s.d. at clock 12, but only the half-s.d. threshold is required.
    arrays = _archive("demo_echolocation_3d").arrays
    mean = arrays["mean"].copy()
    mean[-1, 3] -= 0.00045

    with pytest.raises(
        AssertionError, match="predicted_mean differs from the model at the mean by"
    ):
        _assert_prediction_matches(
            arrays["predicted_mean"][-1],
            arrays["predicted_std"][-1],
            _model_at_final_mean("demo_echolocation_3d", mean),
        )


@pytest.mark.parametrize("name", FILTER_DEMOS)
def test_archive_covariance_orientation_matches_the_dots(name: str) -> None:
    arrays = _archive(name).arrays

    _assert_dots_share_orientation(arrays["cloud"], arrays["cov"])


def test_a_tilted_covariance_is_caught() -> None:
    # The row after the prior with the largest |x-y correlation| (row 5 today,
    # -0.326). A reversed tilt is caught only where |r| exceeds about 0.2: the
    # bound is 0.4 and flipping r moves the dots' correlation by 2|r|.
    arrays = _archive("demo_2d").arrays
    cov = arrays["cov"]
    r = cov[:, 0, 1] / np.sqrt(cov[:, 0, 0] * cov[:, 1, 1])
    row = 1 + int(np.argmax(np.abs(r[1:])))
    assert abs(r[row]) > 0.2
    corrupted = cov.copy()
    corrupted[row, 0, 1] *= -1  # the x-y entry: the drawn ellipse's tilt
    corrupted[row, 1, 0] *= -1

    with pytest.raises(AssertionError, match=f"row {row}:"):
        _assert_dots_share_orientation(arrays["cloud"], corrupted)


def test_model_archive_rows() -> None:
    meta, arrays = (
        _archive("demo_model_comparison").meta,
        _archive("demo_model_comparison").arrays,
    )
    posterior = arrays["model_posterior"]

    assert np.all(arrays["model_log_evidence"][0] == 0.0)
    assert np.all(arrays["model_ess"][0] == meta["n_particles"])
    assert np.array_equal(posterior[0], arrays["model_prior"])
    np.testing.assert_allclose(posterior.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    assert arrays["model_ks"].tolist() == [1, 2, 3]
    assert meta["true_k"] == 2


def test_model_posterior_follows_evidence() -> None:
    arrays = _archive("demo_model_comparison").arrays

    _assert_follows_evidence(
        arrays["model_posterior"], arrays["model_prior"], arrays["model_log_evidence"]
    )


def test_a_posterior_row_that_keeps_its_sum_is_caught() -> None:
    arrays = _archive("demo_model_comparison").arrays
    corrupted = arrays["model_posterior"].copy()
    corrupted[-1] = [0.0, 0.99, 0.01]

    np.testing.assert_allclose(corrupted.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    with pytest.raises(AssertionError):
        _assert_follows_evidence(
            corrupted, arrays["model_prior"], arrays["model_log_evidence"]
        )


def test_an_underflowed_candidate_gives_zero() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        posterior = _posterior_from_evidence(
            np.array([0.5, 0.5]), np.array([[-1000.0, 0.0]])
        )

    assert np.array_equal(posterior, [[0.0, 1.0]])


# --- the gate: a fresh recording against the committed archive ---------------


def test_bounds_cover_every_filter_demo() -> None:
    assert set(BOUNDS) | {"demo_density"} == set(FILTER_DEMOS)


@pytest.mark.parametrize("name", _params(DEMOS))
def test_gate_inputs(name: str) -> None:
    archive, fresh = _archive(name), _fresh(name)

    assert set(fresh.arrays) == set(archive.arrays), _rerecord(name)
    for key, array in archive.arrays.items():
        assert fresh.arrays[key].shape == array.shape, f"{key}; {_rerecord(name)}"
        assert fresh.arrays[key].dtype == array.dtype, f"{key}; {_rerecord(name)}"
    assert _without_provenance(fresh.meta) == _without_provenance(archive.meta), (
        _rerecord(name)
    )
    exact = ["true_params"]
    close = ["clock_positions", "true_rates", "observations"]
    if name == "demo_model_comparison":
        exact += ["model_ks", "model_prior"]
    if name == "demo_echolocation_3d":
        close.append("filter_observations")
    for key in exact:
        assert np.array_equal(fresh.arrays[key], archive.arrays[key]), (
            f"{key}; {_rerecord(name)}"
        )
    for key in close:
        assert_computed_input_matches(
            fresh.arrays[key], archive.arrays[key], f"{key}; {_rerecord(name)}"
        )


def assert_computed_input_matches(
    actual: NDArray, expected: NDArray, label: str
) -> None:
    """Computed inputs match a committed archive to ``1e-12``, not bit for bit.

    The two-mass 2D demo's clocks are drawn from a seeded generator, so their
    low-order bits can differ between processor architectures: an exact comparison
    failed on the x86-64 CI runner against an archive recorded on arm64, while
    the full comparison reproduced.
    """
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, err_msg=label)


def test_the_computed_input_comparison_rejects_a_real_change() -> None:
    positions = _archive("demo_multi_mass_2d").arrays["clock_positions"]
    moved = positions.copy()
    moved[0, 0] *= 1 + 1e-9

    assert_computed_input_matches(positions, positions.copy(), "unchanged")
    with pytest.raises(AssertionError, match="clock_positions"):
        assert_computed_input_matches(moved, positions, "clock_positions")


def _rerecord(name: str) -> str:
    """How to bring the archive up to date when the difference was intended."""
    return (
        "if this change was intended, re-record: "
        "uv run python scripts/record_replays.py --out assets/replays --overwrite "
        f"{name}"
    )


@pytest.mark.parametrize("name", _params(DEMOS))
def test_gate_summaries(name: str) -> None:
    archive, fresh = _archive(name), _fresh(name)
    rows = _checkpoints(archive.meta["n_observations"])

    if name == "demo_model_comparison":
        _check_model_summaries(archive.arrays, fresh.arrays, rows)
        return
    bounds = _column_bounds(name, archive.meta["param_names"])
    assert len(rows) == len(bounds)
    for row, (mean_bound, sd_bound) in zip(rows, bounds):
        mean_gap = np.abs(fresh.arrays["mean"][row] - archive.arrays["mean"][row])
        sd_gap = np.abs(_sd(fresh.arrays["cov"][row]) - _sd(archive.arrays["cov"][row]))
        assert np.all(mean_gap <= mean_bound), (
            f"after observation {row}: mean gap {mean_gap}, bounds {mean_bound}"
        )
        assert np.all(sd_gap <= sd_bound), (
            f"after observation {row}: sd gap {sd_gap}, bounds {sd_bound}"
        )


def _without_provenance(meta: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in meta.items() if key != "created_with"}


def _column_bounds(
    name: str, param_names: Sequence[str]
) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Per checkpoint, the bound on each column's mean and on its sd."""
    if name == "demo_density":
        return [(np.array(mean), np.array(sd)) for mean, sd in DENSITY_BOUNDS]
    kinds = [_column_kind(param) for param in param_names]
    out = []
    for mean_pos, mean_mass, sd_pos, sd_mass in BOUNDS[name]:
        mean = {"position": mean_pos, "mass": mean_mass}
        sd = {"position": sd_pos, "mass": sd_mass}
        out.append(
            (np.array([mean[k] for k in kinds]), np.array([sd[k] for k in kinds]))
        )
    return out


def _column_kind(param: str) -> str:
    if param[0] in "xyz":
        return "position"
    if param[0] == "M":
        return "mass"
    raise ValueError(f"parameter {param!r}: expected a name starting x, y, z or M")


def _check_model_summaries(
    archive: dict[str, NDArray], fresh: dict[str, NDArray], rows: tuple[int, ...]
) -> None:
    first, *later = rows
    for source, arrays in (("archive", archive), ("fresh", fresh)):
        posterior = arrays["model_posterior"]
        assert posterior[first, 0] <= 1e-5, f"{source}: P(K=1) after {first}"
        for row in later:
            assert posterior[row, 0] <= 1e-20, f"{source}: P(K=1) after {row}"
    gap = abs(fresh["model_posterior"][1, 1] - archive["model_posterior"][1, 1])
    assert gap <= 0.45
