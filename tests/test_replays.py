"""The committed replay archives: invariants on the files, and the gate.

The invariants read only the committed files. The gate re-records each demo
and requires its inputs to match the archive exactly and its posterior
summaries, after observations 1, 8 and the last, to match within ``BOUNDS``.

After an intended change to a demo's inputs or inference, re-record its
archive with ``uv run python scripts/record_replays.py --out assets/replays
<demo>`` and commit the file.
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
from clocks.physics import clock_rates
from clocks.types import ClockArray, MassConfig

ROOT = Path(__file__).parents[1]
ARCHIVES = ROOT / "assets/replays"

FILTER_DEMOS = tuple(name for name in DEMOS if name != "demo_model_comparison")
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
    unweighted fraction inside the cube estimates the weight inside it.
    """
    low, high = view_bounds
    positions = cloud[:, :, :3]
    inside = np.all((positions >= low) & (positions <= high), axis=2)
    gap = np.abs(weight_in_view - inside.mean(axis=1))
    assert np.all(gap <= 0.05), (
        f"row {int(np.argmax(gap))}: weight_in_view differs from the dots by "
        f"{gap.max():.3g}"
    )


def _rates_at(
    params: NDArray, clock_positions: NDArray, track_offset: float
) -> NDArray[np.float64]:
    """The forward model's tick rates at one parameter vector.

    ``params`` is laid out as the library lays it out: positions row-major,
    then masses.
    """
    dimensions = clock_positions.shape[1]
    n_masses = len(params) // (dimensions + 1)
    split = n_masses * dimensions
    masses = MassConfig(params[:split].reshape(n_masses, dimensions), params[split:])
    return clock_rates(masses, ClockArray(clock_positions, track_offset=track_offset))


def _assert_prediction_matches(
    predicted_mean: NDArray, predicted_std: NDArray, model: NDArray
) -> None:
    """Assert one row's predicted readings are within half an s.d. of ``model``."""
    gap = np.abs(predicted_mean - model) / predicted_std
    assert np.all(gap <= 0.5), (
        f"predicted_mean differs from the model at the mean by {gap.max():.3g} "
        "predicted s.d."
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
    assert np.array_equal(arrays["clock_positions"], scenario.clock_array.positions)
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


def _model_at_final_mean(name: str) -> NDArray[np.float64]:
    """The forward model at the archive's last posterior mean, as the demo predicts.

    Echolocation predicts centred readings, so the mean over clocks is removed.
    """
    meta, arrays = _archive(name).meta, _archive(name).arrays
    rates = _rates_at(
        arrays["mean"][-1], arrays["clock_positions"], meta["track_offset"]
    )
    if name == "demo_echolocation_3d":
        return rates - rates.mean()
    return rates


@pytest.mark.parametrize("name", ["demo_1d", "demo_multi_mass", "demo_echolocation_3d"])
def test_archive_predictions_match_the_final_estimate(name: str) -> None:
    arrays = _archive(name).arrays

    _assert_prediction_matches(
        arrays["predicted_mean"][-1],
        arrays["predicted_std"][-1],
        _model_at_final_mean(name),
    )


def test_a_shifted_prediction_is_caught() -> None:
    arrays = _archive("demo_1d").arrays

    with pytest.raises(AssertionError):
        _assert_prediction_matches(
            arrays["predicted_mean"][-1] + 0.002,
            arrays["predicted_std"][-1],
            _model_at_final_mean("demo_1d"),
        )


@pytest.mark.parametrize("name", FILTER_DEMOS)
def test_archive_covariance_orientation_matches_the_dots(name: str) -> None:
    arrays = _archive(name).arrays

    _assert_dots_share_orientation(arrays["cloud"], arrays["cov"])


def test_a_tilted_covariance_is_caught() -> None:
    # Row 1, where the x-y correlation is -0.29. At the last row it is -0.14,
    # and flipping it moves the correlation by 0.27, inside the 0.4 bound: the
    # invariant catches a reversed tilt only where the tilt is pronounced.
    arrays = _archive("demo_2d").arrays
    corrupted = arrays["cov"].copy()
    corrupted[1, 0, 1] *= -1  # the x-y entry: the drawn ellipse's tilt
    corrupted[1, 1, 0] *= -1

    with pytest.raises(AssertionError):
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
    exact = ["clock_positions", "true_params"]
    close = ["true_rates", "observations"]
    if name == "demo_model_comparison":
        exact += ["model_ks", "model_prior"]
    if name == "demo_echolocation_3d":
        close.append("filter_observations")
    for key in exact:
        assert np.array_equal(fresh.arrays[key], archive.arrays[key]), (
            f"{key}; {_rerecord(name)}"
        )
    for key in close:
        np.testing.assert_allclose(
            fresh.arrays[key],
            archive.arrays[key],
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"{key}; {_rerecord(name)}",
        )


def _rerecord(name: str) -> str:
    """How to bring the archive up to date when the difference was intended."""
    return (
        "if this change was intended, re-record: "
        f"uv run python scripts/record_replays.py --out assets/replays {name}"
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
