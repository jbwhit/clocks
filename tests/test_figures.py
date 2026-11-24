"""The density figure and the trace helpers, checked through the artists drawn."""

import ast
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from _figure_artists import (
    assert_drawn,
    band_edges,
    gid_census,
    one,
    palette_hexes,
    visible_colours,
)
from matplotlib.colors import to_hex

import clocks.viz
from clocks import _figures
from clocks._demos import demo_1d
from clocks._figures import density_figure, trace_decimals, trace_limits
from clocks._replay import Replay, load_replay, validate_replay
from clocks._style import THEMES, style

ROOT = Path(__file__).resolve().parents[1]
F8 = np.float64
CONTROLS = {
    "ess_target": 0.8,
    "rejuvenation_steps": 2,
    "proposal_scale": 2.38,
    "resampling": "systematic",
}
CREATED = {"python": "3.14.0", "numpy": "2.4.2", "scipy": "1.17.1", "platform": "test"}
TRACES = ["trace-μ", "trace-σ", "trace-A"]
X = np.linspace(-8.0, 8.0, 321)


def density_replay() -> Replay:
    meta = {
        "format": "clocks-replay",
        "version": 1,
        "demo": "demo_density",
        "kind": "filter",
        "seed": 7,
        "n_observations": 2,
        "n_particles": 4,
        "noise_std": 0.005,
        "track_offset": 1.0,
        "param_names": ["μ", "σ", "A"],
        "prior_ranges": [[-8.0, 8.0], [0.1, 5.0], [0.001, 0.03]],
        "controls": CONTROLS,
        "cloud_size": 4,
        "created_with": CREATED,
    }
    sds = (
        np.array([4.6, 1.4, 0.0084]),
        np.array([0.6, 0.5, 0.002]),
        np.array([0.1, 0.15, 0.0006]),
    )
    arrays = {
        "clock_positions": np.array([[-6.0], [-3.0], [0.0], [3.0], [6.0]], F8),
        "true_params": np.array([1.5, 2.0, 0.010], F8),
        "true_rates": np.array([0.999, 0.996, 0.992, 0.991, 0.997], F8),
        "observations": np.array(
            [
                [0.998, 0.995, 0.993, 0.990, 0.996],
                [1.000, 0.997, 0.991, 0.992, 0.998],
            ],
            F8,
        ),
        "mean": np.array(
            [[0.0, 2.55, 0.0155], [1.2, 2.3, 0.011], [1.45, 2.1, 0.0102]], F8
        ),
        "cov": np.stack([np.diag(s**2) for s in sds]).astype(F8),
        "ess": np.array([4.0, 3.5, 3.8], F8),
        "log_evidence": np.array([0.0, 5.0, 11.0], F8),
        "tempering_stages": np.array([2, 1], np.int64),
        "mh_proposals": np.array([8, 8], np.int64),
        "mh_acceptances": np.array([3, 4], np.int64),
        "cloud": np.array(
            [
                [
                    [-5.0, 0.5, 0.004],
                    [-1.0, 2.0, 0.012],
                    [2.0, 3.5, 0.020],
                    [6.0, 4.5, 0.028],
                ],
                [
                    [0.5, 1.8, 0.009],
                    [1.0, 2.2, 0.011],
                    [1.5, 2.6, 0.012],
                    [2.0, 2.9, 0.013],
                ],
                [
                    [1.3, 1.9, 0.0098],
                    [1.4, 2.0, 0.0100],
                    [1.5, 2.2, 0.0104],
                    [1.6, 2.3, 0.0106],
                ],
            ],
            np.float32,
        ),
    }
    return Replay(meta, arrays)


def edited(replay: Replay, **arrays: np.ndarray) -> Replay:
    return Replay(replay.meta, {**replay.arrays, **arrays})


def drawn(replay: Replay, theme: str = "light"):
    """Draw in a theme; return (fig, colors). The caller closes the figure."""
    with style(theme) as colors:
        return density_figure(replay, colors), colors


@pytest.fixture
def themed_figure(request):
    """Figure of the fixture replay in the requested theme, closed afterwards."""
    fig, colors = drawn(density_replay(), request.param)
    try:
        yield fig, colors
    finally:
        plt.close(fig)


each_theme = pytest.mark.parametrize("themed_figure", THEMES, indirect=True)


def artist_data(replay: Replay, gid: str, axes_gid: str, theme: str = "light"):
    fig, _ = drawn(replay, theme)
    try:
        artist = one(next(ax for ax in fig.axes if ax.get_gid() == axes_gid), gid)
        return artist.get_xdata().copy(), artist.get_ydata().copy()
    finally:
        plt.close(fig)


def profiles_of(replay: Replay) -> np.ndarray:
    dots = replay.arrays["cloud"][-1].astype(np.float64)
    return np.array([a * np.exp(-((X - mu) ** 2) / (2 * s**2)) for mu, s, a in dots])


def true_profile(replay: Replay) -> np.ndarray:
    mu, s, a = replay.arrays["true_params"]
    return a * np.exp(-((X - mu) ** 2) / (2 * s**2))


def hexof(value) -> str:
    return to_hex(value, keep_alpha=False).lower()


# --- helper module -----------------------------------------------------


def test_visible_colours_finds_a_literal_colour() -> None:
    with style("light") as colors:
        fig, ax = plt.subplots()
        try:
            ax.plot([0, 1], [0, 1], color="#ff0000")
            assert "#ff0000" in visible_colours(fig)
            assert visible_colours(fig) - palette_hexes(colors) == {"#ff0000"}
        finally:
            plt.close(fig)


# --- the figure --------------------------------------------------------


@each_theme
def test_density_figure_size_and_axes(themed_figure) -> None:
    fig, _ = themed_figure
    assert tuple(fig.get_size_inches()) == (7.5, 5.4)
    assert [ax.get_gid() for ax in fig.axes] == ["profile", *TRACES]
    assert gid_census(fig) == {
        "true-profile": ["profile"],
        "median-profile": ["profile"],
        "profile-band": ["profile"],
        "clocks": ["profile"],
        "trace-mean": TRACES,
        "trace-band": TRACES,
        "trace-truth": TRACES,
    }


@each_theme
def test_an_artist_under_a_hidden_axes_is_not_counted_as_drawn(themed_figure) -> None:
    fig, _ = themed_figure
    one(fig.axes[0], "true-profile")
    fig.axes[0].set_visible(False)
    with pytest.raises(AssertionError, match="got 0"):
        one(fig.axes[0], "true-profile")
    assert "true-profile" not in gid_census(fig)


@each_theme
def test_true_profile(themed_figure) -> None:
    fig, colors = themed_figure
    line = one(fig.axes[0], "true-profile")
    assert np.array_equal(line.get_xdata(), X)
    expected = 0.010 * np.exp(-((X - 1.5) ** 2) / (2 * 2.0**2))
    np.testing.assert_allclose(line.get_ydata(), expected, rtol=1e-12)
    assert hexof(line.get_color()) == hexof(colors.truth)
    assert_drawn(line)


@each_theme
def test_median_and_band_are_pointwise_percentiles_of_the_last_row(
    themed_figure,
) -> None:
    fig, colors = themed_figure
    profiles = profiles_of(density_replay())
    line = one(fig.axes[0], "median-profile")
    band = one(fig.axes[0], "profile-band")
    assert np.array_equal(line.get_xdata(), X)
    np.testing.assert_allclose(
        line.get_ydata(), np.median(profiles, axis=0), rtol=1e-12
    )
    lower, upper = band_edges(band, X)
    np.testing.assert_allclose(lower, np.percentile(profiles, 16, axis=0), rtol=1e-12)
    np.testing.assert_allclose(upper, np.percentile(profiles, 84, axis=0), rtol=1e-12)
    assert hexof(line.get_color()) == hexof(colors.belief)
    assert_drawn(line)
    assert hexof(band.get_facecolor()[0]) == hexof(colors.belief)
    assert band.get_visible()
    assert band.get_facecolor()[0][3] > 0


def profile_lines(replay: Replay):
    fig, _ = drawn(replay)
    try:
        ax = fig.axes[0]
        median = one(ax, "median-profile").get_ydata().copy()
        return median, band_edges(one(ax, "profile-band"), X)
    finally:
        plt.close(fig)


def test_only_the_last_cloud_row_feeds_the_profile() -> None:
    base = density_replay()
    cloud = base.arrays["cloud"].copy()
    cloud[1] = cloud[0]
    median, (lower, upper) = profile_lines(base)
    median2, (lower2, upper2) = profile_lines(edited(base, cloud=cloud))
    assert np.array_equal(median, median2)
    assert np.array_equal(lower, lower2) and np.array_equal(upper, upper2)
    moved = base.arrays["cloud"].copy()
    moved[-1, :, 0] += 1.0
    median3, _ = profile_lines(edited(base, cloud=moved))
    assert np.abs(median3 - median).max() > 1e-4


def test_changing_the_truth_moves_only_the_truth() -> None:
    base = density_replay()
    truth = base.arrays["true_params"].copy()
    truth[0] = 0.0
    other = edited(base, true_params=truth)
    _, before = artist_data(base, "true-profile", "profile")
    _, after = artist_data(other, "true-profile", "profile")
    assert not np.allclose(before, after)
    _, m_before = artist_data(base, "median-profile", "profile")
    _, m_after = artist_data(other, "median-profile", "profile")
    assert np.array_equal(m_before, m_after)
    assert list(artist_data(other, "trace-truth", "trace-μ")[1]) == [0.0, 0.0]
    for gid, value in (("trace-σ", 2.0), ("trace-A", 0.010)):
        assert list(artist_data(other, "trace-truth", gid)[1]) == [value, value]


@each_theme
def test_clocks_are_ink_squares_on_the_axis(themed_figure) -> None:
    fig, colors = themed_figure
    clocks = one(fig.axes[0], "clocks")
    assert list(clocks.get_xdata()) == [-6.0, -3.0, 0.0, 3.0, 6.0]
    assert list(clocks.get_ydata()) == [0.0] * 5
    assert clocks.get_marker() == "s"
    assert clocks.get_linestyle() == "None"
    assert hexof(clocks.get_color()) == hexof(colors.ink)
    assert clocks.get_visible()
    assert clocks.get_clip_on() is False
    assert clocks.get_alpha() in (None, 1)
    assert clocks.get_markersize() > 0


def test_clocks_follow_the_archive() -> None:
    base = density_replay()
    positions = base.arrays["clock_positions"].copy()
    positions[2, 0] = 0.5
    x, _ = artist_data(edited(base, clock_positions=positions), "clocks", "profile")
    assert list(x) == [-6.0, -3.0, 0.5, 3.0, 6.0]


@each_theme
def test_density_labels(themed_figure) -> None:
    fig, colors = themed_figure
    texts = [t for t in fig.axes[0].texts if t.get_visible()]
    for label, role in (
        ("true profile", colors.truth),
        ("pointwise median and 68% band", colors.belief),
        ("clocks", colors.ink),
    ):
        matches = [t for t in texts if t.get_text() == label]
        assert len(matches) == 1, label
        assert hexof(matches[0].get_color()) == hexof(role)
    assert fig.axes[0].get_xlabel() == "position along the track"
    assert fig.axes[0].get_ylabel() == "mass density"
    assert [ax.get_xlabel() for ax in fig.axes[1:]] == ["observations"] * 3


def test_profile_limits() -> None:
    fig, _ = drawn(density_replay())
    try:
        profiles = profiles_of(density_replay())
        top = 1.08 * max(
            true_profile(density_replay()).max(),
            np.percentile(profiles, 84, axis=0).max(),
        )
        assert fig.axes[0].get_xlim() == (-8.0, 8.0)
        bottom, got = fig.axes[0].get_ylim()
        assert bottom == 0.0
        assert got == pytest.approx(top, rel=1e-12)
    finally:
        plt.close(fig)


@each_theme
def test_traces_follow_the_archive(themed_figure) -> None:
    fig, colors = themed_figure
    arrays = density_replay().arrays
    sd = np.sqrt(np.diagonal(arrays["cov"], axis1=1, axis2=2))
    limits = [(0.552, 1.848), (1.76, 2.84), (0.00884, 0.01316)]
    for j, gid in enumerate(TRACES):
        ax = fig.axes[1 + j]
        assert ax.get_gid() == gid
        mean_line = one(ax, "trace-mean")
        band = one(ax, "trace-band")
        truth = one(ax, "trace-truth")
        assert list(mean_line.get_xdata()) == [0, 1, 2]
        np.testing.assert_allclose(
            mean_line.get_ydata(), arrays["mean"][:, j], rtol=1e-12
        )
        lower, upper = band_edges(band, np.arange(3))
        np.testing.assert_allclose(lower, arrays["mean"][:, j] - sd[:, j], rtol=1e-12)
        np.testing.assert_allclose(upper, arrays["mean"][:, j] + sd[:, j], rtol=1e-12)
        t = arrays["true_params"][j]
        assert list(truth.get_xdata()) == [0, 2]
        assert list(truth.get_ydata()) == [t, t]
        assert truth.get_linestyle() == "-"
        assert hexof(mean_line.get_color()) == hexof(colors.belief)
        assert hexof(band.get_facecolor()[0]) == hexof(colors.belief)
        assert hexof(truth.get_color()) == hexof(colors.truth)
        assert_drawn(mean_line)
        assert_drawn(truth)
        assert band.get_visible() and band.get_facecolor()[0][3] > 0
        assert ax.get_xlim() == (0, 2)
        assert ax.get_ylim() == pytest.approx(limits[j], rel=1e-9)


def test_changing_one_row_moves_only_that_point() -> None:
    base = density_replay()
    mean = base.arrays["mean"].copy()
    mean[1, 0] += 0.1
    _, before = artist_data(base, "trace-mean", "trace-μ")
    _, after = artist_data(edited(base, mean=mean), "trace-mean", "trace-μ")
    diff = after - before
    assert diff[1] == pytest.approx(0.1, rel=1e-12)
    assert diff[0] == 0.0 and diff[2] == 0.0


@pytest.mark.parametrize(
    ("prior_range", "expected"),
    [((-8.0, 8.0), 2), ((0.1, 5.0), 2), ((0.001, 0.030), 4)],
    ids=["mu", "sigma", "amplitude"],
)
def test_trace_decimals(prior_range: tuple[float, float], expected: int) -> None:
    assert trace_decimals(prior_range) == expected


def test_trace_headers() -> None:
    fig, _ = drawn(density_replay())
    try:
        assert [ax.get_title() for ax in fig.axes[1:]] == [
            "μ   true 1.50   1.45 ± 0.10",
            "σ   true 2.00   2.10 ± 0.15",
            "A   true 0.0100   0.0102 ± 0.0006",
        ]
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ("mean", "sd", "truth", "expected"),
    [
        pytest.param(
            [0.0, 1.2, 1.45],
            [4.6, 0.6, 0.1],
            1.5,
            (0.552, 1.848),
            id="prior_row_left_out",
        ),
        pytest.param([0.0, 7.9], [1.0, 0.5], 7.5, (7.36, 8.0), id="clipped_to_prior"),
        # Row 1 spans -8.4 to -7.4 and the truth is -7.5, so the interval is
        # [-8.4, -7.4]: width 1.0, pad 0.04; the lower end -8.44 is clipped to -8.0
        # and the upper end is -7.4 + 0.04 = -7.36.
        pytest.param(
            [0.0, -7.9], [1.0, 0.5], -7.5, (-8.0, -7.36), id="clipped_to_prior_below"
        ),
        pytest.param(
            [0.0, 1.5, 1.5], [3.0, 0.0, 0.0], 1.5, (0.86, 2.14), id="zero_width"
        ),
    ],
)
def test_trace_limits(mean, sd, truth, expected) -> None:
    got = trace_limits(np.array(mean), np.array(sd), truth, (-8.0, 8.0))
    assert got == pytest.approx(expected, rel=1e-12)


def test_density_figure_draws_a_one_observation_run() -> None:
    base = density_replay()
    a = base.arrays
    cut = Replay(
        {**base.meta, "n_observations": 1},
        {
            **a,
            "mean": a["mean"][:2],
            "cov": a["cov"][:2],
            "ess": a["ess"][:2],
            "log_evidence": a["log_evidence"][:2],
            "cloud": a["cloud"][:2],
            "observations": a["observations"][:1],
            "tempering_stages": a["tempering_stages"][:1],
            "mh_proposals": a["mh_proposals"][:1],
            "mh_acceptances": a["mh_acceptances"][:1],
        },
    )
    validate_replay(cut)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fig, _ = drawn(cut)
        try:
            fig.canvas.draw()
            for ax in fig.axes[1:]:
                assert list(one(ax, "trace-mean").get_xdata()) == [0, 1]
                assert ax.get_xlim() == (0, 1)
        finally:
            plt.close(fig)


def test_density_figure_refuses_another_demo() -> None:
    other = demo_1d.record(n_observations=1, n_particles=16)
    with style("light") as colors:
        with pytest.raises(
            ValueError,
            match="density_figure needs a demo_density replay, got 'demo_1d'",
        ):
            density_figure(other, colors)


@pytest.fixture(scope="module")
def committed() -> Replay:
    return load_replay(ROOT / "assets/replays/demo_density.npz")


def test_density_figure_on_the_committed_archive(committed: Replay) -> None:
    fig, _ = drawn(committed)
    try:
        ax = fig.axes[0]
        median = one(ax, "median-profile").get_ydata()
        lower, upper = band_edges(one(ax, "profile-band"), X)
        idx = [40, 160, 190]
        approx = lambda values: pytest.approx(values, abs=5e-8)  # noqa: E731
        assert median[idx] == approx([0.0000196, 0.0074162, 0.0095141])
        assert lower[idx] == approx([0.0000095, 0.0071591, 0.0090899])
        assert upper[idx] == approx([0.0000349, 0.0076912, 0.0100017])
        assert ax.get_ylim()[1] == approx(0.0108040)
        assert [a.get_title() for a in fig.axes[1:]] == [
            "μ   true 1.50   1.50 ± 0.08",
            "σ   true 2.00   2.14 ± 0.12",
            "A   true 0.0100   0.0095 ± 0.0005",
        ]
    finally:
        plt.close(fig)


@pytest.mark.parametrize("theme", THEMES)
def test_every_visible_colour_is_in_the_palette(theme: str, committed: Replay) -> None:
    fig, colors = drawn(committed, theme)
    try:
        assert visible_colours(fig) <= palette_hexes(colors)
    finally:
        plt.close(fig)


def test_figures_module_imports_no_forward_model() -> None:
    tree = ast.parse(Path(_figures.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "clocks" or node.module.startswith("clocks."):
                imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(
                a.name for a in node.names if a.name.split(".")[0] == "clocks"
            )
    assert imported == {"clocks._replay", "clocks._style"}


def test_viz_exports_the_density_figure() -> None:
    assert clocks.viz.density_figure is _figures.density_figure
    assert clocks.viz.load_replay is load_replay
    assert "density_figure" in clocks.viz.__all__
    assert "load_replay" in clocks.viz.__all__
