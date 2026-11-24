"""Tests for the echolocation study helpers (pure, no inference runs)."""

import math
from pathlib import Path

import matplotlib
import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
import pytest
from _figure_artists import (
    assert_drawn,
    assert_marks_drawn,
    gid_census,
    one,
    palette_hexes,
    png_size,
    visible_colours,
)
from matplotlib.colors import to_hex, to_rgb

import clocks._echo_study
import clocks.viz
from clocks._echo_study import (
    load_study,
    range_study_figure,
    save_study,
    snr_table,
    summarize,
    write_summary_figure,
)
from clocks._scenarios import (
    ECHO_M_TRUE,
    ECHO_NOISE_STD,
    EchoRunResult,
    echo_mass_position,
)
from clocks._style import THEMES, palette, style

matplotlib.use("Agg")


def _fake_result(seed: int, range_r: float, err: float) -> EchoRunResult:
    mean = np.append(echo_mass_position(range_r), ECHO_M_TRUE)
    mean[0] += err
    std = np.array([max(err, 0.1), 0.2, 0.4, 0.03])
    return EchoRunResult(
        seed=seed,
        range_r=range_r,
        passed=err <= 1.0,
        mean=mean,
        std=std,
        position_error=err,
        mass_error=0.0,
        pos_std=float(np.linalg.norm(std[:3])),
        mass_std=0.03,
        covered_3sigma=True,
        residual_over_noise=1.2,
        normalized_error=err / 2.0,
        forward_model_evaluations=123,
        ess_target=0.9,
        rejuvenation_steps=1,
        proposal_scale=1.5,
    )


def test_snr_table_decreases_with_range() -> None:
    table = snr_table([2.0, 4.0, 8.0])
    assert [row["range_r"] for row in table] == [2.0, 4.0, 8.0]
    signals = [row["signal"] for row in table]
    assert signals[0] > signals[1] > signals[2] > 0.0
    assert np.isclose(
        table[0]["signal_over_noise"], table[0]["signal"] / ECHO_NOISE_STD
    )


def test_save_load_round_trip(tmp_path: Path) -> None:
    results = [_fake_result(s, 2.0, 0.3) for s in range(3)]
    path = tmp_path / "study.json"
    save_study(
        path,
        seed_block=0,
        seeds=range(3),
        control_cells=[(0.9, 1, 1.5)],
        ranges=[2.0],
        results=results,
    )
    study = load_study(path)
    assert study["schema_version"] == 1
    assert study["study"] == "echolocation_range"
    assert study["seed_block"] == 0
    assert study["seed_role"] == "development"
    assert study["seeds"] == [0, 1, 2]
    assert study["control_grid"] == {
        "ess_target": [0.9],
        "proposal_scale": [1.5],
        "rejuvenation_steps": [1],
    }
    assert study["ranges"] == [2.0]
    assert study["tolerances"] == {
        "far_std_factor_min": 20.0,
        "mass_error_max": 0.04,
        "position_error_max": 1.0,
    }
    assert len(study["results"]) == 3
    loaded = study["results"][0]
    assert loaded["position_error"] == 0.3
    assert loaded["mean"] == pytest.approx(
        [*echo_mass_position(2.0) + np.array([0.3, 0.0, 0.0]), ECHO_M_TRUE]
    )


def test_save_study_validation_preserves_existing_evidence(tmp_path: Path) -> None:
    result = _fake_result(0, 2.0, 0.3)
    result["position_error"] = math.inf
    path = tmp_path / "study.json"
    path.write_bytes(b"existing evidence\n")

    with pytest.raises(ValueError, match="position_error.*finite"):
        save_study(
            path,
            seed_block=0,
            seeds=[0],
            control_cells=[(0.9, 1, 1.5)],
            ranges=[2.0],
            results=[result],
        )

    assert path.read_bytes() == b"existing evidence\n"


def test_summarize_medians_per_range(tmp_path: Path) -> None:
    results = [_fake_result(s, 2.0, e) for s, e in enumerate([0.1, 0.3, 0.5])]
    results += [_fake_result(s, 8.0, e) for s, e in enumerate([2.0, 4.0, 6.0])]
    path = tmp_path / "study.json"
    save_study(
        path,
        seed_block=0,
        seeds=range(3),
        control_cells=[(0.9, 1, 1.5)],
        ranges=[2.0, 8.0],
        results=results,
    )
    summary = summarize(load_study(path)["results"])
    assert [row["range_r"] for row in summary] == [2.0, 8.0]
    assert summary[0]["med_position_error"] == 0.3
    assert summary[0]["n_pass"] == 3
    assert summary[1]["med_position_error"] == 4.0
    assert summary[1]["n_pass"] == 0


def test_write_summary_figure_creates_png(tmp_path: Path) -> None:
    results = [_fake_result(s, r, 0.2 * r) for r in (2.0, 4.0) for s in range(3)]
    json_path = tmp_path / "study.json"
    save_study(
        json_path,
        seed_block=0,
        seeds=range(3),
        control_cells=[(0.9, 1, 1.5)],
        ranges=[2.0, 4.0],
        results=results,
    )
    png_path = tmp_path / "study.png"
    write_summary_figure(load_study(json_path), png_path)
    assert png_path.exists()
    assert png_path.stat().st_size > 0


ROOT = Path(__file__).resolve().parents[1]
BOTH = ["position-error", "mass-error"]
TOLERANCES = {
    "position_error_max": 1.0,
    "mass_error_max": 0.04,
    "far_std_factor_min": 20.0,
}


def _run(range_r, position_error, mass_error, pos_std, mass_std, passed):
    return {
        "range_r": range_r,
        "position_error": position_error,
        "mass_error": mass_error,
        "pos_std": pos_std,
        "mass_std": mass_std,
        "passed": passed,
    }


def literal_study(**tolerances: float) -> dict:
    return {
        "tolerances": {**TOLERANCES, **tolerances},
        "ranges": [2.0, 4.0],
        "results": [
            _run(2.0, 0.05, 0.002, 0.04, 0.002, True),
            _run(2.0, 0.10, 0.050, 0.06, 0.003, False),  # inside position, outside mass
            _run(2.0, 0.20, 0.004, 0.08, 0.004, False),  # inside both; flag disagrees
            _run(4.0, 1.50, 0.010, 1.20, 0.020, False),  # outside position
            _run(4.0, 1.00, 0.040, 0.90, 0.030, True),  # exactly on both limits
            _run(4.0, 2.50, 0.030, 1.60, 0.025, False),
        ],
    }


def _drawn(study: dict, theme: str):
    with style(theme) as colors:
        fig = range_study_figure(study, colors)
    return fig, colors


def _axes(fig):
    top, bottom = fig.axes
    return top, bottom


def _texts(ax, text: str) -> list:
    return [t for t in ax.texts if t.get_text() == text and t.get_visible()]


def _counts(fig) -> list[str]:
    top, _ = _axes(fig)
    texts = [t for t in top.texts if t.get_gid() == "pass-count"]
    assert all(t.get_visible() for t in texts)
    return [t.get_text() for t in sorted(texts, key=lambda t: t.get_position()[0])]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("theme", THEMES)
def test_range_study_figure_size_and_axes(theme: str) -> None:
    fig, _ = _drawn(literal_study(), theme)
    top, bottom = _axes(fig)
    assert tuple(fig.get_size_inches()) == (7.5, 5.4)
    assert [ax.get_gid() for ax in fig.axes] == BOTH
    assert top.get_yscale() == "log"
    assert bottom.get_yscale() == "log"
    assert top.get_shared_x_axes().joined(top, bottom)
    assert top.get_title() == ""
    assert bottom.get_title() == ""
    assert fig.get_suptitle() == ""
    assert top.get_ylabel() == "position error"
    assert bottom.get_ylabel() == "mass error"
    assert bottom.get_xlabel() == "range (circumradii)"
    assert gid_census(fig) == {
        "runs": BOTH,
        "median-error": BOTH,
        "median-spread": BOTH,
        "limit": BOTH,
        "pass-count": ["position-error", "position-error"],
        "pass-count-label": ["position-error"],
    }


@pytest.mark.parametrize("theme", THEMES)
def test_runs_are_faint_ink_dots(theme: str) -> None:
    fig, colors = _drawn(literal_study(), theme)
    results = literal_study()["results"]
    for ax, key in zip(_axes(fig), ("position_error", "mass_error")):
        runs = one(ax, "runs")
        assert list(runs.get_xdata()) == [2.0, 2.0, 2.0, 4.0, 4.0, 4.0]
        assert list(runs.get_ydata()) == [r[key] for r in results]
        assert runs.get_marker() == "o"
        assert runs.get_linestyle() == "None"
        assert to_hex(runs.get_color()) == colors.ink
        assert_marks_drawn(runs)


@pytest.mark.parametrize("theme", THEMES)
def test_medians_and_spreads_come_from_summarize(theme: str) -> None:
    fig, colors = _drawn(literal_study(), theme)
    summary = summarize(literal_study()["results"])
    expected = {
        "position-error": (
            [0.10, 1.50],
            [0.06, 1.20],
            "med_position_error",
            "med_pos_std",
        ),
        "mass-error": (
            [0.004, 0.030],
            [0.003, 0.025],
            "med_mass_error",
            "med_mass_std",
        ),
    }
    for ax in _axes(fig):
        errors, spreads, error_key, spread_key = expected[ax.get_gid()]
        median = one(ax, "median-error")
        spread = one(ax, "median-spread")
        for line, values, key, colour, marker in (
            (median, errors, error_key, colors.ink, "o"),
            (spread, spreads, spread_key, colors.belief, "s"),
        ):
            assert_drawn(line)
            assert list(line.get_xdata()) == [2.0, 4.0]
            assert list(line.get_ydata()) == pytest.approx(values)
            assert list(line.get_ydata()) == [row[key] for row in summary]
            assert to_hex(line.get_color()) == colour
            assert line.get_marker() == marker
            assert line.get_linestyle() == "-"


@pytest.mark.parametrize("theme", THEMES)
def test_limits_are_the_studys_tolerances(theme: str) -> None:
    fig, colors = _drawn(literal_study(), theme)
    for ax, value in zip(_axes(fig), (1.0, 0.04)):
        limit = one(ax, "limit")
        assert_drawn(limit)
        assert list(limit.get_ydata()) == [value, value]
        assert limit.get_linestyle() == "--"
        assert to_hex(limit.get_color()) == colors.threshold


def test_limits_follow_the_study() -> None:
    fig, _ = _drawn(
        literal_study(position_error_max=2.0, mass_error_max=0.035), "light"
    )
    top, bottom = _axes(fig)
    assert list(one(top, "limit").get_ydata()) == [2.0, 2.0]
    assert list(one(bottom, "limit").get_ydata()) == [0.035, 0.035]
    assert len(_texts(top, "limit 2.0")) == 1
    assert len(_texts(bottom, "limit 0.035")) == 1
    assert _counts(fig) == ["2", "1"]


@pytest.mark.parametrize(
    ("tolerances", "counts"),
    [
        pytest.param({"position_error_max": 2.0}, ["2", "2"], id="position_changed"),
        pytest.param({"mass_error_max": 0.003}, ["1", "0"], id="mass_changed"),
    ],
)
def test_pass_counts_follow_each_tolerance(tolerances: dict, counts: list[str]) -> None:
    fig, _ = _drawn(literal_study(**tolerances), "light")
    assert _counts(fig) == counts


def test_the_pass_counts_and_their_label_sit_above_the_top_panel() -> None:
    fig, _ = _drawn(literal_study(), "light")
    top, _ = _axes(fig)
    fig.canvas.draw()
    texts = [t for t in top.texts if t.get_gid() in {"pass-count", "pass-count-label"}]
    assert len(texts) > 1
    top_edge = top.get_window_extent().y1
    for text in texts:
        _, y = text.get_transform().transform(text.get_position())
        assert y > top_edge, f"{text.get_text()!r} sits at {y}, panel top {top_edge}"


def test_pass_counts_use_both_limits() -> None:
    fig, colors = _drawn(literal_study(), "light")
    top, _ = _axes(fig)
    texts = [t for t in top.texts if t.get_gid() == "pass-count"]
    pairs = sorted((t.get_position()[0], t.get_text()) for t in texts)
    assert pairs == [(2.0, "2"), (4.0, "1")]
    label = [
        t
        for ax in fig.axes
        for t in ax.texts
        if t.get_text() == "runs within both limits" and t.get_visible()
    ]
    assert len(label) == 1
    for text in [*texts, *label]:
        assert text.get_visible()
        assert to_hex(text.get_color()) == colors.muted


@pytest.mark.parametrize("theme", THEMES)
def test_labels_sit_on_their_axes(theme: str) -> None:
    fig, colors = _drawn(literal_study(), theme)
    top, bottom = _axes(fig)
    for text, colour in (
        ("median error", colors.ink),
        ("median posterior s.d.", colors.belief),
        ("limit 1.0", colors.threshold),
    ):
        (found,) = _texts(top, text)
        assert to_hex(found.get_color()) == colour
    assert len(_texts(top, "runs within both limits")) == 1
    (found,) = _texts(bottom, "limit 0.04")
    assert to_hex(found.get_color()) == colors.threshold
    assert _texts(bottom, "median error") == []
    assert _texts(bottom, "median posterior s.d.") == []


def test_changing_one_run_moves_its_artists() -> None:
    original, _ = _drawn(literal_study(), "light")
    study = literal_study()
    study["results"][0]["position_error"] = 0.5
    changed, _ = _drawn(study, "light")
    top, bottom = _axes(changed)
    assert one(top, "runs").get_ydata()[0] == 0.5
    assert list(one(top, "median-error").get_ydata()) == pytest.approx([0.20, 1.50])
    old_bottom = _axes(original)[1]
    for gid in ("runs", "median-error", "median-spread"):
        assert np.array_equal(
            one(bottom, gid).get_ydata(), one(old_bottom, gid).get_ydata()
        )


def test_range_study_on_the_committed_study() -> None:
    study = load_study(ROOT / "assets/echolocation_range_study.json")
    fig, _ = _drawn(study, "light")
    top, bottom = _axes(fig)
    counts = [t for t in top.texts if t.get_gid() == "pass-count"]
    counts.sort(key=lambda t: t.get_position()[0])
    assert [t.get_text() for t in counts] == ["12", "12", "11", "8", "3", "0"]
    assert [t.get_position()[0] for t in counts] == [2.0, 2.6, 3.5, 4.6, 6.1, 8.0]
    assert list(one(top, "limit").get_ydata()) == [1.0, 1.0]
    assert list(one(bottom, "limit").get_ydata()) == [0.04, 0.04]
    summary = summarize(study["results"])
    assert list(one(top, "median-error").get_ydata()) == [
        r["med_position_error"] for r in summary
    ]
    assert list(one(bottom, "median-error").get_ydata()) == [
        r["med_mass_error"] for r in summary
    ]


@pytest.mark.parametrize("theme", THEMES)
def test_every_visible_colour_is_in_the_palette(theme: str) -> None:
    study = load_study(ROOT / "assets/echolocation_range_study.json")
    fig, colors = _drawn(study, theme)
    assert visible_colours(fig) <= palette_hexes(colors)


_MAGENTA = {"figure.facecolor": "magenta", "savefig.facecolor": "magenta"}


def test_write_summary_figure_saves_the_light_range_study(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = literal_study()
    path = tmp_path / "study.png"
    built: list[tuple] = []
    saved: list[tuple] = []
    save_faces: list[str] = []
    real_build = clocks._echo_study.range_study_figure
    real_save = clocks._echo_study.save_png

    def build_spy(passed_study, colors):
        face = to_hex(matplotlib.rcParams["figure.facecolor"])
        fig = real_build(passed_study, colors)
        built.append((passed_study, colors, fig, face))
        return fig

    def save_spy(fig, target):
        saved.append((fig, target))
        save_faces.append(to_hex(matplotlib.rcParams["figure.facecolor"]))
        real_save(fig, target)

    monkeypatch.setattr(clocks._echo_study, "range_study_figure", build_spy)
    monkeypatch.setattr(clocks._echo_study, "save_png", save_spy)
    with matplotlib.rc_context(_MAGENTA):
        clocks._echo_study.write_summary_figure(study, path)
    assert len(built) == 1
    passed_study, colors, fig, face = built[0]
    assert passed_study is study
    assert colors == palette("light")
    assert face == palette("light").background
    assert len(saved) == 1
    assert saved[0][0] is fig
    assert saved[0][1] == path
    assert save_faces == [palette("light").background]


def test_write_summary_figure_output(tmp_path: Path) -> None:
    path = tmp_path / "study.png"
    with matplotlib.rc_context(_MAGENTA):
        clocks._echo_study.write_summary_figure(literal_study(), path)
    assert png_size(path) == (1500, 1080)
    pixel = matplotlib.image.imread(path)[0, 0]
    np.testing.assert_allclose(
        pixel[:3], to_rgb(palette("light").background), rtol=0, atol=1 / 255
    )
    assert pixel[3] == 1.0


def test_viz_exports_the_range_study_figure() -> None:
    assert clocks.viz.range_study_figure is clocks._echo_study.range_study_figure
    assert "range_study_figure" in clocks.viz.__all__
