"""The clips drawn from replays: their parts, every clip's frames, render_clip."""

import importlib
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from _clip_replays import (
    ARTISTS,
    DOTS,
    ELLIPSES,
    INSETS,
    TRUTHS,
    committed,
    edited,
    one_particle,
    shortest,
    small,
)
from _figure_artists import (
    all_axes,
    assert_hidden,
    assert_marks_drawn,
    assert_shown,
    band_edges,
    off_palette,
    one,
    opaque_fills_above,
)
from matplotlib import patheffects
from matplotlib.collections import LineCollection
from matplotlib.colors import to_hex, to_rgba
from matplotlib.figure import Figure
from matplotlib.markers import MarkerStyle

import clocks._clip_parts
import clocks._clips
import clocks.viz
from clocks._clip_maps import clock_fill
from clocks._clip_parts import (
    Clip,
    ellipse_points,
    label_masses,
    running_slowing,
    status_text,
    zoom_label,
    zoom_window,
)
from clocks._clip_tracks import slowing_limits
from clocks._clips import (
    BUILDERS,
    CLIP_HEIGHTS,
    Shot,
    build_clip,
    final_frame,
    render_clip,
    shots,
)
from clocks._figures import trace_header, trace_limits
from clocks._replay import Replay
from clocks._style import THEMES, Palette, frame_rgba, palette, style
from clocks._video import EncoderError, EncoderNotFound, StreamInfo, dwell_frames, probe

CLIPS = sorted(BUILDERS)


def _rows(table: dict[str, list[tuple]]) -> list:
    """The table's rows for the clips in ``CLIPS``, with IDs from the row."""
    return [
        pytest.param(demo, *row, id=f"{demo}: {row[0]} {row[1]}")
        for demo in CLIPS
        for row in table.get(demo, [])
    ]


@contextmanager
def built(replay: Replay, theme: str = "light") -> Iterator[tuple[Clip, Palette]]:
    """The clip of ``replay`` built in ``theme``; its figure closed afterwards."""
    with style(theme) as colors:
        clip = build_clip(replay, colors)
        try:
            yield clip, colors
        finally:
            plt.close(clip.fig)


def axes_named(fig: Figure, gid: str):
    matches = [ax for ax in all_axes(fig) if ax.get_gid() == gid]
    assert len(matches) == 1, f"expected one axes {gid!r}, got {len(matches)}"
    return matches[0]


def status(fig: Figure):
    matches = [text for text in fig.texts if text.get_gid() == "status"]
    assert len(matches) == 1, f"expected one status text, got {len(matches)}"
    return matches[0]


def fetch(fig: Figure, axes: str | None, gid: str):
    if axes is None:
        matches = [text for text in fig.texts if text.get_gid() == gid]
        assert len(matches) == 1, f"expected one figure text {gid!r}"
        return matches[0]
    return one(axes_named(fig, axes), gid)


def visible_texts(ax, content: str) -> list:
    return [t for t in ax.texts if t.get_visible() and t.get_text() == content]


def magnifications(parent, inset) -> tuple[float, float]:
    """Each axis's inset size over the parent's, times the spans' ratio."""
    outer, inner = parent.get_position(), inset.get_position()
    x = (inner.width / outer.width) * np.ptp(parent.get_xlim())
    y = (inner.height / outer.height) * np.ptp(parent.get_ylim())
    return x / np.ptp(inset.get_xlim()), y / np.ptp(inset.get_ylim())


def series(fig: Figure, name: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Copies of the data of every cloud of ``DOTS[name]``."""
    found = []
    for axes, gid, _, _ in DOTS.get(name, []):
        x, y = fetch(fig, axes, gid).get_data()
        found.append((np.array(x, dtype=float), np.array(y, dtype=float)))
    return found


def archive_series(replay: Replay, name: str, row: int) -> list:
    cloud = replay.arrays["cloud"][row]
    return [
        (cloud[:, cx].astype(np.float64), cloud[:, cy].astype(np.float64))
        for _, _, (cx, cy), _ in DOTS.get(name, [])
    ]


def nothing_to_draw(artist, kind: str) -> bool:
    if kind == "band":
        return len(artist.get_paths()) == 0
    if kind in ("line", "marks"):
        points = len(np.atleast_1d(artist.get_xdata()))
        return points <= (1 if kind == "line" else 0)
    return False


def shows(when: str, row: int) -> bool:
    return {"always": True, "after the prior": row > 0, "only the prior": row == 0}[
        when
    ]


def faces(collection) -> np.ndarray:
    return np.array(collection.get_facecolors(), dtype=float)


def rgba(colour: str) -> np.ndarray:
    return np.array(to_rgba(colour))


# ---------------------------------------------------------------- parts


def test_status_text() -> None:
    assert status_text(0) == "Before any observation"
    assert status_text(1) == "After 1 observation"
    assert status_text(2) == "After 2 observations"
    assert status_text(50) == "After 50 observations"


def test_running_slowing() -> None:
    running = running_slowing(np.array([[0.99, 0.98], [0.97, 1.00]]))
    assert running.shape == (3, 2)
    assert np.all(np.isnan(running[0]))
    np.testing.assert_allclose(running[1:], [[0.01, 0.02], [0.02, 0.01]], rtol=1e-12)


def test_ellipse_points_lie_two_standard_deviations_out() -> None:
    mean = np.array([1.0, -2.0])
    cov = np.array([[0.04, 0.018], [0.018, 0.09]])
    points = ellipse_points(mean, cov)
    assert points.shape == (121, 2)
    np.testing.assert_allclose(points[0], points[-1], rtol=0, atol=1e-12)
    offsets = points - mean
    distances = np.einsum("ni,ij,nj->n", offsets, np.linalg.inv(cov), offsets)
    np.testing.assert_allclose(distances, 4.0, rtol=1e-9)
    turn = np.unwrap(np.arctan2(offsets[:, 1], offsets[:, 0]))
    assert abs(turn[-1] - turn[0]) == pytest.approx(2 * np.pi, rel=1e-9)
    # Small steps all the way: an arc closed by a jump back to its first point
    # also unwraps to a full turn, through one large step.
    assert np.max(np.abs(np.diff(turn))) < 0.2


def test_ellipse_points_of_no_spread() -> None:
    mean = np.array([1.5, -1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        points = ellipse_points(mean, np.zeros((2, 2)))
    assert points.shape == (121, 2)
    np.testing.assert_array_equal(points, np.tile(mean, (121, 1)))


def test_zoom_window() -> None:
    prior = [[-8.0, 8.0], [-8.0, 8.0]]
    window = zoom_window(
        np.array([1.5, -1.0]), np.array([1.42, -1.22]), np.array([0.11, 0.12]), prior
    )
    np.testing.assert_allclose(
        window, [[1.5 - 0.375, 1.5 + 0.375], [-1.0 - 0.575, -1.0 + 0.575]], rtol=1e-12
    )
    on_truth = zoom_window(
        np.array([1.5, -1.0]), np.array([1.5, -1.22]), np.array([0.0, 0.12]), prior
    )
    np.testing.assert_allclose(on_truth[0], [1.5 - 0.16, 1.5 + 0.16], rtol=1e-12)


def test_zoom_label() -> None:
    assert zoom_label(12.2, 11.8, "y") == "zoom ×12"
    assert zoom_label(12.4, 8.6, "M") == "zoom ×12 (x), ×9 (M)"


@pytest.mark.parametrize("theme", THEMES)
def test_clock_fill(theme: str) -> None:
    colors = palette(theme)
    background, ink = rgba(colors.background), rgba(colors.ink)
    fills = clock_fill(np.array([0.0, 0.5, 1.0, 2.0, -0.5, np.nan]), 1.0, colors)
    expected = [background, (background + ink) / 2, ink, ink, background]
    assert len(fills) == 6
    for fill, colour in zip(fills[:5], expected, strict=True):
        np.testing.assert_allclose(fill[:3], colour[:3], rtol=0, atol=1e-12)
        assert fill[3] == 1.0
    assert tuple(fills[5]) == (0, 0, 0, 0)


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize(
    ("values", "largest"),
    [
        pytest.param([-0.2, 0.0, np.nan], 0.0, id="zero"),
        pytest.param([-0.2, -0.1, np.nan], -0.1, id="negative"),
    ],
)
def test_clock_fill_when_no_clock_is_slow(
    theme: str, values: list[float], largest: float
) -> None:
    colors = palette(theme)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fills = clock_fill(np.array(values), largest, colors)
    for fill in fills[:2]:
        np.testing.assert_allclose(fill[:3], rgba(colors.background)[:3], atol=1e-12)
        assert fill[3] == 1.0
    assert tuple(fills[2]) == (0, 0, 0, 0)


def test_assert_shown_needs_one_item_that_draws() -> None:
    fig = plt.figure()
    try:
        ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
        segments = [[(0, 0), (0, 0)], [(0, 0), (1, 1)]]
        opaque, clear = (0, 0, 0, 1), (0, 0, 0, 0)
        wrong = ax.add_collection(LineCollection(segments, colors=[opaque, clear]))
        right = ax.add_collection(LineCollection(segments, colors=[clear, opaque]))
        dots = dict(facecolors=[opaque, clear], edgecolors="none")
        small_first = ax.scatter([0, 1], [0, 1], s=[0, 20], **dots)
        large_first = ax.scatter([0, 1], [0, 1], s=[20, 0], **dots)
        fig.canvas.draw()
        with pytest.raises(AssertionError):
            assert_shown(wrong, "segments")
        assert_shown(right, "segments")
        with pytest.raises(AssertionError):
            assert_shown(small_first, "scatter")
        assert_shown(large_first, "scatter")
    finally:
        plt.close(fig)


# ---------------------------------------------------------------- every clip


@pytest.mark.parametrize("name", CLIPS)
def test_frame_is_1500_by_twice_the_declared_height(name: str) -> None:
    with built(small(name)) as (clip, _):
        assert tuple(clip.fig.get_size_inches()) == (7.5, CLIP_HEIGHTS[name] / 100)
        assert len(frame_rgba(clip.fig)) == 1500 * 2 * CLIP_HEIGHTS[name] * 4


def _placement(fig: Figure) -> dict[str, tuple]:
    placement = {}
    for ax in all_axes(fig):
        gid = ax.get_gid()
        assert gid is not None, "an axes has no gid"
        assert gid not in placement, f"axes gid {gid!r} repeats"
        limits = [ax.get_xlim(), ax.get_ylim()]
        if hasattr(ax, "get_zlim"):
            limits.append(ax.get_zlim())
        placement[gid] = (ax.get_position().bounds, *limits)
    return placement


@pytest.mark.parametrize("name", CLIPS)
def test_axes_never_move(name: str) -> None:
    with built(small(name)) as (clip, _):
        clip.show(0)
        clip.fig.canvas.draw()
        first = _placement(clip.fig)
        assert first
        for row in (1, 4, 9):
            clip.show(row)
            clip.fig.canvas.draw()
            assert _placement(clip.fig) == first, f"row {row}"


@pytest.mark.parametrize("name", CLIPS)
def test_panels_are_on_the_canvas_and_apart(name: str) -> None:
    with built(small(name)) as (clip, _):
        boxes = {ax.get_gid(): ax.get_position(original=True) for ax in clip.fig.axes}
    for gid, box in boxes.items():
        assert 0 <= box.x0 and box.x1 <= 1 and 0 <= box.y0 and box.y1 <= 1, gid
    names = list(boxes)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            p, q = boxes[first], boxes[second]
            width = max(0.0, min(p.x1, q.x1) - max(p.x0, q.x0))
            height = max(0.0, min(p.y1, q.y1) - max(p.y0, q.y0))
            assert width * height <= 1e-9, f"{first} and {second} overlap"


@pytest.mark.parametrize("name", CLIPS)
def test_the_layout_engine_stays_off(name: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with built(small(name)) as (clip, _):
            noted = {id(ax): ax.get_position().bounds for ax in all_axes(clip.fig)}
            clip.fig.canvas.draw()
            for row in (0, 1, 2):
                clip.show(row)
                frame_rgba(clip.fig)
            engine = type(clip.fig.get_layout_engine()).__name__
            after = {id(ax): ax.get_position().bounds for ax in all_axes(clip.fig)}
    assert engine == "PlaceHolderLayoutEngine"
    assert after.keys() == noted.keys()
    for key, bounds in noted.items():
        assert after[key] == pytest.approx(bounds, abs=1e-12)


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("name", CLIPS)
def test_listed_artists_are_drawn(name: str, theme: str) -> None:
    with built(small(name), theme) as (clip, _):
        for row in (0, 1, 9):
            clip.show(row)
            clip.fig.canvas.draw()
            for axes, gid, kind, when in ARTISTS[name]:
                artist = fetch(clip.fig, axes, gid)
                if shows(when, row):
                    assert_shown(artist, kind)
                elif not nothing_to_draw(artist, kind):
                    assert_hidden(artist)


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("name", CLIPS)
def test_status_line(name: str, theme: str) -> None:
    expected = {0: "Before any observation", 1: "After 1 observation"}
    expected[9] = "After 9 observations"
    with built(small(name), theme) as (clip, colors):
        for row, text in expected.items():
            clip.show(row)
            line = status(clip.fig)
            assert line.get_text() == text
            assert line.get_visible()
            assert to_hex(line.get_color()) == colors.ink
            x, y = clip.fig.transFigure.inverted().transform(
                line.get_transform().transform(line.get_position())
            )
            assert x < 0.1 and y > 0.9


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("name", CLIPS)
def test_colours_are_the_palette_or_an_ink_blend(name: str, theme: str) -> None:
    with built(small(name), theme) as (clip, colors):
        for row in (0, 9):
            clip.show(row)
            clip.fig.canvas.draw()
            assert off_palette(clip.fig, colors) == set(), f"row {row}"


@pytest.mark.parametrize(("name", "axes", "gid", "columns", "role"), _rows(DOTS))
def test_dots_are_the_archive_row(
    name: str, axes: str, gid: str, columns: tuple[int, int], role: str
) -> None:
    replay = small(name)
    cx, cy = columns
    with built(replay) as (clip, colors):
        for row in (0, 4, 9):
            clip.show(row)
            line = fetch(clip.fig, axes, gid)
            x, y = line.get_data()
            cloud = replay.arrays["cloud"][row]
            np.testing.assert_array_equal(x, cloud[:, cx].astype(np.float64))
            np.testing.assert_array_equal(y, cloud[:, cy].astype(np.float64))
            assert_marks_drawn(line)
            assert to_hex(line.get_color()) == getattr(colors, role)
            assert opaque_fills_above(axes_named(clip.fig, axes), line) == []


def _trace_names(name: str) -> list[str]:
    return list(small(name).meta["param_names"])


TRACE_CLIPS = [
    name
    for name in CLIPS
    if any(axes and axes.startswith("trace-") for axes, *_ in ARTISTS[name])
]


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("name", TRACE_CLIPS)
def test_traces_follow_the_archive(name: str, theme: str) -> None:
    replay = small(name)
    mean = replay.arrays["mean"]
    sd = np.sqrt(np.diagonal(replay.arrays["cov"], axis1=1, axis2=2))
    priors = replay.meta["prior_ranges"]
    last = len(mean) - 1
    with built(replay, theme) as (clip, colors):
        for row in (0, 4, 9):
            clip.show(row)
            for j, param in enumerate(_trace_names(name)):
                ax = axes_named(clip.fig, f"trace-{param}")
                truth = float(replay.arrays["true_params"][j])
                line = one(ax, "trace-mean")
                steps = np.arange(row + 1)
                np.testing.assert_array_equal(line.get_xdata(), steps)
                np.testing.assert_array_equal(line.get_ydata(), mean[: row + 1, j])
                band = one(ax, "trace-band")
                lower, upper = band_edges(band, steps)
                np.testing.assert_allclose(
                    lower, mean[: row + 1, j] - sd[: row + 1, j], rtol=1e-12
                )
                np.testing.assert_allclose(
                    upper, mean[: row + 1, j] + sd[: row + 1, j], rtol=1e-12
                )
                true_line = one(ax, "trace-truth")
                np.testing.assert_array_equal(true_line.get_xdata(), [0, last])
                np.testing.assert_array_equal(true_line.get_ydata(), [truth, truth])
                assert true_line.get_linestyle() == "-"
                assert ax.get_xlim() == (0, last)
                limits = trace_limits(mean[:, j], sd[:, j], truth, priors[j])
                assert ax.get_ylim() == pytest.approx(limits, rel=1e-12)
                header = trace_header(param, truth, mean[row, j], sd[row, j], priors[j])
                assert ax.get_title() == header
                assert to_hex(line.get_color()) == colors.belief
                assert to_hex(band.get_facecolor()[0]) == colors.belief
                assert to_hex(true_line.get_color()) == colors.truth


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize(("name", "axes", "gid", "entries"), _rows(TRUTHS))
def test_truth_marks_sit_on_the_truth(
    name: str, axes: str, gid: str, entries: tuple[int, int], theme: str
) -> None:
    replay = small(name)
    i, j = entries
    with built(replay, theme) as (clip, colors):
        mark = fetch(clip.fig, axes, gid)
        truth = replay.arrays["true_params"]
        np.testing.assert_array_equal(mark.get_xdata(), [truth[i]])
        np.testing.assert_array_equal(mark.get_ydata(), [truth[j]])
        assert to_hex(mark.get_markeredgecolor()) == colors.truth
        assert mark.get_markerfacecolor() == "none"
        assert mark.get_visible()
        assert mark.get_markersize() > 0


@pytest.mark.parametrize(("name", "axes", "gid", "block", "role"), _rows(ELLIPSES))
def test_ellipse_is_the_two_standard_deviation_contour(
    name: str, axes: str, gid: str, block: tuple[int, int], role: str
) -> None:
    replay = small(name)
    pair = list(block)
    with built(replay) as (clip, colors):
        for row in (1, 4, 9):
            clip.show(row)
            line = fetch(clip.fig, axes, gid)
            points = np.column_stack(line.get_data()).astype(float)
            m = replay.arrays["mean"][row][pair]
            spread = replay.arrays["cov"][row][np.ix_(pair, pair)]
            offsets = points - m
            distances = np.einsum(
                "ni,ij,nj->n", offsets, np.linalg.inv(spread), offsets
            )
            assert len(points) == 121
            np.testing.assert_allclose(distances, 4.0, rtol=1e-9)
            assert to_hex(line.get_color()) == getattr(colors, role)
        assert len(visible_texts(axes_named(clip.fig, axes), "covariance, 2 s.d.")) == 1


@pytest.mark.parametrize(
    ("name", "parent", "inset_gid", "block", "y_name", "copied"), _rows(INSETS)
)
def test_inset_is_the_fixed_window_on_the_truth(
    name: str,
    parent: str,
    inset_gid: str,
    block: tuple[int, int],
    y_name: str,
    copied: tuple[str, ...],
) -> None:
    replay = small(name)
    pair = list(block)
    arrays = replay.arrays
    sd = np.sqrt(np.diagonal(arrays["cov"][-1]))
    priors = replay.meta["prior_ranges"]
    window = zoom_window(
        arrays["true_params"][pair],
        arrays["mean"][-1][pair],
        sd[pair],
        [priors[i] for i in pair],
    )
    with built(replay) as (clip, _):
        outer = axes_named(clip.fig, parent)
        inset = axes_named(clip.fig, inset_gid)
        assert inset in outer.child_axes
        for row in (0, 4, 9):
            clip.show(row)
            np.testing.assert_allclose(inset.get_xlim(), window[0], rtol=1e-12)
            np.testing.assert_allclose(inset.get_ylim(), window[1], rtol=1e-12)
            for gid in copied:
                original = np.asarray(one(outer, gid).get_data())
                copy = np.asarray(one(inset, f"{inset_gid}-{gid}").get_data())
                np.testing.assert_array_equal(copy, original)
        clip.fig.canvas.draw()
        assert len(inset.texts) == 1
        label = zoom_label(*magnifications(outer, inset), y_name)
        assert inset.texts[0].get_text() == label


@pytest.mark.parametrize("name", TRACE_CLIPS)
def test_changing_the_last_mean_moves_the_trace_and_the_header(name: str) -> None:
    original = committed(name)
    width = np.ptp(original.meta["prior_ranges"][0])
    mean = original.arrays["mean"].copy()
    mean[-1, 0] += 0.01 * width
    changed = edited(original, mean=mean)
    first, *others = [f"trace-{param}" for param in original.meta["param_names"]]
    last = len(mean) - 1
    drawn = []
    for replay in (original, changed):
        with built(replay) as (clip, _):
            clip.show(last)
            data = {
                gid: np.asarray(one(axes_named(clip.fig, gid), "trace-mean").get_data())
                for gid in (first, *others)
            }
            title = axes_named(clip.fig, first).get_title()
            drawn.append((data, title))
    (before, title_before), (after, title_after) = drawn
    assert after[first][1, -1] - before[first][1, -1] == pytest.approx(
        0.01 * width, rel=1e-9
    )
    np.testing.assert_array_equal(after[first][:, :-1], before[first][:, :-1])
    for gid in others:
        np.testing.assert_array_equal(after[gid], before[gid])
    assert title_after != title_before


@pytest.mark.parametrize(("name", "axes", "gid", "columns", "role"), _rows(DOTS))
def test_changing_a_middle_cloud_row_moves_only_that_rows_dots(
    name: str, axes: str, gid: str, columns: tuple[int, int], role: str
) -> None:
    original = committed(name)
    last = len(original.arrays["mean"]) - 1
    middle = last // 2
    cloud = original.arrays["cloud"].copy()
    cloud[middle] += 0.5
    changed = edited(original, cloud=cloud)
    rows = (0, middle - 1, middle, middle + 1, last)
    drawn = []
    for replay in (original, changed):
        with built(replay) as (clip, _):
            seen = {}
            for row in rows:
                clip.show(row)
                seen[row] = np.array(fetch(clip.fig, axes, gid).get_data())
            drawn.append(seen)
    before, after = drawn
    assert not np.array_equal(after[middle], before[middle])
    for row in (0, middle - 1, middle + 1, last):
        np.testing.assert_array_equal(after[row], before[row])


@pytest.mark.parametrize("name", CLIPS)
def test_a_one_observation_frame_is_not_blank(name: str) -> None:
    with style("light") as colors:
        fig = final_frame(shortest(name), colors)
        try:
            pixels = np.frombuffer(frame_rgba(fig), dtype=np.uint32)
        finally:
            plt.close(fig)
    assert len(np.unique(pixels)) > 100


@pytest.mark.parametrize("name", CLIPS)
def test_a_one_particle_run_draws(name: str) -> None:
    replay = one_particle(name)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with built(replay) as (clip, _):
            for row in (0, 1, 2):
                clip.show(row)
                assert len(frame_rgba(clip.fig)) > 0


@pytest.mark.parametrize("name", CLIPS)
def test_shots_follow_the_dwell_schedule(name: str) -> None:
    expected = [Shot(k, n, None) for k, n in enumerate(dwell_frames(9))]
    assert shots(small(name)) == expected


@pytest.mark.parametrize("name", CLIPS)
def test_final_frame_shows_the_last_row(name: str) -> None:
    replay = small(name)
    with style("light") as colors:
        fig = final_frame(replay, colors)
        try:
            assert status(fig).get_text() == "After 9 observations"
            for (x, y), (cx, cy) in zip(
                series(fig, name), archive_series(replay, name, 9), strict=True
            ):
                np.testing.assert_array_equal(x, cx)
                np.testing.assert_array_equal(y, cy)
            for ax in all_axes(fig):
                if (ax.get_gid() or "").startswith("trace-"):
                    assert len(one(ax, "trace-mean").get_xdata()) == 10
        finally:
            plt.close(fig)


@pytest.mark.parametrize("name", CLIPS)
def test_render_clip_hands_the_encoder_each_shot_drawn(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replay = small(name)
    log: list[str] = []
    notes: list[dict] = []
    same: list[bool] = []
    repeats: list[int] = []
    sizes: list[tuple[int, int]] = []
    latest: list[bytes] = []
    real_frame_rgba = clocks._clips.frame_rgba

    def spy(fig: Figure) -> bytes:
        scenes = [ax for ax in all_axes(fig) if ax.get_gid() == "scene"]
        notes.append(
            {
                "status": status(fig).get_text(),
                "series": series(fig, name),
                "azimuth": scenes[0].azim if scenes else None,
            }
        )
        log.append("drew")
        frame = real_frame_rgba(fig)
        latest[:] = [frame]
        return frame

    def stand_in(frames, path, *, width, height, encoder=None):
        log.append("entered")
        sizes.append((width, height))
        total = 0
        for frame, repeat in frames:
            log.append("took")
            same.append(bool(latest) and frame is latest[0])
            latest.clear()
            repeats.append(repeat)
            total += repeat
        return total

    monkeypatch.setattr(clocks._clips, "frame_rgba", spy)
    monkeypatch.setattr(clocks._clips, "encode", stand_in)
    count = render_clip(replay, tmp_path / "x.mp4", encoder="unused")

    expected = shots(replay)
    assert count == sum(dwell_frames(9))
    assert log == ["entered"] + ["drew", "took"] * len(expected)
    assert same == [True] * len(expected)
    assert repeats == [shot.repeat for shot in expected]
    assert sizes == [(1500, 2 * CLIP_HEIGHTS[name])]
    for note, shot in zip(notes, expected, strict=True):
        assert note["status"] == status_text(shot.row)
        for (x, y), (cx, cy) in zip(
            note["series"], archive_series(replay, name, shot.row), strict=True
        ):
            np.testing.assert_array_equal(x, cx)
            np.testing.assert_array_equal(y, cy)
        if shot.azimuth is not None:
            assert note["azimuth"] == pytest.approx(shot.azimuth, abs=1e-9)
    assert not (tmp_path / "x.mp4").exists()


@pytest.mark.parametrize("name", CLIPS)
def test_render_clip_encodes_every_shot(name: str, tmp_path: Path) -> None:
    module = importlib.import_module(f"clocks._demos.{name}")
    two = module.record(n_observations=2, n_particles=16)
    path = tmp_path / "clip.mp4"
    assert render_clip(two, path) == 92
    height = 2 * CLIP_HEIGHTS[name]
    assert probe(path) == StreamInfo("h264", "yuv420p", 1500, height, 20.0, 92, True)


@pytest.mark.parametrize("theme", THEMES)
def test_render_clip_draws_in_the_theme_asked_for(
    theme: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kept: list[bytes] = []

    def stand_in(frames, path, *, width, height, encoder=None):
        frame, _ = next(iter(frames))
        kept.append(frame)
        return 0

    monkeypatch.setattr(clocks._clips, "encode", stand_in)
    render_clip(small("demo_2d"), tmp_path / "x.mp4", theme=theme, encoder="unused")
    background = to_rgba(palette(theme).background)
    assert tuple(kept[0][:4]) == (*(round(255 * c) for c in background[:3]), 255)


def test_render_clip_closes_its_figure_when_the_encoder_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def stand_in(frames, path, *, width, height, encoder=None):
        next(iter(frames))
        raise EncoderError("no")

    monkeypatch.setattr(clocks._clips, "encode", stand_in)
    before = plt.get_fignums()
    with pytest.raises(EncoderError, match="no"):
        render_clip(small("demo_2d"), tmp_path / "x.mp4", encoder="unused")
    assert plt.get_fignums() == before


def test_render_clip_looks_for_the_encoder_before_drawing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing() -> str:
        raise EncoderNotFound("none")

    def no_drawing(*args: object, **kwargs: object) -> Clip:
        pytest.fail("the clip was drawn before the encoder was found")

    monkeypatch.setattr(clocks._clips, "find_encoder", missing)
    monkeypatch.setattr(clocks._clips, "build_clip", no_drawing)
    with pytest.raises(EncoderNotFound):
        render_clip(small("demo_2d"), tmp_path / "x.mp4")


def test_build_clip_refuses_a_demo_without_a_clip() -> None:
    with style("light") as colors:
        with pytest.raises(ValueError, match="no clip for demo 'demo_density'"):
            build_clip(committed("demo_density"), colors)


# ---------------------------------------------------------------- the 2D clip


def test_2d_map_spans_the_prior_and_is_square() -> None:
    with built(small("demo_2d")) as (clip, _):
        ax = axes_named(clip.fig, "map")
        assert ax.get_xlim() == (-8.0, 8.0)
        assert ax.get_ylim() == (-8.0, 8.0)
        assert ax.get_aspect() == 1.0
        box = ax.get_position(original=True)
        assert 7.5 * box.width == pytest.approx(
            CLIP_HEIGHTS["demo_2d"] / 100 * box.height, rel=1e-9
        )


@pytest.mark.parametrize("theme", THEMES)
def test_2d_clock_fills_follow_the_running_mean(theme: str) -> None:
    replay = committed("demo_2d")
    observations = replay.arrays["observations"]
    last = len(observations)
    running = np.cumsum(1.0 - observations, axis=0) / np.arange(1, last + 1)[:, None]
    largest = running.max()
    ratios = running[-1] / largest
    assert ratios.max() < 1
    with built(replay, theme) as (clip, colors):
        background, ink = rgba(colors.background), rgba(colors.ink)
        clocks_artist = one(axes_named(clip.fig, "map"), "clocks")
        clip.show(0)
        assert np.all(faces(clocks_artist)[:, 3] == 0)
        for edge in clocks_artist.get_edgecolors():
            assert to_hex(edge) == colors.ink
        clip.show(last)
        expected = background + ratios[:, None] * (ink - background)
        np.testing.assert_allclose(
            faces(clocks_artist)[:, :3], expected[:, :3], atol=1e-9
        )
        np.testing.assert_allclose(faces(clocks_artist)[:, 3], 1.0, atol=1e-9)
        k, c = np.unravel_index(running.argmax(), running.shape)
        clip.show(int(k) + 1)
        np.testing.assert_allclose(faces(clocks_artist)[c], ink, atol=1e-9)


@pytest.mark.parametrize("theme", THEMES)
def test_2d_clocks_are_squares_at_the_clock_positions(theme: str) -> None:
    replay = small("demo_2d")
    square = MarkerStyle("s")
    path = square.get_path().transformed(square.get_transform())
    with built(replay, theme) as (clip, colors):
        clocks_artist = one(axes_named(clip.fig, "map"), "clocks")
        for row in (0, 9):
            clip.show(row)
            np.testing.assert_array_equal(
                clocks_artist.get_offsets(), replay.arrays["clock_positions"]
            )
            (drawn,) = clocks_artist.get_paths()
            np.testing.assert_array_equal(drawn.vertices, path.vertices)
            np.testing.assert_array_equal(drawn.codes, path.codes)
            assert np.all(clocks_artist.get_sizes() > 0)
            for edge in clocks_artist.get_edgecolors():
                assert to_hex(edge) == colors.ink
            assert np.all(np.asarray(clocks_artist.get_linewidths()) > 0)


@pytest.mark.parametrize("reading", [1.0, 1.01], ids=["at_one", "above_one"])
def test_2d_all_readings_at_or_above_one(reading: float) -> None:
    base = small("demo_2d")
    replay = edited(
        base, observations=np.full_like(base.arrays["observations"], reading)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with built(replay) as (clip, colors):
            clocks_artist = one(axes_named(clip.fig, "map"), "clocks")
            clip.show(0)
            assert np.all(faces(clocks_artist)[:, 3] == 0)
            for row in (1, 4, 9):
                clip.show(row)
                fills = faces(clocks_artist)
                np.testing.assert_allclose(
                    fills[:, :3],
                    np.tile(rgba(colors.background)[:3], (len(fills), 1)),
                    atol=1e-12,
                )
                np.testing.assert_array_equal(fills[:, 3], 1.0)
                assert len(frame_rgba(clip.fig)) > 0


def test_2d_a_reading_above_one_gives_a_background_fill() -> None:
    base = small("demo_2d")
    observations = base.arrays["observations"].copy()
    observations[0, 2] = 1.02
    replay = edited(base, observations=observations)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with built(replay) as (clip, colors):
            clip.show(1)
            fill = faces(one(axes_named(clip.fig, "map"), "clocks"))[2]
            np.testing.assert_allclose(fill, rgba(colors.background), atol=1e-12)


def test_2d_labels() -> None:
    with built(small("demo_2d")) as (clip, colors):
        texts = visible_texts(
            axes_named(clip.fig, "map"), "stronger fill = slower clock"
        )
        assert len(texts) == 1
        assert to_hex(texts[0].get_color()) == colors.ink


def test_viz_exports_render_clip() -> None:
    assert clocks.viz.render_clip is clocks._clips.render_clip
    assert "render_clip" in clocks.viz.__all__


# ---------------------------------------------------------------- the track clips

TRACKS = ("demo_1d", "demo_multi_mass")
# The clips whose two masses are labelled by label_masses; Task 5 adds its map.
LABELLED = ["demo_multi_mass"]
OFFSETS = [(9, 7), (-9, 7), (9, -7), (-9, -7), (9, 0), (-9, 0), (0, 7), (0, -7)]
LABEL_SIZE = 8.5


def _bars(collection) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each vertical segment's x, lower y and upper y."""
    segments = np.array(collection.get_segments(), dtype=float)
    assert segments.ndim == 3 and segments.shape[1:] == (2, 2)
    np.testing.assert_array_equal(segments[:, 0, 0], segments[:, 1, 0])
    return segments[:, 0, 0], segments[:, :, 1].min(axis=1), segments[:, :, 1].max(1)


def _running_means(observations: np.ndarray) -> np.ndarray:
    last = len(observations)
    return np.cumsum(1.0 - observations, axis=0) / np.arange(1, last + 1)[:, None]


@pytest.mark.parametrize("name", TRACKS)
def test_track_panels_share_the_x_scale(name: str) -> None:
    replay = small(name)
    priors = replay.meta["prior_ranges"]
    mass = TRUTHS[name][0][2][1]
    with built(replay) as (clip, _):
        panels = [axes_named(clip.fig, gid) for gid in ("cloud", "strip", "slowing")]
        first = panels[0].get_position(original=True)
        for ax in panels:
            assert ax.get_xlim() == pytest.approx(tuple(priors[0]), rel=1e-12)
            box = ax.get_position(original=True)
            assert box.x0 == pytest.approx(first.x0, rel=1e-12)
            assert box.width == pytest.approx(first.width, rel=1e-12)
        assert panels[0].get_ylim() == pytest.approx(tuple(priors[mass]), rel=1e-12)


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("name", TRACKS)
def test_strip_shows_the_clocks(name: str, theme: str) -> None:
    replay = small(name)
    xs = replay.arrays["clock_positions"][:, 0]
    with built(replay, theme) as (clip, colors):
        mark = one(axes_named(clip.fig, "strip"), "clocks")
        np.testing.assert_array_equal(mark.get_xdata(), xs)
        np.testing.assert_array_equal(mark.get_ydata(), np.zeros(len(xs)))
        assert mark.get_marker() == "s"
        assert mark.get_linestyle() == "None"
        assert to_hex(mark.get_color()) == colors.ink
        assert to_hex(mark.get_markerfacecolor()) == colors.ink
        assert to_hex(mark.get_markeredgecolor()) == colors.ink
        assert mark.get_visible()
        assert mark.get_markersize() > 0


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("name", TRACKS)
def test_slowing_marks_follow_the_archive(name: str, theme: str) -> None:
    replay = small(name)
    arrays = replay.arrays
    xs = arrays["clock_positions"][:, 0]
    running = _running_means(arrays["observations"])
    noise = replay.meta["noise_std"]
    with built(replay, theme) as (clip, colors):
        ax = axes_named(clip.fig, "slowing")
        for row in (0, 1, 4, 9):
            clip.show(row)
            truth = one(ax, "slowing-truth")
            np.testing.assert_allclose(
                truth.get_ydata(), 1.0 - arrays["true_rates"], rtol=1e-12
            )
            assert to_hex(truth.get_color()) == colors.truth
            predicted = one(ax, "slowing-predicted")
            centre = 1.0 - arrays["predicted_mean"][row]
            spread = arrays["predicted_std"][row]
            np.testing.assert_allclose(
                predicted.get_xdata(), xs, rtol=1e-12, atol=1e-12
            )
            np.testing.assert_allclose(predicted.get_ydata(), centre, rtol=1e-12)
            bar = one(ax, "slowing-predicted-bar")
            x, lower, upper = _bars(bar)
            np.testing.assert_allclose(x, xs, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(lower, centre - spread, rtol=1e-12)
            np.testing.assert_allclose(upper, centre + spread, rtol=1e-12)
            assert to_hex(predicted.get_color()) == colors.belief
            assert {to_hex(c) for c in bar.get_colors()} == {colors.belief}
            observed = one(ax, "slowing-observed")
            observed_bar = one(ax, "slowing-observed-bar")
            if row == 0:
                assert not observed.get_visible()
                assert not observed_bar.get_visible()
                continue
            assert observed.get_visible() and observed_bar.get_visible()
            mean = running[row - 1]
            half = noise / np.sqrt(row)
            np.testing.assert_allclose(observed.get_ydata(), mean, rtol=1e-12)
            x, lower, upper = _bars(observed_bar)
            np.testing.assert_allclose(x, observed.get_xdata(), rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(lower, mean - half, rtol=1e-12)
            np.testing.assert_allclose(upper, mean + half, rtol=1e-12)
            assert to_hex(observed.get_color()) == colors.ink
            assert {to_hex(c) for c in observed_bar.get_colors()} == {colors.ink}


@pytest.mark.parametrize("name", TRACKS)
def test_slowing_marks_stand_side_by_side(name: str) -> None:
    replay = small(name)
    xs = replay.arrays["clock_positions"][:, 0]
    with built(replay) as (clip, _):
        clip.show(9)
        ax = axes_named(clip.fig, "slowing")
        truth = np.asarray(one(ax, "slowing-truth").get_xdata(), dtype=float)
        predicted = np.asarray(one(ax, "slowing-predicted").get_xdata(), dtype=float)
        observed = np.asarray(one(ax, "slowing-observed").get_xdata(), dtype=float)
    d = observed[0] - xs[0]
    assert d > 0
    np.testing.assert_allclose(truth, xs - d, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(predicted, xs, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(observed, xs + d, rtol=1e-12, atol=1e-12)
    assert 2 * d < np.min(np.diff(np.sort(xs)))


def _slowing_values(replay: Replay) -> np.ndarray:
    """Every value the slowing panel must hold, by the test's own arithmetic."""
    arrays = replay.arrays
    running = _running_means(arrays["observations"])
    half = replay.meta["noise_std"] / np.sqrt(np.arange(1, len(running) + 1))
    centre = 1.0 - arrays["predicted_mean"]
    return np.concatenate(
        [
            (1.0 - arrays["true_rates"]).ravel(),
            (centre - arrays["predicted_std"]).ravel(),
            (centre + arrays["predicted_std"]).ravel(),
            (running - half[:, None]).ravel(),
            (running + half[:, None]).ravel(),
        ]
    )


@pytest.mark.parametrize("name", TRACKS)
def test_slowing_limits_hold_every_row(name: str) -> None:
    replay = small(name)
    values = _slowing_values(replay)
    smallest, largest = values.min(), values.max()
    low, high = slowing_limits(replay)
    assert low < smallest and largest < high
    pad = 0.04 * (largest - smallest)
    assert (low, high) == pytest.approx((smallest - pad, largest + pad), rel=1e-12)
    with built(replay) as (clip, _):
        ax = axes_named(clip.fig, "slowing")
        assert ax.get_ylim() == pytest.approx(
            (smallest - pad, largest + pad), rel=1e-12
        )
        assert ax.get_ylabel() == "slowing (1 − rate)"
    std = replay.arrays["predicted_std"].copy()
    std[0] = (high - low) + 1.0
    wide = edited(replay, predicted_std=std)
    wide_low, wide_high = slowing_limits(wide)
    centre = 1.0 - wide.arrays["predicted_mean"][0]
    assert wide_low < np.min(centre - std[0])
    assert np.max(centre + std[0]) < wide_high
    assert wide_low < low and high < wide_high


@pytest.mark.parametrize("theme", THEMES)
def test_two_masses_are_told_apart_without_colour(theme: str) -> None:
    replay = small("demo_multi_mass")
    with built(replay, theme) as (clip, colors):
        ax = axes_named(clip.fig, "cloud")
        first, second = one(ax, "dots-1"), one(ax, "dots-2")
        assert first.get_marker() != second.get_marker()
        assert to_hex(first.get_color()) == colors.belief
        assert to_hex(second.get_color()) == colors.belief_alt
        renderer = clip.fig.canvas.get_renderer()
        truths = [
            ax.transData.transform(
                np.column_stack(one(ax, gid).get_data()).astype(float)
            )[0]
            for gid in ("truth-1", "truth-2")
        ]
        for n in (1, 2):
            texts = visible_texts(ax, f"mass {n}")
            assert len(texts) == 1
            assert to_hex(texts[0].get_color()) == colors.truth
            box = texts[0].get_window_extent(renderer)
            centre = np.array([(box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2])
            own = np.hypot(*(centre - truths[n - 1]))
            other = np.hypot(*(centre - truths[2 - n]))
            assert own < other


def _alignment(offset: tuple[int, int]) -> tuple[str, str]:
    dx, dy = offset
    ha = "left" if dx > 0 else "right" if dx < 0 else "center"
    va = "bottom" if dy > 0 else "top" if dy < 0 else "center"
    return ha, va


def _candidate_box(ax, text: str, point, offset: tuple[int, int]):
    """The padded window extent of a temporary label at ``offset``, removed after."""
    ha, va = _alignment(offset)
    trial = ax.annotate(
        text,
        point,
        xytext=offset,
        textcoords="offset points",
        ha=ha,
        va=va,
        fontsize=LABEL_SIZE,
    )
    try:
        return trial.get_window_extent(ax.figure.canvas.get_renderer()).padded(2)
    finally:
        trial.remove()


def _display_marks(ax) -> np.ndarray:
    data = [
        np.column_stack(line.get_data()).astype(float)
        for line in ax.get_lines()
        if len(line.get_xdata())
    ]
    return ax.transData.transform(np.vstack(data)) if data else np.empty((0, 2))


def _marks_in(box, marks: np.ndarray) -> int:
    inside = (
        (marks[:, 0] >= box.x0)
        & (marks[:, 0] <= box.x1)
        & (marks[:, 1] >= box.y0)
        & (marks[:, 1] <= box.y1)
    )
    return int(inside.sum())


def _touch(first, second) -> bool:
    return (
        first.x0 <= second.x1
        and second.x0 <= first.x1
        and first.y0 <= second.y1
        and second.y0 <= first.y1
    )


def _exclusions(box, ax, earlier: list) -> tuple[int, int, int]:
    """Leaves the panel; insets touched; earlier labels touched."""
    renderer = ax.figure.canvas.get_renderer()
    frame = ax.get_window_extent(renderer)
    inside = (
        frame.x0 <= box.x0
        and box.x1 <= frame.x1
        and frame.y0 <= box.y0
        and box.y1 <= frame.y1
    )
    insets = sum(
        _touch(box, inset.get_window_extent(renderer)) for inset in ax.child_axes
    )
    labels = sum(_touch(box, other) for other in earlier)
    return int(not inside), insets, labels


def _ranks(ax, point, text: str, earlier: list) -> list:
    """Per offset of ``OFFSETS``: (exclusions by kind, marks covered, padded box)."""
    marks = _display_marks(ax)
    ranks = []
    for offset in OFFSETS:
        box = _candidate_box(ax, text, point, offset)
        ranks.append((_exclusions(box, ax, earlier), _marks_in(box, marks), box))
    return ranks


def _choose(ranks: list) -> tuple[int, int]:
    best = min(range(len(ranks)), key=lambda p: (sum(ranks[p][0]), ranks[p][1], p))
    return OFFSETS[best]


@pytest.mark.parametrize("name", LABELLED)
def test_mass_labels_take_the_least_covered_offset(name: str) -> None:
    replay = small(name)
    truth = replay.arrays["true_params"]
    with built(replay) as (clip, _):
        renderer = clip.fig.canvas.get_renderer()
        earlier: list = []
        chosen = []
        for n, (axes, _, (i, j)) in enumerate(TRUTHS[name], start=1):
            label = fetch(clip.fig, axes, f"mass-label-{n}")
            assert label.get_text() == f"mass {n}"
            assert tuple(label.xy) == (truth[i], truth[j])
            assert label.get_zorder() == 6
            assert label.get_fontsize() == LABEL_SIZE
            ranks = _ranks(label.axes, label.xy, label.get_text(), earlier)
            assert tuple(label.xyann) == _choose(ranks)
            ha, va = _alignment(_choose(ranks))
            assert label.get_horizontalalignment() == ha
            assert label.get_verticalalignment() == va
            earlier.append(label.get_window_extent(renderer).padded(2))
            chosen.append(tuple(label.xyann))
        clip.show(0)
        after = [
            tuple(fetch(clip.fig, axes, f"mass-label-{n}").xyann)
            for n, (axes, _, _) in enumerate(TRUTHS[name], start=1)
        ]
    assert after == chosen


class _RecordedStroke(patheffects.withStroke):
    """A halo that keeps the keyword arguments it was made with."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.made_with = kwargs


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("name", LABELLED)
def test_mass_labels_have_a_background_halo(
    name: str, theme: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(clocks._clip_parts.patheffects, "withStroke", _RecordedStroke)
    with built(small(name), theme) as (clip, colors):
        for n, (axes, _, _) in enumerate(TRUTHS[name], start=1):
            effects = fetch(clip.fig, axes, f"mass-label-{n}").get_path_effects()
            assert len(effects) == 1
            assert type(effects[0]) is _RecordedStroke
            assert effects[0].made_with["linewidth"] == 3
            assert to_hex(effects[0].made_with["foreground"]) == colors.background


# Which exclusion each case gives the offset (9, 7): leaves, inset, other label.
_EXCLUSION_OF = {"panel_edge": 0, "inset": 1, "other_label": 2}


def _exclusion_case(ax, case: str) -> list[tuple[float, float]]:
    """The points to label; marks, and an inset where asked, drawn on ``ax``.

    A mark sits at the centre of every candidate box of the last point but the
    first offset's, so that offset alone covers none.
    """
    if case == "panel_edge":
        points = [(9.6, 5.0)]
    elif case == "inset":
        points = [(5.0, 5.0)]
        box = _candidate_box(ax, "mass 1", points[0], OFFSETS[0])
        corner = ax.transAxes.inverted().transform(
            [((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2)]
        )[0]
        ax.inset_axes((corner[0], corner[1], 0.1, 0.1))
    else:
        first = (3.0, 5.0)
        below = ax.transData.transform([first])[0] - (0.0, 4 * ax.figure.dpi / 72)
        points = [first, tuple(ax.transData.inverted().transform([below])[0])]
    text = f"mass {len(points)}"
    centres = []
    for offset in OFFSETS[1:]:
        box = _candidate_box(ax, text, points[-1], offset)
        centres.append(((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2))
    data = ax.transData.inverted().transform(centres)
    ax.plot(data[:, 0], data[:, 1], linestyle="none", marker="o")
    return points


@pytest.mark.parametrize("case", ["panel_edge", "inset", "other_label", "all_blocked"])
def test_label_masses_keeps_its_exclusions(case: str) -> None:
    fig = plt.figure(figsize=(7.5, 5))
    try:
        width = 0.02 if case == "all_blocked" else 0.8
        ax = fig.add_axes((0.1, 0.1, width, 0.8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        if case == "all_blocked":
            points = [(5.0, 5.0)]
        else:
            points = _exclusion_case(ax, case)
        earlier = []
        if case == "other_label":
            first = _ranks(ax, points[0], "mass 1", [])
            assert first[0][:2] == ((0, 0, 0), 0)
            earlier = [first[0][2]]
        ranks = _ranks(ax, points[-1], f"mass {len(points)}", earlier)
        if case == "all_blocked":
            assert all(sum(kinds) >= 1 for kinds, _, _ in ranks)
        else:
            kinds, covered, _ = ranks[0]
            assert covered == 0
            assert kinds == tuple(int(k == _EXCLUSION_OF[case]) for k in range(3))
            assert all(marks >= 1 for _, marks, _ in ranks[1:])
            assert any(sum(kinds) == 0 for kinds, _, _ in ranks[1:])
        expected = _choose(ranks)

        labels = label_masses(ax, points, palette("light"))

        assert [label.get_text() for label in labels] == [
            f"mass {n}" for n in range(1, len(points) + 1)
        ]
        assert all(label in ax.texts and label.get_visible() for label in labels)
        if case == "other_label":
            assert tuple(labels[0].xyann) == OFFSETS[0]
        assert tuple(labels[-1].xyann) == expected
        if case == "all_blocked":
            assert expected == (9, 7)
        else:
            assert sum(ranks[OFFSETS.index(expected)][0]) == 0
    finally:
        plt.close(fig)
