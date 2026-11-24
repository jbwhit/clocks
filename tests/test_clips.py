"""The clips drawn from replays: their parts, every clip's frames, render_clip."""

import functools
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
    two_candidates,
)
from _figure_artists import (
    _point_count,
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

import clocks._clip_echo
import clocks._clip_parts
import clocks._clips
import clocks.viz
from clocks._clip_echo import azimuth, in_view, view_weight_text
from clocks._clip_maps import clock_fill
from clocks._clip_models import GUTTER, LABEL_GAP, build_models
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
from clocks._replay import DEMOS, Replay
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
    """The archive's clouds at ``row`` for ``DOTS[name]``; none for a cloudless clip."""
    return [
        (
            replay.arrays["cloud"][row][:, cx].astype(np.float64),
            replay.arrays["cloud"][row][:, cy].astype(np.float64),
        )
        for _, _, (cx, cy), _ in DOTS.get(name, [])
    ]


def nothing_to_draw(artist, kind: str) -> bool:
    if kind == "band":
        return len(artist.get_paths()) == 0
    if kind in ("line", "marks"):
        # Not get_xdata(): a Line3D keeps there what it held at the last draw (Q18).
        points = _point_count(artist)
        return points <= (1 if kind == "line" else 0)
    return False


def shows(when: str, row: int, replay: Replay) -> bool:
    if when == "when any is in view":
        low, high = replay.arrays["view_bounds"]
        points = replay.arrays["cloud"][row][:, :3].astype(np.float64)
        return bool(np.any(np.all((points >= low) & (points <= high), axis=1)))
    return {"always": True, "after the prior": row > 0, "only the prior": row == 0}[
        when
    ]


def show_row(clip: Clip, replay: Replay, row: int) -> None:
    """Show ``row`` at the azimuth of its first shot (None without a camera)."""
    clip.show(row, next(shot.azimuth for shot in shots(replay) if shot.row == row))


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
    replay = small(name)
    with built(replay) as (clip, _):
        show_row(clip, replay, 0)
        clip.fig.canvas.draw()
        first = _placement(clip.fig)
        assert first
        for row in (1, 4, 9):
            show_row(clip, replay, row)
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
        replay = small(name)
        with built(replay) as (clip, _):
            noted = {id(ax): ax.get_position().bounds for ax in all_axes(clip.fig)}
            clip.fig.canvas.draw()
            for row in (0, 1, 2):
                show_row(clip, replay, row)
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
    replay = small(name)
    with built(replay, theme) as (clip, _):
        for row in (0, 1, 9):
            show_row(clip, replay, row)
            clip.fig.canvas.draw()
            for axes, gid, kind, when in ARTISTS[name]:
                artist = fetch(clip.fig, axes, gid)
                if shows(when, row, replay):
                    assert_shown(artist, kind)
                elif not nothing_to_draw(artist, kind):
                    assert_hidden(artist)


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("name", CLIPS)
def test_status_line(name: str, theme: str) -> None:
    expected = {0: "Before any observation", 1: "After 1 observation"}
    expected[9] = "After 9 observations"
    replay = small(name)
    with built(replay, theme) as (clip, colors):
        for row, text in expected.items():
            show_row(clip, replay, row)
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
    replay = small(name)
    with built(replay, theme) as (clip, colors):
        for row in (0, 9):
            show_row(clip, replay, row)
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
            show_row(clip, replay, row)
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
            show_row(clip, replay, row)
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
            show_row(clip, replay, last)
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
                show_row(clip, replay, row)
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
                show_row(clip, replay, row)
                assert len(frame_rgba(clip.fig)) > 0


@pytest.mark.parametrize(
    "name", [name for name in CLIPS if name != "demo_echolocation_3d"]
)
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
# The clips whose two masses are labelled by label_masses.
LABELLED = ["demo_multi_mass", "demo_multi_mass_2d"]
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
    _assert_masses_told_apart("demo_multi_mass", "cloud", theme)


def _assert_masses_told_apart(name: str, axes: str, theme: str) -> None:
    """Two markers and two colours; each truth's label is nearer it than the other."""
    with built(small(name), theme) as (clip, colors):
        ax = axes_named(clip.fig, axes)
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
    first offset's, so that offset alone covers none; the second offset's box
    holds a second mark, so that a rule which ranks by order alone, ignoring
    the marks, takes a different offset from the rule.
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
    second = _candidate_box(ax, text, points[-1], OFFSETS[1])
    centres.append(((3 * second.x0 + second.x1) / 4, (second.y0 + second.y1) / 2))
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
            # The second offset is free but covers two marks: ignoring the marks
            # would take it, where the rule takes a later offset covering one.
            assert sum(ranks[1][0]) == 0 and ranks[1][1] == 2
        expected = _choose(ranks)
        if case != "all_blocked":
            assert expected != OFFSETS[1]

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


# ---------------------------------------------------------------- the two-mass map

MULTI_2D = "demo_multi_mass_2d"


@pytest.mark.parametrize("name", TRACE_CLIPS)
def test_trace_headers_are_their_clips_size(name: str) -> None:
    # At 9.5 points the two-mass map's right column of headers leaves the canvas.
    size = 8.5 if name == MULTI_2D else 9.5
    with built(small(name)) as (clip, _):
        for row in (0, 9):
            clip.show(row)
            for param in _trace_names(name):
                title = axes_named(clip.fig, f"trace-{param}").title
                assert title.get_fontsize() == size, (row, param)


def test_two_mass_map_has_one_trace_column_per_mass() -> None:
    with built(small(MULTI_2D)) as (clip, _):
        boxes = {
            name: axes_named(clip.fig, f"trace-{name}").get_position(original=True)
            for name in ("x₁", "y₁", "M₁", "x₂", "y₂", "M₂")
        }
    left, right = boxes["x₁"].x0, boxes["x₂"].x0
    assert right > left
    for name in ("y₁", "M₁"):
        assert boxes[name].x0 == pytest.approx(left, rel=1e-12)
    for name in ("y₂", "M₂"):
        assert boxes[name].x0 == pytest.approx(right, rel=1e-12)
    for first, second in (("x₁", "x₂"), ("y₁", "y₂"), ("M₁", "M₂")):
        assert boxes[first].y0 == pytest.approx(boxes[second].y0, rel=1e-12)
    assert boxes["x₁"].y0 > boxes["y₁"].y0 > boxes["M₁"].y0


@pytest.mark.parametrize("theme", THEMES)
def test_two_mass_map_has_no_inset_and_plain_clocks(theme: str) -> None:
    replay = small(MULTI_2D)
    positions = replay.arrays["clock_positions"]
    with built(replay, theme) as (clip, colors):
        ax = axes_named(clip.fig, "map")
        assert ax.child_axes == []
        for row in (0, 9):
            clip.show(row)
            mark = one(ax, "clocks")
            np.testing.assert_array_equal(mark.get_xdata(), positions[:, 0])
            np.testing.assert_array_equal(mark.get_ydata(), positions[:, 1])
            assert mark.get_marker() == "s"
            assert mark.get_linestyle() == "None"
            assert to_hex(mark.get_color()) == colors.ink
            assert to_hex(mark.get_markerfacecolor()) == colors.ink
            assert to_hex(mark.get_markeredgecolor()) == colors.ink


@pytest.mark.parametrize("theme", THEMES)
def test_two_mass_map_labels_each_truth(theme: str) -> None:
    _assert_masses_told_apart(MULTI_2D, "map", theme)


# ---------------------------------------------------------------- the model clip

MODELS = "demo_model_comparison"


def _expected_label_shifts(ax, row: int, heads) -> list[float]:
    """Decision D50's vertical shifts in points, by the test's own arithmetic.

    Heights from the top, ties in ``model_ks`` order; each label at its head or
    ``LABEL_GAP`` under the one above; then, from the bottom up, the lowest at
    least ``LABEL_GAP / 2`` above the panel's bottom edge and each other at
    least ``LABEL_GAP`` above the one under it.
    """
    scale = 72 / ax.figure.dpi
    ys = [ax.transData.transform((row, head))[1] * scale for head in heads]
    floor = ax.get_window_extent().y0 * scale + LABEL_GAP / 2
    order = sorted(range(len(ys)), key=lambda i: (-ys[i], i))
    placed: dict[int, float] = {}
    previous = None
    for i in order:
        placed[i] = ys[i] if previous is None else min(ys[i], previous - LABEL_GAP)
        previous = placed[i]
    below = None
    for i in reversed(order):
        placed[i] = max(placed[i], floor if below is None else below + LABEL_GAP)
        below = placed[i]
    return [placed[i] - ys[i] for i in range(len(ys))]


@pytest.mark.parametrize("theme", THEMES)
def test_model_lines_grow_from_the_prior(theme: str) -> None:
    replay = small(MODELS)
    posterior = replay.arrays["model_posterior"]
    with built(replay, theme) as (clip, colors):
        ax = axes_named(clip.fig, "posterior")
        for row in (0, 1, 4, 9):
            clip.show(row)
            for i, k in enumerate(replay.arrays["model_ks"]):
                line = one(ax, f"model-{k}")
                np.testing.assert_array_equal(line.get_xdata(), np.arange(row + 1))
                np.testing.assert_array_equal(line.get_ydata(), posterior[: row + 1, i])
                if row == 0:
                    assert line.get_ydata()[0] == replay.arrays["model_prior"][i]
                assert to_hex(line.get_color()) == to_hex(colors.ordered[i])


@pytest.mark.parametrize("theme", THEMES)
def test_model_labels_name_the_truth(theme: str) -> None:
    replay = small(MODELS)
    third = Replay(dict(replay.meta, true_k=3), dict(replay.arrays))
    cases = [
        (replay, ["K = 1", "K = 2 (true model)", "K = 3"]),
        (third, ["K = 1", "K = 2", "K = 3 (true model)"]),
    ]
    for case, texts in cases:
        with style(theme) as colors:
            clip = build_models(case, colors)
            try:
                ax = axes_named(clip.fig, "posterior")
                labels = [one(ax, f"label-{k}") for k in (1, 2, 3)]
                assert [label.get_text() for label in labels] == texts
                for k, label in zip((1, 2, 3), labels, strict=True):
                    line = one(ax, f"model-{k}")
                    assert to_hex(label.get_color()) == to_hex(line.get_color())
            finally:
                plt.close(clip.fig)


@pytest.mark.parametrize("theme", THEMES)
def test_model_labels_follow_the_heads(theme: str) -> None:
    replay = small(MODELS)
    posterior = replay.arrays["model_posterior"]
    # Row 0: three heads tie at one third and stack 12 points apart; rows 4 and
    # 9: the heads are far enough apart that no label moves (Q20).
    literal = {0: [0.0, -12.0, -24.0], 4: [0.0, 0.0, 0.0], 9: [0.0, 0.0, 0.0]}
    with built(replay, theme) as (clip, _):
        ax = axes_named(clip.fig, "posterior")
        for row, shifts in literal.items():
            clip.show(row)
            expected = _expected_label_shifts(ax, row, posterior[row])
            assert expected == pytest.approx(shifts, abs=1e-6)
            for i, k in enumerate(replay.arrays["model_ks"]):
                label = one(ax, f"label-{k}")
                assert tuple(label.xy) == (row, posterior[row, i])
                assert label.anncoords == "offset points"
                assert label.xyann[0] == 6
                assert label.xyann[1] == pytest.approx(expected[i], abs=1e-6)
                assert label.get_horizontalalignment() == "left"
                assert label.get_verticalalignment() == "center"
                assert label.get_fontsize() == 10
        connectors = [
            a for a in ax.findobj() if str(a.get_gid() or "").startswith("connector-")
        ]
        assert connectors == []


@pytest.mark.parametrize(
    "last_row",
    [(0.30, 0.31, 0.39), (0.0, 0.5, 0.5), (0.0, 0.0, 1.0)],
    ids=["close", "tie", "floor"],
)
def test_model_labels_keep_apart_in_head_order(last_row: tuple) -> None:
    base = small(MODELS)
    posterior = base.arrays["model_posterior"].copy()
    posterior[-1] = last_row
    replay = edited(base, model_posterior=posterior)
    row = len(posterior) - 1
    with built(replay) as (clip, _):
        clip.show(row)
        ax = axes_named(clip.fig, "posterior")
        scale = 72 / clip.fig.dpi
        shifts = [one(ax, f"label-{k}").xyann[1] for k in (1, 2, 3)]
        heads = [ax.transData.transform((row, p))[1] * scale for p in last_row]
        frame = ax.get_window_extent()
        bottom, top = frame.y0 * scale, frame.y1 * scale
    centres = [head + shift for head, shift in zip(heads, shifts, strict=True)]
    order = sorted(range(3), key=lambda i: (-last_row[i], i))
    placed = [centres[i] for i in order]
    for upper, lower in zip(placed, placed[1:]):
        assert upper - lower >= LABEL_GAP - 1e-6
    assert placed[-1] >= bottom + LABEL_GAP / 2 - 1e-6
    assert placed[0] <= top - LABEL_GAP / 2 + 1e-6
    if last_row == (0.0, 0.0, 1.0):
        assert shifts[2] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("theme", THEMES)
def test_model_colours_go_by_position_in_model_ks(theme: str) -> None:
    replay = two_candidates()
    with built(replay, theme) as (clip, colors):
        ax = axes_named(clip.fig, "posterior")
        assert to_hex(one(ax, "model-2").get_color()) == to_hex(colors.ordered[0])
        assert to_hex(one(ax, "model-3").get_color()) == to_hex(colors.ordered[1])
        for row in range(len(replay.arrays["model_posterior"])):
            clip.show(row)
            clip.fig.canvas.draw()


def test_model_clip_refuses_a_fourth_candidate() -> None:
    base = small(MODELS)
    arrays = dict(base.arrays)
    arrays["model_ks"] = np.array([1, 2, 3, 4], dtype=np.int64)
    for key in ("model_prior", "model_posterior", "model_log_evidence", "model_ess"):
        column = base.arrays[key]
        arrays[key] = np.concatenate([column, column[..., -1:]], axis=-1)
    four = Replay(dict(base.meta), arrays)
    before = plt.get_fignums()
    with style("light") as colors:
        with pytest.raises(ValueError, match="at most three candidates"):
            build_models(four, colors)
    assert plt.get_fignums() == before
    assert "at most three candidates" in " ".join(render_clip.__doc__.split())


def test_changing_a_posterior_row_moves_only_that_point() -> None:
    original = committed(MODELS)
    posterior = original.arrays["model_posterior"].copy()
    posterior[40, [1, 2]] = posterior[40, [2, 1]]
    changed = edited(original, model_posterior=posterior)
    last = len(posterior) - 1
    drawn = []
    for replay in (original, changed):
        with built(replay) as (clip, _):
            clip.show(last)
            ax = axes_named(clip.fig, "posterior")
            drawn.append(
                {k: np.array(one(ax, f"model-{k}").get_ydata()) for k in (2, 3)}
            )
    before, after = drawn
    for k in (2, 3):
        np.testing.assert_array_equal(np.flatnonzero(after[k] != before[k]), [40])


def test_model_axes() -> None:
    with built(small(MODELS)) as (clip, _):
        ax = axes_named(clip.fig, "posterior")
        assert ax.get_xlim() == pytest.approx((0, 9 + GUTTER), rel=1e-12)
        assert ax.get_ylim() == (-0.03, 1.03)
        assert ax.get_xlabel() == "observations"
        assert ax.get_ylabel() == "posterior probability"


@functools.cache
def _long_model_run() -> Replay:
    """250 observations of sixteen particles per candidate."""
    module = importlib.import_module(f"clocks._demos.{MODELS}")
    return module.record(n_observations=250, n_particles=16)


@pytest.mark.parametrize("theme", THEMES)
def test_model_labels_fit_a_long_run(theme: str) -> None:
    replay = _long_model_run()
    with built(replay, theme) as (clip, _):
        clip.show(250)
        clip.fig.canvas.draw()
        renderer = clip.fig.canvas.get_renderer()
        ax = axes_named(clip.fig, "posterior")
        frame = ax.get_window_extent(renderer)
        for k in replay.arrays["model_ks"]:
            box = one(ax, f"label-{k}").get_window_extent(renderer)
            assert frame.x0 <= box.x0 and box.x1 <= frame.x1, f"K = {k}"
        assert ax.get_xlim()[1] > 250 + GUTTER


# ---------------------------------------------------------------- the echolocation clip

ECHO = "demo_echolocation_3d"


def _inside(points: np.ndarray, bounds) -> np.ndarray:
    """Rows of ``points`` whose x, y and z all lie in the closed interval."""
    low, high = bounds
    xyz = np.asarray(points, dtype=np.float64)[:, :3]
    return xyz[np.all((xyz >= low) & (xyz <= high), axis=1)]


def _scene_dots(fig: Figure) -> np.ndarray:
    """The close view's dots as an (n, 3) array."""
    x, y, z = one(axes_named(fig, "scene"), "dots").get_data_3d()
    return np.column_stack([x, y, z]).astype(np.float64)


def test_every_clip_demo_has_a_builder() -> None:
    assert set(BUILDERS) == set(CLIP_HEIGHTS) == set(DEMOS) - {"demo_density"}


def test_azimuth() -> None:
    for frame, expected in [
        (0, -75.0),
        (76, -69.17850576840875),
        (152, -55.103002582480016),
        (229, -40.82149423159126),
        (305, -35.0),
    ]:
        assert azimuth(frame, 306) == pytest.approx(expected, abs=1e-9), frame
    for frame, expected in [
        (0, -75.0),
        (6, -68.65106286437309),
        (12, -53.63515173270658),
        (23, -35.0),
    ]:
        assert azimuth(frame, 24) == pytest.approx(expected, abs=1e-9), frame


def test_echo_shots_turn_the_camera_and_hold() -> None:
    plan = shots(committed(ECHO))
    assert len(plan) == 307
    turning, hold = plan[:306], plan[306]
    assert all(shot.repeat == 1 for shot in turning)
    np.testing.assert_array_equal(
        [shot.row for shot in turning], np.repeat(np.arange(80), dwell_frames(80)[:-1])
    )
    for f, shot in enumerate(turning):
        assert shot.azimuth == pytest.approx(azimuth(f, 306), abs=1e-9), f
    assert (hold.row, hold.repeat) == (80, 50)
    assert hold.azimuth == pytest.approx(-35.0, abs=1e-9)
    assert sum(shot.repeat for shot in plan) == 356

    short = shots(shortest(ECHO))
    assert len(short) == 25
    assert (short[-1].row, short[-1].repeat) == (1, 50)
    assert short[-1].azimuth == pytest.approx(-35.0, abs=1e-9)


def test_echo_camera_follows_the_shot() -> None:
    replay = committed(ECHO)
    plan = shots(replay)
    with built(replay) as (clip, _):
        scene = axes_named(clip.fig, "scene")
        for index in (0, 76, 306):
            shot = plan[index]
            clip.show(shot.row, shot.azimuth)
            assert scene.azim == pytest.approx(shot.azimuth, abs=1e-9), index
            assert scene.elev == 18
        clip.show(0, plan[0].azimuth)
        clip.show(80)
        assert scene.azim == pytest.approx(-35.0, abs=1e-9)
        assert scene.elev == 18


def test_echo_scene_is_the_fixed_cube() -> None:
    replay = committed(ECHO)
    bounds = tuple(float(b) for b in replay.arrays["view_bounds"])
    with built(replay) as (clip, _):
        scene = axes_named(clip.fig, "scene")
        for row in (0, 1, 80):
            show_row(clip, replay, row)
            assert scene.get_xlim() == bounds
            assert scene.get_ylim() == bounds
            assert scene.get_zlim() == bounds
        aspect = scene.get_box_aspect()
        assert aspect[0] == aspect[1] == aspect[2]


def test_echo_close_view_holds_exactly_the_dots_inside_the_cube() -> None:
    replay = committed(ECHO)
    bounds = replay.arrays["view_bounds"]
    cloud = replay.arrays["cloud"]
    prior_inside = _inside(cloud[0], bounds)
    assert len(prior_inside) == 2 and len(cloud[0]) == 500
    with built(replay) as (clip, _):
        show_row(clip, replay, 0)
        np.testing.assert_array_equal(_scene_dots(clip.fig), prior_inside)
        show_row(clip, replay, 1)
        np.testing.assert_array_equal(
            _scene_dots(clip.fig), cloud[1][:, :3].astype(np.float64)
        )

    base = small(ECHO)
    changed_cloud = base.arrays["cloud"].copy()
    changed_cloud[4, :3, 0] = base.arrays["view_bounds"][1] + 1.0
    changed = edited(base, cloud=changed_cloud)
    with built(changed) as (clip, _):
        show_row(clip, changed, 4)
        expected = changed_cloud[4][3:, :3].astype(np.float64)
        assert len(expected) == 37
        np.testing.assert_array_equal(_scene_dots(clip.fig), expected)


@pytest.mark.parametrize("theme", THEMES)
def test_echo_close_view_draws_the_prior_dots_that_are_inside(theme: str) -> None:
    replay = committed(ECHO)
    with built(replay, theme) as (clip, colors):
        for row, count in ((0, 2), (1, 500), (80, 500)):
            show_row(clip, replay, row)
            clip.fig.canvas.draw()
            dots = one(axes_named(clip.fig, "scene"), "dots")
            assert_shown(dots, "marks")
            assert len(_scene_dots(clip.fig)) == count, row
            assert to_hex(dots.get_color()) == colors.belief


@pytest.mark.parametrize("theme", THEMES)
def test_echo_close_view_dots_reach_the_canvas(theme: str) -> None:
    replay = committed(ECHO)
    with built(replay, theme) as (clip, _):
        fig = clip.fig
        scene = axes_named(fig, "scene")
        dots = one(scene, "dots")
        for row in (0, 1, 80):
            show_row(clip, replay, row)
            shown = np.frombuffer(frame_rgba(fig), dtype=np.uint8).reshape(
                1100, 1500, 4
            )
            dots.set_visible(False)
            hidden = np.frombuffer(frame_rgba(fig), dtype=np.uint8).reshape(
                1100, 1500, 4
            )
            dots.set_visible(True)
            scale = 200 / fig.dpi
            box = scene.get_window_extent()
            left, right = int(np.floor(box.x0 * scale)), int(np.ceil(box.x1 * scale))
            top = int(np.floor(1100 - box.y1 * scale))
            bottom = int(np.ceil(1100 - box.y0 * scale))
            differs = np.any(shown != hidden, axis=2)
            inside = np.zeros_like(differs)
            inside[max(top, 0) : bottom, max(left, 0) : right] = True
            assert int(differs[inside].sum()) >= 10, row
            assert int(differs[~inside].sum()) == 0, row


def _nothing_inside_prior() -> Replay:
    base = small(ECHO)
    cloud = base.arrays["cloud"].copy()
    cloud[0, :, 0] = base.arrays["view_bounds"][1] + 1.0
    return edited(base, cloud=cloud)


def _nothing_inside_after_a_full_row() -> Replay:
    base = small(ECHO)
    low, high = base.arrays["view_bounds"]
    cloud = base.arrays["cloud"].copy()
    cloud[3, :, :3] = (low + high) / 2
    cloud[4, :, 0] = high + 1.0
    return edited(base, cloud=cloud)


@pytest.mark.parametrize(
    ("make", "before", "row"),
    [
        pytest.param(_nothing_inside_prior, None, 0, id="prior"),
        pytest.param(_nothing_inside_after_a_full_row, 3, 4, id="after_a_full_row"),
    ],
)
def test_echo_close_view_with_nothing_inside_is_empty(make, before, row) -> None:
    replay = make()
    with built(replay) as (clip, _):
        if before is not None:
            show_row(clip, replay, before)
            assert len(_scene_dots(clip.fig)) == 40
        show_row(clip, replay, row)
        x, y, z = one(axes_named(clip.fig, "scene"), "dots").get_data_3d()
        assert (len(x), len(y), len(z)) == (0, 0, 0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert len(frame_rgba(clip.fig)) > 0


def test_in_view_counts_the_faces() -> None:
    points = np.array(
        [
            [-1.0, 0.0, 2.0],
            [2.0, 2.0, 2.0],
            [2.0000001, 0.0, 0.0],
            [0.0, -1.0000001, 0.0],
        ]
    )
    assert in_view(points, (-1.0, 2.0)).tolist() == [True, True, False, False]


def test_echo_overview_spans_the_prior_and_marks_the_view() -> None:
    replay = committed(ECHO)
    with built(replay) as (clip, _):
        ax = axes_named(clip.fig, "overview")
        assert ax.get_xlim() == (-16.0, 16.0)
        assert ax.get_ylim() == (-16.0, 16.0)
        x, y = one(ax, "view-box").get_data()
        corners = [(-1.3, -1.3), (4.8, -1.3), (4.8, 4.8), (-1.3, 4.8)]
        visited = list(zip(np.asarray(x, float), np.asarray(y, float), strict=True))
        assert len(visited) == 5
        assert visited[0] == visited[-1]
        assert set(visited[:4]) == set(corners)


def test_view_weight_text() -> None:
    replay = committed(ECHO)
    with built(replay) as (clip, _):
        for row, text in (
            (0, "0.4% of the prior lies inside the close view"),
            (1, "100.0% of the cloud's weight lies inside the close view"),
            (80, "100.0% of the cloud's weight lies inside the close view"),
        ):
            show_row(clip, replay, row)
            assert fetch(clip.fig, None, "view-weight").get_text() == text, row
    assert (
        view_weight_text(3, 0.5)
        == "50.0% of the cloud's weight lies inside the close view"
    )


def test_the_weight_text_follows_the_archive() -> None:
    base = small(ECHO)
    weight = base.arrays["weight_in_view"].copy()
    weight[3] = 0.5
    replay = edited(base, weight_in_view=weight)
    with built(replay) as (clip, _):
        for row in (2, 3, 4):
            show_row(clip, replay, row)
            text = fetch(clip.fig, None, "view-weight").get_text()
            share = "50.0" if row == 3 else f"{100 * weight[row]:.1f}"
            expected = f"{share}% of the cloud's weight lies inside the close view"
            assert text == expected, row


@pytest.mark.parametrize("theme", THEMES)
def test_echo_fit_panel(theme: str) -> None:
    replay = small(ECHO)
    observations = replay.arrays["observations"]
    predicted = replay.arrays["predicted_mean"]
    with built(replay, theme) as (clip, colors):
        ax = axes_named(clip.fig, "fit")
        empty = one(ax, "fit-empty")
        points = one(ax, "fit-points")
        show_row(clip, replay, 0)
        assert empty.get_visible() and empty.get_text() == "no readings yet"
        assert not points.get_visible()
        for row in (1, 4, 9):
            show_row(clip, replay, row)
            assert not empty.get_visible()
            assert points.get_visible()
            np.testing.assert_allclose(
                points.get_xdata(),
                observations[:row].mean(axis=0),
                rtol=1e-12,
                atol=1e-15,
            )
            np.testing.assert_allclose(
                points.get_ydata(), predicted[row], rtol=1e-12, atol=1e-15
            )
            assert to_hex(points.get_color()) == colors.ink
        assert ax.get_xlim() == ax.get_ylim()
        assert ax.get_aspect() == 1.0
        box = ax.get_position(original=True)
        assert 7.5 * box.width == pytest.approx(5.5 * box.height, rel=1e-9)
        diagonal = one(ax, "fit-diagonal")
        low, high = ax.get_xlim()
        np.testing.assert_array_equal(diagonal.get_xdata(), [low, high])
        np.testing.assert_array_equal(diagonal.get_ydata(), [low, high])
        assert to_hex(diagonal.get_color()) == colors.muted
        assert ax.get_xlabel() == "observed mean"
        assert ax.get_ylabel() == "fitted"


@pytest.mark.parametrize("theme", THEMES)
def test_echo_truth_ring_and_drop_line(theme: str) -> None:
    replay = small(ECHO)
    x, y, z = (float(v) for v in replay.arrays["true_params"][:3])
    low = float(replay.arrays["view_bounds"][0])
    with built(replay, theme) as (clip, colors):
        scene = axes_named(clip.fig, "scene")
        truth = one(scene, "truth")
        np.testing.assert_array_equal(
            np.asarray(truth.get_data_3d(), dtype=float), [[x], [y], [z]]
        )
        assert truth.get_markerfacecolor() == "none"
        assert to_hex(truth.get_markeredgecolor()) == colors.truth
        assert truth.get_markeredgewidth() > 0
        assert truth.get_markersize() > one(scene, "dots").get_markersize()
        drop = one(scene, "truth-drop")
        np.testing.assert_array_equal(
            np.asarray(drop.get_data_3d(), dtype=float), [[x, x], [y, y], [z, low]]
        )
        assert to_hex(drop.get_color()) == colors.truth


@pytest.mark.parametrize("theme", THEMES)
def test_echo_head_and_clocks(theme: str, monkeypatch: pytest.MonkeyPatch) -> None:
    made: list[tuple[object, list]] = []
    real = clocks._clip_echo.Line3DCollection

    class Recording(real):
        def __init__(self, segments, *args, **kwargs) -> None:
            made.append((self, [np.array(s, dtype=float) for s in segments]))
            super().__init__(segments, *args, **kwargs)

    monkeypatch.setattr(clocks._clip_echo, "Line3DCollection", Recording)
    base = small(ECHO)
    # The recording's clocks fill the cube from -1 to 1 exactly, which a head
    # of constant size would match; stretched and moved per axis, they do not.
    replay = edited(
        base,
        clock_positions=base.arrays["clock_positions"] * [1.5, 0.75, 1.25]
        + [0.25, -0.5, 0.1],
    )
    positions = replay.arrays["clock_positions"]
    with built(replay, theme) as (clip, colors):
        scene = axes_named(clip.fig, "scene")
        clocks_artist = one(scene, "clocks")
        np.testing.assert_array_equal(
            np.asarray(clocks_artist.get_data_3d(), dtype=float), positions.T
        )
        assert to_hex(clocks_artist.get_color()) == colors.ink
        head = one(scene, "head")
        assert to_hex(head.get_edgecolor()[0], keep_alpha=False) == colors.muted
        assert len(made) == 1
        instance, segments = made[0]
        assert instance is head
    low, high = positions.min(axis=0), positions.max(axis=0)
    corners = [
        (float(a), float(b), float(c))
        for a in (low[0], high[0])
        for b in (low[1], high[1])
        for c in (low[2], high[2])
    ]
    edges = {
        frozenset((p, q))
        for i, p in enumerate(corners)
        for q in corners[i + 1 :]
        if sum(u != v for u, v in zip(p, q, strict=True)) == 1
    }
    assert len(edges) == 12
    drawn = {frozenset(tuple(float(v) for v in end) for end in s) for s in segments}
    assert len(segments) == 12
    assert drawn == edges
