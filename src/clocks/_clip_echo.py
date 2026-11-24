"""The echolocation clip: a turning close view, the whole prior box, the fit.

``build_echo`` draws ``demo_echolocation_3d``: a 3D close view of the cube
``view_bounds`` holding the head (the clocks and their bounding box), the
truth with a line dropped to the floor and the particles that lie inside the
cube; beside it the whole prior box seen from the side, with the cube marked;
under that each clock's fitted reading against the running mean of its
readings; and one trace per parameter. The camera turns from azimuth -75 to
-35 at elevation 18 while the rows play, and stands still through the final
hold. Call it inside ``with style(theme) as colors:``.
"""

import math
from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from mpl_toolkits.mplot3d.art3d import Line3D, Line3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy.typing import NDArray

from clocks._clip_parts import (
    DOT_ALPHA,
    SMALL,
    Clip,
    Rect,
    draw_dots,
    draw_status,
    draw_traces,
    new_figure,
)
from clocks._replay import Replay
from clocks._style import Palette
from clocks._video import dwell_frames

# Figure fractions, from the probe (Task 1), except the fit panel's height:
# 7.5 * 0.17 / 5.5 makes it square in inches, where the probe's 0.232 was not.
LAYOUT_ECHO: dict[str, Rect] = {
    "overview": (0.06, 0.57, 0.19, 0.259),
    "fit": (0.08, 0.11, 0.17, 7.5 * 0.17 / 5.5),
    "scene": (0.255, 0.03, 0.45, 0.90),
    "trace-x": (0.795, 0.75, 0.19, 0.15),
    "trace-y": (0.795, 0.53, 0.19, 0.15),
    "trace-z": (0.795, 0.31, 0.19, 0.15),
    "trace-M": (0.795, 0.09, 0.19, 0.15),
}
HEIGHT_ECHO = 550
ELEVATION = 18.0
FIRST_AZIMUTH = -75.0
TURN = 40.0  # degrees the camera turns before the final hold
SCENE_DOT_SIZE = 1.2
OVERVIEW_DOT_SIZE = 1.2
FIT_PAD = 0.06  # of the fit panel's span, on each side


def azimuth(frame: int, frames: int) -> float:
    """The camera's azimuth at video ``frame`` of the ``frames`` before the hold.

    It eases from -75 to -35 degrees: -75 + 40 (1 - cos(pi f / (F - 1))) / 2.
    """
    eased = (1.0 - math.cos(math.pi * frame / (frames - 1))) / 2.0
    return FIRST_AZIMUTH + TURN * eased


def in_view(points: NDArray, bounds: Sequence[float]) -> NDArray[np.bool_]:
    """True where x, y and z all lie in the closed interval ``bounds``.

    The faces count as inside, as ``weight_in_view`` counts them.
    """
    low, high = bounds
    xyz = np.asarray(points, dtype=np.float64)[:, :3]
    return np.all((xyz >= low) & (xyz <= high), axis=1)


def view_weight_text(row: int, weight: float) -> str:
    """The share of the prior (row 0), or of the cloud's weight, in the view."""
    if row == 0:
        return f"{100 * weight:.1f}% of the prior lies inside the close view"
    return f"{100 * weight:.1f}% of the cloud's weight lies inside the close view"


def build_echo(replay: Replay, colors: Palette) -> Clip:
    """The ``demo_echolocation_3d`` clip, its camera set by each shot."""
    arrays = replay.arrays
    bounds = arrays["view_bounds"]
    last = len(arrays["mean"]) - 1
    turning = sum(dwell_frames(int(replay.meta["n_observations"]))[:-1])
    resting = azimuth(turning - 1, turning)
    fig = new_figure(HEIGHT_ECHO)
    scene, dots = _draw_scene(fig, replay, colors, LAYOUT_ECHO["scene"])
    overview_dots = _draw_overview(fig, replay, colors, LAYOUT_ECHO["overview"])
    fit = _draw_fit(fig, replay, colors, LAYOUT_ECHO["fit"])
    status = draw_status(fig, colors)
    weight = _draw_weight(fig, colors)
    traces = draw_traces(
        fig,
        replay,
        colors,
        {key: rect for key, rect in LAYOUT_ECHO.items() if key.startswith("trace-")},
    )

    def update(row: int, azim: float | None) -> None:
        cloud = arrays["cloud"][row].astype(np.float64)
        inside = cloud[in_view(cloud, bounds)]
        dots.set_data_3d(inside[:, 0], inside[:, 1], inside[:, 2])
        scene.view_init(elev=ELEVATION, azim=resting if azim is None else azim)
        overview_dots.set_data(cloud[:, 0], cloud[:, 2])
        fit(row)
        status(row)
        weight.set_text(view_weight_text(row, float(arrays["weight_in_view"][row])))
        traces(row)

    update(last, None)
    return Clip(fig, update)


def _draw_weight(fig: Figure, colors: Palette) -> Text:
    """The figure text under the status line that the row setter fills."""
    text = fig.text(
        0.012, 0.945, "", ha="left", va="top", fontsize=SMALL + 1, color=colors.muted
    )
    text.set_gid("view-weight")
    return text


def _draw_scene(
    fig: Figure, replay: Replay, colors: Palette, rect: Rect
) -> tuple[Axes3D, Line3D]:
    """The close view: the fixed cube, the head, the truth and an empty cloud."""
    arrays = replay.arrays
    low, high = (float(b) for b in arrays["view_bounds"])
    scene = fig.add_axes(rect, projection="3d")
    scene.set_gid("scene")
    _style_cube(scene, low, high, colors)
    _draw_head(scene, arrays["clock_positions"], colors)
    (dots,) = scene.plot(
        [],
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=SCENE_DOT_SIZE,
        markeredgewidth=0,
        alpha=DOT_ALPHA,
        color=colors.belief,
    )
    dots.set_gid("dots")
    x, y, z = (float(v) for v in arrays["true_params"][:3])
    (truth,) = scene.plot(
        [x],
        [y],
        [z],
        linestyle="none",
        marker="o",
        markersize=11,
        markerfacecolor="none",
        markeredgecolor=colors.truth,
        markeredgewidth=1.4,
    )
    truth.set_gid("truth")
    (drop,) = scene.plot([x, x], [y, y], [z, low], color=colors.truth, linewidth=0.9)
    drop.set_gid("truth-drop")
    return scene, dots


def _style_cube(scene: Axes3D, low: float, high: float, colors: Palette) -> None:
    """Limits ``(low, high)`` on all three axes, drawn as a cube, in the palette.

    Ticks at the even integers inside the view; panes in ``background`` with
    ``rule`` edges and axis lines.
    """
    for setter in (scene.set_xlim, scene.set_ylim, scene.set_zlim):
        setter(low, high)
    scene.set_box_aspect((1, 1, 1), zoom=1.1)
    for axis in (scene.xaxis, scene.yaxis, scene.zaxis):
        axis.set_pane_color(colors.background)
        axis.pane.set_edgecolor(colors.rule)
        axis.line.set_color(colors.rule)
    scene.tick_params(labelsize=SMALL - 1, pad=0)
    ticks = [t for t in range(math.ceil(low), math.floor(high) + 1) if t % 2 == 0]
    for name, set_ticks, set_label in (
        ("x", scene.set_xticks, scene.set_xlabel),
        ("y", scene.set_yticks, scene.set_ylabel),
        ("z", scene.set_zticks, scene.set_zlabel),
    ):
        set_ticks(ticks)
        set_label(name, fontsize=SMALL, labelpad=0)


def _draw_head(scene: Axes3D, positions: NDArray, colors: Palette) -> None:
    """The clocks as squares, and the twelve edges of their bounding box."""
    corners = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)])
    box = np.where(corners == 1, positions.max(axis=0), positions.min(axis=0))
    edges = [
        (box[p], box[q])
        for p in range(8)
        for q in range(p + 1, 8)
        if np.sum(corners[p] != corners[q]) == 1
    ]
    head = Line3DCollection(edges, colors=colors.muted, linewidths=0.9)
    head.set_gid("head")
    scene.add_collection3d(head)
    (clocks,) = scene.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        linestyle="none",
        marker="s",
        markersize=3.5,
        color=colors.ink,
    )
    clocks.set_gid("clocks")


def _draw_overview(fig: Figure, replay: Replay, colors: Palette, rect: Rect) -> Line2D:
    """The whole prior box in (x, z), the close view's square marked on it."""
    priors = replay.meta["prior_ranges"]
    low, high = (float(b) for b in replay.arrays["view_bounds"])
    ax = fig.add_axes(rect)
    ax.set_gid("overview")
    ax.set_xlim(*priors[0])
    ax.set_ylim(*priors[2])
    ax.set_aspect("equal")
    ax.set_xticks([priors[0][0], 0, priors[0][1]])
    ax.set_yticks([priors[2][0], 0, priors[2][1]])
    ax.tick_params(labelsize=SMALL - 1)
    ax.set_xlabel("x", fontsize=SMALL, labelpad=3)
    ax.set_ylabel("z", fontsize=SMALL, labelpad=3)
    ax.set_title("whole prior box", fontsize=SMALL + 0.5)
    dots = draw_dots(ax, colors.belief, gid="overview-dots")
    dots.set_markersize(OVERVIEW_DOT_SIZE)
    (square,) = ax.plot(
        [low, high, high, low, low],
        [low, low, high, high, low],
        color=colors.muted,
        linewidth=1.0,
    )
    square.set_gid("view-box")
    return dots


def _draw_fit(
    fig: Figure, replay: Replay, colors: Palette, rect: Rect
) -> Callable[[int], None]:
    """Each clock's fitted reading against the running mean of its readings.

    Both axes share fixed limits holding every row's values, padded by
    ``FIT_PAD``, on a square panel, so the diagonal is where the two agree.
    """
    observations = replay.arrays["observations"]
    predicted = replay.arrays["predicted_mean"]
    means = (
        np.cumsum(observations, axis=0) / np.arange(1, len(observations) + 1)[:, None]
    )
    values = np.concatenate([means.ravel(), predicted[1:].ravel()])
    low, high = float(values.min()), float(values.max())
    pad = FIT_PAD * (high - low)
    limits = (low - pad, high + pad)
    ax = fig.add_axes(rect)
    ax.set_gid("fit")
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=SMALL - 1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.set_xlabel("observed mean", fontsize=SMALL, labelpad=3)
    ax.set_ylabel("fitted", fontsize=SMALL, labelpad=3)
    (diagonal,) = ax.plot(limits, limits, color=colors.muted, linewidth=0.9)
    diagonal.set_gid("fit-diagonal")
    (points,) = ax.plot(
        [], [], linestyle="none", marker="o", markersize=3, color=colors.ink
    )
    points.set_gid("fit-points")
    empty = ax.text(
        0.04,
        0.96,
        "no readings yet",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=SMALL,
        color=colors.muted,
    )
    empty.set_gid("fit-empty")

    def update(row: int) -> None:
        if row:
            points.set_data(observations[:row].mean(axis=0), predicted[row])
        points.set_visible(row > 0)
        empty.set_visible(row == 0)

    return update
