"""The map clips: a cloud over the (x, y) plane, with its clocks and traces.

``build_2d`` draws ``demo_2d``: the map of the prior box with the particle
cloud, the clocks filled by how slow they have run on average, the truth, the
covariance ellipse and a fixed zoom inset on the truth, beside one trace per
parameter. Call it inside ``with style(theme) as colors:``.
"""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from numpy.typing import NDArray

from clocks._clip_parts import (
    SMALL,
    Clip,
    Rect,
    draw_dots,
    draw_ellipse,
    draw_inset,
    draw_status,
    draw_traces,
    draw_truth,
    ellipse_points,
    new_figure,
    running_slowing,
    standard_deviations,
    zoom_label,
    zoom_window,
)
from clocks._replay import Replay
from clocks._style import Palette

# Figure fractions, from the probe (Task 1); the inset's are fractions of "map".
LAYOUT_2D: dict[str, Rect] = {
    "map": (0.075, 0.12, 0.468, 0.78),
    "inset": (0.0, 0.72, 0.28, 0.28),
    "trace-x": (0.655, 0.70, 0.325, 0.19),
    "trace-y": (0.655, 0.405, 0.325, 0.19),
    "trace-M": (0.655, 0.11, 0.325, 0.19),
}
HEIGHT_2D = 450
CLOCK_SIZE = 42.0


def clock_fill(
    running: NDArray[np.float64], largest: float, colors: Palette
) -> list[tuple[float, float, float, float]]:
    """One RGBA per clock: a blend from ``background`` to ``ink`` by slowing.

    A NaN (the prior row) is fully transparent, a hollow clock. Otherwise the
    blend weight is ``value / largest`` clipped to [0, 1]; when ``largest`` is
    not positive no clock was ever slow on average and none is filled.
    """
    background = np.array(to_rgba(colors.background)[:3])
    ink = np.array(to_rgba(colors.ink)[:3])
    fills: list[tuple[float, float, float, float]] = []
    for value in np.asarray(running, dtype=np.float64):
        if np.isnan(value):
            fills.append((0.0, 0.0, 0.0, 0.0))
            continue
        weight = min(max(value / largest, 0.0), 1.0) if largest > 0 else 0.0
        red, green, blue = background + weight * (ink - background)
        fills.append((float(red), float(green), float(blue), 1.0))
    return fills


def build_2d(replay: Replay, colors: Palette) -> Clip:
    """The ``demo_2d`` clip: the map, its inset and three traces."""
    fig = new_figure(HEIGHT_2D)
    arrays = replay.arrays
    ax = _map_axes(fig.add_axes(LAYOUT_2D["map"]), replay)
    running = running_slowing(arrays["observations"])
    largest = float(np.max(running[np.isfinite(running)]))
    clocks = ax.scatter(
        arrays["clock_positions"][:, 0],
        arrays["clock_positions"][:, 1],
        marker="s",
        s=CLOCK_SIZE,
        facecolors=clock_fill(running[0], largest, colors),
        edgecolors=colors.ink,
        linewidths=1.0,
        zorder=2,
    )
    clocks.set_gid("clocks")
    cloud = _draw_cloud(ax, replay, colors)
    _label_map(ax, colors)
    status = draw_status(fig, colors)
    traces = draw_traces(
        fig,
        replay,
        colors,
        {key: rect for key, rect in LAYOUT_2D.items() if key.startswith("trace-")},
    )

    def update(row: int, azimuth: float | None) -> None:
        clocks.set_facecolors(clock_fill(running[row], largest, colors))
        cloud(row)
        status(row)
        traces(row)

    update(len(arrays["mean"]) - 1, None)
    return Clip(fig, update)


def _map_axes(ax: Axes, replay: Replay) -> Axes:
    """The map: the prior box at equal aspect, on a rectangle square in inches."""
    x_range, y_range = replay.meta["prior_ranges"][:2]
    ax.set_gid("map")
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.tick_params(pad=5)
    return ax


def _draw_cloud(ax: Axes, replay: Replay, colors: Palette):
    """The dots, truth and ellipse on the map, and their copies in the inset."""
    arrays = replay.arrays
    truth = arrays["true_params"]
    dots = draw_dots(ax, colors.belief)
    draw_truth(ax, truth[0], truth[1], colors.truth)
    ellipse = draw_ellipse(ax, colors.belief)
    inset = _draw_inset(ax, replay, colors)
    inset_dots = draw_dots(inset, colors.belief, gid="inset-dots")
    draw_truth(inset, truth[0], truth[1], colors.truth, gid="inset-truth")
    inset_ellipse = draw_ellipse(inset, colors.belief, gid="inset-ellipse")

    def update(row: int) -> None:
        points = arrays["cloud"][row][:, :2].astype(np.float64)
        contour = ellipse_points(arrays["mean"][row][:2], arrays["cov"][row][:2, :2])
        for line in (dots, inset_dots):
            line.set_data(points[:, 0], points[:, 1])
        for line in (ellipse, inset_ellipse):
            line.set_data(contour[:, 0], contour[:, 1])

    return update


def _draw_inset(ax: Axes, replay: Replay, colors: Palette) -> Axes:
    """The fixed zoom window on the truth, from the last row, and its label."""
    arrays, priors = replay.arrays, replay.meta["prior_ranges"]
    rect = LAYOUT_2D["inset"]
    window = zoom_window(
        arrays["true_params"][:2],
        arrays["mean"][-1][:2],
        standard_deviations(replay)[-1][:2],
        priors[:2],
    )
    inset = draw_inset(ax, rect, window, colors)
    x_factor = rect[2] * np.ptp(ax.get_xlim()) / np.ptp(window[0])
    y_factor = rect[3] * np.ptp(ax.get_ylim()) / np.ptp(window[1])
    inset.text(
        0.0,
        1.03,
        zoom_label(x_factor, y_factor, str(replay.meta["param_names"][1])),
        transform=inset.transAxes,
        ha="left",
        va="bottom",
        fontsize=SMALL + 0.5,
        color=colors.muted,
        clip_on=False,
    )
    return inset


def _label_map(ax: Axes, colors: Palette) -> None:
    """The two direct labels: the ellipse's key above, the fill's key below."""
    ax.text(
        1.0,
        1.015,
        "covariance, 2 s.d.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=SMALL + 0.5,
        color=colors.belief,
        clip_on=False,
    )
    ax.text(
        1.0,
        -0.085,
        "stronger fill = slower clock",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=SMALL + 0.5,
        color=colors.ink,
    )
