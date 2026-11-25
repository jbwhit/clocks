"""The map clips: clouds over the (x, y) plane, with their clocks and traces.

``build_2d`` draws ``demo_2d``: the map of the prior box with the particle
cloud, the clocks filled by how slow they have run on average, the truth, the
covariance ellipse and a fixed zoom inset on the truth, beside one trace per
parameter. ``build_multi_2d`` draws ``demo_multi_mass_2d``: both masses'
clouds, truths and ellipses on one map with the plain clocks, each truth
labelled, beside one column of traces per mass. Call them inside
``with style(theme) as colors:``.
"""

from collections.abc import Callable

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
    label_masses,
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
LAYOUT_MULTI_2D: dict[str, Rect] = {
    "map": (0.06, 0.11, 0.484, 0.66),
    "trace-x₁": (0.62, 0.68, 0.14, 0.17),
    "trace-y₁": (0.62, 0.395, 0.14, 0.17),
    "trace-M₁": (0.62, 0.11, 0.14, 0.17),
    "trace-x₂": (0.83, 0.68, 0.14, 0.17),
    "trace-y₂": (0.83, 0.395, 0.14, 0.17),
    "trace-M₂": (0.83, 0.11, 0.14, 0.17),
}
HEIGHT_MULTI_2D = 550
# At 9.5 points the right column's headers leave the canvas (Task 1's sheets).
MULTI_2D_HEADER_SIZE = 8.5
MULTI_2D_CLOCK_SIZE = 5.0
# Each mass: its (x, y) columns of the archive, palette role, marker, gid suffix.
MULTI_2D_MASSES = (((0, 1), "belief", "o", "-1"), ((2, 3), "belief_alt", "^", "-2"))
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
    _key_ellipse(ax, colors)
    _key_fill(ax, colors)
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


def build_multi_2d(replay: Replay, colors: Palette) -> Clip:
    """The ``demo_multi_mass_2d`` clip: two labelled clouds, the clocks, traces."""
    fig = new_figure(HEIGHT_MULTI_2D)
    arrays = replay.arrays
    ax = _map_axes(fig.add_axes(LAYOUT_MULTI_2D["map"]), replay)
    positions = arrays["clock_positions"]
    (clocks,) = ax.plot(
        positions[:, 0],
        positions[:, 1],
        linestyle="none",
        marker="s",
        markersize=MULTI_2D_CLOCK_SIZE,
        color=colors.ink,
        zorder=2,
    )
    clocks.set_gid("clocks")
    clouds = [_draw_mass(ax, replay, colors, *mass) for mass in MULTI_2D_MASSES]
    _key_ellipse(ax, colors)
    status = draw_status(fig, colors)
    traces = draw_traces(
        fig,
        replay,
        colors,
        {key: rect for key, rect in LAYOUT_MULTI_2D.items() if key != "map"},
        header_size=MULTI_2D_HEADER_SIZE,
    )

    def update(row: int, azimuth: float | None) -> None:
        for cloud in clouds:
            cloud(row)
        status(row)
        traces(row)

    update(len(arrays["mean"]) - 1, None)
    truth = arrays["true_params"]
    label_masses(
        ax,
        [(float(truth[i]), float(truth[j])) for (i, j), *_ in MULTI_2D_MASSES],
        colors,
    )
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


def _draw_mass(
    ax: Axes,
    replay: Replay,
    colors: Palette,
    columns: tuple[int, int],
    role: str,
    marker: str,
    suffix: str,
) -> Callable[[int], None]:
    """One mass's dots, truth and ellipse on the map, with no inset."""
    arrays = replay.arrays
    pair = list(columns)
    colour = getattr(colors, role)
    truth = arrays["true_params"][pair]
    dots = draw_dots(ax, colour, gid=f"dots{suffix}", marker=marker)
    draw_truth(ax, truth[0], truth[1], colors.truth, gid=f"truth{suffix}")
    ellipse = draw_ellipse(ax, colour, gid=f"ellipse{suffix}")

    def update(row: int) -> None:
        points = arrays["cloud"][row][:, pair].astype(np.float64)
        contour = ellipse_points(
            arrays["mean"][row][pair], arrays["cov"][row][np.ix_(pair, pair)]
        )
        dots.set_data(points[:, 0], points[:, 1])
        ellipse.set_data(contour[:, 0], contour[:, 1])

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


def _key_ellipse(ax: Axes, colors: Palette) -> None:
    """The ellipse's direct label, just above the map at its right."""
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


def _key_fill(ax: Axes, colors: Palette) -> None:
    """The clock fill's direct label, below the map at its right."""
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
