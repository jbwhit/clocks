"""The track clips: a cloud over (x, M), the clocks under it, per-clock slowing.

``build_1d`` draws ``demo_1d`` and ``build_multi_1d`` draws ``demo_multi_mass``:
the particle cloud over position and mass with its truth, covariance ellipse
and a fixed zoom inset per mass; a strip of the clocks beneath it on the same
x scale; a panel of each clock's slowing (truth, prediction and running mean
of the readings, side by side); and one trace per parameter. Call them inside
``with style(theme) as colors:``.
"""

import math
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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

# Figure fractions, from the probe (Task 1); an inset's are fractions of "cloud".
LAYOUT_1D: dict[str, Rect] = {
    "cloud": (0.085, 0.50, 0.50, 0.40),
    "inset": (0.0, 0.52, 0.30, 0.48),
    "strip": (0.085, 0.43, 0.50, 0.05),
    "slowing": (0.085, 0.11, 0.50, 0.27),
    "trace-x": (0.68, 0.56, 0.30, 0.32),
    "trace-M": (0.68, 0.11, 0.30, 0.32),
}
LAYOUT_MULTI_1D: dict[str, Rect] = {
    "cloud": (0.085, 0.50, 0.50, 0.41),
    "inset-1": (0.0, 0.54, 0.28, 0.46),
    "inset-2": (0.72, 0.54, 0.28, 0.46),
    "strip": (0.085, 0.435, 0.50, 0.045),
    "slowing": (0.085, 0.10, 0.50, 0.28),
    "trace-x₁": (0.68, 0.765, 0.30, 0.13),
    "trace-x₂": (0.68, 0.545, 0.30, 0.13),
    "trace-M₁": (0.68, 0.325, 0.30, 0.13),
    "trace-M₂": (0.68, 0.10, 0.30, 0.13),
}
HEIGHT_1D = 450
HEIGHT_MULTI_1D = 500
MASS_AXIS = "M"  # the cloud's y label, and its insets' y name
SIDE_OFFSET = 0.5  # x distance between a clock's three marks in "slowing"
BAR_WIDTH = 1.4


class _Mass(NamedTuple):
    """One mass of a cloud: its (x, M) columns, how it is drawn, its gids."""

    columns: tuple[int, int]
    role: str
    marker: str
    suffix: str
    inset: str


def slowing_limits(replay: Replay) -> tuple[float, float]:
    """The slowing panel's y-limits: every value it shows, at every row.

    The smallest interval holding ``1 - true_rates``, the prediction's
    ``1 - predicted_mean`` plus or minus ``predicted_std`` at every row (the
    prior's included, decision D19), and each running mean of the readings
    plus or minus ``noise_std / sqrt(k)`` for k from 1; widened by 4% of its
    width on each side.
    """
    arrays = replay.arrays
    running = running_slowing(arrays["observations"])[1:]
    counts = np.arange(1, len(running) + 1)[:, None]
    half = float(replay.meta["noise_std"]) / np.sqrt(counts)
    centre = 1.0 - arrays["predicted_mean"]
    values = np.concatenate(
        [
            (1.0 - arrays["true_rates"]).ravel(),
            (centre - arrays["predicted_std"]).ravel(),
            (centre + arrays["predicted_std"]).ravel(),
            (running - half).ravel(),
            (running + half).ravel(),
        ]
    )
    low, high = float(values.min()), float(values.max())
    pad = 0.04 * (high - low)
    return low - pad, high + pad


def build_1d(replay: Replay, colors: Palette) -> Clip:
    """The ``demo_1d`` clip: one mass's cloud, the clocks, slowing and traces."""
    masses = [_Mass((0, 1), "belief", "o", "", "inset")]
    return _build_track(replay, colors, LAYOUT_1D, HEIGHT_1D, masses)


def build_multi_1d(replay: Replay, colors: Palette) -> Clip:
    """The ``demo_multi_mass`` clip: two labelled clouds, the clocks, slowing."""
    masses = [
        _Mass((0, 2), "belief", "o", "-1", "inset-1"),
        _Mass((1, 3), "belief_alt", "^", "-2", "inset-2"),
    ]
    return _build_track(replay, colors, LAYOUT_MULTI_1D, HEIGHT_MULTI_1D, masses)


def _build_track(
    replay: Replay,
    colors: Palette,
    layout: dict[str, Rect],
    height: int,
    masses: list[_Mass],
) -> Clip:
    fig = new_figure(height)
    arrays = replay.arrays
    ax = _cloud_axes(fig, replay, layout["cloud"], masses[0].columns[1])
    clouds = [
        _draw_mass(ax, replay, colors, mass, layout[mass.inset]) for mass in masses
    ]
    _label_cloud(ax, colors, centred=len(masses) > 1)
    _draw_strip(fig, replay, colors, layout["strip"])
    slowing = _draw_slowing(fig, replay, colors, layout["slowing"])
    status = draw_status(fig, colors)
    traces = draw_traces(
        fig,
        replay,
        colors,
        {key: rect for key, rect in layout.items() if key.startswith("trace-")},
    )

    def update(row: int, azimuth: float | None) -> None:
        for cloud in clouds:
            cloud(row)
        slowing(row)
        status(row)
        traces(row)

    update(len(arrays["mean"]) - 1, None)
    if len(masses) > 1:
        truth = arrays["true_params"]
        label_masses(
            ax,
            [
                (float(truth[i]), float(truth[j]))
                for i, j in (m.columns for m in masses)
            ],
            colors,
        )
    return Clip(fig, update)


def _cloud_axes(fig: Figure, replay: Replay, rect: Rect, mass_column: int) -> Axes:
    """The (x, M) panel over the prior ranges; its x ticks are the strip's."""
    priors = replay.meta["prior_ranges"]
    ax = fig.add_axes(rect)
    ax.set_gid("cloud")
    ax.set_xlim(*priors[0])
    ax.set_ylim(*priors[mass_column])
    ax.set_ylabel(MASS_AXIS)
    ax.tick_params(labelbottom=False)
    return ax


def _draw_mass(
    ax: Axes, replay: Replay, colors: Palette, mass: _Mass, inset_rect: Rect
) -> Callable[[int], None]:
    """One mass's dots, truth and ellipse, and their copies in its inset."""
    arrays = replay.arrays
    pair = list(mass.columns)
    colour = getattr(colors, mass.role)
    truth = arrays["true_params"][pair]
    dots = draw_dots(ax, colour, gid=f"dots{mass.suffix}", marker=mass.marker)
    draw_truth(ax, truth[0], truth[1], colors.truth, gid=f"truth{mass.suffix}")
    ellipse = draw_ellipse(ax, colour, gid=f"ellipse{mass.suffix}")
    inset = _draw_inset(ax, replay, colors, mass, inset_rect)
    prefix = f"{mass.inset}-"
    inset_dots = draw_dots(
        inset, colour, gid=f"{prefix}dots{mass.suffix}", marker=mass.marker
    )
    draw_truth(
        inset, truth[0], truth[1], colors.truth, gid=f"{prefix}truth{mass.suffix}"
    )
    inset_ellipse = draw_ellipse(inset, colour, gid=f"{prefix}ellipse{mass.suffix}")

    def update(row: int) -> None:
        points = arrays["cloud"][row][:, pair].astype(np.float64)
        contour = ellipse_points(
            arrays["mean"][row][pair], arrays["cov"][row][np.ix_(pair, pair)]
        )
        for line in (dots, inset_dots):
            line.set_data(points[:, 0], points[:, 1])
        for line in (ellipse, inset_ellipse):
            line.set_data(contour[:, 0], contour[:, 1])

    return update


def _draw_inset(
    ax: Axes, replay: Replay, colors: Palette, mass: _Mass, rect: Rect
) -> Axes:
    """The fixed zoom window on one mass's truth, from the last row, labelled."""
    arrays, priors = replay.arrays, replay.meta["prior_ranges"]
    pair = list(mass.columns)
    window = zoom_window(
        arrays["true_params"][pair],
        arrays["mean"][-1][pair],
        standard_deviations(replay)[-1][pair],
        [priors[i] for i in pair],
    )
    inset = draw_inset(ax, rect, window, colors, gid=mass.inset)
    x_factor = rect[2] * np.ptp(ax.get_xlim()) / np.ptp(window[0])
    y_factor = rect[3] * np.ptp(ax.get_ylim()) / np.ptp(window[1])
    inset.text(
        0.0,
        1.03,
        zoom_label(x_factor, y_factor, MASS_AXIS),
        transform=inset.transAxes,
        ha="left",
        va="bottom",
        fontsize=SMALL + 0.5,
        color=colors.muted,
        clip_on=False,
    )
    return inset


def _label_cloud(ax: Axes, colors: Palette, *, centred: bool) -> None:
    """The ellipse's key above the panel: at its right, or centred over two."""
    ax.text(
        0.5 if centred else 1.0,
        1.015,
        "covariance, 2 s.d.",
        transform=ax.transAxes,
        ha="center" if centred else "right",
        va="bottom",
        fontsize=SMALL + 0.5,
        color=colors.belief,
        clip_on=False,
    )


def _draw_strip(fig: Figure, replay: Replay, colors: Palette, rect: Rect) -> None:
    """The clocks as squares at their x, under the cloud, on its x scale."""
    xs = replay.arrays["clock_positions"][:, 0]
    strip = fig.add_axes(rect)
    strip.set_gid("strip")
    strip.set_xlim(*replay.meta["prior_ranges"][0])
    strip.set_ylim(-1, 1)
    strip.axis("off")
    (clocks,) = strip.plot(
        xs,
        np.zeros_like(xs),
        linestyle="none",
        marker="s",
        markersize=6,
        color=colors.ink,
    )
    clocks.set_gid("clocks")
    strip.text(
        0.0,
        0.5,
        "clocks ",
        transform=strip.transAxes,
        ha="right",
        va="center",
        fontsize=SMALL + 0.5,
        color=colors.muted,
    )


def _draw_slowing(
    fig: Figure, replay: Replay, colors: Palette, rect: Rect
) -> Callable[[int], None]:
    """Per clock: the true slowing, the prediction and the running mean.

    The three stand side by side, ``SIDE_OFFSET`` apart in x: the truth left
    of the clock, the prediction (with a bar of one standard deviation) on
    it, and the running mean of the readings (with a bar of the noise over
    the square root of their count) to its right, from the first reading.
    """
    arrays = replay.arrays
    ax = fig.add_axes(rect)
    ax.set_gid("slowing")
    ax.set_xlim(*replay.meta["prior_ranges"][0])
    ax.set_ylim(*slowing_limits(replay))
    ax.set_xlabel("x")
    ax.set_ylabel("slowing (1 − rate)")
    ax.tick_params(labelsize=SMALL)
    xs = arrays["clock_positions"][:, 0]
    d = SIDE_OFFSET
    (truth,) = ax.plot(
        xs - d,
        1.0 - arrays["true_rates"],
        linestyle="none",
        marker="_",
        markersize=10,
        markeredgewidth=2.0,
        color=colors.truth,
    )
    truth.set_gid("slowing-truth")
    (predicted,) = ax.plot(
        [], [], linestyle="none", marker="o", markersize=3, color=colors.belief
    )
    predicted.set_gid("slowing-predicted")
    predicted_bar = ax.vlines(xs, 0, 0, colors=colors.belief, linewidth=BAR_WIDTH)
    predicted_bar.set_gid("slowing-predicted-bar")
    (observed,) = ax.plot(
        [], [], linestyle="none", marker="s", markersize=2.6, color=colors.ink
    )
    observed.set_gid("slowing-observed")
    observed_bar = ax.vlines(xs + d, 0, 0, colors=colors.ink, linewidth=BAR_WIDTH)
    observed_bar.set_gid("slowing-observed-bar")
    running = running_slowing(arrays["observations"])
    noise = float(replay.meta["noise_std"])

    def update(row: int) -> None:
        centre = 1.0 - arrays["predicted_mean"][row]
        predicted.set_data(xs, centre)
        predicted_bar.set_segments(_bars(xs, centre, arrays["predicted_std"][row]))
        # Row 0 has no reading: its means are NaN and only visibility hides them.
        observed.set_data(xs + d, running[row])
        half = noise / math.sqrt(row) if row else 0.0
        observed_bar.set_segments(_bars(xs + d, running[row], half))
        observed.set_visible(row > 0)
        observed_bar.set_visible(row > 0)

    return update


def _bars(x: np.ndarray, centre: np.ndarray, half: np.ndarray | float) -> np.ndarray:
    """Vertical segments at ``x`` from ``centre - half`` to ``centre + half``."""
    x, low, high = np.broadcast_arrays(x, centre - half, centre + half)
    return np.stack([np.column_stack([x, low]), np.column_stack([x, high])], axis=1)
