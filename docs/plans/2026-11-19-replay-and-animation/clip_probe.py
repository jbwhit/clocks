"""Probe the six replay clips before any test pins a layout (Task 1, clips plan).

Throwaway code: it draws every clip from the committed archives, times and
decodes them, repeats the small-clip decode measurements and builds a scratch
Quarto project. Tasks 3 to 6 copy ``LAYOUTS``, ``SIDE_OFFSET`` and ``GUTTER``.
Run from the repository root, with exactly one of the four options:

    uv run python docs/plans/2026-11-19-replay-and-animation/clip_probe.py \
        --sheets docs/plans/2026-11-19-replay-and-animation
    uv run --with imageio-ffmpeg==0.8.1 python <this script> --time DIR
    uv run --with imageio-ffmpeg==0.8.1 python <this script> --decode DIR
    uv run --with imageio-ffmpeg==0.8.1 python <this script> --quarto DIR

Machine details (paths, interpreter) are printed for the run's log only.
"""

import argparse
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from clocks._figures import trace_header, trace_limits
from clocks._replay import Replay, load_replay
from clocks._style import THEMES, Palette, style

ROOT = Path(__file__).resolve().parents[3]
FPS = 20
HEIGHTS = {
    "demo_1d": 450,
    "demo_2d": 450,
    "demo_multi_mass": 500,
    "demo_multi_mass_2d": 550,
    "demo_model_comparison": 420,
    "demo_echolocation_3d": 550,
}
EXPECTED_FRAMES = {
    "demo_1d": 296,
    "demo_2d": 296,
    "demo_multi_mass": 356,
    "demo_multi_mass_2d": 356,
    "demo_model_comparison": 356,
    "demo_echolocation_3d": 356,
}

Rect = tuple[float, float, float, float]  # left, bottom, width, height

# Figure fractions; an inset's rectangle is in fractions of its parent axes.
LAYOUTS: dict[str, dict[str, Rect]] = {
    "demo_1d": {
        "cloud": (0.085, 0.50, 0.50, 0.40),
        "inset": (0.0, 0.52, 0.30, 0.48),
        "strip": (0.085, 0.43, 0.50, 0.05),
        "slowing": (0.085, 0.11, 0.50, 0.27),
        "trace-x": (0.68, 0.56, 0.30, 0.32),
        "trace-M": (0.68, 0.11, 0.30, 0.32),
    },
    "demo_2d": {
        "map": (0.075, 0.12, 0.468, 0.78),
        "inset": (0.0, 0.72, 0.28, 0.28),
        "trace-x": (0.655, 0.70, 0.325, 0.19),
        "trace-y": (0.655, 0.405, 0.325, 0.19),
        "trace-M": (0.655, 0.11, 0.325, 0.19),
    },
    "demo_multi_mass": {
        "cloud": (0.085, 0.50, 0.50, 0.41),
        "inset-1": (0.0, 0.54, 0.28, 0.46),
        "inset-2": (0.72, 0.54, 0.28, 0.46),
        "strip": (0.085, 0.435, 0.50, 0.045),
        "slowing": (0.085, 0.10, 0.50, 0.28),
        "trace-x₁": (0.68, 0.765, 0.30, 0.13),
        "trace-x₂": (0.68, 0.545, 0.30, 0.13),
        "trace-M₁": (0.68, 0.325, 0.30, 0.13),
        "trace-M₂": (0.68, 0.10, 0.30, 0.13),
    },
    "demo_multi_mass_2d": {
        "map": (0.06, 0.11, 0.484, 0.66),
        "trace-x₁": (0.62, 0.68, 0.14, 0.17),
        "trace-y₁": (0.62, 0.395, 0.14, 0.17),
        "trace-M₁": (0.62, 0.11, 0.14, 0.17),
        "trace-x₂": (0.83, 0.68, 0.14, 0.17),
        "trace-y₂": (0.83, 0.395, 0.14, 0.17),
        "trace-M₂": (0.83, 0.11, 0.14, 0.17),
    },
    "demo_model_comparison": {
        "posterior": (0.085, 0.14, 0.89, 0.74),
    },
    "demo_echolocation_3d": {
        "overview": (0.06, 0.57, 0.19, 0.259),
        "fit": (0.08, 0.11, 0.17, 0.232),
        "scene": (0.255, 0.03, 0.45, 0.90),
        "trace-x": (0.795, 0.75, 0.19, 0.15),
        "trace-y": (0.795, 0.53, 0.19, 0.15),
        "trace-z": (0.795, 0.31, 0.19, 0.15),
        "trace-M": (0.795, 0.09, 0.19, 0.15),
    },
}
SIDE_OFFSET = 0.5  # x distance between a clock's three marks in the slowing panel
GUTTER = 22.0  # width, in observations, of the model-comparison label gutter

ELEVATION = 18.0
DOT_SIZE = 2.0
DOT_ALPHA = 0.5
TRACE_TITLE_SIZE = 9.5
SMALL = 8.0


# ---------------------------------------------------------------- the schedule


def dwell_frames(n_observations: int) -> list[int]:
    """Video frames per row: 1.2 s, 0.9 s, 0.5 s, 0.25 s, 0.1 s and 2.5 s tiers."""
    counts = [24]
    for k in range(1, n_observations):
        counts.append(18 if k <= 3 else 10 if k <= 8 else 5 if k <= 20 else 2)
    return counts + [50]


def azimuth(frame: int, frames: int) -> float:
    if frames <= 1:
        return -35.0
    return -75.0 + 40.0 * (1.0 - math.cos(math.pi * frame / (frames - 1))) / 2.0


@dataclass
class Shot:
    row: int
    repeat: int
    azimuth: float | None = None


def shots(replay: Replay) -> list[Shot]:
    dwell = dwell_frames(int(replay.meta["n_observations"]))
    if replay.meta["demo"] != "demo_echolocation_3d":
        return [Shot(k, n) for k, n in enumerate(dwell)]
    frames = sum(dwell[:-1])
    rows = np.repeat(np.arange(len(dwell) - 1), dwell[:-1])
    turning = [Shot(int(r), 1, azimuth(f, frames)) for f, r in enumerate(rows)]
    return turning + [Shot(len(dwell) - 1, dwell[-1], azimuth(frames - 1, frames))]


# ---------------------------------------------------------------- shared parts


@dataclass
class Clip:
    fig: Figure
    update: Callable[[int, float | None], None]

    def show(self, row: int, azimuth: float | None = None) -> None:
        self.update(row, azimuth)


def new_figure(height: int) -> Figure:
    fig = plt.figure(figsize=(7.5, height / 100))
    fig.set_layout_engine("none")
    return fig


def status_text(row: int) -> str:
    if row == 0:
        return "Before any observation"
    return "After 1 observation" if row == 1 else f"After {row} observations"


def draw_status(fig: Figure, colors: Palette) -> Callable[[int], None]:
    text = fig.text(
        0.012, 0.985, "", ha="left", va="top", fontsize=11, color=colors.ink
    )
    text.set_gid("status")
    return lambda row: text.set_text(status_text(row))


def standard_deviations(replay: Replay) -> NDArray[np.float64]:
    return np.sqrt(np.diagonal(replay.arrays["cov"], axis1=1, axis2=2))


def draw_traces(
    fig: Figure,
    replay: Replay,
    colors: Palette,
    rects: Mapping[str, Rect],
    title_size: float = TRACE_TITLE_SIZE,
) -> Callable[[int], None]:
    names = list(replay.meta["param_names"])
    priors = replay.meta["prior_ranges"]
    mean, sd = replay.arrays["mean"], standard_deviations(replay)
    last = len(mean) - 1
    keys = [key for key in rects if key.startswith("trace-")]
    bottom = min(rects[key][1] for key in keys)
    updates = []
    for key in keys:
        j = names.index(key.removeprefix("trace-"))
        ax = fig.add_axes(rects[key])
        ax.set_gid(key)
        truth = float(replay.arrays["true_params"][j])
        ax.plot([0, last], [truth, truth], color=colors.truth, linewidth=1.3)[
            0
        ].set_gid("trace-truth")
        (line,) = ax.plot([], [], color=colors.belief, linewidth=1.5)
        line.set_gid("trace-mean")
        ax.set_xlim(0, last)
        ax.set_ylim(*trace_limits(mean[:, j], sd[:, j], truth, priors[j]))
        ax.tick_params(labelsize=SMALL)
        if rects[key][1] == bottom:
            ax.set_xlabel("observations", fontsize=SMALL + 1)
        else:
            ax.tick_params(labelbottom=False)
        updates.append(
            _trace_updater(ax, line, mean, sd, j, truth, priors[j], key, title_size)
        )
    band_colour = colors.belief

    def update(row: int) -> None:
        for function in updates:
            function(row, band_colour)

    return update


def _trace_updater(ax, line, mean, sd, j, truth, prior, key, title_size):
    state: dict[str, object] = {"band": None}
    name = key.removeprefix("trace-")

    def update(row: int, colour: str) -> None:
        steps = np.arange(row + 1)
        line.set_data(steps, mean[: row + 1, j])
        if state["band"] is not None:
            state["band"].remove()
        band = ax.fill_between(
            steps,
            mean[: row + 1, j] - sd[: row + 1, j],
            mean[: row + 1, j] + sd[: row + 1, j],
            color=colour,
            alpha=0.2,
            linewidth=0,
        )
        band.set_gid("trace-band")
        band.set_visible(row > 0)
        state["band"] = band
        header = trace_header(name, truth, mean[row, j], sd[row, j], prior)
        ax.set_title(header, fontsize=title_size)

    return update


def draw_dots(ax: Axes, color: str, gid: str = "dots", marker: str = "o") -> Line2D:
    (line,) = ax.plot(
        [],
        [],
        linestyle="none",
        marker=marker,
        markersize=DOT_SIZE if marker == "o" else DOT_SIZE + 0.8,
        markeredgewidth=0,
        alpha=DOT_ALPHA,
        color=color,
        zorder=3,
    )
    line.set_gid(gid)
    return line


def ellipse_points(mean: NDArray, cov: NDArray) -> NDArray[np.float64]:
    w, v = np.linalg.eigh(np.asarray(cov, dtype=np.float64))
    t = np.linspace(0.0, 2.0 * np.pi, 121)
    t[-1] = 0.0
    circle = np.stack([np.cos(t), np.sin(t)])
    return (
        np.asarray(mean)[:, None] + 2.0 * (v * np.sqrt(np.clip(w, 0, None))) @ circle
    ).T


def draw_ellipse(ax: Axes, color: str, gid: str = "ellipse") -> Line2D:
    (line,) = ax.plot([], [], color=color, linewidth=1.2, zorder=4)
    line.set_gid(gid)
    return line


def draw_truth(ax: Axes, x: float, y: float, color: str, gid: str = "truth") -> Line2D:
    (line,) = ax.plot(
        [x],
        [y],
        linestyle="none",
        marker="o",
        markersize=9,
        markerfacecolor="none",
        markeredgecolor=color,
        markeredgewidth=1.6,
        zorder=5,
    )
    line.set_gid(gid)
    return line


def zoom_window(truth, mean, sd, prior_ranges) -> NDArray[np.float64]:
    window = []
    for t, m, s, (low, high) in zip(truth, mean, sd, prior_ranges, strict=True):
        h = 1.25 * (abs(m - t) + 2.0 * s)
        h = h if h > 0 else 0.01 * (high - low)
        window.append((t - h, t + h))
    return np.array(window)


def zoom_label(x_factor: float, y_factor: float, y_name: str) -> str:
    if round(x_factor) == round(y_factor):
        return f"zoom ×{round(x_factor)}"
    return f"zoom ×{round(x_factor)} (x), ×{round(y_factor)} ({y_name})"


def draw_inset(ax: Axes, rect: Rect, window: NDArray, colors: Palette, gid: str):
    inset = ax.inset_axes(rect)
    inset.set_gid(gid)
    inset.set_xlim(*window[0])
    inset.set_ylim(*window[1])
    inset.set_xticks([])
    inset.set_yticks([])
    inset.grid(False)
    inset.set_facecolor(colors.background)
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(colors.muted)
    x_span, y_span = np.diff(ax.get_xlim())[0], np.diff(ax.get_ylim())[0]
    x_factor = rect[2] * x_span / np.diff(window[0])[0]
    y_factor = rect[3] * y_span / np.diff(window[1])[0]
    return inset, (x_factor, y_factor)


def running_slowing(observations: NDArray) -> NDArray[np.float64]:
    slowing = 1.0 - np.asarray(observations, dtype=np.float64)
    steps = np.arange(1, len(slowing) + 1)[:, None]
    running = np.cumsum(slowing, axis=0) / steps
    nan_row = np.full((1, slowing.shape[1]), np.nan)
    return np.vstack([nan_row, running])


def clock_fill(running: NDArray, largest: float, colors: Palette) -> list[tuple]:
    background = np.array(to_rgba(colors.background))
    ink = np.array(to_rgba(colors.ink))
    fills = []
    for value in running:
        if np.isnan(value):
            fills.append((0.0, 0.0, 0.0, 0.0))
            continue
        t = min(max(value / largest, 0.0), 1.0) if largest > 0 else 0.0
        rgb = background[:3] + t * (ink[:3] - background[:3])
        fills.append((*rgb, 1.0))
    return fills


def add_axes(fig: Figure, rect: Rect, gid: str, **kwargs) -> Axes:
    ax = fig.add_axes(rect, **kwargs)
    ax.set_gid(gid)
    return ax


def cloud_with_inset(ax, replay, colors, columns, role, marker, suffix, rect):
    """Dots, truth, ellipse and (when ``rect``) a zoom inset for one mass."""
    arrays, priors = replay.arrays, replay.meta["prior_ranges"]
    i, j = columns
    tag = f"-{suffix}" if suffix else ""
    dots = draw_dots(ax, getattr(colors, role), f"dots{tag}", marker)
    truth = arrays["true_params"]
    draw_truth(ax, truth[i], truth[j], colors.truth, f"truth{tag}")
    ellipse = draw_ellipse(ax, getattr(colors, role), f"ellipse{tag}")
    copies = []
    if rect is not None:
        sd = standard_deviations(replay)[-1]
        window = zoom_window(
            truth[[i, j]],
            arrays["mean"][-1][[i, j]],
            sd[[i, j]],
            [priors[i], priors[j]],
        )
        gid = f"inset{tag}"
        inset, factors = draw_inset(ax, rect, window, colors, gid)
        copies = [
            draw_dots(inset, getattr(colors, role), f"{gid}-dots{tag}", marker),
            draw_ellipse(inset, getattr(colors, role), f"{gid}-ellipse{tag}"),
        ]
        draw_truth(inset, truth[i], truth[j], colors.truth, f"{gid}-truth{tag}")
        y_name = str(replay.meta["param_names"][j])
        inset.text(
            0.0,
            1.03,
            zoom_label(*factors, y_name),
            transform=inset.transAxes,
            ha="left",
            va="bottom",
            fontsize=SMALL + 0.5,
            color=colors.muted,
            clip_on=False,
        )

    def update(row: int) -> None:
        points = arrays["cloud"][row][:, [i, j]].astype(np.float64)
        contour = ellipse_points(
            arrays["mean"][row][[i, j]], arrays["cov"][row][np.ix_([i, j], [i, j])]
        )
        for artist in [dots, *copies[:1]]:
            artist.set_data(points[:, 0], points[:, 1])
        for artist in [ellipse, *copies[1:]]:
            artist.set_data(contour[:, 0], contour[:, 1])

    return update


# ---------------------------------------------------------------- the map clips


def build_2d(replay: Replay, colors: Palette) -> Clip:
    layout = LAYOUTS["demo_2d"]
    fig = new_figure(HEIGHTS["demo_2d"])
    arrays, priors = replay.arrays, replay.meta["prior_ranges"]
    ax = add_axes(fig, layout["map"], "map")
    ax.set_xlim(*priors[0])
    ax.set_ylim(*priors[1])
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.tick_params(pad=5)
    running = running_slowing(arrays["observations"])
    largest = float(np.nanmax(running))
    clocks = ax.scatter(
        arrays["clock_positions"][:, 0],
        arrays["clock_positions"][:, 1],
        marker="s",
        s=42,
        edgecolors=colors.ink,
        linewidths=1.0,
        zorder=2,
    )
    clocks.set_gid("clocks")
    cloud = cloud_with_inset(
        ax, replay, colors, (0, 1), "belief", "o", "", layout["inset"]
    )
    key_above(ax, "covariance, 2 s.d.", colors.belief)
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
    status = draw_status(fig, colors)
    traces = draw_traces(fig, replay, colors, layout)

    def update(row: int, azim: float | None) -> None:
        clocks.set_facecolors(clock_fill(running[row], largest, colors))
        cloud(row)
        status(row)
        traces(row)

    update(len(arrays["mean"]) - 1, None)
    return Clip(fig, update)


def build_multi_2d(replay: Replay, colors: Palette) -> Clip:
    layout = LAYOUTS["demo_multi_mass_2d"]
    fig = new_figure(HEIGHTS["demo_multi_mass_2d"])
    arrays, priors = replay.arrays, replay.meta["prior_ranges"]
    ax = add_axes(fig, layout["map"], "map")
    ax.set_xlim(*priors[0])
    ax.set_ylim(*priors[1])
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.tick_params(pad=5)
    (clocks,) = ax.plot(
        arrays["clock_positions"][:, 0],
        arrays["clock_positions"][:, 1],
        linestyle="none",
        marker="s",
        markersize=5,
        color=colors.ink,
        zorder=2,
    )
    clocks.set_gid("clocks")
    first = cloud_with_inset(ax, replay, colors, (0, 1), "belief", "o", "1", None)
    second = cloud_with_inset(ax, replay, colors, (2, 3), "belief_alt", "^", "2", None)
    key_above(ax, "covariance, 2 s.d.", colors.belief)
    status = draw_status(fig, colors)
    traces = draw_traces(fig, replay, colors, layout, title_size=8.5)

    def update(row: int, azim: float | None) -> None:
        first(row)
        second(row)
        status(row)
        traces(row)

    update(len(arrays["mean"]) - 1, None)
    truth = arrays["true_params"]
    label_masses(ax, colors, [(truth[0], truth[1]), (truth[2], truth[3])])
    return Clip(fig, update)


def key_above(ax: Axes, text: str, colour: str, x: float = 1.0, ha="right"):
    """A direct label just above a panel, clear of every data mark."""
    ax.text(
        x,
        1.015,
        text,
        transform=ax.transAxes,
        ha=ha,
        va="bottom",
        fontsize=SMALL + 0.5,
        color=colour,
        clip_on=False,
    )


DIRECTIONS = [(1, 1), (-1, 1), (1, -1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)]


def label_masses(ax: Axes, colors: Palette, points: Sequence) -> None:
    """Label each truth on the side where its text covers the fewest final marks.

    Called after the final row is shown: the candidates are the eight compass
    directions, and a text may not leave the panel, cover an inset or another
    label. The choice is made once and fixed for the clip.
    """
    renderer = ax.figure.canvas.get_renderer()
    marks = [
        np.column_stack(line.get_data())
        for line in ax.get_lines()
        if len(line.get_xdata())
    ]
    marks = ax.transData.transform(np.vstack(marks).astype(float))
    frame = ax.get_window_extent(renderer)
    blocked = [inset.get_window_extent(renderer) for inset in ax.child_axes]
    for n, (x, y) in enumerate(points, start=1):
        best = None
        for dx, dy in DIRECTIONS:
            label = ax.annotate(
                f"mass {n}",
                (x, y),
                xytext=(9 * dx, 7 * dy),
                textcoords="offset points",
                ha={1: "left", -1: "right", 0: "center"}[dx],
                va={1: "bottom", -1: "top", 0: "center"}[dy],
                fontsize=SMALL + 0.5,
                color=colors.truth,
                zorder=6,
            )
            box = label.get_window_extent(renderer).padded(2)
            inside = (
                (marks[:, 0] >= box.x0)
                & (marks[:, 0] <= box.x1)
                & (marks[:, 1] >= box.y0)
                & (marks[:, 1] <= box.y1)
            )
            cost = int(inside.sum())
            if not frame.contains(box.x0, box.y0) or not frame.contains(box.x1, box.y1):
                cost += 10_000
            cost += 10_000 * sum(box.overlaps(other) for other in blocked)
            if best is None or cost < best[0]:
                if best is not None:
                    best[1].remove()
                best = (cost, label, box)
            else:
                label.remove()
        blocked.append(best[2])


# ---------------------------------------------------------------- the track clips


def slowing_limits(replay: Replay) -> tuple[float, float]:
    arrays = replay.arrays
    running = running_slowing(arrays["observations"])[1:]
    steps = np.arange(1, len(running) + 1)[:, None]
    half = float(replay.meta["noise_std"]) / np.sqrt(steps)
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


def build_track(replay: Replay, colors: Palette) -> Clip:
    demo = str(replay.meta["demo"])
    layout = LAYOUTS[demo]
    two = demo == "demo_multi_mass"
    fig = new_figure(HEIGHTS[demo])
    arrays, priors = replay.arrays, replay.meta["prior_ranges"]
    m_index = 2 if two else 1
    ax = add_axes(fig, layout["cloud"], "cloud")
    ax.set_xlim(*priors[0])
    ax.set_ylim(*priors[m_index])
    ax.set_ylabel("M")
    ax.tick_params(labelbottom=False)
    if two:
        clouds = [
            cloud_with_inset(
                ax, replay, colors, (0, 2), "belief", "o", "1", layout["inset-1"]
            ),
            cloud_with_inset(
                ax, replay, colors, (1, 3), "belief_alt", "^", "2", layout["inset-2"]
            ),
        ]
        key_above(ax, "covariance, 2 s.d.", colors.belief, x=0.5, ha="center")
    else:
        clouds = [
            cloud_with_inset(
                ax, replay, colors, (0, 1), "belief", "o", "", layout["inset"]
            )
        ]
        key_above(ax, "covariance, 2 s.d.", colors.belief)
    xs = arrays["clock_positions"][:, 0]
    strip = add_axes(fig, layout["strip"], "strip")
    strip.set_xlim(*priors[0])
    strip.set_ylim(-1, 1)
    strip.axis("off")
    (clock_marks,) = strip.plot(
        xs,
        np.zeros_like(xs),
        linestyle="none",
        marker="s",
        markersize=6,
        color=colors.ink,
    )
    clock_marks.set_gid("clocks")
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
    slowing = draw_slowing(fig, replay, colors, layout["slowing"])
    status = draw_status(fig, colors)
    traces = draw_traces(fig, replay, colors, layout)

    def update(row: int, azim: float | None) -> None:
        for cloud in clouds:
            cloud(row)
        slowing(row)
        status(row)
        traces(row)

    update(len(arrays["mean"]) - 1, None)
    if two:
        truth = arrays["true_params"]
        label_masses(ax, colors, [(truth[0], truth[2]), (truth[1], truth[3])])
    return Clip(fig, update)


def draw_slowing(fig, replay, colors, rect) -> Callable[[int], None]:
    arrays, priors = replay.arrays, replay.meta["prior_ranges"]
    ax = add_axes(fig, rect, "slowing")
    ax.set_xlim(*priors[0])
    ax.set_ylim(*slowing_limits(replay))
    ax.set_xlabel("x")
    ax.set_ylabel("slowing (1 − rate)")
    ax.tick_params(labelsize=SMALL)
    xs = arrays["clock_positions"][:, 0]
    d = SIDE_OFFSET
    ax.plot(
        xs - d,
        1.0 - arrays["true_rates"],
        linestyle="none",
        marker="_",
        markersize=10,
        markeredgewidth=2.0,
        color=colors.truth,
    )[0].set_gid("slowing-truth")
    (predicted,) = ax.plot(
        [], [], linestyle="none", marker="o", markersize=3, color=colors.belief
    )
    predicted.set_gid("slowing-predicted")
    (observed,) = ax.plot(
        [], [], linestyle="none", marker="s", markersize=2.6, color=colors.ink
    )
    observed.set_gid("slowing-observed")
    running = running_slowing(arrays["observations"])
    noise = float(replay.meta["noise_std"])
    bars: dict[str, object] = {}

    def update(row: int) -> None:
        for artist in bars.values():
            artist.remove()
        centre = 1.0 - arrays["predicted_mean"][row]
        spread = arrays["predicted_std"][row]
        predicted.set_data(xs, centre)
        bars["predicted"] = ax.vlines(
            xs, centre - spread, centre + spread, color=colors.belief, linewidth=1.4
        )
        bars["predicted"].set_gid("slowing-predicted-bar")
        observed.set_data(xs + d, running[row])
        half = noise / math.sqrt(row) if row else 0.0
        bars["observed"] = ax.vlines(
            xs + d,
            running[row] - half,
            running[row] + half,
            color=colors.ink,
            linewidth=1.4,
        )
        bars["observed"].set_gid("slowing-observed-bar")
        observed.set_visible(row > 0)
        bars["observed"].set_visible(row > 0)

    return update


# ---------------------------------------------------------------- model comparison


def build_models(replay: Replay, colors: Palette) -> Clip:
    layout = LAYOUTS["demo_model_comparison"]
    fig = new_figure(HEIGHTS["demo_model_comparison"])
    arrays = replay.arrays
    ks = [int(k) for k in arrays["model_ks"]]
    if len(ks) > 3:
        raise ValueError("the model clip draws at most three candidates")
    posterior = arrays["model_posterior"]
    last = len(posterior) - 1
    ax = add_axes(fig, layout["posterior"], "posterior")
    ax.set_xlim(0, last + GUTTER)
    ax.set_xticks([t for t in range(0, last + 1, 20 if last > 40 else 2)])
    ax.spines["bottom"].set_bounds(0, last)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("observations")
    ax.set_ylabel("posterior probability")
    label_x = last + 0.12 * GUTTER
    places = [0.80 - 0.30 * i for i in range(len(ks))]
    lines, connectors = [], []
    for i, k in enumerate(ks):
        colour = colors.ordered[i]
        (line,) = ax.plot([], [], color=colour, linewidth=1.8)
        line.set_gid(f"model-{k}")
        (connector,) = ax.plot([], [], color=colour, linewidth=0.7, alpha=0.6)
        connector.set_gid(f"connector-{k}")
        text = f"K = {k} (true model)" if k == replay.meta["true_k"] else f"K = {k}"
        ax.text(
            label_x, places[i], text, ha="left", va="center", color=colour, fontsize=10
        ).set_gid(f"label-{k}")
        lines.append(line)
        connectors.append(connector)
    status = draw_status(fig, colors)

    def update(row: int, azim: float | None) -> None:
        for i, (line, connector) in enumerate(zip(lines, connectors, strict=True)):
            line.set_data(np.arange(row + 1), posterior[: row + 1, i])
            connector.set_data([row, label_x], [posterior[row, i], places[i]])
        status(row)

    update(last, None)
    return Clip(fig, update)


# ---------------------------------------------------------------- echolocation


def in_view(points: NDArray, bounds: Sequence[float]) -> NDArray[np.bool_]:
    low, high = bounds
    return np.all((points >= low) & (points <= high), axis=1)


def view_weight_text(row: int, weight: float) -> str:
    if row == 0:
        return f"{100 * weight:.1f}% of the prior lies inside the close view"
    return f"{100 * weight:.1f}% of the cloud's weight lies inside the close view"


def build_echo(replay: Replay, colors: Palette) -> Clip:
    layout = LAYOUTS["demo_echolocation_3d"]
    fig = new_figure(HEIGHTS["demo_echolocation_3d"])
    arrays = replay.arrays
    bounds = arrays["view_bounds"]
    last = len(arrays["mean"]) - 1
    scene, dots = draw_scene(fig, replay, colors, layout["scene"])
    overview_dots = draw_overview(fig, replay, colors, layout["overview"])
    fit = draw_fit(fig, replay, colors, layout["fit"])
    status = draw_status(fig, colors)
    weight = fig.text(
        0.012, 0.945, "", ha="left", va="top", fontsize=SMALL + 1, color=colors.muted
    )
    weight.set_gid("view-weight")
    traces = draw_traces(fig, replay, colors, layout)
    final_azimuth = azimuth(
        sum(dwell_frames(last)[:-1]) - 1, sum(dwell_frames(last)[:-1])
    )

    def update(row: int, azim: float | None) -> None:
        cloud = arrays["cloud"][row].astype(np.float64)
        inside = cloud[in_view(cloud[:, :3], bounds)]
        dots.set_data_3d(inside[:, 0], inside[:, 1], inside[:, 2])
        dots.set_visible(len(inside) > 0)
        scene.view_init(elev=ELEVATION, azim=final_azimuth if azim is None else azim)
        overview_dots.set_data(cloud[:, 0], cloud[:, 2])
        fit(row)
        status(row)
        weight.set_text(view_weight_text(row, float(arrays["weight_in_view"][row])))
        traces(row)

    update(last, None)
    return Clip(fig, update)


def draw_scene(fig, replay, colors, rect):
    arrays = replay.arrays
    low, high = arrays["view_bounds"]
    scene = add_axes(fig, rect, "scene", projection="3d")
    for setter in (scene.set_xlim, scene.set_ylim, scene.set_zlim):
        setter(low, high)
    scene.set_box_aspect((1, 1, 1), zoom=1.1)
    for axis in (scene.xaxis, scene.yaxis, scene.zaxis):
        axis.set_pane_color(colors.background)
        axis.pane.set_edgecolor(colors.rule)
        axis.line.set_color(colors.rule)
    scene.tick_params(labelsize=SMALL - 1, pad=0)
    for setter, name in (
        (scene.set_xlabel, "x"),
        (scene.set_ylabel, "y"),
        (scene.set_zlabel, "z"),
    ):
        setter(name, fontsize=SMALL, labelpad=0)
    ticks = [t for t in range(math.ceil(low), math.floor(high) + 1) if t % 2 == 0]
    scene.set_xticks(ticks)
    scene.set_yticks(ticks)
    scene.set_zticks(ticks)
    clocks = arrays["clock_positions"]
    corners = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)])
    box = np.where(corners == 1, clocks.max(axis=0), clocks.min(axis=0))
    edges = [
        (box[p], box[q])
        for p in range(8)
        for q in range(p + 1, 8)
        if np.sum(corners[p] != corners[q]) == 1
    ]
    head = Line3DCollection(edges, colors=colors.muted, linewidths=0.9)
    head.set_gid("head")
    scene.add_collection3d(head)
    scene.plot(
        clocks[:, 0],
        clocks[:, 1],
        clocks[:, 2],
        linestyle="none",
        marker="s",
        markersize=3.5,
        color=colors.ink,
    )[0].set_gid("clocks")
    (dots,) = scene.plot(
        [],
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=1.2,
        markeredgewidth=0,
        alpha=DOT_ALPHA,
        color=colors.belief,
    )
    dots.set_gid("dots")
    x, y, z = arrays["true_params"][:3]
    scene.plot(
        [x],
        [y],
        [z],
        linestyle="none",
        marker="o",
        markersize=11,
        markerfacecolor="none",
        markeredgecolor=colors.truth,
        markeredgewidth=1.4,
    )[0].set_gid("truth")
    scene.plot([x, x], [y, y], [z, low], color=colors.truth, linewidth=0.9)[0].set_gid(
        "truth-drop"
    )
    return scene, dots


def draw_overview(fig, replay, colors, rect) -> Line2D:
    priors = replay.meta["prior_ranges"]
    low, high = replay.arrays["view_bounds"]
    ax = add_axes(fig, rect, "overview")
    ax.set_xlim(*priors[0])
    ax.set_ylim(*priors[2])
    ax.set_aspect("equal")
    ax.set_xticks([-16, 0, 16])
    ax.set_yticks([-16, 0, 16])
    ax.tick_params(labelsize=SMALL - 1)
    ax.set_xlabel("x", fontsize=SMALL, labelpad=3)
    ax.set_ylabel("z", fontsize=SMALL, labelpad=3)
    ax.set_title("whole prior box", fontsize=SMALL + 0.5)
    (dots,) = ax.plot(
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=1.2,
        markeredgewidth=0,
        alpha=DOT_ALPHA,
        color=colors.belief,
    )
    dots.set_gid("overview-dots")
    xs = [low, high, high, low, low]
    ys = [low, low, high, high, low]
    ax.plot(xs, ys, color=colors.muted, linewidth=1.0)[0].set_gid("view-box")
    return dots


def draw_fit(fig, replay, colors, rect) -> Callable[[int], None]:
    arrays = replay.arrays
    observations, predicted = arrays["observations"], arrays["predicted_mean"]
    rows = range(1, len(predicted))
    values = np.concatenate(
        [observations[:k].mean(axis=0) for k in rows] + [predicted[1:].ravel()]
    )
    low, high = float(values.min()), float(values.max())
    pad = 0.06 * (high - low)
    ax = add_axes(fig, rect, "fit")
    limits = (low - pad, high + pad)
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=SMALL - 1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.set_xlabel("observed mean", fontsize=SMALL, labelpad=3)
    ax.set_ylabel("fitted", fontsize=SMALL, labelpad=3)
    ax.plot(limits, limits, color=colors.muted, linewidth=0.9)[0].set_gid(
        "fit-diagonal"
    )
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


# ---------------------------------------------------------------- the registry

BUILDERS: dict[str, Callable[[Replay, Palette], Clip]] = {
    "demo_1d": build_track,
    "demo_2d": build_2d,
    "demo_multi_mass": build_track,
    "demo_multi_mass_2d": build_multi_2d,
    "demo_model_comparison": build_models,
    "demo_echolocation_3d": build_echo,
}


def _draw(demo: str, replay: Replay, colors: Palette, row: int, azim) -> Figure:
    clip = BUILDERS[demo](replay, colors)
    clip.show(row, azim)
    return clip.fig


def draw_demo_1d(replay, colors, row, azimuth=None) -> Figure:
    return _draw("demo_1d", replay, colors, row, azimuth)


def draw_demo_2d(replay, colors, row, azimuth=None) -> Figure:
    return _draw("demo_2d", replay, colors, row, azimuth)


def draw_demo_multi_mass(replay, colors, row, azimuth=None) -> Figure:
    return _draw("demo_multi_mass", replay, colors, row, azimuth)


def draw_demo_multi_mass_2d(replay, colors, row, azimuth=None) -> Figure:
    return _draw("demo_multi_mass_2d", replay, colors, row, azimuth)


def draw_demo_model_comparison(replay, colors, row, azimuth=None) -> Figure:
    return _draw("demo_model_comparison", replay, colors, row, azimuth)


def draw_demo_echolocation_3d(replay, colors, row, azimuth=None) -> Figure:
    return _draw("demo_echolocation_3d", replay, colors, row, azimuth)


def frame_rgba(fig: Figure) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(
        buffer, format="rgba", dpi=200, bbox_inches=None, facecolor=fig.get_facecolor()
    )
    return buffer.getvalue()


def committed(demo: str) -> Replay:
    return load_replay(ROOT / "assets" / "replays" / f"{demo}.npz")


def replay_2d_at_010() -> Replay:
    """The 2D run with the true mass at 0.10, recorded in memory (Q6)."""
    from clocks._demos import demo_2d

    demo_2d.TRUE_M = 0.10
    return demo_2d.record()


# ---------------------------------------------------------------- sheets

PHONE_WIDTH = 780
MARGIN = 24
LABEL_HEIGHT = 36


def sheet_rows(replay: Replay) -> list[int]:
    last = int(replay.meta["n_observations"])
    return [0, 1, 8, last]


def render_row(demo: str, replay: Replay, theme: str, row: int) -> Image.Image:
    with style(theme) as colors:
        clip = BUILDERS[demo](replay, colors)
        try:
            azim = next((s.azimuth for s in shots(replay) if s.row == row), None)
            clip.show(row, azim)
            data = frame_rgba(clip.fig)
            width, height = (round(v * 200) for v in clip.fig.get_size_inches())
            report_checks(demo, clip.fig, colors, f"{theme} row {row}")
        finally:
            plt.close(clip.fig)
    return Image.frombuffer("RGBA", (width, height), data).convert("RGB")


def compose_sheet(images: dict[tuple[int, str], Image.Image], rows) -> Image.Image:
    full_w = max(image.width for image in images.values())
    bands = []
    for row in rows:
        full = [images[row, theme] for theme in THEMES]
        phone = [
            image.resize(
                (PHONE_WIDTH, round(image.height * PHONE_WIDTH / image.width)),
                Image.LANCZOS,
            )
            for image in full
        ]
        bands.append((row, full, phone))
    width = MARGIN + 2 * (full_w + MARGIN)
    band_h = [
        2 * LABEL_HEIGHT + full[0].height + phone[0].height + 3 * MARGIN
        for _, full, phone in bands
    ]
    sheet = Image.new("RGB", (width, MARGIN + sum(band_h)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=22)
    y = MARGIN
    for row, full, phone in bands:
        for label, images_row in (("full width", full), ("390 px at 2x", phone)):
            x = MARGIN
            for theme, image in zip(THEMES, images_row, strict=True):
                text = f"row {row}, {theme}, {label} ({image.width} px)"
                draw.text((x, y), text, "black", font)
                sheet.paste(image, (x, y + LABEL_HEIGHT))
                x += full_w + MARGIN
            y += LABEL_HEIGHT + images_row[0].height + MARGIN
        y += MARGIN
    return sheet


def write_sheets(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(demo, f"clip-{demo}.jpg", committed) for demo in BUILDERS]
    jobs.insert(2, ("demo_2d", "clip-demo_2d-at-0.10.jpg", None))
    for demo, filename, loader in jobs:
        replay = loader(demo) if loader else replay_2d_at_010()
        rows = sheet_rows(replay)
        images = {
            (row, theme): render_row(demo, replay, theme, row)
            for row in rows
            for theme in THEMES
        }
        target = out / filename
        compose_sheet(images, rows).save(target, quality=85)
        print(f"wrote {filename}")


# ---------------------------------------------------------------- layout checks


def report_checks(demo: str, fig: Figure, colors: Palette, where: str) -> None:
    for problem in layout_problems(demo, fig) + text_overlaps(fig):
        print(f"  {demo} {where}: {problem}")
    off = off_palette(fig, colors)
    if off:
        print(f"  {demo} {where}: off-palette colours {sorted(off)}")


def layout_problems(demo: str, fig: Figure) -> list[str]:
    problems = []
    boxes = {ax.get_gid(): ax.get_position(original=True) for ax in fig.axes}
    for gid, box in boxes.items():
        if box.x0 < 0 or box.y0 < 0 or box.x1 > 1 or box.y1 > 1:
            problems.append(f"axes {gid} leaves the canvas: {box.bounds}")
    names = list(boxes)
    for a, first in enumerate(names):
        for second in names[a + 1 :]:
            p, q = boxes[first], boxes[second]
            w = min(p.x1, q.x1) - max(p.x0, q.x0)
            h = min(p.y1, q.y1) - max(p.y0, q.y0)
            if w > 1e-9 and h > 1e-9:
                problems.append(f"axes {first} and {second} overlap")
    if "strip" in boxes:
        for gid in ("strip", "slowing"):
            if (boxes[gid].x0, boxes[gid].width) != (
                boxes["cloud"].x0,
                boxes["cloud"].width,
            ):
                problems.append(f"{gid} does not share the cloud's x0 and width")
    return problems


def drawn_texts(fig: Figure) -> list[Text]:
    """The texts a draw actually paints (tick labels out of view are skipped)."""
    drawn: dict[int, Text] = {}
    original = Text.draw

    def recording(self, renderer):
        original(self, renderer)
        if self.get_visible() and self.get_text().strip():
            drawn[id(self)] = self

    Text.draw = recording
    try:
        fig.canvas.draw()
    finally:
        Text.draw = original
    return list(drawn.values())


def text_overlaps(fig: Figure) -> list[str]:
    renderer = fig.canvas.get_renderer()
    boxes = []
    for text in drawn_texts(fig):
        box = text.get_window_extent(renderer).padded(2)
        if box.width > 0 and box.height > 0:
            owner = text.axes.get_gid() if text.axes is not None else "fig"
            boxes.append((f"{owner}:{text.get_text()}", box))
    problems = []
    width, height = fig.bbox.width, fig.bbox.height
    for label, box in boxes:
        if box.x0 < -0.5 or box.y0 < -0.5 or box.x1 > width + 0.5 or box.y1 > height:
            problems.append(f"text {label!r} leaves the canvas")
    for a, (first, p) in enumerate(boxes):
        for second, q in boxes[a + 1 :]:
            w = min(p.x1, q.x1) - max(p.x0, q.x0)
            h = min(p.y1, q.y1) - max(p.y0, q.y0)
            if w > 0.5 and h > 0.5:
                problems.append(f"texts {first!r} and {second!r} overlap")
    return problems


def _palette_rgbs(colors: Palette) -> list[NDArray]:
    roles = [
        colors.background,
        colors.ink,
        colors.muted,
        colors.rule,
        colors.grid,
        colors.truth,
        colors.belief,
        colors.belief_alt,
        colors.threshold,
        *colors.ordered,
    ]
    return [np.array(to_rgba(role)[:3]) for role in roles]


def off_palette(fig: Figure, colors: Palette) -> set[str]:
    """Colours of visible artists that are neither the palette nor an ink blend."""
    fig.canvas.draw()
    allowed = _palette_rgbs(colors)
    background = np.array(to_rgba(colors.background)[:3])
    ink = np.array(to_rgba(colors.ink)[:3])
    found: list[tuple] = []
    for artist in fig.findobj():
        if not artist.get_visible():
            continue
        for getter in (
            "get_color",
            "get_markerfacecolor",
            "get_markeredgecolor",
            "get_facecolor",
            "get_edgecolor",
        ):
            if getter == "get_edgecolor" and _no_edge(artist):
                continue
            if hasattr(artist, getter):
                try:
                    value = getattr(artist, getter)()
                except (TypeError, ValueError, AttributeError):
                    continue
                found.extend(_as_rgba_list(value))
    for ax in fig.axes:
        if ax.name == "3d":
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                found.extend(_as_rgba_list(axis.pane.get_facecolor()))
                found.extend(_as_rgba_list(axis.pane.get_edgecolor()))
                found.extend(_as_rgba_list(axis.line.get_color()))
                found.extend(_as_rgba_list(axis.gridlines.get_color()))
    off = set()
    for rgba in found:
        if rgba[3] == 0:
            continue
        rgb = np.array(rgba[:3])
        if any(np.all(np.abs(rgb - c) <= 1.5 / 255) for c in allowed):
            continue
        span = ink - background
        t = np.clip(np.dot(rgb - background, span) / np.dot(span, span), 0, 1)
        if np.all(np.abs(background + t * span - rgb) <= 1.5 / 255):
            continue
        off.add("#{:02x}{:02x}{:02x}".format(*(round(255 * c) for c in rgb)))
    return off


def _no_edge(artist) -> bool:
    if not hasattr(artist, "get_linewidth"):
        return False
    widths = np.atleast_1d(np.asarray(artist.get_linewidth(), dtype=float))
    return widths.size == 0 or bool(np.all(widths == 0))


def _as_rgba_list(value) -> list[tuple]:
    if isinstance(value, str):
        return [] if value == "none" else [to_rgba(value)]
    array = np.asarray(value, dtype=object)
    if array.ndim == 0:
        return []
    try:
        numeric = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return [to_rgba(v) for v in value if not (isinstance(v, str) and v == "none")]
    if numeric.ndim == 1 and numeric.size in (3, 4):
        return [to_rgba(numeric)]
    if numeric.ndim == 2:
        return [to_rgba(row) for row in numeric]
    return []


# ---------------------------------------------------------------- the encoder


def find_encoder() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        found = shutil.which("ffmpeg")
        if found is None:
            raise SystemExit("no FFmpeg: run with --with imageio-ffmpeg==0.8.1")
        return found


def describe_encoder(encoder: str) -> None:
    result = subprocess.run([encoder, "-version"], capture_output=True, text=True)
    print(f"encoder path (log only): {encoder}")
    print(f"encoder version line: {result.stdout.splitlines()[0]}")
    print(f"interpreter: {sys.version.split()[0]}; prefix (log only): {sys.prefix}")


def encode_frames(
    encoder: str,
    frames: Iterator[tuple[bytes, int]],
    path: Path,
    size: tuple[int, int],
    *,
    fps: int = FPS,
) -> int:
    width, height = size
    command = [
        encoder, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "20", "-preset", "slow", "-movflags", "+faststart", "-f", "mp4",
        str(path),
    ]  # fmt: skip
    written = 0
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
        try:
            for frame, repeat in frames:
                if len(frame) != width * height * 4:
                    raise ValueError(f"frame of {len(frame)} bytes for {size}")
                for _ in range(repeat):
                    process.stdin.write(frame)
                    written += 1
        finally:
            process.stdin.close()
            status = process.wait()
        errors.seek(0)
        text = errors.read().decode(errors="replace")
    if status != 0:
        raise RuntimeError(f"encoder exit {status}: {text[-800:]}")
    return written


@dataclass
class Decoded:
    status: int
    video_line: str | None
    frame_line: str | None
    progress_line: str | None

    def properties(self) -> tuple | None:
        if self.video_line is None:
            return None
        match = re.search(
            r"Video: (\w+).*?, (\w+)[(,].*?(\d+)x(\d+).*?([\d.]+) fps", self.video_line
        )
        if match is None:
            return None
        codec, pix_fmt, w, h, fps = match.groups()
        return codec, pix_fmt, int(w), int(h), float(fps)

    def frames(self) -> int | None:
        if self.frame_line is None:
            return None
        return int(self.frame_line.split("=", 1)[1])


def decode(encoder: str, path: Path) -> Decoded:
    command = [
        encoder, "-hide_banner", "-nostdin", "-i", str(path), "-map", "0:v:0",
        "-f", "null", "-", "-progress", "pipe:1", "-nostats",
    ]  # fmt: skip
    result = subprocess.run(command, capture_output=True, text=True)
    video = next((x for x in result.stderr.splitlines() if "Video:" in x), None)
    lines = [x.strip() for x in result.stdout.splitlines() if x.strip()]
    frame = next((x for x in reversed(lines) if x.startswith("frame=")), None)
    progress = next((x for x in reversed(lines) if x.startswith("progress=")), None)
    return Decoded(result.returncode, video.strip() if video else None, frame, progress)


def box_order(path: Path) -> list[str]:
    data, boxes, offset = path.read_bytes(), [], 0
    while offset + 8 <= len(data):
        size = int.from_bytes(data[offset : offset + 4], "big")
        kind = data[offset + 4 : offset + 8].decode("latin-1")
        boxes.append(kind)
        if size == 1 and offset + 16 <= len(data):
            size = int.from_bytes(data[offset + 8 : offset + 16], "big")
        if size < 8 or offset + size > len(data):
            break
        offset += size
    return boxes


# ---------------------------------------------------------------- --time


def time_clips(directory: Path) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    encoder = find_encoder()
    describe_encoder(encoder)
    failures, total = 0, 0.0
    for demo in BUILDERS:
        replay = committed(demo)
        for theme in THEMES:
            path = directory / f"{demo}.{theme}.mp4"
            started = time.perf_counter()
            drawings = render_clip(replay, theme, encoder, path)
            drawn = time.perf_counter() - started
            started = time.perf_counter()
            decoded = decode(encoder, path)
            decoding = time.perf_counter() - started
            total += drawn + decoding
            frames = decoded.frames()
            ok = (
                decoded.status == 0
                and frames == EXPECTED_FRAMES[demo]
                and decoded.progress_line == "progress=end"
            )
            failures += not ok
            print(
                f"{demo} {theme}: drawings {drawings}, draw+encode {drawn:.2f} s, "
                f"decode {decoding:.2f} s, {path.stat().st_size} bytes, "
                f"decoded frames {frames}, {decoded.progress_line}, "
                f"boxes {box_order(path)}{'' if ok else '  FAILED'}"
            )
            print(f"    {decoded.video_line}")
    print(f"total for the twelve clips: {total:.1f} s")
    return 1 if failures else 0


def render_clip(replay: Replay, theme: str, encoder: str, path: Path) -> int:
    demo = str(replay.meta["demo"])
    clip_shots = shots(replay)
    with style(theme) as colors:
        clip = BUILDERS[demo](replay, colors)
        try:

            def frames() -> Iterator[tuple[bytes, int]]:
                for shot in clip_shots:
                    clip.show(shot.row, shot.azimuth)
                    yield frame_rgba(clip.fig), shot.repeat

            size = (1500, 2 * HEIGHTS[demo])
            written = encode_frames(encoder, frames(), path, size)
        finally:
            plt.close(clip.fig)
    if written != EXPECTED_FRAMES[demo]:
        print(f"  {demo} {theme}: wrote {written} frames")
    return len(clip_shots)


# ---------------------------------------------------------------- --decode


def solid_frames(size: tuple[int, int]) -> list[tuple[bytes, int]]:
    width, height = size
    colours = [(200, 40, 40), (40, 200, 40), (40, 40, 200)]
    return [
        (bytes([*colour, 255]) * (width * height), repeat)
        for colour, repeat in zip(colours, (2, 1, 4), strict=True)
    ]


def transcode(encoder: str, source: Path, target: Path, *options: str) -> None:
    command = [
        encoder,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source),
        *options,
        str(target),
    ]
    subprocess.run(command, check=True)


def decode_probe(directory: Path) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    encoder = find_encoder()
    describe_encoder(encoder)
    small = directory / "small.mp4"
    encode_frames(encoder, iter(solid_frames((64, 48))), small, (64, 48))
    cases: list[tuple[str, Path, tuple | None]] = [
        ("small clip", small, ("h264", "yuv420p", 64, 48, 20.0))
    ]
    mpeg4 = directory / "mpeg4.mp4"
    transcode(encoder, small, mpeg4, "-c:v", "mpeg4")
    cases.append(("re-encoded mpeg4", mpeg4, ("mpeg4", "yuv420p", 64, 48, 20.0)))
    yuv444 = directory / "yuv444p.mp4"
    transcode(encoder, small, yuv444, "-c:v", "libx264", "-pix_fmt", "yuv444p")
    cases.append(("re-encoded yuv444p", yuv444, ("h264", "yuv444p", 64, 48, 20.0)))
    fps25 = directory / "fps25.mp4"
    encode_frames(encoder, iter(solid_frames((64, 48))), fps25, (64, 48), fps=25)
    cases.append(("made at 25 fps", fps25, ("h264", "yuv420p", 64, 48, 25.0)))
    short = directory / "64x40.mp4"
    encode_frames(encoder, iter(solid_frames((64, 40))), short, (64, 40))
    cases.append(("made at 64x40", short, ("h264", "yuv420p", 64, 40, 20.0)))
    copied = directory / "copied.mp4"
    transcode(encoder, small, copied, "-c", "copy", "-f", "mp4")
    cases.append(("copied, no faststart", copied, ("h264", "yuv420p", 64, 48, 20.0)))
    noise = directory / "noise.mp4"
    rng = np.random.default_rng(20261124)
    pixels = rng.integers(0, 256, size=(60, 48, 64, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    encode_frames(encoder, ((frame.tobytes(), 1) for frame in pixels), noise, (64, 48))
    cut = directory / "noise-half.mp4"
    whole = noise.read_bytes()
    cut.write_bytes(whole[: len(whole) // 2])
    cases.append(("noise cut to half", cut, None))
    text = directory / "text.mp4"
    text.write_text("this is not a video\n")
    cases.append(("text file", text, None))
    failures = 0
    for label, path, expected in cases:
        result = decode(encoder, path)
        if label == "noise cut to half":
            frames = result.frames()
            ok = result.status != 0 or (frames is not None and 1 <= frames <= 59)
        elif label == "text file":
            ok = result.status != 0
        else:
            ok = (
                result.status == 0
                and result.frames() == 7
                and result.progress_line == "progress=end"
                and result.properties() == expected
            )
        failures += not ok
        extra = f" ({len(whole)} bytes whole)" if path == cut else ""
        print(f"{label}{extra}: exit {result.status}{'' if ok else '  FAILED'}")
        print(f"    {result.video_line}")
        print(f"    {result.frame_line} {result.progress_line}")
        print(f"    boxes {box_order(path)}, {path.stat().st_size} bytes")
    return 1 if failures else 0


# ---------------------------------------------------------------- --quarto

REPLAY_CSS = "figure.replay video { max-width: 100%; height: auto; }\n"
BLOCK = """```{{=html}}
<figure class="replay">
<video class="light-content" aria-label="Animation: the 2D search" \
aria-describedby="demo-2d-text" src="{up}media/demo_2d.light.mp4" \
poster="{up}media/demo_2d.light.png" width="750" height="450" controls \
playsinline preload="none"></video>
<video class="dark-content" aria-label="Animation: the 2D search" \
aria-describedby="demo-2d-text" src="{up}media/demo_2d.dark.mp4" \
poster="{up}media/demo_2d.dark.png" width="750" height="450" controls \
playsinline preload="none"></video>
<figcaption>The search in the plane, one observation at a time.</figcaption>
</figure>
```

::: {{#demo-2d-text}}
Before any reading the dots cover the whole search area.
:::
"""
DESCRIPTION = "Before any reading the dots cover the whole search area."
QUARTO_YML = """project:
  type: website
  output-dir: _output
  pre-render: prerender.py
  resources:
    - "media/*.mp4"
    - "media/*.png"

website:
  title: "Clip probe"

format:
  html:
    theme:
      light: [custom.scss]
      dark: [custom-dark.scss]
    css: [styles.css, replay.css]
"""
PRERENDER = """import json
import os
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
from PIL import Image

with open({log!r}, "a") as log:
    record = {{
        "cwd": os.getcwd(),
        "prefix": sys.prefix,
        "version": sys.version,
        "render_all": os.environ.get("QUARTO_PROJECT_RENDER_ALL"),
    }}
    log.write(json.dumps(record) + "\\n")
media = Path("media")
media.mkdir(exist_ok=True)
for theme, colour in (("light", (249, 245, 215)), ("dark", (29, 32, 33))):
    frame = bytes([*colour, 255]) * (64 * 48)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
        "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", "64x48",
        "-r", "20", "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-f", "mp4", str(media / f"demo_2d.{{theme}}.mp4"),
    ]
    subprocess.run(command, input=frame * 20, check=True)
    Image.new("RGB", (64, 48), colour).save(media / f"demo_2d.{{theme}}.png")
"""
FAILING_PRERENDER = "import sys\n\nsys.exit(1)\n"


def write_project(project: Path, log: Path) -> None:
    for name in ("custom.scss", "custom-dark.scss", "styles.css"):
        shutil.copyfile(ROOT / "site" / name, project / name)
    (project / "replay.css").write_text(REPLAY_CSS)
    (project / "_quarto.yml").write_text(QUARTO_YML)
    (project / "prerender.py").write_text(PRERENDER.format(log=str(log)))
    (project / "index.qmd").write_text(
        '---\ntitle: "Probe"\n---\n\n' + BLOCK.format(up="")
    )
    (project / "story").mkdir()
    (project / "story" / "page.qmd").write_text(
        '---\ntitle: "Story page"\n---\n\n' + BLOCK.format(up="../")
    )


def quarto(project: Path, *args: str) -> int:
    result = subprocess.run(
        ["quarto", "render", *args], cwd=project, capture_output=True, text=True
    )
    tail = "\n".join((result.stdout + result.stderr).splitlines()[-4:])
    print(f"quarto render {' '.join(args)}: exit {result.returncode}\n{tail}")
    return result.returncode


def log_entries(log: Path) -> list[dict]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line]


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.texts: dict[int, list[str]] = {}
        self.open: list[tuple[str, int | None]] = []
        self.body_classes: list[str] = []
        self.stylesheets: list[str] = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "body":
            self.body_classes = (attributes.get("class") or "").split()
        if tag == "link" and "stylesheet" in (attributes.get("rel") or "").split():
            self.stylesheets.append(attributes.get("href") or "")
        if tag in {"link", "meta", "img", "br", "hr", "input", "source"}:
            return
        index = None
        if attributes.get("id") == "demo-2d-text":
            index = len(self.ids)
            self.ids.append(tag)
            self.texts[index] = []
        self.open.append((tag, index))

    def handle_endtag(self, tag):
        for position in range(len(self.open) - 1, -1, -1):
            if self.open[position][0] == tag:
                del self.open[position:]
                return

    def handle_data(self, data):
        for _, index in self.open:
            if index is not None:
                self.texts[index].append(data)


class EmbedParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.elements: list[tuple[str, dict]] = []

    def handle_starttag(self, tag, attrs):
        if tag in {"figure", "video", "figcaption"}:
            self.elements.append((tag, dict(attrs)))


def embed_attributes(text: str) -> list[tuple[str, dict]]:
    """Each figure, video and figcaption start tag with its attributes."""
    parser = EmbedParser()
    parser.feed(text)
    return parser.elements


def check_page(output: Path, page: Path, up: str, source_css: bytes) -> list[str]:
    html = page.read_text()
    lines = {line.strip() for line in html.splitlines()}
    block_lines = BLOCK.format(up=up).splitlines()[1:5]
    problems = []
    for line in block_lines:
        if line not in lines:
            problems.append(f"(c) line not found whole: {line[:60]}...")
    if any(p.startswith("(c)") for p in problems):
        for found in html.splitlines():
            if "<video" in found or "<figure" in found or "figcaption" in found:
                print(f"    rendered: {found.strip()}")
        source = embed_attributes(BLOCK.format(up=up))
        rendered = embed_attributes(html)
        for (tag, want), (_, got) in zip(source, rendered, strict=False):
            changed = {
                key: (want.get(key), got.get(key))
                for key in sorted(set(want) | set(got))
                if want.get(key) != got.get(key)
            }
            print(f"    <{tag}> attributes changed (source, rendered): {changed}")
    parser = PageParser()
    parser.feed(html)
    if len(parser.ids) != 1:
        problems.append(f"(d) {len(parser.ids)} elements with id demo-2d-text")
    elif " ".join("".join(parser.texts[0]).split()) != DESCRIPTION:
        problems.append(f"(d) description text {''.join(parser.texts[0])!r}")
    if "quarto-light" not in parser.body_classes:
        problems.append(f"(e) body classes {parser.body_classes}")
    target = (output / "replay.css").resolve()
    linked = [
        href
        for href in parser.stylesheets
        if (page.parent / href.split("?")[0]).resolve() == target
    ]
    if not linked:
        problems.append(
            f"(f) no stylesheet link reaches replay.css: {parser.stylesheets}"
        )
    print(f"  {page.relative_to(output)}: stylesheets {parser.stylesheets}")
    if not target.exists() or target.read_bytes() != source_css:
        problems.append("(f) _output/replay.css missing or altered")
    return problems


def theme_rules_present(output: Path) -> bool:
    dark = re.compile(r"body\.quarto-dark\s+\.light-content\s*\{\s*display:\s*none")
    light = re.compile(r"body\.quarto-light\s+\.dark-content\s*\{\s*display:\s*none")
    found_dark = found_light = False
    for css in (output / "site_libs").rglob("*.css"):
        text = css.read_text(errors="replace")
        found_dark = found_dark or bool(dark.search(text))
        found_light = found_light or bool(light.search(text))
    return found_dark and found_light


def quarto_probe(directory: Path) -> int:
    project = directory.resolve()
    if project.is_relative_to(ROOT):
        raise SystemExit("--quarto needs a directory outside the repository")
    project.mkdir(parents=True, exist_ok=True)
    if any(project.iterdir()):
        raise SystemExit("--quarto needs an empty directory")
    version = subprocess.run(["quarto", "--version"], capture_output=True, text=True)
    print(f"quarto version: {version.stdout.strip()}")
    print(f"probe interpreter prefix (log only): {sys.prefix}")
    log = project.parent / f"{project.name}-prerender.log"
    log.unlink(missing_ok=True)
    write_project(project, log)
    output = project / "_output"
    results: dict[str, bool] = {}
    notes: list[str] = []

    first = quarto(project)
    after_first = log_entries(log)
    media = sorted(p.name for p in (output / "media").glob("*"))
    print(f"  _output/media after the clean render: {media}")
    results["b"] = first == 0 and media == sorted(
        f"demo_2d.{t}.{s}" for t in THEMES for s in ("mp4", "png")
    )
    page_problems = []
    css = (project / "replay.css").read_bytes()
    for page, up in ((output / "index.html", ""), (output / "story/page.html", "../")):
        page_problems += check_page(output, page, up, css)
    for letter in "cdf":
        results[letter] = first == 0 and not any(
            p.startswith(f"({letter})") for p in page_problems
        )
    results["e"] = (
        first == 0
        and not any(p.startswith("(e)") for p in page_problems)
        and theme_rules_present(output)
    )
    for problem in page_problems:
        print(f"  {problem}")

    (output / "media" / "output-only.mp4").write_bytes(b"sentinel")
    (project / "media" / "source-only.mp4").write_bytes(b"sentinel")
    second = quarto(project)
    after_second = log_entries(log)
    output_only = (output / "media" / "output-only.mp4").exists()
    source_only = (output / "media" / "source-only.mp4").exists()
    results["h"] = second == 0 and not output_only
    notes.append(
        f"after the second full render: output-only.mp4 "
        f"{'still there' if output_only else 'gone'}, source-only.mp4 "
        f"{'copied to _output/media' if source_only else 'not copied'}"
    )

    single = quarto(project, "story/page.qmd")
    after_single = log_entries(log)

    (project / "prerender.py").write_text(FAILING_PRERENDER)
    failing = quarto(project)
    results["g"] = failing != 0
    notes.append(f"after the failing render, _output exists: {output.exists()}")

    full_entries = after_first + after_second[len(after_first) :]
    single_entries = after_single[len(after_second) :]
    print(
        f"  pre-render log entries: {len(after_first)} after the first render, "
        f"{len(after_second)} after the second, {len(after_single)} after the "
        "single-page render"
    )
    for entry in after_single:
        same_dir = Path(entry["cwd"]).resolve() == project
        same_prefix = Path(entry["prefix"]).resolve() == Path(sys.prefix).resolve()
        print(
            f"    cwd is the project: {same_dir}; same interpreter prefix: "
            f"{same_prefix}; QUARTO_PROJECT_RENDER_ALL={entry['render_all']}; "
            f"python {entry['version'].split()[0]}"
        )
    results["a"] = (
        single == 0
        and len(after_first) == 1
        and len(after_second) == 2
        and len(single_entries) == 1
        and all(
            Path(e["cwd"]).resolve() == project
            and Path(e["prefix"]).resolve() == Path(sys.prefix).resolve()
            for e in after_single
        )
        and all(e["render_all"] == "1" for e in full_entries)
        and all(e["render_all"] is None for e in single_entries)
    )
    for note in notes:
        print(f"  {note}")
    for letter in "abcdefgh":
        print(f"({letter}) {'holds' if results[letter] else 'FAILS'}")
    return 0 if all(results.values()) else 1


# ---------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sheets", type=Path, help="write the contact sheets here")
    group.add_argument("--time", type=Path, help="render and time the twelve clips")
    group.add_argument("--decode", type=Path, help="run the small-clip decode probe")
    group.add_argument("--quarto", type=Path, help="build a scratch Quarto project")
    args = parser.parse_args()
    plt.switch_backend("agg")
    print(f"working directory (log only): {os.getcwd()}")
    if args.sheets:
        write_sheets(args.sheets)
        return 0
    if args.time:
        return time_clips(args.time)
    if args.decode:
        return decode_probe(args.decode)
    return quarto_probe(args.quarto)


if __name__ == "__main__":
    sys.exit(main())
