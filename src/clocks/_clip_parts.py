"""Parts every clip is drawn from: the figure, status line, traces and marks.

A clip is a figure whose axes and limits are fixed once from the replay, and a
function that sets its artists to one row of the archive. These parts draw with
the rcParams in force: call them inside ``with style(theme) as colors:`` and
pass that ``colors``. Every colour is a field of the palette, a blend of its
``background`` and ``ink``, or fully transparent; nothing here calls the
forward model.
"""

from collections.abc import Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Annotation
from matplotlib.transforms import Bbox
from numpy.typing import NDArray

from clocks._figures import trace_header, trace_limits
from clocks._replay import Replay
from clocks._style import Palette

Rect = tuple[float, float, float, float]  # left, bottom, width, height

DOT_SIZE = 2.0
DOT_ALPHA = 0.5
BAND_ALPHA = 0.2
TRACE_TITLE_SIZE = 9.5
SMALL = 8.0
ELLIPSE_POINTS = 121
# Points from the true point, in the order tried (decision D51).
MASS_LABEL_OFFSETS = (
    (9, 7),
    (-9, 7),
    (9, -7),
    (-9, -7),
    (9, 0),
    (-9, 0),
    (0, 7),
    (0, -7),
)
MASS_LABEL_SIZE = SMALL + 0.5
MASS_LABEL_PAD = 2  # pixels round a label's box


class Clip:
    """A figure with fixed axes, and the function that sets its artists to one row."""

    def __init__(
        self, fig: Figure, update: Callable[[int, float | None], None]
    ) -> None:
        self.fig = fig
        self._update = update

    def show(self, row: int, azimuth: float | None = None) -> None:
        """Set every artist to archive row ``row`` (and the camera, where one turns)."""
        self._update(row, azimuth)


def new_figure(height: int) -> Figure:
    """A 7.5 inch wide figure, ``height`` CSS pixels tall, with no layout engine.

    The placeholder engine is set after creation: a figure made with
    ``layout="none"`` under the style regains constrained layout after its
    first save (measured, Q3).
    """
    fig = plt.figure(figsize=(7.5, height / 100))
    fig.set_layout_engine("none")
    return fig


def status_text(row: int) -> str:
    """The status line for ``row``: before any, or after k, observations."""
    if row == 0:
        return "Before any observation"
    if row == 1:
        return "After 1 observation"
    return f"After {row} observations"


def draw_status(fig: Figure, colors: Palette) -> Callable[[int], None]:
    """Add the status line at the top left; return the function that sets it."""
    text = fig.text(
        0.012, 0.985, "", ha="left", va="top", fontsize=11, color=colors.ink
    )
    text.set_gid("status")

    def update(row: int) -> None:
        text.set_text(status_text(row))

    return update


def standard_deviations(replay: Replay) -> NDArray[np.float64]:
    """Each row's marginal standard deviations, shape (T + 1, P)."""
    return np.sqrt(np.diagonal(replay.arrays["cov"], axis1=1, axis2=2))


def draw_traces(
    fig: Figure, replay: Replay, colors: Palette, rects: Mapping[str, Rect]
) -> Callable[[int], None]:
    """Add one trace axes per ``"trace-<name>"`` key; return the row setter.

    Each trace holds the running mean, a one-standard-deviation band and the
    truth as a solid line, with x from 0 to T and the y-limits of
    ``trace_limits``. The lowest traces carry the x label.
    """
    names = [str(name) for name in replay.meta["param_names"]]
    bottom = min((rect[1] for rect in rects.values()), default=0.0)
    updates = []
    for key, rect in rects.items():
        name = key.removeprefix("trace-")
        if not key.startswith("trace-") or name not in names:
            raise ValueError(f"{key!r} names no parameter of {names}")
        ax = fig.add_axes(rect)
        ax.set_gid(key)
        updates.append(_draw_trace(ax, replay, colors, names.index(name)))
        ax.tick_params(labelsize=SMALL)
        if rect[1] == bottom:
            ax.set_xlabel("observations", fontsize=SMALL + 1)
        else:
            ax.tick_params(labelbottom=False)

    def update(row: int) -> None:
        for function in updates:
            function(row)

    return update


def _draw_trace(
    ax: Axes, replay: Replay, colors: Palette, j: int
) -> Callable[[int], None]:
    name = str(replay.meta["param_names"][j])
    prior = replay.meta["prior_ranges"][j]
    mean = replay.arrays["mean"][:, j]
    sd = standard_deviations(replay)[:, j]
    truth = float(replay.arrays["true_params"][j])
    last = len(mean) - 1
    ax.set_xlim(0, last)
    ax.set_ylim(*trace_limits(mean, sd, truth, prior))
    (true_line,) = ax.plot(
        [0, last], [truth, truth], color=colors.truth, linestyle="-", linewidth=1.3
    )
    true_line.set_gid("trace-truth")
    (line,) = ax.plot([], [], color=colors.belief, linewidth=1.5)
    line.set_gid("trace-mean")
    band: list[PolyCollection] = []

    def update(row: int) -> None:
        steps = np.arange(row + 1)
        line.set_data(steps, mean[: row + 1])
        # Matplotlib 3.9 cannot reset a band's data, so it is drawn afresh.
        if band:
            band.pop().remove()
        drawn = ax.fill_between(
            steps,
            mean[: row + 1] - sd[: row + 1],
            mean[: row + 1] + sd[: row + 1],
            facecolor=colors.belief,
            alpha=BAND_ALPHA,
            linewidth=0,
        )
        drawn.set_gid("trace-band")
        # At row 0 the band is one point wide, yet still a path (Q18).
        drawn.set_visible(row > 0)
        band.append(drawn)
        header = trace_header(name, truth, mean[row], sd[row], prior)
        ax.set_title(header, fontsize=TRACE_TITLE_SIZE)

    return update


def draw_dots(ax: Axes, color: str, *, gid: str = "dots", marker: str = "o") -> Line2D:
    """A marker-only line of faint dots of one size, with no data yet."""
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
    """The closed two-standard-deviation contour of a 2 by 2 covariance block.

    121 points, the last equal to the first. Negative eigenvalues from rounding
    are taken as 0, so a block of no spread gives the mean 121 times.
    """
    eigenvalues, vectors = np.linalg.eigh(np.asarray(cov, dtype=np.float64))
    t = np.linspace(0.0, 2.0 * np.pi, ELLIPSE_POINTS)
    t[-1] = 0.0
    circle = np.stack([np.cos(t), np.sin(t)])
    axes = vectors * np.sqrt(np.clip(eigenvalues, 0.0, None))
    centre = np.asarray(mean, dtype=np.float64)[:, None]
    return (centre + 2.0 * axes @ circle).T


def draw_ellipse(ax: Axes, color: str, *, gid: str = "ellipse") -> Line2D:
    """A line for the covariance contour, with no data yet."""
    (line,) = ax.plot([], [], color=color, linewidth=1.2, zorder=4)
    line.set_gid(gid)
    return line


def draw_truth(
    ax: Axes, x: float, y: float, color: str, *, gid: str = "truth"
) -> Line2D:
    """An open ring at the true (x, y)."""
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


def zoom_window(
    truth: NDArray, mean: NDArray, sd: NDArray, prior_ranges: Sequence[Sequence[float]]
) -> NDArray[np.float64]:
    """Per axis, a window centred on the truth: ``(truth - h, truth + h)``.

    ``h`` is 1.25 times the offset of the mean from the truth plus two
    standard deviations, or 1% of the axis's prior range where that is 0.
    """
    window = []
    for t, m, s, (low, high) in zip(truth, mean, sd, prior_ranges, strict=True):
        half = 1.25 * (abs(m - t) + 2.0 * s)
        if half <= 0:
            half = 0.01 * (high - low)
        window.append((t - half, t + half))
    return np.array(window, dtype=np.float64)


def zoom_label(x_factor: float, y_factor: float, y_name: str) -> str:
    """One magnification where both round alike, else one per axis (D18)."""
    if round(x_factor) == round(y_factor):
        return f"zoom ×{round(x_factor)}"
    return f"zoom ×{round(x_factor)} (x), ×{round(y_factor)} ({y_name})"


def draw_inset(
    ax: Axes, rect: Rect, window: NDArray, colors: Palette, *, gid: str = "inset"
) -> Axes:
    """A child axes of ``ax`` at ``rect`` (axes fractions) showing ``window``."""
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
    return inset


def running_slowing(observations: NDArray) -> NDArray[np.float64]:
    """Row k: each clock's mean of 1 - rate over the first k readings; row 0 NaN."""
    slowing = 1.0 - np.asarray(observations, dtype=np.float64)
    counts = np.arange(1, len(slowing) + 1)[:, None]
    running = np.cumsum(slowing, axis=0) / counts
    return np.vstack([np.full((1, slowing.shape[1]), np.nan), running])


def label_masses(
    ax: Axes, points: Sequence[tuple[float, float]], colors: Palette
) -> list[Annotation]:
    """Label point n ``mass n``: fewest exclusions, then marks covered, then order.

    Call once, after the final row is shown; the labels then stay put. Each
    candidate of ``MASS_LABEL_OFFSETS`` is ranked by its exclusions (one for
    leaving the panel, one per inset and per earlier label its padded box
    touches), then by the marks of the axes' own lines inside that box, then
    by its place in the list. Where every candidate has an exclusion, the
    fewest still wins (decision D51).
    """
    renderer = ax.figure.canvas.get_renderer()
    marks = _display_marks(ax)
    frame = ax.get_window_extent(renderer)
    blocked = [inset.get_window_extent(renderer) for inset in ax.child_axes]
    labels: list[Annotation] = []
    for n, point in enumerate(points, start=1):
        candidates = []
        for place, offset in enumerate(MASS_LABEL_OFFSETS):
            label = _mass_label(ax, n, point, offset, colors)
            box = label.get_window_extent(renderer).padded(MASS_LABEL_PAD)
            leaves = not (
                frame.contains(box.x0, box.y0) and frame.contains(box.x1, box.y1)
            )
            exclusions = int(leaves) + sum(box.overlaps(other) for other in blocked)
            candidates.append(
                ((exclusions, _marks_inside(box, marks), place), label, box)
            )
        _, chosen, box = min(candidates, key=lambda candidate: candidate[0])
        for _, label, _ in candidates:
            if label is not chosen:
                label.remove()
        blocked.append(box)
        labels.append(chosen)
    return labels


def _mass_label(
    ax: Axes,
    n: int,
    point: tuple[float, float],
    offset: tuple[int, int],
    colors: Palette,
) -> Annotation:
    """``mass n`` at ``offset`` points from ``point``, aligned away from it."""
    dx, dy = offset
    label = ax.annotate(
        f"mass {n}",
        point,
        xytext=offset,
        textcoords="offset points",
        ha="left" if dx > 0 else "right" if dx < 0 else "center",
        va="bottom" if dy > 0 else "top" if dy < 0 else "center",
        fontsize=MASS_LABEL_SIZE,
        color=colors.truth,
        zorder=6,
        path_effects=[
            patheffects.withStroke(linewidth=3, foreground=colors.background)
        ],
    )
    label.set_gid(f"mass-label-{n}")
    return label


def _display_marks(ax: Axes) -> NDArray[np.float64]:
    """Every data point of the axes' own lines, in display coordinates."""
    data = [
        np.column_stack(line.get_data()).astype(np.float64)
        for line in ax.get_lines()
        if len(line.get_xdata())
    ]
    if not data:
        return np.empty((0, 2))
    return ax.transData.transform(np.vstack(data))


def _marks_inside(box: Bbox, marks: NDArray[np.float64]) -> int:
    inside = (
        (marks[:, 0] >= box.x0)
        & (marks[:, 0] <= box.x1)
        & (marks[:, 1] >= box.y0)
        & (marks[:, 1] <= box.y1)
    )
    return int(inside.sum())
