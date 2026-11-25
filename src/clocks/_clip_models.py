"""The model-comparison clip: each candidate count's posterior probability.

``build_models`` draws ``demo_model_comparison``: one panel of posterior
probability against observations, a line per candidate in ``model_ks`` growing
from the model prior at row 0, coloured by its position in ``model_ks``, with
a dot of its colour at its head. Each line is named by a label just right of
its head, which follows the head from row to row; the labels keep
``LABEL_GAP`` apart without changing their order from the top (decision D50).
Call it inside ``with style(theme) as colors:``.
"""

from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Annotation
from matplotlib.ticker import MaxNLocator

from clocks._clip_parts import Clip, Rect, draw_status, new_figure
from clocks._replay import Replay
from clocks._style import Palette

# Figure fractions, from the probe (Task 1).
LAYOUT_MODELS: dict[str, Rect] = {"posterior": (0.085, 0.14, 0.89, 0.74)}
HEIGHT_MODELS = 420
# The least width, in observations, of the room right of the last observation
# for the final labels; a longer run gets as much more as its labels need.
GUTTER = 22.0
HEAD_SIZE = 5.0  # points: the dot at each line's head
LABEL_GAP = 12.0  # points between adjacent label centres, at least
LABEL_OFFSET = 6.0  # points right of a line's head
LABEL_SIZE = 10.0
LABEL_SPARE = 2.0  # points kept clear of the panel's right edge
MAX_CANDIDATES = 3  # one per colour of ``Palette.ordered`` (decision D16)
Y_LIMITS = (-0.03, 1.03)


def build_models(replay: Replay, colors: Palette) -> Clip:
    """The ``demo_model_comparison`` clip: one line and one label per candidate.

    Raises ``ValueError`` for more than three candidates.
    """
    arrays = replay.arrays
    ks = [int(k) for k in arrays["model_ks"]]
    if len(ks) > MAX_CANDIDATES:
        raise ValueError(
            f"the model clip draws at most three candidates, one per ordered "
            f"colour; model_ks is {ks}"
        )
    posterior = arrays["model_posterior"]
    last = len(posterior) - 1
    fig = new_figure(HEIGHT_MODELS)
    ax = _posterior_axes(fig, last)
    lines, labels = [], []
    for i, k in enumerate(ks):
        colour = colors.ordered[i]
        # The head is marked, so at row 0, where each line is one point at the
        # prior on the y axis, its label sits beside something drawn: a whole
        # dot, unclipped and over the spine (2.5). The data never leave the
        # limits. Earlier candidates are drawn on top, so the dot the tied
        # heads share at row 0 has the colour of the label beside it, the top.
        (line,) = ax.plot(
            [],
            [],
            color=colour,
            linewidth=1.8,
            marker="o",
            markersize=HEAD_SIZE,
            clip_on=False,
            zorder=3.0 - 0.1 * i,
        )
        line.set_markevery([-1])
        line.set_gid(f"model-{k}")
        lines.append(line)
        labels.append(_model_label(ax, k, replay.meta["true_k"], colour))
    ax.set_xlim(0, last + gutter_width(ax, labels, last))
    status = draw_status(fig, colors)

    def update(row: int, azimuth: float | None) -> None:
        heads = posterior[row]
        for i, (line, label) in enumerate(zip(lines, labels, strict=True)):
            line.set_data(np.arange(row + 1), posterior[: row + 1, i])
            label.xy = (row, heads[i])
        for label, shift in zip(labels, label_shifts(ax, row, heads), strict=True):
            label.xyann = (LABEL_OFFSET, shift)
        status(row)

    update(last, None)
    return Clip(fig, update)


def model_label_text(k: int, true_k: object) -> str:
    """``K = k``, marked as the true model when ``k`` is ``true_k``."""
    return f"K = {k} (true model)" if k == true_k else f"K = {k}"


def gutter_width(ax: Axes, labels: Sequence[Annotation], last: int) -> float:
    """Observations of room right of ``last`` for the widest label to fit.

    A label ``w`` points wide (offset and spare included) at observation T fits
    a panel ``W`` points wide whose x-axis ends at T + g when g is at least
    T w / (W - w); the room is never less than ``GUTTER``.
    """
    fig = ax.figure
    scale = 72 / fig.dpi
    renderer = fig.canvas.get_renderer()
    widest = max(label.get_window_extent(renderer).width for label in labels)
    w = widest * scale + LABEL_OFFSET + LABEL_SPARE
    panel = ax.bbox.width * scale
    return max(GUTTER, last * w / (panel - w))


def label_shifts(ax: Axes, row: int, heads: Sequence[float]) -> list[float]:
    """Each label's vertical offset in points from its head at ``row`` (D50).

    Taken from the top (ties in ``model_ks`` order, the earlier above), each
    label sits at its head or ``LABEL_GAP`` below the label above, whichever
    is lower. Then from the bottom up the lowest rises, if it must, to
    ``LABEL_GAP / 2`` above the panel's bottom edge, and each label above it
    only as far as keeping ``LABEL_GAP`` from the one below requires.
    """
    scale = 72 / ax.figure.dpi
    ys = [float(ax.transData.transform((row, head))[1]) * scale for head in heads]
    floor = ax.bbox.y0 * scale + LABEL_GAP / 2
    order = sorted(range(len(ys)), key=lambda i: -ys[i])  # stable: ties keep order
    placed = list(ys)
    for upper, lower in zip(order, order[1:]):
        placed[lower] = min(ys[lower], placed[upper] - LABEL_GAP)
    placed[order[-1]] = max(placed[order[-1]], floor)
    for lower, upper in zip(order[::-1], order[-2::-1]):
        placed[upper] = max(placed[upper], placed[lower] + LABEL_GAP)
    return [q - y for q, y in zip(placed, ys, strict=True)]


def _posterior_axes(fig: Figure, last: int) -> Axes:
    """The panel: y from just below 0 to just above 1; x ticks stop at T."""
    ax = fig.add_axes(LAYOUT_MODELS["posterior"])
    ax.set_gid("posterior")
    ax.set_ylim(*Y_LIMITS)
    ticks = MaxNLocator(nbins=5, integer=True).tick_values(0, last)
    ax.set_xticks([tick for tick in ticks if 0 <= tick <= last])
    ax.spines["bottom"].set_bounds(0, last)
    ax.set_xlabel("observations")
    ax.set_ylabel("posterior probability")
    return ax


def _model_label(ax: Axes, k: int, true_k: object, colour: str) -> Annotation:
    """The label of candidate ``k``, placed at its head by the row setter."""
    label = ax.annotate(
        model_label_text(k, true_k),
        (0, 0),
        xytext=(LABEL_OFFSET, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=LABEL_SIZE,
        color=colour,
        annotation_clip=False,
    )
    label.set_gid(f"label-{k}")
    return label
