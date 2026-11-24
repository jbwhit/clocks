"""Static figures drawn from a replay, and the trace helpers clips reuse.

A figure function draws with the rcParams in force: call it inside
``with style(theme) as colors:`` and pass that ``colors``. It never shows,
saves or closes the figure it returns. The density parameterisation
A·exp(-(x-μ)²/2σ²) is evaluated here, never through the forward model.
"""

import math
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from clocks._replay import Replay
from clocks._style import Palette

PROFILE_POINTS = 321
_TRACE_PAD = 0.04


def trace_limits(
    mean: NDArray[np.float64],
    sd: NDArray[np.float64],
    truth: float,
    prior_range: Sequence[float],
) -> tuple[float, float]:
    """Y-limits of one parameter's trace: rows after the prior and the truth.

    ``mean`` and ``sd`` hold rows 0 to T; row 0 (the prior) is left out. The
    interval is padded by 4% of its width, or of the prior range when it has no
    width, and clipped to the prior range.
    """
    low = min(truth, float(np.min(mean[1:] - sd[1:])))
    high = max(truth, float(np.max(mean[1:] + sd[1:])))
    width = high - low if high > low else prior_range[1] - prior_range[0]
    pad = _TRACE_PAD * width
    return max(low - pad, prior_range[0]), min(high + pad, prior_range[1])


def trace_header(
    name: str,
    truth: float,
    mean: float,
    sd: float,
    prior_range: Sequence[float],
) -> str:
    """The header of a trace: the name, the truth and the final estimate."""
    digits = max(2, 2 - math.floor(math.log10(prior_range[1] - prior_range[0])))
    return f"{name}   true {truth:.{digits}f}   {mean:.{digits}f} ± {sd:.{digits}f}"


def density_figure(replay: Replay, colors: Palette) -> Figure:
    """Draw the final density profile above one trace per parameter."""
    demo = replay.meta["demo"]
    if demo != "demo_density":
        raise ValueError(f"density_figure needs a demo_density replay, got {demo!r}")
    fig = plt.figure(figsize=(7.5, 5.4))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.7, 1.0], hspace=0.08)
    profile_ax = fig.add_subplot(grid[0, :])
    profile_ax.set_gid("profile")
    _draw_profile(profile_ax, replay, colors)
    names = list(replay.meta["param_names"])
    for j, name in enumerate(names):
        ax = fig.add_subplot(grid[1, j])
        ax.set_gid(f"trace-{name}")
        _draw_trace(ax, replay, colors, j)
    return fig


def _gaussian_profiles(
    dots: NDArray[np.float64], x: NDArray[np.float64]
) -> NDArray[np.float64]:
    return np.array([a * np.exp(-((x - mu) ** 2) / (2 * s**2)) for mu, s, a in dots])


def _draw_profile(ax: Axes, replay: Replay, colors: Palette) -> None:
    arrays = replay.arrays
    x_low, x_high = replay.meta["prior_ranges"][0]
    x = np.linspace(x_low, x_high, PROFILE_POINTS)
    mu, sigma, amplitude = arrays["true_params"]
    truth = amplitude * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    profiles = _gaussian_profiles(arrays["cloud"][-1].astype(np.float64), x)
    lower, upper = np.percentile(profiles, [16, 84], axis=0)
    median = np.median(profiles, axis=0)

    ax.fill_between(
        x, lower, upper, color=colors.belief, alpha=0.2, linewidth=0,
        gid="profile-band",
    )  # fmt: skip
    ax.plot(x, truth, color=colors.truth, linewidth=1.3, gid="true-profile")
    ax.plot(x, median, color=colors.belief, gid="median-profile")
    clock_x = arrays["clock_positions"][:, 0]
    ax.plot(
        clock_x, np.zeros_like(clock_x), color=colors.ink, marker="s",
        markersize=6, linestyle="none", clip_on=False, gid="clocks",
    )  # fmt: skip
    top = 1.08 * max(truth.max(), upper.max())
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(0.0, top)
    ax.set_ylabel("clock-rate offset")
    _label_profile(ax, colors, mu, top, float(clock_x[0]))


def _label_profile(
    ax: Axes, colors: Palette, centre: float, top: float, first_clock: float
) -> None:
    """Direct labels: the true profile left of its peak, the estimate right."""
    ax.text(
        centre - 2.2, 0.93 * top, "true profile", color=colors.truth,
        ha="right", va="center",
    )  # fmt: skip
    ax.text(
        ax.get_xlim()[1] - 0.15, 0.93 * top,
        "pointwise median and 68% band", color=colors.belief, ha="right", va="center",
    )  # fmt: skip
    ax.text(
        first_clock, 0.05 * top, "clocks", color=colors.ink, ha="center",
        va="bottom",
    )  # fmt: skip


def _draw_trace(ax: Axes, replay: Replay, colors: Palette, j: int) -> None:
    arrays = replay.arrays
    mean = arrays["mean"][:, j]
    sd = np.sqrt(arrays["cov"][:, j, j])
    steps = np.arange(len(mean))
    last = len(mean) - 1
    truth = float(arrays["true_params"][j])
    prior = replay.meta["prior_ranges"][j]
    ax.fill_between(
        steps, mean - sd, mean + sd, color=colors.belief, alpha=0.2,
        linewidth=0, gid="trace-band",
    )  # fmt: skip
    ax.plot(
        [0, last], [truth, truth], color=colors.truth, linewidth=1.3,
        gid="trace-truth",
    )  # fmt: skip
    ax.plot(steps, mean, color=colors.belief, gid="trace-mean")
    ax.set_xlim(0, last)
    ax.set_ylim(*trace_limits(mean, sd, truth, prior))
    ax.set_xlabel("observations")
    ax.set_title(
        trace_header(
            str(replay.meta["param_names"][j]),
            truth,
            float(mean[-1]),
            float(sd[-1]),
            prior,
        ),
        fontsize=9.5,
    )
