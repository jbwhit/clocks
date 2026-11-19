"""Task 7, step 2: smoke-draw every panel from the candidate archives alone.

Imports from clocks: only ``load_replay`` and names from ``clocks.viz``.
Every drawn value comes from an archive array or its ``meta``; layout constants
(offsets, marker sizes, figure sizes) are the only numbers typed here.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Ellipse, Rectangle

from clocks._replay import load_replay
from clocks.viz import style

HERE = Path(__file__).parent
CANDIDATES = HERE / "candidates"
SHEETS = HERE / "sheets"
DPI = 100
WIDTH = 15.0  # inches: 1500 px at DPI


def frames(replay) -> list[int]:
    return [0, 1, 8, replay.meta["n_observations"]]


def status(k: int) -> str:
    return "Before any observation" if k == 0 else f"After {k} observations"


def sd_rows(replay) -> np.ndarray:
    return np.sqrt(np.diagonal(replay.arrays["cov"], axis1=1, axis2=2))


def running_mean(observations: np.ndarray, k: int) -> np.ndarray | None:
    return None if k == 0 else observations[:k].mean(axis=0)


def column(replay, name: str) -> int:
    return replay.meta["param_names"].index(name)


# --- shared artists ------------------------------------------------------------


def draw_ellipse(ax, centre, cov2, color, label=True) -> None:
    values, vectors = np.linalg.eigh(cov2)
    angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
    width, height = 2 * 2 * np.sqrt(np.maximum(values[::-1], 0.0))
    ax.add_patch(
        Ellipse(centre, width, height, angle=angle, fill=False, color=color, lw=1.4)
    )
    if label:
        ax.annotate(
            "covariance, 2 s.d.",
            (centre[0], centre[1]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color=color,
        )


def draw_trace(ax, replay, j: int, k: int, colors) -> None:
    arrays, meta = replay.arrays, replay.meta
    mean, sd = arrays["mean"][:, j], sd_rows(replay)[:, j]
    truth = arrays["true_params"][j]
    steps = np.arange(meta["n_observations"] + 1)
    shown = steps[: k + 1]
    ax.fill_between(
        shown,
        (mean - sd)[: k + 1],
        (mean + sd)[: k + 1],
        color=colors.belief,
        alpha=0.25,
    )
    ax.plot(shown, mean[: k + 1], color=colors.belief, lw=1.4)
    ax.errorbar(
        [k],
        [mean[k]],
        yerr=[sd[k]],
        fmt="o",
        ms=3,
        color=colors.belief,
        capsize=2,
        clip_on=True,
    )
    ax.axhline(truth, color=colors.truth, lw=1.2)
    ax.set_xlim(0, meta["n_observations"])
    ax.set_ylim(*trace_limits(replay, j))
    name = meta["param_names"][j]
    ax.set_title(
        f"{name}   true {truth:.4g}   {mean[k]:.4g} ± {sd[k]:.2g}",
        fontsize=9,
        loc="left",
    )


def trace_limits(replay, j: int) -> tuple[float, float]:
    """Smallest interval holding the truth and mean ± sd for k >= 1, +8%, clipped."""
    mean, sd = replay.arrays["mean"][1:, j], sd_rows(replay)[1:, j]
    truth = replay.arrays["true_params"][j]
    low = min(truth, float((mean - sd).min()))
    high = max(truth, float((mean + sd).max()))
    pad = 0.08 * (high - low)
    prior_low, prior_high = replay.meta["prior_ranges"][j]
    return max(low - pad, prior_low), min(high + pad, prior_high)


def inset_window(replay, cols: tuple[int, int]) -> list[tuple[float, float]]:
    """Centred on the truth; half-width 1.25 x (final offset + 2 final sd)."""
    truth = replay.arrays["true_params"]
    final, sd = replay.arrays["mean"][-1], sd_rows(replay)[-1]
    window = []
    for c in cols:
        half = 1.25 * (abs(final[c] - truth[c]) + 2 * sd[c])
        window.append((truth[c] - half, truth[c] + half))
    return window


def draw_cloud_with_inset(
    ax,
    replay,
    k,
    cols,
    colors,
    *,
    color,
    marker,
    bounds,
    inset_rect,
    label=None,
    inset=True,
) -> None:
    arrays = replay.arrays
    pts = arrays["cloud"][k][:, list(cols)]
    truth = arrays["true_params"][list(cols)]
    mean = arrays["mean"][k][list(cols)]
    cov = arrays["cov"][k][np.ix_(cols, cols)]
    ax.scatter(pts[:, 0], pts[:, 1], s=4, color=color, alpha=0.5, marker=marker, lw=0)
    ax.plot(*truth, marker="+", ms=12, mew=2, color=colors.truth)
    if label:
        ax.annotate(
            label,
            truth,
            xytext=(6, -10),
            textcoords="offset points",
            fontsize=8,
            color=colors.truth,
        )
    draw_ellipse(ax, mean, cov, color)
    if not inset:
        return
    window = inset_window(replay, cols)
    sub = ax.inset_axes(inset_rect)
    sub.scatter(pts[:, 0], pts[:, 1], s=4, color=color, alpha=0.5, marker=marker, lw=0)
    sub.plot(*truth, marker="+", ms=10, mew=2, color=colors.truth)
    draw_ellipse(sub, mean, cov, color, label=False)
    sub.set_xlim(*window[0])
    sub.set_ylim(*window[1])
    sub.tick_params(labelsize=6)
    magnification = (bounds[0][1] - bounds[0][0]) / (window[0][1] - window[0][0])
    sub.set_title(f"×{magnification:.0f}", fontsize=7)


def draw_clock_strip(ax, replay, colors) -> None:
    xs = replay.arrays["clock_positions"][:, 0]
    ax.scatter(xs, np.zeros_like(xs), marker="s", s=40, color=colors.ink)
    ax.set_xlim(*replay.meta["prior_ranges"][0])
    ax.set_yticks([])
    ax.set_title("clocks", fontsize=8, loc="left")


def draw_slowing(ax, replay, k, colors, slowing_limits) -> None:
    """True tick, predicted bar, running mean of readings: all as 1 - rate."""
    arrays, meta = replay.arrays, replay.meta
    xs = arrays["clock_positions"][:, 0]
    offset = 0.03 * (meta["prior_ranges"][0][1] - meta["prior_ranges"][0][0])
    true_slowing = 1 - arrays["true_rates"]
    ax.scatter(
        xs - offset,
        true_slowing,
        marker="_",
        s=160,
        lw=2.2,
        color=colors.truth,
        label="true",
    )
    ax.errorbar(
        xs,
        1 - arrays["predicted_mean"][k],
        yerr=arrays["predicted_std"][k],
        fmt="o",
        ms=3,
        color=colors.belief,
        capsize=2,
        label="predicted",
    )
    observed = running_mean(arrays["observations"], k)
    if observed is not None:
        ax.errorbar(
            xs + offset,
            1 - observed,
            yerr=meta["noise_std"] / np.sqrt(k),
            fmt="s",
            ms=3,
            color=colors.ink,
            capsize=2,
            label="readings",
        )
    ax.set_xlim(*meta["prior_ranges"][0])
    ax.set_ylim(*slowing_limits)
    ax.set_ylabel("slowing (1 − rate)", fontsize=8)


def slowing_limits(replay) -> tuple[float, float]:
    arrays = replay.arrays
    pm, ps = 1 - arrays["predicted_mean"], arrays["predicted_std"]
    lo = min(float((pm - ps).min()), float((1 - arrays["true_rates"]).min()))
    hi = max(float((pm + ps).max()), float((1 - arrays["true_rates"]).max()))
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def clock_fill(replay, colors):
    """Return f(k) -> per-clock RGB; hollow (None) before any observation."""
    slowing = 1 - replay.arrays["observations"]
    cumulative = np.cumsum(slowing, axis=0) / np.arange(1, len(slowing) + 1)[:, None]
    top = float(cumulative.max())  # largest running mean in the whole run
    back, ink = np.array(to_rgb(colors.background)), np.array(to_rgb(colors.ink))

    def fill(k):
        if k == 0:
            return None
        share = np.clip(cumulative[k - 1] / top, 0, 1)[:, None]
        return back + share * (ink - back)

    return fill


def draw_clock_map(ax, replay, k, fill, colors) -> None:
    positions = replay.arrays["clock_positions"]
    faces = fill(k)
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        marker="s",
        s=60,
        facecolors="none" if faces is None else faces,
        edgecolors=colors.ink,
        lw=1.0,
        zorder=3,
    )
    ax.text(
        0.02,
        0.02,
        "stronger fill = slower clock",
        transform=ax.transAxes,
        fontsize=7,
        color=colors.muted,
    )


# --- per-archive sheets ----------------------------------------------------------


def new_sheet(rows: int, row_height: float, ks: list[int], title: str):
    fig = plt.figure(figsize=(WIDTH, rows * row_height), dpi=DPI)
    grid = fig.add_gridspec(rows, len(ks))
    fig.suptitle(title, fontsize=12)
    return fig, grid


def head(ax, k: int) -> None:
    ax.set_title(f"k = {k}: {status(k)}", fontsize=9, loc="left")


def sheet_1d(replay, colors, multi: bool):
    ks = frames(replay)
    names = replay.meta["param_names"]
    n_traces = len(names)
    fig = plt.figure(figsize=(WIDTH, 2.3 * (2.4 + n_traces)), dpi=DPI)
    fig.suptitle(replay.meta["demo"], fontsize=12)
    grid = fig.add_gridspec(
        3 + n_traces, len(ks), height_ratios=[1.6, 0.3, 1.1] + [1] * n_traces
    )
    bounds = [
        tuple(replay.meta["prior_ranges"][0]),
        tuple(replay.meta["prior_ranges"][len(names) // 2 if multi else 1]),
    ]
    limits = slowing_limits(replay)
    for c, k in enumerate(ks):
        ax = fig.add_subplot(grid[0, c])
        if multi:
            pairs = [
                (column(replay, "x₁"), column(replay, "M₁")),
                (column(replay, "x₂"), column(replay, "M₂")),
            ]
            styles = [
                (colors.belief, "o", "mass 1", [0.02, 0.58, 0.3, 0.38]),
                (colors.belief_alt, "^", "mass 2", [0.68, 0.58, 0.3, 0.38]),
            ]
            for cols, (color, marker, label, rect) in zip(pairs, styles):
                draw_cloud_with_inset(
                    ax,
                    replay,
                    k,
                    cols,
                    colors,
                    color=color,
                    marker=marker,
                    bounds=bounds,
                    inset_rect=rect,
                    label=label,
                )
        else:
            draw_cloud_with_inset(
                ax,
                replay,
                k,
                (0, 1),
                colors,
                color=colors.belief,
                marker="o",
                bounds=bounds,
                inset_rect=[0.04, 0.06, 0.34, 0.42],
            )
        ax.set_xlim(*bounds[0])
        ax.set_ylim(*bounds[1])
        head(ax, k)
        draw_clock_strip(fig.add_subplot(grid[1, c]), replay, colors)
        draw_slowing(fig.add_subplot(grid[2, c]), replay, k, colors, limits)
        for j in range(n_traces):
            draw_trace(fig.add_subplot(grid[3 + j, c]), replay, j, k, colors)
    return [fig]


def sheet_map(replay, colors, multi: bool):
    ks = frames(replay)
    names = replay.meta["param_names"]
    fig, grid = new_sheet(
        1 + (3 if multi else len(names)), 2.6, ks, replay.meta["demo"]
    )
    bounds = [
        tuple(replay.meta["prior_ranges"][0]),
        tuple(replay.meta["prior_ranges"][1]),
    ]
    fill = clock_fill(replay, colors)
    for c, k in enumerate(ks):
        ax = fig.add_subplot(grid[0, c])
        if multi:
            pairs = [
                (column(replay, "x₁"), column(replay, "y₁")),
                (column(replay, "x₂"), column(replay, "y₂")),
            ]
            styles = [
                (colors.belief, "o", "mass 1"),
                (colors.belief_alt, "^", "mass 2"),
            ]
            for cols, (color, marker, label) in zip(pairs, styles):
                draw_cloud_with_inset(
                    ax,
                    replay,
                    k,
                    cols,
                    colors,
                    color=color,
                    marker=marker,
                    bounds=bounds,
                    inset_rect=None,
                    label=label,
                    inset=False,
                )
        else:
            draw_cloud_with_inset(
                ax,
                replay,
                k,
                (0, 1),
                colors,
                color=colors.belief,
                marker="o",
                bounds=bounds,
                inset_rect=[0.64, 0.64, 0.34, 0.34],
            )
        draw_clock_map(ax, replay, k, fill, colors)
        ax.set_xlim(*bounds[0])
        ax.set_ylim(*bounds[1])
        ax.set_aspect("equal")
        head(ax, k)
        if multi:
            sub = grid[1:, c].subgridspec(3, 2)
            order = [("x₁", "x₂"), ("y₁", "y₂"), ("M₁", "M₂")]
            for row, pair in enumerate(order):
                for col, name in enumerate(pair):
                    draw_trace(
                        fig.add_subplot(sub[row, col]),
                        replay,
                        column(replay, name),
                        k,
                        colors,
                    )
        else:
            for j in range(len(names)):
                draw_trace(fig.add_subplot(grid[1 + j, c]), replay, j, k, colors)
    return [fig]


def sheet_model_comparison(replay, colors):
    ks = frames(replay)
    arrays, meta = replay.arrays, replay.meta
    model_ks, posterior = arrays["model_ks"], arrays["model_posterior"]
    T = meta["n_observations"]
    fig, grid = new_sheet(1, 4.2, ks, meta["demo"])
    gutter_x = T * 1.08
    label_y = np.linspace(0.85, 0.15, len(model_ks))  # fixed order, fixed gutter
    for c, k in enumerate(ks):
        ax = fig.add_subplot(grid[0, c])
        for i, (count, color) in enumerate(zip(model_ks, colors.ordered)):
            ax.plot(
                np.arange(k + 1),
                posterior[: k + 1, i],
                color=color,
                lw=1.6,
                marker="o" if k == 0 else None,
                ms=4,
            )
            head_y = posterior[k, i]
            ax.plot([k, gutter_x], [head_y, label_y[i]], color=color, lw=0.6)
            text = f"K = {count}"
            if count == meta["true_k"]:
                text += " (true model)"
            ax.text(
                gutter_x, label_y[i], " " + text, color=color, fontsize=8, va="center"
            )
        ax.set_xlim(0, T * 1.45)
        ax.set_ylim(0, 1)
        ax.set_xlabel("observations")
        ax.set_ylabel("posterior probability")
        head(ax, k)
    return [fig]


def sheet_density(replay, colors):
    arrays, meta = replay.arrays, replay.meta
    fig = plt.figure(figsize=(WIDTH * 0.6, 9.0), dpi=DPI)
    grid = fig.add_gridspec(4, 1, height_ratios=[2.2, 1, 1, 1])
    fig.suptitle(f"{meta['demo']} (static figure, last row)", fontsize=12)
    ax = fig.add_subplot(grid[0])
    x = np.linspace(*meta["prior_ranges"][column(replay, "μ")], 400)

    def profile(mu, sigma, amplitude):
        return amplitude * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

    mu_c, sigma_c, a_c = (column(replay, n) for n in ("μ", "σ", "A"))
    t = arrays["true_params"]
    ax.plot(x, profile(t[mu_c], t[sigma_c], t[a_c]), color=colors.truth, label="true")
    cloud = arrays["cloud"][-1].astype(np.float64)
    curves = profile(cloud[:, [mu_c]], cloud[:, [sigma_c]], cloud[:, [a_c]])
    low, median, high = np.percentile(curves, [16, 50, 84], axis=0)
    ax.fill_between(x, low, high, color=colors.belief, alpha=0.25)
    ax.plot(x, median, color=colors.belief, label="pointwise median and 68% band")
    xs = arrays["clock_positions"][:, 0]
    ax.scatter(
        xs,
        np.zeros_like(xs),
        marker="s",
        s=40,
        color=colors.ink,
        zorder=3,
        clip_on=False,
    )
    ax.legend(fontsize=8)
    ax.set_xlabel("position along the track")
    ax.set_ylabel("line density")
    for j, name in enumerate(("μ", "σ", "A")):
        draw_trace(
            fig.add_subplot(grid[1 + j]),
            replay,
            column(replay, name),
            meta["n_observations"],
            colors,
        )
    return [fig]


def sheet_echolocation(replay, colors):
    ks = frames(replay)
    arrays, meta = replay.arrays, replay.meta
    lo, hi = arrays["view_bounds"]
    clocks = arrays["clock_positions"]
    truth = arrays["true_params"]
    observed_all = (
        np.cumsum(arrays["observations"], axis=0)
        / np.arange(1, meta["n_observations"] + 1)[:, None]
    )
    fitted_all = arrays["predicted_mean"][1:]
    both = np.concatenate([observed_all.ravel(), fitted_all.ravel()])
    pad = 0.05 * (both.max() - both.min())
    agree_limits = (both.min() - pad, both.max() + pad)

    fig_a = plt.figure(figsize=(WIDTH, 13.5), dpi=DPI, layout="none")
    fig_a.suptitle(f"{meta['demo']} (sheet 1 of 2: scene panels)", fontsize=12)
    grid = fig_a.add_gridspec(
        3,
        len(ks),
        height_ratios=[1.5, 1, 1],
        hspace=0.35,
        wspace=0.3,
        left=0.04,
        right=0.98,
        top=0.93,
        bottom=0.04,
    )
    for c, k in enumerate(ks):
        ax = fig_a.add_subplot(grid[0, c], projection="3d")
        draw_cube(ax, lo, hi, colors.muted)
        ax.scatter(*clocks.T, marker="s", s=10, color=colors.ink, depthshade=False)
        pts = arrays["cloud"][k][:, :3]
        inside = np.all((pts >= lo) & (pts <= hi), axis=1)
        ax.scatter(
            *pts[inside].T, s=3, color=colors.belief, alpha=0.5, depthshade=False
        )
        ax.scatter(
            *truth[:3],
            s=260,
            facecolors="none",
            edgecolors=colors.truth,
            lw=1.6,
            depthshade=False,
        )
        ax.plot(
            [truth[0]] * 2, [truth[1]] * 2, [truth[2], lo], color=colors.truth, lw=1
        )
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_zlim(lo, hi)
        ax.view_init(elev=18, azim=-60)
        ax.set_title(
            f"k = {k}: {status(k)}\n{inside.sum()} of {len(pts)} shown dots in view",
            fontsize=9,
        )

        box = fig_a.add_subplot(grid[1, c])
        x_c, z_c = column(replay, "x"), column(replay, "z")
        cloud = arrays["cloud"][k]
        box.scatter(cloud[:, x_c], cloud[:, z_c], s=3, color=colors.belief, alpha=0.5)
        box.add_patch(
            Rectangle((lo, lo), hi - lo, hi - lo, fill=False, color=colors.ink, lw=1)
        )
        box.set_xlim(*meta["prior_ranges"][x_c])
        box.set_ylim(*meta["prior_ranges"][z_c])
        box.set_aspect("equal")
        box.set_xlabel("x")
        box.set_ylabel("z")
        share = 100 * arrays["weight_in_view"][k]
        words = "of the prior lies" if k == 0 else "of the cloud's weight lies"
        box.set_title(f"{share:.1f}% {words} inside the close view", fontsize=8)

        agree = fig_a.add_subplot(grid[2, c])
        agree.plot(agree_limits, agree_limits, color=colors.muted, lw=0.8)
        observed = running_mean(arrays["observations"], k)
        if observed is None:
            agree.text(
                0.5,
                0.5,
                "no readings yet",
                transform=agree.transAxes,
                ha="center",
                color=colors.muted,
            )
        else:
            agree.scatter(observed, arrays["predicted_mean"][k], s=10, color=colors.ink)
        agree.set_xlim(*agree_limits)
        agree.set_ylim(*agree_limits)
        agree.set_aspect("equal")
        agree.set_xlabel("observed mean")
        agree.set_ylabel("fitted")

    fig_b, grid_b = new_sheet(4, 2.4, ks, f"{meta['demo']} (sheet 2 of 2: traces)")
    for c, k in enumerate(ks):
        for j in range(4):
            ax = fig_b.add_subplot(grid_b[j, c])
            draw_trace(ax, replay, j, k, colors)
            if j == 0:
                ax.set_title(
                    f"k = {k}\n" + ax.get_title(loc="left"), fontsize=9, loc="left"
                )
    return [fig_a, fig_b]


def draw_cube(ax, lo, hi, color) -> None:
    corners = np.array(
        [[x, y, z] for x in (lo, hi) for y in (lo, hi) for z in (lo, hi)]
    )
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(corners[i] != corners[j]) == 1:
                ax.plot(*zip(corners[i], corners[j]), color=color, lw=0.6)


DRAWERS = {
    "demo_1d": lambda r, c: sheet_1d(r, c, multi=False),
    "demo_multi_mass": lambda r, c: sheet_1d(r, c, multi=True),
    "demo_2d": lambda r, c: sheet_map(r, c, multi=False),
    "demo_multi_mass_2d": lambda r, c: sheet_map(r, c, multi=True),
    "demo_model_comparison": sheet_model_comparison,
    "demo_density": sheet_density,
    "demo_echolocation_3d": sheet_echolocation,
}


def main() -> None:
    SHEETS.mkdir(exist_ok=True)
    names = sys.argv[1:] or list(DRAWERS)
    for name in names:
        replay = load_replay(CANDIDATES / f"{name}.npz")
        with style("light") as colors:
            figs = DRAWERS[name](replay, colors)
            for i, fig in enumerate(figs):
                suffix = "" if len(figs) == 1 else f"-{i + 1}"
                out = SHEETS / f"{name}{suffix}.png"
                fig.savefig(out, dpi=DPI)
                plt.close(fig)
                print(out)


if __name__ == "__main__":
    main()
