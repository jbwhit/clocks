"""Prototype restyle of the 2D demo (scratch only; nothing written to the repo).

Usage: uv run --frozen --project <repo> python proto_demo2d.py <outdir> <fontdir>
"""

import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse

from clocks.api import build_particle_filter, simulate
from clocks.config import InferenceConfig, NoiseConfig, PriorConfig, SimulationConfig
from clocks.types import ClockArray, MassConfig

OUT = Path(sys.argv[1])
FONTS = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
for ttf in FONTS.glob("*.ttf"):
    font_manager.fontManager.addfont(str(ttf))

THEMES = {
    "light": dict(
        surface="#f9f5d7",
        ink="#282828",
        ink2="#665c54",
        rule="#d5c4a1",
        grid="#ebe2bd",
        belief="#1a6fa3",
        truth="#bf4a0a",
        ramp=["#f1dfc0", "#e39a5f", "#d4691f", "#a83f08", "#6b2603"],
    ),
    "dark": dict(
        surface="#1d2021",
        ink="#fbf1c7",
        ink2="#bdae93",
        rule="#504945",
        grid="#2b2f30",
        belief="#4595cf",
        truth="#e0701a",
        ramp=["#3a2a1c", "#7a3408", "#b4540f", "#e0701a", "#f5a868"],
    ),
}


def rc(t: dict) -> dict:
    return {
        "font.family": "Source Sans 3",
        "font.size": 10.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "figure.facecolor": t["surface"],
        "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"],
        "text.color": t["ink"],
        "axes.labelcolor": t["ink2"],
        "axes.edgecolor": t["rule"],
        "xtick.color": t["ink2"],
        "ytick.color": t["ink2"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.grid": True,
        "grid.color": t["grid"],
        "grid.linewidth": 0.8,
        "lines.linewidth": 1.8,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "axes.titlesize": 10.5,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
    }


# ---- run the real filter with the committed demo_2d configuration ----------
TRUE = np.array([1.5, -1.0, 0.15])
CLOCKS = np.array(
    [
        [-4.0, 0.0],
        [-2.0, 3.0],
        [1.0, 4.0],
        [4.0, 2.0],
        [5.0, -1.0],
        [2.0, -4.0],
        [-1.0, -3.0],
        [-3.0, -1.5],
    ]
)
N_OBS, NOISE = 50, 0.005
clock_array = ClockArray(CLOCKS, track_offset=3.0)
truth = MassConfig(np.array([TRUE[:2]]), np.array([TRUE[2]]))
sim = simulate(
    SimulationConfig(
        clock_array=clock_array,
        ground_truth=truth,
        noise=NoiseConfig(NOISE),
        n_observations=N_OBS,
        seed=42,
    )
)
pf = build_particle_filter(
    InferenceConfig(
        clock_array=clock_array,
        noise=NoiseConfig(NOISE),
        prior=PriorConfig((-8.0, 8.0), (0.005, 0.15)),
        n_particles=2000,
        n_masses=1,
        seed=42,
    )
)
for obs in sim.observations:
    pf.update(obs)
states = pf.history  # states[0] is the prior
assert len(states) == N_OBS + 1
means = np.array([np.average(s.particles, weights=s.weights, axis=0) for s in states])
stds = np.array(
    [
        np.sqrt(np.average((s.particles - m) ** 2, weights=s.weights, axis=0))
        for s, m in zip(states, means)
    ]
)
obs_rates = np.array([o.rates for o in sim.observations])
true_deficit = 1.0 - sim.true_rates
print("final mean", means[-1], "std", stds[-1])
print("M posterior max over particles (final):", states[-1].particles[:, 2].max())


def draw(fig, t: dict, k: int) -> None:
    """Draw the state after k observations (k=0 is the prior)."""
    fig.clear()
    halo = [pe.withStroke(linewidth=2.6, foreground=t["surface"])]
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.18, 1.0],
        left=0.065,
        right=0.985,
        top=0.865,
        bottom=0.115,
        wspace=0.26,
        hspace=0.42,
    )
    ax = fig.add_subplot(gs[:, 0])
    s = states[k]
    # belief: one hue, density carried by overlap (weights are ~uniform after resampling)
    ax.scatter(
        s.particles[:, 0],
        s.particles[:, 1],
        s=5,
        color=t["belief"],
        alpha=0.28,
        linewidths=0,
        zorder=2,
        rasterized=True,
    )
    # clocks: fill = running-mean rate deficit (sequential, one hue)
    cmap = LinearSegmentedColormap.from_list("deficit", t["ramp"])
    if k == 0:
        ax.scatter(
            CLOCKS[:, 0],
            CLOCKS[:, 1],
            marker="s",
            s=70,
            facecolor=t["surface"],
            edgecolor=t["ink"],
            linewidths=1.0,
            zorder=4,
        )
    else:
        deficit = 1.0 - obs_rates[:k].mean(axis=0)
        ax.scatter(
            CLOCKS[:, 0],
            CLOCKS[:, 1],
            marker="s",
            s=70,
            c=deficit,
            cmap=cmap,
            vmin=0.0,
            vmax=true_deficit.max() * 1.1,
            edgecolor=t["ink"],
            linewidths=1.0,
            zorder=4,
        )
    # truth: ring + crosshair, same mark on every page
    ax.plot(
        *TRUE[:2],
        marker="o",
        ms=11,
        mfc="none",
        mec=t["truth"],
        mew=1.8,
        zorder=6,
        path_effects=halo,
    )
    ax.plot(*TRUE[:2], marker="+", ms=17, mec=t["truth"], mew=1.8, zorder=6)
    ax.annotate(
        "hidden mass",
        TRUE[:2],
        xytext=(12, -14),
        textcoords="offset points",
        color=t["truth"],
        fontsize=10,
        fontweight="semibold",
        path_effects=halo,
        zorder=7,
    )
    ax.annotate(
        "clock",
        CLOCKS[2],
        xytext=(9, 4),
        textcoords="offset points",
        color=t["ink2"],
        fontsize=9.5,
        path_effects=halo,
    )
    ax.set(xlim=(-8, 8), ylim=(-8, 8), aspect="equal", xlabel="x", ylabel="y")
    ax.set_xticks(range(-8, 9, 4))
    ax.set_yticks(range(-8, 9, 4))
    ax.set_ylabel("y", rotation=0, labelpad=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(True)
    # zoom inset: fixed window, so the late-time cloud stays readable
    zw = 0.6
    ins = ax.inset_axes([0.03, 0.67, 0.30, 0.30])
    ins.scatter(
        s.particles[:, 0],
        s.particles[:, 1],
        s=5,
        color=t["belief"],
        alpha=0.28,
        linewidths=0,
        rasterized=True,
    )
    ins.plot(*TRUE[:2], marker="o", ms=11, mfc="none", mec=t["truth"], mew=1.8)
    ins.plot(*TRUE[:2], marker="+", ms=17, mec=t["truth"], mew=1.8)
    if k > 0:
        cov = np.cov(s.particles[:, :2].T, aweights=s.weights)
        vals, vecs = np.linalg.eigh(cov)
        ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
        for target in (ax, ins):
            target.add_patch(
                Ellipse(
                    means[k, :2],
                    *(2 * 2 * np.sqrt(vals[::-1])),
                    angle=ang,
                    fill=False,
                    ec=t["belief"],
                    lw=1.2,
                    zorder=5,
                )
            )
    ins.set(
        xlim=(TRUE[0] - zw, TRUE[0] + zw),
        ylim=(TRUE[1] - zw, TRUE[1] + zw),
        aspect="equal",
    )
    ins.set_xticks([])
    ins.set_yticks([])
    ins.grid(False)
    for sp in ins.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(t["ink2"])
    ins.text(
        0.04,
        0.05,
        "zoom ×13",
        transform=ins.transAxes,
        fontsize=8.5,
        color=t["ink2"],
        path_effects=halo,
        zorder=9,
    )
    # inline key for the clock fill
    ax.text(
        0.985,
        0.03,
        "darker clock = slower" if t is THEMES["light"] else "brighter clock = slower",
        transform=ax.transAxes,
        ha="right",
        fontsize=9,
        color=t["ink2"],
        path_effects=halo,
    )

    # small multiples: one parameter per panel, its own scale, fixed axes
    spec = [
        ("x", 0, (0.4, 2.6), "{:.2f}"),
        ("y", 1, (-2.1, 0.1), "{:.2f}"),
        ("M", 2, (0.06, 0.16), "{:.3f}"),
    ]
    steps = np.arange(k + 1)
    for row, (name, j, ylim, fmt) in enumerate(spec):
        a = fig.add_subplot(gs[row, 1])
        a.axhline(TRUE[j], color=t["truth"], lw=1.4, zorder=2)
        a.fill_between(
            steps,
            means[: k + 1, j] - stds[: k + 1, j],
            means[: k + 1, j] + stds[: k + 1, j],
            color=t["belief"],
            alpha=0.22,
            linewidth=0,
            zorder=3,
        )
        a.plot(steps, means[: k + 1, j], color=t["belief"], zorder=4)
        a.plot(
            steps[-1],
            means[k, j],
            "o",
            ms=5.5,
            color=t["belief"],
            mec=t["surface"],
            mew=1.2,
            zorder=5,
        )
        a.set(xlim=(0, N_OBS), ylim=ylim)
        a.set_xticks([0, 10, 20, 30, 40, 50])
        a.yaxis.set_major_locator(plt.MaxNLocator(3))
        a.grid(axis="x", visible=False)
        a.text(
            0.0,
            1.07,
            name,
            transform=a.transAxes,
            fontsize=11.5,
            fontweight="semibold",
            fontstyle="italic",
        )
        a.text(
            1.0,
            1.07,
            (fmt + " ± " + fmt).format(means[k, j], stds[k, j]),
            transform=a.transAxes,
            ha="right",
            color=t["belief"],
            fontweight="semibold",
            fontsize=10.5,
        )
        a.text(
            0.5,
            1.07,
            ("true " + fmt.replace("3", "2")).format(TRUE[j]),
            transform=a.transAxes,
            ha="center",
            color=t["truth"],
            fontsize=10,
        )
        if row < 2:
            a.set_xticklabels([])
        else:
            a.set_xlabel("observations")
    fig.text(
        0.065,
        0.95,
        f"After {k} observation{'s' if k != 1 else ''}"
        if k
        else "Before any observation",
        fontsize=12.5,
        fontweight="semibold",
    )
    fig.text(
        0.985,
        0.955,
        "posterior mean ± 1 s.d.; ellipse = 2 s.d.",
        ha="right",
        fontsize=9.5,
        color=t["ink2"],
    )


for name, t in THEMES.items():
    with plt.rc_context(rc(t)):
        fig = plt.figure(figsize=(7.5, 4.6))
        for k in (0, 1, 8, 50):
            draw(fig, t, k)
            fig.savefig(OUT / f"proto_demo2d_{name}_k{k:02d}.png", dpi=200)
        # paced animation: dwell on the prior and the first updates, speed up, hold the end
        fps = 20
        dwell = [1.2] + [0.9] * 3 + [0.5] * 5 + [0.25] * 12 + [0.1] * (N_OBS - 20)
        dwell[-1] = 2.5
        frames_dir = OUT / f"frames_{name}"
        frames_dir.mkdir(exist_ok=True)
        concat = []
        for k in range(N_OBS + 1):
            draw(fig, t, k)
            path = frames_dir / f"k{k:02d}.png"
            fig.savefig(path, dpi=160)
            concat += [f"file '{path}'", f"duration {dwell[k]}"]
        concat.append(f"file '{path}'")
        (frames_dir / "list.txt").write_text("\n".join(concat))
        plt.close(fig)
        out = OUT / f"proto_demo2d_{name}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(frames_dir / "list.txt"),
                "-vf",
                f"fps={fps},format=yuv420p",
                "-c:v",
                "libx264",
                "-crf",
                "20",
                "-preset",
                "slow",
                "-movflags",
                "+faststart",
                str(out),
            ],
            check=True,
        )
        print(out, out.stat().st_size, "bytes; total s", sum(dwell))
