"""Visualization and animation helpers for gravitational time dilation demos."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from clocks.inference import ParticleFilter
from clocks.physics import clock_rates
from clocks.types import ClockArray, MassConfig, Observation, ParticleState


def plot_clock_setup(
    ax: Axes,
    clock_array: ClockArray,
    mass_config: MassConfig | None = None,
) -> None:
    """Plot the physical layout of clocks (and optionally masses) in 1D."""
    positions = clock_array.positions[:, 0]
    ax.scatter(
        positions,
        np.zeros_like(positions),
        marker="s",
        s=100,
        color="steelblue",
        zorder=5,
        label="Clocks",
    )
    for i, x in enumerate(positions):
        ax.annotate(
            f"C{i}",
            (x, 0),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=9,
        )

    if mass_config is not None:
        mx = mass_config.positions[:, 0]
        ax.scatter(
            mx,
            np.zeros_like(mx),
            marker="*",
            s=200,
            color="red",
            zorder=5,
            label="Mass (true)",
        )

    ax.set_xlabel("Position")
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Physical Setup")


def plot_particle_cloud(
    ax: Axes,
    particle_state: ParticleState,
    true_params: NDArray[np.floating] | None = None,
) -> None:
    """Scatter plot of particles colored by weight (for 2-param case: x, M)."""
    p = particle_state.particles
    w = particle_state.weights

    ax.scatter(
        p[:, 0],
        p[:, 1],
        c=w,
        cmap="viridis",
        s=5,
        alpha=0.6,
    )
    if true_params is not None:
        ax.scatter(
            true_params[0],
            true_params[1],
            marker="x",
            s=100,
            color="red",
            linewidths=2,
            label="True",
        )
        ax.legend(fontsize=8)

    ax.set_xlabel("Position (x)")
    ax.set_ylabel("Mass (M)")
    ax.set_title(f"Particles (n_obs={particle_state.observations_seen})")


def plot_clock_rates(
    ax: Axes,
    rates: NDArray[np.floating],
    clock_array: ClockArray,
    label: str = "Rates",
    color: str = "steelblue",
) -> None:
    """Bar chart of clock rates at their positions."""
    positions = clock_array.positions[:, 0]
    ax.bar(positions, rates, width=0.4, color=color, alpha=0.7, label=label)
    for x, r in zip(positions, rates):
        ax.annotate(
            f"{r:.4f}",
            (x, r),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=8,
        )
    ax.set_xlabel("Clock Position")
    ax.set_ylabel("Tick Rate")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("Clock Rates")


def plot_clock_setup_2d(
    ax: Axes,
    clock_array: ClockArray,
    mass_config: MassConfig | None = None,
) -> None:
    """Plot the physical layout of clocks and masses on a 2D plane."""
    cx = clock_array.positions[:, 0]
    cy = clock_array.positions[:, 1]
    ax.scatter(cx, cy, marker="s", s=100, color="steelblue", zorder=5, label="Clocks")
    for i, (x, y) in enumerate(zip(cx, cy)):
        ax.annotate(
            f"C{i}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )

    if mass_config is not None:
        mx = mass_config.positions[:, 0]
        my = mass_config.positions[:, 1]
        ax.scatter(
            mx, my, marker="*", s=300, color="red", zorder=5, label="Mass (true)"
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Physical Setup")


def plot_particle_cloud_2d(
    ax: Axes,
    particle_state: ParticleState,
    true_params: NDArray[np.floating] | None = None,
) -> None:
    """Scatter particles on x-y plane, colored by weight, sized by mass."""
    p = particle_state.particles
    w = particle_state.weights

    ax.scatter(p[:, 0], p[:, 1], c=w, cmap="viridis", s=5, alpha=0.6)
    if true_params is not None:
        ax.scatter(
            true_params[0],
            true_params[1],
            marker="x",
            s=120,
            color="red",
            linewidths=2,
            label="True",
        )
        ax.legend(fontsize=8)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Particle Cloud (n_obs={particle_state.observations_seen})")


def plot_mass_histogram(
    ax: Axes,
    particle_state: ParticleState,
    true_mass: float | None = None,
) -> None:
    """Histogram of particle mass values (1D marginal of the last parameter)."""
    masses = particle_state.particles[:, -1]
    weights = particle_state.weights

    ax.hist(
        masses,
        bins=40,
        weights=weights,
        color="steelblue",
        alpha=0.7,
        density=True,
    )
    if true_mass is not None:
        ax.axvline(true_mass, color="red", linestyle="--", linewidth=2, label="True M")
        ax.legend(fontsize=8)

    ax.set_xlabel("Mass (M)")
    ax.set_ylabel("Density")
    ax.set_title("Mass Marginal")


def plot_clock_rates_2d(
    ax: Axes,
    rates: NDArray[np.floating],
    clock_array: ClockArray,
    label: str = "Rates",
) -> None:
    """Scatter clocks on 2D plane, colored by tick rate."""
    cx = clock_array.positions[:, 0]
    cy = clock_array.positions[:, 1]
    sc = ax.scatter(
        cx,
        cy,
        c=rates,
        cmap="coolwarm",
        s=120,
        marker="s",
        vmin=0.95,
        vmax=1.0,
        zorder=5,
        edgecolors="black",
        linewidths=0.5,
    )
    for i, (x, y, r) in enumerate(zip(cx, cy, rates)):
        ax.annotate(
            f"{r:.4f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=7,
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_title(label)
    plt.colorbar(sc, ax=ax, label="Rate", shrink=0.8)


def create_inference_dashboard(
    n_params: int = 2,
    figsize: tuple[float, float] = (12, 8),
) -> tuple[Figure, dict[str, Axes]]:
    """Create a 2x2 dashboard figure for inference animation.

    Returns (fig, axes_dict) with keys: 'setup', 'particles', 'rates', 'history'.
    """
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    axes = {
        "setup": axs[0, 0],
        "particles": axs[0, 1],
        "rates": axs[1, 0],
        "history": axs[1, 1],
    }
    fig.tight_layout(pad=3.0)
    return fig, axes


def animate_inference(
    clock_array: ClockArray,
    mass_config: MassConfig,
    observations: list[Observation],
    pf: ParticleFilter,
    output_path: Path,
    fps: int = 4,
    xlim: tuple[float, float] = (-8, 8),
    mlim: tuple[float, float] = (0, 2),
) -> None:
    """Animate the particle filter processing observations and save to file.

    Produces a gif or mp4 depending on output_path extension.
    """
    true_params = np.array(
        [
            mass_config.positions[0, 0],
            mass_config.masses[0],
        ]
    )
    true_rates = clock_rates(mass_config, clock_array)

    fig, axes = create_inference_dashboard()

    # Static: physical setup
    plot_clock_setup(axes["setup"], clock_array, mass_config)
    axes["setup"].set_xlim(xlim)

    # Track estimate history for the history panel
    means: list[NDArray] = []
    stds: list[NDArray] = []

    def update(frame: int) -> None:
        obs = observations[frame]
        pf.update(obs)
        est = pf.estimate()
        means.append(est["mean"])
        stds.append(est["std"])

        # Particles
        ax = axes["particles"]
        ax.clear()
        plot_particle_cloud(ax, pf.state, true_params)
        ax.set_xlim(xlim)
        ax.set_ylim(mlim)

        # Rates
        ax = axes["rates"]
        ax.clear()
        plot_clock_rates(ax, true_rates, clock_array, label="True", color="lightcoral")
        plot_clock_rates(
            ax, obs.rates, clock_array, label="Observed", color="steelblue"
        )

        # History
        ax = axes["history"]
        ax.clear()
        steps = np.arange(1, len(means) + 1)
        m = np.array(means)
        s = np.array(stds)
        ax.plot(steps, m[:, 0], color="tab:blue", label="x estimate")
        ax.fill_between(
            steps,
            m[:, 0] - s[:, 0],
            m[:, 0] + s[:, 0],
            alpha=0.2,
            color="tab:blue",
        )
        ax.axhline(true_params[0], color="tab:blue", linestyle="--", alpha=0.5)
        ax.plot(steps, m[:, 1], color="tab:orange", label="M estimate")
        ax.fill_between(
            steps,
            m[:, 1] - s[:, 1],
            m[:, 1] + s[:, 1],
            alpha=0.2,
            color="tab:orange",
        )
        ax.axhline(true_params[1], color="tab:orange", linestyle="--", alpha=0.5)
        ax.set_xlabel("Observation #")
        ax.set_ylabel("Parameter Value")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_title("Convergence")

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(observations),
        repeat=False,
    )

    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        anim.save(str(output_path), writer="pillow", fps=fps)
    else:
        anim.save(str(output_path), fps=fps)

    plt.close(fig)


def create_inference_dashboard_2d(
    figsize: tuple[float, float] = (13, 10),
) -> tuple[Figure, dict[str, Axes]]:
    """Create a 2x2 dashboard for 2D inference animation.

    Returns (fig, axes_dict) with keys: 'setup', 'particles', 'rates', 'history'.
    """
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    axes = {
        "setup": axs[0, 0],
        "particles": axs[0, 1],
        "rates": axs[1, 0],
        "history": axs[1, 1],
    }
    fig.tight_layout(pad=3.0)
    return fig, axes


def animate_inference_2d(
    clock_array: ClockArray,
    mass_config: MassConfig,
    observations: list[Observation],
    pf: ParticleFilter,
    output_path: Path,
    fps: int = 4,
    xylim: tuple[float, float] = (-8, 8),
    mlim: tuple[float, float] = (0, 2),
) -> None:
    """Animate the 2D particle filter and save to file.

    Particles have 3 columns: [x, y, M].
    """
    true_params = np.array(
        [
            mass_config.positions[0, 0],
            mass_config.positions[0, 1],
            mass_config.masses[0],
        ]
    )
    true_rates = clock_rates(mass_config, clock_array)

    fig, axes = create_inference_dashboard_2d()

    # Static: physical setup
    plot_clock_setup_2d(axes["setup"], clock_array, mass_config)
    axes["setup"].set_xlim(xylim)
    axes["setup"].set_ylim(xylim)

    # Track estimate history
    means: list[NDArray] = []
    stds: list[NDArray] = []

    # Colorbar state for rates panel
    rates_cbar: list = []

    def update(frame: int) -> None:
        obs = observations[frame]
        pf.update(obs)
        est = pf.estimate()
        means.append(est["mean"])
        stds.append(est["std"])

        # Particles — x,y scatter
        ax = axes["particles"]
        ax.clear()
        plot_particle_cloud_2d(ax, pf.state, true_params[:2])
        ax.set_xlim(xylim)
        ax.set_ylim(xylim)
        ax.set_aspect("equal")

        # Rates — scatter colored by observed rate
        ax = axes["rates"]
        # Remove old colorbar if it exists
        if rates_cbar:
            rates_cbar[0].remove()
            rates_cbar.clear()
        ax.clear()
        cx = clock_array.positions[:, 0]
        cy = clock_array.positions[:, 1]
        sc = ax.scatter(
            cx,
            cy,
            c=obs.rates,
            cmap="coolwarm",
            s=120,
            marker="s",
            vmin=min(true_rates) - 0.002,
            vmax=max(true_rates) + 0.002,
            zorder=5,
            edgecolors="black",
            linewidths=0.5,
        )
        for i, (x, y, r) in enumerate(zip(cx, cy, obs.rates)):
            ax.annotate(
                f"{r:.4f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, -14),
                ha="center",
                fontsize=7,
            )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.set_xlim(xylim)
        ax.set_ylim(xylim)
        ax.set_title(f"Observed Rates (t={frame + 1})")
        rates_cbar.append(plt.colorbar(sc, ax=ax, label="Rate", shrink=0.8))

        # History — convergence of x, y, M
        ax = axes["history"]
        ax.clear()
        steps = np.arange(1, len(means) + 1)
        m = np.array(means)
        s = np.array(stds)

        colors = ["tab:blue", "tab:green", "tab:orange"]
        labels = ["x", "y", "M"]
        for j, (color, lbl) in enumerate(zip(colors, labels)):
            ax.plot(steps, m[:, j], color=color, label=f"{lbl} est")
            ax.fill_between(
                steps,
                m[:, j] - s[:, j],
                m[:, j] + s[:, j],
                alpha=0.15,
                color=color,
            )
            ax.axhline(
                true_params[j],
                color=color,
                linestyle="--",
                alpha=0.5,
            )

        ax.set_xlabel("Observation #")
        ax.set_ylabel("Parameter Value")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_title("Convergence")

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(observations),
        repeat=False,
    )

    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        anim.save(str(output_path), writer="pillow", fps=fps)
    else:
        anim.save(str(output_path), fps=fps)

    plt.close(fig)
