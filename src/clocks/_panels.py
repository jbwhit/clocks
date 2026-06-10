"""Static plotting primitives for clock-inference dashboards."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from clocks.types import ClockArray, MassConfig, ParticleState

# Colors for posterior probability bars per K value
_POSTERIOR_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

# Parameter labels and colors for multi-mass convergence plots
_MULTI_COLORS = ["tab:blue", "tab:cyan", "tab:orange", "tab:red"]
_MULTI_LABELS = ["x₁", "x₂", "M₁", "M₂"]

# 6-param multi-mass 2D convergence plots: x1, y1, x2, y2, M1, M2
_MULTI_2D_COLORS = [
    "tab:blue",
    "tab:cyan",
    "tab:orange",
    "tab:red",
    "tab:green",
    "tab:purple",
]
_MULTI_2D_LABELS = ["x₁", "y₁", "x₂", "y₂", "M₁", "M₂"]


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


def plot_particle_cloud_multi_1d(
    ax: Axes,
    particle_state: ParticleState,
    true_params: NDArray[np.floating] | None = None,
) -> None:
    """Scatter plot of x1 vs x2 for multi-mass particles, colored by weight.

    Particles layout: [x1, x2, M1, M2].
    """
    p = particle_state.particles
    w = particle_state.weights

    ax.scatter(p[:, 0], p[:, 1], c=w, cmap="viridis", s=5, alpha=0.6)
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

    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title(f"Position Cloud (n_obs={particle_state.observations_seen})")


def plot_particle_cloud_multi_2d(
    ax: Axes,
    particle_state: ParticleState,
    true_params: NDArray[np.floating] | None = None,
) -> None:
    """Scatter plot of mass 1 position (x1, y1) for multi-mass 2D particles.

    Particles layout: [x1, y1, x2, y2, M1, M2].
    Projects to columns 0 and 1 for the first mass position.
    """
    p = particle_state.particles
    w = particle_state.weights

    ax.scatter(p[:, 0], p[:, 1], c=w, cmap="viridis", s=5, alpha=0.6)
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

    ax.set_xlabel("x₁")
    ax.set_ylabel("y₁")
    ax.set_title(f"Mass 1 Cloud (n_obs={particle_state.observations_seen})")


def _plot_convergence(
    ax: Axes,
    steps: NDArray[np.integer],
    means: NDArray[np.floating],
    stds: NDArray[np.floating],
    true_params: NDArray[np.floating],
    colors: list[str],
    labels: list[str],
    *,
    legend_kwargs: dict | None = None,
) -> None:
    """Plot convergence traces with uncertainty bands and true-value lines."""
    for j, (color, lbl) in enumerate(zip(colors, labels)):
        ax.plot(steps, means[:, j], color=color, label=f"{lbl} est")
        ax.fill_between(
            steps,
            means[:, j] - stds[:, j],
            means[:, j] + stds[:, j],
            alpha=0.15,
            color=color,
        )
        ax.axhline(true_params[j], color=color, linestyle="--", alpha=0.5)
    ax.set_xlabel("Observation #")
    ax.set_ylabel("Parameter Value")
    legend_kw = {"fontsize": 8, "loc": "upper right"}
    if legend_kwargs:
        legend_kw.update(legend_kwargs)
    ax.legend(**legend_kw)
    ax.set_title("Convergence")
