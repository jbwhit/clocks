"""3D plotting primitives for the echolocation dashboard (hero layout)."""

from itertools import combinations, product

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from clocks.types import ClockArray, MassConfig, ParticleState

# Parameter labels/colors for the (x, y, z, M) convergence trace.
_ECHO_COLORS = ["tab:blue", "tab:green", "tab:purple", "tab:orange"]
_ECHO_LABELS = ["x", "y", "z", "M"]


def create_echolocation_dashboard(
    figsize: tuple[float, float] = (14.0, 8.0),
) -> tuple[Figure, dict[str, Axes]]:
    """Hero layout: 3D scene at ~2/3 width, three stacked diagnostics right.

    Returns (fig, axes) with keys: 'scene' (3D), 'history', 'mass', 'rates'.
    """
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(3, 3)
    axes = {
        "scene": fig.add_subplot(gs[:, :2], projection="3d"),
        "history": fig.add_subplot(gs[0, 2]),
        "mass": fig.add_subplot(gs[1, 2]),
        "rates": fig.add_subplot(gs[2, 2]),
    }
    return fig, axes


def _draw_head_wireframe(ax: Axes, half_width: float = 1.0) -> None:
    """The 12 edges of the head cube, so the lattice reads as an object."""
    corners = np.array(list(product((-half_width, half_width), repeat=3)))
    for start, end in combinations(corners, 2):
        if np.count_nonzero(start != end) == 1:  # edge: differs in one axis
            ax.plot3D(*zip(start, end), color="gray", alpha=0.4, linewidth=0.8)


def plot_scene_3d(
    ax: Axes,
    clock_array: ClockArray,
    mass_config: MassConfig,
    particle_state: ParticleState,
    *,
    azim: float,
    elev: float = 18.0,
    max_particles: int = 1500,
) -> None:
    """The hero panel: head lattice, exterior mass, particle cloud."""
    ax.clear()
    _draw_head_wireframe(ax)
    cp = clock_array.positions
    ax.scatter(
        cp[:, 0],
        cp[:, 1],
        cp[:, 2],
        marker="s",
        s=15,
        color="steelblue",
        label="Clocks",
    )
    mp = mass_config.positions[0]
    ax.scatter(
        mp[0],
        mp[1],
        mp[2],
        marker="*",
        s=250,
        color="red",
        label="Mass (true)",
    )
    particles = particle_state.particles
    weights = particle_state.weights
    if len(particles) > max_particles:
        idx = np.linspace(0, len(particles) - 1, max_particles).astype(int)
        particles, weights = particles[idx], weights[idx]
    ax.scatter(
        particles[:, 0],
        particles[:, 1],
        particles[:, 2],
        c=weights,
        cmap="viridis",
        s=3,
        alpha=0.3,
    )
    lim = float(np.max(np.abs(mass_config.positions))) + 2.0
    ax.set_xlim3d(-lim, lim)
    ax.set_ylim3d(-lim, lim)
    ax.set_zlim3d(-lim, lim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=elev, azim=azim)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(
        f"Head lattice + particle cloud (n_obs={particle_state.observations_seen})"
    )


def plot_centered_rates(
    ax: Axes,
    observed: NDArray[np.floating],
    predicted_centered: NDArray[np.floating],
) -> None:
    """Centered (differential) rates by clock index: prediction vs observed.

    ``predicted_centered`` is the forward model at the filter's current
    estimate, centered — so the panel shows the fit improving over frames.
    """
    idx = np.arange(len(observed))
    ax.bar(
        idx,
        predicted_centered,
        width=0.8,
        color="lightcoral",
        alpha=0.7,
        label="Predicted",
    )
    ax.bar(idx, observed, width=0.4, color="steelblue", alpha=0.7, label="Observed")
    ax.axhline(0.0, color="gray", linewidth=0.5)
    ax.set_xlabel("Clock index")
    ax.set_ylabel("Centered rate")
    ax.legend(fontsize=7)
    ax.set_title("Differential Rates")
