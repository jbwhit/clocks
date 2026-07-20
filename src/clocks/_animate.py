"""Animation drivers for clock-inference dashboards."""

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from clocks._panels import (
    _MULTI_2D_COLORS,
    _MULTI_2D_LABELS,
    _MULTI_COLORS,
    _MULTI_LABELS,
    _POSTERIOR_COLORS,
    _plot_convergence,
    create_inference_dashboard,
    plot_clock_rates,
    plot_clock_setup,
    plot_clock_setup_2d,
    plot_mass_histogram,
    plot_particle_cloud,
    plot_particle_cloud_2d,
    plot_particle_cloud_multi_1d,
    plot_particle_cloud_multi_2d,
)
from clocks._panels3d import (
    _ECHO_COLORS,
    _ECHO_LABELS,
    create_echolocation_dashboard,
    plot_centered_rates,
    plot_scene_3d,
)
from clocks.inference import ModelComparison, ParticleFilter
from clocks.physics import clock_rates
from clocks.types import ClockArray, MassConfig, Observation, ParticleState


def _save_animation(
    anim: animation.FuncAnimation,
    fig: Figure,
    output_path: Path,
    fps: int,
) -> None:
    """Save animation to file (gif or mp4) and close the figure."""
    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        anim.save(str(output_path), writer="pillow", fps=fps)
    else:
        anim.save(str(output_path), fps=fps)
    plt.close(fig)


def _precompute_filter_states(
    pf: ParticleFilter,
    observations: list[Observation],
) -> tuple[list[ParticleState], list[NDArray[np.floating]], list[NDArray[np.floating]]]:
    """Run the filter through all observations up front (frame-0 fix)."""
    states: list[ParticleState] = []
    means: list[NDArray[np.floating]] = []
    stds: list[NDArray[np.floating]] = []
    for obs in observations:
        state = pf.update(obs)
        est = pf.estimate()
        states.append(state)
        means.append(est["mean"])
        stds.append(est["std"])
    if pf.state.observations_seen != len(observations):
        raise RuntimeError(
            f"Animation expected a fresh filter: saw "
            f"{pf.state.observations_seen} observations for "
            f"{len(observations)} frames"
        )
    return states, means, stds


def _animate_filter_dashboard(
    fig: Figure,
    axes: dict[str, Axes],
    pf: ParticleFilter,
    observations: list[Observation],
    output_path: Path,
    fps: int,
    render_particles: Callable[[Axes, ParticleState], None],
    render_rates: Callable[[Axes, Observation, int], None],
    render_history: Callable[[Axes, NDArray[np.floating], NDArray[np.floating]], None],
) -> None:
    """Drive a particle filter through observations on the 2x2 dashboard.

    Render callables own their panel completely, including ``ax.clear()``
    and any artist lifecycle (e.g. colorbars).
    """
    states, means, stds = _precompute_filter_states(pf, observations)

    def render(frame: int) -> None:
        render_particles(axes["particles"], states[frame])
        render_rates(axes["rates"], observations[frame], frame)
        render_history(
            axes["history"],
            np.array(means[: frame + 1]),
            np.array(stds[: frame + 1]),
        )

    anim = animation.FuncAnimation(fig, render, frames=len(observations), repeat=False)
    _save_animation(anim, fig, output_path, fps)


def _make_rates_renderer_2d(
    clock_array: ClockArray,
    true_rates: NDArray[np.floating],
    xylim: tuple[float, float],
) -> Callable[[Axes, Observation, int], None]:
    """Per-frame renderer for the 2D observed-rates panel (owns its colorbar)."""
    cbar_state: list = []

    def render(ax: Axes, obs: Observation, frame: int) -> None:
        if cbar_state:
            cbar_state[0].remove()
            cbar_state.clear()
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
        for x, y, r in zip(cx, cy, obs.rates):
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
        cbar_state.append(plt.colorbar(sc, ax=ax, label="Rate", shrink=0.8))

    return render


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
    plot_clock_setup(axes["setup"], clock_array, mass_config)
    axes["setup"].set_xlim(xlim)

    def render_particles(ax: Axes, state: ParticleState) -> None:
        ax.clear()
        plot_particle_cloud(ax, state, true_params)
        ax.set_xlim(xlim)
        ax.set_ylim(mlim)

    def render_rates(ax: Axes, obs: Observation, frame: int) -> None:
        ax.clear()
        plot_clock_rates(ax, true_rates, clock_array, label="True", color="lightcoral")
        plot_clock_rates(
            ax, obs.rates, clock_array, label="Observed", color="steelblue"
        )

    def render_history(
        ax: Axes, means: NDArray[np.floating], stds: NDArray[np.floating]
    ) -> None:
        ax.clear()
        steps = np.arange(1, len(means) + 1)
        _plot_convergence(
            ax,
            steps,
            means,
            stds,
            true_params,
            ["tab:blue", "tab:orange"],
            ["x", "M"],
        )

    _animate_filter_dashboard(
        fig,
        axes,
        pf,
        observations,
        output_path,
        fps,
        render_particles,
        render_rates,
        render_history,
    )


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

    fig, axes = create_inference_dashboard(figsize=(13, 10))
    plot_clock_setup_2d(axes["setup"], clock_array, mass_config)
    axes["setup"].set_xlim(xylim)
    axes["setup"].set_ylim(xylim)

    def render_particles(ax: Axes, state: ParticleState) -> None:
        ax.clear()
        plot_particle_cloud_2d(ax, state, true_params[:2])
        ax.set_xlim(xylim)
        ax.set_ylim(xylim)
        ax.set_aspect("equal")

    render_rates = _make_rates_renderer_2d(clock_array, true_rates, xylim)

    def render_history(
        ax: Axes, means: NDArray[np.floating], stds: NDArray[np.floating]
    ) -> None:
        ax.clear()
        steps = np.arange(1, len(means) + 1)
        _plot_convergence(
            ax,
            steps,
            means,
            stds,
            true_params,
            ["tab:blue", "tab:green", "tab:orange"],
            ["x", "y", "M"],
        )

    _animate_filter_dashboard(
        fig,
        axes,
        pf,
        observations,
        output_path,
        fps,
        render_particles,
        render_rates,
        render_history,
    )


def animate_inference_multi_1d(
    clock_array: ClockArray,
    mass_config: MassConfig,
    observations: list[Observation],
    pf: ParticleFilter,
    output_path: Path,
    fps: int = 4,
    xlim: tuple[float, float] = (-8, 8),
    mlim: tuple[float, float] = (0, 2),
) -> None:
    """Animate multi-mass particle filter (2 masses in 1D) and save to file.

    Particles have 4 columns: [x1, x2, M1, M2].
    Dashboard: setup | x1-vs-x2 scatter | clock rates | 4-trace convergence.
    """
    true_params = np.array(
        [
            mass_config.positions[0, 0],
            mass_config.positions[1, 0],
            mass_config.masses[0],
            mass_config.masses[1],
        ]
    )
    true_rates = clock_rates(mass_config, clock_array)

    fig, axes = create_inference_dashboard()
    plot_clock_setup(axes["setup"], clock_array, mass_config)
    axes["setup"].set_xlim(xlim)

    def render_particles(ax: Axes, state: ParticleState) -> None:
        ax.clear()
        plot_particle_cloud_multi_1d(ax, state, true_params[:2])
        ax.set_xlim(xlim)
        ax.set_ylim(xlim)

    def render_rates(ax: Axes, obs: Observation, frame: int) -> None:
        ax.clear()
        plot_clock_rates(ax, true_rates, clock_array, label="True", color="lightcoral")
        plot_clock_rates(
            ax, obs.rates, clock_array, label="Observed", color="steelblue"
        )

    def render_history(
        ax: Axes, means: NDArray[np.floating], stds: NDArray[np.floating]
    ) -> None:
        ax.clear()
        steps = np.arange(1, len(means) + 1)
        _plot_convergence(
            ax,
            steps,
            means,
            stds,
            true_params,
            _MULTI_COLORS,
            _MULTI_LABELS,
            legend_kwargs={"fontsize": 7, "ncol": 2},
        )

    _animate_filter_dashboard(
        fig,
        axes,
        pf,
        observations,
        output_path,
        fps,
        render_particles,
        render_rates,
        render_history,
    )


def animate_inference_multi_2d(
    clock_array: ClockArray,
    mass_config: MassConfig,
    observations: list[Observation],
    pf: ParticleFilter,
    output_path: Path,
    fps: int = 4,
    xylim: tuple[float, float] = (-8, 8),
    mlim: tuple[float, float] = (0, 2),
) -> None:
    """Animate multi-mass 2D particle filter (2 masses on a plane) and save.

    Particles have 6 columns: [x1, y1, x2, y2, M1, M2].
    Dashboard: setup | mass-1 cloud | observed rates | 6-trace convergence.
    """
    true_params = np.array(
        [
            mass_config.positions[0, 0],
            mass_config.positions[0, 1],
            mass_config.positions[1, 0],
            mass_config.positions[1, 1],
            mass_config.masses[0],
            mass_config.masses[1],
        ]
    )
    true_rates = clock_rates(mass_config, clock_array)

    fig, axes = create_inference_dashboard(figsize=(13, 10))
    plot_clock_setup_2d(axes["setup"], clock_array, mass_config)
    axes["setup"].set_xlim(xylim)
    axes["setup"].set_ylim(xylim)

    def render_particles(ax: Axes, state: ParticleState) -> None:
        ax.clear()
        plot_particle_cloud_multi_2d(ax, state, true_params[:2])
        ax.set_xlim(xylim)
        ax.set_ylim(xylim)
        ax.set_aspect("equal")

    render_rates = _make_rates_renderer_2d(clock_array, true_rates, xylim)

    def render_history(
        ax: Axes, means: NDArray[np.floating], stds: NDArray[np.floating]
    ) -> None:
        ax.clear()
        steps = np.arange(1, len(means) + 1)
        _plot_convergence(
            ax,
            steps,
            means,
            stds,
            true_params,
            _MULTI_2D_COLORS,
            _MULTI_2D_LABELS,
            legend_kwargs={"fontsize": 6, "ncol": 3},
        )

    _animate_filter_dashboard(
        fig,
        axes,
        pf,
        observations,
        output_path,
        fps,
        render_particles,
        render_rates,
        render_history,
    )


def animate_model_comparison(
    clock_array: ClockArray,
    mass_config: MassConfig,
    observations: list[Observation],
    model_comparison: ModelComparison,
    output_path: Path,
    fps: int = 4,
    figsize: tuple[float, float] = (10, 4),
) -> None:
    """Animate Bayesian model comparison: rates + posterior probabilities.

    Left panel: true vs observed clock rates bar chart.
    Right panel: horizontal bars of posterior probability for each K.
    """
    true_rates = clock_rates(mass_config, clock_array)
    k_values = sorted(model_comparison.filters)
    n_obs = len(observations)

    fig, (ax_rates, ax_post) = plt.subplots(1, 2, figsize=figsize)
    fig.tight_layout(pad=3.0)

    posteriors_seq: list[dict[int, float]] = []
    for obs in observations:
        model_comparison.update(obs)
        posteriors_seq.append(model_comparison.evidence()["posterior"])
    for pf in model_comparison.filters.values():
        if pf.state.observations_seen != n_obs:
            raise RuntimeError(
                f"Animation expected a fresh filter: saw "
                f"{pf.state.observations_seen} observations for {n_obs} frames"
            )

    def update(frame: int) -> None:
        obs = observations[frame]
        result_posterior = posteriors_seq[frame]

        # Left panel: clock rates
        ax_rates.clear()
        plot_clock_rates(
            ax_rates, true_rates, clock_array, label="True", color="lightcoral"
        )
        plot_clock_rates(
            ax_rates, obs.rates, clock_array, label="Observed", color="steelblue"
        )

        # Right panel: posterior probabilities
        ax_post.clear()
        posteriors = [result_posterior[k] for k in k_values]
        colors = [
            _POSTERIOR_COLORS[i % len(_POSTERIOR_COLORS)] for i in range(len(k_values))
        ]
        labels = [f"K={k}" for k in k_values]
        bars = ax_post.barh(labels, posteriors, color=colors, height=0.5)
        for bar, p in zip(bars, posteriors):
            ax_post.text(
                bar.get_width() + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{p:.2f}",
                va="center",
                fontsize=9,
            )
        ax_post.set_xlim(0, 1.15)
        ax_post.set_xlabel("Posterior probability")
        ax_post.set_title(f"Observation {frame + 1}/{n_obs}")

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n_obs,
        repeat=False,
    )
    _save_animation(anim, fig, output_path, fps)


def animate_echolocation(
    clock_array: ClockArray,
    mass_config: MassConfig,
    observations: list[Observation],
    pf: ParticleFilter,
    output_path: Path,
    fps: int = 4,
) -> None:
    """Animate the 3D echolocation filter with a slowly orbiting camera.

    ``observations`` must be the centered observations the filter consumes
    (the head has no external reference). One full azimuth orbit spans the
    whole animation. Particles have 4 columns: [x, y, z, M].
    """
    true_params = np.append(mass_config.positions[0], mass_config.masses[0])

    fig, axes = create_echolocation_dashboard()
    states, means, stds = _precompute_filter_states(pf, observations)
    n_frames = len(observations)

    def predicted_centered(frame: int) -> NDArray[np.floating]:
        """Centered forward model at the frame's posterior mean (spec §2)."""
        mean = means[frame]
        rates = clock_rates(
            MassConfig(positions=mean[:3].reshape(1, 3), masses=mean[3:4]),
            clock_array,
        )
        return rates - rates.mean()

    def render(frame: int) -> None:
        azim = -60.0 + 360.0 * frame / n_frames
        plot_scene_3d(axes["scene"], clock_array, mass_config, states[frame], azim=azim)
        axes["history"].clear()
        steps = np.arange(1, frame + 2)
        _plot_convergence(
            axes["history"],
            steps,
            np.array(means[: frame + 1]),
            np.array(stds[: frame + 1]),
            true_params,
            _ECHO_COLORS,
            _ECHO_LABELS,
            legend_kwargs={"fontsize": 7, "ncol": 2},
        )
        axes["mass"].clear()
        plot_mass_histogram(axes["mass"], states[frame], float(true_params[3]))
        axes["rates"].clear()
        plot_centered_rates(
            axes["rates"], observations[frame].rates, predicted_centered(frame)
        )

    anim = animation.FuncAnimation(fig, render, frames=n_frames, repeat=False)
    _save_animation(anim, fig, output_path, fps)
