"""Infer a Gaussian line-density profile from noisy clock observations."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from clocks._demos._common import add_common_arguments
from clocks._support import density_support_mask, sample_conditioned_prior
from clocks.inference import ParticleFilter
from clocks.noise import add_clock_noise
from clocks.physics import (
    clock_rates_density_gaussian,
    clock_rates_density_gaussian_batch,
)
from clocks.types import ClockArray, Observation

TRUE_MU = 1.5
TRUE_SIGMA = 2.0
TRUE_AMPLITUDE = 0.010
CLOCK_POSITIONS = [-6.0, -3.0, 0.0, 3.0, 6.0]
TRACK_OFFSET = 1.0
N_OBSERVATIONS = 60
NOISE_STD = 0.005
N_PARTICLES = 2000
SEED = 42
OUTPUT_PATH = Path("output/demo_density.png")
MU_RANGE = (-8.0, 8.0)
SIGMA_RANGE = (0.1, 5.0)
AMPLITUDE_RANGE = (0.001, 0.030)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without running the simulation."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(
        parser,
        output=OUTPUT_PATH,
        observations=N_OBSERVATIONS,
        particles=N_PARTICLES,
    )
    return parser


def _format_estimate(label: str, estimate: dict, index: int, truth: float) -> str:
    mean, std = estimate["mean"][index], estimate["std"][index]
    return f"  {label} = {mean:.3f} ± {std:.3f}  (true: {truth})"


def build_density_filter(
    clock_array: ClockArray,
    rng: np.random.Generator,
    *,
    n_particles: int = N_PARTICLES,
) -> ParticleFilter:
    """Build a density filter whose sampler and density share exact support."""

    def draw(draw_rng: np.random.Generator, count: int) -> np.ndarray:
        return np.column_stack(
            [
                draw_rng.uniform(*MU_RANGE, count),
                draw_rng.uniform(*SIGMA_RANGE, count),
                draw_rng.uniform(*AMPLITUDE_RANGE, count),
            ]
        )

    def valid(particles: np.ndarray) -> np.ndarray:
        return density_support_mask(
            particles,
            clock_array=clock_array,
            mu_range=MU_RANGE,
            sigma_range=SIGMA_RANGE,
            amplitude_range=AMPLITUDE_RANGE,
        )

    def prior_sampler(draw_rng: np.random.Generator, count: int) -> np.ndarray:
        return sample_conditioned_prior(
            draw_rng,
            count,
            draw,
            valid,
            description="Gaussian density",
        )

    def forward_model(params: np.ndarray) -> np.ndarray:
        return clock_rates_density_gaussian(params, clock_array)

    def forward_model_batch(particles: np.ndarray) -> np.ndarray:
        return clock_rates_density_gaussian_batch(particles, clock_array)

    def log_prior_density(particles: np.ndarray) -> np.ndarray:
        result = np.full(particles.shape[0], -np.inf)
        result[valid(particles)] = 0.0
        return result

    return ParticleFilter(
        n_particles=n_particles,
        prior_sampler=prior_sampler,
        forward_model=forward_model,
        noise_std=NOISE_STD,
        rng=rng,
        forward_model_batch=forward_model_batch,
        log_prior_density=log_prior_density,
    )


def _save_summary(
    output: Path,
    clock_array: ClockArray,
    true_rates: np.ndarray,
    particle_filter: ParticleFilter,
    estimate: dict,
) -> None:
    fig, (density_axis, rates_axis, convergence_axis) = plt.subplots(
        1, 3, figsize=(15, 4)
    )
    positions = np.linspace(-8, 8, 400)
    true_density = TRUE_AMPLITUDE * np.exp(
        -0.5 * ((positions - TRUE_MU) / TRUE_SIGMA) ** 2
    )
    mu_hat, sigma_hat, amplitude_hat = estimate["mean"]
    estimated_density = amplitude_hat * np.exp(
        -0.5 * ((positions - mu_hat) / sigma_hat) ** 2
    )
    density_axis.plot(positions, true_density, color="lightcoral", label="True")
    density_axis.plot(
        positions, estimated_density, color="steelblue", ls="--", label="Inferred"
    )
    density_axis.set(xlabel="x", ylabel="mass density", title="Density profile")
    density_axis.legend()

    predicted = clock_rates_density_gaussian(estimate["mean"], clock_array)
    clock_positions = clock_array.positions[:, 0]
    rates_axis.plot(clock_positions, true_rates, "o-", color="lightcoral", label="True")
    rates_axis.plot(
        clock_positions, predicted, "s--", color="steelblue", label="Inferred"
    )
    rates_axis.set(xlabel="clock position", ylabel="tick rate", title="Clock rates")
    rates_axis.legend()

    history = particle_filter.history[1:]
    means = np.array(
        [
            np.average(state.particles, weights=state.weights, axis=0)
            for state in history
        ]
    )
    stds = np.array(
        [
            np.sqrt(
                np.average(
                    (state.particles - mean) ** 2,
                    weights=state.weights,
                    axis=0,
                )
            )
            for state, mean in zip(history, means, strict=True)
        ]
    )
    steps = np.arange(1, len(history) + 1)
    curves = (
        ("mu", TRUE_MU, "tab:blue"),
        ("sigma", TRUE_SIGMA, "tab:green"),
        ("A", TRUE_AMPLITUDE, "tab:orange"),
    )
    for index, (label, truth, color) in enumerate(curves):
        convergence_axis.plot(steps, means[:, index], color=color, label=f"{label} est")
        convergence_axis.fill_between(
            steps,
            means[:, index] - stds[:, index],
            means[:, index] + stds[:, index],
            alpha=0.15,
            color=color,
        )
        convergence_axis.axhline(truth, color=color, ls="--", alpha=0.5)
    convergence_axis.set(xlabel="Observation #", title="Convergence")
    convergence_axis.legend(fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _run_demo(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(SEED)
    clock_array = ClockArray(
        np.array([[position] for position in CLOCK_POSITIONS]),
        track_offset=TRACK_OFFSET,
    )
    true_params = np.array([TRUE_MU, TRUE_SIGMA, TRUE_AMPLITUDE])
    true_rates = clock_rates_density_gaussian(true_params, clock_array)
    print(f"True density: mu={TRUE_MU}, sigma={TRUE_SIGMA}, amplitude={TRUE_AMPLITUDE}")
    print(f"True rates: {true_rates}\n")
    particle_filter = build_density_filter(clock_array, rng, n_particles=args.particles)
    for step in range(args.observations):
        noisy_rates = add_clock_noise(true_rates, NOISE_STD, rng)
        particle_filter.update(Observation(noisy_rates, float(step)))
        if (step + 1) % 20 == 0:
            estimate = particle_filter.estimate()
            print(f"After {step + 1} observations:")
            print(_format_estimate("mu   ", estimate, 0, TRUE_MU))
            print(_format_estimate("sigma", estimate, 1, TRUE_SIGMA))
            print(_format_estimate("A    ", estimate, 2, TRUE_AMPLITUDE))
            print(f"  ESS   = {estimate['ess']:.0f} / {args.particles}\n")
    estimate = particle_filter.estimate()
    print("Final estimate:")
    print(_format_estimate("mu   ", estimate, 0, TRUE_MU))
    print(_format_estimate("sigma", estimate, 1, TRUE_SIGMA))
    print(_format_estimate("A    ", estimate, 2, TRUE_AMPLITUDE))
    print(f"  ESS   = {estimate['ess']:.0f} / {args.particles}")
    _save_summary(args.output, clock_array, true_rates, particle_filter, estimate)
    print(f"Saved: {args.output}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the demo using command-line arguments from *argv*."""
    args = build_parser().parse_args(argv)
    _run_demo(args)
    return 0


if __name__ == "__main__":
    main()
