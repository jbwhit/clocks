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
from clocks._figures import density_figure, trace_decimals
from clocks._replay import (
    Replay,
    Scenario,
    filter_controls,
    record_filter,
    scenario_meta,
    validate_replay,
)
from clocks._rng import study_generator
from clocks._style import save_png, style
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


def scenario() -> Scenario:
    """The packaged inputs of this demo, read from the module constants."""
    clock_array = ClockArray(
        np.array([[position] for position in CLOCK_POSITIONS]),
        track_offset=TRACK_OFFSET,
    )
    params = np.array([TRUE_MU, TRUE_SIGMA, TRUE_AMPLITUDE])
    return Scenario(
        demo="demo_density",
        clock_array=clock_array,
        true_params=params,
        true_rates=clock_rates_density_gaussian(params, clock_array),
        param_names=("μ", "σ", "A"),
        prior_ranges=(MU_RANGE, SIGMA_RANGE, AMPLITUDE_RANGE),
        seed=SEED,
        noise_std=NOISE_STD,
        n_observations=N_OBSERVATIONS,
        n_particles=N_PARTICLES,
    )


def record(
    *, n_observations: int = N_OBSERVATIONS, n_particles: int = N_PARTICLES
) -> Replay:
    """Run the density inference once and return it as a validated replay.

    Every reading is drawn first, from its own ``study_generator`` of the
    scenario's seed, and the filter runs on them with a second one, so the
    readings do not depend on how many random numbers the sampler consumed. The
    ``demo-density`` command runs exactly this and draws the result.
    """
    packaged = scenario()
    reading_rng = study_generator(packaged.seed)
    readings = [
        add_clock_noise(packaged.true_rates, packaged.noise_std, reading_rng)
        for _ in range(n_observations)
    ]
    observations = [
        Observation(rates, float(step)) for step, rates in enumerate(readings)
    ]
    particle_filter = build_density_filter(
        packaged.clock_array, study_generator(packaged.seed), n_particles=n_particles
    )
    replay = record_filter(
        particle_filter,
        observations,
        meta=scenario_meta(
            packaged,
            kind="filter",
            n_observations=n_observations,
            n_particles=n_particles,
            controls=filter_controls(particle_filter),
        ),
        arrays={
            "clock_positions": packaged.clock_array.positions,
            "true_params": packaged.true_params,
            "true_rates": packaged.true_rates,
        },
    )
    validate_replay(replay)
    return replay


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


def _format_estimate(
    label: str,
    mean: float,
    std: float,
    truth: float,
    prior_range: Sequence[float],
) -> str:
    digits = trace_decimals(prior_range)
    return f"  {label} = {mean:.{digits}f} ± {std:.{digits}f}  (true: {truth})"


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


def _run_demo(args: argparse.Namespace) -> None:
    replay = record(n_observations=args.observations, n_particles=args.particles)
    arrays, meta = replay.arrays, replay.meta
    n_particles = meta["n_particles"]
    mu, sigma, amplitude = (float(value) for value in arrays["true_params"])
    print(f"True density: mu={mu}, sigma={sigma}, amplitude={amplitude}")
    print(f"True rates: {arrays['true_rates']}\n")
    for row in range(20, meta["n_observations"] + 1, 20):
        print(f"After {row} observations:")
        _print_estimate(replay, row, n_particles)
        print()
    print("Final estimate:")
    _print_estimate(replay, -1, n_particles)
    with style("light") as colors:
        fig = density_figure(replay, colors)
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            save_png(fig, args.output)
        finally:
            plt.close(fig)
    print(f"Saved: {args.output}")


def _print_estimate(replay: Replay, row: int, n_particles: int) -> None:
    mean = replay.arrays["mean"][row]
    std = np.sqrt(np.diagonal(replay.arrays["cov"][row]))
    truth = (float(value) for value in replay.arrays["true_params"])
    labels = ("mu   ", "sigma", "A    ")
    ranges = replay.meta["prior_ranges"]
    for index, (label, true_value) in enumerate(zip(labels, truth, strict=True)):
        print(
            _format_estimate(
                label, float(mean[index]), float(std[index]), true_value, ranges[index]
            )
        )
    print(f"  ESS   = {replay.arrays['ess'][row]:.0f} / {n_particles}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the demo using command-line arguments from *argv*."""
    args = build_parser().parse_args(argv)
    _run_demo(args)
    return 0


if __name__ == "__main__":
    main()
