"""End-to-end one-mass inference on a one-dimensional track."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from clocks._demos._common import add_common_arguments
from clocks.api import build_particle_filter, simulate
from clocks.config import InferenceConfig, NoiseConfig, PriorConfig, SimulationConfig
from clocks.types import ClockArray, MassConfig
from clocks.viz import animate_inference

TRUE_X = 2.5
TRUE_M = 0.10
CLOCK_POSITIONS = [-5.0, 0.0, 5.0]
TRACK_OFFSET = 1.0
N_OBSERVATIONS = 50
NOISE_STD = 0.005
N_PARTICLES = 1000
SEED = 42
OUTPUT_PATH = Path("output/demo_1d.gif")


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


def _run_demo(args: argparse.Namespace) -> None:
    mass_config = MassConfig(np.array([[TRUE_X]]), np.array([TRUE_M]))
    clock_array = ClockArray(
        np.array([[position] for position in CLOCK_POSITIONS]),
        track_offset=TRACK_OFFSET,
    )
    simulation = simulate(
        SimulationConfig(
            clock_array=clock_array,
            ground_truth=mass_config,
            noise=NoiseConfig(NOISE_STD),
            n_observations=args.observations,
            seed=SEED,
        )
    )
    print(f"True mass: x={TRUE_X}, M={TRUE_M}")
    print(f"True rates: {simulation.true_rates}")
    particle_filter = build_particle_filter(
        InferenceConfig(
            clock_array=clock_array,
            noise=NoiseConfig(NOISE_STD),
            prior=PriorConfig((-8.0, 8.0), (0.005, 0.15)),
            n_particles=args.particles,
            n_masses=1,
            seed=SEED,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating animation → {args.output}")
    animate_inference(
        clock_array=clock_array,
        mass_config=mass_config,
        observations=simulation.observations,
        pf=particle_filter,
        output_path=args.output,
        fps=4,
    )
    estimate = particle_filter.estimate()
    print(f"\nFinal estimate after {args.observations} observations:")
    print(
        f"  x = {estimate['mean'][0]:.3f} ± {estimate['std'][0]:.3f}  (true: {TRUE_X})"
    )
    print(
        f"  M = {estimate['mean'][1]:.3f} ± {estimate['std'][1]:.3f}  (true: {TRUE_M})"
    )
    print(f"  ESS = {estimate['ess']:.0f} / {args.particles}")
    print(f"\nSaved to {args.output}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the demo using command-line arguments from *argv*."""
    args = build_parser().parse_args(argv)
    _run_demo(args)
    return 0


if __name__ == "__main__":
    main()
