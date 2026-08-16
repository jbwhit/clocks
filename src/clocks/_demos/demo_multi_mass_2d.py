"""Infer two ordered point masses from a random two-dimensional clock array."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from clocks._demos._common import add_common_arguments
from clocks._scenarios import (
    N_CLOCKS,
    N_OBSERVATIONS,
    N_PARTICLES,
    NOISE_STD,
    TRACK_OFFSET,
    TRUE_MASSES,
    TRUE_POSITIONS,
    generate_random_clocks,
)
from clocks.api import build_particle_filter, simulate
from clocks.config import InferenceConfig, NoiseConfig, PriorConfig, SimulationConfig
from clocks.types import ClockArray, MassConfig
from clocks.viz import animate_inference_multi_2d

SEED = 11
OUTPUT_PATH = Path("output/demo_multi_mass_2d.gif")


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
    rng = np.random.default_rng(SEED)
    mass_config = MassConfig(TRUE_POSITIONS, TRUE_MASSES)
    clock_positions = generate_random_clocks(
        N_CLOCKS,
        rng,
        exclude=[tuple(position) for position in TRUE_POSITIONS],
    )
    clock_array = ClockArray(clock_positions, track_offset=TRACK_OFFSET)
    simulation = simulate(
        SimulationConfig(
            clock_array=clock_array,
            ground_truth=mass_config,
            noise=NoiseConfig(NOISE_STD),
            n_observations=args.observations,
            seed=SEED,
        )
    )
    print(
        f"True masses: ({TRUE_POSITIONS[0][0]},{TRUE_POSITIONS[0][1]}) "
        f"M={TRUE_MASSES[0]}, ({TRUE_POSITIONS[1][0]},{TRUE_POSITIONS[1][1]}) "
        f"M={TRUE_MASSES[1]}"
    )
    print(f"Clocks: {N_CLOCKS} randomly placed")
    print(f"True rates: {simulation.true_rates}")
    particle_filter = build_particle_filter(
        InferenceConfig(
            clock_array=clock_array,
            noise=NoiseConfig(NOISE_STD),
            prior=PriorConfig((-8.0, 8.0), (0.005, 0.15)),
            n_particles=args.particles,
            n_masses=2,
            seed=SEED,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating animation → {args.output}")
    animate_inference_multi_2d(
        clock_array=clock_array,
        mass_config=mass_config,
        observations=simulation.observations,
        pf=particle_filter,
        output_path=args.output,
        fps=4,
    )
    estimate = particle_filter.estimate()
    print(f"\nFinal estimate after {args.observations} observations:")
    labels = ("x1", "y1", "x2", "y2", "M1", "M2")
    truths = (*TRUE_POSITIONS[0], *TRUE_POSITIONS[1], *TRUE_MASSES)
    for index, (label, truth) in enumerate(zip(labels, truths, strict=True)):
        print(
            f"  {label} = {estimate['mean'][index]:.3f} ± "
            f"{estimate['std'][index]:.3f}  (true: {truth})"
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
