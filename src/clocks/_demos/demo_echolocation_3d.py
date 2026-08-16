"""Localize an exterior mass from contrast-space 3-D clock readings."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from clocks._demos._common import add_common_arguments
from clocks._scenarios import (
    ECHO_M_TRUE,
    ECHO_N_OBSERVATIONS,
    ECHO_N_PARTICLES,
    build_echolocation_filter,
    build_head_lattice,
    echo_mass_config,
    echo_mass_position,
    make_echo_observations,
    validate_echo_geometry,
)
from clocks.viz import animate_echolocation

DEMO_RANGE_R = 2.0
DEMO_SEED = 4
OUTPUT_PATH = Path("output/demo_echolocation_3d.gif")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without running the simulation."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(
        parser,
        output=OUTPUT_PATH,
        observations=ECHO_N_OBSERVATIONS,
        particles=ECHO_N_PARTICLES,
    )
    return parser


def _run_demo(args: argparse.Namespace) -> None:
    clock_array = build_head_lattice()
    validate_echo_geometry(DEMO_RANGE_R, ECHO_M_TRUE, clock_array)
    mass_config = echo_mass_config(DEMO_RANGE_R)
    truth = np.append(echo_mass_position(DEMO_RANGE_R), ECHO_M_TRUE)
    print(
        f"True mass: M={ECHO_M_TRUE} at {truth[:3].round(2)} "
        f"({DEMO_RANGE_R} circumradii)"
    )
    _, display_observations, filter_observations = make_echo_observations(
        DEMO_SEED,
        DEMO_RANGE_R,
        n_observations=args.observations,
    )
    particle_filter = build_echolocation_filter(DEMO_SEED, n_particles=args.particles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating animation → {args.output}")
    animate_echolocation(
        clock_array=clock_array,
        mass_config=mass_config,
        observations=display_observations,
        filter_observations=filter_observations,
        pf=particle_filter,
        output_path=args.output,
        fps=4,
    )
    estimate = particle_filter.estimate()
    print(f"\nFinal estimate after {args.observations} observations:")
    for index, label in enumerate(("x", "y", "z", "M")):
        print(
            f"  {label} = {estimate['mean'][index]:.3f} ± "
            f"{estimate['std'][index]:.3f}  (true: {truth[index]:.3f})"
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
