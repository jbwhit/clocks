"""Infer two ordered point masses from a random two-dimensional clock array."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")

import numpy as np

from clocks._clips import final_frame
from clocks._demos._common import (
    add_common_arguments,
    add_video_argument,
    make_parser,
    print_estimate,
    print_truth,
    run_command,
    write_outputs,
)
from clocks._replay import (
    Replay,
    Scenario,
    record_point_masses,
)
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
from clocks.physics import clock_rates
from clocks.types import ClockArray, MassConfig

SEED = 11


def scenario() -> Scenario:
    """The packaged inputs of this demo, read from the module constants."""
    clock_array = ClockArray(
        generate_random_clocks(
            N_CLOCKS,
            np.random.default_rng(SEED),
            exclude=[tuple(position) for position in TRUE_POSITIONS],
        ),
        track_offset=TRACK_OFFSET,
    )
    params = np.concatenate([TRUE_POSITIONS.reshape(-1), TRUE_MASSES])
    return Scenario(
        demo="demo_multi_mass_2d",
        clock_array=clock_array,
        true_params=params,
        true_rates=clock_rates(
            MassConfig(params[:4].reshape(2, 2), params[4:]), clock_array
        ),
        param_names=("x₁", "y₁", "x₂", "y₂", "M₁", "M₂"),
        prior_ranges=((-8.0, 8.0),) * 4 + ((0.005, 0.15),) * 2,
        seed=SEED,
        noise_std=NOISE_STD,
        n_observations=N_OBSERVATIONS,
        n_particles=N_PARTICLES,
    )


def record(
    *, n_observations: int = N_OBSERVATIONS, n_particles: int = N_PARTICLES
) -> Replay:
    """Run the packaged inference once and return it as a validated replay."""
    return record_point_masses(
        scenario(),
        n_observations=n_observations,
        n_particles=n_particles,
        predictions=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = make_parser(__doc__)
    add_common_arguments(parser, observations=N_OBSERVATIONS, particles=N_PARTICLES)
    add_video_argument(parser)
    return parser


def _run_demo(args: argparse.Namespace) -> None:
    replay = record(n_observations=args.observations, n_particles=args.particles)
    print_truth(replay)
    print_estimate(replay)
    write_outputs(
        replay, args.output_dir, final_frame, theme=args.theme, encoder=args.encoder
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the demo using command-line arguments from *argv*."""
    return run_command(build_parser(), _run_demo, argv)


if __name__ == "__main__":
    main()
