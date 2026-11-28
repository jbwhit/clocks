"""Infer two ordered point masses on a one-dimensional track."""

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
from clocks.physics import clock_rates
from clocks.types import ClockArray, MassConfig

TRUE_X1 = -2.0
TRUE_X2 = 4.5
TRUE_M1 = 0.045
TRUE_M2 = 0.030
CLOCK_POSITIONS = [-6.0, -3.0, 0.0, 3.0, 6.0]
TRACK_OFFSET = 1.0
N_OBSERVATIONS = 80
NOISE_STD = 0.005
N_PARTICLES = 3000
SEED = 42


def scenario() -> Scenario:
    """The packaged inputs of this demo, read from the module constants."""
    clock_array = ClockArray(
        np.array([[position] for position in CLOCK_POSITIONS]),
        track_offset=TRACK_OFFSET,
    )
    params = np.array([TRUE_X1, TRUE_X2, TRUE_M1, TRUE_M2])
    return Scenario(
        demo="demo_multi_mass",
        clock_array=clock_array,
        true_params=params,
        true_rates=clock_rates(
            MassConfig(params[:2].reshape(2, 1), params[2:]), clock_array
        ),
        param_names=("x₁", "x₂", "M₁", "M₂"),
        prior_ranges=((-8.0, 8.0), (-8.0, 8.0), (0.005, 0.15), (0.005, 0.15)),
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
        predictions=True,
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
