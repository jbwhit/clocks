"""Compare one-, two-, and three-mass models by their SMC evidence."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from clocks._demos._common import add_common_arguments
from clocks.api import build_model_comparison, infer, simulate
from clocks.config import InferenceConfig, NoiseConfig, PriorConfig, SimulationConfig
from clocks.types import ClockArray, MassConfig
from clocks.viz import animate_model_comparison

TRUE_X1 = -2.0
TRUE_X2 = 3.0
TRUE_M1 = 0.045
TRUE_M2 = 0.030
CLOCK_POSITIONS = [-6.0, -3.0, 0.0, 3.0, 6.0]
TRACK_OFFSET = 1.0
N_OBSERVATIONS = 80
NOISE_STD = 0.005
N_PARTICLES = 2000
SEED = 42
OUTPUT_PATH = Path("output/demo_model_comparison.gif")


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
    mass_config = MassConfig(
        np.array([[TRUE_X1], [TRUE_X2]]), np.array([TRUE_M1, TRUE_M2])
    )
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
    inference_config = InferenceConfig(
        clock_array=clock_array,
        noise=NoiseConfig(NOISE_STD),
        prior=PriorConfig((-8.0, 8.0), (0.005, 0.15)),
        n_particles=args.particles,
        n_masses=(1, 2, 3),
        seed=SEED,
    )
    result = infer(simulation.observations, inference_config)
    print(
        f"True model: K=2 masses at x=[{TRUE_X1}, {TRUE_X2}], M=[{TRUE_M1}, {TRUE_M2}]"
    )
    print(f"True rates: {simulation.true_rates}\n")
    for step, posterior in enumerate(result.history, start=1):
        if step % 20 == 0:
            print(f"After {step} observations:")
            for model in sorted(posterior):
                print(f"  K={model}: posterior={posterior[model]:.4f}")
            print()
    estimate = result.result_by_model[result.best_model]
    print(f"MAP model: K={result.best_model}")
    print(f"Estimate: {estimate.posterior_mean}")
    print(f"Std:      {estimate.posterior_std}")
    print(f"ESS:      {estimate.ess:.0f} / {args.particles}")
    print("\nGenerating model comparison GIF...")
    comparison = build_model_comparison(inference_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    animate_model_comparison(
        clock_array=clock_array,
        mass_config=mass_config,
        observations=simulation.observations,
        model_comparison=comparison,
        output_path=args.output,
    )
    print(f"Saved: {args.output}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the demo using command-line arguments from *argv*."""
    args = build_parser().parse_args(argv)
    _run_demo(args)
    return 0


if __name__ == "__main__":
    main()
