"""Compare one-, two-, and three-mass models by their SMC evidence."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from clocks._demos._common import add_common_arguments
from clocks._replay import (
    Replay,
    Scenario,
    filter_controls,
    scenario_meta,
    validate_replay,
)
from clocks.api import build_model_comparison, infer, simulate
from clocks.config import InferenceConfig, NoiseConfig, PriorConfig, SimulationConfig
from clocks.inference import ModelComparison
from clocks.physics import clock_rates
from clocks.types import ClockArray, MassConfig, Observation
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
MODEL_KS = (1, 2, 3)
OUTPUT_PATH = Path("output/demo_model_comparison.gif")


def scenario() -> Scenario:
    """The packaged inputs of this demo, read from the module constants."""
    clock_array = _clock_array()
    params = np.array([TRUE_X1, TRUE_X2, TRUE_M1, TRUE_M2])
    return Scenario(
        demo="demo_model_comparison",
        clock_array=clock_array,
        true_params=params,
        true_rates=clock_rates(
            MassConfig(params[:2].reshape(2, 1), params[2:]), clock_array
        ),
        param_names=("x₁", "x₂", "M₁", "M₂"),
        prior_ranges=((-8.0, 8.0), (0.005, 0.15)),
        seed=SEED,
        noise_std=NOISE_STD,
        n_observations=N_OBSERVATIONS,
        n_particles=N_PARTICLES,
    )


def record(
    *, n_observations: int = N_OBSERVATIONS, n_particles: int = N_PARTICLES
) -> Replay:
    """Run the three-candidate comparison once and return it as a replay."""
    packaged = scenario()
    simulation = simulate(
        SimulationConfig(
            clock_array=packaged.clock_array,
            ground_truth=MassConfig(
                packaged.true_params[:2].reshape(2, 1), packaged.true_params[2:]
            ),
            noise=NoiseConfig(packaged.noise_std),
            n_observations=n_observations,
            seed=packaged.seed,
        )
    )
    comparison = build_model_comparison(
        InferenceConfig(
            clock_array=packaged.clock_array,
            noise=NoiseConfig(packaged.noise_std),
            prior=PriorConfig(*packaged.prior_ranges),
            n_particles=n_particles,
            n_masses=MODEL_KS,
            seed=packaged.seed,
        )
    )
    arrays = _model_arrays(comparison, simulation.observations, n_particles)
    arrays.update(
        clock_positions=packaged.clock_array.positions,
        true_params=packaged.true_params,
        true_rates=packaged.true_rates,
        observations=np.stack([o.rates for o in simulation.observations]),
    )
    replay = Replay(
        meta=scenario_meta(
            packaged,
            kind="model_comparison",
            n_observations=n_observations,
            n_particles=n_particles,
            controls=filter_controls(comparison.filters[1]),
            true_k=2,
        ),
        arrays=arrays,
    )
    validate_replay(replay)
    return replay


def _model_arrays(
    comparison: ModelComparison, observations: list[Observation], n_particles: int
) -> dict[str, np.ndarray]:
    """Update ``comparison`` observation by observation and collect its history.

    Row 0 of each per-observation array is the prior, written as exact values.
    """
    ks = comparison.k_values
    prior = np.array([comparison.model_prior[k] for k in ks])
    posterior, log_evidence = [prior], [np.zeros(len(ks))]
    ess = [np.full(len(ks), float(n_particles))]
    for observation in observations:
        comparison.update(observation)
        evidence = comparison.evidence()
        posterior.append(np.array([evidence["posterior"][k] for k in ks]))
        log_evidence.append(np.array([evidence["log_evidence"][k] for k in ks]))
        ess.append(np.array([comparison.estimate(k)["ess"] for k in ks]))
    arrays = {
        "model_ks": np.array(ks, dtype=np.int64),
        "model_prior": prior,
        "model_posterior": np.stack(posterior),
        "model_log_evidence": np.stack(log_evidence),
        "model_ess": np.stack(ess),
    }
    for k in ks:
        estimate = comparison.estimate(k)
        arrays[f"final_mean_k{k}"] = np.asarray(estimate["mean"], dtype=np.float64)
        arrays[f"final_std_k{k}"] = np.asarray(estimate["std"], dtype=np.float64)
    return arrays


def _clock_array() -> ClockArray:
    return ClockArray(
        np.array([[position] for position in CLOCK_POSITIONS]),
        track_offset=TRACK_OFFSET,
    )


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
