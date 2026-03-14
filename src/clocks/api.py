"""Public end-to-end API for clocks simulation and inference."""

import numpy as np

from clocks.config import InferenceConfig, SimulationConfig
from clocks.noise import add_clock_noise
from clocks.physics import clock_rates
from clocks.results import (
    InferenceResult,
    ModelComparisonInferenceResult,
    SimulationResult,
)
from clocks.types import Observation


def simulate(config: SimulationConfig) -> SimulationResult:
    """Generate synthetic observations from a ground-truth mass configuration."""
    rng = np.random.default_rng(config.seed)
    true_rates = clock_rates(config.ground_truth, config.clock_array)
    observations = [
        Observation(
            rates=add_clock_noise(true_rates, config.noise.observation_std, rng),
            time=float(t),
        )
        for t in range(config.n_observations)
    ]
    return SimulationResult(
        clock_array=config.clock_array,
        ground_truth=config.ground_truth,
        true_rates=true_rates,
        observations=observations,
        noise=config.noise,
        seed=config.seed,
    )


def infer(
    observations: list[Observation], config: InferenceConfig
) -> InferenceResult | ModelComparisonInferenceResult:
    """Run inference against a list of observations."""
    raise NotImplementedError("infer() is implemented in a later task")


def simulate_and_infer(
    simulation_config: SimulationConfig,
    inference_config: InferenceConfig,
) -> InferenceResult | ModelComparisonInferenceResult:
    """Generate synthetic data and immediately run inference over it."""
    raise NotImplementedError("simulate_and_infer() is implemented in a later task")
