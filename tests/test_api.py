"""Tests for the public library API."""

import numpy as np
import pytest

from clocks.api import infer, simulate, simulate_and_infer
from clocks.config import InferenceConfig, NoiseConfig, PriorConfig, SimulationConfig
from clocks.results import SimulationResult
from clocks.types import ClockArray, MassConfig, Observation


def _make_clock_array() -> ClockArray:
    return ClockArray(
        positions=np.array([[-6.0], [-3.0], [0.0], [3.0], [6.0]]),
        track_offset=1.0,
    )


def _make_ground_truth() -> MassConfig:
    return MassConfig(
        positions=np.array([[-2.0], [4.5]]),
        masses=np.array([0.6, 0.4]),
    )


def _make_model_comparison_ground_truth() -> MassConfig:
    return MassConfig(
        positions=np.array([[-2.0], [3.0]]),
        masses=np.array([0.6, 0.4]),
    )


def _make_noise() -> NoiseConfig:
    return NoiseConfig(observation_std=0.005)


def _make_prior() -> PriorConfig:
    return PriorConfig(position_range=(-8.0, 8.0), mass_range=(0.1, 2.0))


def _make_simulation_config(
    n_observations: int = 5,
    seed: int = 42,
    ground_truth: MassConfig | None = None,
) -> SimulationConfig:
    return SimulationConfig(
        clock_array=_make_clock_array(),
        ground_truth=ground_truth or _make_ground_truth(),
        noise=_make_noise(),
        n_observations=n_observations,
        seed=seed,
    )


def _make_inference_config(
    n_masses: int | tuple[int, ...] = 2,
    n_particles: int = 400,
    seed: int = 42,
) -> InferenceConfig:
    return InferenceConfig(
        clock_array=_make_clock_array(),
        noise=_make_noise(),
        prior=_make_prior(),
        n_particles=n_particles,
        n_masses=n_masses,
        jitter_std=0.02,
        seed=seed,
    )


def test_simulation_result_exposes_ground_truth() -> None:
    result = SimulationResult(
        clock_array=_make_clock_array(),
        ground_truth=_make_ground_truth(),
        true_rates=np.array([0.98, 0.97, 0.96, 0.97, 0.98]),
        observations=[
            Observation(rates=np.array([0.98, 0.97, 0.96, 0.97, 0.98]), time=0.0)
        ],
        noise=_make_noise(),
        seed=42,
    )

    np.testing.assert_allclose(result.ground_truth.masses, [0.6, 0.4])
    assert result.clock_array.positions.shape == (5, 1)


def test_inference_config_rejects_nonpositive_particles() -> None:
    with pytest.raises(ValueError, match="n_particles must be > 0"):
        InferenceConfig(
            clock_array=_make_clock_array(),
            noise=_make_noise(),
            prior=_make_prior(),
            n_particles=0,
            n_masses=2,
        )


def test_simulation_result_to_dict_serializes_arrays() -> None:
    result = SimulationResult(
        clock_array=_make_clock_array(),
        ground_truth=_make_ground_truth(),
        true_rates=np.array([0.98, 0.97, 0.96, 0.97, 0.98]),
        observations=[
            Observation(rates=np.array([0.98, 0.97, 0.96, 0.97, 0.98]), time=0.0)
        ],
        noise=_make_noise(),
        seed=42,
    )

    payload = result.to_dict()

    assert payload["ground_truth"]["masses"] == [0.6, 0.4]
    assert payload["clock_array"]["positions"][0] == [-6.0]
    assert payload["observations"][0]["rates"][2] == 0.96


def test_simulate_returns_observations_and_ground_truth() -> None:
    result = simulate(_make_simulation_config())

    assert len(result.observations) == 5
    assert result.ground_truth.masses.shape == (2,)
    assert result.observations[0].rates.shape == (5,)


def test_infer_multi_mass_returns_summary_history() -> None:
    simulation = simulate(_make_simulation_config(n_observations=8, seed=123))

    result = infer(
        simulation.observations,
        _make_inference_config(n_masses=2, seed=123),
    )

    assert result.posterior_mean.shape == (4,)
    assert result.posterior_std.shape == (4,)
    assert len(result.history) == len(simulation.observations)


def test_infer_model_comparison_returns_model_probabilities() -> None:
    simulation = simulate(
        _make_simulation_config(
            n_observations=40,
            seed=42,
            ground_truth=_make_model_comparison_ground_truth(),
        )
    )

    result = infer(
        simulation.observations,
        _make_inference_config(n_masses=(1, 2, 3), n_particles=1500, seed=42),
    )

    assert set(result.posterior_by_model) == {1, 2, 3}
    assert result.best_model == 2
    assert len(result.history) == len(simulation.observations)


def test_simulate_and_infer_preserves_simulation_output() -> None:
    result = simulate_and_infer(
        _make_simulation_config(n_observations=6, seed=99),
        _make_inference_config(n_masses=2, seed=99),
    )

    assert result.simulation is not None
    assert result.simulation.ground_truth.masses.shape == (2,)
