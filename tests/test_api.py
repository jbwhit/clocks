"""Tests for the public library API."""

import numpy as np
import pytest

from clocks.config import InferenceConfig, NoiseConfig, PriorConfig
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


def test_simulation_result_exposes_ground_truth() -> None:
    result = SimulationResult(
        clock_array=_make_clock_array(),
        ground_truth=_make_ground_truth(),
        true_rates=np.array([0.98, 0.97, 0.96, 0.97, 0.98]),
        observations=[
            Observation(rates=np.array([0.98, 0.97, 0.96, 0.97, 0.98]), time=0.0)
        ],
        noise=NoiseConfig(observation_std=0.005),
        seed=42,
    )

    np.testing.assert_allclose(result.ground_truth.masses, [0.6, 0.4])
    assert result.clock_array.positions.shape == (5, 1)


def test_inference_config_rejects_nonpositive_particles() -> None:
    with pytest.raises(ValueError, match="n_particles must be > 0"):
        InferenceConfig(
            clock_array=_make_clock_array(),
            noise=NoiseConfig(observation_std=0.005),
            prior=PriorConfig(position_range=(-8.0, 8.0), mass_range=(0.1, 2.0)),
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
        noise=NoiseConfig(observation_std=0.005),
        seed=42,
    )

    payload = result.to_dict()

    assert payload["ground_truth"]["masses"] == [0.6, 0.4]
    assert payload["clock_array"]["positions"][0] == [-6.0]
    assert payload["observations"][0]["rates"][2] == 0.96
