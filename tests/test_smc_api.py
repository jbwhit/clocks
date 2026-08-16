"""Public API tests for rigorous SMC configuration and results."""

import numpy as np
import pytest

from clocks import (
    ClockArray,
    InferenceConfig,
    NoiseConfig,
    Observation,
    PriorConfig,
    build_model_comparison,
    build_particle_filter,
    infer,
)
from clocks._support import point_mass_support_mask


def _config(*, n_masses: int | tuple[int, ...] = 2, seed: int = 23) -> InferenceConfig:
    return InferenceConfig(
        clock_array=ClockArray(np.linspace(-6.0, 6.0, 8), track_offset=3.0),
        noise=NoiseConfig(0.005),
        prior=PriorConfig((-8.0, 8.0), (0.005, 0.15)),
        n_particles=300,
        n_masses=n_masses,
        resampling="residual",
        ess_target=0.75,
        rejuvenation_steps=3,
        proposal_scale=1.7,
        seed=seed,
    )


def test_inference_controls_are_validated_and_plumbed() -> None:
    config = _config()
    particle_filter = build_particle_filter(config)

    assert particle_filter.resampling == "residual"
    assert particle_filter.ess_target == 0.75
    assert particle_filter.rejuvenation_steps == 3
    assert particle_filter.proposal_scale == 1.7

    for field, value in [
        ("ess_target", 0.0),
        ("rejuvenation_steps", True),
        ("proposal_scale", np.nan),
        ("resampling", "bogus"),
    ]:
        kwargs = {field: value}
        with pytest.raises(ValueError, match=field):
            InferenceConfig(
                clock_array=config.clock_array,
                noise=config.noise,
                prior=config.prior,
                n_particles=10,
                n_masses=1,
                **kwargs,
            )


def test_api_initial_and_moved_particles_stay_in_actual_support() -> None:
    config = _config()
    particle_filter = build_particle_filter(config)
    observations = [Observation(np.full(8, 0.99), float(t)) for t in range(3)]

    states = [particle_filter.state]
    states.extend(particle_filter.update(observation) for observation in observations)
    for state in states:
        assert np.all(
            point_mass_support_mask(
                state.particles,
                n_masses=2,
                n_dims=1,
                clock_array=config.clock_array,
                position_range=config.prior.position_range,
                mass_range=config.prior.mass_range,
            )
        )


def test_model_comparison_builder_uses_independent_repeatable_streams() -> None:
    first = build_model_comparison(_config(n_masses=(1, 2), seed=8))
    second = build_model_comparison(_config(n_masses=(1, 2), seed=8))

    assert set(first.filters) == {1, 2}
    for key in first.filters:
        np.testing.assert_array_equal(
            first.filters[key].state.particles, second.filters[key].state.particles
        )
    assert not np.array_equal(
        first.filters[1].state.particles[:, 0],
        first.filters[2].state.particles[:, 0],
    )


def test_fixed_k_result_serializes_evidence_and_diagnostics() -> None:
    config = _config(n_masses=1)
    observations = [Observation(np.full(8, 0.99), float(t)) for t in range(2)]
    result = infer(observations, config)
    payload = result.to_dict()

    assert result.log_evidence == payload["log_evidence"]
    assert len(result.history) == 2
    assert payload["history"][0]["diagnostics"]["tempering_stages"] >= 1
    assert "mh_acceptances" in payload["history"][0]["diagnostics"]


def test_infer_rejects_wrong_observation_channel_count() -> None:
    with pytest.raises(ValueError, match="channel"):
        infer([Observation(np.ones(7), 0.0)], _config(n_masses=1))


def test_impossible_conditional_prior_names_weak_field_policy() -> None:
    config = InferenceConfig(
        clock_array=ClockArray([[0.0]], track_offset=1.0),
        noise=NoiseConfig(0.01),
        prior=PriorConfig((-0.001, 0.001), (0.06, 0.07)),
        n_particles=3,
        n_masses=1,
        seed=4,
    )

    with pytest.raises(ValueError, match="conditional prior.*weak-field"):
        build_particle_filter(config)
