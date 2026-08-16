"""Tests for the public library API."""

import numpy as np
import pytest

from clocks.api import (
    _make_log_prior,
    build_model_comparison,
    build_particle_filter,
    infer,
    simulate,
    simulate_and_infer,
)
from clocks.config import InferenceConfig, NoiseConfig, PriorConfig, SimulationConfig
from clocks.results import InferenceResult, SimulationResult
from clocks.types import ClockArray, MassConfig, Observation


def _make_clock_array() -> ClockArray:
    return ClockArray(
        positions=np.array([[-6.0], [-3.0], [0.0], [3.0], [6.0]]),
        track_offset=1.0,
    )


def _make_ground_truth() -> MassConfig:
    return MassConfig(
        positions=np.array([[-2.0], [4.5]]),
        masses=np.array([0.045, 0.030]),
    )


def _make_model_comparison_ground_truth() -> MassConfig:
    return MassConfig(
        positions=np.array([[-2.0], [3.0]]),
        masses=np.array([0.045, 0.030]),
    )


def _make_noise() -> NoiseConfig:
    return NoiseConfig(observation_std=0.005)


def _make_prior() -> PriorConfig:
    return PriorConfig(position_range=(-8.0, 8.0), mass_range=(0.005, 0.15))


def test_fixed_k_initial_particles_use_the_actual_conditional_prior() -> None:
    config = InferenceConfig(
        clock_array=_make_clock_array(),
        noise=_make_noise(),
        prior=PriorConfig(position_range=(-8.0, 8.0), mass_range=(0.005, 0.04)),
        n_particles=500,
        n_masses=2,
        seed=12,
    )
    particle_filter = build_particle_filter(config)
    log_prior = _make_log_prior(config, n_masses=2, n_dims=1)

    assert np.all(np.isfinite(log_prior(particle_filter.state.particles)))
    masses = particle_filter.state.particles[:, 2:]
    assert np.all(masses >= 0.005)
    assert np.all(masses <= 0.04)
    assert np.all(
        particle_filter.state.particles[:, 0] < particle_filter.state.particles[:, 1]
    )
    assert particle_filter.log_prior_density is not None


def test_fixed_k_particles_remain_in_conditional_support_after_updates() -> None:
    config = InferenceConfig(
        clock_array=_make_clock_array(),
        noise=NoiseConfig(observation_std=1e-4),
        prior=PriorConfig(position_range=(-8.0, 8.0), mass_range=(0.005, 0.04)),
        n_particles=400,
        n_masses=2,
        proposal_scale=5.0,
        seed=13,
    )
    particle_filter = build_particle_filter(config)
    log_prior = _make_log_prior(config, n_masses=2, n_dims=1)
    observation = Observation(np.full(5, 0.99), time=0.0)

    particle_filter.update(observation)

    assert np.all(np.isfinite(log_prior(particle_filter.state.particles)))


def test_impossible_conditional_prior_fails_clearly() -> None:
    config = InferenceConfig(
        clock_array=ClockArray(np.array([[0.0]]), track_offset=1.0),
        noise=_make_noise(),
        prior=PriorConfig(position_range=(-0.001, 0.001), mass_range=(0.06, 0.07)),
        n_particles=8,
        n_masses=1,
        seed=14,
    )

    with pytest.raises(ValueError, match="conditional prior.*weak-field"):
        build_particle_filter(config)


def test_model_comparison_initial_particles_use_true_support() -> None:
    comparison = build_model_comparison(
        InferenceConfig(
            clock_array=_make_clock_array(),
            noise=_make_noise(),
            prior=PriorConfig((-8.0, 8.0), (0.005, 0.04)),
            n_particles=150,
            n_masses=(1, 2, 3),
            seed=15,
        )
    )

    for particle_filter in comparison.filters.values():
        assert np.all(
            np.isfinite(
                particle_filter.log_prior_density(particle_filter.state.particles)
            )
        )


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
    proposal_scale: float = 2.38,
    rejuvenation_steps: int = 2,
    ess_target: float = 0.8,
) -> InferenceConfig:
    return InferenceConfig(
        clock_array=_make_clock_array(),
        noise=_make_noise(),
        prior=_make_prior(),
        n_particles=n_particles,
        n_masses=n_masses,
        proposal_scale=proposal_scale,
        rejuvenation_steps=rejuvenation_steps,
        ess_target=ess_target,
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

    np.testing.assert_allclose(result.ground_truth.masses, [0.045, 0.030])
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

    assert payload["ground_truth"]["masses"] == [0.045, 0.03]
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


def test_build_particle_filter_is_public_and_runs() -> None:
    simulation = simulate(_make_simulation_config(n_observations=3, seed=7))
    pf = build_particle_filter(_make_inference_config(n_masses=2, seed=7))

    assert pf.n_particles == _make_inference_config(n_masses=2).n_particles
    for obs in simulation.observations:
        pf.update(obs)
    assert pf.state.observations_seen == 3

    with pytest.raises(TypeError, match="fixed-K"):
        build_particle_filter(_make_inference_config(n_masses=(1, 2)))


def test_infer_rejects_empty_observations() -> None:
    for n_masses in (1, (1, 2)):
        config = _make_inference_config(n_masses=n_masses)
        with pytest.raises(ValueError, match="observations must not be empty"):
            infer([], config)


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


def test_inference_result_to_dict_includes_particle_state_and_simulation() -> None:
    result = simulate_and_infer(
        _make_simulation_config(n_observations=6, seed=99),
        _make_inference_config(n_masses=2, seed=99),
    )

    payload = result.to_dict()

    assert "particle_state" in payload
    assert payload["particle_state"]["observations_seen"] == 6
    assert payload["simulation"]["ground_truth"]["masses"] == [0.045, 0.03]


def test_model_comparison_result_to_dict_includes_nested_results() -> None:
    result = simulate_and_infer(
        _make_simulation_config(
            n_observations=40,
            seed=42,
            ground_truth=_make_model_comparison_ground_truth(),
        ),
        _make_inference_config(n_masses=(1, 2, 3), n_particles=1500, seed=42),
    )

    payload = result.to_dict()

    assert payload["best_model"] == 2
    assert set(payload["result_by_model"]) == {1, 2, 3}
    assert "simulation" in payload


def test_noise_config_rejects_nonpositive_std() -> None:
    with pytest.raises(ValueError, match="observation_std must be > 0"):
        NoiseConfig(observation_std=0.0)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_noise_config_rejects_nonfinite_std(value: float) -> None:
    with pytest.raises(ValueError, match="observation_std must be finite"):
        NoiseConfig(observation_std=value)


def test_prior_config_rejects_invalid_position_range() -> None:
    with pytest.raises(ValueError, match="position_range must be increasing"):
        PriorConfig(position_range=(2.0, -2.0), mass_range=(0.1, 2.0))


@pytest.mark.parametrize(
    ("position_range", "mass_range"),
    [
        ((-np.inf, 1.0), (0.1, 2.0)),
        ((-1.0, np.inf), (0.1, 2.0)),
        ((-1.0, 1.0), (np.nan, 2.0)),
        ((-1.0, 1.0), (0.1, np.inf)),
    ],
)
def test_prior_config_rejects_nonfinite_endpoints(
    position_range: tuple[float, float], mass_range: tuple[float, float]
) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        PriorConfig(position_range=position_range, mass_range=mass_range)


def test_inference_config_rejects_nonpositive_n_masses() -> None:
    with pytest.raises(ValueError, match="n_masses must be > 0"):
        InferenceConfig(
            clock_array=_make_clock_array(),
            noise=_make_noise(),
            prior=_make_prior(),
            n_particles=100,
            n_masses=0,
        )


def test_inference_config_rejects_empty_candidate_models() -> None:
    with pytest.raises(ValueError, match="n_masses candidates must not be empty"):
        InferenceConfig(
            clock_array=_make_clock_array(),
            noise=_make_noise(),
            prior=_make_prior(),
            n_particles=100,
            n_masses=(),
        )


@pytest.mark.parametrize("field", ["n_particles", "n_masses"])
def test_inference_config_rejects_bool_counts(field: str) -> None:
    kwargs = {field: True}
    with pytest.raises(ValueError, match=field):
        _make_inference_config(**kwargs)


@pytest.mark.parametrize("n_masses", [(1, True), (1, 2.5)])
def test_inference_config_rejects_noninteger_model_counts(n_masses: tuple) -> None:
    with pytest.raises(ValueError, match="n_masses"):
        _make_inference_config(n_masses=n_masses)


@pytest.mark.parametrize("seed", [True, 1.5, "1"])
def test_inference_config_rejects_invalid_seed(seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        _make_inference_config(seed=seed)


@pytest.mark.parametrize("field", ["ess_target", "proposal_scale"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_inference_config_rejects_nonfinite_numeric_controls(
    field: str, value: float
) -> None:
    with pytest.raises(ValueError, match=field):
        _make_inference_config(**{field: value})


@pytest.mark.parametrize("n_observations", [True, 1.5])
def test_simulation_config_rejects_noninteger_observation_count(
    n_observations: object,
) -> None:
    with pytest.raises(ValueError, match="n_observations"):
        _make_simulation_config(n_observations=n_observations)


@pytest.mark.parametrize("seed", [True, 1.5, "1"])
def test_simulation_config_rejects_invalid_seed(seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        _make_simulation_config(seed=seed)


def test_public_api_is_exported_from_package() -> None:
    import clocks

    assert clocks.build_particle_filter is not None
    assert clocks.infer is not None
    assert clocks.simulate is not None
    assert clocks.simulate_and_infer is not None
    assert clocks.SimulationConfig is not None


class TestDefaultFlipRecovery:
    """Numerical recovery at the new defaults (spec: default-flip regressions).

    Single-mass 1D recovery and correct-K model comparison already exist;
    these add the missing single-mass 2D and multi-mass 1D coverage.
    """

    def test_single_mass_2d_recovery(self) -> None:
        rng = np.random.default_rng(3)
        ca = ClockArray(positions=rng.uniform(-5.0, 5.0, (8, 2)), track_offset=3.0)
        truth = MassConfig(positions=np.array([[1.5, -2.0]]), masses=np.array([0.15]))
        sim = simulate(
            SimulationConfig(
                clock_array=ca,
                ground_truth=truth,
                noise=NoiseConfig(observation_std=0.005),
                n_observations=60,
                seed=3,
            )
        )
        result = infer(
            sim.observations,
            InferenceConfig(
                clock_array=ca,
                noise=NoiseConfig(observation_std=0.005),
                prior=PriorConfig(position_range=(-8.0, 8.0), mass_range=(0.005, 0.15)),
                n_particles=2000,
                n_masses=1,
                seed=3,
            ),
        )
        error = np.abs(result.posterior_mean - np.array([1.5, -2.0, 0.15]))
        assert np.all(error <= np.array([0.5, 0.5, 0.1]))

    def test_multi_mass_1d_recovery(self) -> None:
        ca = ClockArray(
            positions=np.linspace(-6.0, 6.0, 10).reshape(-1, 1),
            track_offset=3.0,
        )
        truth = MassConfig(
            positions=np.array([[-3.0], [4.5]]), masses=np.array([0.045, 0.030])
        )
        sim = simulate(
            SimulationConfig(
                clock_array=ca,
                ground_truth=truth,
                noise=NoiseConfig(observation_std=0.005),
                n_observations=80,
                seed=5,
            )
        )
        result = infer(
            sim.observations,
            InferenceConfig(
                clock_array=ca,
                noise=NoiseConfig(observation_std=0.005),
                prior=PriorConfig(position_range=(-8.0, 8.0), mass_range=(0.005, 0.15)),
                n_particles=4000,
                n_masses=2,
                seed=5,
            ),
        )
        truth_vec = np.array([-3.0, 4.5, 0.045, 0.030])
        error = np.abs(result.posterior_mean - truth_vec)
        assert np.all(error <= np.array([0.5, 0.5, 0.1, 0.1]))


class TestConditionalSupportPlumbing:
    def test_build_particle_filter_uses_reject_and_stay(self) -> None:
        ca = ClockArray(
            positions=np.linspace(-5, 5, 6).reshape(-1, 1), track_offset=3.0
        )
        pf = build_particle_filter(
            InferenceConfig(
                clock_array=ca,
                noise=NoiseConfig(observation_std=0.01),
                prior=PriorConfig(position_range=(-8.0, 8.0), mass_range=(0.005, 0.15)),
                n_particles=50,
                n_masses=2,
            )
        )
        assert np.all(np.isfinite(pf.log_prior_density(pf.state.particles)))

    def test_model_comparison_filters_get_conditional_support(self) -> None:
        ca = ClockArray(
            positions=np.linspace(-5, 5, 6).reshape(-1, 1), track_offset=3.0
        )
        result = infer(
            [Observation(rates=np.ones(6), time=0.0)],
            InferenceConfig(
                clock_array=ca,
                noise=NoiseConfig(observation_std=0.01),
                prior=PriorConfig(position_range=(-8.0, 8.0), mass_range=(0.005, 0.15)),
                n_particles=50,
                n_masses=(1, 2),
            ),
        )
        assert result.best_model in (1, 2)

        mc = build_model_comparison(
            InferenceConfig(
                clock_array=ca,
                noise=NoiseConfig(0.01),
                prior=PriorConfig((-8.0, 8.0), (0.005, 0.15)),
                n_particles=50,
                n_masses=(1, 2),
            )
        )
        for particle_filter in mc.filters.values():
            assert np.all(
                np.isfinite(
                    particle_filter.log_prior_density(particle_filter.state.particles)
                )
            )


class TestInference3D:
    def test_single_mass_3d_recovery(self) -> None:
        """(x, y, z, M) inference works end-to-end through the public API."""
        rng = np.random.default_rng(3)
        clock_array = ClockArray(
            positions=rng.uniform(-3, 3, size=(12, 3)), track_offset=0.0
        )
        truth = MassConfig(
            positions=np.array([[1.0, -1.5, 0.5]]), masses=np.array([0.05])
        )
        sim = simulate(
            SimulationConfig(
                clock_array=clock_array,
                ground_truth=truth,
                noise=NoiseConfig(observation_std=0.005),
                n_observations=40,
                seed=3,
            )
        )
        result = infer(
            sim.observations,
            InferenceConfig(
                clock_array=clock_array,
                noise=NoiseConfig(observation_std=0.005),
                prior=PriorConfig(position_range=(-5.0, 5.0), mass_range=(0.005, 0.15)),
                n_particles=2000,
                n_masses=1,
                seed=3,
            ),
        )
        assert isinstance(result, InferenceResult)
        expected = np.array([1.0, -1.5, 0.5, 0.05])
        assert np.all(np.abs(result.posterior_mean - expected) < 0.5)
