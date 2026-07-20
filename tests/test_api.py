"""Tests for the public library API."""

import numpy as np
import pytest

from clocks.api import build_particle_filter, infer, simulate, simulate_and_infer
from clocks.config import InferenceConfig, NoiseConfig, PriorConfig, SimulationConfig
from clocks.inference import ModelComparison
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
    assert payload["simulation"]["ground_truth"]["masses"] == [0.6, 0.4]


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


def test_prior_config_rejects_invalid_position_range() -> None:
    with pytest.raises(ValueError, match="position_range must be increasing"):
        PriorConfig(position_range=(2.0, -2.0), mass_range=(0.1, 2.0))


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


def test_public_api_is_exported_from_package() -> None:
    import clocks

    assert clocks.build_particle_filter is not None
    assert clocks.infer is not None
    assert clocks.simulate is not None
    assert clocks.simulate_and_infer is not None
    assert clocks.SimulationConfig is not None


class TestAnnealedDefaultsAPI:
    def _config(self, **kwargs: object) -> InferenceConfig:
        ca = ClockArray(
            positions=np.linspace(-5, 5, 6).reshape(-1, 1), track_offset=3.0
        )
        defaults: dict = dict(
            clock_array=ca,
            noise=NoiseConfig(observation_std=0.01),
            prior=PriorConfig(position_range=(-8.0, 8.0), mass_range=(0.1, 2.0)),
            n_particles=100,
            n_masses=1,
        )
        defaults.update(kwargs)
        return InferenceConfig(**defaults)

    def test_inference_config_default_jitter_is_annealed(self) -> None:
        assert self._config().jitter == "annealed"

    def test_jitter_tau_plumbs_through_build(self) -> None:
        pf = build_particle_filter(self._config(jitter_tau=7.0))
        assert pf.jitter_tau == 7.0
        assert pf.jitter == "annealed"

    @pytest.mark.parametrize("bad_tau", [0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_jitter_tau_raises(self, bad_tau: float) -> None:
        with pytest.raises(ValueError, match="jitter_tau"):
            self._config(jitter_tau=bad_tau)

    @pytest.mark.parametrize("bad_std", [-0.1, float("nan"), float("inf")])
    def test_invalid_jitter_std_raises(self, bad_std: float) -> None:
        with pytest.raises(ValueError, match="jitter_std"):
            self._config(jitter_std=bad_std)


class TestDefaultFlipRecovery:
    """Numerical recovery at the new defaults (spec: default-flip regressions).

    Single-mass 1D recovery and correct-K model comparison already exist;
    these add the missing single-mass 2D and multi-mass 1D coverage.
    """

    def test_single_mass_2d_recovery(self) -> None:
        rng = np.random.default_rng(3)
        ca = ClockArray(positions=rng.uniform(-5.0, 5.0, (8, 2)), track_offset=3.0)
        truth = MassConfig(positions=np.array([[1.5, -2.0]]), masses=np.array([0.5]))
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
                prior=PriorConfig(position_range=(-8.0, 8.0), mass_range=(0.1, 2.0)),
                n_particles=2000,
                n_masses=1,
                seed=3,
            ),
        )
        error = np.abs(result.posterior_mean - np.array([1.5, -2.0, 0.5]))
        assert np.all(error <= np.array([0.5, 0.5, 0.1]))

    def test_multi_mass_1d_recovery(self) -> None:
        ca = ClockArray(
            positions=np.linspace(-6.0, 6.0, 10).reshape(-1, 1),
            track_offset=3.0,
        )
        truth = MassConfig(
            positions=np.array([[-3.0], [4.5]]), masses=np.array([0.6, 0.4])
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
                prior=PriorConfig(position_range=(-8.0, 8.0), mass_range=(0.1, 2.0)),
                n_particles=4000,
                n_masses=2,
                seed=5,
            ),
        )
        truth_vec = np.array([-3.0, 4.5, 0.6, 0.4])
        error = np.abs(result.posterior_mean - truth_vec)
        assert np.all(error <= np.array([0.5, 0.5, 0.1, 0.1]))


class TestSupportBoundsPlumbing:
    def test_build_particle_filter_constructs_bounds(self) -> None:
        ca = ClockArray(
            positions=np.linspace(-5, 5, 6).reshape(-1, 1), track_offset=3.0
        )
        pf = build_particle_filter(
            InferenceConfig(
                clock_array=ca,
                noise=NoiseConfig(observation_std=0.01),
                prior=PriorConfig(position_range=(-8.0, 8.0), mass_range=(0.1, 2.0)),
                n_particles=50,
                n_masses=2,
            )
        )
        lower, upper = pf.support_bounds
        # 2 masses x 1 dim -> params [x1, x2, M1, M2]
        assert np.allclose(lower[:2], -8.0) and np.allclose(upper[:2], 8.0)
        assert np.all(lower[2:] > 0.0) and np.all(lower[2:] < 1e-100)
        assert np.all(np.isinf(upper[2:]))

    def test_model_comparison_filters_get_bounds(self) -> None:
        ca = ClockArray(
            positions=np.linspace(-5, 5, 6).reshape(-1, 1), track_offset=3.0
        )
        result = infer(
            [Observation(rates=np.ones(6), time=0.0)],
            InferenceConfig(
                clock_array=ca,
                noise=NoiseConfig(observation_std=0.01),
                prior=PriorConfig(position_range=(-8.0, 8.0), mass_range=(0.1, 2.0)),
                n_particles=50,
                n_masses=(1, 2),
            ),
        )
        assert result.best_model in (1, 2)

        # Assert that ModelComparison-constructed filters carry support_bounds.
        mc = ModelComparison(clock_array=ca, noise_std=0.01, k_max=2)
        n_dims = ca.positions.shape[1]  # 1 for this test
        for k in mc.filters:
            assert mc.filters[k].support_bounds is not None
            lower, upper = mc.filters[k].support_bounds
            # For filter k, params are [positions (k*n_dims), masses (k)].
            position_end = k * n_dims
            # Position bounds: lower and upper should be -8.0 and 8.0 respectively.
            assert np.allclose(lower[:position_end], -8.0)
            assert np.allclose(upper[:position_end], 8.0)
            # Mass bounds: lower should be > 0 and finite, upper should be +inf.
            assert np.all(lower[position_end:] > 0.0)
            assert np.all(np.isfinite(lower[position_end:]))
            assert np.all(np.isinf(upper[position_end:]))


class TestInference3D:
    def test_single_mass_3d_recovery(self) -> None:
        """(x, y, z, M) inference works end-to-end through the public API."""
        rng = np.random.default_rng(3)
        clock_array = ClockArray(
            positions=rng.uniform(-3, 3, size=(12, 3)), track_offset=0.0
        )
        truth = MassConfig(
            positions=np.array([[1.0, -1.5, 0.5]]), masses=np.array([0.5])
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
                prior=PriorConfig(position_range=(-5.0, 5.0), mass_range=(0.1, 2.0)),
                n_particles=2000,
                n_masses=1,
                seed=3,
            ),
        )
        assert isinstance(result, InferenceResult)
        expected = np.array([1.0, -1.5, 0.5, 0.5])
        assert np.all(np.abs(result.posterior_mean - expected) < 0.5)
