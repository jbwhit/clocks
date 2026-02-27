"""Tests for the particle filter inference engine."""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from clocks.inference import ConvergenceInfo, ParticleFilter
from clocks.noise import add_clock_noise
from clocks.physics import clock_rates, clock_rates_batch
from clocks.types import ClockArray, MassConfig, Observation


def _make_1d_scenario(
    true_x: float = 2.5,
    true_m: float = 0.8,
    clock_positions: list[float] | None = None,
    track_offset: float = 1.0,
) -> tuple[MassConfig, ClockArray]:
    if clock_positions is None:
        clock_positions = [-5.0, 0.0, 5.0]
    mc = MassConfig(
        positions=np.array([[true_x]]),
        masses=np.array([true_m]),
    )
    ca = ClockArray(
        positions=np.array([[x] for x in clock_positions]),
        track_offset=track_offset,
    )
    return mc, ca


def _make_forward_model(
    clock_array: ClockArray,
) -> Callable[[NDArray[np.floating]], NDArray[np.floating]]:
    """Create forward model callable for particle filter: params → rates."""

    def forward(params: np.ndarray) -> np.ndarray:
        mc = MassConfig(
            positions=np.array([[params[0]]]),
            masses=np.array([params[1]]),
        )
        return clock_rates(mc, clock_array)

    return forward


class TestParticleFilter:
    def test_convergence(self) -> None:
        """Particle filter should converge near the true parameters."""
        rng = np.random.default_rng(42)
        true_x, true_m = 2.5, 0.8
        mc, ca = _make_1d_scenario(true_x, true_m)
        true_rates = clock_rates(mc, ca)
        forward = _make_forward_model(ca)

        def prior_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
            x = rng.uniform(-8, 8, n)
            m = rng.uniform(0.1, 2.0, n)
            return np.column_stack([x, m])

        pf = ParticleFilter(
            n_particles=500,
            prior_sampler=prior_sampler,
            forward_model=forward,
            noise_std=0.005,
            jitter_std=0.05,
            rng=rng,
        )

        for t in range(30):
            noisy = add_clock_noise(true_rates, noise_std=0.005, rng=rng)
            obs = Observation(rates=noisy, time=float(t))
            pf.update(obs)

        est = pf.estimate()
        assert abs(est["mean"][0] - true_x) < 1.0, (
            f"x estimate {est['mean'][0]} too far from {true_x}"
        )
        assert abs(est["mean"][1] - true_m) < 0.3, (
            f"M estimate {est['mean'][1]} too far from {true_m}"
        )

    def test_uncertainty_narrows(self) -> None:
        """Standard deviation should decrease with more observations."""
        rng = np.random.default_rng(123)
        true_x, true_m = 2.5, 0.8
        mc, ca = _make_1d_scenario(true_x, true_m)
        true_rates = clock_rates(mc, ca)
        forward = _make_forward_model(ca)

        def prior_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
            x = rng.uniform(-8, 8, n)
            m = rng.uniform(0.1, 2.0, n)
            return np.column_stack([x, m])

        pf = ParticleFilter(
            n_particles=1000,
            prior_sampler=prior_sampler,
            forward_model=forward,
            noise_std=0.005,
            jitter_std=0.01,
            rng=rng,
        )

        # Measure after 5 observations
        for t in range(5):
            noisy = add_clock_noise(true_rates, noise_std=0.005, rng=rng)
            pf.update(Observation(rates=noisy, time=float(t)))
        early_std = pf.estimate()["std"]

        # Continue to 30 observations
        for t in range(5, 30):
            noisy = add_clock_noise(true_rates, noise_std=0.005, rng=rng)
            pf.update(Observation(rates=noisy, time=float(t)))
        late_std = pf.estimate()["std"]

        # At least one parameter's uncertainty should decrease
        assert np.any(late_std < early_std), (
            f"Uncertainty didn't narrow: {early_std} → {late_std}"
        )

    def test_history_length(self) -> None:
        """History should grow with each observation."""
        rng = np.random.default_rng(0)
        _, ca = _make_1d_scenario()
        forward = _make_forward_model(ca)

        def prior_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
            return rng.uniform(-5, 5, (n, 2))

        pf = ParticleFilter(
            n_particles=50,
            prior_sampler=prior_sampler,
            forward_model=forward,
            noise_std=0.01,
            rng=rng,
        )

        for t in range(10):
            obs = Observation(rates=np.array([0.9, 0.95, 0.99]), time=float(t))
            pf.update(obs)

        # 1 initial + 10 updates
        assert len(pf.history) == 11
        assert pf.state.observations_seen == 10

    def test_batch_matches_scalar(self) -> None:
        """Batch forward model should produce same estimates as scalar."""
        true_x, true_m = 2.5, 0.8
        mc, ca = _make_1d_scenario(true_x, true_m)
        true_rates = clock_rates(mc, ca)
        forward = _make_forward_model(ca)

        def forward_batch(particles: np.ndarray) -> np.ndarray:
            return clock_rates_batch(particles[:, :1], particles[:, 1], ca)

        def prior_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
            x = rng.uniform(-8, 8, n)
            m = rng.uniform(0.1, 2.0, n)
            return np.column_stack([x, m])

        # Run scalar version
        rng1 = np.random.default_rng(42)
        pf_scalar = ParticleFilter(
            n_particles=200,
            prior_sampler=prior_sampler,
            forward_model=forward,
            noise_std=0.005,
            jitter_std=0.01,
            rng=rng1,
        )

        # Run batch version with same seed
        rng2 = np.random.default_rng(42)
        pf_batch = ParticleFilter(
            n_particles=200,
            prior_sampler=prior_sampler,
            forward_model=forward,
            noise_std=0.005,
            jitter_std=0.01,
            rng=rng2,
            forward_model_batch=forward_batch,
        )

        # Same observations for both
        rng_obs = np.random.default_rng(99)
        for t in range(10):
            noisy = add_clock_noise(true_rates, noise_std=0.005, rng=rng_obs)
            obs = Observation(rates=noisy, time=float(t))
            pf_scalar.update(obs)
            pf_batch.update(obs)

        est_scalar = pf_scalar.estimate()
        est_batch = pf_batch.estimate()
        np.testing.assert_allclose(est_scalar["mean"], est_batch["mean"], atol=1e-10)
        np.testing.assert_allclose(est_scalar["std"], est_batch["std"], atol=1e-10)

    def test_constraint_fn_applied(self) -> None:
        """Constraint function should be called after resampling."""
        _, ca = _make_1d_scenario()
        forward = _make_forward_model(ca)

        call_count = [0]

        def counting_constraint(particles: np.ndarray) -> np.ndarray:
            call_count[0] += 1
            return particles

        def prior_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
            return rng.uniform(-5, 5, (n, 2))

        rng = np.random.default_rng(0)
        pf = ParticleFilter(
            n_particles=50,
            prior_sampler=prior_sampler,
            forward_model=forward,
            noise_std=0.01,
            resample_threshold=1.0,  # always resample
            rng=rng,
            constraint_fn=counting_constraint,
        )

        for t in range(5):
            obs = Observation(rates=np.array([0.9, 0.95, 0.99]), time=float(t))
            pf.update(obs)

        assert call_count[0] == 5, (
            f"Constraint should be called on every resample, got {call_count[0]}"
        )

    # --- Resampling methods ---

    def test_stratified_resampling_converges(self) -> None:
        """Stratified resampling should converge near the true parameters."""
        rng = np.random.default_rng(42)
        true_x, true_m = 2.5, 0.8
        mc, ca = _make_1d_scenario(true_x, true_m)
        true_rates = clock_rates(mc, ca)
        forward = _make_forward_model(ca)

        def prior_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
            x = rng.uniform(-8, 8, n)
            m = rng.uniform(0.1, 2.0, n)
            return np.column_stack([x, m])

        pf = ParticleFilter(
            n_particles=500,
            prior_sampler=prior_sampler,
            forward_model=forward,
            noise_std=0.005,
            jitter_std=0.05,
            rng=rng,
            resampling="stratified",
        )

        for t in range(30):
            noisy = add_clock_noise(true_rates, noise_std=0.005, rng=rng)
            pf.update(Observation(rates=noisy, time=float(t)))

        est = pf.estimate()
        assert abs(est["mean"][0] - true_x) < 1.0
        assert abs(est["mean"][1] - true_m) < 0.3

    def test_residual_resampling_converges(self) -> None:
        """Residual resampling should converge near the true parameters."""
        rng = np.random.default_rng(42)
        true_x, true_m = 2.5, 0.8
        mc, ca = _make_1d_scenario(true_x, true_m)
        true_rates = clock_rates(mc, ca)
        forward = _make_forward_model(ca)

        def prior_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
            x = rng.uniform(-8, 8, n)
            m = rng.uniform(0.1, 2.0, n)
            return np.column_stack([x, m])

        pf = ParticleFilter(
            n_particles=500,
            prior_sampler=prior_sampler,
            forward_model=forward,
            noise_std=0.005,
            jitter_std=0.05,
            rng=rng,
            resampling="residual",
        )

        for t in range(30):
            noisy = add_clock_noise(true_rates, noise_std=0.005, rng=rng)
            pf.update(Observation(rates=noisy, time=float(t)))

        est = pf.estimate()
        assert abs(est["mean"][0] - true_x) < 1.0
        assert abs(est["mean"][1] - true_m) < 0.3

    def test_invalid_resampling_raises(self) -> None:
        """Unknown resampling method should raise ValueError."""

        def prior_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
            return rng.uniform(-5, 5, (n, 2))

        with pytest.raises(ValueError, match="Unknown resampling method"):
            ParticleFilter(
                n_particles=50,
                prior_sampler=prior_sampler,
                forward_model=lambda p: np.array([1.0]),
                noise_std=0.01,
                resampling="bogus",
            )

    # --- Adaptive jitter ---

    def test_adaptive_jitter_converges(self) -> None:
        """Adaptive jitter should still converge near the true parameters."""
        rng = np.random.default_rng(42)
        true_x, true_m = 2.5, 0.8
        mc, ca = _make_1d_scenario(true_x, true_m)
        true_rates = clock_rates(mc, ca)
        forward = _make_forward_model(ca)

        def prior_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
            x = rng.uniform(-8, 8, n)
            m = rng.uniform(0.1, 2.0, n)
            return np.column_stack([x, m])

        pf = ParticleFilter(
            n_particles=500,
            prior_sampler=prior_sampler,
            forward_model=forward,
            noise_std=0.005,
            jitter_std=0.5,
            rng=rng,
            adaptive_jitter=True,
        )

        for t in range(30):
            noisy = add_clock_noise(true_rates, noise_std=0.005, rng=rng)
            pf.update(Observation(rates=noisy, time=float(t)))

        est = pf.estimate()
        assert abs(est["mean"][0] - true_x) < 1.0
        assert abs(est["mean"][1] - true_m) < 0.3

    def test_adaptive_jitter_scales_with_spread(self) -> None:
        """Wider prior should produce wider jitter with adaptive_jitter."""
        _, ca = _make_1d_scenario()
        forward = _make_forward_model(ca)

        # Narrow prior
        def narrow_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
            x = rng.uniform(-0.1, 0.1, n)
            m = rng.uniform(0.49, 0.51, n)
            return np.column_stack([x, m])

        # Wide prior
        def wide_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
            x = rng.uniform(-100, 100, n)
            m = rng.uniform(0.01, 10.0, n)
            return np.column_stack([x, m])

        rng1 = np.random.default_rng(0)
        pf_narrow = ParticleFilter(
            n_particles=500,
            prior_sampler=narrow_sampler,
            forward_model=forward,
            noise_std=0.01,
            jitter_std=1.0,
            adaptive_jitter=True,
            resample_threshold=1.0,  # force resample
            rng=rng1,
        )

        rng2 = np.random.default_rng(0)
        pf_wide = ParticleFilter(
            n_particles=500,
            prior_sampler=wide_sampler,
            forward_model=forward,
            noise_std=0.01,
            jitter_std=1.0,
            adaptive_jitter=True,
            resample_threshold=1.0,
            rng=rng2,
        )

        narrow_std = pf_narrow.estimate()["std"]
        wide_std = pf_wide.estimate()["std"]
        # The wide prior should have larger spread
        assert np.all(wide_std > narrow_std)

    # --- Convergence diagnostics ---

    def test_converged_false_initially(self) -> None:
        """Filter should not be converged with 0-1 observations."""
        rng = np.random.default_rng(0)

        def prior_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
            return rng.uniform(-5, 5, (n, 2))

        _, ca = _make_1d_scenario()
        forward = _make_forward_model(ca)

        pf = ParticleFilter(
            n_particles=100,
            prior_sampler=prior_sampler,
            forward_model=forward,
            noise_std=0.01,
            rng=rng,
        )

        info = pf.converged()
        assert not info["converged"]
        assert not info["estimates_stable"]

    def test_converged_true_after_many_observations(self) -> None:
        """Filter should converge after enough observations with generous thresholds."""
        rng = np.random.default_rng(42)
        true_x, true_m = 2.5, 0.8
        mc, ca = _make_1d_scenario(true_x, true_m)
        true_rates = clock_rates(mc, ca)
        forward = _make_forward_model(ca)

        def prior_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
            x = rng.uniform(-8, 8, n)
            m = rng.uniform(0.1, 2.0, n)
            return np.column_stack([x, m])

        pf = ParticleFilter(
            n_particles=500,
            prior_sampler=prior_sampler,
            forward_model=forward,
            noise_std=0.005,
            jitter_std=0.05,
            rng=rng,
        )

        for t in range(50):
            noisy = add_clock_noise(true_rates, noise_std=0.005, rng=rng)
            pf.update(Observation(rates=noisy, time=float(t)))

        info = pf.converged(std_threshold=1.0, stability_threshold=0.5)
        assert info["converged"]
        assert info["estimates_stable"]
        assert np.all(info["per_param_converged"])

    def test_convergence_info_structure(self) -> None:
        """ConvergenceInfo should have correct keys, types, and shapes."""
        rng = np.random.default_rng(0)

        def prior_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
            return rng.uniform(-5, 5, (n, 3))

        pf = ParticleFilter(
            n_particles=50,
            prior_sampler=prior_sampler,
            forward_model=lambda p: np.array([1.0]),
            noise_std=0.01,
            rng=rng,
        )

        info = pf.converged()
        assert isinstance(info, dict)
        assert set(info.keys()) == set(ConvergenceInfo.__annotations__)
        assert isinstance(info["converged"], bool)
        assert isinstance(info["estimates_stable"], bool)
        assert isinstance(info["ess"], float)
        assert info["per_param_std"].shape == (3,)
        assert info["per_param_converged"].shape == (3,)

    def test_estimates_stable_requires_window(self) -> None:
        """estimates_stable should be False when history < window."""
        rng = np.random.default_rng(0)
        _, ca = _make_1d_scenario()
        forward = _make_forward_model(ca)

        def prior_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
            return rng.uniform(-5, 5, (n, 2))

        pf = ParticleFilter(
            n_particles=50,
            prior_sampler=prior_sampler,
            forward_model=forward,
            noise_std=0.01,
            rng=rng,
        )

        # Add 5 observations, but check with window=10
        for t in range(5):
            obs = Observation(rates=np.array([0.9, 0.95, 0.99]), time=float(t))
            pf.update(obs)

        info = pf.converged(window=10)
        assert not info["estimates_stable"]
