"""Tests for the particle filter inference engine."""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.special import logsumexp

from clocks.inference import ConvergenceInfo, ModelComparison, ParticleFilter
from clocks.noise import add_clock_noise, log_likelihood_gaussian
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

    # --- Covariance jitter ---

    def test_covariance_jitter_converges(self) -> None:
        """Covariance jitter should still converge near the true parameters."""
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
            jitter="covariance",
        )

        for t in range(30):
            noisy = add_clock_noise(true_rates, noise_std=0.005, rng=rng)
            pf.update(Observation(rates=noisy, time=float(t)))

        est = pf.estimate()
        assert abs(est["mean"][0] - true_x) < 1.0
        assert abs(est["mean"][1] - true_m) < 0.3

    def test_covariance_jitter_scales_with_spread(self) -> None:
        """Wider prior should produce wider jitter with covariance jitter."""
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
            jitter="covariance",
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
            jitter="covariance",
            resample_threshold=1.0,
            rng=rng2,
        )

        narrow_std = pf_narrow.estimate()["std"]
        wide_std = pf_wide.estimate()["std"]
        # The wide prior should have larger spread
        assert np.all(wide_std > narrow_std)

    def test_invalid_jitter_raises(self) -> None:
        """Unknown jitter mode should raise ValueError."""

        def prior_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
            return rng.uniform(-5, 5, (n, 2))

        with pytest.raises(ValueError, match="Unknown jitter mode"):
            ParticleFilter(
                n_particles=50,
                prior_sampler=prior_sampler,
                forward_model=lambda p: np.array([1.0]),
                noise_std=0.01,
                jitter="bogus",
            )

    # --- Log-prior ---

    def test_log_prior_zeroes_invalid_particles(self) -> None:
        """Particles with -inf log_prior should get zero weight."""
        _, ca = _make_1d_scenario()
        forward = _make_forward_model(ca)

        def prior_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
            x = rng.uniform(-8, 8, n)
            m = rng.uniform(0.1, 2.0, n)
            return np.column_stack([x, m])

        # log_prior that kills negative-x particles
        def log_prior(particles: np.ndarray) -> np.ndarray:
            lp = np.zeros(particles.shape[0])
            lp[particles[:, 0] < 0] = -np.inf
            return lp

        rng = np.random.default_rng(42)
        pf = ParticleFilter(
            n_particles=500,
            prior_sampler=prior_sampler,
            forward_model=forward,
            noise_std=0.005,
            rng=rng,
            log_prior=log_prior,
        )

        obs = Observation(rates=np.array([0.98, 0.95, 0.99]), time=0.0)
        pf.update(obs)

        # After update, all weight should be on non-negative-x particles
        particles = pf.state.particles
        weights = pf.state.weights
        negative_weight = weights[particles[:, 0] < 0].sum()
        assert negative_weight < 1e-10, (
            f"Negative-x particles should have ~0 weight, got {negative_weight}"
        )

    def test_log_prior_called_each_update(self) -> None:
        """log_prior should be called once per update."""
        _, ca = _make_1d_scenario()
        forward = _make_forward_model(ca)

        def prior_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
            return rng.uniform(-5, 5, (n, 2))

        call_count = [0]

        def counting_prior(particles: np.ndarray) -> np.ndarray:
            call_count[0] += 1
            return np.zeros(particles.shape[0])

        rng = np.random.default_rng(0)
        pf = ParticleFilter(
            n_particles=50,
            prior_sampler=prior_sampler,
            forward_model=forward,
            noise_std=0.01,
            rng=rng,
            log_prior=counting_prior,
        )

        for t in range(5):
            obs = Observation(rates=np.array([0.9, 0.95, 0.99]), time=float(t))
            pf.update(obs)

        assert call_count[0] == 5, (
            f"log_prior should be called 5 times, got {call_count[0]}"
        )

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

    # --- Log-evidence ---

    def test_log_evidence_starts_at_zero(self) -> None:
        """Log-evidence should be 0.0 before any updates."""
        rng = np.random.default_rng(0)

        def prior_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
            return rng.uniform(-5, 5, (n, 2))

        pf = ParticleFilter(
            n_particles=50,
            prior_sampler=prior_sampler,
            forward_model=lambda p: np.array([1.0]),
            noise_std=0.01,
            rng=rng,
        )
        assert pf.log_evidence == 0.0

    def test_log_evidence_tracked(self) -> None:
        """Log-evidence should be a finite negative float after updates."""
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
            n_particles=200,
            prior_sampler=prior_sampler,
            forward_model=forward,
            noise_std=0.005,
            rng=rng,
        )

        for t in range(10):
            noisy = add_clock_noise(true_rates, noise_std=0.005, rng=rng)
            pf.update(Observation(rates=noisy, time=float(t)))

        assert np.isfinite(pf.log_evidence)
        assert pf.log_evidence < 0

    def test_log_evidence_accumulates(self) -> None:
        """Log-evidence magnitude should increase with more observations."""
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
            n_particles=200,
            prior_sampler=prior_sampler,
            forward_model=forward,
            noise_std=0.005,
            rng=rng,
        )

        for t in range(5):
            noisy = add_clock_noise(true_rates, noise_std=0.005, rng=rng)
            pf.update(Observation(rates=noisy, time=float(t)))
        early_evidence = pf.log_evidence

        for t in range(5, 15):
            noisy = add_clock_noise(true_rates, noise_std=0.005, rng=rng)
            pf.update(Observation(rates=noisy, time=float(t)))
        late_evidence = pf.log_evidence

        # More observations → more accumulated evidence (larger magnitude)
        assert abs(late_evidence) > abs(early_evidence)

    def test_log_evidence_matches_direct_computation(self) -> None:
        """Each update's log-evidence increment is log(sum(prev_w * L))."""
        mc, ca = _make_1d_scenario()
        true_rates = clock_rates(mc, ca)
        forward = _make_forward_model(ca)

        def prior_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
            x = rng.uniform(-8, 8, n)
            m = rng.uniform(0.1, 2.0, n)
            return np.column_stack([x, m])

        # resample_threshold=0 → never resample → previous weights stay
        # NONUNIFORM after the first update, which is the case the bias
        # claim is about.
        pf = ParticleFilter(
            n_particles=50,
            prior_sampler=prior_sampler,
            forward_model=forward,
            noise_std=0.005,
            resample_threshold=0.0,
            rng=np.random.default_rng(1),
        )

        obs_rng = np.random.default_rng(0)
        expected = 0.0
        for t in range(3):
            obs = Observation(
                rates=add_clock_noise(true_rates, 0.005, obs_rng), time=float(t)
            )
            prev_weights = pf.state.weights.copy()
            # log(0) → -inf is intended for fully underflowed weights
            with np.errstate(divide="ignore"):
                log_prev = np.log(prev_weights)
            log_w = log_prev + np.array(
                [
                    log_likelihood_gaussian(obs.rates, forward(p), 0.005)
                    for p in pf.state.particles
                ]
            )
            expected += logsumexp(log_w)
            pf.update(obs)
            assert pf.log_evidence == pytest.approx(expected, rel=1e-9)


class TestModelComparison:
    def _make_clock_array(self) -> ClockArray:
        return ClockArray(
            positions=np.array([[-6.0], [-3.0], [0.0], [3.0], [6.0]]),
            track_offset=1.0,
        )

    def test_model_comparison_selects_true_k(self) -> None:
        """K=2 ground truth should be favored after enough observations."""
        rng = np.random.default_rng(42)
        ca = self._make_clock_array()
        mc = MassConfig(
            positions=np.array([[-2.0], [3.0]]),
            masses=np.array([0.6, 0.4]),
        )
        true_rates = clock_rates(mc, ca)

        comp = ModelComparison(
            clock_array=ca,
            noise_std=0.005,
            n_dims=1,
            k_max=3,
            n_particles=1500,
            jitter_std=0.02,
            rng=rng,
        )

        for t in range(40):
            noisy = add_clock_noise(true_rates, noise_std=0.005, rng=rng)
            comp.update(Observation(rates=noisy, time=float(t)))

        result = comp.evidence()
        map_k = max(result["posterior"], key=lambda x: result["posterior"][x])
        assert map_k == 2, f"Expected MAP K=2, got K={map_k}: {result['posterior']}"

    def test_model_comparison_estimate_returns_map(self) -> None:
        """estimate() with no k should use the MAP model."""
        rng = np.random.default_rng(42)
        ca = self._make_clock_array()
        mc = MassConfig(
            positions=np.array([[2.0]]),
            masses=np.array([0.5]),
        )
        true_rates = clock_rates(mc, ca)

        comp = ModelComparison(
            clock_array=ca,
            noise_std=0.005,
            n_dims=1,
            k_max=2,
            n_particles=500,
            jitter_std=0.02,
            rng=rng,
        )

        for t in range(20):
            noisy = add_clock_noise(true_rates, noise_std=0.005, rng=rng)
            comp.update(Observation(rates=noisy, time=float(t)))

        est = comp.estimate()
        assert "mean" in est
        assert "std" in est
        assert "ess" in est

    def test_model_comparison_evidence_structure(self) -> None:
        """Evidence result should have correct keys and posteriors sum to 1."""
        rng = np.random.default_rng(0)
        ca = self._make_clock_array()

        comp = ModelComparison(
            clock_array=ca,
            noise_std=0.01,
            n_dims=1,
            k_max=3,
            n_particles=100,
            rng=rng,
        )

        # Need at least one observation
        obs = Observation(rates=np.array([0.98, 0.95, 0.90, 0.95, 0.98]), time=0.0)
        comp.update(obs)

        result = comp.evidence()
        assert set(result["log_evidence"].keys()) == {1, 2, 3}
        assert set(result["posterior"].keys()) == {1, 2, 3}
        np.testing.assert_allclose(sum(result["posterior"].values()), 1.0, atol=1e-10)

    def test_model_comparison_invalid_k_raises(self) -> None:
        """estimate(k=5) should raise ValueError."""
        rng = np.random.default_rng(0)
        ca = self._make_clock_array()
        comp = ModelComparison(
            clock_array=ca, noise_std=0.01, k_max=3, n_particles=50, rng=rng
        )
        with pytest.raises(ValueError, match="No filter for K=5"):
            comp.estimate(k=5)

    def test_model_comparison_constraint_enforced(self) -> None:
        """K=2 particles should have x1 <= x2 after updates."""
        rng = np.random.default_rng(42)
        ca = self._make_clock_array()

        comp = ModelComparison(
            clock_array=ca,
            noise_std=0.01,
            n_dims=1,
            k_max=2,
            n_particles=200,
            rng=rng,
        )

        obs = Observation(rates=np.array([0.98, 0.95, 0.90, 0.95, 0.98]), time=0.0)
        comp.update(obs)

        # Check K=2 filter: positions x1 <= x2
        particles = comp.filters[2].state.particles
        x1 = particles[:, 0]  # first position (dim 0)
        x2 = particles[:, 1]  # second position (dim 0)
        assert np.all(x1 <= x2 + 1e-10), "Ordering constraint not enforced"

    def test_model_comparison_accepts_explicit_k_values(self) -> None:
        """Explicit k_values should only build and report the requested models."""
        rng = np.random.default_rng(0)
        ca = self._make_clock_array()

        comp = ModelComparison(
            clock_array=ca,
            noise_std=0.01,
            n_dims=1,
            n_particles=100,
            rng=rng,
            k_values=(2, 3),
        )

        obs = Observation(rates=np.array([0.98, 0.95, 0.90, 0.95, 0.98]), time=0.0)
        comp.update(obs)

        result = comp.evidence()
        assert set(comp.filters) == {2, 3}
        assert set(result["log_evidence"]) == {2, 3}
        assert set(result["posterior"]) == {2, 3}
