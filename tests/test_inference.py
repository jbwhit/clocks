"""Tests for the particle filter inference engine."""

import numpy as np

from clocks.inference import ParticleFilter
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
) -> callable:
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
