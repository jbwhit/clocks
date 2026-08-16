"""Tests for the particle filter inference engine."""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.special import logsumexp

from clocks.inference import (
    ConvergenceInfo,
    ModelComparison,
    ParticleFilter,
    _reflect_into_bounds,
    _repair_support,
    _residual_indices,
    _state_collapsed_ess,
    _stratified_indices,
    _systematic_indices,
)
from clocks.noise import add_clock_noise, log_likelihood_gaussian
from clocks.physics import clock_rates, clock_rates_batch
from clocks.types import ClockArray, MassConfig, Observation, ParticleState


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
        # ParticleFilter is a generic numerical primitive whose legacy jitter
        # can propose negative components. Use the array kernel here rather
        # than constructing a now-strict public MassConfig for each candidate.
        return clock_rates_batch(params[:1].reshape(1, 1), params[1:2], clock_array)[0]

    return forward


class TestResamplingIndices:
    class _EndpointRng:
        """RNG double returning the largest legal uniform value below one."""

        def uniform(self, size: int | None = None) -> float | NDArray[np.float64]:
            value = np.nextafter(1.0, 0.0)
            if size is None:
                return value
            return np.full(size, value)

    @pytest.mark.parametrize(
        "helper", [_systematic_indices, _stratified_indices, _residual_indices]
    )
    @pytest.mark.parametrize(
        "weights",
        [
            np.array(1.0),
            np.array([[0.5, 0.5]]),
            np.array([]),
            np.array([0.5, np.nan]),
            np.array([0.5, np.inf]),
            np.array([1.1, -0.1]),
            np.array([0.4, 0.5]),
            np.array([0.4, 0.600000000002]),
            np.array([0.2, 0.2, 0.2, 0.2, 0.19], dtype=np.float32),
        ],
    )
    def test_rejects_invalid_weights(
        self,
        helper: Callable[
            [NDArray[np.floating], int, np.random.Generator], NDArray[np.intp]
        ],
        weights: NDArray[np.floating],
    ) -> None:
        with pytest.raises(ValueError):
            helper(weights, 5, np.random.default_rng(0))

    @pytest.mark.parametrize(
        "helper", [_systematic_indices, _stratified_indices, _residual_indices]
    )
    @pytest.mark.parametrize("n_draws", [True, False, 0, -1, 1.5])
    def test_rejects_invalid_draw_counts(
        self,
        helper: Callable[
            [NDArray[np.floating], int, np.random.Generator], NDArray[np.intp]
        ],
        n_draws: object,
    ) -> None:
        with pytest.raises(ValueError):
            helper(np.array([0.5, 0.5]), n_draws, np.random.default_rng(0))  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "helper", [_systematic_indices, _stratified_indices, _residual_indices]
    )
    def test_output_length_dtype_and_bounds(
        self,
        helper: Callable[
            [NDArray[np.floating], int, np.random.Generator], NDArray[np.intp]
        ],
    ) -> None:
        indices = helper(np.array([0.1, 0.2, 0.3, 0.4]), 11, np.random.default_rng(4))

        assert indices.shape == (11,)
        assert indices.dtype == np.intp
        assert np.all((0 <= indices) & (indices < 4))

    @pytest.mark.parametrize("helper", [_systematic_indices, _stratified_indices])
    def test_clips_against_source_length_not_draw_count(
        self,
        helper: Callable[
            [NDArray[np.floating], int, np.random.Generator], NDArray[np.intp]
        ],
    ) -> None:
        indices = helper(
            np.array([0.1, 0.1, 0.1, 0.1, 0.6]),
            1,
            np.random.default_rng(0),
        )

        np.testing.assert_array_equal(indices, np.array([4], dtype=np.intp))

    def test_residual_regression_selects_second_source_index(self) -> None:
        weights = np.array([0.20, 0.19, 0.21, 0.20, 0.20])

        indices = _residual_indices(weights, 5, np.random.default_rng(0))

        np.testing.assert_array_equal(indices, np.array([0, 2, 3, 4, 1], dtype=np.intp))

    def test_residual_empirical_frequencies_match_weights(self) -> None:
        weights = np.array([0.20, 0.19, 0.21, 0.20, 0.20])
        rng = np.random.default_rng(1234)
        counts = np.zeros(len(weights), dtype=int)

        for _ in range(2_000):
            counts += np.bincount(
                _residual_indices(weights, 5, rng), minlength=len(weights)
            )

        np.testing.assert_allclose(counts / counts.sum(), weights, atol=0.01)

    def test_residual_zero_remainder_uses_deterministic_copies(self) -> None:
        indices = _residual_indices(
            np.array([0.25, 0.5, 0.25]), 4, np.random.default_rng(0)
        )

        np.testing.assert_array_equal(indices, np.array([0, 1, 1, 2], dtype=np.intp))

    @pytest.mark.parametrize("helper", [_systematic_indices, _stratified_indices])
    def test_rounded_endpoint_never_selects_zero_weight_source(
        self,
        helper: Callable[
            [NDArray[np.floating], int, np.random.Generator], NDArray[np.intp]
        ],
    ) -> None:
        weights = np.array([0.0, 0.25, 0.0, 0.75, 0.0])

        indices = helper(weights, 4, self._EndpointRng())  # type: ignore[arg-type]

        assert np.all(weights[indices] > 0)

    def test_residual_rounded_endpoint_never_selects_zero_weight_source(
        self,
    ) -> None:
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.0])

        indices = _residual_indices(
            weights,
            4,
            self._EndpointRng(),  # type: ignore[arg-type]
        )

        assert np.all(weights[indices] > 0)

    @pytest.mark.parametrize(
        "helper", [_systematic_indices, _stratified_indices, _residual_indices]
    )
    def test_accepts_normalized_float32_weights(
        self,
        helper: Callable[
            [NDArray[np.floating], int, np.random.Generator], NDArray[np.intp]
        ],
    ) -> None:
        indices = helper(
            np.array([0.2] * 5, dtype=np.float32), 7, np.random.default_rng(0)
        )

        assert indices.shape == (7,)

    def test_rejects_large_float32_vector_summing_to_point_99(self) -> None:
        weights = np.full(100_000, 0.99 / 100_000, dtype=np.float32)
        assert float(weights.astype(np.float64).sum()) == pytest.approx(0.99)

        with pytest.raises(ValueError, match="sum to one"):
            _systematic_indices(weights, 5, np.random.default_rng(0))

    def test_rejects_zero_total_float16_without_dividing(self) -> None:
        weights = np.zeros(2_048, dtype=np.float16)

        with (
            np.errstate(all="raise"),
            pytest.raises(ValueError, match="strictly positive"),
        ):
            _systematic_indices(weights, 5, np.random.default_rng(0))

    @pytest.mark.parametrize(
        "dtype", [np.float16, np.float32, np.float64, np.longdouble]
    )
    def test_rejects_materially_non_normalized_floating_dtypes(
        self, dtype: type[np.floating]
    ) -> None:
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.19], dtype=dtype)

        with pytest.raises(ValueError, match="sum to one"):
            _systematic_indices(weights, 5, np.random.default_rng(0))


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

        # 1 call on the initial cloud (support-repair soundness) + 1 per
        # resample; resample_threshold=1.0 forces one per update.
        assert call_count[0] == 6, (
            f"Constraint: 1 init + 5 resamples expected, got {call_count[0]}"
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
        """log_prior should be called once per update to reweight, plus
        once per resample for support repair."""
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
            resample_threshold=1.0,
            rng=rng,
            log_prior=counting_prior,
        )

        for t in range(5):
            obs = Observation(rates=np.array([0.9, 0.95, 0.99]), time=float(t))
            pf.update(obs)

        # 5 reweight calls + 5 support-repair calls (threshold forces a
        # resample every update; all particles valid => one check each).
        assert call_count[0] == 10, (
            f"log_prior: 5 reweight + 5 repair calls expected, got {call_count[0]}"
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

    def test_covariance_jitter_survives_weight_collapse(self) -> None:
        """ESS≈1 must not crash covariance jitter (np.cov dof underflow).

        With a razor-thin likelihood one particle takes all the weight,
        making the weighted covariance undefined; the filter must fall
        back to isotropic jitter instead of feeding inf/NaN to cholesky.
        """

        def prior_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
            return np.column_stack(
                [np.linspace(-1.0, 1.0, n), np.linspace(0.5, 1.5, n)]
            )

        def forward(params: np.ndarray) -> np.ndarray:
            return params.copy()

        pf = ParticleFilter(
            n_particles=20,
            prior_sampler=prior_sampler,
            forward_model=forward,
            noise_std=1e-6,
            resample_threshold=1.0,
            jitter="covariance",
            jitter_std=0.05,
            rng=np.random.default_rng(0),
        )
        target = prior_sampler(np.random.default_rng(0), 20)[7]

        state = pf.update(Observation(rates=target, time=0.0))

        assert np.all(np.isfinite(state.particles))
        assert state.particles.std(axis=0).min() > 0

    def test_update_raises_when_all_particles_have_zero_weight(self) -> None:
        pf = ParticleFilter(
            n_particles=10,
            prior_sampler=lambda rng, n: rng.uniform(-1, 1, (n, 1)),
            forward_model=lambda params: np.array([1.0]),
            noise_std=0.01,
            log_prior=lambda particles: np.full(particles.shape[0], -np.inf),
            rng=np.random.default_rng(0),
        )
        obs = Observation(rates=np.array([1.0]), time=0.0)

        with pytest.raises(RuntimeError, match="zero weight"):
            pf.update(obs)

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


def _interval_log_prior(particles: np.ndarray) -> np.ndarray:
    """Support: every component in [-1, 1]."""
    lp = np.zeros(particles.shape[0])
    lp[np.any(np.abs(particles) > 1.0, axis=1)] = -np.inf
    return lp


class TestSupportRepair:
    def test_valid_proposals_pass_through_unchanged(self) -> None:
        proposals = np.array([[0.5], [-0.5]])
        parents = np.array([[0.1], [0.2]])
        out, reverted = _repair_support(
            proposals, parents, _interval_log_prior, np.random.default_rng(0)
        )
        assert np.array_equal(out, proposals)
        assert reverted is False

    def test_rejected_proposals_revert_to_parent(self) -> None:
        proposals = np.array([[0.7], [1.5]])  # second is out of support
        parents = np.array([[0.5], [0.6]])
        out, reverted = _repair_support(
            proposals, parents, _interval_log_prior, np.random.default_rng(0)
        )
        assert out[0, 0] == 0.7  # valid proposal kept (reject-and-stay)
        assert out[1, 0] == 0.6  # invalid proposal reverted to parent
        assert reverted is True

    def test_invalid_parent_replaced_from_valid_particles(self) -> None:
        proposals = np.array([[3.0], [0.4]])  # both rows: proposal 0 invalid
        parents = np.array([[2.0], [0.5]])  # ... and its parent is too
        out, reverted = _repair_support(
            proposals, parents, _interval_log_prior, np.random.default_rng(0)
        )
        assert out[0, 0] == 0.4  # safety net: drawn from the only valid particle
        assert out[1, 0] == 0.4
        assert reverted is True

    def test_raises_when_no_valid_particles_exist(self) -> None:
        proposals = np.array([[3.0], [4.0]])
        parents = np.array([[2.0], [5.0]])
        with pytest.raises(RuntimeError, match="no valid particles"):
            _repair_support(
                proposals, parents, _interval_log_prior, np.random.default_rng(0)
            )

    def test_public_state_stays_in_support_after_resample(self) -> None:
        """Huge jitter + log_prior: no out-of-support particle may survive."""
        rng = np.random.default_rng(42)
        pf = ParticleFilter(
            n_particles=200,
            prior_sampler=lambda r, n: r.uniform(-1.0, 1.0, (n, 1)),
            forward_model=lambda p: p,
            noise_std=0.1,
            resample_threshold=1.1,  # force a resample every update
            jitter_std=5.0,  # most proposals leave [-1, 1]
            jitter="fixed",
            log_prior=_interval_log_prior,
            rng=rng,
        )
        pf.update(Observation(rates=np.array([0.0]), time=0.0))
        assert np.all(np.abs(pf.state.particles) <= 1.0)

    def test_initial_cloud_constraint_applied(self) -> None:
        """Unconstrained prior sampler + sorting constraint: stored initial
        particles must already satisfy the constraint."""

        def sort_rows(particles: np.ndarray) -> np.ndarray:
            return np.sort(particles, axis=1)

        pf = ParticleFilter(
            n_particles=50,
            prior_sampler=lambda r, n: r.uniform(-1.0, 1.0, (n, 2)),
            forward_model=lambda p: p,
            noise_std=0.1,
            constraint_fn=sort_rows,
            rng=np.random.default_rng(7),
        )
        p = pf.state.particles
        assert np.all(p[:, 0] <= p[:, 1])


class TestAnnealedJitter:
    def _make_pf(self, **kwargs: object) -> ParticleFilter:
        defaults: dict = dict(
            n_particles=500,
            prior_sampler=lambda r, n: r.uniform(-8.0, 8.0, (n, 2)),
            forward_model=lambda p: p,
            noise_std=0.1,
            jitter="annealed",
            jitter_std=0.01,
            jitter_tau=5.0,
            rng=np.random.default_rng(0),
        )
        defaults.update(kwargs)
        return ParticleFilter(**defaults)

    def test_schedule_starts_at_initial_cloud_scale(self) -> None:
        pf = self._make_pf()
        expected = np.maximum(pf.state.particles.std(axis=0), 0.01)
        assert np.allclose(pf._annealed_std(0), expected)

    def test_schedule_decays_to_floor(self) -> None:
        pf = self._make_pf()
        assert np.allclose(pf._annealed_std(10_000), 0.01)

    def test_schedule_never_anneals_upward(self) -> None:
        # Tight prior (std ~0.001) with a larger floor: constant at floor.
        pf = self._make_pf(
            prior_sampler=lambda r, n: r.uniform(-0.001, 0.001, (n, 2)),
            jitter_std=0.5,
        )
        assert np.allclose(pf._annealed_std(0), 0.5)
        assert np.allclose(pf._annealed_std(100), 0.5)

    @pytest.mark.parametrize("bad_tau", [0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_jitter_tau_raises(self, bad_tau: float) -> None:
        with pytest.raises(ValueError, match="jitter_tau"):
            self._make_pf(jitter_tau=bad_tau)

    @pytest.mark.parametrize("bad_std", [-0.1, float("nan"), float("inf")])
    def test_invalid_jitter_std_raises(self, bad_std: float) -> None:
        with pytest.raises(ValueError, match="jitter_std"):
            self._make_pf(jitter_std=bad_std)

    def test_annealed_mode_converges_1d(self) -> None:
        """Annealed jitter recovers a single 1D mass (standard scenario)."""
        rng = np.random.default_rng(3)
        true_params = np.array([2.0, 0.5])
        positions = np.linspace(-6, 6, 8).reshape(-1, 1)
        ca = ClockArray(positions=positions, track_offset=3.0)
        mc = MassConfig(positions=true_params[:1].reshape(1, 1), masses=true_params[1:])
        rates = clock_rates(mc, ca)

        def forward(params: np.ndarray) -> np.ndarray:
            return clock_rates_batch(params[:1].reshape(1, 1), params[1:2], ca)[0]

        pf = ParticleFilter(
            n_particles=2000,
            prior_sampler=lambda r, n: np.column_stack(
                [r.uniform(-8, 8, n), r.uniform(0.1, 2.0, n)]
            ),
            forward_model=forward,
            noise_std=0.005,
            jitter="annealed",
            jitter_std=0.02,
            jitter_tau=5.0,
            rng=rng,
        )
        for t in range(60):
            noisy = rates + rng.normal(0, 0.005, size=rates.shape)
            pf.update(Observation(rates=noisy, time=float(t)))
        est = pf.estimate()
        assert abs(est["mean"][0] - 2.0) < 0.5
        assert abs(est["mean"][1] - 0.5) < 0.1


class TestAnnealedDefaults:
    def test_particle_filter_default_jitter_is_annealed(self) -> None:
        pf = ParticleFilter(
            n_particles=10,
            prior_sampler=lambda r, n: r.uniform(-1, 1, (n, 1)),
            forward_model=lambda p: p,
            noise_std=0.1,
        )
        assert pf.jitter == "annealed"

    def test_model_comparison_default_jitter_is_annealed(self) -> None:
        ca = ClockArray(
            positions=np.linspace(-5, 5, 6).reshape(-1, 1), track_offset=3.0
        )
        mc = ModelComparison(clock_array=ca, noise_std=0.01, k_max=2)
        assert all(pf.jitter == "annealed" for pf in mc.filters.values())

    def test_model_comparison_jitter_tau_plumbs_through(self) -> None:
        ca = ClockArray(
            positions=np.linspace(-5, 5, 6).reshape(-1, 1), track_offset=3.0
        )
        mc = ModelComparison(clock_array=ca, noise_std=0.01, k_max=2, jitter_tau=7.0)
        assert all(pf.jitter_tau == 7.0 for pf in mc.filters.values())


class TestReflectIntoBounds:
    def test_interior_points_unchanged(self) -> None:
        x = np.array([[0.5, 1.0]])
        lower = np.array([0.0, 0.0])
        upper = np.array([2.0, 2.0])
        assert np.allclose(_reflect_into_bounds(x, lower, upper), x)

    def test_single_bounce(self) -> None:
        x = np.array([[2.5]])  # 0.5 past upper=2 -> reflects to 1.5
        out = _reflect_into_bounds(x, np.array([0.0]), np.array([2.0]))
        assert np.allclose(out, [[1.5]])

    def test_repeated_reflection_handles_large_overshoot(self) -> None:
        # 5.0 in [0, 2]: period-4 triangular wave -> 5 mod 4 = 1 -> 1.0
        out = _reflect_into_bounds(np.array([[5.0]]), np.array([0.0]), np.array([2.0]))
        assert np.allclose(out, [[1.0]])
        assert 0.0 <= out[0, 0] <= 2.0

    def test_one_sided_lower_reflection(self) -> None:
        # mass-style: lower bound ~0, no upper
        out = _reflect_into_bounds(
            np.array([[-0.3]]), np.array([0.0]), np.array([np.inf])
        )
        assert np.allclose(out, [[0.3]])

    def test_unbounded_column_passes_through(self) -> None:
        out = _reflect_into_bounds(
            np.array([[-7.5]]), np.array([-np.inf]), np.array([np.inf])
        )
        assert np.allclose(out, [[-7.5]])


class TestStateCollapsedEss:
    def test_clones_share_one_group(self) -> None:
        # 4 particles, 3 identical clones with 0.3 weight each + 1 other
        particles = np.array([[1.0], [1.0], [1.0], [2.0]])
        weights = np.array([0.3, 0.3, 0.3, 0.1])
        # groups: {1.0: 0.9, 2.0: 0.1} -> 1/(0.81+0.01) ~= 1.2195
        ess = _state_collapsed_ess(particles, weights)
        assert abs(ess - 1.0 / 0.82) < 1e-9

    def test_all_distinct_matches_ordinary_ess(self) -> None:
        particles = np.arange(4.0).reshape(-1, 1)
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        assert abs(_state_collapsed_ess(particles, weights) - 4.0) < 1e-9


class TestSupportBoundsReflection:
    def _make_pf(self, **kwargs: object) -> ParticleFilter:
        defaults: dict = dict(
            n_particles=400,
            prior_sampler=lambda r, n: r.uniform(0.2, 0.8, (n, 2)),
            forward_model=lambda p: p,
            noise_std=0.1,
            resample_threshold=1.1,  # resample every update
            jitter="fixed",
            jitter_std=5.0,  # most proposals leave [0, 1] without repair
            support_bounds=(np.array([0.0, 0.0]), np.array([1.0, 1.0])),
            rng=np.random.default_rng(0),
        )
        defaults.update(kwargs)
        return ParticleFilter(**defaults)

    def test_reflection_keeps_diagonal_modes_in_bounds(self) -> None:
        pf = self._make_pf()
        pf.update(Observation(rates=np.array([0.5, 0.5]), time=0.0))
        p = pf.state.particles
        assert np.all((p >= 0.0) & (p <= 1.0))
        # reflection must not create clone pileups
        assert len(np.unique(p, axis=0)) == pf.n_particles

    def test_reflection_runs_before_constraint(self) -> None:
        pf = self._make_pf(constraint_fn=lambda p: np.sort(p, axis=1))
        pf.update(Observation(rates=np.array([0.5, 0.5]), time=0.0))
        p = pf.state.particles
        assert np.all((p >= 0.0) & (p <= 1.0))
        assert np.all(p[:, 0] <= p[:, 1])

    def test_bounds_contradicting_log_prior_raise(self) -> None:
        def strict_log_prior(particles: np.ndarray) -> np.ndarray:
            lp = np.zeros(particles.shape[0])
            lp[np.any(particles > 0.5, axis=1)] = -np.inf  # tighter than bounds
            return lp

        pf = self._make_pf(log_prior=strict_log_prior)
        with pytest.raises(RuntimeError, match="support_bounds"):
            pf.update(Observation(rates=np.array([0.2, 0.2]), time=0.0))

    def test_covariance_mode_keeps_reject_and_stay(self) -> None:
        """Reflection is not valid for correlated kernels; covariance mode
        must still repair via reject-and-stay (particles stay in support
        via log_prior, not bounds reflection)."""
        pf = self._make_pf(
            jitter="covariance",
            jitter_std=0.5,
            log_prior=_interval_log_prior,  # support [-1, 1] on all comps
            support_bounds=None,
        )
        pf.update(Observation(rates=np.array([0.5, 0.5]), time=0.0))
        assert np.all(np.abs(pf.state.particles) <= 1.0)


class TestCloneAwareResampleTrigger:
    def test_state_collapsed_trigger_fires_on_clone_majority(self) -> None:
        """A clone-majority cloud with high weight-ESS must still resample
        after a repair reverted proposals (the seed-101 freeze shape)."""
        pf = ParticleFilter(
            n_particles=100,
            prior_sampler=lambda r, n: r.uniform(-1.0, 1.0, (n, 1)),
            forward_model=lambda p: p,
            noise_std=0.5,
            resample_threshold=0.5,
            jitter="fixed",
            jitter_std=0.01,
            log_prior=_interval_log_prior,
            rng=np.random.default_rng(1),
        )
        clones = np.full((80, 1), 0.3)
        scattered = np.linspace(-0.9, 0.9, 20).reshape(-1, 1)
        pf._state = ParticleState(
            particles=np.vstack([clones, scattered]),
            weights=np.ones(100) / 100,
            observations_seen=1,
        )
        pf._repair_reverted = True
        state = pf.update(Observation(rates=np.array([0.3]), time=1.0))
        # weight-ESS of ~80 equally-weighted clones stays over threshold
        # (50); the state-collapsed backstop must fire the resample, which
        # re-diversifies the cloud via jitter.
        assert len(np.unique(state.particles, axis=0)) > 25
