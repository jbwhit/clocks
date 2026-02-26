"""Tests for the observation noise model."""

import numpy as np

from clocks.noise import add_clock_noise, log_likelihood_gaussian


class TestAddClockNoise:
    def test_shape_preserved(self) -> None:
        rng = np.random.default_rng(42)
        rates = np.array([0.9, 0.95, 0.99])
        noisy = add_clock_noise(rates, noise_std=0.01, rng=rng)
        assert noisy.shape == rates.shape

    def test_zero_noise_returns_true(self) -> None:
        rng = np.random.default_rng(42)
        rates = np.array([0.9, 0.95, 0.99])
        noisy = add_clock_noise(rates, noise_std=0.0, rng=rng)
        np.testing.assert_allclose(noisy, rates)

    def test_noise_has_correct_std(self) -> None:
        rng = np.random.default_rng(42)
        rates = np.ones(10_000) * 0.9
        noise_std = 0.05
        noisy = add_clock_noise(rates, noise_std=noise_std, rng=rng)
        empirical_std = np.std(noisy - rates)
        np.testing.assert_allclose(empirical_std, noise_std, atol=0.005)


class TestLogLikelihood:
    def test_maximized_at_truth(self) -> None:
        observed = np.array([0.9, 0.95, 0.99])
        noise_std = 0.01
        ll_at_truth = log_likelihood_gaussian(observed, observed, noise_std)
        ll_away = log_likelihood_gaussian(observed, observed + 0.05, noise_std)
        assert ll_at_truth > ll_away

    def test_higher_noise_lower_peak(self) -> None:
        observed = np.array([0.9, 0.95])
        ll_tight = log_likelihood_gaussian(observed, observed, noise_std=0.01)
        ll_loose = log_likelihood_gaussian(observed, observed, noise_std=0.1)
        # Tighter noise → higher peak likelihood
        assert ll_tight > ll_loose

    def test_symmetric(self) -> None:
        observed = np.array([0.9])
        noise_std = 0.01
        ll_above = log_likelihood_gaussian(observed, observed + 0.02, noise_std)
        ll_below = log_likelihood_gaussian(observed, observed - 0.02, noise_std)
        np.testing.assert_allclose(ll_above, ll_below)
