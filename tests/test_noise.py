"""Tests for the observation noise model."""

import numpy as np
import pytest

from clocks.noise import (
    add_clock_noise,
    log_likelihood_gaussian,
    log_likelihood_gaussian_batch,
)


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


class TestLogLikelihoodBatch:
    def test_matches_scalar(self) -> None:
        """Batch result should match calling scalar version per row."""
        observed = np.array([0.9, 0.95, 0.99])
        predicted_batch = np.array(
            [
                [0.9, 0.95, 0.99],
                [0.88, 0.93, 0.97],
                [0.92, 0.96, 1.00],
            ]
        )
        noise_std = 0.01
        batch_ll = log_likelihood_gaussian_batch(observed, predicted_batch, noise_std)
        scalar_ll = np.array(
            [
                log_likelihood_gaussian(observed, predicted_batch[i], noise_std)
                for i in range(3)
            ]
        )
        np.testing.assert_allclose(batch_ll, scalar_ll)

    def test_shape(self) -> None:
        observed = np.array([0.9, 0.95])
        predicted_batch = np.ones((50, 2)) * 0.9
        result = log_likelihood_gaussian_batch(observed, predicted_batch, 0.01)
        assert result.shape == (50,)


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("noise_std", [0.0, -0.01, np.nan, np.inf])
def test_likelihood_rejects_invalid_noise(batch: bool, noise_std: float) -> None:
    observed = np.array([0.9, 0.95, 0.99])
    function = log_likelihood_gaussian_batch if batch else log_likelihood_gaussian
    predicted = observed[None, :] if batch else observed
    with pytest.raises(ValueError, match="noise_std"):
        function(observed, predicted, noise_std)


@pytest.mark.parametrize("batch", [False, True])
def test_likelihood_rejects_broadcast_channel_mismatch(batch: bool) -> None:
    function = log_likelihood_gaussian_batch if batch else log_likelihood_gaussian
    predicted = np.array([[0.9]]) if batch else np.array([0.9])
    with pytest.raises(ValueError, match="shape|channel"):
        function(np.array([0.9, 0.95, 0.99]), predicted, 0.01)


@pytest.mark.parametrize("batch", [False, True])
def test_likelihood_rejects_column_vector_observation(batch: bool) -> None:
    function = log_likelihood_gaussian_batch if batch else log_likelihood_gaussian
    predicted = np.array([[0.9, 0.95]]) if batch else np.array([0.9, 0.95])
    with pytest.raises(ValueError, match="1-D"):
        function(np.array([[0.9], [0.95]]), predicted, 0.01)
