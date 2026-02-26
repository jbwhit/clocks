"""Observation noise model for clock measurements."""

import numpy as np
from numpy.typing import NDArray


def add_clock_noise(
    true_rates: NDArray[np.floating],
    noise_std: float,
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    """Add Gaussian noise to true clock rates.

    Returns: (n_clocks,) array of noisy observed rates.
    """
    return true_rates + rng.normal(0.0, noise_std, size=true_rates.shape)


def log_likelihood_gaussian(
    observed: NDArray[np.floating],
    predicted: NDArray[np.floating],
    noise_std: float,
) -> float:
    """Log-likelihood of observed rates given predicted rates and Gaussian noise.

    Returns: scalar log-likelihood (sum over clocks).
    """
    residuals = observed - predicted
    return float(
        -0.5 * np.sum((residuals / noise_std) ** 2)
        - len(residuals) * np.log(noise_std * np.sqrt(2 * np.pi))
    )


def log_likelihood_gaussian_batch(
    observed: NDArray[np.floating],
    predicted_batch: NDArray[np.floating],
    noise_std: float,
) -> NDArray[np.floating]:
    """Batch log-likelihood: one observed vector vs many predicted vectors.

    observed: (n_clocks,)
    predicted_batch: (n_particles, n_clocks)
    Returns: (n_particles,) log-likelihoods.
    """
    residuals = observed[np.newaxis, :] - predicted_batch
    n_clocks = observed.shape[0]
    return -0.5 * np.sum((residuals / noise_std) ** 2, axis=1) - n_clocks * np.log(
        noise_std * np.sqrt(2 * np.pi)
    )
