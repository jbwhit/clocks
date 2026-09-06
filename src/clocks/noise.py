"""Observation noise model for clock measurements."""

import numpy as np
from numpy.typing import NDArray

from clocks._validation import finite_float, finite_float_array


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

    Inputs must be finite matching nonempty 1-D channel vectors, with a
    finite, strictly positive noise standard deviation.
    Returns: scalar log-likelihood (sum over clocks).
    """
    observed = finite_float_array("observed", observed, ndim=1)
    predicted = finite_float_array("predicted", predicted, ndim=1)
    if predicted.shape != observed.shape:
        raise ValueError(
            f"predicted must have shape {observed.shape}, got {predicted.shape}"
        )
    noise_std = finite_float("noise_std", noise_std)
    if noise_std <= 0:
        raise ValueError("noise_std must be > 0")
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
    All inputs must be finite; noise_std must be strictly positive.
    Returns: (n_particles,) log-likelihoods.
    """
    observed = finite_float_array("observed", observed, ndim=1)
    predicted_batch = finite_float_array(
        "predicted_batch", predicted_batch, ndim=2, nonempty=False
    )
    if predicted_batch.shape[1] != observed.size:
        raise ValueError(
            f"predicted_batch must have {observed.size} channels, "
            f"got shape {predicted_batch.shape}"
        )
    noise_std = finite_float("noise_std", noise_std)
    if noise_std <= 0:
        raise ValueError("noise_std must be > 0")
    residuals = observed[np.newaxis, :] - predicted_batch
    n_clocks = observed.shape[0]
    return -0.5 * np.sum((residuals / noise_std) ** 2, axis=1) - n_clocks * np.log(
        noise_std * np.sqrt(2 * np.pi)
    )
