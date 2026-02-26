"""Particle filter (Sequential Monte Carlo) for mass inference."""

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from clocks.noise import log_likelihood_gaussian, log_likelihood_gaussian_batch
from clocks.types import Observation, ParticleState


class ParticleFilter:
    """Sequential Monte Carlo filter for inferring mass parameters.

    Infers mass parameters from noisy clock observations.

    Each particle represents a hypothesis for the unknown parameters
    (e.g., mass position and magnitude). Observations reweight particles
    via the forward model likelihood, and resampling prevents degeneracy.

    Parameters
    ----------
    n_particles : Number of particles.
    prior_sampler : Callable(rng, n) → (n, n_params) array of prior samples.
    forward_model : Callable(params) → (n_clocks,) predicted rates for one particle.
    noise_std : Observation noise standard deviation.
    resample_threshold : Resample when ESS / n_particles drops below this.
    jitter_std : Std of Gaussian jitter applied after resampling.
    rng : Numpy random generator.
    forward_model_batch : Optional Callable(particles) → (n_particles, n_clocks).
        If provided, used instead of looping forward_model per particle.
    """

    def __init__(
        self,
        n_particles: int,
        prior_sampler: Callable[[np.random.Generator, int], NDArray[np.floating]],
        forward_model: Callable[[NDArray[np.floating]], NDArray[np.floating]],
        noise_std: float,
        resample_threshold: float = 0.5,
        jitter_std: float = 0.01,
        rng: np.random.Generator | None = None,
        forward_model_batch: Callable[[NDArray[np.floating]], NDArray[np.floating]]
        | None = None,
        constraint_fn: Callable[[NDArray[np.floating]], NDArray[np.floating]]
        | None = None,
    ) -> None:
        self.n_particles = n_particles
        self.forward_model = forward_model
        self.forward_model_batch = forward_model_batch
        self.noise_std = noise_std
        self.resample_threshold = resample_threshold
        self.jitter_std = jitter_std
        self.constraint_fn = constraint_fn
        self.rng = rng or np.random.default_rng()

        particles = prior_sampler(self.rng, n_particles)
        weights = np.ones(n_particles) / n_particles
        self._state = ParticleState(
            particles=particles,
            weights=weights,
            observations_seen=0,
        )
        self._history: list[ParticleState] = [self._state]

    @property
    def state(self) -> ParticleState:
        return self._state

    @property
    def history(self) -> list[ParticleState]:
        return list(self._history)

    def update(self, observation: Observation) -> ParticleState:
        """Incorporate one observation: reweight particles, resample if needed."""
        particles = self._state.particles
        weights = self._state.weights.copy()

        # Reweight each particle by its likelihood
        log_weights = np.log(weights)
        if self.forward_model_batch is not None:
            predicted_batch = self.forward_model_batch(particles)
            log_weights += log_likelihood_gaussian_batch(
                observation.rates, predicted_batch, self.noise_std
            )
        else:
            for i in range(self.n_particles):
                predicted = self.forward_model(particles[i])
                ll = log_likelihood_gaussian(
                    observation.rates, predicted, self.noise_std
                )
                log_weights[i] += ll

        # Normalize weights (log-sum-exp for numerical stability)
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        weights /= weights.sum()

        # Resample if effective sample size is too low
        ess = 1.0 / np.sum(weights**2)
        if ess < self.resample_threshold * self.n_particles:
            particles, weights = self._resample(particles, weights)

        self._state = ParticleState(
            particles=particles,
            weights=weights,
            observations_seen=self._state.observations_seen + 1,
        )
        self._history.append(self._state)
        return self._state

    def _resample(
        self,
        particles: NDArray[np.floating],
        weights: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Systematic resampling with jitter."""
        n = self.n_particles
        cumsum = np.cumsum(weights)
        u = (self.rng.uniform() + np.arange(n)) / n
        indices = np.searchsorted(cumsum, u)
        indices = np.clip(indices, 0, n - 1)

        new_particles = particles[indices].copy()
        new_particles += self.rng.normal(0, self.jitter_std, size=new_particles.shape)
        if self.constraint_fn is not None:
            new_particles = self.constraint_fn(new_particles)
        new_weights = np.ones(n) / n
        return new_particles, new_weights

    def estimate(self) -> dict[str, Any]:
        """Weighted mean and standard deviation of current particles."""
        p = self._state.particles
        w = self._state.weights
        mean = np.average(p, weights=w, axis=0)
        var = np.average((p - mean) ** 2, weights=w, axis=0)
        return {
            "mean": mean,
            "std": np.sqrt(var),
            "ess": 1.0 / np.sum(w**2),
        }
