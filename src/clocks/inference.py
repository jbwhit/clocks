"""Particle filter (Sequential Monte Carlo) for mass inference."""

from collections.abc import Callable
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from clocks.noise import log_likelihood_gaussian, log_likelihood_gaussian_batch
from clocks.types import Observation, ParticleState

_RESAMPLING_METHODS = {"systematic", "stratified", "residual"}


class Estimate(TypedDict):
    """Return type for ParticleFilter.estimate()."""

    mean: NDArray[np.floating]
    std: NDArray[np.floating]
    ess: float


class ConvergenceInfo(TypedDict):
    """Return type for ParticleFilter.converged()."""

    converged: bool
    per_param_std: NDArray[np.floating]
    per_param_converged: NDArray[np.bool_]
    ess: float
    estimates_stable: bool


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
        resampling: str = "systematic",
        adaptive_jitter: bool = False,
    ) -> None:
        if resampling not in _RESAMPLING_METHODS:
            msg = (
                f"Unknown resampling method {resampling!r},"
                f" expected one of {sorted(_RESAMPLING_METHODS)}"
            )
            raise ValueError(msg)

        self.n_particles = n_particles
        self.forward_model = forward_model
        self.forward_model_batch = forward_model_batch
        self.noise_std = noise_std
        self.resample_threshold = resample_threshold
        self.jitter_std = jitter_std
        self.constraint_fn = constraint_fn
        self.resampling = resampling
        self.adaptive_jitter = adaptive_jitter
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

    def _systematic_indices(
        self, weights: NDArray[np.floating], n: int
    ) -> NDArray[np.intp]:
        cumsum = np.cumsum(weights)
        u = (self.rng.uniform() + np.arange(n)) / n
        return np.clip(np.searchsorted(cumsum, u), 0, n - 1)

    def _stratified_indices(
        self, weights: NDArray[np.floating], n: int
    ) -> NDArray[np.intp]:
        cumsum = np.cumsum(weights)
        u = (self.rng.uniform(size=n) + np.arange(n)) / n
        return np.clip(np.searchsorted(cumsum, u), 0, n - 1)

    def _residual_indices(
        self, weights: NDArray[np.floating], n: int
    ) -> NDArray[np.intp]:
        counts = np.floor(n * weights).astype(int)
        remainder = n - counts.sum()
        residual_w = n * weights - counts
        if remainder > 0:
            residual_w /= residual_w.sum()
            extra = self._systematic_indices(residual_w, remainder)
            indices = np.concatenate([np.repeat(np.arange(n), counts), extra])
        else:
            indices = np.repeat(np.arange(n), counts)
        return indices.astype(np.intp)

    def _resample(
        self,
        particles: NDArray[np.floating],
        weights: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Resample particles with jitter."""
        n = self.n_particles

        if self.resampling == "stratified":
            indices = self._stratified_indices(weights, n)
        elif self.resampling == "residual":
            indices = self._residual_indices(weights, n)
        else:
            indices = self._systematic_indices(weights, n)

        new_particles = particles[indices].copy()

        if self.adaptive_jitter:
            mean = np.average(particles, weights=weights, axis=0)
            var = np.average((particles - mean) ** 2, weights=weights, axis=0)
            jitter_scale = self.jitter_std * np.maximum(np.sqrt(var), 1e-10)
            new_particles += (
                self.rng.normal(0, 1, size=new_particles.shape) * jitter_scale
            )
        else:
            new_particles += self.rng.normal(
                0, self.jitter_std, size=new_particles.shape
            )

        if self.constraint_fn is not None:
            new_particles = self.constraint_fn(new_particles)
        new_weights = np.ones(n) / n
        return new_particles, new_weights

    def estimate(self) -> Estimate:
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

    def converged(
        self,
        std_threshold: float = 0.01,
        window: int = 10,
        stability_threshold: float = 0.001,
    ) -> ConvergenceInfo:
        """Check whether the filter has converged.

        Two criteria must both be met:
        1. All per-parameter weighted stds are below ``std_threshold``.
        2. The max step-to-step change in the weighted mean over the last
           ``window`` states is below ``stability_threshold``.
        """
        est = self.estimate()
        per_param_std = est["std"]
        per_param_converged = per_param_std < std_threshold
        ess = est["ess"]

        # Stability: check recent means aren't changing
        if len(self._history) < window:
            estimates_stable = False
        else:
            recent = self._history[-window:]
            means = np.array(
                [np.average(s.particles, weights=s.weights, axis=0) for s in recent]
            )
            max_change = np.max(np.abs(np.diff(means, axis=0)))
            estimates_stable = bool(max_change < stability_threshold)

        return {
            "converged": bool(np.all(per_param_converged)) and estimates_stable,
            "per_param_std": per_param_std,
            "per_param_converged": per_param_converged,
            "ess": ess,
            "estimates_stable": estimates_stable,
        }
