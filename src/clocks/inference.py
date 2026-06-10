"""Particle filter (Sequential Monte Carlo) for mass inference."""

from collections.abc import Callable
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cholesky

from clocks.noise import log_likelihood_gaussian, log_likelihood_gaussian_batch
from clocks.physics import clock_rates, clock_rates_batch, clock_rates_batch_multi
from clocks.types import ClockArray, MassConfig, Observation, ParticleState

_RESAMPLING_METHODS = {"systematic", "stratified", "residual"}
_JITTER_MODES = {"fixed", "covariance"}


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
    jitter_std : Scale of the post-resampling jitter. With ``jitter="fixed"``
        this is an absolute standard deviation applied isotropically; with
        ``jitter="covariance"`` it scales the Cholesky factor of the weighted
        empirical covariance, so 0.02 means "2% of the cloud's own spread
        along its correlation structure".
    rng : Numpy random generator.
    forward_model_batch : Optional Callable(particles) → (n_particles, n_clocks).
        If provided, used instead of looping forward_model per particle.
    jitter : Jitter mode after resampling. ``"fixed"`` uses isotropic Gaussian
        noise; ``"covariance"`` draws from the weighted empirical covariance
        so correlated parameters jitter along their joint structure.
    log_prior : Optional Callable(particles) → (n_particles,) log-prior values.
        Particles with ``-inf`` log-prior get zero weight.  Use this instead
        of constraint clamping for boundary enforcement.
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
        jitter: str = "fixed",
        log_prior: Callable[[NDArray[np.floating]], NDArray[np.floating]] | None = None,
    ) -> None:
        if resampling not in _RESAMPLING_METHODS:
            msg = (
                f"Unknown resampling method {resampling!r},"
                f" expected one of {sorted(_RESAMPLING_METHODS)}"
            )
            raise ValueError(msg)
        if jitter not in _JITTER_MODES:
            msg = (
                f"Unknown jitter mode {jitter!r},"
                f" expected one of {sorted(_JITTER_MODES)}"
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
        self.jitter = jitter
        self.log_prior = log_prior
        self.rng = rng or np.random.default_rng()
        self.log_evidence: float = 0.0

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

        # Reweight each particle by its likelihood.
        # log(0) → -inf is intended: a zero-weight particle stays dead.
        with np.errstate(divide="ignore"):
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

        # Add log-prior (gives -inf weight to invalid particles)
        if self.log_prior is not None:
            log_weights += self.log_prior(particles)

        # Normalize weights (log-sum-exp for numerical stability)
        max_lw = np.max(log_weights)
        log_weights -= max_lw
        weights = np.exp(log_weights)
        self.log_evidence += max_lw + np.log(weights.sum())
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

        if self.jitter == "covariance":
            cov = np.cov(particles.T, aweights=weights)
            n_params = particles.shape[1]
            if n_params == 1:
                cov = cov.reshape(1, 1)
            cov += 1e-10 * np.eye(n_params)
            L = cholesky(cov, lower=True)
            z = self.rng.normal(0, 1, size=new_particles.shape)
            new_particles += self.jitter_std * (z @ L.T)
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


class ModelComparisonResult(TypedDict):
    """Return type for ModelComparison.evidence()."""

    log_evidence: dict[int, float]
    posterior: dict[int, float]


class ModelComparison:
    """Bayesian model comparison over number of point masses.

    Runs parallel particle filters for K=1..k_max and compares
    accumulated log-evidence to infer the most likely number of masses.
    """

    def __init__(
        self,
        clock_array: ClockArray,
        noise_std: float,
        n_dims: int = 1,
        k_max: int = 3,
        k_values: tuple[int, ...] | None = None,
        n_particles: int = 1000,
        jitter_std: float = 0.02,
        position_range: tuple[float, float] = (-8.0, 8.0),
        mass_range: tuple[float, float] = (0.1, 2.0),
        rng: np.random.Generator | None = None,
        resampling: str = "systematic",
        jitter: str = "fixed",
    ) -> None:
        self.clock_array = clock_array
        self.noise_std = noise_std
        self.n_dims = n_dims
        if k_values is None:
            self.k_values = tuple(range(1, k_max + 1))
        else:
            if not k_values:
                raise ValueError("k_values must not be empty")
            if any(k <= 0 for k in k_values):
                raise ValueError("k_values must all be > 0")
            self.k_values = tuple(sorted(set(k_values)))
        self.k_max = max(self.k_values)
        self.position_range = position_range
        self.rng = rng or np.random.default_rng()

        self.filters: dict[int, ParticleFilter] = {}
        for k in self.k_values:
            self.filters[k] = ParticleFilter(
                n_particles=n_particles,
                prior_sampler=self._make_prior_sampler(k, position_range, mass_range),
                forward_model=self._make_forward_model(k),
                noise_std=noise_std,
                jitter_std=jitter_std,
                rng=np.random.default_rng(self.rng.integers(2**63)),
                forward_model_batch=self._make_forward_model_batch(k),
                constraint_fn=self._make_constraint_fn(k) if k > 1 else None,
                resampling=resampling,
                jitter=jitter,
                log_prior=self._make_log_prior(k, position_range, mass_range),
            )

    def _make_prior_sampler(
        self,
        k: int,
        position_range: tuple[float, float],
        mass_range: tuple[float, float],
    ) -> Callable[[np.random.Generator, int], NDArray[np.floating]]:
        """Create prior sampler for K masses in n_dims dimensions."""
        n_dims = self.n_dims

        def sampler(rng: np.random.Generator, n: int) -> NDArray[np.floating]:
            # Positions: (n, K*n_dims)
            positions = rng.uniform(
                position_range[0], position_range[1], (n, k * n_dims)
            )
            # Enforce ordering on dim-0 for K>1
            if k > 1:
                pos_reshaped = positions.reshape(n, k, n_dims)
                sort_idx = np.argsort(pos_reshaped[:, :, 0], axis=1)
                for i in range(n_dims):
                    col = pos_reshaped[:, :, i]
                    pos_reshaped[:, :, i] = np.take_along_axis(col, sort_idx, axis=1)
                positions = pos_reshaped.reshape(n, k * n_dims)
            # Masses: (n, K)
            masses = rng.uniform(mass_range[0], mass_range[1], (n, k))
            return np.column_stack([positions, masses])

        return sampler

    def _make_forward_model(
        self, k: int
    ) -> Callable[[NDArray[np.floating]], NDArray[np.floating]]:
        """Create scalar forward model for K masses."""
        n_dims = self.n_dims
        ca = self.clock_array

        def forward(params: NDArray[np.floating]) -> NDArray[np.floating]:
            positions = params[: k * n_dims].reshape(k, n_dims)
            masses = params[k * n_dims :]
            mc = MassConfig(positions=positions, masses=masses)
            return clock_rates(mc, ca)

        return forward

    def _make_forward_model_batch(
        self, k: int
    ) -> Callable[[NDArray[np.floating]], NDArray[np.floating]]:
        """Create batch forward model for K masses."""
        n_dims = self.n_dims
        ca = self.clock_array

        if k == 1:

            def batch_single(
                particles: NDArray[np.floating],
            ) -> NDArray[np.floating]:
                return clock_rates_batch(
                    particles[:, :n_dims], particles[:, n_dims], ca
                )

            return batch_single

        def batch_multi(particles: NDArray[np.floating]) -> NDArray[np.floating]:
            pos = particles[:, : k * n_dims].reshape(-1, k, n_dims)
            masses = particles[:, k * n_dims :]
            return clock_rates_batch_multi(pos, masses, ca)

        return batch_multi

    def _make_constraint_fn(
        self, k: int
    ) -> Callable[[NDArray[np.floating]], NDArray[np.floating]]:
        """Create ordering constraint for K>1 masses (sort by first spatial dim)."""
        n_dims = self.n_dims

        def constraint(particles: NDArray[np.floating]) -> NDArray[np.floating]:
            n = particles.shape[0]
            pos = particles[:, : k * n_dims].reshape(n, k, n_dims)
            masses = particles[:, k * n_dims :].reshape(n, k)
            sort_idx = np.argsort(pos[:, :, 0], axis=1)
            for i in range(n_dims):
                col = pos[:, :, i]
                pos[:, :, i] = np.take_along_axis(col, sort_idx, axis=1)
            masses = np.take_along_axis(masses, sort_idx, axis=1)
            particles[:, : k * n_dims] = pos.reshape(n, k * n_dims)
            particles[:, k * n_dims :] = masses
            return particles

        return constraint

    def _make_log_prior(
        self,
        k: int,
        position_range: tuple[float, float],
        mass_range: tuple[float, float],
    ) -> Callable[[NDArray[np.floating]], NDArray[np.floating]]:
        """Log-prior: -inf for negative masses or out-of-range positions."""
        n_dims = self.n_dims

        def log_prior(particles: NDArray[np.floating]) -> NDArray[np.floating]:
            lp = np.zeros(particles.shape[0])
            # Check positions
            positions = particles[:, : k * n_dims]
            out_of_range = np.any(
                (positions < position_range[0]) | (positions > position_range[1]),
                axis=1,
            )
            lp[out_of_range] = -np.inf
            # Check masses
            masses = particles[:, k * n_dims :]
            invalid_mass = np.any(masses <= 0, axis=1)
            lp[invalid_mass] = -np.inf
            return lp

        return log_prior

    def update(self, observation: Observation) -> None:
        """Feed one observation to all filters."""
        for pf in self.filters.values():
            pf.update(observation)

    def evidence(self) -> ModelComparisonResult:
        """Per-K log-evidence and posterior probabilities (uniform prior over K)."""
        log_ev = {k: pf.log_evidence for k, pf in self.filters.items()}

        # Compute posterior via logsumexp + normalize
        ks = sorted(log_ev)
        log_vals = np.array([log_ev[k] for k in ks])
        max_log = np.max(log_vals)
        log_norm = max_log + np.log(np.sum(np.exp(log_vals - max_log)))
        posterior = {k: float(np.exp(log_ev[k] - log_norm)) for k in ks}

        return {"log_evidence": log_ev, "posterior": posterior}

    def estimate(self, k: int | None = None) -> Estimate:
        """Weighted mean/std for a specific K, or the MAP model if k is None."""
        if k is None:
            result = self.evidence()
            k = max(result["posterior"], key=lambda x: result["posterior"][x])
        if k not in self.filters:
            msg = f"No filter for K={k}, available: {sorted(self.filters)}"
            raise ValueError(msg)
        return self.filters[k].estimate()
