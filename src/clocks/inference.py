"""Particle filter (Sequential Monte Carlo) for mass inference."""

import math
from collections.abc import Callable
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cholesky

from clocks.noise import log_likelihood_gaussian, log_likelihood_gaussian_batch
from clocks.physics import clock_rates, clock_rates_batch, clock_rates_batch_multi
from clocks.types import ClockArray, MassConfig, Observation, ParticleState

_RESAMPLING_METHODS = {"systematic", "stratified", "residual"}
_JITTER_MODES = {"fixed", "covariance", "annealed"}
_WEIGHT_SUM_ATOL = 1e-12
_UNIT_INTERVAL_MAX = np.nextafter(1.0, 0.0)


def _validated_resampling_inputs(
    weights: NDArray[np.floating], n_draws: int
) -> tuple[NDArray[np.float64], int]:
    """Validate inputs, allowing dtype-scaled summation roundoff only."""
    source_weights = np.asarray(weights)
    if source_weights.ndim != 1:
        raise ValueError("weights must be a 1-D array")
    if source_weights.size == 0:
        raise ValueError("weights must be nonempty")
    weights_array = np.asarray(source_weights, dtype=float)
    if not np.all(np.isfinite(weights_array)):
        raise ValueError("weights must be finite")
    if np.any(weights_array < 0):
        raise ValueError("weights must be nonnegative")

    total = float(weights_array.sum())
    source_epsilon = (
        np.finfo(source_weights.dtype).eps
        if np.issubdtype(source_weights.dtype, np.floating)
        else np.finfo(np.float64).eps
    )
    sum_atol = max(_WEIGHT_SUM_ATOL, source_weights.size * source_epsilon)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=sum_atol):
        raise ValueError(f"weights must sum to one within {sum_atol:g}, got {total}")
    if isinstance(n_draws, (bool, np.bool_)) or not isinstance(
        n_draws, (int, np.integer)
    ):
        raise ValueError("n_draws must be a positive integer")
    if n_draws <= 0:
        raise ValueError("n_draws must be a positive integer")

    return weights_array / total, int(n_draws)


def _systematic_indices(
    weights: NDArray[np.floating], n_draws: int, rng: np.random.Generator
) -> NDArray[np.intp]:
    """Draw source indices using systematic resampling."""
    weights_array, n_draws = _validated_resampling_inputs(weights, n_draws)
    cumsum = np.cumsum(weights_array)
    cumsum[-1] = 1.0
    positions = (rng.uniform() + np.arange(n_draws)) / n_draws
    positions = np.minimum(positions, _UNIT_INTERVAL_MAX)
    indices = np.searchsorted(cumsum, positions, side="right")
    return np.clip(indices, 0, len(weights_array) - 1).astype(np.intp)


def _stratified_indices(
    weights: NDArray[np.floating], n_draws: int, rng: np.random.Generator
) -> NDArray[np.intp]:
    """Draw source indices using stratified resampling."""
    weights_array, n_draws = _validated_resampling_inputs(weights, n_draws)
    cumsum = np.cumsum(weights_array)
    cumsum[-1] = 1.0
    positions = (rng.uniform(size=n_draws) + np.arange(n_draws)) / n_draws
    positions = np.minimum(positions, _UNIT_INTERVAL_MAX)
    indices = np.searchsorted(cumsum, positions, side="right")
    return np.clip(indices, 0, len(weights_array) - 1).astype(np.intp)


def _residual_indices(
    weights: NDArray[np.floating], n_draws: int, rng: np.random.Generator
) -> NDArray[np.intp]:
    """Draw source indices using deterministic and systematic residual draws."""
    weights_array, n_draws = _validated_resampling_inputs(weights, n_draws)
    expected_counts = n_draws * weights_array
    counts = np.floor(expected_counts).astype(np.intp)
    remainder = n_draws - int(counts.sum())
    deterministic = np.repeat(np.arange(len(weights_array), dtype=np.intp), counts)
    if remainder == 0:
        return deterministic.astype(np.intp, copy=False)

    fractional = expected_counts - counts
    fractional /= fractional.sum()
    stochastic = _systematic_indices(fractional, remainder, rng)
    return np.concatenate((deterministic, stochastic)).astype(np.intp, copy=False)


def _repair_support(
    proposals: NDArray[np.floating],
    parents: NDArray[np.floating],
    log_prior: Callable[[NDArray[np.floating]], NDArray[np.floating]],
    rng: np.random.Generator,
) -> tuple[NDArray[np.floating], bool]:
    """One-shot reject-and-stay support repair for post-jitter proposals.

    Proposals with -inf log-prior revert to their resampled parent's value.
    Parents that are themselves invalid are replaced by a uniform draw from
    the valid repaired particles. Deliberately NOT retry-until-valid:
    retrying samples a parent-dependent truncated proposal that biases
    particles away from support boundaries.

    Returns the repaired particles and whether anything was reverted. The
    caller uses the flag to trigger the clone-aware ESS backstop (see
    ``_state_collapsed_ess``): reverting many proposals to a dominant
    parent can create a clone-majority cloud whose weight-ESS stays high
    despite collapsed state diversity (the clone-freeze degeneracy).
    """
    repaired = proposals.copy()
    invalid = np.isneginf(log_prior(repaired))
    if not invalid.any():
        return repaired, False
    repaired[invalid] = parents[invalid]
    still_invalid = np.isneginf(log_prior(repaired))
    if still_invalid.any():
        valid_idx = np.flatnonzero(~still_invalid)
        if valid_idx.size == 0:
            raise RuntimeError(
                "Support repair failed: no valid particles remain after "
                "reverting to parents; prior support and proposals are "
                "fully disjoint"
            )
        donors = rng.choice(valid_idx, size=int(still_invalid.sum()))
        repaired[np.flatnonzero(still_invalid)] = repaired[donors]
    return repaired, True


def _reflect_into_bounds(
    x: NDArray[np.floating],
    lower: NDArray[np.floating],
    upper: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Repeated triangular-wave reflection into [lower, upper], per column.

    Handles overshoots larger than the interval width (a single bounce or
    clipping would not) and one-sided intervals where only one bound is
    finite; doubly-infinite columns pass through unchanged. The reflected
    diagonal Gaussian kernel is symmetric and cannot create clones.
    """
    out = x.copy()
    both = np.isfinite(lower) & np.isfinite(upper)
    if both.any():
        lo, hi = lower[both], upper[both]
        width = hi - lo
        y = np.mod(out[:, both] - lo, 2.0 * width)
        out[:, both] = lo + np.where(y > width, 2.0 * width - y, y)
    lower_only = np.isfinite(lower) & ~np.isfinite(upper)
    if lower_only.any():
        lo = lower[lower_only]
        out[:, lower_only] = lo + np.abs(out[:, lower_only] - lo)
    upper_only = ~np.isfinite(lower) & np.isfinite(upper)
    if upper_only.any():
        hi = upper[upper_only]
        out[:, upper_only] = hi - np.abs(out[:, upper_only] - hi)
    return out


def _state_collapsed_ess(
    particles: NDArray[np.floating], weights: NDArray[np.floating]
) -> float:
    """ESS over unique particle values: clones pool into one weight group.

    Ordinary ESS counts weight diversity; a clone-majority cloud can hold
    ESS above the resample threshold with zero state diversity (the
    clone-freeze). Grouping by value exposes that collapse.
    """
    _, inverse = np.unique(particles, axis=0, return_inverse=True)
    group_weights = np.bincount(inverse, weights=weights)
    return float(1.0 / np.sum(group_weights**2))


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
        along its correlation structure"; with ``jitter="annealed"`` it is
        the floor the per-parameter schedule decays toward.
    rng : Numpy random generator.
    forward_model_batch : Optional Callable(particles) → (n_particles, n_clocks).
        If provided, used instead of looping forward_model per particle.
    constraint_fn : Optional Callable(particles) -> particles applied after
        resampling (and at init) to enforce structural constraints, e.g. mass
        ordering for multi-mass models.
    resampling : Resampling scheme after ESS drops below threshold: "systematic"
        (default, minimal-variance), "stratified", or "residual".
    jitter : Jitter mode after resampling. ``"fixed"`` uses isotropic Gaussian
        noise; ``"covariance"`` draws from the weighted empirical covariance
        so correlated parameters jitter along their joint structure;
        ``"annealed"`` uses an axis-aligned diagonal Gaussian whose
        per-parameter scale decays from the initial cloud's std down to the
        ``jitter_std`` floor with time constant ``jitter_tau`` observations.
    jitter_tau : Time constant (in observations) for the ``"annealed"``
        jitter schedule's exponential decay. Unused by other jitter modes.
    log_prior : Optional Callable(particles) → (n_particles,) log-prior values.
        Particles with ``-inf`` log-prior get zero weight.  Use this instead
        of constraint clamping for boundary enforcement.
    support_bounds : Optional (lower, upper) per-parameter arrays enabling
        bounds reflection repair for the diagonal jitter modes (``"fixed"``
        and ``"annealed"``); ``"covariance"`` always uses reject-and-stay
        since coordinate-wise reflection is not symmetric for correlated
        kernels. Masses use ``(np.nextafter(0.0, 1.0), np.inf)`` via the API
        layer: reflection enforces "mass > 0, no upper bound", not
        ``mass_range``.
    """

    def __init__(
        self,
        n_particles: int,
        prior_sampler: Callable[[np.random.Generator, int], NDArray[np.floating]],
        forward_model: Callable[[NDArray[np.floating]], NDArray[np.floating]],
        noise_std: float,
        resample_threshold: float = 0.5,
        jitter_std: float = 0.02,
        rng: np.random.Generator | None = None,
        forward_model_batch: Callable[[NDArray[np.floating]], NDArray[np.floating]]
        | None = None,
        constraint_fn: Callable[[NDArray[np.floating]], NDArray[np.floating]]
        | None = None,
        resampling: str = "systematic",
        jitter: str = "annealed",
        jitter_tau: float = 15.0,
        log_prior: Callable[[NDArray[np.floating]], NDArray[np.floating]] | None = None,
        support_bounds: tuple[NDArray[np.floating], NDArray[np.floating]] | None = None,
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
        if not math.isfinite(jitter_std) or jitter_std < 0:
            msg = f"jitter_std must be finite and >= 0, got {jitter_std}"
            raise ValueError(msg)
        if not math.isfinite(jitter_tau) or jitter_tau <= 0:
            msg = f"jitter_tau must be finite and > 0, got {jitter_tau}"
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
        self.jitter_tau = jitter_tau
        self.log_prior = log_prior
        self.support_bounds = support_bounds
        self.rng = rng or np.random.default_rng()
        self.log_evidence: float = 0.0
        self._repair_reverted = False

        particles = prior_sampler(self.rng, n_particles)
        if constraint_fn is not None:
            # Parents must satisfy the constraint for reject-and-stay
            # reversion to be sound on the very first resample.
            particles = constraint_fn(particles)
        if support_bounds is not None:
            lower, upper = support_bounds
            if len(lower) != particles.shape[1] or len(upper) != particles.shape[1]:
                msg = (
                    f"support_bounds arrays must have length {particles.shape[1]}"
                    f" (n_params), got lower={len(lower)}, upper={len(upper)}"
                )
                raise ValueError(msg)
            if not np.all(lower < upper):
                raise ValueError("support_bounds requires lower < upper elementwise")
        # Prior scale for the annealed schedule: the initial cloud is a
        # prior sample. Clamped so the schedule never anneals upward.
        self._jitter_init = np.maximum(particles.std(axis=0), jitter_std)
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
        if not np.isfinite(max_lw):
            raise RuntimeError(
                "All particles have zero weight (every log-weight is -inf); "
                "the prior or forward model is inconsistent with the "
                "observations"
            )
        log_weights -= max_lw
        weights = np.exp(log_weights)
        self.log_evidence += max_lw + np.log(weights.sum())
        weights /= weights.sum()

        # Resample if effective sample size is too low. A repair that
        # reverted proposals to a dominant parent can create a clone-
        # majority cloud whose weight-ESS stays high with zero state
        # diversity (the clone-freeze degeneracy) -- the state-collapsed
        # ESS backstop catches that case even when the plain ESS doesn't.
        ess = 1.0 / np.sum(weights**2)
        needs_resample = ess < self.resample_threshold * self.n_particles
        if not needs_resample and self._repair_reverted:
            ess_state = _state_collapsed_ess(particles, weights)
            needs_resample = ess_state < self.resample_threshold * self.n_particles
        if needs_resample:
            particles, weights = self._resample(particles, weights)

        self._state = ParticleState(
            particles=particles,
            weights=weights,
            observations_seen=self._state.observations_seen + 1,
        )
        self._history.append(self._state)
        return self._state

    def _annealed_std(self, t: float) -> NDArray[np.floating]:
        """Scheduled per-parameter jitter std at 0-based observation index t."""
        decay = np.exp(-t / self.jitter_tau)
        return self.jitter_std + (self._jitter_init - self.jitter_std) * decay

    def _resample(
        self,
        particles: NDArray[np.floating],
        weights: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Resample particles with jitter."""
        n = self.n_particles

        if self.resampling == "stratified":
            indices = _stratified_indices(weights, n, self.rng)
        elif self.resampling == "residual":
            indices = _residual_indices(weights, n, self.rng)
        else:
            indices = _systematic_indices(weights, n, self.rng)

        parents = particles[indices]
        new_particles = parents.copy()

        # The weighted covariance needs at least ~2 effective samples;
        # below that np.cov's normalization (1 - sum(w^2)) underflows to
        # inf/NaN. Fall back to isotropic jitter to restore diversity. A
        # clone-collapsed cloud has a degenerate covariance even when
        # weight-ESS looks healthy, so also require state-collapsed ESS.
        ess = 1.0 / np.sum(weights**2)
        if (
            self.jitter == "covariance"
            and ess >= 2.0
            and _state_collapsed_ess(particles, weights) >= 2.0
        ):
            cov = np.cov(particles.T, aweights=weights)
            n_params = particles.shape[1]
            if n_params == 1:
                cov = cov.reshape(1, 1)
            cov += 1e-10 * np.eye(n_params)
            L = cholesky(cov, lower=True)
            z = self.rng.normal(0, 1, size=new_particles.shape)
            new_particles += self.jitter_std * (z @ L.T)
        elif self.jitter == "annealed":
            # observations_seen has not been incremented for the update in
            # progress, so this is the 0-based index of the current one.
            sigma = self._annealed_std(self._state.observations_seen)
            z = self.rng.normal(0.0, 1.0, size=new_particles.shape)
            new_particles += z * sigma
        else:
            new_particles += self.rng.normal(
                0, self.jitter_std, size=new_particles.shape
            )

        if self.support_bounds is not None and self.jitter in ("fixed", "annealed"):
            lower, upper = self.support_bounds
            new_particles = _reflect_into_bounds(new_particles, lower, upper)
            if self.constraint_fn is not None:
                new_particles = self.constraint_fn(new_particles)
            self._repair_reverted = False
            if self.log_prior is not None:
                if np.isneginf(self.log_prior(new_particles)).any():
                    raise RuntimeError(
                        "support_bounds contradict log_prior: reflected "
                        "particles are still outside the prior support"
                    )
        else:
            if self.constraint_fn is not None:
                new_particles = self.constraint_fn(new_particles)
            if self.log_prior is not None:
                new_particles, self._repair_reverted = _repair_support(
                    new_particles, parents, self.log_prior, self.rng
                )
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

    Runs parallel particle filters for `k_values` when provided; otherwise
    for K=1..k_max, and compares accumulated log-evidence to infer the most
    likely number of masses.
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
        jitter: str = "annealed",
        jitter_tau: float = 15.0,
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
                jitter_tau=jitter_tau,
                log_prior=self._make_log_prior(k, position_range, mass_range),
                support_bounds=self._make_support_bounds(k, position_range),
            )

    def _make_support_bounds(
        self, k: int, position_range: tuple[float, float]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Reflection bounds for K masses: positions in range, mass > 0."""
        n_params = k * self.n_dims + k
        lower = np.empty(n_params)
        upper = np.empty(n_params)
        lower[: k * self.n_dims] = position_range[0]
        upper[: k * self.n_dims] = position_range[1]
        lower[k * self.n_dims :] = np.nextafter(0.0, 1.0)
        upper[k * self.n_dims :] = np.inf
        return lower, upper

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
        """Log-prior: -inf for non-positive masses or positions outside
        position_range; mass_range only shapes initial sampling, not the
        prior support."""
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
