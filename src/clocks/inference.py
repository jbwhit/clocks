"""Adaptive tempered resample-move SMC for static parameter inference."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from clocks.types import Observation, ParticleState, UpdateDiagnostics

_RESAMPLING_METHODS = {"systematic", "stratified", "residual"}
_WEIGHT_SUM_ATOL = 1e-12
_WEIGHT_SUM_EPS_MULTIPLIER = 4.0
_UNIT_INTERVAL_MAX = np.nextafter(1.0, 0.0)


def _finite_float(name: str, value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validated_resampling_inputs(
    weights: NDArray[np.floating], n_draws: int
) -> tuple[NDArray[np.float64], int]:
    """Validate resampling inputs without silently renormalizing bad weights."""
    source_weights = np.asarray(weights)
    if source_weights.ndim != 1:
        raise ValueError("weights must be a 1-D array")
    if source_weights.size == 0:
        raise ValueError("weights must be nonempty")
    weights_array = np.asarray(source_weights, dtype=np.float64)
    if not np.all(np.isfinite(weights_array)):
        raise ValueError("weights must be finite")
    if np.any(weights_array < 0):
        raise ValueError("weights must be nonnegative")
    total = float(weights_array.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("weights total must be finite and strictly positive")
    source_epsilon = (
        np.finfo(source_weights.dtype).eps
        if np.issubdtype(source_weights.dtype, np.floating)
        else np.finfo(np.float64).eps
    )
    sum_atol = max(_WEIGHT_SUM_ATOL, _WEIGHT_SUM_EPS_MULTIPLIER * source_epsilon)
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
    cumulative = np.cumsum(weights_array)
    cumulative[-1] = 1.0
    positions = (rng.uniform() + np.arange(n_draws)) / n_draws
    positions = np.minimum(positions, _UNIT_INTERVAL_MAX)
    indices = np.searchsorted(cumulative, positions, side="right")
    return np.clip(indices, 0, len(weights_array) - 1).astype(np.intp)


def _stratified_indices(
    weights: NDArray[np.floating], n_draws: int, rng: np.random.Generator
) -> NDArray[np.intp]:
    """Draw source indices using stratified resampling."""
    weights_array, n_draws = _validated_resampling_inputs(weights, n_draws)
    cumulative = np.cumsum(weights_array)
    cumulative[-1] = 1.0
    positions = (rng.uniform(size=n_draws) + np.arange(n_draws)) / n_draws
    positions = np.minimum(positions, _UNIT_INTERVAL_MAX)
    indices = np.searchsorted(cumulative, positions, side="right")
    return np.clip(indices, 0, len(weights_array) - 1).astype(np.intp)


def _residual_indices(
    weights: NDArray[np.floating], n_draws: int, rng: np.random.Generator
) -> NDArray[np.intp]:
    """Draw deterministic copies plus systematic fractional-residual draws."""
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


@dataclass(frozen=True)
class GaussianObservationStats:
    """Immutable sufficient statistics for iid vector Gaussian observations."""

    n: float
    sum_y: NDArray[np.float64]
    sum_y2: float

    def __post_init__(self) -> None:
        count = float(self.n)
        sums = np.array(self.sum_y, dtype=np.float64, copy=True)
        sum_squares = float(self.sum_y2)
        if not math.isfinite(count) or count < 0:
            raise ValueError("n must be finite and nonnegative")
        if sums.ndim != 1 or sums.size == 0 or not np.all(np.isfinite(sums)):
            raise ValueError("sum_y must be a nonempty finite 1-D array")
        if not math.isfinite(sum_squares) or sum_squares < 0:
            raise ValueError("sum_y2 must be finite and nonnegative")
        sums.setflags(write=False)
        object.__setattr__(self, "n", count)
        object.__setattr__(self, "sum_y", sums)
        object.__setattr__(self, "sum_y2", sum_squares)

    @classmethod
    def empty(cls, n_channels: int) -> GaussianObservationStats:
        if isinstance(n_channels, bool) or not isinstance(n_channels, Integral):
            raise ValueError("n_channels must be a positive integer")
        if n_channels <= 0:
            raise ValueError("n_channels must be a positive integer")
        return cls(0.0, np.zeros(int(n_channels)), 0.0)

    def add(self, observation: NDArray[np.floating]) -> GaussianObservationStats:
        return self.with_fraction(observation, 1.0)

    def with_fraction(
        self, observation: NDArray[np.floating], beta: float
    ) -> GaussianObservationStats:
        rates = np.asarray(observation, dtype=np.float64)
        fraction = float(beta)
        if rates.shape != self.sum_y.shape:
            raise ValueError(
                f"observation shape must be {self.sum_y.shape}, got {rates.shape}"
            )
        if not np.all(np.isfinite(rates)):
            raise ValueError("observation must be finite")
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("beta must be finite and in [0, 1]")
        return GaussianObservationStats(
            self.n + fraction,
            self.sum_y + fraction * rates,
            self.sum_y2 + fraction * float(rates @ rates),
        )

    def log_likelihood(
        self, predicted: NDArray[np.floating], noise_std: float
    ) -> NDArray[np.float64]:
        predictions = np.asarray(predicted, dtype=np.float64)
        if predictions.ndim != 2 or predictions.shape[1] != len(self.sum_y):
            raise ValueError(
                "predicted must have shape "
                f"(N, {len(self.sum_y)}), got {predictions.shape}"
            )
        if not np.all(np.isfinite(predictions)):
            raise ValueError("predictions must be finite")
        sigma = float(noise_std)
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("noise_std must be finite and > 0")
        quadratic = (
            self.sum_y2
            - 2.0 * predictions @ self.sum_y
            + self.n * np.sum(predictions**2, axis=1)
        )
        normalizer = (
            self.n * len(self.sum_y) * math.log(sigma * math.sqrt(2.0 * math.pi))
        )
        return -normalizer - quadratic / (2.0 * sigma**2)


def _effective_sample_size(weights: NDArray[np.floating]) -> float:
    return float(1.0 / np.sum(np.asarray(weights, dtype=np.float64) ** 2))


def _normalize_log_weights(
    log_weights: NDArray[np.floating],
) -> tuple[NDArray[np.float64], float]:
    values = np.asarray(log_weights, dtype=np.float64)
    log_normalizer = float(logsumexp(values))
    if not math.isfinite(log_normalizer):
        raise RuntimeError(
            "All particles have zero weight; the prior or forward model is "
            "inconsistent with the observation"
        )
    return np.exp(values - log_normalizer), log_normalizer


def _next_beta(
    weights: NDArray[np.floating],
    observation_log_likelihood: NDArray[np.floating],
    beta: float,
    *,
    target_ess: float,
) -> float:
    """Choose the largest next exponent whose importance weights meet ESS."""
    weights_array = np.asarray(weights, dtype=np.float64)
    likelihood = np.asarray(observation_log_likelihood, dtype=np.float64)
    if weights_array.shape != likelihood.shape or weights_array.ndim != 1:
        raise ValueError("weights and log likelihood must be matching 1-D arrays")
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in [0, 1)")
    with np.errstate(divide="ignore"):
        base = np.log(weights_array)

    def ess_at(candidate: float) -> float:
        normalized, _ = _normalize_log_weights(base + (candidate - beta) * likelihood)
        return _effective_sample_size(normalized)

    if ess_at(1.0) >= target_ess:
        return 1.0
    low, high = beta, 1.0
    for _ in range(60):
        middle = (low + high) / 2.0
        if ess_at(middle) >= target_ess:
            low = middle
        else:
            high = middle
    if low <= beta:
        raise RuntimeError("adaptive tempering could not make positive progress")
    return low


class Estimate(TypedDict):
    mean: NDArray[np.floating]
    std: NDArray[np.floating]
    ess: float


class ConvergenceInfo(TypedDict):
    converged: bool
    per_param_std: NDArray[np.floating]
    per_param_converged: NDArray[np.bool_]
    ess: float
    estimates_stable: bool


class ParticleFilter:
    """Adaptive tempered resample-move SMC for a static parameter vector."""

    def __init__(
        self,
        n_particles: int,
        prior_sampler: Callable[[np.random.Generator, int], NDArray[np.floating]],
        forward_model: Callable[[NDArray[np.floating]], NDArray[np.floating]],
        noise_std: float,
        *,
        log_prior_density: Callable[[NDArray[np.floating]], NDArray[np.floating]],
        forward_model_batch: Callable[[NDArray[np.floating]], NDArray[np.floating]]
        | None = None,
        resampling: str = "systematic",
        ess_target: float = 0.8,
        rejuvenation_steps: int = 2,
        proposal_scale: float = 2.38,
        rng: np.random.Generator | None = None,
    ) -> None:
        if isinstance(n_particles, (bool, np.bool_)) or not isinstance(
            n_particles, (int, np.integer)
        ):
            raise ValueError("n_particles must be a positive integer")
        if n_particles <= 0:
            raise ValueError("n_particles must be a positive integer")
        if not callable(prior_sampler):
            raise ValueError("prior_sampler must be callable")
        if not callable(forward_model):
            raise ValueError("forward_model must be callable")
        if not callable(log_prior_density):
            raise ValueError("log_prior_density must be callable")
        if forward_model_batch is not None and not callable(forward_model_batch):
            raise ValueError("forward_model_batch must be callable or None")
        noise_std = _finite_float("noise_std", noise_std)
        if noise_std <= 0:
            raise ValueError("noise_std must be finite and > 0")
        if not isinstance(resampling, str) or resampling not in _RESAMPLING_METHODS:
            raise ValueError(
                f"resampling must be one of {sorted(_RESAMPLING_METHODS)}, "
                f"got {resampling!r}"
            )
        ess_target = _finite_float("ess_target", ess_target)
        if not 0.0 < ess_target < 1.0:
            raise ValueError("ess_target must be finite and in (0, 1)")
        if isinstance(rejuvenation_steps, (bool, np.bool_)) or not isinstance(
            rejuvenation_steps, (int, np.integer)
        ):
            raise ValueError("rejuvenation_steps must be a positive integer")
        if rejuvenation_steps <= 0:
            raise ValueError("rejuvenation_steps must be a positive integer")
        proposal_scale = _finite_float("proposal_scale", proposal_scale)
        if proposal_scale <= 0:
            raise ValueError("proposal_scale must be finite and > 0")

        self.n_particles = int(n_particles)
        self.forward_model = forward_model
        self.forward_model_batch = forward_model_batch
        self.noise_std = float(noise_std)
        self.log_prior_density = log_prior_density
        self.resampling = resampling
        self.ess_target = float(ess_target)
        self.rejuvenation_steps = int(rejuvenation_steps)
        self.proposal_scale = float(proposal_scale)
        self.rng = rng if rng is not None else np.random.default_rng()

        particles = np.asarray(
            prior_sampler(self.rng, self.n_particles), dtype=np.float64
        )
        if particles.ndim != 2 or particles.shape[0] != self.n_particles:
            raise ValueError(
                "prior_sampler must return shape "
                f"({self.n_particles}, D), got {particles.shape}"
            )
        if particles.shape[1] == 0 or not np.all(np.isfinite(particles)):
            raise ValueError("prior_sampler must return finite nonempty parameters")
        initial_log_prior = self._log_prior(particles)
        if not np.all(np.isfinite(initial_log_prior)):
            raise ValueError(
                "prior_sampler must draw every particle with finite log prior density"
            )
        self._initial_scale = particles.std(axis=0)
        weights = np.full(self.n_particles, 1.0 / self.n_particles)
        self._state = ParticleState(particles, weights, 0)
        self._history: list[ParticleState] = [self._state]
        self._diagnostics_history: list[UpdateDiagnostics] = []
        self._log_evidence_history: list[float] = []
        self._completed_stats: GaussianObservationStats | None = None
        self.log_evidence = 0.0
        self.last_diagnostics = UpdateDiagnostics()
        self._last_log_evidence_increments: tuple[float, ...] = ()

    @property
    def state(self) -> ParticleState:
        return self._state

    @property
    def history(self) -> list[ParticleState]:
        return list(self._history)

    @property
    def diagnostics_history(self) -> list[UpdateDiagnostics]:
        return list(self._diagnostics_history)

    @property
    def log_evidence_history(self) -> list[float]:
        return list(self._log_evidence_history)

    @property
    def last_log_evidence_increments(self) -> tuple[float, ...]:
        """Incremental normalizers from the most recent public update."""
        return self._last_log_evidence_increments

    def _log_prior(self, particles: NDArray[np.floating]) -> NDArray[np.float64]:
        values = np.asarray(self.log_prior_density(particles), dtype=np.float64)
        if values.shape != (len(particles),):
            raise ValueError(
                f"log_prior_density must return shape ({len(particles)},), "
                f"got {values.shape}"
            )
        if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
            raise ValueError("log prior values must be finite or -inf")
        return values

    def _predictions(
        self,
        particles: NDArray[np.floating],
        valid: NDArray[np.bool_],
        n_channels: int,
    ) -> NDArray[np.float64]:
        valid_particles = particles[valid]
        expected = (len(valid_particles), n_channels)
        if self.forward_model_batch is not None:
            predicted = np.asarray(
                self.forward_model_batch(valid_particles), dtype=np.float64
            )
        else:
            rows = [
                np.asarray(self.forward_model(row), dtype=np.float64)
                for row in valid_particles
            ]
            predicted = np.asarray(rows, dtype=np.float64)
            if not rows:
                predicted = np.empty(expected)
        if predicted.shape != expected:
            raise ValueError(
                f"prediction shape must be {expected}, got {predicted.shape}"
            )
        if not np.all(np.isfinite(predicted)):
            raise ValueError("predictions must be finite")
        return predicted

    def _observation_log_likelihood(
        self, particles: NDArray[np.floating], observation: Observation
    ) -> NDArray[np.float64]:
        prior = self._log_prior(particles)
        valid = np.isfinite(prior)
        result = np.full(len(particles), -np.inf)
        if np.any(valid):
            predictions = self._predictions(particles, valid, observation.rates.size)
            residual = predictions - observation.rates
            result[valid] = -observation.rates.size * math.log(
                self.noise_std * math.sqrt(2.0 * math.pi)
            ) - np.sum(residual**2, axis=1) / (2.0 * self.noise_std**2)
        return result

    def _log_target(
        self,
        particles: NDArray[np.floating],
        *,
        beta: float,
        observation: Observation,
    ) -> NDArray[np.float64]:
        prior = self._log_prior(particles)
        valid = np.isfinite(prior)
        result = np.full(len(particles), -np.inf)
        if not np.any(valid):
            return result
        predictions = self._predictions(particles, valid, observation.rates.size)
        completed = self._completed_stats
        if completed is None:
            completed = GaussianObservationStats.empty(observation.rates.size)
        elif len(completed.sum_y) != observation.rates.size:
            raise ValueError(
                "observation channel count does not match previous observations"
            )
        target_stats = completed.with_fraction(observation.rates, beta)
        result[valid] = prior[valid] + target_stats.log_likelihood(
            predictions, self.noise_std
        )
        return result

    def _proposal_cholesky(
        self,
        particles: NDArray[np.floating],
        weights: NDArray[np.floating],
    ) -> NDArray[np.float64]:
        mean = np.sum(weights[:, np.newaxis] * particles, axis=0)
        centered = particles - mean
        covariance = centered.T @ (weights[:, np.newaxis] * centered)
        ridge = 1e-6 * np.diag(np.maximum(self._initial_scale, 1e-12) ** 2)
        covariance = (self.proposal_scale**2 / particles.shape[1]) * (
            covariance + ridge
        )
        return np.linalg.cholesky(covariance)

    def _metropolis_move(
        self,
        particles: NDArray[np.floating],
        *,
        beta: float,
        observation: Observation,
        proposal_chol: NDArray[np.floating],
    ) -> tuple[NDArray[np.float64], UpdateDiagnostics]:
        current = np.array(particles, dtype=np.float64, copy=True)
        current_target = self._log_target(current, beta=beta, observation=observation)
        acceptances = 0
        for _ in range(self.rejuvenation_steps):
            noise = np.asarray(self.rng.normal(size=current.shape), dtype=np.float64)
            proposal = current + noise @ np.asarray(proposal_chol).T
            proposal_target = self._log_target(
                proposal, beta=beta, observation=observation
            )
            log_alpha = proposal_target - current_target
            uniforms = np.asarray(self.rng.random(len(current)), dtype=np.float64)
            with np.errstate(divide="ignore"):
                accept = np.log(uniforms) < np.minimum(0.0, log_alpha)
            current[accept] = proposal[accept]
            current_target[accept] = proposal_target[accept]
            acceptances += int(np.sum(accept))
        diagnostics = UpdateDiagnostics(
            tempering_stages=0,
            mh_proposals=self.rejuvenation_steps * len(current),
            mh_acceptances=acceptances,
        )
        return current, diagnostics

    def _resample_indices(self, weights: NDArray[np.floating]) -> NDArray[np.intp]:
        helper = {
            "systematic": _systematic_indices,
            "stratified": _stratified_indices,
            "residual": _residual_indices,
        }[self.resampling]
        return helper(weights, self.n_particles, self.rng)

    def update(self, observation: Observation) -> ParticleState:
        if not isinstance(observation, Observation):
            raise TypeError("observation must be an Observation")
        if self._completed_stats is not None and observation.rates.shape != (
            len(self._completed_stats.sum_y),
        ):
            raise ValueError(
                "observation channel count does not match previous observations"
            )
        particles = self._state.particles.copy()
        weights = self._state.weights.copy()
        beta = 0.0
        stages = proposals = acceptances = 0
        evidence_increments: list[float] = []
        target_ess = self.ess_target * self.n_particles

        while beta < 1.0:
            observation_ll = self._observation_log_likelihood(particles, observation)
            next_beta = _next_beta(
                weights,
                observation_ll,
                beta,
                target_ess=target_ess,
            )
            delta = next_beta - beta
            with np.errstate(divide="ignore", invalid="ignore"):
                candidate_log_weights = np.log(weights) + delta * observation_ll
            weights, log_increment = _normalize_log_weights(candidate_log_weights)
            self.log_evidence += log_increment
            evidence_increments.append(log_increment)
            beta = next_beta
            stages += 1

            ess = _effective_sample_size(weights)
            if beta < 1.0 or ess <= target_ess + 1e-9:
                proposal_chol = self._proposal_cholesky(particles, weights)
                indices = self._resample_indices(weights)
                particles = particles[indices].copy()
                weights = np.full(self.n_particles, 1.0 / self.n_particles)
                particles, move = self._metropolis_move(
                    particles,
                    beta=beta,
                    observation=observation,
                    proposal_chol=proposal_chol,
                )
                proposals += move.mh_proposals
                acceptances += move.mh_acceptances

        if self._completed_stats is None:
            self._completed_stats = GaussianObservationStats.empty(
                observation.rates.size
            )
        self._completed_stats = self._completed_stats.add(observation.rates)
        diagnostics = UpdateDiagnostics(stages, proposals, acceptances)
        self.last_diagnostics = diagnostics
        self._last_log_evidence_increments = tuple(evidence_increments)
        self._diagnostics_history.append(diagnostics)
        self._log_evidence_history.append(self.log_evidence)
        self._state = ParticleState(
            particles, weights, self._state.observations_seen + 1
        )
        self._history.append(self._state)
        return self._state

    def estimate(self) -> Estimate:
        particles = self._state.particles
        weights = self._state.weights
        mean = np.average(particles, weights=weights, axis=0)
        variance = np.average((particles - mean) ** 2, weights=weights, axis=0)
        return {
            "mean": mean,
            "std": np.sqrt(variance),
            "ess": _effective_sample_size(weights),
        }

    def converged(
        self,
        std_threshold: float = 0.01,
        window: int = 10,
        stability_threshold: float = 0.001,
    ) -> ConvergenceInfo:
        estimate = self.estimate()
        per_param_std = estimate["std"]
        per_param_converged = per_param_std < std_threshold
        if len(self._history) < window:
            stable = False
        else:
            means = np.array(
                [
                    np.average(state.particles, weights=state.weights, axis=0)
                    for state in self._history[-window:]
                ]
            )
            stable = bool(np.max(np.abs(np.diff(means, axis=0))) < stability_threshold)
        return {
            "converged": bool(np.all(per_param_converged)) and stable,
            "per_param_std": per_param_std,
            "per_param_converged": per_param_converged,
            "ess": estimate["ess"],
            "estimates_stable": stable,
        }


class ModelComparisonResult(TypedDict):
    log_evidence: dict[int, float]
    posterior: dict[int, float]


class ModelComparison:
    """Bayesian comparison of already-constructed fixed-model filters."""

    def __init__(
        self,
        filters: dict[int, ParticleFilter],
        model_prior: Mapping[int, float] | None = None,
    ) -> None:
        if not filters:
            raise ValueError("filters must be nonempty")
        if any(
            isinstance(k, bool) or not isinstance(k, Integral) or k <= 0
            for k in filters
        ):
            raise ValueError("filter keys must be positive integer model identifiers")
        self.filters = dict(sorted(filters.items()))
        self.k_values = tuple(self.filters)
        if model_prior is None:
            probability = 1.0 / len(filters)
            self.model_prior = {k: probability for k in self.filters}
        else:
            if set(model_prior) != set(filters):
                raise ValueError("model_prior keys must exactly match filters")
            probabilities = {k: float(model_prior[k]) for k in self.filters}
            if any(not math.isfinite(p) or p <= 0 for p in probabilities.values()):
                raise ValueError(
                    "model_prior probabilities must be finite and positive"
                )
            if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-12):
                raise ValueError("model_prior probabilities must sum to one")
            self.model_prior = probabilities

    def update(self, observation: Observation) -> None:
        for particle_filter in self.filters.values():
            particle_filter.update(observation)

    def evidence(self) -> ModelComparisonResult:
        log_evidence = {k: pf.log_evidence for k, pf in self.filters.items()}
        keys = list(self.filters)
        log_joint = np.array(
            [log_evidence[k] + math.log(self.model_prior[k]) for k in keys]
        )
        log_norm = float(logsumexp(log_joint))
        posterior = {
            k: float(math.exp(value - log_norm))
            for k, value in zip(keys, log_joint, strict=True)
        }
        return {"log_evidence": log_evidence, "posterior": posterior}

    def estimate(self, k: int | None = None) -> Estimate:
        if k is None:
            posterior = self.evidence()["posterior"]
            k = max(posterior, key=posterior.__getitem__)
        if k not in self.filters:
            raise ValueError(f"No filter for K={k}, available: {list(self.filters)}")
        return self.filters[k].estimate()
