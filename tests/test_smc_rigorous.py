"""Correctness tests for adaptive tempered resample-move SMC."""

import math

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm

import clocks.inference as inference_module
from clocks.inference import (
    GaussianObservationStats,
    ModelComparison,
    ParticleFilter,
    _next_beta,
)
from clocks.types import Observation


def _normal_filter(seed: int, n_particles: int = 40_000) -> ParticleFilter:
    tau = 1.2

    def sample(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.normal(0.0, tau, size=(n, 1))

    def log_prior(values: np.ndarray) -> np.ndarray:
        return norm.logpdf(values[:, 0], loc=0.0, scale=tau)

    return ParticleFilter(
        n_particles,
        sample,
        lambda value: value.copy(),
        0.25,
        log_prior_density=log_prior,
        forward_model_batch=lambda values: values.copy(),
        ess_target=0.8,
        rejuvenation_steps=2,
        rng=np.random.default_rng(seed),
    )


def test_sufficient_statistics_match_direct_gaussian_sum() -> None:
    observations = np.array([[0.2, -0.1], [0.3, 0.0], [0.4, 0.2]])
    predicted = np.array([[0.25, 0.05], [0.5, -0.2]])
    stats = GaussianObservationStats.empty(2)
    for row in observations:
        stats = stats.add(row)

    actual = stats.log_likelihood(predicted, noise_std=0.3)
    expected = np.array(
        [
            sum(np.sum(norm.logpdf(row, loc=mu, scale=0.3)) for row in observations)
            for mu in predicted
        ]
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("beta", [0.0, 0.37, 1.0])
def test_fractional_current_observation_matches_direct_likelihood(beta: float) -> None:
    completed = GaussianObservationStats.empty(2).add(np.array([0.2, -0.1]))
    current = np.array([0.4, 0.3])
    predicted = np.array([[0.25, 0.05], [0.5, -0.2]])

    actual = completed.with_fraction(current, beta).log_likelihood(predicted, 0.3)
    expected = completed.log_likelihood(predicted, 0.3) + beta * np.array(
        [np.sum(norm.logpdf(current, loc=mu, scale=0.3)) for mu in predicted]
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_invalid_prior_rows_never_reach_strict_batch_forward() -> None:
    initial = np.array([[0.2], [0.8]])

    def strict_batch(values: np.ndarray) -> np.ndarray:
        assert np.all((0.0 <= values[:, 0]) & (values[:, 0] <= 1.0))
        return values.copy()

    pf = ParticleFilter(
        2,
        lambda rng, n: initial.copy(),
        lambda value: value.copy(),
        0.2,
        log_prior_density=lambda values: np.where(
            (0.0 <= values[:, 0]) & (values[:, 0] <= 1.0), 0.0, -np.inf
        ),
        forward_model_batch=strict_batch,
        rng=np.random.default_rng(1),
    )

    target = pf._log_target(
        np.array([[0.2], [2.0]]), beta=1.0, observation=Observation([0.3], 0.0)
    )

    assert np.isfinite(target[0])
    assert target[1] == -np.inf


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_particles": True}, "n_particles"),
        ({"noise_std": 0.0}, "noise_std"),
        ({"noise_std": "invalid"}, "noise_std"),
        ({"ess_target": 1.0}, "ess_target"),
        ({"ess_target": "invalid"}, "ess_target"),
        ({"rejuvenation_steps": False}, "rejuvenation_steps"),
        ({"proposal_scale": np.inf}, "proposal_scale"),
        ({"proposal_scale": "invalid"}, "proposal_scale"),
        ({"resampling": "unknown"}, "resampling"),
        ({"resampling": []}, "resampling"),
    ],
)
def test_constructor_rejects_invalid_controls(kwargs: dict, message: str) -> None:
    arguments = {
        "n_particles": 4,
        "prior_sampler": lambda rng, n: np.zeros((n, 1)),
        "forward_model": lambda value: value,
        "noise_std": 1.0,
        "log_prior_density": lambda values: np.zeros(len(values)),
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        ParticleFilter(**arguments)


def test_prior_sampler_must_return_exact_finite_supported_shape() -> None:
    with pytest.raises(ValueError, match="prior_sampler.*shape"):
        ParticleFilter(
            4,
            lambda rng, n: np.zeros((n - 1, 1)),
            lambda value: value,
            1.0,
            log_prior_density=lambda values: np.zeros(len(values)),
        )
    with pytest.raises(ValueError, match="finite log prior"):
        ParticleFilter(
            4,
            lambda rng, n: np.arange(n, dtype=float).reshape(n, 1),
            lambda value: value,
            1.0,
            log_prior_density=lambda values: np.where(values[:, 0] < 3, 0.0, -np.inf),
        )


def test_next_beta_hits_target_ess_and_evidence_identity() -> None:
    weights = np.full(4, 0.25)
    log_likelihood = np.array([0.0, -2.0, -5.0, -9.0])
    next_beta = _next_beta(weights, log_likelihood, 0.0, target_ess=3.2)
    log_increment = logsumexp(np.log(weights) + next_beta * log_likelihood)
    normalized = np.exp(np.log(weights) + next_beta * log_likelihood - log_increment)

    assert 0.0 < next_beta < 1.0
    assert 1.0 / np.sum(normalized**2) == pytest.approx(3.2, abs=1e-10)
    assert log_increment == pytest.approx(
        logsumexp(np.log(weights) + next_beta * log_likelihood), abs=1e-12
    )


def test_one_stage_filter_evidence_equals_weighted_likelihood_average() -> None:
    initial = np.array([[-1.0], [-0.25], [0.5], [1.5]])
    observation = Observation([0.3], 0.0)
    sigma = 10.0
    pf = ParticleFilter(
        4,
        lambda rng, n: initial.copy(),
        lambda value: value,
        sigma,
        log_prior_density=lambda values: np.zeros(len(values)),
        forward_model_batch=lambda values: values,
        ess_target=0.1,
        rng=np.random.default_rng(2),
    )
    log_likelihood = norm.logpdf(observation.rates[0], initial[:, 0], sigma)

    pf.update(observation)

    assert pf.log_evidence == pytest.approx(
        logsumexp(np.log(np.full(4, 0.25)) + log_likelihood), abs=1e-12
    )
    assert pf.last_diagnostics.tempering_stages == 1
    assert pf.last_diagnostics.mh_proposals == 0


def test_update_never_leaves_ess_below_tempering_target() -> None:
    initial = np.linspace(-2.0, 2.0, 200).reshape(-1, 1)
    pf = ParticleFilter(
        len(initial),
        lambda rng, n: initial.copy(),
        lambda value: value,
        0.05,
        log_prior_density=lambda values: np.where(
            np.abs(values[:, 0]) <= 3.0, 0.0, -np.inf
        ),
        forward_model_batch=lambda values: values,
        ess_target=0.8,
        rng=np.random.default_rng(7),
    )

    pf.update(Observation([0.1], 0.0))

    assert pf.last_diagnostics.tempering_stages > 1
    assert pf.estimate()["ess"] >= 0.8 * pf.n_particles - 1e-9


class _ZeroMoveRng:
    """Deterministic RNG: fixed systematic offset and zero MH displacement."""

    def uniform(self, size: int | None = None) -> float | np.ndarray:
        if size is None:
            return 0.25
        return np.full(size, 0.25)

    def normal(self, size: tuple[int, int]) -> np.ndarray:
        return np.zeros(size)

    def random(self, size: int) -> np.ndarray:
        return np.full(size, 0.5)


def _independent_tempered_evidence_trace(
    particles: np.ndarray,
    *,
    observation: float,
    noise_std: float,
    ess_target: float,
    beta_schedule: tuple[float, ...],
) -> list[float]:
    """Independent reference implementation for the deterministic test case."""
    n_particles = len(particles)
    weights = np.full(n_particles, 1.0 / n_particles)
    beta = 0.0
    increments: list[float] = []
    for next_beta in beta_schedule:
        log_likelihood = norm.logpdf(observation, loc=particles[:, 0], scale=noise_std)
        log_weight = np.log(weights) + (next_beta - beta) * log_likelihood
        increment = float(logsumexp(log_weight))
        increments.append(increment)
        weights = np.exp(log_weight - increment)
        beta = next_beta

        ess = 1.0 / np.sum(weights**2)
        if beta < 1.0 or ess <= ess_target * n_particles + 1e-9:
            cumulative = np.cumsum(weights)
            positions = (0.25 + np.arange(n_particles)) / n_particles
            indices = np.searchsorted(cumulative, positions, side="right")
            particles = particles[indices]
            weights = np.full(n_particles, 1.0 / n_particles)
    return increments


def test_multistage_update_records_every_pre_resample_evidence_increment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = np.linspace(-2.0, 2.0, 41).reshape(-1, 1)
    observation = Observation([0.17], 0.0)
    pf = ParticleFilter(
        len(initial),
        lambda rng, n: initial.copy(),
        lambda value: value,
        0.08,
        log_prior_density=lambda values: np.zeros(len(values)),
        forward_model_batch=lambda values: values,
        ess_target=0.8,
        rejuvenation_steps=1,
        rng=_ZeroMoveRng(),  # type: ignore[arg-type]
    )
    expected = _independent_tempered_evidence_trace(
        initial.copy(),
        observation=0.17,
        noise_std=0.08,
        ess_target=0.8,
        beta_schedule=(0.2, 1.0),
    )
    beta_schedule = iter((0.2, 1.0))
    monkeypatch.setattr(
        inference_module,
        "_next_beta",
        lambda *args, **kwargs: next(beta_schedule),
    )

    pf.update(observation)

    assert len(expected) == 2
    assert pf.last_log_evidence_increments == pytest.approx(expected, abs=1e-12)
    assert pf.log_evidence == pytest.approx(sum(expected), abs=1e-12)
    assert pf.last_diagnostics.mh_proposals == len(expected) * len(initial)
    np.testing.assert_array_equal(
        pf.state.weights, np.full(len(initial), 1.0 / len(initial))
    )
    with pytest.raises(AttributeError):
        pf.last_log_evidence_increments = ()


def test_stage_proposal_uses_one_frozen_symmetric_gaussian_kernel() -> None:
    pf = _normal_filter(27, n_particles=200)
    factor = pf._proposal_cholesky(pf.state.particles, pf.state.weights)
    assert not hasattr(pf, "last_proposal_cholesky_factors")
    displacement = np.array([0.37])
    forward_whitened = np.linalg.solve(factor, displacement)
    reverse_whitened = np.linalg.solve(factor, -displacement)
    log_q_forward = -0.5 * float(forward_whitened @ forward_whitened)
    log_q_reverse = -0.5 * float(reverse_whitened @ reverse_whitened)
    assert log_q_forward == pytest.approx(log_q_reverse, abs=0.0)


@pytest.mark.parametrize(
    "prediction",
    [np.array(1.0), np.ones((1, 1)), np.array([np.nan])],
)
def test_scalar_forward_prediction_contract(prediction: np.ndarray) -> None:
    pf = ParticleFilter(
        4,
        lambda rng, n: np.zeros((n, 1)),
        lambda value: prediction,
        1.0,
        log_prior_density=lambda values: np.zeros(len(values)),
        ess_target=0.1,
    )
    message = "finite" if np.any(~np.isfinite(prediction)) else "prediction shape"
    with pytest.raises(ValueError, match=message):
        pf.update(Observation([0.0], 0.0))


@pytest.mark.parametrize(
    "prediction",
    [
        lambda n: np.ones(n),
        lambda n: np.ones((n, 2)),
        lambda n: np.full((n, 1), np.nan),
    ],
)
def test_batch_forward_prediction_contract(prediction: object) -> None:
    pf = ParticleFilter(
        4,
        lambda rng, n: np.zeros((n, 1)),
        lambda value: value,
        1.0,
        log_prior_density=lambda values: np.zeros(len(values)),
        forward_model_batch=lambda values: prediction(len(values)),  # type: ignore[operator]
        ess_target=0.1,
    )
    produced = prediction(4)  # type: ignore[operator]
    message = "finite" if np.any(~np.isfinite(produced)) else "prediction shape"
    with pytest.raises(ValueError, match=message):
        pf.update(Observation([0.0], 0.0))


def test_later_update_rejects_changed_channel_count() -> None:
    pf = ParticleFilter(
        4,
        lambda rng, n: np.zeros((n, 1)),
        lambda value: np.ones(1),
        1.0,
        log_prior_density=lambda values: np.zeros(len(values)),
        ess_target=0.1,
    )
    pf.update(Observation([0.0], 0.0))

    with pytest.raises(ValueError, match="channel count"):
        pf.update(Observation([0.0, 0.0], 1.0))


@pytest.mark.parametrize("seed", [3, 11])
def test_conjugate_normal_posterior_and_evidence(seed: int) -> None:
    tau = 1.2
    sigma = 0.25
    observations = np.array([0.45, 0.36, 0.51, 0.40])
    pf = _normal_filter(seed)

    for time, value in enumerate(observations):
        pf.update(Observation([value], float(time)))

    posterior_var = 1.0 / (1.0 / tau**2 + len(observations) / sigma**2)
    posterior_mean = posterior_var * observations.sum() / sigma**2
    covariance = sigma**2 * np.eye(len(observations)) + tau**2 * np.ones(
        (len(observations), len(observations))
    )
    expected_log_evidence = multivariate_normal.logpdf(
        observations, mean=np.zeros(len(observations)), cov=covariance
    )
    estimate = pf.estimate()

    assert estimate["mean"][0] == pytest.approx(posterior_mean, abs=0.01)
    assert estimate["std"][0] == pytest.approx(math.sqrt(posterior_var), abs=0.01)
    assert pf.log_evidence == pytest.approx(expected_log_evidence, abs=0.04)
    assert len(pf.history) == len(observations) + 1
    assert pf.last_diagnostics.tempering_stages >= 1
    assert pf.last_diagnostics.mh_proposals > 0
    assert pf.last_diagnostics.mh_acceptances > 0


class _CrossingRng:
    def normal(self, size: tuple[int, int]) -> np.ndarray:
        return np.ones(size)

    def random(self, size: int) -> np.ndarray:
        return np.full(size, 0.5)


def test_mh_rejects_support_crossing_without_repair() -> None:
    pf = ParticleFilter(
        1,
        lambda rng, n: np.array([[0.99]]),
        lambda value: value,
        0.2,
        log_prior_density=lambda values: np.where(
            (values[:, 0] >= 0.0) & (values[:, 0] <= 1.0), 0.0, -np.inf
        ),
        proposal_scale=50.0,
        rejuvenation_steps=1,
        rng=np.random.default_rng(0),
    )
    pf.rng = _CrossingRng()  # type: ignore[assignment]
    current = np.array([[0.99]])

    moved, diagnostics = pf._metropolis_move(
        current,
        beta=1.0,
        observation=Observation([0.99], 0.0),
        proposal_chol=np.array([[1.0]]),
    )

    np.testing.assert_array_equal(moved, current)
    assert diagnostics.mh_proposals == 1
    assert diagnostics.mh_acceptances == 0


def test_mh_does_not_change_evidence() -> None:
    pf = _normal_filter(19, n_particles=100)
    before = pf.log_evidence
    particles = pf.state.particles.copy()
    chol = np.array([[0.1]])

    pf._metropolis_move(
        particles,
        beta=0.5,
        observation=Observation([0.4], 0.0),
        proposal_chol=chol,
    )

    assert pf.log_evidence == before


def test_model_comparison_combines_evidence_with_model_prior() -> None:
    def filter_with_evidence(value: float) -> ParticleFilter:
        particle_filter = ParticleFilter(
            2,
            lambda rng, n: np.zeros((n, 1)),
            lambda parameter: parameter,
            1.0,
            log_prior_density=lambda parameters: np.zeros(len(parameters)),
        )
        particle_filter.log_evidence = value
        return particle_filter

    comparison = ModelComparison(
        {1: filter_with_evidence(-2.0), 2: filter_with_evidence(-1.0)},
        model_prior={1: 0.25, 2: 0.75},
    )

    result = comparison.evidence()
    expected = np.exp(np.array([-2.0 + np.log(0.25), -1.0 + np.log(0.75)]))
    expected /= expected.sum()
    assert result["posterior"] == pytest.approx({1: expected[0], 2: expected[1]})
