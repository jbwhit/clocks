"""Correctness tests for adaptive tempered resample-move SMC."""

import copy
import math
import warnings

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


def test_sufficient_statistics_are_stable_near_unit_rates_at_low_noise() -> None:
    scale = 1e-9
    observations = np.array(
        [
            [1.0 + scale, 1.0 - 2.0 * scale],
            [1.0 - scale, 1.0 + scale],
            [1.0 + 2.0 * scale, 1.0 - scale],
        ]
    )
    predicted = np.array(
        [
            [1.0, 1.0],
            [1.0 + scale / 2.0, 1.0 - scale / 2.0],
        ]
    )
    noise_std = scale / 10.0
    stats = GaussianObservationStats.empty(2)
    for row in observations:
        stats = stats.add(row)

    actual = stats.log_likelihood(predicted, noise_std)
    residuals = observations[:, np.newaxis, :] - predicted[np.newaxis, :, :]
    expected = -observations.shape[0] * observations.shape[1] * math.log(
        noise_std * math.sqrt(2.0 * math.pi)
    ) - np.sum(residuals**2, axis=(0, 2)) / (2.0 * noise_std**2)

    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)


def test_sufficient_statistics_match_direct_sum_for_symmetric_unit_rates() -> None:
    values = 1.0 + 1e-9 * np.where(np.arange(100) % 2, 1, -1)
    prediction = np.array([[1.0]])
    sigma = 1e-9
    stats = GaussianObservationStats.empty(1)
    for value in values:
        stats = stats.add(np.array([value]))

    actual = stats.log_likelihood(prediction, sigma)[0]
    expected = sum(
        -math.log(sigma * math.sqrt(2.0 * math.pi))
        - (value - prediction[0, 0]) ** 2 / (2.0 * sigma**2)
        for value in values
    )

    assert actual == pytest.approx(expected, rel=1e-13, abs=1e-13)


def test_centered_sufficient_statistics_are_deeply_immutable() -> None:
    origin = np.array([1.0, 2.0])
    centered_sum = np.array([0.2, -0.1])
    stats = GaussianObservationStats(3.0, origin, centered_sum, 5.0)
    origin[0] = 99.0
    centered_sum[0] = 99.0

    np.testing.assert_array_equal(stats.origin, [1.0, 2.0])
    np.testing.assert_array_equal(stats.centered_sum, [0.2, -0.1])
    with pytest.raises(ValueError, match="WRITEABLE"):
        stats.origin.setflags(write=True)
    with pytest.raises(ValueError, match="WRITEABLE"):
        stats.centered_sum.setflags(write=True)


def test_centered_statistics_avoid_raw_square_overflow_for_zero_residual() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stats = GaussianObservationStats.empty(1).add(np.array([1e308]))
        actual = stats.log_likelihood(np.array([[1e308]]), noise_std=1.0)

    expected = np.array([-math.log(math.sqrt(2.0 * math.pi))])
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_centered_statistics_reject_unrepresentable_update_without_warning() -> None:
    stats = GaussianObservationStats.empty(1).add(np.array([1e308]))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="centered.*finite"):
            stats.add(np.array([-1e308]))


def test_extreme_prediction_has_negative_infinite_likelihood_without_warning() -> None:
    stats = GaussianObservationStats.empty(1).add(np.array([1e308]))
    predictions = np.array([[-1e308], [1e308]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = stats.log_likelihood(predictions, noise_std=1.0)

    assert actual[0] == -np.inf
    assert actual[1] == -math.log(math.sqrt(2.0 * math.pi))


def test_large_noise_has_finite_normalizer_and_stable_scaled_residuals() -> None:
    sigma = 1e308
    stats = GaussianObservationStats.empty(1).add(np.array([1e308]))
    predictions = np.array([[-1e308], [1e308]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = stats.log_likelihood(predictions, noise_std=sigma)

    normalizer = math.log(sigma) + 0.5 * math.log(2.0 * math.pi)
    np.testing.assert_allclose(actual, [-normalizer - 2.0, -normalizer])


def test_completed_square_avoids_overflowing_cross_term_cancellation() -> None:
    observations = np.array([0.0, 1e154])
    stats = GaussianObservationStats.empty(1)
    for value in observations:
        stats = stats.add(np.array([value]))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = stats.log_likelihood(np.array([[1e154]]), noise_std=1.0)[0]

    expected = sum(
        -0.5 * math.log(2.0 * math.pi) - 0.5 * (value - 1e154) ** 2
        for value in observations
    )
    assert np.isfinite(actual)
    assert actual == pytest.approx(expected, rel=1e-15)


@pytest.mark.parametrize(
    ("n", "centered_sum", "centered_sum_squares"),
    [
        (1.0, np.array([2.0]), 1.0),
        (0.0, np.array([0.0]), 1.0),
    ],
)
def test_centered_statistics_reject_materially_inconsistent_constructor(
    n: float, centered_sum: np.ndarray, centered_sum_squares: float
) -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        GaussianObservationStats(
            n=n,
            origin=np.array([0.0]),
            centered_sum=centered_sum,
            centered_sum_squares=centered_sum_squares,
        )


@pytest.mark.parametrize(
    "operation",
    [
        lambda stats: stats.log_likelihood(np.array([[1.0 + 1.0j]]), 1.0),
        lambda stats: stats.log_likelihood(np.array([[1.0]]), 1.0 + 1.0j),
    ],
)
def test_sufficient_statistics_reject_complex_likelihood_inputs_without_warning(
    operation,
) -> None:
    stats = GaussianObservationStats.empty(1).add(np.array([1.0]))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError):
            operation(stats)


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


def test_prior_sampler_rejects_complex_output_without_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="real-valued"):
            ParticleFilter(
                2,
                lambda rng, n: np.ones((n, 1), dtype=np.complex128),
                lambda value: value,
                1.0,
                log_prior_density=lambda values: np.zeros(len(values)),
            )


def test_log_prior_rejects_complex_output_during_construction_without_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="real-valued"):
            ParticleFilter(
                2,
                lambda rng, n: np.zeros((n, 1)),
                lambda value: value,
                1.0,
                log_prior_density=lambda values: np.zeros(
                    len(values), dtype=np.complex128
                ),
            )


def test_log_prior_rejects_complex_output_during_update_without_warning() -> None:
    calls = 0

    def log_prior(values: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        dtype = np.float64 if calls == 1 else np.complex128
        return np.zeros(len(values), dtype=dtype)

    particle_filter = ParticleFilter(
        2,
        lambda rng, n: np.zeros((n, 1)),
        lambda value: value,
        1.0,
        log_prior_density=log_prior,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="real-valued"):
            particle_filter.update(Observation([0.0], 0.0))


@pytest.mark.parametrize("batch", [False, True])
def test_forward_models_reject_complex_output_without_warning(batch: bool) -> None:
    kwargs = {}
    if batch:
        kwargs["forward_model_batch"] = lambda values: np.ones(
            (len(values), 1), dtype=np.complex128
        )
    particle_filter = ParticleFilter(
        2,
        lambda rng, n: np.zeros((n, 1)),
        lambda value: np.ones(1, dtype=np.complex128),
        1.0,
        log_prior_density=lambda values: np.zeros(len(values)),
        **kwargs,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="real-valued"):
            particle_filter.update(Observation([0.0], 0.0))


def test_update_maps_unrepresentable_residual_to_zero_weight_without_warning() -> None:
    initial = np.array([[-1.0], [1.0]])
    particle_filter = ParticleFilter(
        2,
        lambda rng, n: initial.copy(),
        lambda value: np.array([np.copysign(1e308, value[0])]),
        1.0,
        log_prior_density=lambda values: np.zeros(len(values)),
        forward_model_batch=lambda values: np.copysign(1e308, values),
        ess_target=0.1,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        state = particle_filter.update(Observation([1e308], 0.0))

    np.testing.assert_array_equal(state.weights, np.array([0.0, 1.0]))


def test_large_noise_update_uses_stable_scaled_residuals_without_warning() -> None:
    sigma = 1e308
    initial = np.array([[-1.0], [1.0]])
    particle_filter = ParticleFilter(
        2,
        lambda rng, n: initial.copy(),
        lambda value: np.array([np.copysign(1e308, value[0])]),
        sigma,
        log_prior_density=lambda values: np.zeros(len(values)),
        forward_model_batch=lambda values: np.copysign(1e308, values),
        ess_target=0.1,
    )
    observation = Observation([1e308], 0.0)
    normalizer = math.log(sigma) + 0.5 * math.log(2.0 * math.pi)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        direct = particle_filter._observation_log_likelihood(
            particle_filter.state.particles, observation
        )
        state = particle_filter.update(observation)

    np.testing.assert_allclose(direct, [-normalizer - 2.0, -normalizer])
    expected_weights = np.exp(np.array([-2.0, 0.0]))
    expected_weights /= expected_weights.sum()
    np.testing.assert_allclose(state.weights, expected_weights)


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


class _UnitMoveRng:
    def normal(self, size: tuple[int, int]) -> np.ndarray:
        return np.ones(size)

    def random(self, size: int) -> np.ndarray:
        return np.full(size, 0.25)


def test_update_freezes_one_pre_resample_proposal_factor_per_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = np.array([[-1.0], [0.0], [1.0]])
    particle_filter = ParticleFilter(
        3,
        lambda rng, n: initial.copy(),
        lambda value: value,
        0.1,
        log_prior_density=lambda values: np.zeros(len(values)),
        rejuvenation_steps=3,
        rng=np.random.default_rng(1),
    )
    particle_filter.rng = _UnitMoveRng()  # type: ignore[assignment]
    beta_schedule = iter((0.5, 1.0))
    factors = iter((np.array([[0.25]]), np.array([[0.5]])))
    events: list[tuple[str, object]] = []
    original_move = particle_filter._metropolis_move

    def factor_before_resampling(
        particles: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        factor = next(factors)
        events.append(("factor", factor))
        return factor

    def identity_resampling(weights: np.ndarray) -> np.ndarray:
        events.append(("resample", weights.copy()))
        return np.arange(len(weights))

    def recorded_move(
        particles: np.ndarray,
        *,
        beta: float,
        observation: Observation,
        proposal_chol: np.ndarray,
    ) -> tuple[np.ndarray, object]:
        events.append(("move", proposal_chol))
        return original_move(
            particles,
            beta=beta,
            observation=observation,
            proposal_chol=proposal_chol,
        )

    monkeypatch.setattr(
        inference_module, "_next_beta", lambda *args, **kwargs: next(beta_schedule)
    )
    monkeypatch.setattr(
        particle_filter,
        "_observation_log_likelihood",
        lambda particles, observation: np.array([0.0, -10.0, -20.0]),
    )
    monkeypatch.setattr(particle_filter, "_proposal_cholesky", factor_before_resampling)
    monkeypatch.setattr(particle_filter, "_resample_indices", identity_resampling)
    monkeypatch.setattr(
        particle_filter,
        "_log_target",
        lambda particles, **kwargs: np.zeros(len(particles)),
    )
    monkeypatch.setattr(particle_filter, "_metropolis_move", recorded_move)

    state = particle_filter.update(Observation([0.0], 0.0))

    assert [event for event, _ in events] == [
        "factor",
        "resample",
        "move",
        "factor",
        "resample",
        "move",
    ]
    assert events[2][1] is events[0][1]
    assert events[5][1] is events[3][1]
    np.testing.assert_array_equal(state.particles, initial + 3 * 0.25 + 3 * 0.5)
    assert particle_filter.last_diagnostics.mh_proposals == 2 * 3 * 3


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


def _assert_filter_matches_snapshot(
    particle_filter: ParticleFilter,
    *,
    state: object,
    history: list[object],
    diagnostics_history: list[object],
    log_evidence_history: list[float],
    log_evidence: float,
    last_diagnostics: object,
    last_increments: tuple[float, ...],
    rng_state: dict[str, object],
) -> None:
    assert particle_filter.state is state
    assert len(particle_filter.history) == len(history)
    assert all(
        actual is expected
        for actual, expected in zip(particle_filter.history, history, strict=True)
    )
    assert particle_filter.diagnostics_history == diagnostics_history
    assert particle_filter.log_evidence_history == log_evidence_history
    assert particle_filter.log_evidence == log_evidence
    assert particle_filter.last_diagnostics == last_diagnostics
    assert particle_filter.last_log_evidence_increments == last_increments
    assert particle_filter.rng.bit_generator.state == rng_state


def test_failed_particle_update_rolls_back_all_state_and_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    particle_filter = _normal_filter(31, n_particles=200)
    control = _normal_filter(31, n_particles=200)
    snapshot = {
        "state": particle_filter.state,
        "history": particle_filter.history,
        "diagnostics_history": particle_filter.diagnostics_history,
        "log_evidence_history": particle_filter.log_evidence_history,
        "log_evidence": particle_filter.log_evidence,
        "last_diagnostics": particle_filter.last_diagnostics,
        "last_increments": particle_filter.last_log_evidence_increments,
        "rng_state": copy.deepcopy(particle_filter.rng.bit_generator.state),
    }
    original_move = particle_filter._metropolis_move

    def fail_after_rng_draw(*args: object, **kwargs: object) -> None:
        particle_filter.rng.random()
        raise RuntimeError("injected later-stage failure")

    monkeypatch.setattr(particle_filter, "_metropolis_move", fail_after_rng_draw)
    observation = Observation([0.45], 0.0)
    with pytest.raises(RuntimeError, match="injected later-stage failure"):
        particle_filter.update(observation)

    _assert_filter_matches_snapshot(particle_filter, **snapshot)

    monkeypatch.setattr(particle_filter, "_metropolis_move", original_move)
    actual = particle_filter.update(observation)
    expected = control.update(observation)
    np.testing.assert_array_equal(actual.particles, expected.particles)
    np.testing.assert_array_equal(actual.weights, expected.weights)
    assert particle_filter.log_evidence == control.log_evidence
    assert particle_filter.rng.bit_generator.state == control.rng.bit_generator.state


def test_failed_model_comparison_update_rolls_back_every_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filters = {1: _normal_filter(41, 200), 2: _normal_filter(42, 200)}
    comparison = ModelComparison(filters)
    snapshots = {
        key: {
            "state": particle_filter.state,
            "history": particle_filter.history,
            "diagnostics_history": particle_filter.diagnostics_history,
            "log_evidence_history": particle_filter.log_evidence_history,
            "log_evidence": particle_filter.log_evidence,
            "last_diagnostics": particle_filter.last_diagnostics,
            "last_increments": particle_filter.last_log_evidence_increments,
            "rng_state": copy.deepcopy(particle_filter.rng.bit_generator.state),
        }
        for key, particle_filter in filters.items()
    }

    def fail_after_rng_draw(*args: object, **kwargs: object) -> None:
        filters[2].rng.random()
        raise RuntimeError("injected second-filter failure")

    monkeypatch.setattr(filters[2], "_metropolis_move", fail_after_rng_draw)
    before_evidence = comparison.evidence()
    with pytest.raises(RuntimeError, match="injected second-filter failure"):
        comparison.update(Observation([0.45], 0.0))

    assert comparison.evidence() == before_evidence
    for key, particle_filter in filters.items():
        _assert_filter_matches_snapshot(particle_filter, **snapshots[key])
