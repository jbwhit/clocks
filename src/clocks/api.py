"""Public end-to-end API for simulation and rigorous SMC inference."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from clocks._rng import study_generator
from clocks._support import make_point_mass_prior_sampler, point_mass_support_mask
from clocks.config import InferenceConfig, SimulationConfig
from clocks.inference import ModelComparison, ParticleFilter
from clocks.noise import add_clock_noise
from clocks.physics import clock_rates, clock_rates_batch, clock_rates_batch_multi
from clocks.results import (
    HistoryEntry,
    InferenceResult,
    ModelComparisonInferenceResult,
    SimulationResult,
)
from clocks.types import MassConfig, Observation, ParticleState, UpdateDiagnostics


def simulate(config: SimulationConfig) -> SimulationResult:
    """Generate synthetic observations from a ground-truth configuration."""
    rng = study_generator(config.seed)
    true_rates = clock_rates(config.ground_truth, config.clock_array)
    observations = [
        Observation(
            add_clock_noise(true_rates, config.noise.observation_std, rng), float(t)
        )
        for t in range(config.n_observations)
    ]
    return SimulationResult(
        clock_array=config.clock_array,
        ground_truth=config.ground_truth,
        true_rates=true_rates,
        observations=observations,
        noise=config.noise,
        seed=config.seed,
    )


def _validate_observations(
    observations: list[Observation], config: InferenceConfig
) -> None:
    if not observations:
        raise ValueError("observations must not be empty")
    n_channels = len(config.clock_array.positions)
    for index, observation in enumerate(observations):
        if not isinstance(observation, Observation):
            raise TypeError(f"observations[{index}] must be an Observation")
        if observation.rates.shape != (n_channels,):
            raise ValueError(
                f"observation channel count must be {n_channels}, "
                f"got {observation.rates.size} at index {index}"
            )


def infer(
    observations: list[Observation], config: InferenceConfig
) -> InferenceResult | ModelComparisonInferenceResult:
    """Run inference against a nonempty sequence of observations."""
    _validate_observations(observations, config)
    if isinstance(config.n_masses, tuple):
        return _infer_model_comparison(observations, config)
    particle_filter = build_particle_filter(config)
    for observation in observations:
        particle_filter.update(observation)
    return _inference_result_from_particle_filter(particle_filter)


def simulate_and_infer(
    simulation_config: SimulationConfig,
    inference_config: InferenceConfig,
) -> InferenceResult | ModelComparisonInferenceResult:
    """Generate synthetic data and immediately infer its parameters."""
    simulation = simulate(simulation_config)
    result = infer(simulation.observations, inference_config)
    return result.with_simulation(simulation)


def _make_prior_sampler(config: InferenceConfig, n_masses: int, n_dims: int):
    return make_point_mass_prior_sampler(
        n_masses=n_masses,
        n_dims=n_dims,
        clock_array=config.clock_array,
        position_range=config.prior.position_range,
        mass_range=config.prior.mass_range,
    )


def _make_forward_model(config: InferenceConfig, n_masses: int, n_dims: int):
    def forward(params: NDArray[np.floating]) -> NDArray[np.floating]:
        positions = params[: n_masses * n_dims].reshape(n_masses, n_dims)
        masses = params[n_masses * n_dims :]
        return clock_rates(MassConfig(positions, masses), config.clock_array)

    return forward


def _make_forward_model_batch(config: InferenceConfig, n_masses: int, n_dims: int):
    if n_masses == 1:

        def forward_batch_single(
            particles: NDArray[np.floating],
        ) -> NDArray[np.floating]:
            return clock_rates_batch(
                particles[:, :n_dims], particles[:, n_dims], config.clock_array
            )

        return forward_batch_single

    def forward_batch_multi(
        particles: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        positions = particles[:, : n_masses * n_dims].reshape(-1, n_masses, n_dims)
        masses = particles[:, n_masses * n_dims :]
        return clock_rates_batch_multi(positions, masses, config.clock_array)

    return forward_batch_multi


def _make_log_prior(config: InferenceConfig, n_masses: int, n_dims: int):
    def log_prior_density(
        particles: NDArray[np.floating],
    ) -> NDArray[np.float64]:
        valid = point_mass_support_mask(
            particles,
            n_masses=n_masses,
            n_dims=n_dims,
            clock_array=config.clock_array,
            position_range=config.prior.position_range,
            mass_range=config.prior.mass_range,
        )
        return np.where(valid, 0.0, -np.inf)

    return log_prior_density


def _build_fixed_filter(
    config: InferenceConfig, n_masses: int, rng: np.random.Generator
) -> ParticleFilter:
    n_dims = config.clock_array.positions.shape[1]
    return ParticleFilter(
        config.n_particles,
        _make_prior_sampler(config, n_masses, n_dims),
        _make_forward_model(config, n_masses, n_dims),
        config.noise.observation_std,
        log_prior_density=_make_log_prior(config, n_masses, n_dims),
        forward_model_batch=_make_forward_model_batch(config, n_masses, n_dims),
        resampling=config.resampling,
        ess_target=config.ess_target,
        rejuvenation_steps=config.rejuvenation_steps,
        proposal_scale=config.proposal_scale,
        rng=rng,
    )


def build_particle_filter(config: InferenceConfig) -> ParticleFilter:
    """Construct the fixed-K filter used by :func:`infer`."""
    if isinstance(config.n_masses, tuple):
        raise TypeError("expected int for n_masses in fixed-K mode")
    return _build_fixed_filter(
        config, cast(int, config.n_masses), study_generator(config.seed)
    )


def build_model_comparison(config: InferenceConfig) -> ModelComparison:
    """Construct independently seeded fixed-K filters for model comparison."""
    if not isinstance(config.n_masses, tuple):
        raise TypeError("expected tuple for n_masses in model-comparison mode")
    candidate_models = tuple(sorted(set(config.n_masses)))
    child_sequences = np.random.SeedSequence(config.seed).spawn(len(candidate_models))
    filters = {
        k: _build_fixed_filter(config, k, study_generator(child_sequence))
        for k, child_sequence in zip(candidate_models, child_sequences, strict=True)
    }
    return ModelComparison(filters)


def _history_entry_from_state(
    state: ParticleState, log_evidence: float, diagnostics: UpdateDiagnostics
) -> HistoryEntry:
    mean = np.average(state.particles, weights=state.weights, axis=0)
    variance = np.average((state.particles - mean) ** 2, weights=state.weights, axis=0)
    return HistoryEntry(
        mean=mean,
        std=np.sqrt(variance),
        ess=float(1.0 / np.sum(state.weights**2)),
        observations_seen=state.observations_seen,
        log_evidence=log_evidence,
        diagnostics=diagnostics,
    )


def _inference_result_from_particle_filter(
    particle_filter: ParticleFilter,
) -> InferenceResult:
    estimate = particle_filter.estimate()
    history = [
        _history_entry_from_state(state, evidence, diagnostics)
        for state, evidence, diagnostics in zip(
            particle_filter.history[1:],
            particle_filter.log_evidence_history,
            particle_filter.diagnostics_history,
            strict=True,
        )
    ]
    return InferenceResult(
        posterior_mean=estimate["mean"],
        posterior_std=estimate["std"],
        ess=estimate["ess"],
        log_evidence=particle_filter.log_evidence,
        history=history,
        particle_state=particle_filter.state,
    )


def _infer_model_comparison(
    observations: list[Observation], config: InferenceConfig
) -> ModelComparisonInferenceResult:
    comparison = build_model_comparison(config)
    history: list[dict[int, float]] = []
    for observation in observations:
        comparison.update(observation)
        history.append(comparison.evidence()["posterior"])
    evidence = comparison.evidence()
    posterior = evidence["posterior"]
    return ModelComparisonInferenceResult(
        posterior_by_model=posterior,
        log_evidence_by_model=evidence["log_evidence"],
        best_model=max(posterior, key=posterior.__getitem__),
        result_by_model={
            k: _inference_result_from_particle_filter(particle_filter)
            for k, particle_filter in comparison.filters.items()
        },
        history=history,
    )
