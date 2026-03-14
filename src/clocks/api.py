"""Public end-to-end API for clocks simulation and inference."""

import numpy as np
from numpy.typing import NDArray

from clocks.config import InferenceConfig, SimulationConfig
from clocks.inference import ParticleFilter
from clocks.noise import add_clock_noise
from clocks.physics import clock_rates, clock_rates_batch, clock_rates_batch_multi
from clocks.results import (
    HistoryEntry,
    InferenceResult,
    ModelComparisonInferenceResult,
    SimulationResult,
)
from clocks.types import MassConfig, Observation, ParticleState


def simulate(config: SimulationConfig) -> SimulationResult:
    """Generate synthetic observations from a ground-truth mass configuration."""
    rng = np.random.default_rng(config.seed)
    true_rates = clock_rates(config.ground_truth, config.clock_array)
    observations = [
        Observation(
            rates=add_clock_noise(true_rates, config.noise.observation_std, rng),
            time=float(t),
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


def infer(
    observations: list[Observation], config: InferenceConfig
) -> InferenceResult | ModelComparisonInferenceResult:
    """Run inference against a list of observations."""
    if isinstance(config.n_masses, tuple):
        raise NotImplementedError("model comparison is implemented in a later task")

    particle_filter = _build_particle_filter(config)
    for observation in observations:
        particle_filter.update(observation)
    return _inference_result_from_particle_filter(particle_filter)


def simulate_and_infer(
    simulation_config: SimulationConfig,
    inference_config: InferenceConfig,
) -> InferenceResult | ModelComparisonInferenceResult:
    """Generate synthetic data and immediately run inference over it."""
    raise NotImplementedError("simulate_and_infer() is implemented in a later task")


def _build_particle_filter(config: InferenceConfig) -> ParticleFilter:
    n_masses = config.n_masses
    assert isinstance(n_masses, int)
    n_dims = config.clock_array.positions.shape[1]
    rng = np.random.default_rng(config.seed)

    return ParticleFilter(
        n_particles=config.n_particles,
        prior_sampler=_make_prior_sampler(config, n_masses, n_dims),
        forward_model=_make_forward_model(config, n_masses, n_dims),
        noise_std=config.noise.observation_std,
        jitter_std=config.jitter_std,
        rng=rng,
        forward_model_batch=_make_forward_model_batch(config, n_masses, n_dims),
        constraint_fn=_make_constraint_fn(n_masses, n_dims) if n_masses > 1 else None,
        resampling=config.resampling,
        jitter=config.jitter,
        log_prior=_make_log_prior(config, n_masses, n_dims),
    )


def _make_prior_sampler(config: InferenceConfig, n_masses: int, n_dims: int):
    position_range = config.prior.position_range
    mass_range = config.prior.mass_range

    def sampler(rng: np.random.Generator, n: int) -> NDArray[np.floating]:
        positions = rng.uniform(
            position_range[0],
            position_range[1],
            (n, n_masses, n_dims),
        )
        if n_masses > 1:
            sort_idx = np.argsort(positions[:, :, 0], axis=1)
            for dim in range(n_dims):
                positions[:, :, dim] = np.take_along_axis(
                    positions[:, :, dim], sort_idx, axis=1
                )
        masses = rng.uniform(mass_range[0], mass_range[1], (n, n_masses))
        return np.concatenate([positions.reshape(n, n_masses * n_dims), masses], axis=1)

    return sampler


def _make_forward_model(config: InferenceConfig, n_masses: int, n_dims: int):
    def forward(params: NDArray[np.floating]) -> NDArray[np.floating]:
        positions = params[: n_masses * n_dims].reshape(n_masses, n_dims)
        masses = params[n_masses * n_dims :]
        return clock_rates(
            MassConfig(positions=positions, masses=masses),
            config.clock_array,
        )

    return forward


def _make_forward_model_batch(config: InferenceConfig, n_masses: int, n_dims: int):
    if n_masses == 1:

        def forward_batch_single(
            particles: NDArray[np.floating],
        ) -> NDArray[np.floating]:
            return clock_rates_batch(
                particles[:, :n_dims],
                particles[:, n_dims],
                config.clock_array,
            )

        return forward_batch_single

    def forward_batch_multi(particles: NDArray[np.floating]) -> NDArray[np.floating]:
        positions = particles[:, : n_masses * n_dims].reshape(-1, n_masses, n_dims)
        masses = particles[:, n_masses * n_dims :]
        return clock_rates_batch_multi(positions, masses, config.clock_array)

    return forward_batch_multi


def _make_constraint_fn(n_masses: int, n_dims: int):
    def constraint(particles: NDArray[np.floating]) -> NDArray[np.floating]:
        positions = particles[:, : n_masses * n_dims].reshape(-1, n_masses, n_dims)
        masses = particles[:, n_masses * n_dims :].reshape(-1, n_masses)
        sort_idx = np.argsort(positions[:, :, 0], axis=1)
        for dim in range(n_dims):
            positions[:, :, dim] = np.take_along_axis(
                positions[:, :, dim], sort_idx, axis=1
            )
        masses = np.take_along_axis(masses, sort_idx, axis=1)
        particles[:, : n_masses * n_dims] = positions.reshape(-1, n_masses * n_dims)
        particles[:, n_masses * n_dims :] = masses
        return particles

    return constraint


def _make_log_prior(config: InferenceConfig, n_masses: int, n_dims: int):
    position_range = config.prior.position_range

    def log_prior(particles: NDArray[np.floating]) -> NDArray[np.floating]:
        log_prob = np.zeros(particles.shape[0])
        positions = particles[:, : n_masses * n_dims]
        masses = particles[:, n_masses * n_dims :]
        out_of_range = np.any(
            (positions < position_range[0]) | (positions > position_range[1]),
            axis=1,
        )
        invalid_masses = np.any(masses <= 0, axis=1)
        log_prob[out_of_range | invalid_masses] = -np.inf
        return log_prob

    return log_prior


def _history_entry_from_state(state: ParticleState) -> HistoryEntry:
    mean = np.average(state.particles, weights=state.weights, axis=0)
    var = np.average((state.particles - mean) ** 2, weights=state.weights, axis=0)
    return HistoryEntry(
        mean=mean,
        std=np.sqrt(var),
        ess=float(1.0 / np.sum(state.weights**2)),
        observations_seen=state.observations_seen,
    )


def _inference_result_from_particle_filter(
    particle_filter: ParticleFilter,
) -> InferenceResult:
    estimate = particle_filter.estimate()
    history = [
        _history_entry_from_state(state) for state in particle_filter.history[1:]
    ]
    return InferenceResult(
        posterior_mean=estimate["mean"],
        posterior_std=estimate["std"],
        ess=estimate["ess"],
        history=history,
        particle_state=particle_filter.state,
    )
