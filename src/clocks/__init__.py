"""Gravitational time dilation simulation and inference library."""

from clocks.api import (
    build_model_comparison,
    build_particle_filter,
    infer,
    simulate,
    simulate_and_infer,
)
from clocks.config import InferenceConfig, NoiseConfig, PriorConfig, SimulationConfig
from clocks.inference import (
    ConvergenceInfo,
    Estimate,
    GaussianObservationStats,
    ModelComparison,
    ModelComparisonResult,
    ParticleFilter,
)
from clocks.noise import (
    add_clock_noise,
    log_likelihood_gaussian,
    log_likelihood_gaussian_batch,
)
from clocks.physics import (
    WEAK_FIELD_LIMIT,
    PhysicsDomainError,
    clock_rates,
    clock_rates_batch,
    clock_rates_batch_multi,
    clock_rates_density_gaussian,
    clock_rates_density_gaussian_batch,
    compute_distances,
    gravitational_potential,
    time_dilation_factor,
)
from clocks.results import (
    HistoryEntry,
    InferenceResult,
    ModelComparisonInferenceResult,
    SimulationResult,
)
from clocks.types import (
    ClockArray,
    MassConfig,
    Observation,
    ParticleState,
    UpdateDiagnostics,
)

__all__ = [
    "ClockArray",
    "ConvergenceInfo",
    "Estimate",
    "GaussianObservationStats",
    "HistoryEntry",
    "InferenceConfig",
    "InferenceResult",
    "MassConfig",
    "ModelComparison",
    "ModelComparisonInferenceResult",
    "ModelComparisonResult",
    "NoiseConfig",
    "Observation",
    "ParticleFilter",
    "ParticleState",
    "PriorConfig",
    "PhysicsDomainError",
    "SimulationConfig",
    "SimulationResult",
    "UpdateDiagnostics",
    "WEAK_FIELD_LIMIT",
    "add_clock_noise",
    "build_particle_filter",
    "build_model_comparison",
    "clock_rates",
    "clock_rates_batch",
    "clock_rates_batch_multi",
    "clock_rates_density_gaussian",
    "clock_rates_density_gaussian_batch",
    "compute_distances",
    "gravitational_potential",
    "infer",
    "log_likelihood_gaussian",
    "log_likelihood_gaussian_batch",
    "simulate",
    "simulate_and_infer",
    "time_dilation_factor",
]
