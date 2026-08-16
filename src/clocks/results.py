"""Public result objects for the clocks library API."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from clocks._validation import finite_float_array
from clocks.config import NoiseConfig
from clocks.types import (
    ClockArray,
    MassConfig,
    Observation,
    ParticleState,
    UpdateDiagnostics,
)


def _mass_config_to_dict(config: MassConfig) -> dict[str, object]:
    return {
        "positions": config.positions.tolist(),
        "masses": config.masses.tolist(),
    }


def _clock_array_to_dict(clock_array: ClockArray) -> dict[str, object]:
    return {
        "positions": clock_array.positions.tolist(),
        "track_offset": clock_array.track_offset,
    }


def _observation_to_dict(observation: Observation) -> dict[str, object]:
    return {
        "rates": observation.rates.tolist(),
        "time": observation.time,
    }


@dataclass(frozen=True)
class HistoryEntry:
    """Posterior summary for one observation step."""

    mean: NDArray[np.floating]
    std: NDArray[np.floating]
    ess: float
    observations_seen: int
    log_evidence: float
    diagnostics: UpdateDiagnostics

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean", finite_float_array("mean", self.mean, ndim=1))
        object.__setattr__(self, "std", finite_float_array("std", self.std, ndim=1))

    def to_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "ess": self.ess,
            "observations_seen": self.observations_seen,
            "log_evidence": self.log_evidence,
            "diagnostics": {
                "tempering_stages": self.diagnostics.tempering_stages,
                "mh_proposals": self.diagnostics.mh_proposals,
                "mh_acceptances": self.diagnostics.mh_acceptances,
                "acceptance_rate": self.diagnostics.acceptance_rate,
            },
        }


@dataclass(frozen=True)
class SimulationResult:
    """Synthetic observations and their generating state."""

    clock_array: ClockArray
    ground_truth: MassConfig
    true_rates: NDArray[np.floating]
    observations: tuple[Observation, ...]
    noise: NoiseConfig
    seed: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "true_rates",
            finite_float_array("true_rates", self.true_rates, ndim=1),
        )
        object.__setattr__(self, "observations", tuple(self.observations))

    def to_dict(self) -> dict[str, object]:
        return {
            "clock_array": _clock_array_to_dict(self.clock_array),
            "ground_truth": _mass_config_to_dict(self.ground_truth),
            "true_rates": self.true_rates.tolist(),
            "observations": [_observation_to_dict(obs) for obs in self.observations],
            "noise": {"observation_std": self.noise.observation_std},
            "seed": self.seed,
        }


@dataclass(frozen=True)
class InferenceResult:
    """Posterior summary for a fixed-K inference run."""

    posterior_mean: NDArray[np.floating]
    posterior_std: NDArray[np.floating]
    ess: float
    log_evidence: float
    history: tuple[HistoryEntry, ...]
    particle_state: ParticleState | None = None
    simulation: SimulationResult | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "posterior_mean",
            finite_float_array("posterior_mean", self.posterior_mean, ndim=1),
        )
        object.__setattr__(
            self,
            "posterior_std",
            finite_float_array("posterior_std", self.posterior_std, ndim=1),
        )
        object.__setattr__(self, "history", tuple(self.history))

    def with_simulation(self, simulation: SimulationResult) -> "InferenceResult":
        return InferenceResult(
            posterior_mean=self.posterior_mean,
            posterior_std=self.posterior_std,
            ess=self.ess,
            log_evidence=self.log_evidence,
            history=self.history,
            particle_state=self.particle_state,
            simulation=simulation,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "posterior_mean": self.posterior_mean.tolist(),
            "posterior_std": self.posterior_std.tolist(),
            "ess": self.ess,
            "log_evidence": self.log_evidence,
            "history": [entry.to_dict() for entry in self.history],
        }
        if self.particle_state is not None:
            payload["particle_state"] = {
                "particles": self.particle_state.particles.tolist(),
                "weights": self.particle_state.weights.tolist(),
                "observations_seen": self.particle_state.observations_seen,
            }
        if self.simulation is not None:
            payload["simulation"] = self.simulation.to_dict()
        return payload


@dataclass(frozen=True)
class ModelComparisonInferenceResult:
    """Posterior summary for model comparison runs."""

    posterior_by_model: Mapping[int, float]
    log_evidence_by_model: Mapping[int, float]
    best_model: int
    result_by_model: Mapping[int, InferenceResult]
    history: tuple[Mapping[int, float], ...]
    simulation: SimulationResult | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "posterior_by_model",
            MappingProxyType(dict(self.posterior_by_model)),
        )
        object.__setattr__(
            self,
            "log_evidence_by_model",
            MappingProxyType(dict(self.log_evidence_by_model)),
        )
        object.__setattr__(
            self,
            "result_by_model",
            MappingProxyType(dict(self.result_by_model)),
        )
        object.__setattr__(
            self,
            "history",
            tuple(MappingProxyType(dict(entry)) for entry in self.history),
        )

    def with_simulation(
        self, simulation: SimulationResult
    ) -> "ModelComparisonInferenceResult":
        return ModelComparisonInferenceResult(
            posterior_by_model=self.posterior_by_model,
            log_evidence_by_model=self.log_evidence_by_model,
            best_model=self.best_model,
            result_by_model=self.result_by_model,
            history=self.history,
            simulation=simulation,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "posterior_by_model": dict(self.posterior_by_model),
            "log_evidence_by_model": dict(self.log_evidence_by_model),
            "best_model": self.best_model,
            "result_by_model": {
                k: result.to_dict() for k, result in self.result_by_model.items()
            },
            "history": [dict(entry) for entry in self.history],
        }
        if self.simulation is not None:
            payload["simulation"] = self.simulation.to_dict()
        return payload
