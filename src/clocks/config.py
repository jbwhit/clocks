"""Public configuration objects for the clocks library API."""

import math
from dataclasses import dataclass

from clocks.types import ClockArray, MassConfig


@dataclass(frozen=True)
class NoiseConfig:
    """Observation noise settings."""

    observation_std: float

    def __post_init__(self) -> None:
        if self.observation_std <= 0:
            raise ValueError("observation_std must be > 0")


@dataclass(frozen=True)
class PriorConfig:
    """Prior bounds for position and mass parameters."""

    position_range: tuple[float, float]
    mass_range: tuple[float, float]

    def __post_init__(self) -> None:
        if self.position_range[0] >= self.position_range[1]:
            raise ValueError("position_range must be increasing")
        if self.mass_range[0] <= 0 or self.mass_range[0] >= self.mass_range[1]:
            raise ValueError("mass_range must be positive and increasing")


@dataclass(frozen=True)
class InferenceConfig:
    """Top-level config for end-to-end inference.

    ``jitter_std`` scales the post-resampling jitter: an absolute standard
    deviation when ``jitter="fixed"``, a fraction of the particle cloud's
    weighted covariance when ``jitter="covariance"``, or the floor (late-run
    asymptote) when ``jitter="annealed"``. ``jitter_tau`` is the anneal time
    constant, in observations, for the ``"annealed"`` schedule.
    """

    clock_array: ClockArray
    noise: NoiseConfig
    prior: PriorConfig
    n_particles: int
    n_masses: int | tuple[int, ...]
    jitter_std: float = 0.02
    resampling: str = "systematic"
    jitter: str = "annealed"
    jitter_tau: float = 15.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.n_particles <= 0:
            raise ValueError("n_particles must be > 0")
        if isinstance(self.n_masses, int):
            if self.n_masses <= 0:
                raise ValueError("n_masses must be > 0")
        else:
            if not self.n_masses:
                raise ValueError("n_masses candidates must not be empty")
            if any(k <= 0 for k in self.n_masses):
                raise ValueError("n_masses candidates must all be > 0")
        if not math.isfinite(self.jitter_std) or self.jitter_std < 0:
            raise ValueError(
                f"jitter_std must be finite and >= 0, got {self.jitter_std}"
            )
        if not math.isfinite(self.jitter_tau) or self.jitter_tau <= 0:
            raise ValueError(
                f"jitter_tau must be finite and > 0, got {self.jitter_tau}"
            )


@dataclass(frozen=True)
class SimulationConfig:
    """Top-level config for generating synthetic observations."""

    clock_array: ClockArray
    ground_truth: MassConfig
    noise: NoiseConfig
    n_observations: int
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.n_observations <= 0:
            raise ValueError("n_observations must be > 0")
