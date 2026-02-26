"""Data structures for the gravitational time dilation simulation."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MassConfig:
    """One or more point masses in space.

    positions: (n_masses, n_dims) array of mass locations
    masses: (n_masses,) array of mass values
    """

    positions: NDArray[np.floating]
    masses: NDArray[np.floating]

    def __post_init__(self) -> None:
        if self.positions.ndim == 1:
            object.__setattr__(self, "positions", self.positions.reshape(-1, 1))
        if self.masses.ndim == 0:
            object.__setattr__(self, "masses", self.masses.reshape(1))
        if self.positions.shape[0] != self.masses.shape[0]:
            raise ValueError(
                f"Number of positions ({self.positions.shape[0]}) must match "
                f"number of masses ({self.masses.shape[0]})"
            )


@dataclass(frozen=True)
class ClockArray:
    """Array of clocks at fixed positions.

    positions: (n_clocks, n_dims) array of clock locations
    track_offset: perpendicular offset from mass track (for 1D mass on track)
    """

    positions: NDArray[np.floating]
    track_offset: float = 0.0

    def __post_init__(self) -> None:
        if self.positions.ndim == 1:
            object.__setattr__(self, "positions", self.positions.reshape(-1, 1))


@dataclass(frozen=True)
class Observation:
    """A single noisy observation of clock rates.

    rates: (n_clocks,) array of observed tick rates
    time: simulation time of this observation
    """

    rates: NDArray[np.floating]
    time: float


@dataclass(frozen=True)
class ParticleState:
    """State of the particle filter at a point in time.

    particles: (n_particles, n_params) array of parameter guesses
    weights: (n_particles,) normalized weights
    observations_seen: number of observations incorporated so far
    """

    particles: NDArray[np.floating]
    weights: NDArray[np.floating]
    observations_seen: int
