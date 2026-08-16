"""Data structures for the gravitational time dilation simulation."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from clocks._validation import finite_float, finite_float_array


def _positions_array(value: object) -> NDArray[np.float64]:
    positions = np.array(value, dtype=np.float64, copy=True)
    if positions.ndim == 1:
        positions = positions.reshape(-1, 1)
    return finite_float_array("positions", positions, ndim=2)


def _masses_array(value: object) -> NDArray[np.float64]:
    masses = np.array(value, dtype=np.float64, copy=True)
    if masses.ndim == 0:
        masses = masses.reshape(1)
    return finite_float_array("masses", masses, ndim=1)


@dataclass(frozen=True)
class MassConfig:
    """One or more point masses in space.

    positions: (n_masses, n_dims) array of mass locations
    masses: (n_masses,) array of mass values
    """

    positions: NDArray[np.float64]
    masses: NDArray[np.float64]

    def __post_init__(self) -> None:
        positions = _positions_array(self.positions)
        masses = _masses_array(self.masses)
        if positions.shape[0] != masses.shape[0]:
            raise ValueError(
                f"Number of positions ({positions.shape[0]}) must match "
                f"number of masses ({masses.shape[0]})"
            )
        if np.any(masses < 0):
            raise ValueError("masses must be nonnegative")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "masses", masses)


@dataclass(frozen=True)
class ClockArray:
    """Array of clocks at fixed positions.

    positions: (n_clocks, n_dims) array of clock locations
    track_offset: perpendicular distance between the clock array and the
        subspace the masses live in (a parallel track in 1D, an offset
        plane in 2D; 0.0 for co-located, as in 3D)
    """

    positions: NDArray[np.float64]
    track_offset: float = 0.0

    def __post_init__(self) -> None:
        positions = _positions_array(self.positions)
        track_offset = finite_float("track_offset", self.track_offset)
        if track_offset < 0:
            raise ValueError("track_offset must be nonnegative")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "track_offset", track_offset)


@dataclass(frozen=True)
class Observation:
    """A single noisy observation of clock rates.

    rates: (n_clocks,) array of observed tick rates
    time: simulation time of this observation
    """

    rates: NDArray[np.float64]
    time: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "rates", finite_float_array("rates", self.rates, ndim=1)
        )
        object.__setattr__(self, "time", finite_float("time", self.time))


@dataclass(frozen=True)
class ParticleState:
    """State of the particle filter at a point in time.

    particles: (n_particles, n_params) array of parameter guesses
    weights: (n_particles,) normalized weights
    observations_seen: number of observations incorporated so far
    """

    particles: NDArray[np.float64]
    weights: NDArray[np.float64]
    observations_seen: int

    def __post_init__(self) -> None:
        particles = finite_float_array("particles", self.particles, ndim=2)
        weights = finite_float_array("weights", self.weights, ndim=1)
        if particles.shape[0] != weights.shape[0]:
            raise ValueError(
                f"Number of particles ({particles.shape[0]}) must match "
                f"number of weights ({weights.shape[0]})"
            )
        if np.any(weights < 0):
            raise ValueError("weights must be nonnegative")
        if abs(float(weights.sum()) - 1.0) > 1e-12:
            raise ValueError("weights must sum to 1 within 1e-12")
        if isinstance(self.observations_seen, bool) or not isinstance(
            self.observations_seen, Integral
        ):
            raise ValueError("observations_seen must be a nonnegative integer")
        observations_seen = int(self.observations_seen)
        if observations_seen < 0:
            raise ValueError("observations_seen must be a nonnegative integer")
        object.__setattr__(self, "particles", particles)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "observations_seen", observations_seen)


@dataclass(frozen=True)
class UpdateDiagnostics:
    """Tempering and Metropolis-Hastings counts for one observation update."""

    tempering_stages: int = 0
    mh_proposals: int = 0
    mh_acceptances: int = 0

    @property
    def acceptance_rate(self) -> float:
        if self.mh_proposals == 0:
            return 0.0
        return self.mh_acceptances / self.mh_proposals
