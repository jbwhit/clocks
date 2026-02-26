"""Forward model: gravitational potential and time dilation.

Uses simulation units where G = 1, c = 1.
"""

import numpy as np
from numpy.typing import NDArray

from clocks.types import ClockArray, MassConfig

# Guard against singularities inside the Schwarzschild radius
_EPS = 1e-15


def compute_distances(
    clock_positions: NDArray[np.floating],
    mass_positions: NDArray[np.floating],
    track_offset: float = 0.0,
) -> NDArray[np.floating]:
    """Euclidean distance from each clock to each mass.

    Returns: (n_clocks, n_masses) distance array.
    """
    # (n_clocks, 1, n_dims) - (1, n_masses, n_dims)
    diff = clock_positions[:, np.newaxis, :] - mass_positions[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1) + track_offset**2
    return np.sqrt(dist_sq)


def gravitational_potential(
    distances: NDArray[np.floating],
    masses: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Newtonian gravitational potential Phi at each clock (negative by convention).

    Phi_i = -sum_j (M_j / r_ij)

    Returns: (n_clocks,) potential array.
    """
    # distances: (n_clocks, n_masses), masses: (n_masses,)
    safe_dist = np.maximum(distances, _EPS)
    return -np.sum(masses[np.newaxis, :] / safe_dist, axis=1)


def time_dilation_factor(
    potential: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Gravitational time dilation factor: dtau/dt = sqrt(1 + 2*Phi).

    In GR with c=1, the metric gives dtau/dt = sqrt(1 + 2*Phi/c^2).
    Since Phi is negative, clocks in deeper potential wells tick slower.
    Clamped above eps to avoid sqrt of negative (black hole regime).

    Returns: (n_clocks,) dilation factors in [0, 1].
    """
    argument = 1.0 + 2.0 * potential
    return np.sqrt(np.maximum(argument, _EPS))


def clock_rates(
    mass_config: MassConfig,
    clock_array: ClockArray,
) -> NDArray[np.floating]:
    """Full forward model: mass configuration → predicted clock tick rates.

    Returns: (n_clocks,) array of rates in (0, 1].
    """
    distances = compute_distances(
        clock_array.positions,
        mass_config.positions,
        clock_array.track_offset,
    )
    potential = gravitational_potential(distances, mass_config.masses)
    return time_dilation_factor(potential)
