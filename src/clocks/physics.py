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
    """Distance from each clock to each mass, with `track_offset` added as
    an orthogonal (perpendicular) component.

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

    In general relativity, dtau/dt = sqrt(1 + 2*Phi/c^2); in simulation
    units (c=1) this simplifies to sqrt(1 + 2*Phi).
    Since Phi is negative, clocks in deeper potential wells tick slower.
    Clamped above eps to avoid sqrt of negative (black hole regime).

    Returns: (n_clocks,) dilation factors in (0, 1].
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


def clock_rates_batch(
    mass_positions: NDArray[np.floating],
    masses: NDArray[np.floating],
    clock_array: ClockArray,
) -> NDArray[np.floating]:
    """Batch forward model for single-mass particles.

    Computes predicted clock rates for many hypothesized point masses at once.

    mass_positions: (n_particles, n_dims) — position of each particle's mass
    masses: (n_particles,) — mass value for each particle
    clock_array: the fixed clock layout

    Returns: (n_particles, n_clocks) array of predicted rates.
    """
    # clock_array.positions: (n_clocks, n_dims)
    # mass_positions: (n_particles, n_dims)
    # diff: (n_clocks, n_particles, n_dims)
    diff = clock_array.positions[:, np.newaxis, :] - mass_positions[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1) + clock_array.track_offset**2
    distances = np.sqrt(dist_sq)  # (n_clocks, n_particles)

    # potential: (n_clocks, n_particles)
    safe_dist = np.maximum(distances, _EPS)
    potential = -masses[np.newaxis, :] / safe_dist

    # dilation: (n_clocks, n_particles)
    argument = 1.0 + 2.0 * potential
    rates = np.sqrt(np.maximum(argument, _EPS))

    return rates.T  # (n_particles, n_clocks)


def clock_rates_batch_multi(
    mass_positions: NDArray[np.floating],
    masses: NDArray[np.floating],
    clock_array: ClockArray,
) -> NDArray[np.floating]:
    """Batch forward model for multi-mass particles.

    Computes predicted clock rates for many hypothesized K-mass configurations.

    mass_positions: (n_particles, K, n_dims) — positions of each particle's masses
    masses: (n_particles, K) — mass values for each particle
    clock_array: the fixed clock layout

    Returns: (n_particles, n_clocks) array of predicted rates.
    """
    # clock_array.positions: (n_clocks, n_dims)
    # mass_positions: (n_particles, K, n_dims)
    # diff: (n_clocks, n_particles, K, n_dims)
    diff = (
        clock_array.positions[:, np.newaxis, np.newaxis, :]
        - mass_positions[np.newaxis, :, :, :]
    )
    dist_sq = np.sum(diff**2, axis=-1) + clock_array.track_offset**2
    distances = np.sqrt(dist_sq)  # (n_clocks, n_particles, K)

    # potential per mass: (n_clocks, n_particles, K)
    safe_dist = np.maximum(distances, _EPS)
    potential_per_mass = -masses[np.newaxis, :, :] / safe_dist

    # total potential: sum over K masses → (n_clocks, n_particles)
    potential = np.sum(potential_per_mass, axis=-1)

    # dilation: (n_clocks, n_particles)
    argument = 1.0 + 2.0 * potential
    rates = np.sqrt(np.maximum(argument, _EPS))

    return rates.T  # (n_particles, n_clocks)


def clock_rates_density_gaussian(
    params: NDArray[np.floating],
    clock_array: ClockArray,
    integration_limit: float = 10.0,
) -> NDArray[np.floating]:
    """Forward model for a Gaussian mass density profile (1D).

    params: [mu, sigma, amplitude] — center, width, and peak density
    Returns: (n_clocks,) array of time dilation factors.
    """
    from scipy.integrate import quad

    mu, sigma, amplitude = params[0], params[1], params[2]
    h = clock_array.track_offset
    positions = clock_array.positions[:, 0]  # 1D only

    lo = mu - integration_limit * sigma
    hi = mu + integration_limit * sigma

    potential = np.empty(len(positions))
    for i, x_c in enumerate(positions):

        def integrand(x: float, xc: float = x_c) -> float:
            density = amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            r = np.sqrt((x - xc) ** 2 + h**2)
            return -density / r

        potential[i], _ = quad(integrand, lo, hi)

    return time_dilation_factor(potential)


def clock_rates_density_gaussian_batch(
    params_batch: NDArray[np.floating],
    clock_array: ClockArray,
    integration_limit: float = 10.0,
    n_quad: int = 200,
) -> NDArray[np.floating]:
    """Batch forward model for Gaussian density (vectorized via trapezoid rule).

    params_batch: (n_particles, 3) — each row is [mu, sigma, amplitude]
    Returns: (n_particles, n_clocks) array of time dilation factors.
    """
    n_particles = params_batch.shape[0]
    mu = params_batch[:, 0]  # (n_particles,)
    sigma = params_batch[:, 1]  # (n_particles,)
    amplitude = params_batch[:, 2]  # (n_particles,)
    h = clock_array.track_offset
    positions = clock_array.positions[:, 0]  # (n_clocks,)

    # Per-particle bounds: mu ± limit*sigma
    lo = mu - integration_limit * sigma  # (n_particles,)
    hi = mu + integration_limit * sigma  # (n_particles,)

    # Map to common [0, 1] grid, then scale per particle
    t = np.linspace(0, 1, n_quad)  # (n_quad,)
    # x_grid: (n_particles, n_quad)
    x_grid = lo[:, np.newaxis] + (hi - lo)[:, np.newaxis] * t[np.newaxis, :]

    # Density at grid points: (n_particles, n_quad)
    z = (x_grid - mu[:, np.newaxis]) / sigma[:, np.newaxis]
    density = amplitude[:, np.newaxis] * np.exp(-0.5 * z**2)

    # For each clock, compute potential via trapezoid
    n_clocks = len(positions)
    potential = np.empty((n_particles, n_clocks))

    for c in range(n_clocks):
        # distance: (n_particles, n_quad)
        r = np.sqrt((x_grid - positions[c]) ** 2 + h**2)
        integrand = -density / r  # (n_particles, n_quad)
        potential[:, c] = np.trapezoid(integrand, x_grid, axis=1)

    # Time dilation for each particle's potential
    argument = 1.0 + 2.0 * potential
    return np.sqrt(np.maximum(argument, _EPS))
