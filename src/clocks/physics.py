"""Strict weak-field gravitational forward models.

The project uses simulation units where ``G = c = 1`` and deliberately limits
the pedagogical rate map to ``|2 Phi| <= 0.1``.
"""

from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from clocks._validation import finite_float
from clocks.types import ClockArray, MassConfig

WEAK_FIELD_LIMIT = 0.1
_MAX_ABS_POTENTIAL = WEAK_FIELD_LIMIT / 2.0


class PhysicsDomainError(ValueError):
    """A state lies outside the documented gravitational model."""


def _finite_array(name: str, value: object, *, ndim: int) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {array.shape}")
    if 0 in array.shape:
        raise ValueError(f"{name} must be nonempty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _nonnegative_float(name: str, value: object) -> float:
    result = finite_float(name, value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _positive_float(name: str, value: object) -> float:
    result = finite_float(name, value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _validate_spatial_dimensions(
    first_name: str,
    first: NDArray[np.float64],
    second_name: str,
    second: NDArray[np.float64],
) -> None:
    if first.shape[-1] != second.shape[-1]:
        raise ValueError(
            f"{first_name} and {second_name} must have matching spatial dimensions, "
            f"got {first.shape[-1]} and {second.shape[-1]}"
        )


def compute_distances(
    clock_positions: NDArray[np.floating],
    mass_positions: NDArray[np.floating],
    track_offset: float = 0.0,
) -> NDArray[np.float64]:
    """Return the exact clock-to-mass distances with an orthogonal offset."""
    clocks = _finite_array("clock_positions", clock_positions, ndim=2)
    positions = _finite_array("mass_positions", mass_positions, ndim=2)
    offset = _nonnegative_float("track_offset", track_offset)
    _validate_spatial_dimensions("clock_positions", clocks, "mass_positions", positions)
    with np.errstate(over="ignore", invalid="ignore"):
        diff = clocks[:, np.newaxis, :] - positions[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=-1) + np.square(np.float64(offset)))
    if not np.all(np.isfinite(distances)):
        raise PhysicsDomainError("computed distances must be finite")
    return distances


def gravitational_potential(
    distances: NDArray[np.floating],
    masses: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Return ``Phi_i = -sum_j M_j / r_ij`` without numerical floors."""
    distance_array = _finite_array("distances", distances, ndim=2)
    mass_array = _finite_array("masses", masses, ndim=1)
    if distance_array.shape[1] != mass_array.shape[0]:
        raise ValueError(
            "distances must have one column per mass; "
            f"got {distance_array.shape[1]} columns for {mass_array.shape[0]} masses"
        )
    if np.any(distance_array < 0):
        raise ValueError("distances must be nonnegative")
    if np.any(mass_array < 0):
        raise ValueError("masses must be nonnegative")
    singular = (distance_array == 0.0) & (mass_array[np.newaxis, :] > 0.0)
    if np.any(singular):
        raise PhysicsDomainError("positive mass at zero distance is singular")

    terms = np.zeros_like(distance_array)
    with np.errstate(over="ignore", invalid="ignore"):
        np.divide(
            mass_array[np.newaxis, :],
            distance_array,
            out=terms,
            where=distance_array > 0.0,
        )
        potential = -np.sum(terms, axis=1)
    if not np.all(np.isfinite(potential)):
        raise PhysicsDomainError("computed potential must be finite")
    return potential


def _validate_potential(potential: NDArray[np.float64]) -> None:
    if not np.all(np.isfinite(potential)):
        raise PhysicsDomainError("potential must be finite")
    if np.any(potential > 0.0):
        raise PhysicsDomainError("potential must be nonpositive")
    if np.any(np.abs(potential) > _MAX_ABS_POTENTIAL):
        raise PhysicsDomainError(
            f"weak-field policy requires |2*Phi| <= {WEAK_FIELD_LIMIT}"
        )


def time_dilation_factor(
    potential: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Return exactly ``sqrt(1 + 2 Phi)`` inside the weak-field domain."""
    array = np.asarray(potential, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"potential must be 1-D, got shape {array.shape}")
    if array.size == 0:
        raise ValueError("potential must be nonempty")
    _validate_potential(array)
    return np.sqrt(1.0 + 2.0 * array)


def clock_rates(
    mass_config: MassConfig,
    clock_array: ClockArray,
) -> NDArray[np.float64]:
    """Evaluate one point-mass configuration against a clock array."""
    distances = compute_distances(
        clock_array.positions,
        mass_config.positions,
        clock_array.track_offset,
    )
    return time_dilation_factor(gravitational_potential(distances, mass_config.masses))


def _validate_point_mass_batch_shapes(
    mass_positions: NDArray[np.float64],
    masses: NDArray[np.float64],
    clock_array: ClockArray,
) -> None:
    if mass_positions.shape[:2] != masses.shape:
        raise ValueError(
            "mass_positions and masses must have matching particle and mass "
            f"dimensions, got {mass_positions.shape[:2]} and {masses.shape}"
        )
    _validate_spatial_dimensions(
        "mass_positions", mass_positions, "clock positions", clock_array.positions
    )


def _point_mass_potential_batch(
    mass_positions: NDArray[np.floating],
    masses: NDArray[np.floating],
    clock_array: ClockArray,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Return raw candidate potentials and a no-warning physical-validity mask.

    Unlike public forward functions, candidate invalidity is normal control
    flow here. Shape mismatches still raise because they are programming
    errors; non-finite, negative, singular, and strong-field rows are marked
    invalid.
    """
    positions = np.asarray(mass_positions, dtype=np.float64)
    mass_array = np.asarray(masses, dtype=np.float64)
    if positions.ndim != 3:
        raise ValueError(f"mass_positions must be 3-D, got shape {positions.shape}")
    if mass_array.ndim != 2:
        raise ValueError(f"masses must be 2-D, got shape {mass_array.shape}")
    if 0 in positions.shape or 0 in mass_array.shape:
        raise ValueError("mass_positions and masses must be nonempty")
    _validate_point_mass_batch_shapes(positions, mass_array, clock_array)

    finite_rows = np.all(np.isfinite(positions), axis=(1, 2)) & np.all(
        np.isfinite(mass_array), axis=1
    )
    nonnegative_rows = np.all(mass_array >= 0.0, axis=1)
    clean_positions = np.where(np.isfinite(positions), positions, 0.0)
    clean_masses = np.where(
        np.isfinite(mass_array) & (mass_array >= 0.0), mass_array, 0.0
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        diff = (
            clock_array.positions[np.newaxis, :, np.newaxis, :]
            - clean_positions[:, np.newaxis, :, :]
        )
        distance = np.sqrt(
            np.sum(diff**2, axis=-1) + np.square(np.float64(clock_array.track_offset))
        )
        singular = (distance == 0.0) & (clean_masses[:, np.newaxis, :] > 0.0)
        terms = np.zeros_like(distance)
        np.divide(
            clean_masses[:, np.newaxis, :],
            distance,
            out=terms,
            where=distance > 0.0,
        )
        potential = -np.sum(terms, axis=2)

    computed_finite_rows = np.all(np.isfinite(distance), axis=(1, 2)) & np.all(
        np.isfinite(potential), axis=1
    )
    potential[~(finite_rows & computed_finite_rows)] = np.nan
    valid = (
        finite_rows
        & nonnegative_rows
        & computed_finite_rows
        & ~np.any(singular, axis=(1, 2))
        & np.all(np.isfinite(potential), axis=1)
        & np.all(potential <= 0.0, axis=1)
        & np.all(np.abs(potential) <= _MAX_ABS_POTENTIAL, axis=1)
    )
    return potential, valid


def _strict_rates_from_point_mass_batch(
    positions: NDArray[np.float64],
    masses: NDArray[np.float64],
    clock_array: ClockArray,
) -> NDArray[np.float64]:
    potential, valid = _point_mass_potential_batch(positions, masses, clock_array)
    if not np.all(valid):
        raise PhysicsDomainError(
            "point-mass state is singular or violates the weak-field policy"
        )
    return np.sqrt(1.0 + 2.0 * potential)


def clock_rates_batch(
    mass_positions: NDArray[np.floating],
    masses: NDArray[np.floating],
    clock_array: ClockArray,
) -> NDArray[np.float64]:
    """Evaluate a batch of single-mass candidates with exact shape checks."""
    positions = _finite_array("mass_positions", mass_positions, ndim=2)
    mass_array = _finite_array("masses", masses, ndim=1)
    if positions.shape[0] != mass_array.shape[0]:
        raise ValueError("mass_positions and masses must contain the same number")
    if positions.shape[1] != clock_array.positions.shape[1]:
        raise ValueError(
            "mass_positions and clock positions must have matching spatial dimensions"
        )
    if np.any(mass_array < 0):
        raise ValueError("masses must be nonnegative")
    return _strict_rates_from_point_mass_batch(
        positions[:, np.newaxis, :], mass_array[:, np.newaxis], clock_array
    )


def clock_rates_batch_multi(
    mass_positions: NDArray[np.floating],
    masses: NDArray[np.floating],
    clock_array: ClockArray,
) -> NDArray[np.float64]:
    """Evaluate a batch of multi-mass candidates with exact shape checks."""
    positions = _finite_array("mass_positions", mass_positions, ndim=3)
    mass_array = _finite_array("masses", masses, ndim=2)
    _validate_point_mass_batch_shapes(positions, mass_array, clock_array)
    if np.any(mass_array < 0):
        raise ValueError("masses must be nonnegative")
    return _strict_rates_from_point_mass_batch(positions, mass_array, clock_array)


def _validate_density_context(
    clock_array: ClockArray, integration_limit: object
) -> float:
    if clock_array.positions.shape[1] != 1:
        raise ValueError("Gaussian density model requires one spatial dimension")
    if clock_array.track_offset <= 0.0:
        raise ValueError("density track_offset must be positive")
    return _positive_float("integration_limit", integration_limit)


def _validate_density_params(value: object, *, batch: bool) -> NDArray[np.float64]:
    name = "params_batch" if batch else "params"
    ndim = 2 if batch else 1
    params = _finite_array(name, value, ndim=ndim)
    if params.shape[-1] != 3:
        raise ValueError(f"{name} must contain exactly three parameters")
    if np.any(params[..., 1] <= 0.0):
        raise ValueError("density sigma must be positive")
    if np.any(params[..., 2] < 0.0):
        raise ValueError("density amplitude must be nonnegative")
    return params


def _density_potential_batch(
    params_batch: NDArray[np.float64],
    clock_array: ClockArray,
    integration_limit: float,
    n_quad: int,
) -> NDArray[np.float64]:
    mu = params_batch[:, 0]
    sigma = params_batch[:, 1]
    amplitude = params_batch[:, 2]
    lo = mu - integration_limit * sigma
    hi = mu + integration_limit * sigma
    t = np.linspace(0.0, 1.0, n_quad)
    x_grid = lo[:, np.newaxis] + (hi - lo)[:, np.newaxis] * t
    z = (x_grid - mu[:, np.newaxis]) / sigma[:, np.newaxis]
    density = amplitude[:, np.newaxis] * np.exp(-0.5 * z**2)

    potential = np.empty((params_batch.shape[0], len(clock_array.positions)))
    for index, clock_position in enumerate(clock_array.positions[:, 0]):
        distance = np.sqrt((x_grid - clock_position) ** 2 + clock_array.track_offset**2)
        potential[:, index] = np.trapezoid(-density / distance, x_grid, axis=1)
    return potential


def clock_rates_density_gaussian(
    params: NDArray[np.floating],
    clock_array: ClockArray,
    integration_limit: float = 10.0,
) -> NDArray[np.float64]:
    """Evaluate a one-dimensional Gaussian line-density profile."""
    from scipy.integrate import quad

    values = _validate_density_params(params, batch=False)
    limit = _validate_density_context(clock_array, integration_limit)
    mu, sigma, amplitude = values
    lo = mu - limit * sigma
    hi = mu + limit * sigma
    potential = np.empty(len(clock_array.positions))

    for index, clock_position in enumerate(clock_array.positions[:, 0]):

        def integrand(x: float, xc: float = float(clock_position)) -> float:
            density = amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            distance = np.sqrt((x - xc) ** 2 + clock_array.track_offset**2)
            return float(-density / distance)

        potential[index], _ = quad(integrand, lo, hi)
    return time_dilation_factor(potential)


def clock_rates_density_gaussian_batch(
    params_batch: NDArray[np.floating],
    clock_array: ClockArray,
    integration_limit: float = 10.0,
    n_quad: int = 200,
) -> NDArray[np.float64]:
    """Evaluate Gaussian density candidates using vectorized quadrature."""
    values = _validate_density_params(params_batch, batch=True)
    limit = _validate_density_context(clock_array, integration_limit)
    if isinstance(n_quad, (bool, np.bool_)) or not isinstance(n_quad, Integral):
        raise ValueError("n_quad must be an integer >= 2")
    count = int(n_quad)
    if count < 2:
        raise ValueError("n_quad must be an integer >= 2")
    potential = _density_potential_batch(values, clock_array, limit, count)
    rates = time_dilation_factor(potential.reshape(-1))
    return rates.reshape(potential.shape)
