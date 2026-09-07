"""Strict weak-field gravitational forward models.

The project uses simulation units where ``G = c = 1`` and deliberately limits
the pedagogical rate map to ``|2 Phi| <= 0.1``.
"""

from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from clocks._validation import finite_float, real_float_array
from clocks.types import ClockArray, MassConfig

WEAK_FIELD_LIMIT = 0.1
_MAX_ABS_POTENTIAL = WEAK_FIELD_LIMIT / 2.0
# Relative agreement required between a grid and its refinement, and between
# the two independent grids, before a potential is reported at all.
_QUADRATURE_RTOL = 1e-8


class PhysicsDomainError(ValueError):
    """A state lies outside the documented gravitational model."""


def _finite_array(name: str, value: object, *, ndim: int) -> NDArray[np.float64]:
    array = real_float_array(name, value)
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
    array = real_float_array("potential", potential)
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


def _gaussian_shape(
    offset_from_mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the unit-amplitude Gaussian profile at ``offset_from_mu``.

    Callers pass the displacement from the centre rather than an absolute
    coordinate: reconstructing ``x`` and subtracting ``mu`` afterwards quantises
    the profile to the float spacing at ``mu``, which for ``mu ~ 1e15`` is 0.125
    and destroys the resolution the quadrature depends on.
    """
    z = offset_from_mu / sigma[:, np.newaxis]
    return np.exp(-0.5 * z**2)


def _plain_quadrature(
    params: NDArray[np.float64],
    bounds: tuple[NDArray[np.float64], NDArray[np.float64]],
    clock_position: float,
    track_offset: float,
    fractions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Integrate the kernel directly on the profile's own uniform ``x`` grid."""
    mu, sigma, amplitude = params[:, 0], params[:, 1], params[:, 2]
    lo, hi = bounds
    x_grid = lo[:, np.newaxis] + (hi - lo)[:, np.newaxis] * fractions
    density = amplitude[:, np.newaxis] * _gaussian_shape(
        x_grid - mu[:, np.newaxis], sigma
    )
    distance = np.sqrt((x_grid - clock_position) ** 2 + track_offset**2)
    return np.trapezoid(-density / distance, x_grid, axis=1)


def _substituted_quadrature(
    params: NDArray[np.float64],
    u_bounds: tuple[NDArray[np.float64], NDArray[np.float64]],
    clock_position: float,
    track_offset: float,
    fractions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Integrate under ``x = c + h * sinh(u)``, where the kernel cancels.

    The amplitude is applied after the quadrature so that an extreme but finite
    amplitude cannot overflow the summation of the ordinates.
    """
    mu, sigma, amplitude = params[:, 0], params[:, 1], params[:, 2]
    u_lo, u_hi = u_bounds
    u_grid = u_lo[:, np.newaxis] + (u_hi - u_lo)[:, np.newaxis] * fractions
    offset_from_mu = (clock_position - mu)[:, np.newaxis] + track_offset * np.sinh(
        u_grid
    )
    return amplitude * np.trapezoid(
        -_gaussian_shape(offset_from_mu, sigma), u_grid, axis=1
    )


def _quadrature_converged(
    coarse: NDArray[np.float64], fine: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """Whether refining the grid stopped moving the answer.

    ``fine`` doubles the interval count, so ``|fine - coarse|`` estimates the
    error of ``coarse`` and bounds the error of ``fine`` conservatively. This is
    an accuracy criterion, not a resolution one: it asks what the quadrature
    actually did rather than whether a spacing looks small enough.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        difference = np.abs(fine - coarse)
        return (
            np.isfinite(coarse)
            & np.isfinite(fine)
            & (difference <= _QUADRATURE_RTOL * np.abs(fine))
        )


def _density_potential_batch(
    params_batch: NDArray[np.float64],
    clock_array: ClockArray,
    integration_limit: float,
    n_quad: int,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Integrate each Gaussian profile against the clock kernel.

    Returns the potential and, alongside it, whether each entry is trustworthy.
    An entry that no grid here could certify comes back as ``nan`` with its flag
    clear, so a caller must decide what to do about it rather than receive a
    confident wrong number. ``density_support_mask`` rejects such a candidate;
    the batch forward model raises.

    Two uniform grids are needed because no single one resolves every geometry.
    The plain ``x`` grid spanning the profile's own support steps over the kernel
    ``1 / sqrt((x - c)^2 + h^2)`` -- only ``h = track_offset`` wide -- once
    ``track_offset / sigma`` gets small. Substituting ``x = c + h * sinh(u)``
    makes ``sqrt((x - c)^2 + h^2) = h * cosh(u)`` and ``dx = h * cosh(u) du``, so
    the kernel cancels analytically; but that grid spends its points near the
    clock and starves the profile when the clock lies many sigma away.

    Each grid is therefore run at ``n_quad`` and again at ``2 * n_quad - 1``
    points, and believed only where refinement stops moving its answer
    (:func:`_quadrature_converged`). Refinement is a real check here because the
    finer grid contains the coarse nodes and halves their weight, so a feature
    caught by a shared node is weighted differently by the two.

    What refinement alone cannot see is a feature that *neither* grid samples.
    That needs the kernel to be narrower than a step and its peak to lie inside
    the integrated span; there, and where the plain grid failed outright, the
    substituted grid is run too and the two must agree before either is
    believed. The discretizations fail in unrelated ways, so a shared answer is
    evidence and a split one means neither is knowable here. Where no feature
    can hide -- the shipped configuration included -- the second grid is not
    computed at all.

    This replaced a criterion that asked whether a *spacing* looked fine enough
    relative to ``track_offset``. That question has no bearing on the achieved
    error: at ``track_offset = 1e-50`` the plain grid's accuracy is decided by
    where its nodes happen to fall, and it was measured right to 6.5e-12 at
    ``n_quad = 200`` and wrong by a factor of 22 at ``n_quad = 400``. A spacing
    rule certifies the first and the second equally.

    Residual limitation, stated because it is not closed: the corroboration rule
    rests on a sufficient condition for a feature being sampled, not a proof
    that the integrand has no other structure a fixed grid can miss. Two
    resolutions agreeing on the same wrong answer is what no fixed-grid scheme
    can rule out; requiring an independent grid wherever a spike could hide is
    what makes it unlikely rather than merely undetected.
    """
    mu = params_batch[:, 0]
    sigma = params_batch[:, 1]
    half_width = integration_limit * sigma
    lo = mu - half_width
    hi = mu + half_width
    track_offset = clock_array.track_offset
    fractions = np.linspace(0.0, 1.0, n_quad)
    fine_fractions = np.linspace(0.0, 1.0, 2 * n_quad - 1)
    spacing = (hi - lo) / (n_quad - 1)

    shape = (params_batch.shape[0], len(clock_array.positions))
    potential = np.full(shape, np.nan)
    converged = np.zeros(shape, dtype=bool)
    with np.errstate(over="ignore", invalid="ignore"):
        for index, clock_position in enumerate(clock_array.positions[:, 0]):
            plain_coarse = _plain_quadrature(
                params_batch, (lo, hi), clock_position, track_offset, fractions
            )
            plain_fine = _plain_quadrature(
                params_batch, (lo, hi), clock_position, track_offset, fine_fractions
            )
            plain_ok = _quadrature_converged(plain_coarse, plain_fine)
            # A spike can only hide between nodes when the kernel is narrower
            # than a step AND its peak lies inside the integrated span. Wider,
            # or with the clock outside, every feature is sampled and refinement
            # alone is honest; otherwise refinement can agree on two equally
            # wrong answers, so the other grid must corroborate before either is
            # believed.
            undersampled = (
                (spacing >= track_offset)
                & (clock_position > lo - spacing)
                & (clock_position < hi + spacing)
            )

            substituted_fine = np.full(params_batch.shape[0], np.nan)
            substituted_ok = np.zeros(params_batch.shape[0], dtype=bool)
            # The substituted grid is only worth its cost where the plain one
            # failed or cannot be taken on its own word, so the transform is
            # evaluated for those rows alone rather than the whole batch.
            candidate = np.flatnonzero(~plain_ok | undersampled)
            if candidate.size:
                u_lo = np.arcsinh((lo[candidate] - clock_position) / track_offset)
                u_hi = np.arcsinh((hi[candidate] - clock_position) / track_offset)
                usable = np.isfinite(u_lo) & np.isfinite(u_hi) & (u_hi > u_lo)
                if np.any(usable):
                    rows = candidate[usable]
                    sub_coarse = _substituted_quadrature(
                        params_batch[rows],
                        (u_lo[usable], u_hi[usable]),
                        clock_position,
                        track_offset,
                        fractions,
                    )
                    sub_fine = _substituted_quadrature(
                        params_batch[rows],
                        (u_lo[usable], u_hi[usable]),
                        clock_position,
                        track_offset,
                        fine_fractions,
                    )
                    substituted_fine[rows] = sub_fine
                    substituted_ok[rows] = _quadrature_converged(sub_coarse, sub_fine)

            # Two independent discretizations of the same integral: agreement is
            # evidence, and a split verdict means neither is knowable here.
            corroborated = (
                plain_ok
                & substituted_ok
                & (
                    np.abs(plain_fine - substituted_fine)
                    <= _QUADRATURE_RTOL * np.abs(plain_fine)
                )
            )
            trust_plain = plain_ok & (~undersampled | corroborated)
            trust_substituted = substituted_ok & (~plain_ok | corroborated)

            column_ok = trust_plain | trust_substituted
            column = np.where(trust_plain, plain_fine, substituted_fine)
            potential[:, index] = np.where(column_ok, column, np.nan)
            converged[:, index] = column_ok
    return potential, converged


def _density_integration_bounds(
    mu: NDArray[np.float64] | np.float64,
    sigma: NDArray[np.float64] | np.float64,
    integration_limit: float,
    clock_array: ClockArray,
) -> tuple[NDArray[np.float64] | np.float64, NDArray[np.float64] | np.float64]:
    with np.errstate(over="ignore", invalid="ignore"):
        half_width = integration_limit * sigma
        lo = mu - half_width
        hi = mu + half_width
        span = np.asarray(hi) - np.asarray(lo)
        endpoints = np.stack((np.asarray(lo), np.asarray(hi)), axis=-1)
        offsets = endpoints[..., np.newaxis] - clock_array.positions[:, 0]
        distance_squared = offsets**2 + clock_array.track_offset**2
    if (
        not np.all(np.isfinite(lo))
        or not np.all(np.isfinite(hi))
        or not np.all(np.isfinite(span))
        or not np.all(span > 0.0)
        or not np.all(np.isfinite(distance_squared))
    ):
        raise PhysicsDomainError(
            "density integration bounds and geometry must be finite"
        )
    return lo, hi


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
    lo, hi = _density_integration_bounds(mu, sigma, limit, clock_array)
    potential = np.empty(len(clock_array.positions))

    for index, clock_position in enumerate(clock_array.positions[:, 0]):

        def integrand(x: float, xc: float = float(clock_position)) -> float:
            density = amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            distance = np.sqrt((x - xc) ** 2 + clock_array.track_offset**2)
            return float(-density / distance)

        quadrature = quad(integrand, lo, hi, full_output=1)
        if len(quadrature) != 3 or not np.isfinite(quadrature[0]):
            raise PhysicsDomainError("density quadrature failed")
        potential[index] = quadrature[0]
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
    _density_integration_bounds(values[:, 0], values[:, 1], limit, clock_array)
    potential, converged = _density_potential_batch(values, clock_array, limit, count)
    if not np.all(converged):
        raise PhysicsDomainError(
            "density quadrature did not converge; refine n_quad or widen "
            "track_offset relative to sigma"
        )
    if not np.all(np.isfinite(potential)):
        raise PhysicsDomainError("computed density potential must be finite")
    rates = time_dilation_factor(potential.reshape(-1))
    return rates.reshape(potential.shape)
