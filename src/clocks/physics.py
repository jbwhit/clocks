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
# Relative agreement required between a grid and its refinement before a
# potential is reported at all. It bounds the error of the coarser grid, so
# it is conservative for the extrapolated value actually returned: shipped
# draws estimate up to 1.3e-4 while landing within about 1e-8 of a tight
# reference.
_QUADRATURE_RTOL = 1e-3
# Grid points per sigma below which the profile itself is unresolved and
# refinement compares two equally blind answers.
_MIN_POINTS_PER_SIGMA = 4.0


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


def _asinh_ratio(
    displacement: NDArray[np.float64], track_offset: float
) -> NDArray[np.float64]:
    """``arcsinh(displacement / track_offset)``, without forming the ratio.

    ``displacement / track_offset`` overflows for a small enough offset -- at
    ``h = 1e-310`` a displacement of 10 is already ``inf`` -- and the difference
    of two infinities that follows is NaN. For large arguments
    ``arcsinh(x) -> sign(x) * (log 2 + log|x|)``, which splits the ratio into a
    subtraction of logarithms that cannot overflow.
    """
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ratio = displacement / track_offset
        asymptotic = np.sign(displacement) * (
            np.log(2.0) + np.log(np.abs(displacement)) - np.log(track_offset)
        )
        return np.where(np.isfinite(ratio), np.arcsinh(ratio), asymptotic)


def _smooth_integral_pair(
    params: NDArray[np.float64],
    clock_offset: NDArray[np.float64],
    track_offset: float,
    integration_limit: float,
    intervals: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The singularity-subtracted integral at ``intervals`` and at twice that.

    ``[f(t) - f(c)] / sqrt((t-c)^2 + h^2)`` has a removable singularity at the
    clock, but not a smooth one: away from ``t = c`` it tends to
    ``f'(c) * sign(t - c)``, so it turns a corner there. A grid that steps over
    the corner is stuck at second order however fine it gets, which is what held
    this to about 1e-5. Splitting at ``t = c`` puts the corner on an endpoint and
    leaves each side smooth.

    The split is branch-free: clipping the clock into the interval collapses the
    unused side to zero width when the clock lies outside it, and a zero-width
    piece integrates to zero on its own.

    Both resolutions come from **one** set of ordinates, because the finer grid
    contains the coarser one's nodes -- every second point is the coarse grid.
    The refinement therefore costs nothing beyond the finer evaluation itself.
    """
    sigma = params[:, 1]
    half_width = integration_limit * sigma
    lower, upper = -half_width, half_width
    at_clock = _gaussian_shape(clock_offset[:, np.newaxis], sigma)[:, 0]
    split = np.clip(clock_offset, lower, upper)
    fractions = np.linspace(0.0, 1.0, 2 * intervals + 1)

    coarse = np.zeros(params.shape[0])
    fine = np.zeros(params.shape[0])
    for piece_lo, piece_hi in ((lower, split), (split, upper)):
        grid = (
            piece_lo[:, np.newaxis] + (piece_hi - piece_lo)[:, np.newaxis] * fractions
        )
        # hypot, not the square root of a sum of squares: ``track_offset**2``
        # underflows to zero below about 1e-154, and the split puts a node
        # exactly on the clock, so the distance there would be 0 and the
        # quotient 0/0.
        integrand = (_gaussian_shape(grid, sigma) - at_clock[:, np.newaxis]) / np.hypot(
            grid - clock_offset[:, np.newaxis], track_offset
        )
        fine = fine + np.trapezoid(integrand, grid, axis=1)
        coarse = coarse + np.trapezoid(integrand[:, ::2], grid[:, ::2], axis=1)
    return coarse, fine


def _closed_form_peak(
    params: NDArray[np.float64],
    clock_offset: NDArray[np.float64],
    track_offset: float,
    integration_limit: float,
) -> NDArray[np.float64]:
    """The part of the integral carrying the kernel's peak, done exactly.

    ``f(c) * integral of 1/sqrt((t-c)^2 + h^2)`` is
    ``f(c) * [asinh((b-c)/h) - asinh((a-c)/h)]`` for any ``h``, however small. No
    grid ever samples this term, which is why no grid has to be placed so as to
    catch a peak of width ``track_offset``.
    """
    sigma = params[:, 1]
    half_width = integration_limit * sigma
    at_clock = _gaussian_shape(clock_offset[:, np.newaxis], sigma)[:, 0]
    return at_clock * (
        _asinh_ratio(half_width - clock_offset, track_offset)
        - _asinh_ratio(-half_width - clock_offset, track_offset)
    )


def _quadrature_converged(
    coarse: NDArray[np.float64], fine: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """Whether refining the grid stopped moving the answer.

    ``fine`` doubles the interval count, so ``|fine - coarse|`` estimates the
    error of ``coarse`` and bounds the error of ``fine`` conservatively.

    Refinement is only an honest estimate because the singularity is gone. When
    the kernel's peak was integrated numerically, two resolutions could miss it
    identically and agree on the same wrong answer; with the peak in closed form
    there is no such feature left to miss. Measured against a tight reference,
    the estimate lands within a factor of 1.5 of the true error wherever the
    grid resolves the profile at all.
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
    An entry no grid here could certify comes back as ``nan`` with its flag
    clear, so a caller decides what to do rather than receiving a confident wrong
    number: ``density_support_mask`` rejects such a candidate, and the batch
    forward model raises.

    The kernel's peak is integrated **analytically** rather than sampled:
    :func:`_closed_form_peak` takes ``f(c)`` times the exact integral of the
    kernel, and :func:`_smooth_integral_pair` handles what is left. Earlier
    revisions instead chose between grids according to whether a spacing looked
    fine enough to catch a peak of width ``track_offset``. No fixed grid can be
    relied on to catch it, and the measurement that settles it is that the old
    plain grid was right to 6.5e-12 at ``n_quad = 200`` and wrong by a factor of
    22 at ``n_quad = 400`` for the same state, purely by where a node fell.

    What remains is smooth on the profile's own scale, so **refinement is an
    honest error estimate** -- which it was not while a narrow peak could hide
    between nodes and be missed identically at both resolutions.

    The two resolutions are then combined rather than merely compared. Halving
    the step quarters a second-order error, so ``(4 * fine - coarse) / 3``
    cancels it; the returned value is typically three or four orders better than
    the gap the certificate is drawn from, which is why that certificate is
    loose but never optimistic.

    Refinement still cannot see a grid too coarse to resolve the profile
    *itself*, where both resolutions are equally blind and agree on nothing.
    That is a property of ``integration_limit`` and ``n_quad`` alone -- the
    spacing is ``2 * integration_limit * sigma / (n_quad - 1)`` and the profile's
    scale is ``sigma``, so ``sigma`` cancels -- so it is checked directly. It is
    not a formality: of 39,054 sampled geometries this gate blocks, refinement
    alone would have certified 1,546, every one of them as exactly zero.
    """
    track_offset = clock_array.track_offset
    intervals = n_quad - 1
    # Points per sigma, independent of sigma. Below this the profile is not
    # resolved at either resolution, so refinement would compare two equally
    # blind answers and certify them.
    resolves_profile = (n_quad - 1) >= 2.0 * integration_limit * _MIN_POINTS_PER_SIGMA

    shape = (params_batch.shape[0], len(clock_array.positions))
    potential = np.full(shape, np.nan)
    converged = np.zeros(shape, dtype=bool)
    if not resolves_profile:
        return potential, converged

    with np.errstate(over="ignore", invalid="ignore"):
        for index, clock_position in enumerate(clock_array.positions[:, 0]):
            # Displacement from the profile centre, never an absolute coordinate.
            clock_offset = clock_position - params_batch[:, 0]
            peak = _closed_form_peak(
                params_batch, clock_offset, track_offset, integration_limit
            )
            coarse_smooth, fine_smooth = _smooth_integral_pair(
                params_batch, clock_offset, track_offset, integration_limit, intervals
            )
            extrapolated = (4.0 * fine_smooth - coarse_smooth) / 3.0
            amplitude = params_batch[:, 2]
            coarse = -amplitude * (coarse_smooth + peak)
            fine = -amplitude * (extrapolated + peak)
            certified = _quadrature_converged(coarse, fine)
            potential[:, index] = np.where(certified, fine, np.nan)
            converged[:, index] = certified
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
