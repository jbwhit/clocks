"""Conditional-prior support shared by API-built point-mass models."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from clocks.physics import (
    WEAK_FIELD_LIMIT,
    _density_potential_batch,
    _point_mass_potential_batch,
)
from clocks.types import ClockArray


def point_mass_support_mask(
    particles: NDArray[np.floating],
    *,
    n_masses: int,
    n_dims: int,
    clock_array: ClockArray,
    position_range: tuple[float, float],
    mass_range: tuple[float, float],
) -> NDArray[np.bool_]:
    """Return membership in bounded, ordered, physically valid support."""
    values = np.asarray(particles, dtype=np.float64)
    expected_columns = n_masses * n_dims + n_masses
    if values.ndim != 2 or values.shape[1] != expected_columns:
        raise ValueError(
            f"particles must have shape (N, {expected_columns}), got {values.shape}"
        )
    if values.shape[0] == 0:
        return np.zeros(0, dtype=bool)

    positions = values[:, : n_masses * n_dims].reshape(-1, n_masses, n_dims)
    masses = values[:, n_masses * n_dims :]
    finite = np.all(np.isfinite(values), axis=1)
    in_position_range = np.all(
        (positions >= position_range[0]) & (positions <= position_range[1]),
        axis=(1, 2),
    )
    in_mass_range = np.all(
        (masses >= mass_range[0]) & (masses <= mass_range[1]), axis=1
    )
    ordered = np.ones(len(values), dtype=bool)
    if n_masses > 1:
        ordered = np.all(np.diff(positions[:, :, 0], axis=1) > 0.0, axis=1)
    _, physical = _point_mass_potential_batch(positions, masses, clock_array)
    return finite & in_position_range & in_mass_range & ordered & physical


def sample_conditioned_prior(
    rng: np.random.Generator,
    n: int,
    draw: Callable[[np.random.Generator, int], NDArray[np.float64]],
    valid: Callable[[NDArray[np.float64]], NDArray[np.bool_]],
    *,
    description: str,
    max_batches: int = 1_000,
) -> NDArray[np.float64]:
    """Rejection-sample exactly ``n`` rows from a conditioned prior."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("conditional prior sample size must be a positive integer")
    if (
        isinstance(max_batches, bool)
        or not isinstance(max_batches, (int, np.integer))
        or max_batches <= 0
    ):
        raise ValueError("max_batches must be a positive integer")
    accepted: list[NDArray[np.float64]] = []
    accepted_count = 0
    draws = 0
    for _ in range(int(max_batches)):
        remaining = n - accepted_count
        batch_size = max(1_024, 2 * remaining)
        candidates = np.asarray(draw(rng, batch_size), dtype=np.float64)
        if candidates.ndim != 2 or candidates.shape[0] != batch_size:
            raise ValueError(
                "conditional-prior draw must return shape "
                f"({batch_size}, D), got {candidates.shape}"
            )
        if not np.all(np.isfinite(candidates)):
            raise ValueError("conditional-prior draw must return finite candidates")
        mask = np.asarray(valid(candidates), dtype=bool)
        if mask.shape != (batch_size,):
            raise ValueError(
                f"conditional-prior validity mask must have shape ({batch_size},)"
            )
        selected = candidates[mask]
        if selected.size:
            accepted.append(selected)
            accepted_count += len(selected)
        draws += batch_size
        if accepted_count >= n:
            return np.concatenate(accepted, axis=0)[:n]

    raise ValueError(
        f"conditional prior for {description} produced only {accepted_count} "
        f"valid samples from {draws} bounded draws under the weak-field policy "
        f"|2*Phi| <= {WEAK_FIELD_LIMIT}; the requested support may have no "
        "usable valid volume"
    )


def make_point_mass_prior_sampler(
    *,
    n_masses: int,
    n_dims: int,
    clock_array: ClockArray,
    position_range: tuple[float, float],
    mass_range: tuple[float, float],
) -> Callable[[np.random.Generator, int], NDArray[np.float64]]:
    """Build a sampler for the ordered, weak-field-conditioned rectangle."""

    def draw(rng: np.random.Generator, n: int) -> NDArray[np.float64]:
        positions = rng.uniform(
            position_range[0], position_range[1], (n, n_masses, n_dims)
        )
        masses = rng.uniform(mass_range[0], mass_range[1], (n, n_masses))
        if n_masses > 1:
            order = np.argsort(positions[:, :, 0], axis=1)
            positions = np.take_along_axis(positions, order[:, :, np.newaxis], axis=1)
            masses = np.take_along_axis(masses, order, axis=1)
        return np.concatenate((positions.reshape(n, n_masses * n_dims), masses), axis=1)

    def sampler(rng: np.random.Generator, n: int) -> NDArray[np.float64]:
        return sample_conditioned_prior(
            rng,
            n,
            draw,
            lambda candidates: point_mass_support_mask(
                candidates,
                n_masses=n_masses,
                n_dims=n_dims,
                clock_array=clock_array,
                position_range=position_range,
                mass_range=mass_range,
            ),
            description=f"K={n_masses} point masses",
        )

    return sampler


def density_support_mask(
    params_batch: NDArray[np.floating],
    *,
    clock_array: ClockArray,
    mu_range: tuple[float, float],
    sigma_range: tuple[float, float],
    amplitude_range: tuple[float, float],
    integration_limit: float = 10.0,
    n_quad: int = 200,
) -> NDArray[np.bool_]:
    """Return bounded and weak-field-valid Gaussian-density candidates."""
    values = np.asarray(params_batch, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(
            f"density particles must have shape (N, 3), got {values.shape}"
        )
    valid = (
        np.all(np.isfinite(values), axis=1)
        & (values[:, 0] >= mu_range[0])
        & (values[:, 0] <= mu_range[1])
        & (values[:, 1] >= sigma_range[0])
        & (values[:, 1] <= sigma_range[1])
        & (values[:, 2] >= amplitude_range[0])
        & (values[:, 2] <= amplitude_range[1])
    )
    physical = np.zeros(len(values), dtype=bool)
    if np.any(valid):
        potential = _density_potential_batch(
            values[valid], clock_array, integration_limit, n_quad
        )
        physical[valid] = (
            np.all(np.isfinite(potential), axis=1)
            & np.all(potential <= 0.0, axis=1)
            & np.all(np.abs(2.0 * potential) <= WEAK_FIELD_LIMIT, axis=1)
        )
    return valid & physical
