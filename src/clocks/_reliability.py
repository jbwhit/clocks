"""Exact local-identifiability mathematics for the echolocation study."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from clocks._scenarios import contrast_matrix
from clocks._validation import finite_float, finite_float_array, real_float_array
from clocks.physics import (
    PhysicsDomainError,
    compute_distances,
    gravitational_potential,
    time_dilation_factor,
)
from clocks.types import ClockArray

PARAMETER_NAMES = ("angular_1", "angular_2", "log_range", "log_mass")


@dataclass(frozen=True)
class IdentifiabilityResult:
    """Immutable local sensitivity diagnostics in dimensionless coordinates."""

    jacobian: NDArray[np.float64]
    scaled_jacobian: NDArray[np.float64]
    fisher_information: NDArray[np.float64]
    singular_values: NDArray[np.float64]
    rank: int
    rank_tolerance: float
    condition_number: float | None
    crlb_std: NDArray[np.float64] | None
    weakest_direction: NDArray[np.float64]
    weakest_mode_loadings: Mapping[str, float]

    parameter_names: ClassVar[tuple[str, str, str, str]] = PARAMETER_NAMES

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "jacobian", finite_float_array("jacobian", self.jacobian, ndim=2)
        )
        object.__setattr__(
            self,
            "scaled_jacobian",
            finite_float_array("scaled_jacobian", self.scaled_jacobian, ndim=2),
        )
        object.__setattr__(
            self,
            "fisher_information",
            finite_float_array("fisher_information", self.fisher_information, ndim=2),
        )
        object.__setattr__(
            self,
            "singular_values",
            finite_float_array("singular_values", self.singular_values, ndim=1),
        )
        if self.crlb_std is not None:
            object.__setattr__(
                self,
                "crlb_std",
                finite_float_array("crlb_std", self.crlb_std, ndim=1),
            )
        object.__setattr__(
            self,
            "weakest_direction",
            finite_float_array("weakest_direction", self.weakest_direction, ndim=1),
        )
        object.__setattr__(
            self,
            "weakest_mode_loadings",
            MappingProxyType(
                {
                    str(name): finite_float(f"loading {name}", value)
                    for name, value in self.weakest_mode_loadings.items()
                }
            ),
        )


def _position_vector(position: object) -> NDArray[np.float64]:
    """Return a finite, nonzero three-dimensional position vector."""
    vector = real_float_array("position", position)
    if vector.shape != (3,):
        raise ValueError(f"position must have shape (3,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("position must contain only finite values")
    if not np.any(vector):
        raise ValueError("position must be nonzero")
    return vector


def _positive_mass(mass: object) -> float:
    if isinstance(mass, (bool, np.bool_)):
        raise ValueError("mass must be a real-valued positive number")
    value = finite_float("mass", mass)
    if value <= 0.0:
        raise ValueError("mass must be positive")
    return value


def tangent_basis(position: NDArray[np.floating]) -> NDArray[np.float64]:
    """Return a deterministic orthonormal basis tangent to a radial direction."""
    vector = _position_vector(position)
    norm = math.hypot(*(float(component) for component in vector))
    if not math.isfinite(norm):
        raise ValueError("position norm must be finite")
    direction = vector / norm

    reference = np.zeros(3, dtype=np.float64)
    reference[int(np.argmin(np.abs(direction)))] = 1.0
    first = reference - np.dot(reference, direction) * direction
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    return np.column_stack((first, second))


def contrast_jacobian(
    position: NDArray[np.floating],
    mass: float,
    clock_array: ClockArray,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return exact position and mass derivatives of orthonormal contrasts."""
    vector = _position_vector(position)
    mass_value = _positive_mass(mass)
    if clock_array.positions.shape[1] != 3:
        raise ValueError(
            "position and clock positions must have matching spatial dimensions"
        )

    distances = compute_distances(
        clock_array.positions, vector.reshape(1, 3), clock_array.track_offset
    )[:, 0]
    masses = np.array([mass_value], dtype=np.float64)
    potential = gravitational_potential(distances[:, np.newaxis], masses)
    rates = time_dilation_factor(potential)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        differences = clock_array.positions - vector
        directions = differences / distances[:, np.newaxis]
        mass_over_rate_distance = (mass_value / distances) / rates
        position_derivatives = (
            -mass_over_rate_distance[:, np.newaxis]
            * directions
            / distances[:, np.newaxis]
        )
        mass_derivatives = -(1.0 / distances) / rates
    if not np.all(np.isfinite(position_derivatives)) or not np.all(
        np.isfinite(mass_derivatives)
    ):
        raise PhysicsDomainError("clock-rate derivatives must be finite")

    contrasts = contrast_matrix(len(clock_array.positions))
    with np.errstate(invalid="ignore", over="ignore"):
        contrast_position = contrasts @ position_derivatives
        contrast_mass = contrasts @ mass_derivatives
    if not np.all(np.isfinite(contrast_position)) or not np.all(
        np.isfinite(contrast_mass)
    ):
        raise PhysicsDomainError("contrast derivatives must be finite")
    return contrast_position, contrast_mass


def _validated_tangent_basis(
    position: NDArray[np.float64], basis: object | None
) -> NDArray[np.float64]:
    if basis is None:
        return tangent_basis(position)
    vectors = real_float_array("basis", basis)
    if vectors.shape != (3, 2):
        raise ValueError(f"basis must have shape (3, 2), got {vectors.shape}")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("basis must contain only finite values")
    direction = position / math.hypot(*(float(value) for value in position))
    if not np.allclose(vectors.T @ vectors, np.eye(2), rtol=1e-12, atol=1e-12):
        raise ValueError("basis must be orthonormal")
    if not np.allclose(direction @ vectors, np.zeros(2), rtol=0.0, atol=1e-12):
        raise ValueError("basis must be perpendicular to position")
    return vectors


def _dimensionless_jacobian(
    position: object,
    mass: object,
    clock_array: ClockArray,
    *,
    basis: object | None = None,
) -> NDArray[np.float64]:
    """Return contrast derivatives in angular/log-range/log-mass coordinates."""
    vector = _position_vector(position)
    mass_value = _positive_mass(mass)
    vectors = _validated_tangent_basis(vector, basis)
    position_jacobian, mass_jacobian = contrast_jacobian(
        vector, mass_value, clock_array
    )
    radius = math.hypot(*(float(component) for component in vector))
    with np.errstate(invalid="ignore", over="ignore"):
        jacobian = np.column_stack(
            (
                position_jacobian @ (radius * vectors[:, 0]),
                position_jacobian @ (radius * vectors[:, 1]),
                position_jacobian @ vector,
                mass_jacobian * mass_value,
            )
        )
    if not np.all(np.isfinite(jacobian)):
        raise PhysicsDomainError("dimensionless contrast derivatives must be finite")
    return jacobian


def _positive_noise(noise_std: object) -> float:
    if isinstance(noise_std, (bool, np.bool_)):
        raise ValueError("noise_std must be a real-valued positive number")
    value = finite_float("noise_std", noise_std)
    if value <= 0.0:
        raise ValueError("noise_std must be positive")
    return value


def _positive_count(n_observations: object) -> int:
    if isinstance(n_observations, (bool, np.bool_)) or not isinstance(
        n_observations, Integral
    ):
        raise ValueError("n_observations must be a positive non-bool integer")
    value = int(n_observations)
    if value <= 0:
        raise ValueError("n_observations must be a positive non-bool integer")
    return value


def local_identifiability(
    position: NDArray[np.floating],
    mass: float,
    clock_array: ClockArray,
    *,
    n_observations: int,
    noise_std: float,
) -> IdentifiabilityResult:
    """Compute exact local identifiability for one exterior point mass."""
    count = _positive_count(n_observations)
    noise = _positive_noise(noise_std)
    jacobian = _dimensionless_jacobian(position, mass, clock_array)
    try:
        scale = math.sqrt(count) / noise
    except OverflowError as error:
        raise ValueError("n_observations is too large") from error
    if not math.isfinite(scale):
        raise ValueError("sqrt(n_observations) / noise_std must be finite")
    with np.errstate(invalid="ignore", over="ignore"):
        scaled_jacobian = scale * jacobian
    if not np.all(np.isfinite(scaled_jacobian)):
        raise PhysicsDomainError("scaled contrast derivatives must be finite")

    full_matrices = scaled_jacobian.shape[0] < scaled_jacobian.shape[1]
    _, singular_values, right_vectors = np.linalg.svd(
        scaled_jacobian, full_matrices=full_matrices
    )
    if singular_values.size < len(PARAMETER_NAMES):
        singular_values = np.append(
            singular_values,
            np.zeros(len(PARAMETER_NAMES) - singular_values.size),
        )
    if not np.all(np.isfinite(singular_values)) or np.any(singular_values < 0.0):
        raise PhysicsDomainError("singular values must be finite and nonnegative")
    largest = float(singular_values[0])
    rank_tolerance = np.finfo(np.float64).eps * max(scaled_jacobian.shape) * largest
    rank = int(np.count_nonzero(singular_values > rank_tolerance))

    condition_number: float | None = None
    crlb_std: NDArray[np.float64] | None = None
    if rank == len(PARAMETER_NAMES):
        condition_number = largest / float(singular_values[-1])
        crlb_std = np.array(
            [
                math.hypot(
                    *(
                        float(right_vectors[row, column]) / float(singular_values[row])
                        for row in range(len(PARAMETER_NAMES))
                    )
                )
                for column in range(len(PARAMETER_NAMES))
            ]
        )
        if not np.all(np.isfinite(crlb_std)):
            raise PhysicsDomainError("CRLB standard deviations must be finite")

    with np.errstate(invalid="ignore", over="ignore"):
        fisher_information = scaled_jacobian.T @ scaled_jacobian
    if not np.all(np.isfinite(fisher_information)):
        raise PhysicsDomainError("Fisher information must be finite")

    weakest_direction = right_vectors[-1]
    squared = np.square(weakest_direction)
    squared /= np.sum(squared)
    weakest_mode_loadings = {
        "angular": float(np.sum(squared[:2])),
        "log_range": float(squared[2]),
        "log_mass": float(squared[3]),
    }
    return IdentifiabilityResult(
        jacobian=jacobian,
        scaled_jacobian=scaled_jacobian,
        fisher_information=fisher_information,
        singular_values=singular_values,
        rank=rank,
        rank_tolerance=float(rank_tolerance),
        condition_number=condition_number,
        crlb_std=crlb_std,
        weakest_direction=weakest_direction,
        weakest_mode_loadings=weakest_mode_loadings,
    )
