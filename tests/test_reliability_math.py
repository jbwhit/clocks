"""Exact mathematical checks for the echolocation reliability diagnostics."""

import math
from collections.abc import Callable
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from numpy.typing import NDArray

from clocks._reliability import (
    IdentifiabilityResult,
    _dimensionless_jacobian,
    _stable_fisher_information,
    contrast_jacobian,
    local_identifiability,
    tangent_basis,
)
from clocks._scenarios import build_head_lattice, contrast_matrix
from clocks.physics import PhysicsDomainError, clock_rates
from clocks.types import ClockArray, MassConfig


def _contrasts(position: NDArray[np.float64], mass: float) -> NDArray[np.float64]:
    clocks = build_head_lattice()
    rates = clock_rates(MassConfig(position.reshape(1, 3), np.array([mass])), clocks)
    return contrast_matrix(len(clocks.positions)) @ rates


def _central_difference(
    function: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    point: NDArray[np.float64],
    step: float,
) -> NDArray[np.float64]:
    columns = []
    for index in range(point.size):
        delta = np.zeros_like(point)
        delta[index] = step
        columns.append((function(point + delta) - function(point - delta)) / (2 * step))
    return np.column_stack(columns)


def _independent_dimensionless_jacobian(
    position: NDArray[np.float64], mass: float, clocks: ClockArray
) -> NDArray[np.float64]:
    position_jacobian, mass_jacobian = contrast_jacobian(position, mass, clocks)
    basis = tangent_basis(position)
    radius = np.linalg.norm(position)
    return np.column_stack(
        (
            position_jacobian @ (radius * basis[:, 0]),
            position_jacobian @ (radius * basis[:, 1]),
            position_jacobian @ position,
            mass_jacobian * mass,
        )
    )


def _independent_stable_contrast_jacobian(
    position: NDArray[np.float64], mass: float, clocks: ClockArray
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Scalar-loop oracle: contrasts of the mean-removed clock-rate derivatives.

    Contrast rows sum to zero, so removing the across-clock mean changes nothing
    in real arithmetic. The mean is taken with ``math.fsum`` so this oracle does
    not simply repeat production's summation order.
    """
    position_rows = []
    mass_rows = []
    for clock in clocks.positions:
        difference = clock - position
        distance = math.hypot(
            *(float(component) for component in difference), clocks.track_offset
        )
        rate = math.sqrt(1.0 - 2.0 * mass / distance)
        position_scale = -((mass / distance) / rate) / distance
        position_rows.append(position_scale * (difference / distance))
        mass_rows.append(-(1.0 / distance) / rate)
    count = len(clocks.positions)
    position_array = np.array(position_rows)
    mass_array = np.array(mass_rows)
    position_mean = np.array(
        [
            math.fsum(float(value) for value in position_array[:, axis]) / count
            for axis in range(position_array.shape[1])
        ]
    )
    mass_mean = math.fsum(float(value) for value in mass_array) / count
    contrasts = contrast_matrix(count)
    return (
        contrasts @ (position_array - position_mean),
        contrasts @ (mass_array - mass_mean),
    )


def _independent_stable_gram(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    scale = float(np.max(np.abs(matrix)))
    if scale == 0.0:
        return np.zeros((matrix.shape[1], matrix.shape[1]))
    normalized = matrix / scale
    gram = np.empty((matrix.shape[1], matrix.shape[1]))
    for first in range(matrix.shape[1]):
        for second in range(matrix.shape[1]):
            normalized_entry = math.fsum(
                float(row[first]) * float(row[second]) for row in normalized
            )
            gram[first, second] = (normalized_entry * scale) * scale
    return gram


@pytest.mark.parametrize(
    "position",
    [
        np.array([2.0, 3.0, 6.0]) / 7.0 * (2.4 * np.sqrt(3.0)),
        np.array([4.2, -3.1, 5.7]),
        np.array([-5.3, 3.2, 4.8]),
    ],
)
def test_tangent_basis_is_deterministic_orthonormal_and_perpendicular(
    position: NDArray[np.float64],
) -> None:
    first = tangent_basis(position)
    second = tangent_basis(position.copy())
    direction = position / np.linalg.norm(position)

    assert first.shape == (3, 2)
    assert_allclose(first, second, rtol=0.0, atol=0.0)
    assert_allclose(first.T @ first, np.eye(2), rtol=0.0, atol=2e-15)
    assert_allclose(direction @ first, np.zeros(2), rtol=0.0, atol=2e-15)


@pytest.mark.parametrize(
    "position, message",
    [
        (np.zeros(3), "nonzero"),
        (np.array([1.0, 2.0]), "shape"),
        (np.ones((1, 3)), "shape"),
        (np.array([1.0, np.nan, 3.0]), "finite"),
        (np.array([1.0, np.inf, 3.0]), "finite"),
        (np.array([1.0 + 0.0j, 2.0, 3.0]), "real-valued"),
    ],
)
def test_tangent_basis_rejects_invalid_positions_without_warnings(
    position: NDArray[np.generic], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        tangent_basis(position)


@pytest.mark.parametrize(
    "position, mass",
    [
        (np.array([3.0, 4.0, 7.0]), 0.080),
        (np.array([5.5, -3.0, 4.0]), 0.045),
        (np.array([-4.5, 5.0, 6.5]), 0.120),
    ],
)
def test_exact_contrast_jacobian_matches_central_finite_differences(
    position: NDArray[np.float64], mass: float
) -> None:
    position_jacobian, mass_jacobian = contrast_jacobian(
        position, mass, build_head_lattice()
    )
    numerical_position = _central_difference(
        lambda trial: _contrasts(trial, mass), position, 2e-5
    )
    numerical_mass = _central_difference(
        lambda trial: _contrasts(position, float(trial[0])),
        np.array([mass]),
        2e-5,
    )[:, 0]

    assert position_jacobian.shape == (26, 3)
    assert mass_jacobian.shape == (26,)
    assert_allclose(position_jacobian, numerical_position, rtol=2e-6, atol=2e-11)
    assert_allclose(mass_jacobian, numerical_mass, rtol=2e-7, atol=2e-10)


@pytest.mark.parametrize(
    "mass, message",
    [
        (0.0, "positive"),
        (-0.1, "positive"),
        (True, "positive"),
        (np.nan, "finite"),
        (np.inf, "finite"),
        (0.1 + 0.0j, "real-valued"),
    ],
)
def test_contrast_jacobian_rejects_invalid_mass_without_warnings(
    mass: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        contrast_jacobian(np.array([3.0, 4.0, 7.0]), mass, build_head_lattice())


@pytest.mark.parametrize(
    "position, message",
    [
        (np.zeros(3), "nonzero"),
        (np.array([1.0, 2.0]), "shape"),
        (np.array([1.0, 2.0, np.nan]), "finite"),
        (np.array([1.0 + 0.0j, 2.0, 3.0]), "real-valued"),
    ],
)
def test_contrast_jacobian_rejects_invalid_positions_without_warnings(
    position: NDArray[np.generic], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        contrast_jacobian(position, 0.08, build_head_lattice())


def test_contrast_jacobian_honors_a_nonzero_track_offset() -> None:
    """The offset must reach the derivatives, not just the distances.

    Every other geometry test uses the 3-D head, whose ``track_offset`` is zero,
    so dropping the offset from the Jacobian's distance computation would be
    invisible to them while silently changing 1-D and 2-D sensitivities.
    """
    clocks = ClockArray(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.5, -0.4],
                [-0.7, 1.1, 0.3],
                [0.4, -1.2, 0.9],
            ]
        ),
        track_offset=2.5,
    )
    position = np.array([3.0, 4.0, 7.0])
    mass = 0.05

    def contrasts(trial_position: NDArray[np.float64]) -> NDArray[np.float64]:
        rates = clock_rates(
            MassConfig(trial_position.reshape(1, 3), np.array([mass])), clocks
        )
        return contrast_matrix(len(clocks.positions)) @ rates

    position_jacobian, _ = contrast_jacobian(position, mass, clocks)
    numerical = _central_difference(contrasts, position, 2e-5)
    zero_offset_jacobian, _ = contrast_jacobian(
        position, mass, ClockArray(clocks.positions)
    )

    assert_allclose(position_jacobian, numerical, rtol=2e-6, atol=2e-12)
    # The offset genuinely changes the answer, so the agreement above is not
    # satisfiable by ignoring it.
    assert not np.allclose(position_jacobian, zero_offset_jacobian, rtol=1e-3)


def test_contrast_jacobian_rejects_wrong_clock_dimension() -> None:
    clocks = ClockArray(np.array([[0.0, 0.0], [1.0, 1.0]]))

    with pytest.raises(ValueError, match="spatial dimensions"):
        contrast_jacobian(np.array([3.0, 4.0, 7.0]), 0.08, clocks)


def test_contrast_jacobian_does_not_clamp_singular_or_strong_field_inputs() -> None:
    clocks = build_head_lattice()

    with pytest.raises(PhysicsDomainError, match="singular"):
        contrast_jacobian(np.array([1.0, 1.0, 1.0]), 0.08, clocks)
    with pytest.raises(PhysicsDomainError, match="weak-field"):
        contrast_jacobian(np.array([1.01, 1.0, 1.0]), 0.08, clocks)


@pytest.mark.parametrize(
    "scale, mass",
    [(1e103, 0.08), (1e-109, 1e-112)],
    ids=("large-distance", "small-distance"),
)
def test_contrast_jacobian_is_stable_across_wide_distance_scales(
    scale: float, mass: float
) -> None:
    """Agreement with a scalar-loop oracle across 200 decades of coordinates.

    These are wide scales, not the float64 limits: ``compute_distances`` squares
    coordinates, so it overflows above ``sqrt(DBL_MAX)`` (~1.34e154) and
    underflows a nonzero distance to zero below ~2.2e-162. Both are far outside
    any declared scenario and are a property of the shared forward model, not of
    this Jacobian.
    """
    position = scale * np.array([1.5, -1.2, 0.9])
    clocks = ClockArray(
        scale
        * np.array(
            [
                [0.0, 0.0, 0.0],
                [0.3, -0.2, 0.1],
                [-0.4, 0.1, 0.2],
                [0.2, 0.4, -0.3],
                [-0.1, -0.3, 0.4],
            ]
        )
    )
    expected_position, expected_mass = _independent_stable_contrast_jacobian(
        position, mass, clocks
    )

    assert np.all(np.isfinite(expected_position))
    assert np.any(expected_position != 0.0)
    assert np.all(np.isfinite(expected_mass))
    actual_position, actual_mass = contrast_jacobian(position, mass, clocks)
    # Individual contrast entries can cancel to a small fraction of the matrix
    # scale, and neither code path resolves those to full relative precision.
    # Accuracy is asserted against the magnitude that is actually present, so
    # this cannot pass merely because both paths round identically.
    assert_allclose(
        actual_position,
        expected_position,
        rtol=2e-14,
        atol=1e-15 * float(np.max(np.abs(expected_position))),
    )
    assert_allclose(
        actual_mass,
        expected_mass,
        rtol=2e-14,
        atol=1e-15 * float(np.max(np.abs(expected_mass))),
    )


def test_dimensionless_jacobian_has_exact_parameter_order_and_shape() -> None:
    position = np.array([3.0, 4.0, 7.0])
    mass = 0.08
    clocks = build_head_lattice()
    basis = tangent_basis(position)
    position_jacobian, mass_jacobian = contrast_jacobian(position, mass, clocks)

    jacobian, _ = _dimensionless_jacobian(position, mass, clocks, basis=basis)
    radius = np.linalg.norm(position)
    expected = np.column_stack(
        (
            position_jacobian @ (radius * basis[:, 0]),
            position_jacobian @ (radius * basis[:, 1]),
            position_jacobian @ position,
            mass_jacobian * mass,
        )
    )

    assert jacobian.shape == (26, 4)
    assert_allclose(jacobian, expected, rtol=2e-15, atol=2e-17)


def test_singular_values_are_invariant_under_tangent_plane_rotation() -> None:
    position = np.array([4.2, -3.1, 5.7])
    clocks = build_head_lattice()
    basis = tangent_basis(position)
    angle = 0.731
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )

    original, _ = _dimensionless_jacobian(position, 0.08, clocks, basis=basis)
    rotated, _ = _dimensionless_jacobian(position, 0.08, clocks, basis=basis @ rotation)

    assert_allclose(
        np.linalg.svd(original, compute_uv=False),
        np.linalg.svd(rotated, compute_uv=False),
        rtol=2e-14,
        atol=1e-16,
    )


def test_local_identifiability_reports_exact_shapes_and_parameter_order() -> None:
    result = local_identifiability(
        np.array([3.0, 4.0, 7.0]),
        0.08,
        build_head_lattice(),
        n_observations=80,
        noise_std=0.001,
    )

    assert isinstance(result, IdentifiabilityResult)
    assert result.parameter_names == (
        "angular_1",
        "angular_2",
        "log_range",
        "log_mass",
    )
    assert result.jacobian.shape == (26, 4)
    assert result.scaled_jacobian.shape == (26, 4)
    assert result.fisher_information.shape == (4, 4)
    assert result.singular_values.shape == (4,)
    assert result.weakest_direction.shape == (4,)


def test_local_identifiability_scales_with_count_and_noise() -> None:
    args = (np.array([3.0, 4.0, 7.0]), 0.08, build_head_lattice())
    baseline = local_identifiability(*args, n_observations=20, noise_std=0.002)
    four_times_count = local_identifiability(*args, n_observations=80, noise_std=0.002)
    half_noise = local_identifiability(*args, n_observations=20, noise_std=0.001)

    assert_allclose(four_times_count.scaled_jacobian, 2.0 * baseline.scaled_jacobian)
    assert_allclose(four_times_count.singular_values, 2.0 * baseline.singular_values)
    assert_allclose(half_noise.scaled_jacobian, 2.0 * baseline.scaled_jacobian)
    assert_allclose(half_noise.singular_values, 2.0 * baseline.singular_values)


def test_scaled_jacobian_and_singular_values_match_independent_oracles() -> None:
    position = np.array([5.5, -3.0, 4.0])
    mass = 0.045
    clocks = build_head_lattice()
    n_observations = 91
    noise_std = 0.002
    result = local_identifiability(
        position,
        mass,
        clocks,
        n_observations=n_observations,
        noise_std=noise_std,
    )
    dimensionless = _independent_dimensionless_jacobian(position, mass, clocks)
    expected_scaled = np.sqrt(n_observations) / noise_std * dimensionless
    # Take the singular values from the independently built matrix, not from the
    # result's own array: the latter would make this assertion svd(X) == svd(X).
    expected_singular_values = np.linalg.svd(expected_scaled, compute_uv=False)

    assert_allclose(result.scaled_jacobian, expected_scaled, rtol=2e-15, atol=2e-15)
    assert_allclose(
        result.singular_values,
        expected_singular_values,
        rtol=2e-15,
        atol=2e-15,
    )


def _independent_projection_noise_floor(
    position: NDArray[np.float64],
    mass: float,
    clocks: ClockArray,
    *,
    n_observations: int,
    noise_std: float,
) -> float:
    """Whitened size of a contrast value that is pure floating-point residue."""
    common_mode = 0.0
    for clock in clocks.positions:
        difference = clock - position
        distance = math.hypot(
            *(float(component) for component in difference), clocks.track_offset
        )
        rate = math.sqrt(1.0 - 2.0 * mass / distance)
        position_scale = -((mass / distance) / rate) / distance
        common_mode = max(
            common_mode,
            max(
                abs(float(value)) for value in position_scale * (difference / distance)
            ),
            abs(-(1.0 / distance) / rate),
        )
    radius = math.hypot(*(float(component) for component in position))
    return (
        (math.sqrt(n_observations) / noise_std)
        * float(np.finfo(np.float64).eps)
        * len(clocks.positions)
        * common_mode
        * max(radius, mass)
    )


def test_rank_tolerance_is_the_svd_tolerance_floored_by_projection_residue() -> None:
    position = np.array([3.0, 4.0, 7.0])
    clocks = build_head_lattice()
    result = local_identifiability(
        position, 0.08, clocks, n_observations=80, noise_std=0.001
    )
    relative = float(
        np.finfo(np.float64).eps
        * max(result.scaled_jacobian.shape)
        * result.singular_values[0]
    )
    floor = _independent_projection_noise_floor(
        position, 0.08, clocks, n_observations=80, noise_std=0.001
    )

    assert result.rank_tolerance == pytest.approx(max(relative, floor), rel=2e-15)
    assert result.rank == int(
        np.count_nonzero(result.singular_values > result.rank_tolerance)
    )
    # Both branches are far below the physical signal, so the head is full rank
    # either way: the floor exists for degenerate heads, not for this one.
    assert result.rank == 4
    assert result.rank_tolerance < 1e-6 * float(result.singular_values[-1])


def test_degenerate_head_reports_no_rank_instead_of_projection_residue() -> None:
    """Coincident clocks have an exactly zero contrast Jacobian.

    Every contrast of identical clock rates vanishes, so there is no information
    at all. Reporting rank 4 with a benign condition number would be reporting
    the structure of floating-point residue.
    """
    coincident = ClockArray(np.tile(np.array([[0.1, 0.2, -0.3]]), (5, 1)))
    result = local_identifiability(
        np.array([3.0, 4.0, 7.0]),
        0.08,
        coincident,
        n_observations=80,
        noise_std=0.001,
    )

    assert result.rank == 0
    assert result.condition_number is None
    assert result.crlb_std is None
    assert result.weakest_direction is None
    assert result.weakest_mode_loadings is None
    assert float(np.max(np.abs(result.scaled_jacobian))) < result.rank_tolerance


def test_singular_values_are_descending_finite_and_nonnegative() -> None:
    result = local_identifiability(
        np.array([-4.5, 5.0, 6.5]),
        0.12,
        build_head_lattice(),
        n_observations=37,
        noise_std=0.003,
    )

    assert np.all(np.isfinite(result.singular_values))
    assert np.all(result.singular_values >= 0.0)
    assert np.all(np.diff(result.singular_values) <= 0.0)


def test_condition_number_and_crlb_are_only_available_at_full_rank() -> None:
    position = np.array([3.0, 4.0, 7.0])
    full_rank = local_identifiability(
        position,
        0.08,
        build_head_lattice(),
        n_observations=80,
        noise_std=0.001,
    )
    deficient = local_identifiability(
        position,
        0.08,
        ClockArray(build_head_lattice().positions[:4]),
        n_observations=80,
        noise_std=0.001,
    )

    assert full_rank.rank == 4
    assert full_rank.condition_number is not None
    assert np.isfinite(full_rank.condition_number)
    assert full_rank.condition_number >= 1.0
    expected_scaled = (
        np.sqrt(80)
        / 0.001
        * _independent_dimensionless_jacobian(position, 0.08, build_head_lattice())
    )
    expected_singular_values = np.linalg.svd(expected_scaled, compute_uv=False)
    assert full_rank.condition_number == pytest.approx(
        expected_singular_values[0] / expected_singular_values[-1], rel=2e-15
    )
    assert full_rank.crlb_std is not None
    expected_covariance = np.linalg.inv(full_rank.fisher_information)
    assert_allclose(
        full_rank.crlb_std,
        np.sqrt(np.diag(expected_covariance)),
        rtol=2e-5,
        atol=0.0,
    )

    assert deficient.rank < 4
    assert deficient.condition_number is None
    assert deficient.crlb_std is None


def test_crlb_std_is_stable_for_tiny_valid_singular_values() -> None:
    position = np.array([3.0, 4.0, 7.0])
    mass = 0.08
    clocks = build_head_lattice()
    n_observations = 80
    noise_std = 1e160
    expected_scaled = (
        np.sqrt(n_observations)
        / noise_std
        * _independent_dimensionless_jacobian(position, mass, clocks)
    )
    _, expected_singular_values, expected_right_vectors = np.linalg.svd(
        expected_scaled, full_matrices=False
    )
    expected_std = np.array(
        [
            math.hypot(
                *(
                    float(expected_right_vectors[row, column])
                    / float(expected_singular_values[row])
                    for row in range(4)
                )
            )
            for column in range(4)
        ]
    )
    expected_fisher = _independent_stable_gram(expected_scaled)
    direct_fisher = expected_scaled.T @ expected_scaled

    assert np.all(np.isfinite(expected_std))
    assert np.all(expected_std > 1e160)
    assert not np.array_equal(direct_fisher, expected_fisher)
    result = local_identifiability(
        position,
        mass,
        clocks,
        n_observations=n_observations,
        noise_std=noise_std,
    )
    assert result.crlb_std is not None
    assert_allclose(result.crlb_std, expected_std, rtol=2e-14, atol=0.0)
    assert_array_equal(result.fisher_information, expected_fisher)
    assert_array_equal(result.fisher_information, result.fisher_information.T)

    # Raw eigensolvers can emit a negative subnormal quantum. Normalize first
    # so the tolerance tracks floating-point operations, not absolute scale.
    fisher_scale = float(np.max(np.abs(result.fisher_information)))
    normalized_fisher = result.fisher_information / fisher_scale
    minimum_eigenvalue = float(np.linalg.eigvalsh(normalized_fisher)[0])
    operation_scale = max(1.0, float(np.max(np.sum(np.abs(normalized_fisher), axis=1))))
    psd_tolerance = (
        np.finfo(np.float64).eps * max(normalized_fisher.shape) * operation_scale
    )
    assert minimum_eigenvalue >= -psd_tolerance


def test_symmetrization_is_exact_at_both_ends_of_the_float64_range() -> None:
    """Making the Gram symmetric must not itself perturb the Gram.

    Both arithmetic routes to symmetry fail somewhere this function has to
    work. ``(F + F.T) / 2`` overflows to infinity for entries just under the
    float64 maximum -- the eigensolver then reports 'Eigenvalues did not
    converge' rather than anything a caller could act on. ``0.5*F + 0.5*F.T``
    survives that but rounds away real bits of a subnormal, and subnormal Gram
    matrices are precisely what the surrounding rescaling exists to preserve.
    Mirroring one triangle is exact at both ends.
    """
    near_maximum = _stable_fisher_information(np.full((1, 4), 1e154))

    assert np.all(np.isfinite(near_maximum))
    assert_array_equal(near_maximum, np.full((4, 4), 1e308))
    assert_array_equal(near_maximum, near_maximum.T)

    scaled_jacobian = np.random.default_rng(3).normal(size=(26, 4)) * 1e-161
    subnormal = _stable_fisher_information(scaled_jacobian)
    scale = float(np.max(np.abs(scaled_jacobian)))
    normalized = scaled_jacobian / scale
    raw = (normalized.T @ normalized * scale) * scale

    assert np.min(np.abs(subnormal[subnormal != 0.0])) < np.finfo(np.float64).tiny
    assert_array_equal(subnormal, subnormal.T)
    # Every published entry is exactly one of the two computed triangles, so no
    # bits were invented or lost on the way to symmetry.
    assert np.all((subnormal == raw) | (subnormal == raw.T))
    # And that is a real distinction here: averaging would have moved entries.
    assert np.any(subnormal != 0.5 * raw + 0.5 * raw.T)


def test_exact_zero_matrix_has_exact_zero_fisher_information() -> None:
    assert_array_equal(_stable_fisher_information(np.zeros((3, 4))), np.zeros((4, 4)))


def test_complete_scaled_jacobian_underflow_is_rejected() -> None:
    position = np.array([3e20, 4e20, 7e20])
    clocks = build_head_lattice()
    dimensionless = _independent_dimensionless_jacobian(position, 0.08, clocks)

    assert np.any(dimensionless != 0.0)
    previous_errors = np.seterr(under="warn")
    try:
        with pytest.raises(PhysicsDomainError, match="underflowed completely"):
            local_identifiability(
                position,
                0.08,
                clocks,
                n_observations=1,
                noise_std=np.finfo(np.float64).max,
            )
    finally:
        np.seterr(**previous_errors)


def test_unrepresentable_subnormal_fisher_information_is_rejected() -> None:
    clocks = ClockArray(build_head_lattice().positions[:2])

    previous_errors = np.seterr(under="warn")
    try:
        with pytest.raises(
            PhysicsDomainError, match="Fisher information is not representably PSD"
        ):
            local_identifiability(
                np.array([3.0, 4.0, 7.0]),
                0.08,
                clocks,
                n_observations=80,
                noise_std=2.5588262897963646e159,
            )
    finally:
        np.seterr(**previous_errors)


def test_underdetermined_array_reports_implicit_zero_and_matching_null() -> None:
    position = np.array([3.0, 4.0, 7.0])
    clocks = ClockArray(build_head_lattice().positions[:4])
    result = local_identifiability(
        position,
        0.08,
        clocks,
        n_observations=80,
        noise_std=0.001,
    )
    _, explicit_singular_values, expected_right_vectors = np.linalg.svd(
        result.scaled_jacobian, full_matrices=True
    )
    expected_singular_values = np.append(explicit_singular_values, 0.0)
    expected_null = expected_right_vectors[-1]
    direction_error = min(
        np.linalg.norm(result.weakest_direction - expected_null),
        np.linalg.norm(result.weakest_direction + expected_null),
    )
    null_residual = math.hypot(
        *(float(value) for value in result.scaled_jacobian @ result.weakest_direction)
    )

    assert result.singular_values.shape == (4,)
    assert_array_equal(result.singular_values, expected_singular_values)
    assert result.singular_values[-1] == 0.0
    assert direction_error <= 2e-15
    assert null_residual <= 2.0 * result.rank_tolerance
    assert result.rank == 3
    assert result.condition_number is None
    assert result.crlb_std is None


def test_repeated_smallest_singular_value_withholds_the_weak_direction() -> None:
    """A multi-dimensional null space has no single weakest direction.

    Two clocks give one contrast row, so three of the four singular values are
    zero and LAPACK's last right-singular vector is an arbitrary basis choice
    inside that subspace. Publishing its loadings would report solver detail as
    geometry.
    """
    two_clocks = ClockArray(build_head_lattice().positions[:2])
    result = local_identifiability(
        np.array([3.0, 4.0, 7.0]),
        0.08,
        two_clocks,
        n_observations=80,
        noise_std=0.001,
    )

    assert result.scaled_jacobian.shape == (1, 4)
    assert result.rank == 1
    assert result.singular_values[-2] == result.singular_values[-1]
    assert result.weakest_direction is None
    assert result.weakest_mode_loadings is None


def test_isolated_smallest_singular_value_still_publishes_the_weak_direction() -> None:
    """The real head is nowhere near degenerate, so the guard must not fire."""
    result = local_identifiability(
        np.array([3.0, 4.0, 7.0]),
        0.08,
        build_head_lattice(),
        n_observations=80,
        noise_std=0.001,
    )
    assert result.weakest_direction is not None
    assert result.weakest_mode_loadings is not None
    gap = float(result.singular_values[-2]) - float(result.singular_values[-1])
    assert gap > 1e6 * result.rank_tolerance


def test_weakest_mode_loadings_are_grouped_squared_components() -> None:
    position = np.array([3.0, 4.0, 7.0])
    clocks = build_head_lattice()
    result = local_identifiability(
        position,
        0.08,
        clocks,
        n_observations=80,
        noise_std=0.001,
    )
    loadings = result.weakest_mode_loadings
    expected_scaled = (
        np.sqrt(80)
        / 0.001
        * _independent_dimensionless_jacobian(position, 0.08, clocks)
    )
    _, _, expected_right_vectors = np.linalg.svd(expected_scaled, full_matrices=False)
    expected_weakest = expected_right_vectors[-1]
    direction_error = min(
        np.linalg.norm(result.weakest_direction - expected_weakest),
        np.linalg.norm(result.weakest_direction + expected_weakest),
    )
    expected_squared = np.square(expected_weakest)
    expected_squared /= expected_squared.sum()

    assert direction_error <= 2e-15
    assert set(loadings) == {"angular", "log_range", "log_mass"}
    assert loadings["angular"] == pytest.approx(
        float(np.sum(expected_squared[:2])), abs=2e-15
    )
    assert loadings["log_range"] == pytest.approx(float(expected_squared[2]), abs=2e-15)
    assert loadings["log_mass"] == pytest.approx(float(expected_squared[3]), abs=2e-15)
    assert all(value >= 0.0 for value in loadings.values())
    assert sum(loadings.values()) == pytest.approx(1.0, abs=2e-15)


def test_fisher_information_is_the_symmetric_psd_gram_matrix() -> None:
    result = local_identifiability(
        np.array([5.5, -3.0, 4.0]),
        0.045,
        build_head_lattice(),
        n_observations=91,
        noise_std=0.002,
    )

    raw_gram = result.scaled_jacobian.T @ result.scaled_jacobian

    # Exact symmetry is a guarantee of the published value, not of the product
    # it comes from: on Linux x86-64 raw_gram's [0, 3] and [3, 0] entries differ
    # by ~2e-14, because a blocked gemm need not sum the two triangles in the
    # same order. Production averages them, so what callers receive is symmetric
    # bit for bit on every platform.
    assert_array_equal(result.fisher_information, result.fisher_information.T)
    assert np.linalg.eigvalsh(result.fisher_information)[0] >= -1e-12

    # Production deliberately does not form this product directly -- it
    # normalizes, takes the Gram, and rescales, so that extreme scales do not
    # underflow (see test_crlb_std_is_stable_for_tiny_valid_singular_values,
    # which asserts the two algorithms differ). The agreement below is
    # therefore up to the rounding of two different routes to the same matrix,
    # still ~1e13 tighter than any real change to the Jacobian would produce.
    assert_allclose(
        result.fisher_information,
        0.5 * (raw_gram + raw_gram.T),
        rtol=1e-13,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "n_observations",
    [0, -1, 1.0, True, np.nan, np.inf, 1 + 0j],
)
def test_local_identifiability_rejects_invalid_observation_count(
    n_observations: object,
) -> None:
    with pytest.raises(ValueError, match="n_observations"):
        local_identifiability(
            np.array([3.0, 4.0, 7.0]),
            0.08,
            build_head_lattice(),
            n_observations=n_observations,
            noise_std=0.001,
        )


@pytest.mark.parametrize(
    "noise_std",
    [0.0, -0.001, True, np.nan, np.inf, 0.001 + 0.0j],
)
def test_local_identifiability_rejects_invalid_noise_without_warnings(
    noise_std: object,
) -> None:
    with pytest.raises(ValueError, match="noise_std"):
        local_identifiability(
            np.array([3.0, 4.0, 7.0]),
            0.08,
            build_head_lattice(),
            n_observations=80,
            noise_std=noise_std,
        )


def test_local_identifiability_result_is_deeply_immutable() -> None:
    result = local_identifiability(
        np.array([3.0, 4.0, 7.0]),
        0.08,
        build_head_lattice(),
        n_observations=80,
        noise_std=0.001,
    )
    arrays = [
        result.jacobian,
        result.scaled_jacobian,
        result.fisher_information,
        result.singular_values,
        result.weakest_direction,
    ]
    if result.crlb_std is not None:
        arrays.append(result.crlb_std)

    with pytest.raises(FrozenInstanceError):
        result.rank = 0  # type: ignore[misc]
    for array in arrays:
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE"):
            array.setflags(write=True)
    with pytest.raises(TypeError):
        result.weakest_mode_loadings["angular"] = 0.0  # type: ignore[index]
