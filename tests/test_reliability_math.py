"""Exact mathematical checks for the echolocation reliability diagnostics."""

from collections.abc import Callable
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from clocks._reliability import (
    IdentifiabilityResult,
    _dimensionless_jacobian,
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


def test_dimensionless_jacobian_has_exact_parameter_order_and_shape() -> None:
    position = np.array([3.0, 4.0, 7.0])
    mass = 0.08
    clocks = build_head_lattice()
    basis = tangent_basis(position)
    position_jacobian, mass_jacobian = contrast_jacobian(position, mass, clocks)

    jacobian = _dimensionless_jacobian(position, mass, clocks, basis=basis)
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

    original = _dimensionless_jacobian(position, 0.08, clocks, basis=basis)
    rotated = _dimensionless_jacobian(position, 0.08, clocks, basis=basis @ rotation)

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


def test_rank_uses_the_documented_machine_precision_tolerance() -> None:
    result = local_identifiability(
        np.array([3.0, 4.0, 7.0]),
        0.08,
        build_head_lattice(),
        n_observations=80,
        noise_std=0.001,
    )
    expected = (
        np.finfo(np.float64).eps
        * max(result.scaled_jacobian.shape)
        * result.singular_values[0]
    )

    assert result.rank_tolerance == expected
    assert result.rank == int(np.count_nonzero(result.singular_values > expected))


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


def test_weakest_mode_loadings_are_grouped_squared_components() -> None:
    result = local_identifiability(
        np.array([3.0, 4.0, 7.0]),
        0.08,
        build_head_lattice(),
        n_observations=80,
        noise_std=0.001,
    )
    loadings = result.weakest_mode_loadings

    assert loadings["angular"] == pytest.approx(
        float(np.sum(result.weakest_direction[:2] ** 2)), abs=2e-15
    )
    assert loadings["log_range"] == pytest.approx(
        float(result.weakest_direction[2] ** 2), abs=2e-15
    )
    assert loadings["log_mass"] == pytest.approx(
        float(result.weakest_direction[3] ** 2), abs=2e-15
    )
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

    assert_allclose(
        result.fisher_information,
        result.scaled_jacobian.T @ result.scaled_jacobian,
        rtol=2e-15,
        atol=2e-15,
    )
    assert_allclose(
        result.fisher_information,
        result.fisher_information.T,
        rtol=0.0,
        atol=0.0,
    )
    assert np.linalg.eigvalsh(result.fisher_information)[0] >= -1e-12


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
