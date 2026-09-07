"""Tests for the gravitational time dilation forward model."""

import warnings

import numpy as np
import pytest

from clocks._support import density_support_mask
from clocks.physics import (
    WEAK_FIELD_LIMIT,
    PhysicsDomainError,
    _density_potential_batch,
    _point_mass_potential_batch,
    clock_rates,
    clock_rates_batch,
    clock_rates_batch_multi,
    clock_rates_density_gaussian,
    clock_rates_density_gaussian_batch,
    compute_distances,
    gravitational_potential,
    time_dilation_factor,
)
from clocks.types import ClockArray, MassConfig


@pytest.mark.parametrize(
    "operation",
    [
        lambda: compute_distances(np.array([[0.0 + 1.0j]]), np.array([[1.0]])),
        lambda: compute_distances(
            np.array([[0.0]]), np.array([[1.0]]), track_offset=1.0 + 1.0j
        ),
        lambda: gravitational_potential(np.array([[1.0]]), np.array([0.01 + 1.0j])),
        lambda: time_dilation_factor(np.array([-0.01 + 1.0j])),
        lambda: clock_rates_batch(
            np.array([[1.0 + 1.0j]]),
            np.array([0.01]),
            ClockArray([[0.0]], track_offset=1.0),
        ),
        lambda: clock_rates_density_gaussian_batch(
            np.array([[0.0, 1.0, 0.01 + 1.0j]]),
            ClockArray([[0.0]], track_offset=1.0),
        ),
        lambda: clock_rates_density_gaussian_batch(
            np.array([[0.0, 1.0, 0.01]]),
            ClockArray([[0.0]], track_offset=1.0),
            integration_limit=10.0 + 1.0j,
        ),
    ],
)
def test_public_physics_inputs_reject_complex_values_without_warning(operation) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="real-valued"):
            operation()


class TestComputeDistances:
    def test_1d_single(self) -> None:
        clocks = np.array([[0.0], [3.0], [5.0]])
        masses = np.array([[1.0]])
        dist = compute_distances(clocks, masses)
        np.testing.assert_allclose(dist, [[1.0], [2.0], [4.0]])

    def test_2d(self) -> None:
        clocks = np.array([[0.0, 0.0], [3.0, 4.0]])
        masses = np.array([[0.0, 0.0]])
        dist = compute_distances(clocks, masses)
        np.testing.assert_allclose(dist, [[0.0], [5.0]])

    def test_track_offset(self) -> None:
        clocks = np.array([[0.0]])
        masses = np.array([[0.0]])
        dist = compute_distances(clocks, masses, track_offset=3.0)
        np.testing.assert_allclose(dist, [[3.0]])

    def test_multiple_masses(self) -> None:
        clocks = np.array([[0.0]])
        masses = np.array([[1.0], [4.0]])
        dist = compute_distances(clocks, masses)
        np.testing.assert_allclose(dist, [[1.0, 4.0]])

    def test_computed_distance_overflow_is_a_domain_error_without_warning(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(PhysicsDomainError, match="distance.*finite"):
                compute_distances(np.array([[1e308]]), np.array([[-1e308]]))

    def test_track_offset_overflow_is_a_domain_error_without_warning(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(PhysicsDomainError, match="distance.*finite"):
                compute_distances(np.array([[0.0]]), np.array([[0.0]]), 1e308)

    @pytest.mark.parametrize(
        ("clocks", "masses", "offset", "message"),
        [
            (np.array([0.0]), np.array([[1.0]]), 0.0, "clock_positions must be 2-D"),
            (
                np.array([[0.0]]),
                np.array([1.0]),
                0.0,
                "mass_positions must be 2-D",
            ),
            (
                np.array([[0.0]]),
                np.array([[1.0, 2.0]]),
                0.0,
                "spatial dimensions",
            ),
            (
                np.array([[np.nan]]),
                np.array([[1.0]]),
                0.0,
                "finite",
            ),
            (np.array([[0.0]]), np.array([[1.0]]), -1.0, "nonnegative"),
        ],
    )
    def test_rejects_malformed_inputs(
        self,
        clocks: np.ndarray,
        masses: np.ndarray,
        offset: float,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            compute_distances(clocks, masses, offset)


class TestGravitationalPotential:
    def test_zero_mass(self) -> None:
        dist = np.array([[1.0, 2.0]])
        masses = np.array([0.0, 0.0])
        pot = gravitational_potential(dist, masses)
        np.testing.assert_allclose(pot, [0.0])

    def test_single_mass(self) -> None:
        dist = np.array([[2.0]])
        masses = np.array([0.04])
        pot = gravitational_potential(dist, masses)
        np.testing.assert_allclose(pot, [-0.02])

    def test_potential_is_negative(self) -> None:
        dist = np.array([[1.0], [2.0], [5.0]])
        masses = np.array([0.04])
        pot = gravitational_potential(dist, masses)
        assert np.all(pot < 0)

    def test_closer_is_deeper(self) -> None:
        dist = np.array([[1.0], [2.0], [5.0]])
        masses = np.array([0.04])
        pot = gravitational_potential(dist, masses)
        # More negative potential at closer distances
        assert pot[0] < pot[1] < pot[2]

    def test_zero_mass_at_zero_distance_contributes_exactly_zero(self) -> None:
        result = gravitational_potential(np.array([[0.0, 2.0]]), np.array([0.0, 0.04]))
        np.testing.assert_array_equal(result, np.array([-0.02]))

    def test_positive_mass_at_zero_distance_is_a_domain_error(self) -> None:
        with pytest.raises(PhysicsDomainError, match="zero distance"):
            gravitational_potential(np.array([[0.0]]), np.array([0.01]))

    def test_computed_potential_overflow_is_a_domain_error_without_warning(
        self,
    ) -> None:
        distance = np.nextafter(0.0, 1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(PhysicsDomainError, match="potential.*finite"):
                gravitational_potential(np.array([[distance]]), np.array([1.0]))

    @pytest.mark.parametrize(
        ("distances", "masses", "message"),
        [
            (np.array([1.0]), np.array([0.01]), "distances must be 2-D"),
            (np.array([[1.0]]), np.array([[0.01]]), "masses must be 1-D"),
            (np.array([[1.0, 2.0]]), np.array([0.01]), "one column per mass"),
            (np.array([[np.inf]]), np.array([0.01]), "finite"),
            (np.array([[-1.0]]), np.array([0.01]), "nonnegative"),
            (np.array([[1.0]]), np.array([-0.01]), "nonnegative"),
        ],
    )
    def test_rejects_malformed_inputs(
        self, distances: np.ndarray, masses: np.ndarray, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            gravitational_potential(distances, masses)


class TestTimeDilation:
    def test_zero_potential(self) -> None:
        factor = time_dilation_factor(np.array([0.0, 0.0]))
        np.testing.assert_allclose(factor, [1.0, 1.0])

    def test_negative_potential_slows_clocks(self) -> None:
        factor = time_dilation_factor(np.array([-0.01]))
        assert 0.0 < factor[0] < 1.0

    def test_monotonic(self) -> None:
        pot = np.array([0.0, -0.01, -0.02, -0.03])
        factor = time_dilation_factor(pot)
        # Less negative potential → faster clock
        assert np.all(np.diff(factor) < 0)

    def test_accepts_validity_boundary_without_clamping(self) -> None:
        assert WEAK_FIELD_LIMIT == 0.1
        result = time_dilation_factor(np.array([-0.05, 0.0]))
        np.testing.assert_array_equal(result, np.sqrt(np.array([0.9, 1.0])))

    @pytest.mark.parametrize("potential", [-0.0500001, 0.001, np.nan, -np.inf])
    def test_rejects_outside_model_domain(self, potential: float) -> None:
        with pytest.raises(PhysicsDomainError):
            time_dilation_factor(np.array([potential]))

    def test_extreme_finite_potential_rejects_without_overflow_warning(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(PhysicsDomainError, match="weak-field"):
                time_dilation_factor(np.array([-1e308]))

    def test_requires_exact_nonempty_vector(self) -> None:
        with pytest.raises(ValueError, match="potential must be 1-D"):
            time_dilation_factor(np.array([[-0.01]]))
        with pytest.raises(ValueError, match="nonempty"):
            time_dilation_factor(np.array([]))


class TestClockRates:
    def test_zero_mass_gives_rate_one(self) -> None:
        mc = MassConfig(
            positions=np.array([[0.0]]),
            masses=np.array([0.0]),
        )
        ca = ClockArray(positions=np.array([[-5.0], [0.0], [5.0]]))
        rates = clock_rates(mc, ca)
        np.testing.assert_allclose(rates, [1.0, 1.0, 1.0])

    def test_symmetry(self) -> None:
        """Clocks equidistant from mass should have equal rates."""
        mc = MassConfig(
            positions=np.array([[0.0]]),
            masses=np.array([0.04]),
        )
        ca = ClockArray(
            positions=np.array([[-3.0], [3.0]]),
            track_offset=1.0,
        )
        rates = clock_rates(mc, ca)
        np.testing.assert_allclose(rates[0], rates[1])

    def test_closer_clock_is_slower(self) -> None:
        mc = MassConfig(
            positions=np.array([[0.0]]),
            masses=np.array([0.04]),
        )
        ca = ClockArray(
            positions=np.array([[1.0], [5.0]]),
            track_offset=1.0,
        )
        rates = clock_rates(mc, ca)
        assert rates[0] < rates[1]

    def test_dimension_agnostic_2d(self) -> None:
        mc = MassConfig(
            positions=np.array([[0.0, 0.0]]),
            masses=np.array([0.03]),
        )
        ca = ClockArray(positions=np.array([[3.0, 4.0], [-3.0, -4.0]]))
        rates = clock_rates(mc, ca)
        assert rates.shape == (2,)
        np.testing.assert_allclose(rates[0], rates[1])

    def test_dimension_agnostic_3d(self) -> None:
        mc = MassConfig(
            positions=np.array([[0.0, 0.0, 0.0]]),
            masses=np.array([0.02]),
        )
        ca = ClockArray(positions=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        rates = clock_rates(mc, ca)
        np.testing.assert_allclose(rates[0], rates[1])

    def test_rejects_mismatched_spatial_dimensions(self) -> None:
        mass = MassConfig(np.array([[0.0, 1.0]]), np.array([0.01]))
        clocks = ClockArray(np.array([[0.0]]), track_offset=1.0)
        with pytest.raises(ValueError, match="spatial dimensions"):
            clock_rates(mass, clocks)

    def test_rejects_singular_and_outside_weak_field_states(self) -> None:
        clocks = ClockArray(np.array([[0.0]]))
        with pytest.raises(PhysicsDomainError, match="zero distance"):
            clock_rates(MassConfig([[0.0]], [0.01]), clocks)

        offset_clocks = ClockArray(np.array([[0.0]]), track_offset=1.0)
        with pytest.raises(PhysicsDomainError, match="weak-field"):
            clock_rates(MassConfig([[0.0]], [0.051]), offset_clocks)

    def test_computed_overflow_is_a_domain_error_without_warning(self) -> None:
        mass = MassConfig(np.array([[1e308]]), np.array([0.01]))
        clocks = ClockArray(np.array([[-1e308]]))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(PhysicsDomainError, match="distance.*finite"):
                clock_rates(mass, clocks)


class TestClockRatesBatch:
    def test_matches_scalar_1d(self) -> None:
        """Batch should match calling clock_rates per particle."""
        ca = ClockArray(
            positions=np.array([[-5.0], [0.0], [5.0]]),
            track_offset=1.0,
        )
        mass_positions = np.array([[1.0], [3.0], [-2.0]])
        masses = np.array([0.02, 0.03, 0.01])

        batch_result = clock_rates_batch(mass_positions, masses, ca)
        assert batch_result.shape == (3, 3)

        for i in range(3):
            mc = MassConfig(
                positions=mass_positions[i : i + 1], masses=masses[i : i + 1]
            )
            scalar_result = clock_rates(mc, ca)
            np.testing.assert_allclose(batch_result[i], scalar_result)

    def test_matches_scalar_2d(self) -> None:
        ca = ClockArray(
            positions=np.array([[0.0, 0.0], [3.0, 4.0], [-2.0, 1.0]]),
            track_offset=2.0,
        )
        mass_positions = np.array([[1.0, -1.0], [2.0, 3.0]])
        masses = np.array([0.02, 0.03])

        batch_result = clock_rates_batch(mass_positions, masses, ca)
        assert batch_result.shape == (2, 3)

        for i in range(2):
            mc = MassConfig(
                positions=mass_positions[i : i + 1], masses=masses[i : i + 1]
            )
            scalar_result = clock_rates(mc, ca)
            np.testing.assert_allclose(batch_result[i], scalar_result)

    def test_shape(self) -> None:
        ca = ClockArray(positions=np.array([[-5.0], [0.0], [5.0]]))
        mass_positions = np.array([[1.0], [2.0], [3.0], [4.0]])
        masses = np.array([0.01, 0.02, 0.03, 0.04])
        result = clock_rates_batch(mass_positions, masses, ca)
        assert result.shape == (4, 3)

    @pytest.mark.parametrize(
        ("positions", "masses", "message"),
        [
            (np.array([1.0]), np.array([0.01]), "mass_positions must be 2-D"),
            (np.array([[1.0]]), np.array([[0.01]]), "masses must be 1-D"),
            (np.array([[1.0], [2.0]]), np.array([0.01]), "same number"),
            (np.array([[1.0, 2.0]]), np.array([0.01]), "spatial dimensions"),
            (np.array([[np.nan]]), np.array([0.01]), "finite"),
            (np.array([[1.0]]), np.array([-0.01]), "nonnegative"),
        ],
    )
    def test_rejects_malformed_inputs(
        self, positions: np.ndarray, masses: np.ndarray, message: str
    ) -> None:
        clocks = ClockArray(np.array([[0.0]]), track_offset=1.0)
        with pytest.raises(ValueError, match=message):
            clock_rates_batch(positions, masses, clocks)

    def test_rejects_any_invalid_candidate(self) -> None:
        clocks = ClockArray(np.array([[0.0]]), track_offset=1.0)
        with pytest.raises(PhysicsDomainError, match="weak-field"):
            clock_rates_batch(np.array([[2.0], [0.0]]), np.array([0.01, 0.051]), clocks)

    def test_computed_distance_overflow_is_a_domain_error_without_warning(self) -> None:
        clocks = ClockArray(np.array([[-1e308]]), track_offset=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(PhysicsDomainError, match="point-mass state"):
                clock_rates_batch(np.array([[1e308]]), np.array([0.01]), clocks)

    def test_candidate_overflow_is_invalid_without_warning(self) -> None:
        clocks = ClockArray(np.array([[-1e308]]), track_offset=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            potential, valid = _point_mass_potential_batch(
                np.array([[[1e308]]]), np.array([[0.01]]), clocks
            )

        assert not valid[0]
        assert not np.all(np.isfinite(potential[0]))

    def test_extreme_finite_potential_is_invalid_without_warning(self) -> None:
        clocks = ClockArray(np.array([[0.0]]), track_offset=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            potential, valid = _point_mass_potential_batch(
                np.array([[[1.0]]]), np.array([[1e308]]), clocks
            )

        np.testing.assert_array_equal(potential, np.array([[-1e308]]))
        assert not valid[0]

    def test_public_batch_extreme_potential_rejects_without_warning(self) -> None:
        clocks = ClockArray(np.array([[0.0]]), track_offset=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(PhysicsDomainError, match="weak-field"):
                clock_rates_batch(np.array([[1.0]]), np.array([1e308]), clocks)


class TestClockRatesBatchMulti:
    def test_matches_scalar_1d_k2(self) -> None:
        """Batch multi should match calling clock_rates per particle for K=2 in 1D."""
        ca = ClockArray(
            positions=np.array([[-5.0], [0.0], [5.0]]),
            track_offset=1.0,
        )
        # 3 particles, each with 2 masses in 1D
        mass_positions = np.array(
            [
                [[1.0], [3.0]],
                [[-2.0], [4.0]],
                [[0.0], [-1.0]],
            ]
        )
        masses = np.array(
            [
                [0.02, 0.01],
                [0.03, 0.01],
                [0.01, 0.02],
            ]
        )

        batch_result = clock_rates_batch_multi(mass_positions, masses, ca)
        assert batch_result.shape == (3, 3)

        for i in range(3):
            mc = MassConfig(positions=mass_positions[i], masses=masses[i])
            scalar_result = clock_rates(mc, ca)
            np.testing.assert_allclose(batch_result[i], scalar_result)

    def test_matches_scalar_2d_k2(self) -> None:
        """Batch multi should match scalar for K=2 in 2D."""
        ca = ClockArray(
            positions=np.array([[0.0, 0.0], [3.0, 4.0], [-2.0, 1.0]]),
            track_offset=2.0,
        )
        mass_positions = np.array(
            [
                [[1.0, -1.0], [2.0, 3.0]],
                [[-1.0, 2.0], [0.5, -0.5]],
            ]
        )
        masses = np.array(
            [
                [0.02, 0.03],
                [0.01, 0.03],
            ]
        )

        batch_result = clock_rates_batch_multi(mass_positions, masses, ca)
        assert batch_result.shape == (2, 3)

        for i in range(2):
            mc = MassConfig(positions=mass_positions[i], masses=masses[i])
            scalar_result = clock_rates(mc, ca)
            np.testing.assert_allclose(batch_result[i], scalar_result)

    def test_k1_matches_single_mass_batch(self) -> None:
        """K=1 multi-mass batch should match the single-mass batch function."""
        ca = ClockArray(
            positions=np.array([[-5.0], [0.0], [5.0]]),
            track_offset=1.0,
        )
        mass_positions_single = np.array([[1.0], [3.0], [-2.0]])
        masses_single = np.array([0.02, 0.03, 0.01])

        # Single-mass batch
        result_single = clock_rates_batch(mass_positions_single, masses_single, ca)

        # Multi-mass batch with K=1
        mass_positions_multi = mass_positions_single[:, np.newaxis, :]  # (3, 1, 1)
        masses_multi = masses_single[:, np.newaxis]  # (3, 1)
        result_multi = clock_rates_batch_multi(mass_positions_multi, masses_multi, ca)

        np.testing.assert_allclose(result_multi, result_single)

    def test_shape(self) -> None:
        ca = ClockArray(positions=np.array([[-5.0], [0.0], [5.0]]))
        mass_positions = np.array(
            [
                [[1.0], [2.0]],
                [[3.0], [4.0]],
                [[-1.0], [-2.0]],
                [[0.0], [1.0]],
            ]
        )
        masses = np.array(
            [
                [0.01, 0.01],
                [0.01, 0.02],
                [0.02, 0.01],
                [0.0, 0.02],
            ]
        )
        result = clock_rates_batch_multi(mass_positions, masses, ca)
        assert result.shape == (4, 3)

    @pytest.mark.parametrize(
        ("positions", "masses", "message"),
        [
            (np.ones((2, 1)), np.ones((2, 1)), "mass_positions must be 3-D"),
            (np.ones((2, 1, 1)), np.ones(2), "masses must be 2-D"),
            (np.ones((2, 2, 1)), np.ones((2, 1)), "matching"),
            (np.ones((2, 1, 2)), np.ones((2, 1)), "spatial dimensions"),
            (np.array([[[np.nan]]]), np.array([[0.01]]), "finite"),
            (np.ones((1, 1, 1)), np.array([[-0.01]]), "nonnegative"),
        ],
    )
    def test_rejects_malformed_inputs(
        self, positions: np.ndarray, masses: np.ndarray, message: str
    ) -> None:
        clocks = ClockArray(np.array([[0.0]]), track_offset=1.0)
        with pytest.raises(ValueError, match=message):
            clock_rates_batch_multi(positions, masses, clocks)

    def test_private_candidate_evaluation_marks_invalid_without_warning(self) -> None:
        clocks = ClockArray(np.array([[0.0]]))
        positions = np.array([[[0.0]], [[1.0]], [[np.nan]]])
        masses = np.array([[0.01], [0.06], [0.01]])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            potential, valid = _point_mass_potential_batch(positions, masses, clocks)
        assert potential.shape == (3, 1)
        np.testing.assert_array_equal(valid, [False, False, False])


class TestGaussianDensity:
    def _make_clock_array(self) -> ClockArray:
        return ClockArray(
            positions=np.array([[-6.0], [-3.0], [0.0], [3.0], [6.0]]),
            track_offset=1.0,
        )

    def test_zero_amplitude_gives_rate_one(self) -> None:
        """Zero amplitude (no mass) should give rate 1.0 everywhere."""
        ca = self._make_clock_array()
        params = np.array([0.0, 1.0, 0.0])
        rates = clock_rates_density_gaussian(params, ca)
        np.testing.assert_allclose(rates, 1.0, atol=1e-10)

    def test_density_rates_below_one(self) -> None:
        """Nonzero mass density should produce rates below 1."""
        ca = self._make_clock_array()
        params = np.array([0.0, 2.0, 0.01])
        rates = clock_rates_density_gaussian(params, ca)
        assert np.all(rates < 1.0)

    def test_density_closer_clock_slower(self) -> None:
        """Clock nearer to the density center should tick slower."""
        ca = ClockArray(
            positions=np.array([[0.0], [5.0]]),
            track_offset=1.0,
        )
        params = np.array([0.0, 2.0, 0.01])
        rates = clock_rates_density_gaussian(params, ca)
        assert rates[0] < rates[1]

    def test_density_symmetry(self) -> None:
        """Equidistant clocks should have equal rates."""
        ca = ClockArray(
            positions=np.array([[-3.0], [3.0]]),
            track_offset=1.0,
        )
        params = np.array([0.0, 2.0, 0.01])
        rates = clock_rates_density_gaussian(params, ca)
        np.testing.assert_allclose(rates[0], rates[1], atol=1e-10)

    def test_narrow_density_approximates_point_mass(self) -> None:
        """Very narrow Gaussian should approximate a point mass."""
        ca = ClockArray(
            positions=np.array([[-5.0], [0.0], [5.0]]),
            track_offset=1.0,
        )
        # Narrow Gaussian: sigma=0.01, total mass ≈ A * sigma * sqrt(2*pi)
        sigma = 0.01
        total_mass = 0.02
        amplitude = total_mass / (sigma * np.sqrt(2 * np.pi))
        params = np.array([2.0, sigma, amplitude])

        density_rates = clock_rates_density_gaussian(params, ca)

        # Compare with point mass of same total mass at same location
        mc = MassConfig(positions=np.array([[2.0]]), masses=np.array([total_mass]))
        point_rates = clock_rates(mc, ca)

        np.testing.assert_allclose(density_rates, point_rates, atol=1e-3)

    def test_density_batch_matches_scalar(self) -> None:
        """Batch version should match scalar within numerical tolerance."""
        ca = self._make_clock_array()
        params_batch = np.array(
            [
                [0.0, 2.0, 0.01],
                [1.5, 1.0, 0.02],
                [-2.0, 3.0, 0.005],
            ]
        )

        batch_result = clock_rates_density_gaussian_batch(params_batch, ca)

        for i in range(len(params_batch)):
            scalar_result = clock_rates_density_gaussian(params_batch[i], ca)
            np.testing.assert_allclose(
                batch_result[i],
                scalar_result,
                atol=1e-4,
                err_msg=f"Mismatch for particle {i}",
            )

    @pytest.mark.parametrize("track_offset", [1.0, 0.5, 0.1, 0.05, 0.02, 0.01, 0.005])
    def test_density_batch_matches_scalar_across_offset_ratios(
        self, track_offset: float
    ) -> None:
        """Batch quadrature must resolve the kernel for any track_offset/sigma.

        The kernel ``1 / sqrt((x - c)^2 + h^2)`` has width ``h = track_offset``.
        A fixed grid over +/-10 sigma steps over that peak once
        ``track_offset / sigma`` falls below roughly 0.1, so the assertion is on
        the relative error of the *delay* ``1 - rate``, which is the physically
        meaningful quantity; the rates themselves all sit near 1.
        """
        ca = ClockArray(positions=np.array([[0.0]]), track_offset=track_offset)
        params = np.array([0.0, 1.0, 0.001])

        batch_rate = clock_rates_density_gaussian_batch(params[np.newaxis], ca)[0, 0]
        scalar_rate = clock_rates_density_gaussian(params, ca)[0]

        scalar_delay = 1.0 - scalar_rate
        assert scalar_delay > 0.0
        relative_delay_error = abs(batch_rate - scalar_rate) / scalar_delay
        assert relative_delay_error < 1e-6, (
            f"track_offset={track_offset}: batch delay differs from adaptive "
            f"scalar quadrature by {relative_delay_error:.3%}"
        )

    def test_density_batch_rejects_candidate_outside_weak_field(self) -> None:
        """A state the scalar model rejects must not survive the batch path.

        With ``track_offset=0.01`` the fixed grid underestimates ``|Phi|`` enough
        to accept a candidate that genuinely violates ``|2 Phi| <= 0.1``, and the
        support mask shares the same routine, so both must reject it.
        """
        ca = ClockArray(positions=np.array([[0.0]]), track_offset=0.01)
        params = np.array([0.0, 1.0, 0.005])

        with pytest.raises(PhysicsDomainError):
            clock_rates_density_gaussian(params, ca)
        with pytest.raises(PhysicsDomainError):
            clock_rates_density_gaussian_batch(params[np.newaxis], ca)

        mask = density_support_mask(
            params[np.newaxis],
            clock_array=ca,
            mu_range=(-8.0, 8.0),
            sigma_range=(0.1, 5.0),
            amplitude_range=(0.001, 0.03),
        )
        assert not bool(mask[0])

    def test_density_batch_matches_scalar_at_shipped_offset(self) -> None:
        """The shipped demo geometry must not move when quadrature changes.

        Sweeps in-support draws from the demo's own prior ranges at
        ``track_offset=1.0`` and pins batch/scalar agreement far below the
        demo's ``NOISE_STD = 0.005``.
        """
        ca = self._make_clock_array()
        rng = np.random.default_rng(20240514)
        draws = np.column_stack(
            (
                rng.uniform(-8.0, 8.0, 400),
                rng.uniform(0.1, 5.0, 400),
                rng.uniform(0.001, 0.030, 400),
            )
        )
        supported = draws[
            density_support_mask(
                draws,
                clock_array=ca,
                mu_range=(-8.0, 8.0),
                sigma_range=(0.1, 5.0),
                amplitude_range=(0.001, 0.030),
            )
        ]
        assert len(supported) > 50

        batch_rates = clock_rates_density_gaussian_batch(supported, ca)
        scalar_rates = np.array(
            [clock_rates_density_gaussian(params, ca) for params in supported]
        )

        max_difference = float(np.max(np.abs(batch_rates - scalar_rates)))
        assert max_difference < 1e-6, (
            f"shipped configuration moved by {max_difference:.3e}, which is not "
            "negligible against NOISE_STD=0.005"
        )

    def test_density_batch_rejects_far_clock_with_tiny_offset(self) -> None:
        """A clock many sigma from the profile must not lose the profile.

        The ``sinh`` substitution concentrates its grid at the clock, so with the
        clock 9.25 sigma away and a tiny track_offset it starves the profile and
        understates ``|Phi|`` by about 14%, wrongly accepting this state. Here
        the true ``|2 Phi|`` is about 0.1042, so both paths must reject it.
        """
        ca = ClockArray(positions=np.array([[9.25]]), track_offset=1e-12)
        params = np.array([0.0, 1.0, 0.19])

        with pytest.raises(PhysicsDomainError):
            clock_rates_density_gaussian(params, ca)
        with pytest.raises(PhysicsDomainError):
            clock_rates_density_gaussian_batch(params[np.newaxis], ca)

        mask = density_support_mask(
            params[np.newaxis],
            clock_array=ca,
            mu_range=(-10.0, 10.0),
            sigma_range=(0.1, 5.0),
            amplitude_range=(0.001, 0.5),
        )
        assert not bool(mask[0])

    def test_density_batch_keeps_resolution_at_large_coordinates(self) -> None:
        """Absolute coordinates must not quantise the profile.

        Reconstructing ``x = c + h * sinh(u)`` and only then subtracting ``mu``
        rounds the profile onto the float spacing at ``mu`` -- 0.125 near 1e15 --
        which understated ``|Phi|`` enough to accept this state. The true
        ``|2 Phi|`` is about 0.1004, so it must be rejected.
        """
        centre = 1e15
        ca = ClockArray(positions=np.array([[centre + 8.0]]), track_offset=1e-4)
        params = np.array([centre, 1.0, 0.1575])

        with pytest.raises(PhysicsDomainError):
            clock_rates_density_gaussian_batch(params[np.newaxis], ca)

    def test_density_batch_keeps_plain_grid_for_truncated_profiles(self) -> None:
        """A short integration limit leaves the profile alive at the endpoints.

        The substituted grid's accuracy rests on the integrand decaying to zero
        at both ends, which a limit of two sigma does not give, so the plain grid
        must be kept. With the substitution forced here the potential shrinks to
        -0.0496 and the state is wrongly accepted.
        """
        ca = ClockArray(positions=np.array([[0.0]]), track_offset=1.0)
        params = np.array([0.0, 1.0, 0.0264])

        with pytest.raises(PhysicsDomainError):
            clock_rates_density_gaussian_batch(
                params[np.newaxis], ca, integration_limit=2.0, n_quad=5
            )

    def test_density_substituted_branch_survives_extreme_amplitude(self) -> None:
        """The substituted branch must not carry the amplitude into the sum.

        This geometry takes the substituted branch, and summing ordinates scaled
        by a 1e308 amplitude overflows to ``-inf`` before the interval width is
        applied. Factoring the amplitude out keeps the result exact.
        """
        ca = ClockArray(positions=np.array([[10.0]]), track_offset=0.5)
        params = np.array([[0.0, 1.0, 1e308]])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            potential, converged = _density_potential_batch(params, ca, 10.0, 200)

        assert converged.all()
        assert np.isfinite(potential).all()
        np.testing.assert_allclose(potential[0, 0], -2.529157059929335e307, rtol=1e-12)

    @pytest.mark.parametrize(
        ("params", "clock_position", "track_offset", "expected"),
        [
            # Bounds so narrow that both endpoints minus the clock round to the
            # same float: the substituted grid collapses to zero width.
            ((0.0, 1e-18, 3e16), 1.0, 1.0, -0.0531736155271655),
            # Amplitude large enough to overflow a summation that carries it.
            ((0.0, 1e-310, 1e308), 0.0, 10.0, -0.00250662827463),
            # track_offset so small that (x - c) / h overflows to infinity.
            ((0.0, 1.0, 0.001), 20.0, 1e-310, -0.00012564712213),
        ],
        ids=["collapsed-bounds", "overflowing-amplitude", "overflowing-offset"],
    )
    def test_density_potential_survives_extreme_scales(
        self,
        params: tuple[float, float, float],
        clock_position: float,
        track_offset: float,
        expected: float,
    ) -> None:
        """Extreme but finite states must stay computable, not go to 0/inf/NaN.

        Each of these is representable and integrable; a quadrature that leans on
        ``arcsinh((x - c) / h)`` alone returns exactly zero, ``-inf`` and ``NaN``
        for them respectively.
        """
        ca = ClockArray(
            positions=np.array([[clock_position]]), track_offset=track_offset
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            potential, converged = _density_potential_batch(
                np.array([params]), ca, 10.0, 200
            )

        assert converged.all()
        assert np.isfinite(potential).all()
        np.testing.assert_allclose(potential[0, 0], expected, rtol=1e-9)

    def test_density_batch_shape(self) -> None:
        """Batch output should have correct shape."""
        ca = self._make_clock_array()
        params_batch = np.array(
            [
                [0.0, 1.0, 0.01],
                [1.0, 2.0, 0.01],
                [-1.0, 0.5, 0.01],
                [2.0, 3.0, 0.005],
            ]
        )
        result = clock_rates_density_gaussian_batch(params_batch, ca)
        assert result.shape == (4, 5)

    @pytest.mark.parametrize(
        ("params", "message"),
        [
            (np.array([[0.0, 1.0, 0.01]]), "params must be 1-D"),
            (np.array([0.0, 1.0]), "exactly three"),
            (np.array([0.0, 0.0, 0.01]), "sigma must be positive"),
            (np.array([0.0, 1.0, -0.01]), "amplitude must be nonnegative"),
            (np.array([0.0, np.inf, 0.01]), "finite"),
        ],
    )
    def test_scalar_rejects_invalid_parameters(
        self, params: np.ndarray, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            clock_rates_density_gaussian(params, self._make_clock_array())

    @pytest.mark.parametrize(
        ("params", "message"),
        [
            (np.array([0.0, 1.0, 0.01]), "params_batch must be 2-D"),
            (np.ones((2, 2)), "exactly three"),
            (np.array([[0.0, -1.0, 0.01]]), "sigma must be positive"),
            (np.array([[0.0, 1.0, -0.01]]), "amplitude must be nonnegative"),
            (np.array([[0.0, 1.0, np.nan]]), "finite"),
        ],
    )
    def test_batch_rejects_invalid_parameters(
        self, params: np.ndarray, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            clock_rates_density_gaussian_batch(params, self._make_clock_array())

    def test_density_requires_positive_track_offset(self) -> None:
        clocks = ClockArray(np.array([[0.0]]), track_offset=0.0)
        with pytest.raises(ValueError, match="track_offset must be positive"):
            clock_rates_density_gaussian(np.array([0.0, 1.0, 0.01]), clocks)

    @pytest.mark.parametrize("limit", [0.0, -1.0, np.inf])
    def test_density_requires_positive_finite_integration_limit(
        self, limit: float
    ) -> None:
        with pytest.raises(ValueError, match="integration_limit"):
            clock_rates_density_gaussian(
                np.array([0.0, 1.0, 0.01]),
                self._make_clock_array(),
                integration_limit=limit,
            )

    @pytest.mark.parametrize("n_quad", [True, 1, 2.5])
    def test_density_batch_requires_integer_quadrature_count(
        self, n_quad: object
    ) -> None:
        with pytest.raises(ValueError, match="n_quad"):
            clock_rates_density_gaussian_batch(
                np.array([[0.0, 1.0, 0.01]]),
                self._make_clock_array(),
                n_quad=n_quad,
            )

    def test_density_scalar_and_batch_reject_same_invalid_domain(self) -> None:
        params = np.array([0.0, 2.0, 0.1])
        with pytest.raises(PhysicsDomainError, match="weak-field"):
            clock_rates_density_gaussian(params, self._make_clock_array())
        with pytest.raises(PhysicsDomainError, match="weak-field"):
            clock_rates_density_gaussian_batch(
                params.reshape(1, 3), self._make_clock_array()
            )

    @pytest.mark.parametrize("sigma", [1e308, 1e307, 1e306])
    def test_density_scalar_rejects_extreme_finite_width_without_warning(
        self, sigma: float
    ) -> None:
        params = np.array([0.0, sigma, 0.01])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(PhysicsDomainError, match="integration bounds"):
                clock_rates_density_gaussian(params, self._make_clock_array())

    @pytest.mark.parametrize("sigma", [1e308, 1e307, 1e306])
    def test_density_batch_rejects_extreme_finite_width_without_warning(
        self, sigma: float
    ) -> None:
        params = np.array([[0.0, sigma, 0.01]])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(PhysicsDomainError, match="integration bounds"):
                clock_rates_density_gaussian_batch(params, self._make_clock_array())

    def test_density_scalar_rejects_extreme_finite_amplitude_without_warning(
        self,
    ) -> None:
        params = np.array([0.0, 1.0, 1e308])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(PhysicsDomainError, match="density quadrature"):
                clock_rates_density_gaussian(params, self._make_clock_array())

    def test_density_batch_rejects_extreme_finite_amplitude_without_warning(
        self,
    ) -> None:
        params = np.array([[0.0, 1.0, 1e308]])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(
                PhysicsDomainError, match="density (potential|quadrature)"
            ):
                clock_rates_density_gaussian_batch(params, self._make_clock_array())


class TestBatchEquivalence3D:
    def test_clock_rates_batch_matches_loop_in_3d(self) -> None:
        rng = np.random.default_rng(7)
        clock_array = ClockArray(
            positions=rng.uniform(-2, 2, size=(9, 3)), track_offset=0.0
        )
        mass_positions = rng.uniform(3, 8, size=(20, 3))
        masses = rng.uniform(0.005, 0.04, size=20)
        batch = clock_rates_batch(mass_positions, masses, clock_array)
        assert batch.shape == (20, 9)
        for i in range(20):
            single = clock_rates(
                MassConfig(
                    positions=mass_positions[i].reshape(1, 3),
                    masses=masses[i : i + 1],
                ),
                clock_array,
            )
            assert np.allclose(batch[i], single)

    @pytest.mark.parametrize(
        ("label", "params", "clock", "offset", "limit", "n_quad"),
        [
            ("nondefault-coarse", [0.2, 1.0, 0.0034], 0.0, 0.001, 6.0, 5),
            ("nondefault-wide-offset", [2.8, 1.0, 0.05458], 0.0, 1.08, 6.0, 15),
            (
                "default-settings",
                [0.0, 1.0, 0.000286],
                0.7542139006700481,
                1e-50,
                10.0,
                200,
            ),
        ],
    )
    def test_density_batch_never_accepts_what_the_reference_rejects(
        self,
        label: str,
        params: list[float],
        clock: float,
        offset: float,
        limit: float,
        n_quad: int,
    ) -> None:
        """A grid that cannot resolve the integrand must not certify a state.

        Each of these was measured against two independent references -- adaptive
        quadrature with the log singularity subtracted in closed form, and a
        4,000,001-node ``sinh`` grid, agreeing to 2.2e-16.  In every one the true
        ``|2 Phi|`` exceeds the weak-field limit, so the state must be rejected.
        A fixed grid is entitled to fail here; it is not entitled to return a
        confident wrong number.
        """
        ca = ClockArray(positions=np.array([[clock]]), track_offset=offset)
        values = np.asarray(params, dtype=float)[np.newaxis]

        mask = density_support_mask(
            values,
            clock_array=ca,
            mu_range=(-10.0, 10.0),
            sigma_range=(0.1, 5.0),
            amplitude_range=(1e-6, 0.5),
            integration_limit=limit,
            n_quad=n_quad,
        )
        assert not bool(mask[0]), f"{label}: support mask accepted a rejected state"

        with pytest.raises(PhysicsDomainError):
            clock_rates_density_gaussian_batch(
                values, ca, integration_limit=limit, n_quad=n_quad
            )

    def test_density_batch_does_not_trust_a_lucky_node(self) -> None:
        """Agreement at one resolution is not convergence.

        At ``track_offset=1e-50`` the plain grid's accuracy is decided by where
        its nodes happen to fall relative to the clock.  With ``n_quad=200`` the
        nearest node lands 4.45e-4 away and the answer is right to 6.5e-12; at
        ``n_quad=400`` the nearest node is 0.023 away and the same grid returns
        -0.0023 against a true -0.0500, off by a factor of 22.  Neither
        resolution may be reported as a result.
        """
        ca = ClockArray(positions=np.array([[0.7542139006700481]]), track_offset=1e-50)
        values = np.array([[0.0, 1.0, 0.000286]])

        for n_quad in (200, 400):
            with pytest.raises(PhysicsDomainError):
                clock_rates_density_gaussian_batch(
                    values, ca, integration_limit=10.0, n_quad=n_quad
                )

    def test_density_batch_requires_a_second_opinion_where_a_spike_can_hide(
        self,
    ) -> None:
        """Refinement alone can certify a wrong answer, so it is not enough.

        Refining a grid detects a feature that one resolution samples and the
        other does not.  It is blind to a feature that *neither* samples: with
        ``track_offset`` far below the step size, both resolutions can miss the
        kernel's peak in the same way and agree to better than the convergence
        tolerance while sharing an error four orders larger than it.

        This point was found by searching 60,000 randomized geometries; 128 of
        them self-converged on the plain grid while materially wrong.  Here the
        plain grid agrees with its own refinement to within 1e-8 and is still
        wrong by 8.9e-05 against a reference with the log singularity subtracted
        in closed form.  The substituted grid disagrees, so neither is reported.
        """
        params = np.array(
            [[2.2268092007579856, 0.36267764966515176, 0.016122017435951475]]
        )
        ca = ClockArray(
            positions=np.array([[0.24053235159921216]]),
            track_offset=2.4457603407890153e-33,
        )

        potential, converged = _density_potential_batch(params, ca, 10.0, 400)

        assert not bool(converged[0, 0])
        assert np.isnan(potential[0, 0])
