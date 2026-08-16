"""Tests for the gravitational time dilation forward model."""

import warnings

import numpy as np
import pytest

from clocks.physics import (
    WEAK_FIELD_LIMIT,
    PhysicsDomainError,
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
