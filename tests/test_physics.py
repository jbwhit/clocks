"""Tests for the gravitational time dilation forward model."""

import numpy as np

from clocks.physics import (
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


class TestGravitationalPotential:
    def test_zero_mass(self) -> None:
        dist = np.array([[1.0, 2.0]])
        masses = np.array([0.0, 0.0])
        pot = gravitational_potential(dist, masses)
        np.testing.assert_allclose(pot, [0.0])

    def test_single_mass(self) -> None:
        dist = np.array([[2.0]])
        masses = np.array([1.0])
        pot = gravitational_potential(dist, masses)
        np.testing.assert_allclose(pot, [-0.5])

    def test_potential_is_negative(self) -> None:
        dist = np.array([[1.0], [2.0], [5.0]])
        masses = np.array([1.0])
        pot = gravitational_potential(dist, masses)
        assert np.all(pot < 0)

    def test_closer_is_deeper(self) -> None:
        dist = np.array([[1.0], [2.0], [5.0]])
        masses = np.array([1.0])
        pot = gravitational_potential(dist, masses)
        # More negative potential at closer distances
        assert pot[0] < pot[1] < pot[2]


class TestTimeDilation:
    def test_zero_potential(self) -> None:
        factor = time_dilation_factor(np.array([0.0, 0.0]))
        np.testing.assert_allclose(factor, [1.0, 1.0])

    def test_negative_potential_slows_clocks(self) -> None:
        factor = time_dilation_factor(np.array([-0.1]))
        assert 0.0 < factor[0] < 1.0

    def test_monotonic(self) -> None:
        pot = np.array([0.0, -0.1, -0.2, -0.3])
        factor = time_dilation_factor(pot)
        # Less negative potential → faster clock
        assert np.all(np.diff(factor) < 0)

    def test_black_hole_guard(self) -> None:
        """Extremely deep potential doesn't produce NaN."""
        factor = time_dilation_factor(np.array([-10.0]))
        assert np.isfinite(factor[0])
        assert factor[0] > 0


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
            masses=np.array([1.0]),
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
            masses=np.array([0.5]),
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
            masses=np.array([0.3]),
        )
        ca = ClockArray(positions=np.array([[3.0, 4.0], [-3.0, -4.0]]))
        rates = clock_rates(mc, ca)
        assert rates.shape == (2,)
        np.testing.assert_allclose(rates[0], rates[1])

    def test_dimension_agnostic_3d(self) -> None:
        mc = MassConfig(
            positions=np.array([[0.0, 0.0, 0.0]]),
            masses=np.array([0.2]),
        )
        ca = ClockArray(positions=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        rates = clock_rates(mc, ca)
        np.testing.assert_allclose(rates[0], rates[1])


class TestClockRatesBatch:
    def test_matches_scalar_1d(self) -> None:
        """Batch should match calling clock_rates per particle."""
        ca = ClockArray(
            positions=np.array([[-5.0], [0.0], [5.0]]),
            track_offset=1.0,
        )
        mass_positions = np.array([[1.0], [3.0], [-2.0]])
        masses = np.array([0.5, 0.8, 0.3])

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
        masses = np.array([0.4, 0.6])

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
        masses = np.array([0.1, 0.2, 0.3, 0.4])
        result = clock_rates_batch(mass_positions, masses, ca)
        assert result.shape == (4, 3)


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
                [0.5, 0.3],
                [0.8, 0.2],
                [0.4, 0.6],
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
                [0.4, 0.6],
                [0.3, 0.7],
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
        masses_single = np.array([0.5, 0.8, 0.3])

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
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
                [0.7, 0.8],
            ]
        )
        result = clock_rates_batch_multi(mass_positions, masses, ca)
        assert result.shape == (4, 3)


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
        params = np.array([0.0, 2.0, 0.3])
        rates = clock_rates_density_gaussian(params, ca)
        assert np.all(rates < 1.0)

    def test_density_closer_clock_slower(self) -> None:
        """Clock nearer to the density center should tick slower."""
        ca = ClockArray(
            positions=np.array([[0.0], [5.0]]),
            track_offset=1.0,
        )
        params = np.array([0.0, 2.0, 0.3])
        rates = clock_rates_density_gaussian(params, ca)
        assert rates[0] < rates[1]

    def test_density_symmetry(self) -> None:
        """Equidistant clocks should have equal rates."""
        ca = ClockArray(
            positions=np.array([[-3.0], [3.0]]),
            track_offset=1.0,
        )
        params = np.array([0.0, 2.0, 0.3])
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
        total_mass = 0.5
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
                [0.0, 2.0, 0.3],
                [1.5, 1.0, 0.5],
                [-2.0, 3.0, 0.1],
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
                [0.0, 1.0, 0.2],
                [1.0, 2.0, 0.3],
                [-1.0, 0.5, 0.4],
                [2.0, 3.0, 0.1],
            ]
        )
        result = clock_rates_density_gaussian_batch(params_batch, ca)
        assert result.shape == (4, 5)


class TestBatchEquivalence3D:
    def test_clock_rates_batch_matches_loop_in_3d(self) -> None:
        rng = np.random.default_rng(7)
        clock_array = ClockArray(
            positions=rng.uniform(-2, 2, size=(9, 3)), track_offset=0.0
        )
        mass_positions = rng.uniform(3, 8, size=(20, 3))
        masses = rng.uniform(0.05, 0.5, size=20)
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
