"""Tests for the gravitational time dilation forward model."""

import numpy as np

from clocks.physics import (
    clock_rates,
    clock_rates_batch,
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
