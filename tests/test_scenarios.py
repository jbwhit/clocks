"""Fast tests for the shared multi-mass-2D scenario module."""

import numpy as np
import pytest

from clocks._scenarios import (
    ECHO_DIRECTION,
    ECHO_M_TRUE,
    ECHO_MIN_RANGE_R,
    ECHO_R_HEAD,
    MIN_SEPARATION,
    PASS_TOLERANCE,
    TRUTH,
    build_head_lattice,
    echo_mass_config,
    echo_mass_position,
    generate_random_clocks,
    passes,
    run_multi_mass_2d,
    validate_echo_geometry,
)


class TestPassRule:
    def test_truth_passes(self) -> None:
        assert passes(TRUTH)

    def test_position_error_at_tolerance_passes(self) -> None:
        assert passes(TRUTH + np.array([0.5, 0, 0, 0, 0, 0]))

    def test_position_error_beyond_tolerance_fails(self) -> None:
        assert not passes(TRUTH + np.array([0.51, 0, 0, 0, 0, 0]))

    def test_mass_error_beyond_tolerance_fails(self) -> None:
        assert not passes(TRUTH + np.array([0, 0, 0, 0, 0.11, 0]))

    def test_tolerance_values(self) -> None:
        assert np.array_equal(PASS_TOLERANCE, np.array([0.5, 0.5, 0.5, 0.5, 0.1, 0.1]))


class TestClockPlacement:
    def test_respects_min_separation_and_exclusions(self) -> None:
        rng = np.random.default_rng(11)
        exclude = [(-3.0, 2.0), (4.0, -1.0)]
        clocks = generate_random_clocks(10, rng, exclude=exclude)
        assert clocks.shape == (10, 2)
        for i in range(10):
            for j in range(i + 1, 10):
                assert np.linalg.norm(clocks[i] - clocks[j]) >= MIN_SEPARATION
            for p in exclude:
                assert np.linalg.norm(clocks[i] - np.array(p)) >= MIN_SEPARATION


class TestFreezeRegression:
    def test_seed_101_does_not_freeze(self) -> None:
        """Seed 101 froze at t=1 under reject-and-stay (clone-freeze,
        docs/superpowers/specs/2026-07-03-clone-freeze-diagnosis.md).
        Under reflection it must keep a live posterior."""
        result = run_multi_mass_2d(101, jitter_std=0.02, jitter_tau=5.0)
        assert result["max_posterior_std"] > 1e-6


class TestEchoGeometry:
    def test_lattice_is_3x3x3_cube(self) -> None:
        head = build_head_lattice()
        assert head.positions.shape == (27, 3)
        assert head.track_offset == 0.0
        # Every coordinate is exactly -1, 0, or 1; all 27 cells distinct.
        assert set(np.unique(head.positions)) == {-1.0, 0.0, 1.0}
        assert len({tuple(p) for p in head.positions}) == 27

    def test_circumradius_is_sqrt_3(self) -> None:
        head = build_head_lattice()
        radii = np.linalg.norm(head.positions, axis=1)
        assert np.isclose(radii.max(), ECHO_R_HEAD)
        assert np.isclose(ECHO_R_HEAD, np.sqrt(3.0))

    def test_direction_is_unit_and_off_axis(self) -> None:
        assert np.isclose(np.linalg.norm(ECHO_DIRECTION), 1.0)
        # No zero component (off-axis) and components distinct (off-diagonal).
        assert np.all(np.abs(ECHO_DIRECTION) > 0.1)
        assert len(set(np.round(np.abs(ECHO_DIRECTION), 6))) == 3

    def test_mass_position_at_requested_range(self) -> None:
        pos = echo_mass_position(3.0)
        assert pos.shape == (3,)
        assert np.isclose(np.linalg.norm(pos), 3.0 * ECHO_R_HEAD)
        config = echo_mass_config(3.0)
        assert config.positions.shape == (1, 3)
        assert np.isclose(config.masses[0], ECHO_M_TRUE)

    def test_validate_rejects_interior_range(self) -> None:
        head = build_head_lattice()
        with pytest.raises(ValueError, match="exterior"):
            validate_echo_geometry(ECHO_MIN_RANGE_R - 0.5, ECHO_M_TRUE, head)

    def test_validate_rejects_weak_field_violation(self) -> None:
        head = build_head_lattice()
        # Heavy mass at minimum range: d_min ~ 2.0 < 10 * 0.5.
        with pytest.raises(ValueError, match="weak-field"):
            validate_echo_geometry(ECHO_MIN_RANGE_R, 0.5, head)

    def test_validate_accepts_shipped_defaults(self) -> None:
        head = build_head_lattice()
        validate_echo_geometry(ECHO_MIN_RANGE_R, ECHO_M_TRUE, head)
