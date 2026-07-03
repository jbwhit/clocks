"""Fast tests for the shared multi-mass-2D scenario module."""

import numpy as np

from clocks._scenarios import (
    MIN_SEPARATION,
    PASS_TOLERANCE,
    TRUTH,
    generate_random_clocks,
    passes,
    run_multi_mass_2d,
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
