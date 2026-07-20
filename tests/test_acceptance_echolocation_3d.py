"""Slow acceptance pin: echolocation scenario on certification seeds.

Deterministic re-execution of the certified runs (same seeds + same code
=> same result) — a regression pin, not a re-certification and not a
population reliability estimate. The exactly-once rule (spec section 3a)
bars using these seeds for tuning or selection; re-executing the frozen
configuration is permitted, exactly as test_acceptance_multi_mass_2d.py
re-executes its holdout. Excluded from default runs; execute with
`uv run pytest -m slow`. Rerun whenever inference defaults or the
scenario change.
"""

import inspect
import statistics

import numpy as np
import pytest

from clocks._scenarios import (
    ECHO_DIRECTION,
    ECHO_M_TRUE,
    ECHO_MASS_RANGE,
    ECHO_N_OBSERVATIONS,
    ECHO_N_PARTICLES,
    ECHO_NOISE_STD,
    ECHO_PASS_MASS_TOL,
    ECHO_PASS_POS_TOL,
    ECHO_POSITION_HALFWIDTH,
    ECHO_SWEEP_RANGES,
    build_echolocation_filter,
    run_echolocation_3d,
)

# Certified configuration, pinned as LITERALS (frozen in Task 9): the pin
# must not follow later edits to the live scenario constants. The fast
# guard below asserts the live constants still match, so drift fails
# loudly instead of silently redefining the certified run. On a burned
# block (spec section 3a), update CERT_SEED_BLOCK — nothing else here
# encodes the block.
CERT_SEED_BLOCK = 300
CERT_SEEDS = tuple(range(CERT_SEED_BLOCK, CERT_SEED_BLOCK + 12))
CERT_CLOSE_RANGE = 2.0
CERT_FAR_RANGE = 8.0
CERT_POS_TOL = 1.0
CERT_MASS_TOL = 0.075
CERT_SWEEP_RANGES = (2.0, 2.6, 3.5, 4.6, 6.1, 8.0)
CERT_M_TRUE = 0.15
CERT_NOISE_STD = 0.005
CERT_N_PARTICLES = 6000
CERT_N_OBSERVATIONS = 80
CERT_MASS_RANGE = (0.05, 2.0)
CERT_POSITION_HALFWIDTH = 16.0
CERT_DIRECTION = (2.0 / 7.0, 3.0 / 7.0, 6.0 / 7.0)
# Far-range posterior std must be at least this multiple of close-range.
ECHO_FAR_STD_FACTOR = 7.6


def _cert_passed(result: dict) -> bool:
    """Pass gate against the CERTIFIED tolerances (not the live ones)."""
    return (
        result["position_error"] <= CERT_POS_TOL
        and result["mass_error"] <= CERT_MASS_TOL
    )


@pytest.mark.slow
def test_certified_gates_hold_on_certification_seeds() -> None:
    """Both certified gates in one pass so each run executes exactly once."""
    close = [run_echolocation_3d(seed, CERT_CLOSE_RANGE) for seed in CERT_SEEDS]
    far = [run_echolocation_3d(seed, CERT_FAR_RANGE) for seed in CERT_SEEDS]

    failed = [r["seed"] for r in close if not _cert_passed(r)]
    assert len(CERT_SEEDS) - len(failed) >= 10, (
        f"close-range acceptance below 10/12; failing seeds: {failed}"
    )

    med_close = statistics.median(r["pos_std"] for r in close)
    med_far = statistics.median(r["pos_std"] for r in far)
    assert med_far >= ECHO_FAR_STD_FACTOR * med_close, (
        f"far-range posterior std {med_far:.3f} not >= "
        f"{ECHO_FAR_STD_FACTOR}x close-range {med_close:.3f}"
    )


def test_scenario_matches_certified_configuration() -> None:
    """Fast guard: live scenario constants equal the certified pins."""
    params = inspect.signature(run_echolocation_3d).parameters
    assert params["n_particles"].default == ECHO_N_PARTICLES
    assert params["n_observations"].default == ECHO_N_OBSERVATIONS
    assert ECHO_N_PARTICLES == CERT_N_PARTICLES
    assert ECHO_N_OBSERVATIONS == CERT_N_OBSERVATIONS
    assert ECHO_SWEEP_RANGES == CERT_SWEEP_RANGES
    assert ECHO_SWEEP_RANGES[0] == CERT_CLOSE_RANGE
    assert ECHO_SWEEP_RANGES[-1] == CERT_FAR_RANGE
    assert ECHO_PASS_POS_TOL == CERT_POS_TOL
    assert ECHO_PASS_MASS_TOL == CERT_MASS_TOL
    assert ECHO_M_TRUE == CERT_M_TRUE
    assert ECHO_NOISE_STD == CERT_NOISE_STD
    assert ECHO_MASS_RANGE == CERT_MASS_RANGE
    assert ECHO_POSITION_HALFWIDTH == CERT_POSITION_HALFWIDTH
    assert np.allclose(ECHO_DIRECTION, CERT_DIRECTION)


def test_filter_construction_matches_certified_configuration() -> None:
    """Fast guard: the built filter is certified too, not just constants."""
    pf = build_echolocation_filter(seed=0, n_particles=10)
    assert pf.jitter == "annealed"
    assert pf.jitter_tau == 15.0
    assert pf.jitter_std == 0.02
    assert pf.noise_std == CERT_NOISE_STD
    assert pf.support_bounds is not None
    lower, upper = pf.support_bounds
    hw = CERT_POSITION_HALFWIDTH
    assert np.allclose(lower, [-hw, -hw, -hw, CERT_MASS_RANGE[0]])
    assert np.allclose(upper, [hw, hw, hw, CERT_MASS_RANGE[1]])
