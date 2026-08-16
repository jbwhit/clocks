"""Slow acceptance pin: echolocation scenario on certification seeds.

Deterministic re-execution of the frozen runs (same seeds + same code
=> same result) — a regression pin, not a re-certification and not a
population reliability estimate. The exactly-once rule (spec section 3a)
bars using these seeds for tuning or selection. Excluded from default runs;
execute with `uv run pytest -m slow`. Rerun whenever inference defaults or
the scenario changes.
"""

import importlib.util
import inspect
import statistics
from pathlib import Path
from types import ModuleType

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

# Certification configuration, pinned as LITERALS: the pin
# must not follow later edits to the live scenario constants. The fast
# guard below asserts the live constants still match, so drift fails
# loudly instead of silently redefining the certified run. On a burned
# block (spec section 3a), update CERT_SEED_BLOCK — nothing else here
# encodes the block.
CERT_SEED_BLOCK = 400
CERT_SEEDS = tuple(range(CERT_SEED_BLOCK, CERT_SEED_BLOCK + 12))
CERT_CLOSE_RANGE = 2.0
CERT_FAR_RANGE = 8.0
CERT_POS_TOL = 1.0
CERT_MASS_TOL = 0.04
CERT_SWEEP_RANGES = (2.0, 2.6, 3.5, 4.6, 6.1, 8.0)
CERT_M_TRUE = 0.080
CERT_NOISE_STD = 0.001
CERT_N_PARTICLES = 6000
CERT_N_OBSERVATIONS = 80
CERT_MASS_RANGE = (0.005, 0.15)
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
    assert pf.ess_target == 0.8
    assert pf.rejuvenation_steps == 2
    assert pf.proposal_scale == 2.38
    assert pf.noise_std == CERT_NOISE_STD
    assert np.all(np.isfinite(pf.log_prior_density(pf.state.particles)))


def test_run_and_filter_accept_only_rigorous_smc_controls() -> None:
    for callable_ in (build_echolocation_filter, run_echolocation_3d):
        params = inspect.signature(callable_).parameters
        assert "jitter" not in params
        assert params["ess_target"].default == 0.8
        assert params["rejuvenation_steps"].default == 2
        assert params["proposal_scale"].default == 2.38


def test_seed_block_policy_reserves_fresh_certification_blocks() -> None:
    scan = _load_scan()
    assert hasattr(scan, "_seeds_for_block")
    _seeds_for_block = scan._seeds_for_block
    assert _seeds_for_block(0) == tuple(range(12))
    assert _seeds_for_block(400) == tuple(range(400, 412))
    for burned_or_invalid in (-100, 200, 300, 401):
        with pytest.raises(ValueError, match="seed block"):
            _seeds_for_block(burned_or_invalid)


def test_development_grid_and_single_certification_cell() -> None:
    scan = _load_scan()
    assert hasattr(scan, "_control_cells")
    development = scan._control_cells(0, None, None, None)
    assert len(development) == 27
    assert {cell[0] for cell in development} == {0.7, 0.8, 0.9}
    assert {cell[1] for cell in development} == {1, 2, 4}
    assert {cell[2] for cell in development} == {1.5, 2.38, 3.0}
    assert scan._control_cells(400, None, None, None) == [(0.8, 2, 2.38)]
    with pytest.raises(ValueError, match="single control cell"):
        scan._control_cells(400, [0.7, 0.8], None, None)
    for controls in (([0.0], None, None), (None, [0], None), (None, None, [np.inf])):
        with pytest.raises(ValueError, match="control"):
            scan._control_cells(0, *controls)


def _load_scan() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "scan_echolocation_range.py"
    spec = importlib.util.spec_from_file_location("scan_echolocation_range", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
