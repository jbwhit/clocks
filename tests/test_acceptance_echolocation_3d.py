"""Echolocation acceptance guards before one-shot certification exists.

Task 11 will install the deterministic slow replay only after development
calibration is frozen and the protected block has been executed exactly once.
"""

import inspect

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

EXPECTED_CLOSE_RANGE = 2.0
EXPECTED_FAR_RANGE = 8.0
EXPECTED_POS_TOL = 1.0
EXPECTED_MASS_TOL = 0.04
EXPECTED_SWEEP_RANGES = (2.0, 2.6, 3.5, 4.6, 6.1, 8.0)
EXPECTED_M_TRUE = 0.080
EXPECTED_NOISE_STD = 0.001
EXPECTED_N_PARTICLES = 6000
EXPECTED_N_OBSERVATIONS = 80
EXPECTED_MASS_RANGE = (0.005, 0.15)
EXPECTED_POSITION_HALFWIDTH = 16.0
EXPECTED_DIRECTION = (2.0 / 7.0, 3.0 / 7.0, 6.0 / 7.0)


@pytest.mark.slow
@pytest.mark.skip(reason="Task 11 installs this replay after one-shot certification")
def test_certification_replay_pending_task_11() -> None:
    """No protected seed values belong in runnable tests before Task 11."""


def test_scenario_matches_declared_configuration() -> None:
    """Fast guard: live scenario constants equal Task 8's declared values."""
    params = inspect.signature(run_echolocation_3d).parameters
    assert params["n_particles"].default == ECHO_N_PARTICLES
    assert params["n_observations"].default == ECHO_N_OBSERVATIONS
    assert ECHO_N_PARTICLES == EXPECTED_N_PARTICLES
    assert ECHO_N_OBSERVATIONS == EXPECTED_N_OBSERVATIONS
    assert ECHO_SWEEP_RANGES == EXPECTED_SWEEP_RANGES
    assert ECHO_SWEEP_RANGES[0] == EXPECTED_CLOSE_RANGE
    assert ECHO_SWEEP_RANGES[-1] == EXPECTED_FAR_RANGE
    assert ECHO_PASS_POS_TOL == EXPECTED_POS_TOL
    assert ECHO_PASS_MASS_TOL == EXPECTED_MASS_TOL
    assert ECHO_M_TRUE == EXPECTED_M_TRUE
    assert ECHO_NOISE_STD == EXPECTED_NOISE_STD
    assert ECHO_MASS_RANGE == EXPECTED_MASS_RANGE
    assert ECHO_POSITION_HALFWIDTH == EXPECTED_POSITION_HALFWIDTH
    assert np.allclose(ECHO_DIRECTION, EXPECTED_DIRECTION)


def test_filter_construction_matches_declared_configuration() -> None:
    """Fast guard: the built filter uses Task 8's declared defaults."""
    pf = build_echolocation_filter(seed=0, n_particles=10)
    assert pf.ess_target == 0.8
    assert pf.rejuvenation_steps == 2
    assert pf.proposal_scale == 2.38
    assert pf.noise_std == EXPECTED_NOISE_STD
    assert np.all(np.isfinite(pf.log_prior_density(pf.state.particles)))


def test_run_and_filter_accept_only_rigorous_smc_controls() -> None:
    for callable_ in (build_echolocation_filter, run_echolocation_3d):
        params = inspect.signature(callable_).parameters
        assert "jitter" not in params
        assert params["ess_target"].default == 0.8
        assert params["rejuvenation_steps"].default == 2
        assert params["proposal_scale"].default == 2.38
