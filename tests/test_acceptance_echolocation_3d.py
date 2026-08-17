"""Frozen guards and deterministic replay for certified echolocation."""

import inspect
import math
import statistics
from collections.abc import Mapping

import numpy as np
import pytest

import clocks._scenarios as scenarios
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
    ECHO_R_HEAD,
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
EXPECTED_ESS_TARGET = 0.9
EXPECTED_REJUVENATION_STEPS = 1
EXPECTED_PROPOSAL_SCALE = 1.5
EXPECTED_FAR_STD_FACTOR = 20.0
CERTIFICATION_SEEDS = (400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411)
CERT_DIRECTION = (2.0 / 7.0, 3.0 / 7.0, 6.0 / 7.0)
CERT_HEAD_CIRCUMRADIUS = math.sqrt(3.0)
CERT_M_TRUE = 0.080
CERT_CLOSE_RANGE = 2.0
CERT_FAR_RANGE = 8.0
CERT_POSITION_TOLERANCE = 1.0
CERT_MASS_TOLERANCE = 0.04
CERT_FAR_STD_FACTOR = 20.0


def _cert_metrics(
    result: Mapping[str, object], *, range_r: float
) -> tuple[float, float, float]:
    mean = np.asarray(result["mean"], dtype=float)
    std = np.asarray(result["std"], dtype=float)
    truth_position = np.asarray(CERT_DIRECTION) * range_r * CERT_HEAD_CIRCUMRADIUS
    position_error = float(np.linalg.norm(mean[:3] - truth_position))
    mass_error = float(abs(mean[3] - CERT_M_TRUE))
    position_std = float(np.linalg.norm(std[:3]))
    return position_error, mass_error, position_std


@pytest.mark.slow
def test_certification_endpoints_are_a_deterministic_regression_replay() -> None:
    """Replay fixed endpoints; this is not recertification or population evidence."""
    close = [
        run_echolocation_3d(
            seed,
            range_r=CERT_CLOSE_RANGE,
            ess_target=0.9,
            rejuvenation_steps=1,
            proposal_scale=1.5,
        )
        for seed in CERTIFICATION_SEEDS
    ]
    far = [
        run_echolocation_3d(
            seed,
            range_r=CERT_FAR_RANGE,
            ess_target=0.9,
            rejuvenation_steps=1,
            proposal_scale=1.5,
        )
        for seed in CERTIFICATION_SEEDS
    ]
    close_metrics = [
        _cert_metrics(result, range_r=CERT_CLOSE_RANGE) for result in close
    ]
    far_metrics = [_cert_metrics(result, range_r=CERT_FAR_RANGE) for result in far]
    close_passes = sum(
        position_error <= CERT_POSITION_TOLERANCE and mass_error <= CERT_MASS_TOLERANCE
        for position_error, mass_error, _ in close_metrics
    )
    close_median_std = statistics.median(item[2] for item in close_metrics)
    far_median_std = statistics.median(item[2] for item in far_metrics)

    assert close_passes >= 10, (
        "deterministic replay requires at least 10 of 12 close-range passes"
    )
    assert far_median_std >= CERT_FAR_STD_FACTOR * close_median_std


def test_scenario_matches_frozen_development_configuration() -> None:
    """Fast guard for constants fixed without protected-seed evidence."""
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
    assert scenarios.ECHO_FAR_STD_FACTOR == EXPECTED_FAR_STD_FACTOR
    assert ECHO_R_HEAD == pytest.approx(CERT_HEAD_CIRCUMRADIUS)
    assert ECHO_M_TRUE == CERT_M_TRUE
    assert ECHO_SWEEP_RANGES[0] == CERT_CLOSE_RANGE
    assert ECHO_SWEEP_RANGES[-1] == CERT_FAR_RANGE
    assert ECHO_PASS_POS_TOL == CERT_POSITION_TOLERANCE
    assert ECHO_PASS_MASS_TOL == CERT_MASS_TOLERANCE
    assert np.allclose(ECHO_DIRECTION, CERT_DIRECTION)
    assert scenarios.ECHO_FAR_STD_FACTOR == CERT_FAR_STD_FACTOR


def test_filter_construction_matches_declared_configuration() -> None:
    """Fast guard: the built filter uses the development-selected controls."""
    pf = build_echolocation_filter(seed=0, n_particles=10)
    assert pf.ess_target == EXPECTED_ESS_TARGET
    assert pf.rejuvenation_steps == EXPECTED_REJUVENATION_STEPS
    assert pf.proposal_scale == EXPECTED_PROPOSAL_SCALE
    assert pf.noise_std == EXPECTED_NOISE_STD
    assert np.all(np.isfinite(pf.log_prior_density(pf.state.particles)))


def test_certification_metrics_recompute_from_mean_and_std() -> None:
    truth_position = (
        np.array(CERT_DIRECTION) * CERT_CLOSE_RANGE * CERT_HEAD_CIRCUMRADIUS
    )
    result = {
        "mean": np.append(truth_position + np.array([0.3, 0.4, 0.0]), 0.09),
        "std": np.array([3.0, 4.0, 0.0, 0.03]),
        "passed": True,
        "position_error": 0.0,
        "mass_error": 0.0,
        "pos_std": 0.0,
    }

    position_error, mass_error, pos_std = _cert_metrics(
        result, range_r=CERT_CLOSE_RANGE
    )

    assert position_error == pytest.approx(0.5)
    assert mass_error == pytest.approx(0.01)
    assert pos_std == pytest.approx(5.0)


def test_run_and_filter_accept_only_rigorous_smc_controls() -> None:
    for callable_ in (build_echolocation_filter, run_echolocation_3d):
        params = inspect.signature(callable_).parameters
        assert "jitter" not in params
        assert params["ess_target"].default == EXPECTED_ESS_TARGET
        assert params["rejuvenation_steps"].default == EXPECTED_REJUVENATION_STEPS
        assert params["proposal_scale"].default == EXPECTED_PROPOSAL_SCALE


def test_arbitrary_case_api_keeps_seed_streams_and_controls_explicit() -> None:
    params = inspect.signature(scenarios.run_echolocation_case).parameters

    assert list(params) == [
        "truth_position",
        "truth_mass",
        "observation_seed",
        "inference_seed",
        "n_particles",
        "n_observations",
        "noise_std",
        "ess_target",
        "rejuvenation_steps",
        "proposal_scale",
    ]
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in params.values()
    )
    assert params["n_particles"].default == EXPECTED_N_PARTICLES
    assert params["n_observations"].default == EXPECTED_N_OBSERVATIONS
    assert params["noise_std"].default == EXPECTED_NOISE_STD
    assert params["ess_target"].default == EXPECTED_ESS_TARGET
    assert params["rejuvenation_steps"].default == EXPECTED_REJUVENATION_STEPS
    assert params["proposal_scale"].default == EXPECTED_PROPOSAL_SCALE
