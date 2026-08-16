"""Frozen guards and deterministic replay for the certified 2-D scenario."""

import inspect
from collections.abc import Mapping

import numpy as np
import pytest

from clocks._scenarios import PASS_TOLERANCE, TRUTH, run_multi_mass_2d
from clocks.config import InferenceConfig
from clocks.inference import ParticleFilter

EXPECTED_SCENARIO_ESS_TARGET = 0.7
EXPECTED_SCENARIO_REJUVENATION_STEPS = 2
EXPECTED_SCENARIO_PROPOSAL_SCALE = 3.0
EXPECTED_PASS_TOLERANCE = np.array([2.5, 2.5, 2.5, 2.5, 0.012, 0.012])
CERTIFICATION_SEEDS = (400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411)
CERT_TRUTH = (-3.0, 2.0, 4.0, -1.0, 0.050, 0.030)
CERT_TOLERANCE = (2.5, 2.5, 2.5, 2.5, 0.012, 0.012)


def _cert_passed(result: Mapping[str, object]) -> bool:
    mean = np.asarray(result["mean"], dtype=float)
    error = np.abs(mean - np.asarray(CERT_TRUTH))
    return bool(np.all(error <= np.asarray(CERT_TOLERANCE)))


@pytest.mark.slow
def test_certification_block_is_a_deterministic_regression_replay() -> None:
    """Replay the fixed block; this is not recertification or population evidence."""
    results = [
        run_multi_mass_2d(
            seed,
            ess_target=0.7,
            rejuvenation_steps=2,
            proposal_scale=3.0,
        )
        for seed in CERTIFICATION_SEEDS
    ]
    n_pass = sum(_cert_passed(result) for result in results)

    assert n_pass >= 10, "deterministic replay requires at least 10 of 12 passes"


def test_core_defaults_remain_the_general_rigorous_smc_configuration() -> None:
    """Scenario calibration must not silently change general API defaults."""
    assert InferenceConfig.__dataclass_fields__["ess_target"].default == 0.8
    assert InferenceConfig.__dataclass_fields__["rejuvenation_steps"].default == 2
    assert InferenceConfig.__dataclass_fields__["proposal_scale"].default == 2.38
    params = inspect.signature(ParticleFilter.__init__).parameters
    assert params["ess_target"].default == 0.8
    assert params["rejuvenation_steps"].default == 2
    assert params["proposal_scale"].default == 2.38


def test_certification_gate_recomputes_from_mean_not_returned_metrics() -> None:
    exact = {
        "mean": np.array(CERT_TRUTH),
        "passed": False,
        "max_abs_error": np.inf,
        "normalized_error": np.inf,
    }
    outside = {
        **exact,
        "mean": np.array(CERT_TRUTH) + np.array([2.5001, 0, 0, 0, 0, 0]),
        "passed": True,
        "max_abs_error": 0.0,
        "normalized_error": 0.0,
    }

    assert _cert_passed(exact)
    assert not _cert_passed(outside)


def test_development_selected_scenario_defaults_are_frozen() -> None:
    """Literal guard for the cell selected without protected-seed evidence."""
    runner = inspect.signature(run_multi_mass_2d).parameters
    assert runner["ess_target"].default == EXPECTED_SCENARIO_ESS_TARGET
    assert runner["rejuvenation_steps"].default == EXPECTED_SCENARIO_REJUVENATION_STEPS
    assert runner["proposal_scale"].default == EXPECTED_SCENARIO_PROPOSAL_SCALE


def test_recovery_gate_is_one_uniform_position_and_mass_rule() -> None:
    """The gate is scientific regression evidence, not population reliability."""
    np.testing.assert_array_equal(PASS_TOLERANCE, EXPECTED_PASS_TOLERANCE)
    np.testing.assert_array_equal(TRUTH, CERT_TRUTH)
    np.testing.assert_array_equal(PASS_TOLERANCE, CERT_TOLERANCE)
