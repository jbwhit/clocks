"""Frozen development guards before the one-shot certification exists.

Development calibration is frozen here before any protected seed is inspected.
Task 11 will install the deterministic slow replay only after the protected
block has been executed exactly once.
"""

import inspect

import numpy as np
import pytest

from clocks._scenarios import PASS_TOLERANCE, TRUTH, passes, run_multi_mass_2d
from clocks.config import InferenceConfig
from clocks.inference import ParticleFilter

EXPECTED_SCENARIO_ESS_TARGET = 0.7
EXPECTED_SCENARIO_REJUVENATION_STEPS = 2
EXPECTED_SCENARIO_PROPOSAL_SCALE = 3.0
EXPECTED_PASS_TOLERANCE = np.array([2.5, 2.5, 2.5, 2.5, 0.012, 0.012])


@pytest.mark.slow
@pytest.mark.skip(reason="Task 11 installs this replay after one-shot certification")
def test_certification_replay_pending_task_11() -> None:
    """Protected-seed outcomes are added only after their one-shot run."""


def test_core_defaults_remain_the_general_rigorous_smc_configuration() -> None:
    """Scenario calibration must not silently change general API defaults."""
    assert InferenceConfig.__dataclass_fields__["ess_target"].default == 0.8
    assert InferenceConfig.__dataclass_fields__["rejuvenation_steps"].default == 2
    assert InferenceConfig.__dataclass_fields__["proposal_scale"].default == 2.38
    params = inspect.signature(ParticleFilter.__init__).parameters
    assert params["ess_target"].default == 0.8
    assert params["rejuvenation_steps"].default == 2
    assert params["proposal_scale"].default == 2.38


def test_development_selected_scenario_defaults_are_frozen() -> None:
    """Literal guard for the cell selected without protected-seed evidence."""
    runner = inspect.signature(run_multi_mass_2d).parameters
    assert runner["ess_target"].default == EXPECTED_SCENARIO_ESS_TARGET
    assert runner["rejuvenation_steps"].default == EXPECTED_SCENARIO_REJUVENATION_STEPS
    assert runner["proposal_scale"].default == EXPECTED_SCENARIO_PROPOSAL_SCALE


def test_recovery_gate_is_one_uniform_position_and_mass_rule() -> None:
    """The gate is scientific regression evidence, not population reliability."""
    np.testing.assert_array_equal(PASS_TOLERANCE, EXPECTED_PASS_TOLERANCE)
    assert passes(TRUTH + EXPECTED_PASS_TOLERANCE)
    for index in range(len(TRUTH)):
        outside = TRUTH.copy()
        outside[index] += EXPECTED_PASS_TOLERANCE[index] + 1e-12
        assert not passes(outside)
