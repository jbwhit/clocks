"""Acceptance guards before the one-shot certification exists.

Task 11 will install the deterministic slow replay only after development
calibration is frozen and the protected block has been executed exactly once.
"""

import inspect

import pytest

from clocks._scenarios import run_multi_mass_2d
from clocks.config import InferenceConfig
from clocks.inference import ParticleFilter

EXPECTED_ESS_TARGET = 0.8
EXPECTED_REJUVENATION_STEPS = 2
EXPECTED_PROPOSAL_SCALE = 2.38


@pytest.mark.slow
@pytest.mark.skip(reason="Task 11 installs this replay after one-shot certification")
def test_certification_replay_pending_task_11() -> None:
    """No protected seed values belong in runnable tests before Task 11."""


def test_shipped_defaults_match_rigorous_smc_configuration() -> None:
    """Fast guard for the declared adaptive resample-move defaults."""
    assert InferenceConfig.__dataclass_fields__["ess_target"].default == 0.8
    assert InferenceConfig.__dataclass_fields__["rejuvenation_steps"].default == 2
    assert InferenceConfig.__dataclass_fields__["proposal_scale"].default == 2.38
    params = inspect.signature(ParticleFilter.__init__).parameters
    assert params["ess_target"].default == EXPECTED_ESS_TARGET
    assert params["rejuvenation_steps"].default == EXPECTED_REJUVENATION_STEPS
    assert params["proposal_scale"].default == EXPECTED_PROPOSAL_SCALE
    runner = inspect.signature(run_multi_mass_2d).parameters
    assert runner["ess_target"].default == EXPECTED_ESS_TARGET
    assert runner["rejuvenation_steps"].default == EXPECTED_REJUVENATION_STEPS
    assert runner["proposal_scale"].default == EXPECTED_PROPOSAL_SCALE
