"""Slow acceptance scan: annealed-jitter defaults on the holdout seeds.

Deterministic regression pin (same seeds + same code => same result), not
a population reliability estimate. Excluded from default runs; execute
with `uv run pytest -m slow`. Rerun whenever inference defaults change.
"""

import inspect

import pytest

from clocks._scenarios import run_multi_mass_2d
from clocks.config import InferenceConfig
from clocks.inference import ParticleFilter

# Seeds 100-111 are burned (diagnostics only: they exposed the clone-freeze
# degeneracy, see docs/superpowers/specs/2026-07-03-clone-freeze-diagnosis.md).
# Certification now uses 200-211 per the retry protocol.
HOLDOUT_SEEDS = tuple(range(200, 212))

CERTIFIED_ESS_TARGET = 0.8
CERTIFIED_REJUVENATION_STEPS = 2
CERTIFIED_PROPOSAL_SCALE = 2.38


@pytest.mark.slow
def test_annealed_defaults_pass_holdout_scan() -> None:
    results = [run_multi_mass_2d(seed) for seed in HOLDOUT_SEEDS]
    failed = [r["seed"] for r in results if not r["passed"]]
    assert len(HOLDOUT_SEEDS) - len(failed) >= 10, (
        f"holdout acceptance below 10/12; failing seeds: {failed}"
    )


def test_shipped_defaults_match_rigorous_smc_configuration() -> None:
    """Fast guard for the declared adaptive resample-move defaults."""
    assert InferenceConfig.__dataclass_fields__["ess_target"].default == 0.8
    assert InferenceConfig.__dataclass_fields__["rejuvenation_steps"].default == 2
    assert InferenceConfig.__dataclass_fields__["proposal_scale"].default == 2.38
    params = inspect.signature(ParticleFilter.__init__).parameters
    assert params["ess_target"].default == CERTIFIED_ESS_TARGET
    assert params["rejuvenation_steps"].default == CERTIFIED_REJUVENATION_STEPS
    assert params["proposal_scale"].default == CERTIFIED_PROPOSAL_SCALE
    runner = inspect.signature(run_multi_mass_2d).parameters
    assert runner["ess_target"].default == CERTIFIED_ESS_TARGET
    assert runner["rejuvenation_steps"].default == CERTIFIED_REJUVENATION_STEPS
    assert runner["proposal_scale"].default == CERTIFIED_PROPOSAL_SCALE
