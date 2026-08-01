"""Slow acceptance scan: annealed-jitter defaults on the holdout seeds.

Deterministic regression pin (same seeds + same code => same result), not
a population reliability estimate. Excluded from default runs; execute
with `uv run pytest -m slow`. Rerun whenever inference defaults change.
"""

import inspect

import pytest

from clocks._scenarios import run_multi_mass_2d
from clocks.config import InferenceConfig
from clocks.inference import ModelComparison, ParticleFilter

# Seeds 100-111 are burned (diagnostics only: they exposed the clone-freeze
# degeneracy, see docs/superpowers/specs/2026-07-03-clone-freeze-diagnosis.md).
# Certification now uses 200-211 per the retry protocol.
HOLDOUT_SEEDS = tuple(range(200, 212))

CERTIFIED_TAU = 15.0  # scan winner (Task 7B certification, seeds 200-211)
CERTIFIED_FLOOR = 0.02  # scan winner floor


@pytest.mark.slow
def test_annealed_defaults_pass_holdout_scan() -> None:
    results = [run_multi_mass_2d(seed) for seed in HOLDOUT_SEEDS]
    failed = [r["seed"] for r in results if not r["passed"]]
    assert len(HOLDOUT_SEEDS) - len(failed) >= 10, (
        f"holdout acceptance below 10/12; failing seeds: {failed}"
    )


def test_shipped_defaults_match_certified_cell() -> None:
    """Fast guard: every shipped jitter_tau and jitter_std default equals
    the certified scan winner (spec §3). Runs in regular CI (not marked
    slow)."""
    tau_field = InferenceConfig.__dataclass_fields__["jitter_tau"]
    assert tau_field.default == CERTIFIED_TAU
    floor_field = InferenceConfig.__dataclass_fields__["jitter_std"]
    assert floor_field.default == CERTIFIED_FLOOR
    for fn in (ParticleFilter.__init__, ModelComparison.__init__):
        params = inspect.signature(fn).parameters
        assert params["jitter_tau"].default == CERTIFIED_TAU
        assert params["jitter_std"].default == CERTIFIED_FLOOR
    runner = inspect.signature(run_multi_mass_2d).parameters
    assert runner["jitter_tau"].default == CERTIFIED_TAU
    assert runner["jitter_std"].default == CERTIFIED_FLOOR
