"""Slow acceptance replay for the rigorous SMC configuration.

Deterministic regression pin (same seeds + same code => same result), not
a population reliability estimate. Excluded from default runs; execute
with `uv run pytest -m slow`. Rerun whenever inference defaults change.
"""

import importlib.util
import inspect
from pathlib import Path
from types import ModuleType

import pytest

from clocks._scenarios import run_multi_mass_2d
from clocks.config import InferenceConfig
from clocks.inference import ParticleFilter

# Blocks 100, 200, and 300 are burned. Task 11 reserves 400-411 for the next
# one-shot certification and runs it only after freezing development choices.
CERTIFICATION_SEEDS = tuple(range(400, 412))

CERTIFIED_ESS_TARGET = 0.8
CERTIFIED_REJUVENATION_STEPS = 2
CERTIFIED_PROPOSAL_SCALE = 2.38


@pytest.mark.slow
def test_rigorous_defaults_pass_certification_replay() -> None:
    results = [run_multi_mass_2d(seed) for seed in CERTIFICATION_SEEDS]
    failed = [r["seed"] for r in results if not r["passed"]]
    assert len(CERTIFICATION_SEEDS) - len(failed) >= 10, (
        f"certification acceptance below 10/12; failing seeds: {failed}"
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
    for controls in (
        ([0.0], None, None),
        (None, [0], None),
        (None, None, [float("inf")]),
    ):
        with pytest.raises(ValueError, match="control"):
            scan._control_cells(0, *controls)


def _load_scan() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "scan_multi_mass_2d.py"
    spec = importlib.util.spec_from_file_location("scan_multi_mass_2d", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
