"""Slow acceptance scan: annealed-jitter defaults on the holdout seeds.

Deterministic regression pin (same seeds + same code => same result), not
a population reliability estimate. Excluded from default runs; execute
with `uv run pytest -m slow`. Rerun whenever inference defaults change.
"""

import pytest

from clocks._scenarios import run_multi_mass_2d

# Seeds 100-111 are burned (diagnostics only: they exposed the clone-freeze
# degeneracy, see docs/superpowers/specs/2026-07-03-clone-freeze-diagnosis.md).
# Certification now uses 200-211 per the retry protocol.
HOLDOUT_SEEDS = tuple(range(200, 212))


@pytest.mark.slow
def test_annealed_defaults_pass_holdout_scan() -> None:
    results = [run_multi_mass_2d(seed) for seed in HOLDOUT_SEEDS]
    failed = [r["seed"] for r in results if not r["passed"]]
    assert len(HOLDOUT_SEEDS) - len(failed) >= 10, (
        f"holdout acceptance below 10/12; failing seeds: {failed}"
    )
