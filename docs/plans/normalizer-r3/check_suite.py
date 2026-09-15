# /// script
# dependencies = ["numpy", "scipy", "pytest"]
# ///
"""Run the actual package suite against the proposed normalizer and ESS split.

Only this interpreter's bindings and four known xfail marks change. No project
source or virtual environment is modified. Run with uv run --no-sync python.
"""

import importlib.util
import sys
from pathlib import Path
from types import FunctionType

import pytest
from mutations import MutationPlugin, apply_mutation

import clocks.inference as inference

ROOT = Path(__file__).resolve().parents[3]
assert Path(inference.__file__).resolve().is_relative_to(ROOT)
spec = importlib.util.spec_from_file_location(
    "candidate", Path(__file__).with_name("prototype.py")
)
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
apply_mutation(candidate)
search_globals = dict(inference._next_beta.__globals__)
search_globals["_normalize_log_weights"] = candidate.normalize_branch
inference._next_beta = FunctionType(
    inference._next_beta.__code__, search_globals, "_next_beta"
)
inference._normalize_log_weights = candidate.normalize_proto

UNXFAIL = {
    "test_normalization_keeps_a_weight_that_is_barely_representable",
    "test_subnormal_recovery_matches_exact_arithmetic_at_the_boundary[16-heads]",
    "test_subnormal_recovery_matches_exact_arithmetic_at_the_boundary[128-heads]",
    "test_subnormal_recovery_matches_exact_arithmetic_at_the_boundary[peak-3.6e12]",
}


class CandidatePlugin:
    def pytest_collection_modifyitems(self, items):
        removed = 0
        for item in items:
            if item.path.name == "test_smc_rigorous.py" and item.name in UNXFAIL:
                assert item.get_closest_marker("xfail") is not None
                item.own_markers = [m for m in item.own_markers if m.name != "xfail"]
                removed += 1
        # pytest's -m deselection runs after this hook.
        assert removed == 4, f"expected four known strict xfails, found {removed}"


if __name__ == "__main__":
    print("Verified package import:", inference.__file__, flush=True)
    raise SystemExit(
        pytest.main(
            sys.argv[1:] or ["--tb=short"],
            plugins=[CandidatePlugin(), MutationPlugin()],
        )
    )
