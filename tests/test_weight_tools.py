"""Focused source-provenance checks for production normalizer tools."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "script_name", ["check_normalizer_mutations", "benchmark_normalizer"]
)
def test_tool_guards_accept_exact_production_module_paths(script_name: str) -> None:
    tool = _load_script(script_name)

    normalizer, inference = tool.verify_production_modules()

    assert (
        Path(normalizer.__file__).resolve()
        == ROOT / "src/clocks/_weight_normalization.py"
    )
    assert Path(inference.__file__).resolve() == ROOT / "src/clocks/inference.py"


@pytest.mark.parametrize(
    "script_name", ["check_normalizer_mutations", "benchmark_normalizer"]
)
def test_tool_guards_reject_nonproduction_normalizer_path(
    monkeypatch: pytest.MonkeyPatch, script_name: str
) -> None:
    tool = _load_script(script_name)
    import clocks._weight_normalization as normalizer

    monkeypatch.setattr(normalizer, "__file__", str(ROOT / "stale-normalizer.py"))

    with pytest.raises(RuntimeError, match="normalizer did not import"):
        tool.verify_production_modules()


@pytest.mark.parametrize(
    "script_name", ["check_normalizer_mutations", "benchmark_normalizer"]
)
def test_tool_guards_reject_nonproduction_inference_path(
    monkeypatch: pytest.MonkeyPatch, script_name: str
) -> None:
    tool = _load_script(script_name)
    import clocks.inference as inference

    monkeypatch.setattr(inference, "__file__", str(ROOT / "stale-inference.py"))

    with pytest.raises(RuntimeError, match="inference did not import"):
        tool.verify_production_modules()
