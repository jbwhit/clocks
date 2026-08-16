"""Contracts for packaged demo commands."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

DEMO_CASES = [
    ("demo_1d", "demo-1d", Path("output/demo_1d.gif")),
    ("demo_2d", "demo-2d", Path("output/demo_2d.gif")),
    ("demo_multi_mass", "demo-multi-mass", Path("output/demo_multi_mass.gif")),
    (
        "demo_multi_mass_2d",
        "demo-multi-mass-2d",
        Path("output/demo_multi_mass_2d.gif"),
    ),
    (
        "demo_model_comparison",
        "demo-model-comparison",
        Path("output/demo_model_comparison.gif"),
    ),
    ("demo_density", "demo-density", Path("output/demo_density.png")),
    (
        "demo_echolocation_3d",
        "demo-echolocation-3d",
        Path("output/demo_echolocation_3d.gif"),
    ),
]


def _module(name: str):
    return importlib.import_module(f"clocks._demos.{name}")


@pytest.mark.parametrize("module_name,executable,default_output", DEMO_CASES)
def test_demo_help_exits_without_running(
    module_name: str,
    executable: str,
    default_output: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    del executable, default_output
    module = _module(module_name)

    def fail_if_run(*args, **kwargs):
        raise AssertionError("--help must exit before starting the demo")

    monkeypatch.setattr(module, "_run_demo", fail_if_run)
    with pytest.raises(SystemExit) as exc:
        module.main(["--help"])

    assert exc.value.code == 0
    assert "usage:" in capsys.readouterr().out


@pytest.mark.parametrize("module_name,executable,default_output", DEMO_CASES)
def test_demo_parser_preserves_defaults_and_accepts_fast_overrides(
    module_name: str,
    executable: str,
    default_output: Path,
    tmp_path: Path,
) -> None:
    del executable
    parser = _module(module_name).build_parser()
    defaults = parser.parse_args([])
    assert defaults.output == default_output
    assert defaults.observations > 1
    assert defaults.particles > 16

    requested_output = tmp_path / default_output.name
    args = parser.parse_args(
        [
            "--output",
            str(requested_output),
            "--observations",
            "1",
            "--particles",
            "16",
        ]
    )
    assert args.output == requested_output
    assert args.observations == 1
    assert args.particles == 16


@pytest.mark.parametrize("module_name,executable,default_output", DEMO_CASES)
def test_demo_main_executes_with_fast_overrides(
    module_name: str,
    executable: str,
    default_output: Path,
    tmp_path: Path,
) -> None:
    del executable
    module = _module(module_name)
    requested_output = tmp_path / default_output.name
    assert (
        module.main(
            [
                "--output",
                str(requested_output),
                "--observations",
                "1",
                "--particles",
                "16",
            ]
        )
        == 0
    )
    assert requested_output.stat().st_size > 0


def test_project_scripts_point_directly_to_packaged_demo_modules() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    expected = {
        executable: f"clocks._demos.{module_name}:main"
        for module_name, executable, _ in DEMO_CASES
    }
    assert pyproject["project"]["scripts"] == expected


@pytest.mark.parametrize("module_name,executable,default_output", DEMO_CASES)
def test_demo_selects_agg_before_importing_pyplot(
    module_name: str, executable: str, default_output: Path
) -> None:
    del executable, default_output
    check = """
import importlib.abc
import sys

class PyplotGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != "matplotlib.pyplot":
            return None
        matplotlib = sys.modules.get("matplotlib")
        if matplotlib is None or str(matplotlib.get_backend()).lower() != "agg":
            raise RuntimeError("matplotlib.pyplot imported before selecting Agg")
        sys.meta_path.remove(self)
        return None

sys.meta_path.insert(0, PyplotGuard())
__import__(sys.argv[1])
"""
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "svg"
    result = subprocess.run(
        [sys.executable, "-c", check, f"clocks._demos.{module_name}"],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("module_name,executable,default_output", DEMO_CASES)
def test_source_demo_is_only_a_thin_wrapper(
    module_name: str, executable: str, default_output: Path
) -> None:
    del executable, default_output
    source = Path("scripts", f"{module_name}.py").read_text()
    assert source == (
        f"from clocks._demos.{module_name} import main\n\n"
        'if __name__ == "__main__":\n'
        "    main()\n"
    )
