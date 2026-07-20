"""Entry points for demo scripts."""

import importlib
import runpy
from pathlib import Path


def _run_script(name: str) -> None:
    """Run a demo script by name from the scripts/ directory."""
    # Find the project root (where scripts/ lives) relative to this package
    script = Path(__file__).resolve().parents[2] / "scripts" / name
    if script.exists():
        runpy.run_path(str(script), run_name="__main__")
    else:
        # Fallback: import as module if installed without scripts/
        importlib.import_module(f"scripts.{Path(name).stem}")


def demo_1d() -> None:
    _run_script("demo_1d.py")


def demo_2d() -> None:
    _run_script("demo_2d.py")


def demo_multi_mass() -> None:
    _run_script("demo_multi_mass.py")


def demo_model_comparison() -> None:
    _run_script("demo_model_comparison.py")


def demo_multi_mass_2d() -> None:
    _run_script("demo_multi_mass_2d.py")


def demo_density() -> None:
    _run_script("demo_density.py")


def demo_echolocation_3d() -> None:
    _run_script("demo_echolocation_3d.py")
