"""Smoke-test every console command from an installed wheel."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

EXECUTABLES = (
    "demo-1d",
    "demo-2d",
    "demo-multi-mass",
    "demo-multi-mass-2d",
    "demo-model-comparison",
    "demo-density",
    "demo-echolocation-3d",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path, help="wheel artifact to install")
    parser.add_argument("python", help="Python executable for the temporary venv")
    return parser


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        rendered = " ".join(command)
        raise RuntimeError(
            f"command failed ({result.returncode}): {rendered}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    wheel = args.wheel.resolve()
    if not wheel.is_file():
        raise FileNotFoundError(f"wheel does not exist: {wheel}")
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv executable was not found on PATH")

    with tempfile.TemporaryDirectory(prefix="clocks-wheel-smoke-") as directory:
        environment = Path(directory, "venv")
        _run((uv, "venv", "--python", args.python, str(environment)))
        scripts_directory = environment / ("Scripts" if os.name == "nt" else "bin")
        python_name = "python.exe" if os.name == "nt" else "python"
        _run(
            (
                uv,
                "pip",
                "install",
                "--python",
                str(scripts_directory / python_name),
                str(wheel),
            )
        )
        suffix = ".exe" if os.name == "nt" else ""
        for name in EXECUTABLES:
            result = _run((str(scripts_directory / f"{name}{suffix}"), "--help"))
            if "usage:" not in result.stdout:
                raise RuntimeError(
                    f"{name} --help did not print argparse usage to stdout:\n"
                    f"{result.stdout}"
                )
            print(f"ok: {name} --help")
    return 0


if __name__ == "__main__":
    main()
