"""Smoke-test every console command from an installed wheel."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import zipfile
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

REQUIRED_WHEEL_FILES = (
    "clocks/_fonts/SourceSans3-Regular.ttf",
    "clocks/_fonts/SourceSans3-Semibold.ttf",
    "clocks/_fonts/LICENSE.md",
)
LICENSE_FILES = ("LICENSE", "src/clocks/_fonts/LICENSE.md")
LICENSE_EXPRESSION = "MIT AND OFL-1.1-RFN"
STYLE_SMOKE = (
    "from clocks.viz import style; "
    "cm = style('light'); cm.__enter__(); cm.__exit__(None, None, None)"
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


def _check_wheel_contents(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        missing = [name for name in REQUIRED_WHEEL_FILES if name not in names]
        if missing:
            raise RuntimeError(f"wheel {wheel.name} is missing files: {missing}")
        metadata_names = [n for n in names if n.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise RuntimeError(f"expected one METADATA in {wheel.name}")
        dist_info = metadata_names[0].removesuffix("/METADATA")
        missing = [
            name
            for name in LICENSE_FILES
            if f"{dist_info}/licenses/{name}" not in names
        ]
        if missing:
            raise RuntimeError(
                f"wheel {wheel.name} lacks licence files under "
                f"{dist_info}/licenses/: {missing}"
            )
        metadata = archive.read(metadata_names[0]).decode()
    lines = set(metadata.splitlines())
    wanted = [f"License-Expression: {LICENSE_EXPRESSION}"]
    wanted += [f"License-File: {name}" for name in LICENSE_FILES]
    absent = [line for line in wanted if line not in lines]
    if absent:
        raise RuntimeError(f"METADATA of {wheel.name} lacks lines: {absent}")
    print("ok: wheel ships the fonts and both licences")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    wheel = args.wheel.resolve()
    if not wheel.is_file():
        raise FileNotFoundError(f"wheel does not exist: {wheel}")
    _check_wheel_contents(wheel)
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
        _run((str(scripts_directory / python_name), "-c", STYLE_SMOKE))
        print("ok: style('light') enters and exits")
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
