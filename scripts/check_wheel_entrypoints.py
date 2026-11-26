"""Smoke-test every console command from an installed wheel."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
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


_INHERITED_WAYS_IN = frozenset({"PATH", "PYTHONPATH", "IMAGEIO_FFMPEG_EXE"})


def bare_environment(
    scripts_directory: Path, inherited: Mapping[str, str]
) -> dict[str, str]:
    """``inherited`` without ``PYTHONPATH`` or ``IMAGEIO_FFMPEG_EXE``, and with
    ``PATH`` the environment's scripts directory alone, so neither a checkout
    nor an encoder can be reached by inheritance (K7 of the commands plan)."""
    environment = {k: v for k, v in inherited.items() if k not in _INHERITED_WAYS_IN}
    environment["PATH"] = str(scripts_directory)
    return environment


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


def _run_demo(
    command: Sequence[str], directory: Path, environment: Mapping[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        cwd=directory,
        env=dict(environment),
    )


def _demo_failure(
    command: Sequence[str], result: subprocess.CompletedProcess[str], why: str
) -> RuntimeError:
    return RuntimeError(
        f"{why}: {' '.join(command)}\nstatus: {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def _check_the_commands_without_an_encoder(
    scripts_directory: Path, suffix: str
) -> None:
    executable = str(scripts_directory / f"demo-1d{suffix}")
    command = (executable, "--output-dir", "out", "--observations", "2")
    command += ("--particles", "16")
    environment = bare_environment(scripts_directory, os.environ)
    wanted = {"demo_1d.npz", "demo_1d.png"}
    with tempfile.TemporaryDirectory(prefix="clocks-wheel-bare-") as directory:
        work = Path(directory)
        output = work / "out"

        result = _run_demo(command, work, environment)
        if result.returncode != 0:
            raise _demo_failure(command, result, "demo-1d failed with no encoder")
        if {p.name for p in output.iterdir()} != wanted:
            raise _demo_failure(command, result, f"out/ does not hold exactly {wanted}")
        print("ok: demo-1d writes its archive and PNG with no encoder")
        before = {name: (output / name).read_bytes() for name in wanted}

        video = (*command, "--video")
        result = _run_demo(video, work, environment)
        if result.returncode != 2 or "imageio-ffmpeg" not in result.stderr:
            raise _demo_failure(
                video,
                result,
                "--video did not stop with status 2 naming imageio-ffmpeg",
            )
        after = {p.name: p.read_bytes() for p in output.iterdir()}
        if after != before:
            raise _demo_failure(video, result, "--video changed the output directory")
        print(
            "ok: demo-1d --video stops and names imageio-ffmpeg "
            "when there is no encoder"
        )


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
        _check_the_commands_without_an_encoder(scripts_directory, suffix)
    return 0


if __name__ == "__main__":
    main()
