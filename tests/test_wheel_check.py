"""The wheel check's environment. The check itself installs a wheel, which is
CI's job (decision C12 of the commands plan)."""

import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "check_wheel_entrypoints", ROOT / "scripts" / "check_wheel_entrypoints.py"
)
assert _SPEC is not None and _SPEC.loader is not None
check = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check)


def test_the_bare_environment_reaches_no_checkout_and_no_encoder() -> None:
    """Only this catches the wheel check inheriting a way to the checkout or to
    an encoder: `PATH` beyond the environment's scripts, `PYTHONPATH`, or
    `IMAGEIO_FFMPEG_EXE`."""
    inherited = {
        "PATH": os.pathsep.join(["usr-bin", "opt-bin"]),
        "PYTHONPATH": "src",
        "IMAGEIO_FFMPEG_EXE": "opt-bin/ffmpeg",
        "HOME": "home",
        "LANG": "C",
    }
    before = dict(inherited)

    environment = check.bare_environment(Path("venv", "bin"), inherited)

    assert environment == {
        "PATH": str(Path("venv", "bin")),
        "HOME": "home",
        "LANG": "C",
    }
    assert inherited == before
