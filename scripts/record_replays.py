"""Record the packaged demos as replay archives and report how they compare.

Writes ``<out>/<demo>.npz`` for each requested demo (all seven by default).
With ``--compare DIR`` it also compares each fresh recording in full with
``DIR/<demo>.npz`` and prints a short report, writing it to ``--report FILE``
as well when given. The exit status is 0 when nothing was compared or every
compared demo reproduced, and 1 otherwise; a workflow that only wants the
report ignores it.

    uv run python scripts/record_replays.py --out assets/replays
    uv run python scripts/record_replays.py --out recorded \
        --compare assets/replays --report replay-report.md demo_1d
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from clocks._replay import (
    DEMOS,
    Replay,
    first_difference,
    load_replay,
    provenance,
    save_replay,
)


def record_demo(name: str) -> Replay:
    """Run the packaged inference of the named demo and return its replay."""
    return importlib.import_module(f"clocks._demos.{name}").record()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    names = args.demos or DEMOS
    recorded = {name: _record_and_save(name, args.out) for name in names}
    if args.compare is None:
        outcomes: list[tuple[str, str]] = []
    else:
        outcomes = [
            (name, _compare_one(name, args.compare, recorded[name])) for name in names
        ]
    _emit(_report_text(outcomes), args.report)
    return 0 if all(line == "reproduced" for _, line in outcomes) else 1


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="directory to write")
    parser.add_argument(
        "--compare", type=Path, help="directory of committed archives to compare with"
    )
    parser.add_argument("--report", type=Path, help="file to write the report to")
    # No default: Python 3.12 rejects a tuple default as one invalid choice.
    parser.add_argument("demos", nargs="*", choices=DEMOS, metavar="demo")
    return parser.parse_args(argv)


def _record_and_save(name: str, out: Path) -> Replay:
    replay = record_demo(name)
    out.mkdir(parents=True, exist_ok=True)
    save_replay(out / f"{name}.npz", replay)
    return replay


def _compare_one(name: str, committed_dir: Path, recorded: Replay) -> str:
    """One demo's outcome: the text after ``<name>: `` in the report."""
    path = committed_dir / f"{name}.npz"
    if not path.exists():
        return "no committed archive"
    try:
        committed = load_replay(path)
    except ValueError as error:
        return f"committed archive unreadable: {error}"
    difference = first_difference(committed, recorded)
    if difference is None:
        return "reproduced"
    text = f"differs at {difference.key}"
    if difference.row is not None:
        text += f", row {difference.row}"
    if not np.isnan(difference.largest):
        text += f" (largest difference {difference.largest:.3g})"
    return text


def _report_text(outcomes: Sequence[tuple[str, str]]) -> str:
    lines = [
        f"run: {os.environ.get('GITHUB_RUN_ID', 'local')}",
        f"head: {os.environ.get('REPLAY_HEAD_SHA', 'unknown')}",
        f"checkout: {_checkout_sha()}",
        *(f"{key}: {value}" for key, value in provenance().items()),
        f"blas: {_blas()}",
        *(f"{name}: {outcome}" for name, outcome in outcomes),
    ]
    return "\n".join(lines) + "\n"


def _checkout_sha() -> str:
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return done.stdout.strip() or "unknown"


def _blas() -> str:
    blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
    return f"{blas['name']} {blas['version']}"


def _emit(text: str, report: Path | None) -> None:
    if report is not None:
        report.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    raise SystemExit(main())
