"""Record the packaged demos as replay archives and report how they compare.

Writes ``<out>/<demo>.npz`` for each requested demo (all seven by default).
With ``--compare DIR`` it also compares each fresh recording in full with
``DIR/<demo>.npz`` and prints a short report, writing it to ``--report FILE``
as well when given; ``DIR`` must not be the output directory. A demo whose
recording fails gets a ``recording failed`` line, with or without
``--compare``, one whose comparison fails a ``comparison failed`` line, and
the others still run. The exit status
is 0 when every demo was recorded and every compared demo reproduced, and 1
otherwise; a workflow that only wants the report ignores it.

    uv run python scripts/record_replays.py --out assets/replays
    uv run python scripts/record_replays.py --out recorded \
        --compare assets/replays --report replay-report.md demo_1d
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import tempfile
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
    outcomes: list[tuple[str, str]] = []
    for name in args.demos or DEMOS:
        outcome = _record_and_compare(name, args.out, args.compare)
        if outcome is not None:
            outcomes.append((name, outcome))
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
    args = parser.parse_args(argv)
    collision = _path_collision(args, args.demos or DEMOS)
    if collision is not None:
        parser.error(collision)
    return args


def _path_collision(args: argparse.Namespace, names: Sequence[str]) -> str | None:
    """Say which two arguments name one file or directory, or return None.

    Recording saves before comparing and the report is renamed into place last,
    so a baseline that is also an output, or a report that is an archive, would
    be overwritten and then compared with or lost.
    """
    if args.compare is not None and _same_file(args.out, args.compare):
        return (
            "the comparison directory must differ from the output directory, "
            f"got --out {args.out} and --compare {args.compare}"
        )
    groups = {"--out": [args.out / f"{name}.npz" for name in names]}
    if args.compare is not None:
        groups["--compare"] = [args.compare / f"{name}.npz" for name in names]
    if args.report is not None:
        groups["--report"] = [args.report]
    labels = list(groups)
    for i, first in enumerate(labels):
        for second in labels[i + 1 :]:
            for a in groups[first]:
                for b in groups[second]:
                    if _same_file(a, b):
                        return f"{first} and {second} name the same file: {a} and {b}"
    return None


def _same_file(a: Path, b: Path) -> bool:
    """Whether two paths name one file or directory, now or once written.

    When both exist, ``samefile`` sees through symlinks, hard links and
    case-variant spellings. When one does not exist yet, its parent is compared
    by ``samefile`` and its name with ``casefold()``: this refuses a few
    legitimate pairs on a case-sensitive filesystem (names differing only in
    case) rather than let a case-insensitive one through.
    """
    a, b = a.resolve(), b.resolve()
    if a == b:
        return True
    if a.exists() and b.exists():
        return a.samefile(b)
    return (
        a.parent.exists()
        and b.parent.exists()
        and a.parent.samefile(b.parent)
        and a.name.casefold() == b.name.casefold()
    )


def _record_and_compare(name: str, out: Path, committed_dir: Path | None) -> str | None:
    """One demo's report line after ``<name>: ``, or None when there is none.

    A demo whose recording or comparison fails is reported, and the others
    still run.
    """
    try:
        recorded = _record_and_save(name, out)
    # The report must be written when its surroundings are broken; this is the
    # outermost per-demo boundary, and the error is reported and fails the run.
    except Exception as error:
        return f"recording failed: {type(error).__name__}: {error}"
    if committed_dir is None:
        return None
    try:
        return _compare_one(name, committed_dir, recorded)
    # As above: reported with its type, counted as a failure, others carry on.
    except Exception as error:
        return f"comparison failed: {type(error).__name__}: {error}"


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
    """The commit of the checkout holding this script, wherever it is run from."""
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return done.stdout.strip() or "unknown"


def _blas() -> str:
    try:
        blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        return f"{blas['name']} {blas['version']}"
    except (KeyError, TypeError):  # a NumPy build that records no BLAS
        return "unknown"


def _emit(text: str, report: Path | None) -> None:
    """Print the report, then write it; printed first, a failed write loses nothing."""
    print(text, end="")
    if report is not None:
        _write_atomically(report, text)


def _write_atomically(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` whole or not at all.

    The text goes to a temporary file beside ``path`` that is then renamed over
    it, so a reader never sees a partial report.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(text)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)  # gone already once it has replaced path


if __name__ == "__main__":
    raise SystemExit(main())
