"""Record the packaged demos as replay archives and report how they compare.

Writes ``<out>/<demo>.npz`` for each requested demo (all seven by default).
With ``--compare DIR`` it also compares each fresh recording in full with
``DIR/<demo>.npz``, every one of which is read before anything is written, and
prints a short report, writing it to ``--report FILE`` as well when given.
``DIR`` must not be the output directory, and ``FILE`` must not end in
``.npz``. Whatever the name, the report is never written over a file that
holds an archive (checked just before the rename, so an archive the run
itself saved through a link is protected too): the report is printed, one
line on standard error says it was not written, and the exit status is 1. A
demo whose
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
import sys
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from clocks._replay import (
    DEMOS,
    Difference,
    Replay,
    first_difference,
    load_replay,
    provenance,
    save_replay,
)


def record_demo(name: str) -> Replay:
    """Run the packaged inference of the named demo and return its replay."""
    return importlib.import_module(f"clocks._demos.{name}").record()


class _ArchiveInTheWay(Exception):
    """The report's destination holds a replay archive, which it must not replace."""


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    names = args.demos or DEMOS
    # Every baseline is read before anything is written, and each comparison is
    # with the recording in memory, so no spelling of a path can make the run
    # compare against its own output. (Two outputs the user has linked to one
    # file hold whatever the links make them; the report lines stay true.)
    baselines = (
        {name: _read_baseline(args.compare, name) for name in names}
        if args.compare is not None
        else {}
    )
    outcomes: list[tuple[str, str]] = []
    for name in names:
        outcome = _record_and_compare(name, args.out, baselines.get(name))
        if outcome is not None:
            outcomes.append((name, outcome))
    written = _emit(_report_text(outcomes), args.report)
    reproduced = all(line == "reproduced" for _, line in outcomes)
    return 0 if written and reproduced else 1


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
    if args.compare is not None and _same_directory(args.out, args.compare):
        parser.error(
            "the comparison directory must differ from the output directory, "
            f"got --out {args.out} and --compare {args.compare}"
        )
    if args.report is not None and args.report.suffix.lower() == ".npz":
        parser.error(f"--report must not be an archive (.npz), got {args.report}")
    if args.report is not None and args.report.is_dir():
        parser.error(f"--report must be a file, got the directory {args.report}")
    return args


def _same_directory(out: Path, compare: Path) -> bool:
    """Whether ``--out`` and ``--compare`` are one directory, as far as can be seen.

    A courtesy for the common mistake, with a clear message; correctness does
    not rest on it, because every baseline is read before anything is written.
    """
    if out.resolve() == compare.resolve():
        return True
    return out.exists() and compare.exists() and out.samefile(compare)


def _read_baseline(committed_dir: Path, name: str) -> Replay | str:
    """The committed replay for ``name``, or the report line saying why not."""
    path = committed_dir / f"{name}.npz"
    try:
        if not path.exists():
            return "no committed archive"
        return load_replay(path)
    except ValueError as error:
        return f"committed archive unreadable: {error}"
    # The report must be written when its surroundings are broken; this is the
    # outermost per-demo boundary, and the error is reported and fails the run.
    except Exception as error:
        return f"comparison failed: {type(error).__name__}: {error}"


def _record_and_compare(
    name: str, out: Path, baseline: Replay | str | None
) -> str | None:
    """One demo's report line after ``<name>: ``, or None when there is none.

    ``baseline`` is what was read before recording began, or None when the run
    compares nothing. A demo whose recording or comparison fails is reported,
    and the others still run.
    """
    try:
        recorded = _record_and_save(name, out)
    # As in _read_baseline: reported with its type, counted as a failure.
    except Exception as error:
        return f"recording failed: {type(error).__name__}: {error}"
    if baseline is None:
        return None
    if isinstance(baseline, str):
        return baseline
    try:
        return _describe(first_difference(baseline, recorded))
    # As above: reported with its type, counted as a failure, others carry on.
    except Exception as error:
        return f"comparison failed: {type(error).__name__}: {error}"


def _record_and_save(name: str, out: Path) -> Replay:
    replay = record_demo(name)
    out.mkdir(parents=True, exist_ok=True)
    save_replay(out / f"{name}.npz", replay)
    return replay


def _describe(difference: Difference | None) -> str:
    """One demo's outcome: the text after ``<name>: `` in the report."""
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


def _emit(text: str, report: Path | None) -> bool:
    """Print the report, then write it; return False if the write failed.

    Printed first, so a failed write loses nothing; the failure is one line on
    standard error. A destination that holds an archive is left as it is.
    """
    print(text, end="")
    if report is None:
        return True
    try:
        _write_atomically(report, text)
    except OSError as error:
        print(f"the report could not be written to {report}: {error}", file=sys.stderr)
        return False
    except _ArchiveInTheWay:
        print(
            f"the report was not written because {report} holds an archive",
            file=sys.stderr,
        )
        return False
    return True


def _write_atomically(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` whole or not at all.

    The text goes to a temporary file beside ``path`` that is then renamed over
    it, so a reader never sees a partial report. Raises ``_ArchiveInTheWay``,
    leaving ``path`` untouched, when ``path`` is a ZIP file: every replay
    archive is one, and no text report is.
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
        if zipfile.is_zipfile(path):
            raise _ArchiveInTheWay(str(path))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)  # gone already once it has replaced path


if __name__ == "__main__":
    raise SystemExit(main())
