"""Record the packaged demos as replay archives and report how they compare.

Writes ``<out>/<demo>.npz`` for each requested demo (all seven by default).
Archives are created, never replaced, unless ``--overwrite`` is given: without
it, anything already at one of those paths is a usage error, so no link into
``--out`` can lose its content.
With ``--compare DIR`` it also compares each fresh recording in full with
``DIR/<demo>.npz``, every one of which is read before anything is written, and
prints a short report, writing it to ``--report FILE`` as well when given.
``DIR`` must not be the output directory, ``FILE`` must not end in ``.npz``,
and nothing may exist at ``FILE``: the report is created, never replaced (a
usage error up front, and the file is put in place with a hard link, which
fails rather than replace, in case something appears during the run). A demo
whose
recording fails gets a ``recording failed`` line, with or without
``--compare``, one whose comparison fails a ``comparison failed`` line, and
the others still run. The exit status
is 0 when every demo was recorded and every compared demo reproduced, and 1
otherwise; a workflow that only wants the report ignores it.

    uv run python scripts/record_replays.py --out assets/replays --overwrite
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
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from clocks._replay import (
    DEMOS,
    Difference,
    Replay,
    create_replay,
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
    names = args.demos
    # Every baseline is read before anything is written, and each comparison is
    # with the recording in memory, so no spelling of a path can make the run
    # compare against its own output. (Two outputs the user has linked to one
    # file hold whatever the links make them; the report lines stay true.)
    baselines = (
        {name: _read_baseline(args.compare, name) for name in names}
        if args.compare is not None
        else {}
    )
    write = save_replay if args.overwrite else create_replay
    outcomes: list[tuple[str, str]] = []
    for name in names:
        outcome = _record_and_compare(name, args.out, baselines.get(name), write)
        if outcome is not None:
            outcomes.append((name, outcome))
    written = _emit(_report_text(outcomes), args.report)
    reproduced = all(line == "reproduced" for _, line in outcomes)
    return 0 if written and reproduced else 1


_FREE_PATH_ADVICE = "remove it or choose another path"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="directory to write")
    parser.add_argument(
        "--compare", type=Path, help="directory of committed archives to compare with"
    )
    parser.add_argument("--report", type=Path, help="file to write the report to")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace archives that already exist in --out; anything else that "
        "links to those files changes with them",
    )
    # No default: Python 3.12 rejects a tuple default as one invalid choice.
    parser.add_argument("demos", nargs="*", choices=DEMOS, metavar="demo")
    args = parser.parse_args(argv)
    args.demos = list(dict.fromkeys(args.demos or DEMOS))  # each demo once, in order
    if args.compare is not None and _same_directory(args.out, args.compare):
        parser.error(
            "the comparison directory must differ from the output directory, "
            f"got --out {args.out} and --compare {args.compare}"
        )
    if args.report is not None and args.report.suffix.lower() == ".npz":
        parser.error(f"--report must not be an archive (.npz), got {args.report}")
    if args.report is not None and os.path.lexists(args.report):
        # lexists: a link to nothing is an entry too. Nothing is inspected, so
        # an archive, a FIFO or an unreadable file at the path is all one case.
        parser.error(f"--report {args.report} already exists; {_FREE_PATH_ADVICE}")
    if not args.overwrite:
        existing = [
            args.out / f"{name}.npz"
            for name in args.demos
            if os.path.lexists(args.out / f"{name}.npz")
        ]
        if existing:
            parser.error(
                f"{', '.join(str(path) for path in existing)} already exist(s); "
                "pass --overwrite to replace them or choose another --out"
            )
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
    name: str,
    out: Path,
    baseline: Replay | str | None,
    write: Callable[[Path, Replay], None],
) -> str | None:
    """One demo's report line after ``<name>: ``, or None when there is none.

    ``baseline`` is what was read before recording began, or None when the run
    compares nothing. ``write`` is ``create_replay`` (which refuses to replace
    anything, even something that appeared during the run) or, with
    ``--overwrite``, ``save_replay``. A demo whose recording or comparison
    fails is reported, and the others still run.
    """
    try:
        recorded = _record_and_save(name, out, write)
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


def _record_and_save(
    name: str, out: Path, write: Callable[[Path, Replay], None]
) -> Replay:
    replay = record_demo(name)
    out.mkdir(parents=True, exist_ok=True)
    write(out / f"{name}.npz", replay)
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
    """The report: ``key: value`` lines, every value on one line.

    Values can carry text from outside (an unreadable archive's member names, an
    environment variable), so ``_one_line`` keeps any of it from starting a line
    of its own, such as a forged ``demo_1d: reproduced``.
    """
    lines = [
        f"run: {_one_line(os.environ.get('GITHUB_RUN_ID', 'local'))}",
        f"head: {_one_line(os.environ.get('REPLAY_HEAD_SHA', 'unknown'))}",
        f"checkout: {_one_line(_checkout_sha())}",
        *(f"{key}: {_one_line(str(value))}" for key, value in provenance().items()),
        f"blas: {_one_line(_blas())}",
        *(f"{name}: {_one_line(outcome)}" for name, outcome in outcomes),
    ]
    return "\n".join(lines) + "\n"


def _one_line(text: str) -> str:
    """``text`` with every character that is not printable escaped.

    Escaping by printability, not from a short list, because ``str.splitlines``
    breaks lines at more than ``\\n`` and ``\\r`` (``\\x0b``, ``\\x85``, U+2028, ...).
    """
    return "".join(
        ch if ch.isprintable() or ch == " " else ch.encode("unicode_escape").decode()
        for ch in text
    )


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
    """Print the report and write it, each regardless of the other.

    Returns True only if everything attempted succeeded. A failed print (a
    closed pipe, say) does not stop the file being written, and a failed write
    loses nothing already printed; each failure is one line on standard error,
    if that is still open. Anything at ``report`` is left untouched.
    """
    succeeded = True
    try:
        print(text, end="")
        sys.stdout.flush()
    except OSError as error:
        succeeded = False
        _warn(f"the report could not be printed: {error}")
    if report is None:
        return succeeded
    try:
        _create_atomically(report, text)
    except FileExistsError:
        succeeded = False
        _warn(
            f"the report could not be written to {report}: it already exists "
            f"and was left untouched; {_FREE_PATH_ADVICE}"
        )
    except OSError as error:
        succeeded = False
        _warn(f"the report could not be written to {report}: {error}")
    return succeeded


def _warn(message: str) -> None:
    """One line on standard error; nothing more can be done if that is gone too."""
    try:
        print(message, file=sys.stderr)
    except OSError:
        pass


def _create_atomically(path: Path, text: str) -> None:
    """Create ``path`` holding ``text`` whole, or not at all; never replace.

    The text goes to a temporary file beside ``path`` that is then hard-linked
    to ``path``: ``os.link`` fails with ``FileExistsError`` when any directory
    entry is there, so nothing can be replaced, and a reader never sees a
    partial report.
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
        os.link(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as error:
            # Not raised: it would mask the write error being propagated.
            print(
                f"the temporary file {temporary} could not be removed: {error}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    raise SystemExit(main())
