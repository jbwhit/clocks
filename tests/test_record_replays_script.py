import importlib.util
import os
import platform
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from clocks import WeightRoundingUndecided
from clocks._demos import demo_1d
from clocks._replay import DEMOS, Replay, load_replay, save_replay

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "record_replays.py"
_SPEC = importlib.util.spec_from_file_location("record_replays", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
record_replays = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(record_replays)


def _record_into(directory: Path) -> None:
    """Record demo_1d into ``directory`` through the script itself."""
    assert record_replays.main(["--out", str(directory), "demo_1d"]) == 0


def _compare(
    a: Path, b: Path, report: Path, names: tuple[str, ...] = ("demo_1d",)
) -> int:
    return record_replays.main(
        ["--out", str(b), "--compare", str(a), "--report", str(report), *names]
    )


def test_records_the_named_demo(tmp_path: Path) -> None:
    out = tmp_path / "out"

    assert record_replays.main(["--out", str(out), "demo_1d"]) == 0

    assert [p.name for p in out.iterdir()] == ["demo_1d.npz"]
    load_replay(out / "demo_1d.npz")


def test_no_names_records_all_seven(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    small = demo_1d.record(n_observations=1, n_particles=16)
    calls: list[str] = []

    def stub(name: str) -> Replay:
        calls.append(name)
        return small

    monkeypatch.setattr(record_replays, "record_demo", stub)
    out = tmp_path / "out"

    assert record_replays.main(["--out", str(out)]) == 0

    assert calls == list(DEMOS)
    assert sorted(p.name for p in out.iterdir()) == sorted(f"{n}.npz" for n in DEMOS)


def test_an_unknown_name_is_a_usage_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as raised:
        record_replays.main(["--out", str(tmp_path / "out"), "demo_nope"])

    assert raised.value.code == 2


@pytest.mark.parametrize("spelling", ["same", "relative", "symlink"])
def test_comparing_with_the_output_directory_is_a_usage_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    spelling: str,
) -> None:
    committed = tmp_path / "a"
    committed.mkdir()
    save_replay(
        committed / "demo_1d.npz", demo_1d.record(n_observations=1, n_particles=16)
    )
    before = {p.name: p.read_bytes() for p in committed.iterdir()}
    calls: list[str] = []
    monkeypatch.setattr(record_replays, "record_demo", calls.append)
    monkeypatch.chdir(tmp_path)
    if spelling == "symlink":
        (tmp_path / "link").symlink_to(committed, target_is_directory=True)
    out = {"same": str(committed), "relative": "a/", "symlink": "link"}[spelling]

    with pytest.raises(SystemExit) as raised:
        record_replays.main(["--out", out, "--compare", str(committed), "demo_1d"])

    assert raised.value.code == 2
    assert "must differ from the output directory" in capsys.readouterr().err
    assert calls == []
    assert {p.name: p.read_bytes() for p in committed.iterdir()} == before


def test_compare_reports_a_reproduced_demo(tmp_path: Path) -> None:
    a, b, report = tmp_path / "a", tmp_path / "b", tmp_path / "report.md"
    _record_into(a)

    assert _compare(a, b, report) == 0

    assert "demo_1d: reproduced" in report.read_text().splitlines()


def test_compare_names_the_first_difference(tmp_path: Path) -> None:
    a, b, report = tmp_path / "a", tmp_path / "b", tmp_path / "report.md"
    _record_into(a)
    committed = load_replay(a / "demo_1d.npz")
    mean = committed.arrays["mean"].copy()
    mean[3, 0] += 1e-6
    save_replay(
        a / "demo_1d.npz", Replay(committed.meta, {**committed.arrays, "mean": mean})
    )

    assert _compare(a, b, report) == 1

    lines = report.read_text().splitlines()
    assert any(
        line.startswith("demo_1d: differs at mean, row 3 (largest") for line in lines
    )


def test_a_difference_that_is_not_numeric_omits_the_size(tmp_path: Path) -> None:
    a, b, report = tmp_path / "a", tmp_path / "b", tmp_path / "report.md"
    _record_into(a)
    committed = load_replay(a / "demo_1d.npz")
    meta = {**committed.meta, "seed": int(committed.meta["seed"]) + 1}
    save_replay(a / "demo_1d.npz", replace(committed, meta=meta))

    assert _compare(a, b, report) == 1

    assert "demo_1d: differs at meta" in report.read_text().splitlines()


def test_compare_ignores_created_with(tmp_path: Path) -> None:
    a, b, report = tmp_path / "a", tmp_path / "b", tmp_path / "report.md"
    _record_into(a)
    committed = load_replay(a / "demo_1d.npz")
    elsewhere = {k: "elsewhere" for k in committed.meta["created_with"]}
    save_replay(
        a / "demo_1d.npz",
        replace(committed, meta={**committed.meta, "created_with": elsewhere}),
    )

    assert _compare(a, b, report) == 0

    assert "demo_1d: reproduced" in report.read_text().splitlines()


@pytest.mark.parametrize("kind", ["missing", "unreadable"])
def test_a_missing_or_unreadable_archive_is_reported(tmp_path: Path, kind: str) -> None:
    a, b, report = tmp_path / "a", tmp_path / "b", tmp_path / "report.md"
    a.mkdir()
    if kind == "unreadable":
        (a / "demo_1d.npz").write_bytes(b"not a zip")

    assert _compare(a, b, report) == 1

    expected = (
        "demo_1d: no committed archive"
        if kind == "missing"
        else "demo_1d: committed archive unreadable"
    )
    assert any(line.startswith(expected) for line in report.read_text().splitlines())
    assert (b / "demo_1d.npz").exists()


def test_report_header_records_where_it_ran(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("REPLAY_HEAD_SHA", "abc123")
    report = tmp_path / "report.md"

    record_replays.main(
        ["--out", str(tmp_path / "b"), "--report", str(report), "demo_1d"]
    )

    lines = report.read_text().splitlines()
    assert "run: 123" in lines
    assert "head: abc123" in lines
    assert f"python: {platform.python_version()}" in lines
    for prefix in ("checkout:", "numpy:", "scipy:", "blas:", "platform:"):
        assert any(line.startswith(prefix) for line in lines), prefix


def test_report_survives_a_missing_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_git(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(record_replays.subprocess, "run", no_git)
    report = tmp_path / "report.md"

    record_replays.main(
        ["--out", str(tmp_path / "b"), "--report", str(report), "demo_1d"]
    )

    assert "checkout: unknown" in report.read_text().splitlines()


@pytest.mark.parametrize("compare", [True, False], ids=["compare", "no_compare"])
def test_a_failed_recording_is_reported_and_the_others_still_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, compare: bool
) -> None:
    small = demo_1d.record(n_observations=1, n_particles=16)

    def stub(name: str) -> Replay:
        if name == "demo_2d":
            raise WeightRoundingUndecided("precision cap reached")
        return small

    monkeypatch.setattr(record_replays, "record_demo", stub)
    committed, out, report = tmp_path / "a", tmp_path / "b", tmp_path / "report.md"
    committed.mkdir()
    save_replay(committed / "demo_1d.npz", small)
    compare_args = ["--compare", str(committed)] if compare else []

    status = record_replays.main(
        [
            "--out",
            str(out),
            *compare_args,
            "--report",
            str(report),
            "demo_2d",
            "demo_1d",
        ]
    )

    lines = report.read_text().splitlines()
    assert status == 1
    assert (
        "demo_2d: recording failed: WeightRoundingUndecided: precision cap reached"
        in lines
    )
    assert ("demo_1d: reproduced" in lines) is compare
    assert [p.name for p in out.iterdir()] == ["demo_1d.npz"]


def _two_demo_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorder
) -> tuple[int, list[str]]:
    """Run demo_2d then demo_1d against committed copies of one small replay."""
    monkeypatch.setattr(record_replays, "record_demo", recorder)
    committed, out, report = tmp_path / "a", tmp_path / "b", tmp_path / "report.md"
    committed.mkdir()
    small = demo_1d.record(n_observations=1, n_particles=16)
    for name in ("demo_1d", "demo_2d"):
        save_replay(committed / f"{name}.npz", small)
    status = record_replays.main(
        [
            "--out",
            str(out),
            "--compare",
            str(committed),
            "--report",
            str(report),
            "demo_2d",
            "demo_1d",
        ]
    )
    return status, report.read_text().splitlines()


def test_an_unexpected_recording_error_is_reported_and_the_others_still_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    small = demo_1d.record(n_observations=1, n_particles=16)

    def stub(name: str) -> Replay:
        if name == "demo_2d":
            raise TypeError("an unexpected argument")
        return small

    status, lines = _two_demo_run(tmp_path, monkeypatch, stub)

    assert status == 1
    assert "demo_2d: recording failed: TypeError: an unexpected argument" in lines
    assert "demo_1d: reproduced" in lines


def test_a_failed_comparison_is_reported_and_the_others_still_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    small = demo_1d.record(n_observations=1, n_particles=16)
    real_first_difference = record_replays.first_difference
    calls: list[int] = []

    def first_call_raises(reference: Replay, other: Replay):
        calls.append(1)
        if len(calls) == 1:  # demo_2d is compared first
            raise KeyError("cloud")
        return real_first_difference(reference, other)

    monkeypatch.setattr(record_replays, "first_difference", first_call_raises)

    status, lines = _two_demo_run(tmp_path, monkeypatch, lambda name: small)

    assert status == 1
    assert "demo_2d: comparison failed: KeyError: 'cloud'" in lines
    assert "demo_1d: reproduced" in lines
    assert len(calls) == 2


def test_the_report_directory_is_created(tmp_path: Path) -> None:
    report = tmp_path / "not" / "yet" / "report.md"

    record_replays.main(
        ["--out", str(tmp_path / "b"), "--report", str(report), "demo_1d"]
    )

    assert any(line.startswith("run: ") for line in report.read_text().splitlines())


def test_the_report_is_printed_before_it_is_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    unwritable = tmp_path / "a-directory"
    unwritable.mkdir()

    with pytest.raises(OSError):
        record_replays.main(
            ["--out", str(tmp_path / "b"), "--report", str(unwritable), "demo_1d"]
        )

    assert "run: " in capsys.readouterr().out


def test_the_report_is_written_whole_and_leaves_nothing_beside_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = tmp_path / "reports" / "report.md"

    record_replays.main(
        ["--out", str(tmp_path / "b"), "--report", str(report), "demo_1d"]
    )

    assert [p.name for p in report.parent.iterdir()] == ["report.md"]
    assert report.read_text(encoding="utf-8") == capsys.readouterr().out


def test_a_failed_write_leaves_the_previous_report_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "reports" / "report.md"
    report.parent.mkdir()
    report.write_text("the previous report\n", encoding="utf-8")

    def interrupted(source: object, target: object) -> None:
        raise OSError("interrupted before the rename")

    monkeypatch.setattr(os, "replace", interrupted)

    with pytest.raises(OSError, match="interrupted"):
        record_replays.main(
            ["--out", str(tmp_path / "b"), "--report", str(report), "demo_1d"]
        )

    assert report.read_text(encoding="utf-8") == "the previous report\n"
    assert [p.name for p in report.parent.iterdir()] == ["report.md"]


def test_a_numpy_without_build_config_gives_unknown_blas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(np, "show_config", lambda mode=None: {})

    assert "blas: unknown" in record_replays._report_text([]).splitlines()


def test_the_checkout_is_read_beside_the_script(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(kwargs)
        return subprocess.CompletedProcess(args, 0, stdout="0123abcd\n", stderr="")

    monkeypatch.setattr(record_replays.subprocess, "run", run)

    lines = record_replays._report_text([]).splitlines()

    assert "checkout: 0123abcd" in lines
    assert [call["cwd"] for call in calls] == [_SCRIPT.parent]


def test_a_failed_git_gives_an_unknown_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_git(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(128, ["git", "rev-parse", "HEAD"])

    monkeypatch.setattr(record_replays.subprocess, "run", failing_git)

    assert "checkout: unknown" in record_replays._report_text([]).splitlines()


def test_report_is_also_printed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = tmp_path / "report.md"

    record_replays.main(
        ["--out", str(tmp_path / "b"), "--report", str(report), "demo_1d"]
    )

    assert report.read_text() in capsys.readouterr().out


def test_the_command_exits_with_the_comparison_status(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--out",
            str(tmp_path / "b"),
            "--compare",
            str(empty),
            "demo_1d",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1, result.stderr
