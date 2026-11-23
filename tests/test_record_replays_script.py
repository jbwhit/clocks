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
from clocks._demos import demo_1d, demo_2d
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


def _is_case_insensitive(directory: Path) -> bool:
    probe = directory / "case-probe"
    probe.touch()
    try:
        return (directory / "CASE-PROBE").exists()
    finally:
        probe.unlink()


def _snapshot(root: Path) -> dict[str, tuple[str, bytes | str]]:
    """Every entry under ``root``: a symlink's target, a file's bytes, or a dir."""
    entries: dict[str, tuple[str, bytes | str]] = {}
    for path in sorted(root.rglob("*")):
        key = str(path.relative_to(root))
        if path.is_symlink():
            entries[key] = ("link", str(path.readlink()))
        elif path.is_file():
            entries[key] = ("file", path.read_bytes())
        else:
            entries[key] = ("dir", "")
    return entries


def _small() -> Replay:
    return demo_1d.record(n_observations=1, n_particles=16)


_SMALL_1D = _small()
_SMALL_2D = demo_2d.record(n_observations=1, n_particles=16)


@pytest.mark.parametrize("spelling", ["same", "relative", "symlink", "case_variant"])
def test_comparing_with_the_output_directory_is_a_usage_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    spelling: str,
) -> None:
    if spelling == "case_variant" and not _is_case_insensitive(tmp_path):
        pytest.skip("a case-variant spelling names another directory here")
    committed = tmp_path / "a"
    committed.mkdir()
    save_replay(committed / "demo_1d.npz", _small())
    calls: list[str] = []
    monkeypatch.setattr(record_replays, "record_demo", calls.append)
    monkeypatch.chdir(tmp_path)
    if spelling == "symlink":
        (tmp_path / "link").symlink_to(committed, target_is_directory=True)
    before = _snapshot(tmp_path)
    out = {
        "same": str(committed),
        "relative": "a/",
        "symlink": "link",
        "case_variant": "A",
    }[spelling]

    with pytest.raises(SystemExit) as raised:
        record_replays.main(["--out", out, "--compare", str(committed), "demo_1d"])

    assert raised.value.code == 2
    assert "must differ from the output directory" in capsys.readouterr().err
    assert calls == []
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("link", ["symlink", "hard_link"])
def test_an_output_linked_to_a_baseline_does_not_overwrite_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, link: str
) -> None:
    # The directories differ, so the courtesy check passes, but out/demo_1d.npz
    # is a link to the baseline file. Saving puts a new file at the output path
    # instead of writing through the link, so the baseline keeps its bytes and
    # the comparison is with it. (Reading order is tested by
    # test_baselines_are_all_read_before_any_demo_is_recorded.)
    small = _small()
    committed, out, report = tmp_path / "a", tmp_path / "b", tmp_path / "report.md"
    committed.mkdir()
    out.mkdir()
    save_replay(committed / "demo_1d.npz", small)
    if link == "symlink":
        (out / "demo_1d.npz").symlink_to(committed / "demo_1d.npz")
    else:
        (out / "demo_1d.npz").hardlink_to(committed / "demo_1d.npz")
    baseline_bytes = (committed / "demo_1d.npz").read_bytes()
    mean = small.arrays["mean"].copy()
    mean[0, 0] += 1e-3
    different = Replay(small.meta, {**small.arrays, "mean": mean})
    monkeypatch.setattr(record_replays, "record_demo", lambda name: different)

    status = _compare(committed, out, report)

    lines = report.read_text().splitlines()
    assert status == 1
    assert any(line.startswith("demo_1d: differs at mean, row 0") for line in lines)
    assert (committed / "demo_1d.npz").read_bytes() == baseline_bytes
    assert not (out / "demo_1d.npz").is_symlink()


def test_a_baseline_in_another_case_is_read_before_anything_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not _is_case_insensitive(tmp_path):
        pytest.skip("a case-variant spelling names another directory here")
    out = tmp_path / "new" / "inner"
    compare = tmp_path.parent / tmp_path.name.upper() / "new" / "inner"
    assert str(compare) != str(out)
    monkeypatch.setattr(record_replays, "record_demo", lambda name: _small())

    status = _compare(compare, out, tmp_path / "report.md")

    lines = (tmp_path / "report.md").read_text().splitlines()
    assert status == 1
    assert "demo_1d: no committed archive" in lines
    assert (out / "demo_1d.npz").exists()


@pytest.mark.parametrize("name", ["x.npz", "X.NPZ"], ids=["lower", "upper"])
def test_a_report_path_may_not_be_an_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    name: str,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(record_replays, "record_demo", calls.append)

    with pytest.raises(SystemExit) as raised:
        record_replays.main(
            ["--out", str(tmp_path / "b"), "--report", str(tmp_path / name), "demo_1d"]
        )

    assert raised.value.code == 2
    assert "must not be an archive" in capsys.readouterr().err
    assert calls == []
    assert list(tmp_path.iterdir()) == []


_REPORT_ENTRIES = [
    "previous_report",
    "other_text",
    "archive",
    "empty",
    "unreadable",
    "dangling_symlink",
    "symlink_to_a_file",
    "fifo",
    "directory",
]


@pytest.mark.parametrize("kind", _REPORT_ENTRIES)
def test_an_existing_report_path_is_refused_before_anything_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    kind: str,
) -> None:
    if kind == "unreadable" and os.geteuid() == 0:
        pytest.skip("permissions do not bind a process running as root")
    if kind == "fifo" and not hasattr(os, "mkfifo"):
        pytest.skip("this platform has no named pipes")
    report, target = tmp_path / "report.md", tmp_path / "target.bin"
    contents = {
        "previous_report": b"run: local\nan earlier report\n",
        "other_text": b"run: payroll\n",
        "empty": b"",
    }
    if kind in contents:
        report.write_bytes(contents[kind])
    elif kind in ("archive", "unreadable"):
        save_replay(tmp_path / "x.npz", _small())
        (tmp_path / "x.npz").rename(report)
    elif kind == "dangling_symlink":
        report.symlink_to(target)
    elif kind == "symlink_to_a_file":
        target.write_bytes(b"run: local\nbytes behind the link\n")
        report.symlink_to(target)
    elif kind == "fifo":
        os.mkfifo(report)
    else:
        report.mkdir()
    # The link and the pipe are never opened: bytes are read only from files
    # that are readable regular files.
    before_bytes = report.read_bytes() if kind in ("archive", *contents) else None
    before_stat = report.lstat()
    before_target = target.read_bytes() if target.exists() else None
    calls: list[str] = []
    monkeypatch.setattr(record_replays, "record_demo", calls.append)
    if kind == "unreadable":
        report.chmod(0o000)
        before_stat = report.lstat()

    try:
        with pytest.raises(SystemExit) as raised:
            record_replays.main(
                ["--out", str(tmp_path / "b"), "--report", str(report), "demo_1d"]
            )
    finally:
        if kind == "unreadable":
            report.chmod(0o644)

    errors = capsys.readouterr().err
    assert raised.value.code == 2
    assert f"--report {report} already exists" in errors
    assert calls == []
    assert not (tmp_path / "b").exists()
    after = report.lstat()
    assert (after.st_ino, after.st_dev) == (before_stat.st_ino, before_stat.st_dev)
    if before_bytes is not None:
        assert report.read_bytes() == before_bytes
    if kind in ("dangling_symlink", "symlink_to_a_file"):
        assert report.is_symlink() and report.readlink() == target
        assert (target.read_bytes() if target.exists() else None) == before_target
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(".")) == []


def test_a_report_that_appears_during_the_run_is_not_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = tmp_path / "report.md"

    def recording_creates_the_report(name: str) -> Replay:
        report.write_bytes(b"run: arrived while recording\n")
        return _small()

    monkeypatch.setattr(record_replays, "record_demo", recording_creates_the_report)

    status = record_replays.main(
        ["--out", str(tmp_path / "b"), "--report", str(report), "demo_1d"]
    )

    printed, errors = capsys.readouterr()
    assert status == 1
    assert printed.startswith("run: ")
    assert f"the report could not be written to {report}: it already exists" in errors
    assert report.read_bytes() == b"run: arrived while recording\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["b", "report.md"]


def test_a_baseline_behind_a_link_at_the_report_path_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The reviewer's scenario: the baseline archive is stored in the very file
    # the report path names, under a text name.
    small = _small()
    shared, recorded, baseline = (tmp_path / n for n in ("shared", "recorded", "base"))
    for directory in (shared, baseline):
        directory.mkdir()
    save_replay(tmp_path / "x.npz", small)
    report = shared / "report.md"
    (tmp_path / "x.npz").rename(report)
    (baseline / "demo_1d.npz").symlink_to(report)
    monkeypatch.setattr(record_replays, "record_demo", lambda name: small)

    with pytest.raises(SystemExit) as raised:
        record_replays.main(
            ["--out", str(recorded), "--compare", str(baseline)]
            + ["--report", str(report), "demo_1d"]
        )

    assert raised.value.code == 2
    assert load_replay(baseline / "demo_1d.npz").meta == small.meta


def test_baselines_are_all_read_before_any_demo_is_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Same directory for --out and --compare (the courtesy check is disabled),
    # so each save overwrites that demo's baseline. Read after the saves, every
    # comparison would be of a recording with itself and say "reproduced".
    monkeypatch.setattr(record_replays, "_same_directory", lambda out, compare: False)
    stubs = {}
    for name, small in (("demo_1d", _SMALL_1D), ("demo_2d", _SMALL_2D)):
        save_replay(tmp_path / f"{name}.npz", small)
        stubs[name] = Replay(
            {**small.meta, "seed": small.meta["seed"] + 1}, small.arrays
        )
    monkeypatch.setattr(record_replays, "record_demo", stubs.__getitem__)

    status = record_replays.main(
        ["--out", str(tmp_path), "--compare", str(tmp_path), "demo_1d", "demo_2d"]
    )

    lines = capsys.readouterr().out.splitlines()
    assert [line for line in lines if line.startswith("demo_")] == [
        "demo_1d: differs at meta",
        "demo_2d: differs at meta",
    ]
    assert status == 1


def test_a_report_beside_the_archives_is_not_a_collision(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    _record_into(a)

    assert _compare(a, b, b / "report.md") == 0


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


def test_the_report_is_written_whole_and_leaves_nothing_beside_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = tmp_path / "reports" / "report.md"

    record_replays.main(
        ["--out", str(tmp_path / "b"), "--report", str(report), "demo_1d"]
    )

    assert [p.name for p in report.parent.iterdir()] == ["report.md"]
    assert report.read_text(encoding="utf-8") == capsys.readouterr().out


def test_a_failed_write_is_reported_and_leaves_nothing_beside_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = tmp_path / "reports" / "report.md"
    monkeypatch.setattr(record_replays, "record_demo", lambda name: _small())

    def interrupted(source: object, target: object) -> None:
        raise OSError("interrupted before the link")

    monkeypatch.setattr(os, "link", interrupted)

    status = record_replays.main(
        ["--out", str(tmp_path / "b"), "--report", str(report), "demo_1d"]
    )

    printed, errors = capsys.readouterr()
    assert status == 1
    assert "run: " in printed
    assert errors == (
        f"the report could not be written to {report}: interrupted before the link\n"
    )
    assert list(report.parent.iterdir()) == []


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
