import importlib.util
import platform
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

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
