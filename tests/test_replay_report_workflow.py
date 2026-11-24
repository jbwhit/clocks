"""Pin the properties of the replay-report workflow, read as text.

No YAML library is a declared dependency, so the file is read line by line.
"""

import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW = _ROOT / ".github" / "workflows" / "replay-report.yml"
_CI = _ROOT / ".github" / "workflows" / "ci.yml"
_OWN_PATH = ".github/workflows/replay-report.yml"


def _text() -> str:
    return _WORKFLOW.read_text(encoding="utf-8")


def _steps() -> list[str]:
    after = _text().split("steps:\n", 1)[1]
    items: list[list[str]] = []
    for line in after.splitlines():
        if line.startswith("      - "):
            items.append([])
        if items:
            items[-1].append(line)
    return ["\n".join(item) for item in items]


def test_every_step_may_fail() -> None:
    steps = _steps()
    assert len(steps) >= 2
    for step in steps:
        assert "continue-on-error: true" in step, step


def test_the_last_step_runs_whatever_came_before() -> None:
    # Without it, a failed earlier step would skip the warning about a missing
    # report, and the summary would say nothing.
    last = _steps()[-1]
    assert "if: always()" in [line.strip() for line in last.splitlines()], last


def test_the_head_commit_is_the_pull_requests_own() -> None:
    # On a pull request github.sha is the merge commit, which no one pushed.
    job = _text().split("jobs:\n", 1)[1].split("steps:\n", 1)[0]
    env = job.split("    env:\n", 1)[1].splitlines()
    assert (
        "      REPLAY_HEAD_SHA: ${{ github.event.pull_request.head.sha || github.sha }}"
        in env
    )


def test_the_report_job_is_a_three_way_matrix() -> None:
    text = _text()
    assert "sample: [1, 2, 3]" in text
    assert "fail-fast: false" in text


_TRIGGER_PATHS = [
    "src/clocks/**",
    "assets/replays/**",
    "scripts/record_replays.py",
    "uv.lock",
    "pyproject.toml",
    _OWN_PATH,
]


def _assert_listed_once_per_trigger(text: str, path: str) -> None:
    """Assert ``path`` is listed exactly once under push and once under pull_request."""
    triggers = text.split("\non:\n", 1)[1].split("\nconcurrency:", 1)[0]
    push, pull_request = triggers.split("\n  pull_request:\n", 1)
    for trigger, section in (("push", push), ("pull_request", pull_request)):
        entries = [
            line for line in section.splitlines() if line.strip() == f'- "{path}"'
        ]
        assert len(entries) == 1, (
            f"{path} is listed {len(entries)} times under {trigger}"
        )


@pytest.mark.parametrize("path", _TRIGGER_PATHS)
def test_triggers_are_filtered_to_what_can_change_a_replay(path: str) -> None:
    _assert_listed_once_per_trigger(_text(), path)


def test_a_path_moved_from_pull_request_to_push_is_caught() -> None:
    entry = '      - "uv.lock"\n'
    text = _text()
    push, pull_request = text.split("\n  pull_request:\n", 1)
    assert entry in push and entry in pull_request
    moved = (
        push.replace(entry, entry + entry, 1)
        + "\n  pull_request:\n"
        + pull_request.replace(entry, "", 1)
    )
    # Two entries in all, which a count over the whole file would accept.
    assert moved.count(entry) == 2

    with pytest.raises(AssertionError, match="listed 2 times under push"):
        _assert_listed_once_per_trigger(moved, "uv.lock")


def test_drafts_are_skipped_and_jobs_time_out() -> None:
    text = _text()
    assert "if: github.event_name == 'push' || !github.event.pull_request.draft" in text
    assert "timeout-minutes: 10" in text


def test_the_recording_uses_the_locked_environment() -> None:
    record = [s for s in _steps() if "scripts/record_replays.py" in s]
    assert len(record) == 1
    assert "uv run --locked" in record[0]


def test_artifacts_are_named_per_attempt_and_sample() -> None:
    names = [
        line
        for line in _text().splitlines()
        if line.strip().startswith("name: replay-report-")
    ]
    assert len(names) == 1
    for part in ("github.run_id", "github.run_attempt", "matrix.sample"):
        assert part in names[0]


def test_the_report_never_touches_the_test_workflow() -> None:
    assert "record_replays" not in _CI.read_text(encoding="utf-8")


def _last_script() -> str:
    lines = _text().splitlines()
    start = max(i for i, line in enumerate(lines) if line.strip() == "run: |")
    return textwrap.dedent("\n".join(lines[start + 1 :])) + "\n"


def _run_last_step(
    tmp_path: Path,
    report: str | None,
    upload_outcome: str,
    summary_name: str,
    path_prefix: Path | None = None,
    seed_summary: str | None = None,
) -> tuple[str, str]:
    if report is not None:
        (tmp_path / "replay-report.md").write_text(report, encoding="utf-8")
    summary = tmp_path / summary_name
    if seed_summary is not None:
        summary.write_text(seed_summary, encoding="utf-8")
    path = "/usr/bin:/bin" if path_prefix is None else f"{path_prefix}:/usr/bin:/bin"
    bash = shutil.which("bash")
    assert bash is not None
    result = subprocess.run(
        # -e, as GitHub runs a step that names no shell: a failed command ends it.
        [bash, "-e", "-c", _last_script()],
        cwd=tmp_path,
        env={
            "PATH": path,
            "GITHUB_STEP_SUMMARY": str(summary),
            "SAMPLE": "2",
            "UPLOAD_OUTCOME": upload_outcome,
        },
        capture_output=True,
        text=True,
        check=True,
    )
    written = summary.read_text(encoding="utf-8") if summary.exists() else ""
    return result.stdout, written


_SUMMARY = "summary.md"
_UNWRITABLE_SUMMARY = "no-such-directory/summary.md"


def test_the_fence_closes_on_its_own_line_without_a_trailing_newline(
    tmp_path: Path,
) -> None:
    stdout, summary = _run_last_step(tmp_path, "head: abc", "success", _SUMMARY)

    assert summary == "```text\nhead: abc\n```\n"
    assert "::warning" not in stdout


def test_an_unreadable_report_gives_the_warning(tmp_path: Path) -> None:
    if os.geteuid() == 0:
        pytest.skip("permissions do not bind a process running as root")
    report = tmp_path / "replay-report.md"
    report.write_text("# report\n", encoding="utf-8")
    report.chmod(0o000)
    try:
        stdout, _ = _run_last_step(tmp_path, None, "success", _SUMMARY)
    finally:
        report.chmod(0o644)

    assert "the report for sample 2 could not be added to the summary" in stdout


@pytest.mark.parametrize(
    ("report", "outcome", "summary_name", "in_stdout", "summary_has_report"),
    [
        pytest.param(
            "# report\nhead: abc\n", "success", _SUMMARY, None, True, id="shown"
        ),
        pytest.param(
            "",
            "success",
            _SUMMARY,
            "no report was written for sample 2",
            False,
            id="empty_report",
        ),
        pytest.param(
            None,
            "success",
            _SUMMARY,
            "no report was written for sample 2",
            False,
            id="no_report",
        ),
        pytest.param(
            "# report\n",
            "failure",
            _SUMMARY,
            "was not uploaded (failure)",
            True,
            id="upload_failed",
        ),
        pytest.param(
            "# report\n",
            "success",
            _UNWRITABLE_SUMMARY,
            "the report for sample 2 could not be added to the summary",
            False,
            id="summary_unwritable",
        ),
    ],
)
def test_the_last_step_shows_or_warns(
    tmp_path: Path,
    report: str | None,
    outcome: str,
    summary_name: str,
    in_stdout: str | None,
    summary_has_report: bool,
) -> None:
    stdout, summary = _run_last_step(tmp_path, report, outcome, summary_name)
    if in_stdout is None:
        assert "::warning" not in stdout
    else:
        assert in_stdout in stdout
    if summary_has_report:
        # Fenced as text, so Markdown in a report cannot render as report lines.
        assert summary == f"```text\n{report}```\n"
    else:
        assert summary == ""


def test_a_failed_read_leaves_the_summary_untouched(tmp_path: Path) -> None:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_cat = fake_bin / "cat"
    fake_cat.write_text("#!/bin/sh\nprintf 'partial'\nexit 1\n", encoding="utf-8")
    fake_cat.chmod(0o755)

    stdout, summary = _run_last_step(
        tmp_path,
        "# report\n",
        "success",
        _SUMMARY,
        path_prefix=fake_bin,
        seed_summary="previous summary\n",
    )

    assert "the report for sample 2 could not be added to the summary" in stdout
    assert summary == "previous summary\n"


def test_a_report_with_format_characters_is_added_unchanged(tmp_path: Path) -> None:
    report = "100% done \\n and %s\n"

    stdout, summary = _run_last_step(tmp_path, report, "success", _SUMMARY)

    assert summary == f"```text\n{report}```\n"
    assert "::warning" not in stdout


def test_the_upload_runs_on_node_24() -> None:
    steps = [
        step
        for step in _steps()
        if "id: upload" in [line.strip() for line in step.splitlines()]
    ]
    assert len(steps) == 1
    uses = [
        line.strip()
        for line in steps[0].splitlines()
        if line.strip().startswith("uses:")
    ]
    assert uses == ["uses: actions/upload-artifact@v7"]
