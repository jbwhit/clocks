"""Pin the properties of the replay-report workflow, read as text.

No YAML library is a declared dependency, so the file is read line by line.
"""

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


@pytest.mark.parametrize(
    "path",
    [
        "src/clocks/**",
        "assets/replays/**",
        "scripts/record_replays.py",
        "uv.lock",
        "pyproject.toml",
        _OWN_PATH,
    ],
)
def test_triggers_are_filtered_to_what_can_change_a_replay(path: str) -> None:
    entries = [line for line in _text().splitlines() if line.strip() == f'- "{path}"']
    assert len(entries) == 2, f"{path} must be listed under push and pull_request"


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
    tmp_path: Path, report: str | None, upload_outcome: str
) -> tuple[str, str]:
    if report is not None:
        (tmp_path / "replay-report.md").write_text(report, encoding="utf-8")
    summary = tmp_path / "summary.md"
    bash = shutil.which("bash")
    assert bash is not None
    result = subprocess.run(
        [bash, "-c", _last_script()],
        cwd=tmp_path,
        env={
            "PATH": "/usr/bin:/bin",
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


@pytest.mark.parametrize(
    ("report", "outcome", "in_stdout", "summary_has_report"),
    [
        pytest.param("# report\nhead: abc\n", "success", None, True, id="shown"),
        pytest.param(
            "",
            "success",
            "no report was written for sample 2",
            False,
            id="empty_report",
        ),
        pytest.param(
            None, "success", "no report was written for sample 2", False, id="no_report"
        ),
        pytest.param(
            "# report\n",
            "failure",
            "was not uploaded (failure)",
            True,
            id="upload_failed",
        ),
    ],
)
def test_the_last_step_shows_or_warns(
    tmp_path: Path,
    report: str | None,
    outcome: str,
    in_stdout: str | None,
    summary_has_report: bool,
) -> None:
    stdout, summary = _run_last_step(tmp_path, report, outcome)
    if in_stdout is None:
        assert "::warning" not in stdout
    else:
        assert in_stdout in stdout
    if summary_has_report:
        assert summary == report
    else:
        assert summary == ""
