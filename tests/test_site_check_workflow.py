"""The site check must run when the data its figures are drawn from change."""

from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]

FIXTURE = """name: Fixture
on:
  push:
    branches: [main]
    paths:
      - "assets/replays/**"
  pull_request:
    branches: [main]
    types: [opened, synchronize, ready_for_review]
    paths:
      - "site/**"
      # a comment inside the list
      - "src/clocks/**"

concurrency:
  group: fixture
"""


def _pull_request_paths(text: str) -> list[str]:
    """Entries of the ``paths:`` list directly under ``on:`` -> ``pull_request:``."""
    lines = text.splitlines()
    in_on = in_pull_request = in_paths = False
    entries: list[str] = []
    for line in lines:
        indent = len(line) - len(line.lstrip(" "))
        if not line.strip():
            continue
        if indent == 0:
            in_on = line.rstrip() == "on:"
            in_pull_request = in_paths = False
        elif in_on and indent == 2:
            in_pull_request = line.strip() == "pull_request:"
            in_paths = False
        elif in_pull_request and indent == 4:
            in_paths = line.strip() == "paths:"
        elif in_paths and indent == 6:
            stripped = line.strip()
            if stripped.startswith("- "):
                entries.append(stripped[2:].strip().strip('"'))
    return entries


def _pull_request_types(text: str) -> list[str]:
    """Entries of the inline ``types: [...]`` list under ``pull_request:``."""
    in_on = in_pull_request = False
    for line in text.splitlines():
        indent = len(line) - len(line.lstrip(" "))
        if not line.strip():
            continue
        if indent == 0:
            in_on = line.rstrip() == "on:"
            in_pull_request = False
        elif in_on and indent == 2:
            in_pull_request = line.strip() == "pull_request:"
        elif in_pull_request and indent == 4 and line.strip().startswith("types:"):
            inner = line.strip()[len("types:") :].strip().strip("[]")
            return [item.strip() for item in inner.split(",") if item.strip()]
    return []


def test_the_types_parser_reads_only_the_pull_request_types() -> None:
    assert _pull_request_types(FIXTURE) == [
        "opened",
        "synchronize",
        "ready_for_review",
    ]


def test_site_check_runs_when_a_draft_is_marked_ready() -> None:
    text = (ROOT / ".github/workflows/site-check.yml").read_text()
    assert "ready_for_review" in _pull_request_types(text)


def test_the_path_parser_reads_only_the_pull_request_list() -> None:
    assert _pull_request_paths(FIXTURE) == ["site/**", "src/clocks/**"]


@pytest.mark.parametrize(
    "path", ["assets/replays/**", "assets/echolocation_range_study.json"]
)
def test_site_check_runs_when_the_figure_data_change(path: str) -> None:
    text = (ROOT / ".github/workflows/site-check.yml").read_text()
    assert _pull_request_paths(text).count(path) == 1
