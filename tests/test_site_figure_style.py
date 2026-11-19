"""Source contract for the site's build-time plots.

Every plotting cell draws its figure once per theme through the shared style
and gives one identical caption per rendering. These are string checks on the
page source: they stop a plot from bypassing the conventions, but they do not
prove that a plot looks right or that the style block encloses the drawing.
"""

import re
from pathlib import Path

import pytest

SITE = Path(__file__).parents[1] / "site"
PLOT_PAGES = (
    "method/the-particle-filter.qmd",
    "story/beyond-point-masses.qmd",
    "story/clocks-as-gravimeters.qmd",
    "story/gravitational-echolocation.qmd",
    "story/how-many-masses.qmd",
    "story/into-the-plane.qmd",
    "story/one-clock-is-not-enough.qmd",
    "story/the-search-in-one-dimension.qmd",
    "story/two-hidden-masses.qmd",
)
_EXCLUDED_DIRS = {"_output", ".quarto"}
_CELL = re.compile(r"^```\{python\}[^\n]*\n(.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)
_CAPTION_ENTRY = re.compile(r'^#\|\s{2,}-\s*"(.*)"\s*$')
_FORBIDDEN = (
    r"plt\.show\(",
    r"\bcmap\s*=",
    r"[\"']tab:\w+[\"']",
    r"[\"'](steelblue|lightcoral|red|black|gray|grey)[\"']",
)


def _python_cells(path: Path) -> list[str]:
    """Return the text of each executable ```{python} cell in a page."""
    return _CELL.findall(path.read_text(encoding="utf-8"))


def _plot_cells(path: Path) -> list[str]:
    """Return the Python cells that create a matplotlib figure."""
    return [
        cell
        for cell in _python_cells(path)
        if "plt.subplots(" in cell or "plt.figure(" in cell
    ]


def _only_plot_cell(page: str) -> str:
    cells = _plot_cells(SITE / page)
    assert len(cells) == 1, f"{page}: expected one plot cell, found {len(cells)}"
    return cells[0]


def _caption_entries(cell: str) -> list[str]:
    """Return the quoted entries of the list-form ``fig-cap`` option."""
    lines = cell.splitlines()
    try:
        start = lines.index("#| fig-cap:")
    except ValueError:
        return []
    entries = []
    for line in lines[start + 1 :]:
        match = _CAPTION_ENTRY.match(line)
        if match is None:
            break
        entries.append(match.group(1))
    return entries


def test_plot_pages_are_exactly_the_known_nine() -> None:
    found = sorted(
        path.relative_to(SITE).as_posix()
        for path in SITE.rglob("*.qmd")
        if not _EXCLUDED_DIRS & set(path.relative_to(SITE).parts) and _plot_cells(path)
    )
    assert found == sorted(PLOT_PAGES)


@pytest.mark.parametrize("page", PLOT_PAGES)
def test_cell_declares_both_renderings(page: str) -> None:
    cell = _only_plot_cell(page)
    assert "#| renderings: [light, dark]" in cell.splitlines()


@pytest.mark.parametrize("page", PLOT_PAGES)
def test_cell_gives_one_identical_caption_per_rendering(page: str) -> None:
    entries = _caption_entries(_only_plot_cell(page))
    assert len(entries) == 2, f"{page}: caption entries {entries!r}"
    assert entries[0], f"{page}: empty caption"
    assert entries[0] == entries[1], f"{page}: captions differ"


@pytest.mark.parametrize("page", PLOT_PAGES)
def test_cell_draws_through_the_style_and_show(page: str) -> None:
    cell = _only_plot_cell(page)
    assert "for theme in THEMES:" in cell
    assert "with style(theme) as" in cell
    assert "show(fig)" in cell


@pytest.mark.parametrize("page", PLOT_PAGES)
def test_cell_has_no_unstyled_colour_or_display(page: str) -> None:
    cell = _only_plot_cell(page)
    matches = [pattern for pattern in _FORBIDDEN if re.search(pattern, cell)]
    assert matches == [], f"{page}: forbidden patterns {matches}"
