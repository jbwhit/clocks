"""Source contract for the site's build-time plots.

Every plotting cell draws its figure once per theme through the shared style
and gives one identical caption per rendering. These are checks on the page
source: the header and loop checks match text, and the colour and display check
reads the cell's syntax tree. They stop a plot from bypassing the conventions,
but they do not prove that a plot looks right or that the style block encloses
the drawing.
"""

import ast
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
_COLOUR_KEYWORDS = frozenset(
    {
        "color",
        "c",
        "facecolor",
        "facecolors",
        "edgecolor",
        "edgecolors",
        "markerfacecolor",
        "markeredgecolor",
        "mfc",  # alias of markerfacecolor
        "mec",  # alias of markeredgecolor
        "ecolor",
        "colors",  # vlines, hlines, contour, tick_params
    }
)
# Plotting calls whose positional string arguments are format strings (or, for
# fill, colours), which can carry a colour; markers and line styles are given
# by keyword instead.
_FORMAT_STRING_CALLS = frozenset(
    {"plot", "semilogy", "semilogx", "loglog", "step", "errorbar", "fill"}
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


def _style_violations(cell: str) -> list[str]:
    """Return what a cell does that bypasses the shared style.

    The cell's ``#|`` option lines are comments, so the whole cell parses.
    """
    violations: list[str] = []
    for node in ast.walk(ast.parse(cell)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "show"
            and isinstance(func.value, ast.Name)
            and func.value.id == "plt"
        ):
            violations.append("calls plt.show()")
        for keyword in node.keywords:
            if keyword.arg == "cmap":
                violations.append("passes cmap")
            elif keyword.arg == "fmt" and _is_string(keyword.value):
                violations.append("format string")
            elif keyword.arg in _COLOUR_KEYWORDS:
                if any(text != "none" for text in _string_constants(keyword.value)):
                    violations.append(f"literal colour in {keyword.arg}")
        if isinstance(func, ast.Attribute) and func.attr in _FORMAT_STRING_CALLS:
            # With data=, the first two positional strings are column names;
            # any further one is a format string.
            has_data = any(keyword.arg == "data" for keyword in node.keywords)
            strings = [arg for arg in node.args if _is_string(arg)]
            if len(strings) > (2 if has_data else 0):
                violations.append("positional format string")
    return violations


def _is_string(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _string_constants(node: ast.expr) -> list[str]:
    """Return the string constants a value is or, as a list or tuple, holds."""
    elements = node.elts if isinstance(node, (ast.List, ast.Tuple)) else [node]
    return [element.value for element in elements if _is_string(element)]


@pytest.mark.parametrize("page", PLOT_PAGES)
def test_cell_has_no_unstyled_colour_or_display(page: str) -> None:
    violations = _style_violations(_only_plot_cell(page))
    assert violations == [], f"{page}: {violations}"


@pytest.mark.parametrize(
    ("source", "violation"),
    [
        ('ax.plot(x, y, "r-")', "positional format string"),
        ('ax.plot(x, y, "o-")', "positional format string"),
        ('ax.plot(x, y, color="blue")', "literal colour in color"),
        ("ax.scatter(x, y, c='C0')", "literal colour in c"),
        ('ax.scatter(x, y, cmap="viridis")', "passes cmap"),
        ("plt.show ()", "calls plt.show()"),
        ('ax.scatter(x, y, c=["red", "black"])', "literal colour in c"),
        ('ax.errorbar(x, y, fmt="r-")', "format string"),
        ('ax.plot("step", "rate", "r-", data=frame)', "positional format string"),
        ('ax.vlines(x, 0, 1, colors="red")', "literal colour in colors"),
        ('ax.fill(x, y, "r")', "positional format string"),
    ],
)
def test_style_violations_reports_unstyled_calls(source: str, violation: str) -> None:
    assert _style_violations(source) == [violation]


def test_style_violations_reports_marker_colour_aliases() -> None:
    assert _style_violations('ax.plot(x, y, mfc="red", mec="blue")') == [
        "literal colour in mfc",
        "literal colour in mec",
    ]


@pytest.mark.parametrize(
    "source",
    [
        'ax.set_ylabel("red")',
        'ax.scatter(x, y, facecolors="none", edgecolors=colors.truth)',
        "ax.plot(x, y, color=colors.ink)",
        "ax.vlines(x, 0, 1, colors=colors.rule)",
        "# do not call plt.show()",
        'ax.plot("step", "rate", data=frame, color=colors.ink)',
    ],
)
def test_style_violations_allows_styled_calls(source: str) -> None:
    assert _style_violations(source) == []
