"""Source contract for the site's build-time plots.

Every plot cell on a page (a page may hold several) draws its figure once per
theme through the shared style and gives one identical caption per rendering.
A cell that calls a figure builder from ``clocks.viz`` must have one canonical
form. The page source is also checked against the figure manifest of
``scripts/check_site_figures.py``, and no page or the README may mention a
retired image file. Library modules that draw figures get the same colour check
as the cells. These are checks on source: the header and loop checks match
text, and the colour and display check reads the cell's syntax tree. They stop
a plot from bypassing the conventions, but they do not prove that a plot looks
right, nor, for a cell that calls no ``clocks.viz`` builder, that its style
block encloses the drawing (a builder cell does get that check).

For the two cells that draw committed data through a ``clocks.viz`` builder,
this module pins the cell's program and its option block exactly, and requires
any cell in which a builder call is found to be one of them. A builder call
is found when it is resolved from a ``clocks.viz`` import in the cell (any of
the four import forms, with aliases) or when the called name itself is a
builder's name; a builder imported from a private module under another name is
not found. It checks page source, so it does not try to prove that a figure is
visible on the rendered page: a surrounding block that hides its contents, a
stylesheet, or a script could still hide one. The rendered-site check
(``scripts/check_site_figures.py``) covers the rendered structure, and the
contact sheets and the phone-width measurements cover what a visitor sees.

Clip embeds are pinned whole: each is the text ``_embed_block`` builds from
``REPLAY_EMBEDS``, present exactly once and in the table's order, followed by
its description, and video markup anywhere else on a page is a violation. This
catches an embed's markup changed by mistake. It does not check what surrounds
a block, so it does not catch a canonical block inside a code fence or an HTML
comment, which Markdown does not render, or inside a wrapper that hides it. The
rendered-site check (``scripts/check_site_figures.py``) catches the first, and
part of the second: a block that was not rendered leaves no video, and the
count of themed elements against the manifest fails; a wrapper carrying one of
the markers that script names (``HIDING_CLASSES``, ``hidden``, ``aria-hidden``,
two inline styles, ``template``, ``noscript``, a closed ``details``) fails its
hidden-ancestor rule. A wrapper that hides by any other means, an opacity of
zero or a ``clip-path`` for two, passes both. Actual visibility, keyboard reach
and playback are browser checks, made by hand for each pull request that
changes a clip or its page.
"""

import ast
import html
import importlib.util
import re
import string
import sys
import urllib.parse
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import NamedTuple

import numpy as np
import pytest

import clocks.viz
from clocks import (
    ClockArray,
    InferenceConfig,
    MassConfig,
    NoiseConfig,
    PriorConfig,
    SimulationConfig,
    infer,
    simulate,
)
from clocks._clip_parts import standard_deviations
from clocks._clips import CLIP_HEIGHTS
from clocks._demos import demo_2d
from clocks._figures import trace_decimals
from clocks._replay import Replay, load_replay

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_site_figures.py"
_SPEC = importlib.util.spec_from_file_location("check_site_figures", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
check_site_figures = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_site_figures)

REPO = Path(__file__).resolve().parents[1]

# The figure builders of the facade, by name, whatever way a cell reaches them.
BUILDER_NAMES = frozenset(
    name for name in clocks.viz.__all__ if name.endswith("_figure")
)

SITE = REPO / "site"
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


class Embed(NamedTuple):
    """One clip embedded on a page."""

    clip: str
    label: str  # the aria-label of both videos
    caption: str


# Page (relative to site/) -> its clip embeds, in document order.
REPLAY_EMBEDS: dict[str, tuple[Embed, ...]] = {
    "index.qmd": (
        Embed(
            "demo_2d",
            "Animation: the 2D search",
            "One simulated run: a particle filter finding a hidden mass on a plane "
            "from clock rates alone. Left: the cloud of hypotheses, the eight clocks "
            "and the true position. Right: the estimates of x, y and M after each "
            "observation.",
        ),
    ),
    "story/the-search-in-one-dimension.qmd": (
        Embed(
            "demo_1d",
            "Animation: the search in one dimension",
            "The 1D search, one simulated run. Left: the cloud of hypotheses in "
            "position and mass, above the three clocks and, clock by clock, the "
            "true, predicted and measured slowing. Right: the estimates of x and M "
            "after each observation.",
        ),
    ),
    "story/into-the-plane.qmd": (
        Embed(
            "demo_2d",
            "Animation: the 2D search",
            "The 2D search, one simulated run: 8 clocks, one hidden mass, three "
            "unknowns (x, y, M). Left: the cloud and the clocks, each given a "
            "stronger fill the slower it runs. Right: the three estimates after "
            "each observation.",
        ),
    ),
    "story/two-hidden-masses.qmd": (
        Embed(
            "demo_multi_mass",
            "Animation: two masses on a line",
            "Two masses in 1D: five clocks, four unknowns. One simulated run.",
        ),
        Embed(
            "demo_multi_mass_2d",
            "Animation: two masses on a plane",
            "Two masses in 2D: ten randomly placed clocks, six unknowns. "
            "One simulated run.",
        ),
    ),
    "story/how-many-masses.qmd": (
        Embed(
            "demo_model_comparison",
            "Animation: the probability of one, two and three masses",
            "One simulated run: posterior probability over K=1, 2, 3 as 80 "
            "observations accumulate. Truth is K=2.",
        ),
    ),
    "story/gravitational-echolocation.qmd": (
        Embed(
            "demo_echolocation_3d",
            "Animation: the head locating an exterior mass",
            "The echolocation demo, one simulated run. Center: the head, its "
            "clocks, the cloud and the true mass (ring). Upper left: the whole "
            "search box. Lower left: the fitted readings against the observed "
            "ones: agreement with the readings used in the fit, not a test on new "
            "data. Right: the four estimates after each observation.",
        ),
    ),
}
FIGURE_PAGES: tuple[str, ...] = tuple(sorted(set(PLOT_PAGES) | set(REPLAY_EMBEDS)))
# A digit, ASCII, subscript or superscript: what a description's leftover text
# may not hold.
STRAY = re.compile(r"[0-9₀-₉¹²³⁰⁴-⁹]")
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
    """Return every Python cell of a page that draws a figure, in order."""
    return [cell for cell in _python_cells(path) if _is_plot_cell(cell)]


# Heights of the figures the builders draw (CSS pixels), by name in clocks.viz.
FIGURE_FUNCTION_HEIGHTS: dict[str, int] = {
    "density_figure": 540,
    "range_study_figure": 540,
}

# Retired image files and library modules that draw figures; later tasks add to
# these.
RETIRED_ASSETS: tuple[str, ...] = (
    "demo_density.png",
    "echolocation_range_study.png",
    "demo_1d.gif",
    "demo_2d.gif",
    "demo_multi_mass.gif",
    "demo_multi_mass_2d.gif",
    "demo_model_comparison.gif",
    "demo_echolocation_3d.gif",
)
LIBRARY_DRAWING_MODULES: tuple[str, ...] = (
    "src/clocks/_figures.py",
    "src/clocks/_demos/demo_density.py",
    "src/clocks/_echo_study.py",
    "src/clocks/_clip_parts.py",
    "src/clocks/_clip_maps.py",
    "src/clocks/_clip_models.py",
    "src/clocks/_clip_tracks.py",
    "src/clocks/_clip_echo.py",
    "src/clocks/_clips.py",
)


def _is_plot_cell(cell: str) -> bool:
    if any(line.startswith("#| renderings:") for line in cell.splitlines()):
        return True
    for node in ast.walk(ast.parse(cell)):
        if not isinstance(node, ast.Call):
            continue
        if _is_show_call(node) or _dotted(node.func) in {"plt.subplots", "plt.figure"}:
            return True
    return bool(_facade_builders(cell))


def _dotted(node: ast.expr) -> str | None:
    """Return ``a.b.c`` for a chain of names and attributes, else None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return None if base is None else f"{base}.{node.attr}"
    return None


def _is_show_call(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Name) and node.func.id == "show"


def _builder_calls(tree: ast.AST) -> list[tuple[ast.Call, str]]:
    """Return each call of a figure builder, in call order.

    Two routes find them. By import: a ``clocks.viz`` ``*_figure`` name, alias
    or module imported in the same tree. By name: any call whose function is a
    name, or an attribute whose last part, equal to a real builder's name, so
    a builder reached through another module still counts. An unrelated
    ``helper_figure()`` is neither.
    """
    names: dict[str, str] = {}
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "clocks.viz":
            for alias in node.names:
                if alias.name.endswith("_figure"):
                    names[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module == "clocks":
            for alias in node.names:
                if alias.name == "viz":
                    modules.add(alias.asname or "viz")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "clocks.viz":
                    modules.add(alias.asname or "clocks.viz")
    found: list[tuple[ast.Call, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in names:
            found.append((node, names[func.id]))
        elif isinstance(func, ast.Name) and func.id in BUILDER_NAMES:
            found.append((node, func.id))
        elif (
            isinstance(func, ast.Attribute)
            and func.attr.endswith("_figure")
            and _dotted(func.value) in modules
        ):
            found.append((node, func.attr))
        elif isinstance(func, ast.Attribute) and func.attr in BUILDER_NAMES:
            found.append((node, func.attr))
    found.sort(key=lambda item: (item[0].lineno, item[0].col_offset))
    return found


def _facade_builders(cell: str) -> list[str]:
    """Return the ``clocks.viz`` ``*_figure`` names the cell calls, in order."""
    return [name for _, name in _builder_calls(ast.parse(cell))]


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }


def _style_block(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> ast.With | None:
    """Return the innermost ``with`` around a node if it is a style block.

    A style block is ``with style(v)`` as a statement of the body (not the
    ``orelse``) of ``for v in THEMES``.
    """
    current = parents.get(node)
    while current is not None and not isinstance(current, ast.With):
        current = parents.get(current)
    if current is None:
        return None
    loop = parents.get(current)
    if not (
        isinstance(loop, ast.For)
        and isinstance(loop.target, ast.Name)
        and isinstance(loop.iter, ast.Name)
        and loop.iter.id == "THEMES"
        and any(current is statement for statement in loop.body)
    ):
        return None
    for item in current.items:
        call = item.context_expr
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "style"
            and len(call.args) == 1
            and not call.keywords
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id == loop.target.id
        ):
            return current
    return None


def _is_show_of(statement: ast.stmt, name: str) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and _is_show_call(statement.value)
        and len(statement.value.args) == 1
        and not statement.value.keywords
        and isinstance(statement.value.args[0], ast.Name)
        and statement.value.args[0].id == name
    )


def _is_canonical(call: ast.Call, block: ast.With) -> bool:
    """Does the block hold exactly the builder assignment and its ``show``?

    The ``with`` has one item, ``style(...) as <palette>``; its body is exactly
    ``<fig> = <builder>(<data>, <palette>)`` then ``show(<fig>)``. The palette
    is passed positionally and alone, and nothing else can rebind it, so a
    figure cannot be drawn with another theme's colours.
    """
    if len(block.items) != 1 or len(block.body) != 2:
        return False
    palette = block.items[0].optional_vars
    statement, following = block.body
    return (
        isinstance(palette, ast.Name)
        and isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.value is call
        and not call.keywords
        and len(call.args) == 2
        and isinstance(call.args[1], ast.Name)
        and call.args[1].id == palette.id
        and _is_show_of(following, statement.targets[0].id)
    )


def _style_block_violations(cell: str) -> list[str]:
    """Return how a cell with figure builders departs from the canonical form."""
    tree = ast.parse(cell)
    parents = _parents(tree)
    builders = _builder_calls(tree)
    shows = sorted(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and _is_show_call(node)
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )
    blocks = {call: _style_block(call, parents) for call, _ in builders}
    messages = [
        f"{name}() is not inside a style(theme) block of a THEMES loop"
        for call, name in builders
        if blocks[call] is None
    ]
    messages += [
        "show() is not inside a style(theme) block of a THEMES loop"
        for call in shows
        if _style_block(call, parents) is None
    ]
    messages += [
        f"{name}() is not `<name> = {name}(<data>, <palette>)` followed by "
        "`show(<name>)`, the only two statements of a one-item "
        "`with style(theme) as <palette>`"
        for call, name in builders
        if blocks[call] is not None and not _is_canonical(call, blocks[call])
    ]
    return messages


def _cell_height(cell: str) -> int:
    builders = _facade_builders(cell)
    if builders:
        return FIGURE_FUNCTION_HEIGHTS[builders[0]]
    return 540 if "figsize=(7.5, 5.4)" in cell else 420


def _retired_asset_mentions(text: str, retired: Sequence[str]) -> list[str]:
    """Return each mention of a retired file that is not the demo's output path.

    The text is normalised the way a reader of the link would resolve it, each
    step working on what the one before decoded: HTML entities, percent
    escapes, Markdown backslash escapes, then tabs and carriage returns (a URL
    parser strips them; newlines stay so line numbers do not move). No link
    grammar is parsed: a name must follow a standalone ``output/``.
    """
    text = html.unescape(text)
    text = urllib.parse.unquote(text)
    text = re.sub(rf"\\([{re.escape(string.punctuation)}])", r"\1", text)
    text = text.replace("\t", "").replace("\r", "")
    mentions = []
    for number, line in enumerate(text.split("\n"), start=1):
        for name in retired:
            for match in re.finditer(re.escape(name), line):
                if not re.search(r"(?<![\w./-])output/$", line[: match.start()]):
                    mentions.append(f"line {number}: {name}")
    return mentions


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


PLOT_CELLS = [
    pytest.param(page, index, id=f"{page}#{index + 1}")
    for page in PLOT_PAGES
    for index in range(len(_plot_cells(SITE / page)))
]


def test_plot_pages_are_exactly_the_known_nine() -> None:
    found = sorted(
        path.relative_to(SITE).as_posix()
        for path in SITE.rglob("*.qmd")
        if not _EXCLUDED_DIRS & set(path.relative_to(SITE).parts) and _plot_cells(path)
    )
    assert found == sorted(PLOT_PAGES)


@pytest.mark.parametrize(("page", "index"), PLOT_CELLS)
def test_cell_declares_both_renderings(page: str, index: int) -> None:
    cell = _plot_cells(SITE / page)[index]
    assert "#| renderings: [light, dark]" in cell.splitlines()


@pytest.mark.parametrize(("page", "index"), PLOT_CELLS)
def test_cell_gives_one_identical_caption_per_rendering(page: str, index: int) -> None:
    entries = _caption_entries(_plot_cells(SITE / page)[index])
    where = f"{page}#{index + 1}"
    assert len(entries) == 2, f"{where}: caption entries {entries!r}"
    assert entries[0], f"{where}: empty caption"
    assert entries[0] == entries[1], f"{where}: captions differ"


@pytest.mark.parametrize(("page", "index"), PLOT_CELLS)
def test_cell_draws_through_the_style_and_show(page: str, index: int) -> None:
    cell = _plot_cells(SITE / page)[index]
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
            # With data=, the first two positional arguments may be column
            # names; a string at any later position is a format string.
            has_data = any(keyword.arg == "data" for keyword in node.keywords)
            positional = node.args[2:] if has_data else node.args
            if any(_is_string(arg) for arg in positional):
                violations.append("positional format string")
    return violations


def _is_string(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _string_constants(node: ast.expr) -> list[str]:
    """Return the string constants a value is or, as a list or tuple, holds."""
    elements = node.elts if isinstance(node, (ast.List, ast.Tuple)) else [node]
    return [element.value for element in elements if _is_string(element)]


@pytest.mark.parametrize(("page", "index"), PLOT_CELLS)
def test_cell_has_no_unstyled_colour_or_display(page: str, index: int) -> None:
    violations = _style_violations(_plot_cells(SITE / page)[index])
    assert violations == [], f"{page}#{index + 1}: {violations}"


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
        pytest.param(
            'ax.plot(x, "rate", "r-", data=frame)',
            "positional format string",
            id="data_third_argument",
        ),
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


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        pytest.param("#| renderings: [light, dark]\nx = 1", True, id="renderings"),
        pytest.param("fig, ax = plt.subplots()", True, id="subplots"),
        pytest.param("fig = plt.figure()", True, id="figure"),
        pytest.param("show(fig)", True, id="show"),
        pytest.param(
            "from clocks.viz import density_figure\n"
            "fig = density_figure(replay, colors)",
            True,
            id="figure_function",
        ),
        pytest.param(
            "from clocks.viz import density_figure as draw\nfig = draw(replay, colors)",
            True,
            id="aliased",
        ),
        pytest.param(
            "import clocks.viz as viz\nfig = viz.range_study_figure(study, colors)",
            True,
            id="viz_module",
        ),
        pytest.param(
            "import clocks.viz\nfig = clocks.viz.density_figure(replay, colors)",
            True,
            id="qualified",
        ),
        pytest.param(
            "def helper_figure():\n    return 1\n\nhelper_figure()",
            False,
            id="helper_figure",
        ),
        pytest.param(
            "import other\nfig = other.helper_figure()", False, id="attribute_helper"
        ),
        pytest.param(
            "fig = density_figure(replay, colors)", True, id="unimported_figure"
        ),
        # The stated limit: a builder imported from a private module under
        # another name is found by neither route.
        pytest.param(
            "from clocks._figures import density_figure as draw\nfig = draw(r, c)",
            False,
            id="private_alias",
        ),
        pytest.param(
            "from clocks import viz\nfig = viz.density_figure(replay, colors)",
            True,
            id="from_clocks_import_viz",
        ),
        pytest.param(
            "from clocks import viz as v\nfig = v.density_figure(replay, colors)",
            True,
            id="from_clocks_import_viz_aliased",
        ),
        pytest.param(
            "from clocks import _figures\nfig = _figures.density_figure(r, c)",
            True,
            id="through_another_module",
        ),
        pytest.param(
            "from clocks._figures import density_figure\nfig = density_figure(r, c)",
            True,
            id="bare_name_from_private_module",
        ),
        pytest.param("table = Markdown(text)\nprint(table)", False, id="not_a_plot"),
        pytest.param("ax.figure.canvas.draw()", False, id="a_method_named_figure"),
    ],
)
def test_a_cell_is_a_plot_cell(cell: str, expected: bool) -> None:
    assert _is_plot_cell(cell) is expected


def test_builder_names_are_the_two_real_builders() -> None:
    assert BUILDER_NAMES == {"density_figure", "range_study_figure"}
    assert BUILDER_NAMES == set(BUILDER_CELL_PROGRAMS)


def test_facade_builders_are_resolved() -> None:
    cell = (
        "from clocks.viz import density_figure as draw\n"
        "import clocks.viz as viz\n"
        "import clocks.viz\n"
        "draw(r, c)\n"
        "viz.range_study_figure(s, c)\n"
        "clocks.viz.density_figure(r, c)\n"
        "helper_figure()\n"
        "other.helper_figure()"
    )
    assert _facade_builders(cell) == [
        "density_figure",
        "range_study_figure",
        "density_figure",
    ]


def test_facade_builders_come_back_in_source_order() -> None:
    cell = (
        "from clocks.viz import density_figure, range_study_figure\n"
        "for theme in THEMES:\n"
        "    with style(theme) as colors:\n"
        "        range_study_figure(s, colors)\n"
        "density_figure(r, colors)\n"
    )
    assert _facade_builders(cell) == ["range_study_figure", "density_figure"]


_HEAD = "from clocks.viz import THEMES, density_figure, show, style\n"
_NOT_IN_BLOCK = "density_figure() is not inside a style(theme) block of a THEMES loop"
_SHOW_NOT_IN_BLOCK = "show() is not inside a style(theme) block of a THEMES loop"
_NOT_CANONICAL = (
    "density_figure() is not `<name> = density_figure(<data>, <palette>)` "
    "followed by `show(<name>)`, the only two statements of a one-item "
    "`with style(theme) as <palette>`"
)
_CANONICAL = (
    "for theme in THEMES:\n"
    "    with style(theme) as colors:\n"
    "        fig = density_figure(replay, colors)\n"
    "        show(fig)\n"
)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        pytest.param(_CANONICAL, [], id="canonical"),
        pytest.param(
            _CANONICAL.replace("style(theme)", 'style("light")'),
            [_NOT_IN_BLOCK, _SHOW_NOT_IN_BLOCK],
            id="fixed_theme",
        ),
        pytest.param(
            "other_theme = THEMES[0]\n"
            + _CANONICAL.replace("style(theme)", "style(other_theme)"),
            [_NOT_IN_BLOCK, _SHOW_NOT_IN_BLOCK],
            id="other_variable",
        ),
        pytest.param(
            "for theme in THEMES:\n"
            "    pass\n"
            "else:\n"
            "    with style(theme) as colors:\n"
            "        fig = density_figure(replay, colors)\n"
            "        show(fig)\n",
            [_NOT_IN_BLOCK, _SHOW_NOT_IN_BLOCK],
            id="loop_else",
        ),
        pytest.param(
            "# for theme in THEMES: with style(theme) as colors: show(fig)\n"
            "fig = density_figure(replay, colors)\n"
            "show(fig)\n",
            [_NOT_IN_BLOCK, _SHOW_NOT_IN_BLOCK],
            id="comment_only",
        ),
        pytest.param(
            "for theme in THEMES:\n"
            "    with style(theme) as colors:\n"
            "        fig = density_figure(replay, colors)\n"
            "    show(fig)\n",
            [_SHOW_NOT_IN_BLOCK, _NOT_CANONICAL],
            id="show_outside",
        ),
        pytest.param(
            "for theme in THEMES:\n"
            "    fig = density_figure(replay, colors)\n"
            "    with style(theme) as colors:\n"
            "        show(fig)\n",
            [_NOT_IN_BLOCK],
            id="builder_outside",
        ),
        pytest.param(
            "for theme in THEMES:\n"
            "    with style(theme) as colors:\n"
            "        fig = density_figure(replay, colors)\n"
            "    with style(theme) as colors:\n"
            "        show(fig)\n",
            [_NOT_CANONICAL],
            id="two_blocks",
        ),
        pytest.param(
            _CANONICAL.replace("show(fig)", "# show(fig)"),
            [_NOT_CANONICAL],
            id="commented_out_show",
        ),
        pytest.param(
            _CANONICAL.replace(
                "        fig = density_figure",
                "        other = plt.figure()\n        fig = density_figure",
            ).replace("show(fig)", "show(other)"),
            [_NOT_CANONICAL],
            id="show_of_another_figure",
        ),
        pytest.param(
            _CANONICAL.replace(
                "        show(fig)",
                "        fig = plt.figure(figsize=(7.5, 5.4))\n        show(fig)",
            ),
            [_NOT_CANONICAL],
            id="reassigned_before_show",
        ),
        pytest.param(
            _CANONICAL.replace(
                "        show(fig)", "        print(theme)\n        show(fig)"
            ),
            [_NOT_CANONICAL],
            id="statement_between",
        ),
        pytest.param(
            "for theme in THEMES:\n"
            "    with style(theme) as colors:\n"
            "        show(density_figure(replay, colors))\n",
            [_NOT_CANONICAL],
            id="show_of_the_call",
        ),
        pytest.param(
            _CANONICAL.replace("THEMES:", '("light", "dark"):'),
            [_NOT_IN_BLOCK, _SHOW_NOT_IN_BLOCK],
            id="other_loop",
        ),
        pytest.param(
            _CANONICAL.replace(
                "density_figure(replay", "wrap(density_figure(replay"
            ).replace("colors)\n", "colors))\n", 1),
            [_NOT_CANONICAL],
            id="wrapped_call",
        ),
        pytest.param(
            _CANONICAL.replace("fig = ", "a = fig = "),
            [_NOT_CANONICAL],
            id="two_targets",
        ),
        pytest.param(
            _CANONICAL.replace("fig = ", "fig = a = "),
            [_NOT_CANONICAL],
            id="two_targets_shown_first",
        ),
        pytest.param(
            _CANONICAL.replace("THEMES:", "OTHER_THEMES:"),
            [_NOT_IN_BLOCK, _SHOW_NOT_IN_BLOCK],
            id="other_loop_name",
        ),
        pytest.param(
            _CANONICAL.replace(
                "density_figure(replay, colors)",
                'density_figure(replay, palette("light"))',
            ),
            [_NOT_CANONICAL],
            id="fixed_palette",
        ),
        pytest.param(
            'other = palette("light")\n'
            + _CANONICAL.replace("(replay, colors)", "(replay, other)"),
            [_NOT_CANONICAL],
            id="other_palette_name",
        ),
        pytest.param(
            'colors = palette("light")\n' + _CANONICAL.replace(" as colors", ""),
            [_NOT_CANONICAL],
            id="no_as_name",
        ),
        pytest.param(
            _CANONICAL.replace("replay, colors)", "replay, colors, **{})"),
            [_NOT_CANONICAL],
            id="palette_by_keyword",
        ),
        pytest.param(
            _CANONICAL.replace(
                "        fig = density_figure",
                '        colors = palette("light")\n        fig = density_figure',
            ),
            [_NOT_CANONICAL],
            id="palette_rebound_in_block",
        ),
        pytest.param(
            _CANONICAL + "        print(theme)\n",
            [_NOT_CANONICAL],
            id="statement_after_show",
        ),
        pytest.param(
            _CANONICAL.replace("as colors:", 'as colors, open("x") as f:'),
            [_NOT_CANONICAL],
            id="two_with_items",
        ),
        pytest.param(
            "for theme in THEMES:\n"
            "    with style(theme) as colors, density_figure(replay, colors) as fig:\n"
            "        show(fig)\n",
            [_NOT_CANONICAL],
            id="builder_in_with_item",
        ),
    ],
)
def test_style_block_violations(body: str, expected: list[str]) -> None:
    assert _style_block_violations(_HEAD + body) == expected


BUILDER_CELLS = [
    case
    for case in PLOT_CELLS
    if _facade_builders(_plot_cells(SITE / case.values[0])[case.values[1]])
]


@pytest.mark.parametrize(("page", "index"), BUILDER_CELLS)
def test_builder_calls_and_show_are_inside_the_style_block(
    page: str, index: int
) -> None:
    cell = _plot_cells(SITE / page)[index]
    assert _style_block_violations(cell) == [], f"{page}#{index + 1}"


def _manifest_page(page: str) -> str:
    return page.replace(".qmd", ".html")


def _description_id(clip: str) -> str:
    """Return the id of a clip's description: ``demo_2d`` -> ``demo-2d-text``."""
    return f"{clip.replace('_', '-')}-text"


def _embed_block(embed: Embed, page: str) -> str:
    """Return the canonical raw block that embeds a clip on a page.

    Its paths are relative to the page, and its last line ends with a newline,
    so the closing fence is a whole line.
    """
    up = "../" * (len(PurePosixPath(page).parts) - 1)
    videos = [
        f'<video class="{theme}-content" aria-label="{embed.label}" '
        f'aria-describedby="{_description_id(embed.clip)}" '
        f'src="{up}media/{embed.clip}.{theme}.mp4" '
        f'poster="{up}media/{embed.clip}.{theme}.png" '
        f'width="750" height="{CLIP_HEIGHTS[embed.clip]}" '
        'controls playsinline preload="none"></video>'
        for theme in ("light", "dark")
    ]
    lines = [
        "```{=html}",
        '<figure class="replay">',
        *videos,
        f"<figcaption>{embed.caption}</figcaption>",
        "</figure>",
        "```",
    ]
    return "\n".join(lines) + "\n"


_VIDEO_MARKUP = (
    "<video",
    "<source",
    "<figure",
    "<iframe",
    ".mp4",
    "{=html}",
    "{{< video",
)
_DIV_CLOSE = re.compile(r"^:{3,}[ \t]*$")


def _replay_embed_violations(
    text: str, page: str, embeds: Sequence[Embed]
) -> list[str]:
    """Return how a page's source departs from its pinned clip embeds.

    Each embed's block must occur exactly once, the blocks in the table's
    order, each followed by its non-empty description div whose id occurs once;
    with every pinned block removed, no line may hold video markup. Nothing is
    said about what surrounds a block (see the module docstring).
    """
    blocks = [_embed_block(embed, page) for embed in embeds]
    messages: list[str] = []
    starts: list[int] = []
    for number, (embed, block) in enumerate(zip(embeds, blocks, strict=True), 1):
        count = text.count(block)
        if count == 0:
            messages.append(
                f"embed {number} ({embed.clip}): the pinned block is missing or altered"
            )
        elif count > 1:
            messages.append(
                f"embed {number} ({embed.clip}): the pinned block occurs {count} times"
            )
        else:
            starts.append(text.index(block))
    if len(starts) == len(blocks) and starts != sorted(starts):
        messages.append("the embeds are out of order")
    for number, (embed, block) in enumerate(zip(embeds, blocks, strict=True), 1):
        if text.count(block) == 1:
            messages += _description_violations(text, number, embed, block)
    remainder = text
    for block in blocks:
        remainder = remainder.replace(block, "\n" * block.count("\n"))
    messages += [
        f"unpinned video markup: line {number}"
        for number, line in enumerate(remainder.split("\n"), start=1)
        if any(marker in line.lower() for marker in _VIDEO_MARKUP)
    ]
    return messages


def _description_violations(
    text: str, number: int, embed: Embed, block: str
) -> list[str]:
    """Return how the description after an embed's one block departs from the pin."""
    where = f"embed {number} ({embed.clip})"
    div_id = _description_id(embed.clip)
    messages: list[str] = []
    following = text[text.index(block) + len(block) :].split("\n")
    nonblank = [index for index, line in enumerate(following) if line.strip()]
    opening = f"::: {{#{div_id}}}"
    if not nonblank or following[nonblank[0]] != opening:
        messages.append(f"{where}: not followed by its description `{opening}`")
    else:
        body = []
        for line in following[nonblank[0] + 1 :]:
            if _DIV_CLOSE.match(line):
                break
            body.append(line)
        if not any(line.strip() for line in body):
            messages.append(f"{where}: the description is empty")
    count = text.count(f"{{#{div_id}}}")
    if count > 1:
        messages.append(f"{where}: the description id occurs {count} times")
    return messages


_DIV_OPEN = re.compile(r"^:{3,}[ \t]*\{#([^\s{}]+)\}[ \t]*$")


def _descriptions(text: str) -> dict[str, str]:
    """Map the id of each ``::: {#<id>}`` div to its whitespace-normalised body.

    A body runs to the next closing ``:::`` line. Divs with classes or other
    attributes (a callout, say) are not descriptions and are left out.
    """
    found: dict[str, str] = {}
    lines = text.split("\n")
    for index, line in enumerate(lines):
        match = _DIV_OPEN.match(line)
        if match is None:
            continue
        body = []
        for following in lines[index + 1 :]:
            if _DIV_CLOSE.match(following):
                break
            body.append(following)
        found.setdefault(match.group(1), " ".join(" ".join(body).split()))
    return found


def _page_figures(text: str, page: str) -> list[tuple[str, int, str | None]]:
    """Return a page's plot cells and pinned clip blocks, in source order.

    A plot cell is ``("plot", height, None)``; each occurrence of a block that
    ``REPLAY_EMBEDS`` lists for the page is ``("clip", height, clip)``.
    """
    found: list[tuple[int, tuple[str, int, str | None]]] = [
        (match.start(), ("plot", _cell_height(match.group(1)), None))
        for match in _CELL.finditer(text)
        if _is_plot_cell(match.group(1))
    ]
    for embed in REPLAY_EMBEDS.get(page, ()):
        block = _embed_block(embed, page)
        found += [
            (match.start(), ("clip", CLIP_HEIGHTS[embed.clip], embed.clip))
            for match in re.finditer(re.escape(block), text)
        ]
    return [figure for _, figure in sorted(found, key=lambda item: item[0])]


def _without(description: str, sentences: Sequence[str]) -> str:
    """Return the description with each sentence removed once, in turn.

    Raises AssertionError naming the first sentence not in what is left.
    """
    for sentence in sentences:
        assert sentence in description, f"not in the description: {sentence!r}"
        description = description.replace(sentence, "", 1)
    return description


@pytest.mark.parametrize("page", FIGURE_PAGES)
def test_manifest_matches_the_source(page: str) -> None:
    figures = check_site_figures.EXPECTED_FIGURES[_manifest_page(page)]
    from_manifest = tuple((fig.kind, fig.height, fig.clip) for fig in figures)
    text = (SITE / page).read_text(encoding="utf-8")
    assert from_manifest == tuple(_page_figures(text, page))


def test_the_manifest_lists_exactly_the_figure_pages() -> None:
    assert set(check_site_figures.EXPECTED_FIGURES) == {
        _manifest_page(page) for page in FIGURE_PAGES
    }
    assert len(PLOT_PAGES) == 9
    assert len(FIGURE_PAGES) == 10


_PAGE = "story/into-the-plane.qmd"
_EMBED_2D = Embed("demo_2d", "Animation: the 2D search", "A caption.")


def test_embed_block_is_the_pinned_markup() -> None:
    """Only this catches a generated block that drifts from the design's markup."""
    pinned = (
        "```{=html}\n"
        '<figure class="replay">\n'
        '<video class="light-content" aria-label="Animation: the 2D search" '
        'aria-describedby="demo-2d-text" src="../media/demo_2d.light.mp4" '
        'poster="../media/demo_2d.light.png" width="750" height="450" '
        'controls playsinline preload="none"></video>\n'
        '<video class="dark-content" aria-label="Animation: the 2D search" '
        'aria-describedby="demo-2d-text" src="../media/demo_2d.dark.mp4" '
        'poster="../media/demo_2d.dark.png" width="750" height="450" '
        'controls playsinline preload="none"></video>\n'
        "<figcaption>A caption.</figcaption>\n"
        "</figure>\n"
        "```\n"
    )
    assert _embed_block(_EMBED_2D, _PAGE) == pinned
    at_root = _embed_block(_EMBED_2D, "index.qmd")
    paths = re.findall(r'(?:src|poster)="([^"]*)"', at_root)
    assert len(paths) == 4
    assert all(path.startswith("media/") for path in paths)
    assert at_root == pinned.replace('"../media/', '"media/')


# Built with _embed_block, which the test above pins, so each test below fails
# for its own rule.
_BLOCK_2D = _embed_block(_EMBED_2D, _PAGE)
PAGE_TEXT = "A paragraph.\n\n" + _BLOCK_2D + "\n::: {#demo-2d-text}\nWords.\n:::\n"
_ALTERED = ["embed 1 (demo_2d): the pinned block is missing or altered"]
_NOT_FOLLOWED = [
    "embed 1 (demo_2d): not followed by its description `::: {#demo-2d-text}`"
]
_MASS = Embed("demo_multi_mass", "Animation: one mass", "One.")
_MASS_2D = Embed("demo_multi_mass_2d", "Animation: two masses", "Two.")
_SWAPPED = (
    _embed_block(_MASS_2D, _PAGE)
    + "\n::: {#demo-multi-mass-2d-text}\nB.\n:::\n\n"
    + _embed_block(_MASS, _PAGE)
    + "\n::: {#demo-multi-mass-text}\nA.\n:::\n"
)


def _altered(pattern: str, new: str, count: int = 0) -> str:
    """Return PAGE_TEXT with a change made where only the pinned block matches."""
    return re.sub(pattern, new, PAGE_TEXT, count=count)


@pytest.mark.parametrize(
    ("text", "embeds", "expected"),
    [
        pytest.param(PAGE_TEXT, (_EMBED_2D,), [], id="canonical"),
        pytest.param(
            _altered("></video>", " autoplay></video>", 1),
            (_EMBED_2D,),
            _ALTERED,
            id="autoplay_added",
        ),
        pytest.param(
            _altered("A caption.", "Another caption."),
            (_EMBED_2D,),
            _ALTERED,
            id="caption_changed",
        ),
        pytest.param(
            _altered('height="450"', 'height="500"'),
            (_EMBED_2D,),
            _ALTERED,
            id="height_changed",
        ),
        pytest.param(
            _altered('<video class="dark-content".*\n', ""),
            (_EMBED_2D,),
            _ALTERED,
            id="dark_video_removed",
        ),
        pytest.param(
            _altered("```\\{=html\\}\n|(?<=</figure>\n)```\n", ""),
            (_EMBED_2D,),
            _ALTERED,
            id="fence_removed",
        ),
        pytest.param(
            PAGE_TEXT + "\n" + _BLOCK_2D,
            (_EMBED_2D,),
            ["embed 1 (demo_2d): the pinned block occurs 2 times"],
            id="block_twice",
        ),
        pytest.param(
            PAGE_TEXT + '<video src="x.mp4"></video>\n',
            (_EMBED_2D,),
            ["unpinned video markup: line 14"],
            id="a_second_video",
        ),
        pytest.param(
            PAGE_TEXT.replace(
                "A paragraph.", "A paragraph, [the clip](../media/demo_2d.light.mp4)."
            ),
            (_EMBED_2D,),
            ["unpinned video markup: line 1"],
            id="a_link_to_a_clip",
        ),
        pytest.param(
            PAGE_TEXT + '<iframe src="x.html"></iframe>\n',
            (_EMBED_2D,),
            ["unpinned video markup: line 14"],
            id="an_iframe",
        ),
        pytest.param(
            PAGE_TEXT + '<source src="x.webm">\n',
            (_EMBED_2D,),
            ["unpinned video markup: line 14"],
            id="a_source_element",
        ),
        pytest.param(
            PAGE_TEXT + '<VIDEO SRC="x.webm"></VIDEO>\n',
            (_EMBED_2D,),
            ["unpinned video markup: line 14"],
            id="upper_case_video",
        ),
        pytest.param(
            PAGE_TEXT + "\n```{=html}\n<p>Words.</p>\n```\n",
            (_EMBED_2D,),
            ["unpinned video markup: line 15"],
            id="another_raw_block",
        ),
        pytest.param(
            PAGE_TEXT + "{{< video https://www.example.com/clip >}}\n",
            (_EMBED_2D,),
            ["unpinned video markup: line 14"],
            id="a_video_shortcode",
        ),
        pytest.param(
            PAGE_TEXT.replace("\n::: {#demo-2d-text}\nWords.\n:::\n", ""),
            (_EMBED_2D,),
            _NOT_FOLLOWED,
            id="description_missing",
        ),
        pytest.param(
            PAGE_TEXT.replace(
                "\n::: {#demo-2d-text}", "\nA paragraph between.\n\n::: {#demo-2d-text}"
            ),
            (_EMBED_2D,),
            _NOT_FOLLOWED,
            id="description_not_adjacent",
        ),
        pytest.param(
            PAGE_TEXT.replace("Words.\n", "\n"),
            (_EMBED_2D,),
            ["embed 1 (demo_2d): the description is empty"],
            id="description_empty",
        ),
        pytest.param(
            PAGE_TEXT + "\n::: {#demo-2d-text}\nMore words.\n:::\n",
            (_EMBED_2D,),
            ["embed 1 (demo_2d): the description id occurs 2 times"],
            id="description_id_twice",
        ),
        pytest.param(
            _SWAPPED,
            (_MASS, _MASS_2D),
            ["the embeds are out of order"],
            id="two_embeds_swapped",
        ),
    ],
)
def test_replay_embed_violations(
    text: str, embeds: tuple[Embed, ...], expected: list[str]
) -> None:
    """Catches an embed edited by hand, a second video outside the table, and
    a description that the videos name but that is not there.

    An altered block is reported as missing, and its markup, no longer pinned,
    as unpinned video markup.
    """
    messages = _replay_embed_violations(text, _PAGE, embeds)
    if expected is _ALTERED:
        assert messages[:1] == _ALTERED
        assert len(messages) > 1
        assert all(
            message.startswith("unpinned video markup: line ")
            for message in messages[1:]
        )
    else:
        assert messages == expected


def test_descriptions_are_found_by_id() -> None:
    text = (
        PAGE_TEXT
        + "\n::: {#other}\nOther   words\nhere.\n:::\n"
        + "\n::: {.callout-note}\nA note.\n:::\n"
    )
    assert _descriptions(text) == {
        "demo-2d-text": "Words.",
        "other": "Other words here.",
    }


def test_page_figures_merges_cells_and_embeds_in_source_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only this catches clips listed after all plots whatever their place."""
    monkeypatch.setattr(sys.modules[__name__], "REPLAY_EMBEDS", {_PAGE: (_EMBED_2D,)})
    text = (
        "```{python}\n#| renderings: [light, dark]\n"
        "fig, ax = plt.subplots(figsize=(7.5, 5.4))\n```\n\n"
        + _BLOCK_2D
        + "\n::: {#demo-2d-text}\nWords.\n:::\n\n"
        "```{python}\n#| renderings: [light, dark]\nfig, ax = plt.subplots()\n```\n"
    )
    assert _page_figures(text, _PAGE) == [
        ("plot", 540, None),
        ("clip", 450, "demo_2d"),
        ("plot", 420, None),
    ]


def test_without_removes_pinned_sentences_and_reports_a_missing_one() -> None:
    assert _without("A 1. B 2.", ["A 1."]) == " B 2."
    assert _without("A 1. B 2.", ["A 1.", " B 2."]) == ""
    with pytest.raises(AssertionError, match="C 3"):
        _without("A 1. B 2.", ["A 1.", "C 3."])


def test_stray_numbers_are_found() -> None:
    """Only this catches a subscript or superscript digit, which ``[0-9]`` does
    not match, passing as no number."""
    assert STRAY.pattern == r"[0-9₀-₉¹²³⁰⁴-⁹]"
    assert STRAY.search("after 12 readings")
    assert STRAY.search("x₃")
    for superscript in "⁰¹²³⁴⁵⁶⁷⁸⁹":
        assert STRAY.search(f"one over R{superscript}"), superscript
    assert STRAY.search("0.4%")
    assert not STRAY.search("after twelve readings, x and M")


def test_embed_labels_and_captions_are_plain_text() -> None:
    for page, embeds in REPLAY_EMBEDS.items():
        for embed in embeds:
            for text in (embed.label, embed.caption):
                assert not set(text) & set('<>&"'), f"{page}: {embed.clip}: {text!r}"


def test_each_clip_is_embedded_at_its_declared_height() -> None:
    """Only this catches a clip dropped consistently from the embeds, the page
    and the manifest (the set equality with ``CLIP_HEIGHTS`` and the count of
    seven)."""
    embedded: list[str] = []
    for page, embeds in REPLAY_EMBEDS.items():
        figures = check_site_figures.EXPECTED_FIGURES[_manifest_page(page)]
        clips = [fig for fig in figures if fig.kind == "clip"]
        for embed, fig in zip(embeds, clips, strict=True):
            assert fig.clip == embed.clip, page
            assert fig.height == CLIP_HEIGHTS[embed.clip], page
            embedded.append(embed.clip)
    assert len(embedded) == 7
    assert set(embedded) == set(CLIP_HEIGHTS)


ARCHIVES = REPO / "assets" / "replays"
ORDINAL = {10: "tenth", 20: "twentieth"}


def _listing(items: Sequence[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    return ", ".join(items[:-1]) + " and " + items[-1] if len(items) > 1 else items[0]


def _number(replay: Replay, j: int, value: float) -> str:
    """``value`` at the trace headers' decimals for parameter ``j``."""
    return f"{value:.{trace_decimals(replay.meta['prior_ranges'][j])}f}"


def _run_ends(replay: Replay) -> str:
    """``After T readings the run ends at x = m ± s, ...``, from the last row."""
    mean = replay.arrays["mean"][-1]
    sd = standard_deviations(replay)[-1]
    ends = [
        f"{name} = {_number(replay, j, mean[j])} ± {_number(replay, j, sd[j])}"
        for j, name in enumerate(replay.meta["param_names"])
    ]
    return (
        f"After {replay.meta['n_observations']} readings the run ends at "
        f"{_listing(ends)}"
    )


def _against_a_true(replay: Replay) -> str:
    truth = replay.arrays["true_params"]
    named = [
        f"{name} = {_number(replay, j, truth[j])}"
        for j, name in enumerate(replay.meta["param_names"])
    ]
    return f", against a true {_listing(named)}."


def _against_true_values(replay: Replay) -> str:
    truth = replay.arrays["true_params"]
    values = [_number(replay, j, value) for j, value in enumerate(truth)]
    return f", against true values of {_listing(values)}."


def _sentences_1d(replay: Replay) -> list[str]:
    sd = standard_deviations(replay)
    cloud = replay.arrays["cloud"][0]
    for j, (lo, hi) in enumerate(replay.meta["prior_ranges"]):
        # "span the prior range": the dots reach within 5% of both ends
        assert cloud[:, j].min() - lo < 0.05 * (hi - lo)
        assert hi - cloud[:, j].max() < 0.05 * (hi - lo)
    assert sd[1, 0] < 0.1 * sd[0, 0]  # "a narrow band" after the first reading
    return [_run_ends(replay) + _against_a_true(replay)]


def _sentences_2d(replay: Replay) -> list[str]:
    sd = standard_deviations(replay)
    mean, truth = replay.arrays["mean"][-1], replay.arrays["true_params"]
    assert sd[8, :2].max() < 0.1 * sd[0, :2].min()  # "by the eighth"
    assert 1.5 < abs(mean[1] - truth[1]) / sd[-1, 1] < 2.5  # "about two of those"
    return [_run_ends(replay) + _against_a_true(replay)]


def _sentences_multi_mass(replay: Replay) -> list[str]:
    sd = standard_deviations(replay)
    mean = replay.arrays["mean"]
    assert sd[1, 1] > 1.5 * sd[1, 0]  # the second "takes longer"
    early = (
        f"after one reading its position is x₂ = {_number(replay, 1, mean[1, 1])} "
        f"± {_number(replay, 1, sd[1, 1])}, and after eight it is "
        f"{_number(replay, 1, mean[8, 1])} ± {_number(replay, 1, sd[8, 1])}."
    )
    # Two of the ± values are from readings 1 and 8, not from the final cloud.
    at_reading = (
        "Each ± is one standard deviation of the cloud for that parameter at that "
        "reading."
    )
    return [early, _run_ends(replay) + _against_true_values(replay), at_reading]


def _sentences_multi_mass_2d(replay: Replay) -> list[str]:
    sd = standard_deviations(replay)
    # "narrow them only partly, and they remain broad": after three readings
    # every position's spread is below its prior's, yet above 40% of it.
    assert (sd[3, :4] < sd[0, :4]).all()
    assert (sd[3, :4] > 0.4 * sd[0, :4]).all()
    return [_run_ends(replay) + _against_true_values(replay)]


def _sentences_model_comparison(replay: Replay) -> list[str]:
    ks = [int(k) for k in replay.arrays["model_ks"]]
    posterior = replay.arrays["model_posterior"]
    prior = replay.arrays["model_prior"]
    share = round(1 / prior[0])
    # "start level": an equal prior, which is the clip's row 0
    np.testing.assert_allclose(prior, 1 / share, rtol=1e-12)
    np.testing.assert_allclose(posterior[0], prior, rtol=1e-12)
    assert len(ks) == share == 3  # "All three candidates"
    assert posterior[1, ks.index(1)] < 1e-5  # "all but ruled out by the first"
    leader = [ks[int(row.argmax())] for row in posterior]
    changes = [k for k in range(2, len(leader)) if leader[k] != leader[k - 1]]
    assert leader[1] == 3
    assert changes == [10, 19]
    assert [leader[k] for k in changes] == [2, 3]
    p2, p3 = posterior[-1, ks.index(2)], posterior[-1, ks.index(3)]
    return [
        f"All three candidates start level, at a prior probability of 1/{share} each",
        "Three masses lead at first, two take over around the "
        f"{ORDINAL[round(changes[0], -1)]} observation, and from about the "
        f"{ORDINAL[round(changes[1], -1)]} three stay ahead: the run ends at about "
        f"{round(100 * p2)}% for two masses and {round(100 * p3)}% for three",
    ]


def _sentences_echolocation(replay: Replay) -> list[str]:
    lo, hi = replay.meta["prior_ranges"][0]
    weight = replay.arrays["weight_in_view"]
    return [
        f"Before any reading the dots fill the whole search box, {hi - lo:g} units "
        f"on a side, and only {100 * weight[0]:.1f}% of the prior lies inside the "
        "close view around the head.",
        f"where {100 * weight[1:].min():.1f}% of its weight stays",
        _run_ends(replay) + _against_true_values(replay),
    ]


# Clip -> the sentences of its description that hold a number, built from the
# archive (with the qualitative words checked against it on the way).
DESCRIPTION_SENTENCES = {
    "demo_1d": _sentences_1d,
    "demo_2d": _sentences_2d,
    "demo_multi_mass": _sentences_multi_mass,
    "demo_multi_mass_2d": _sentences_multi_mass_2d,
    "demo_model_comparison": _sentences_model_comparison,
    "demo_echolocation_3d": _sentences_echolocation,
}
EMBED_CASES = [
    pytest.param(page, embed, id=f"{page}:{embed.clip}")
    for page, embeds in REPLAY_EMBEDS.items()
    for embed in embeds
]


@pytest.mark.parametrize(("page", "embed"), EMBED_CASES)
def test_description_numbers_are_the_archives(page: str, embed: Embed) -> None:
    """Only this catches a description whose numbers belong to another run, or
    a number added to a description without a check."""
    text = (SITE / page).read_text(encoding="utf-8")
    description = _descriptions(text)[_description_id(embed.clip)]
    replay = load_replay(ARCHIVES / f"{embed.clip}.npz")
    sentences = DESCRIPTION_SENTENCES[embed.clip](replay)
    assert STRAY.search(_without(description, sentences)) is None


def test_the_two_2d_descriptions_are_the_same() -> None:
    """Only this catches one of the two being edited alone (decision D24)."""
    landing = _descriptions((SITE / "index.qmd").read_text(encoding="utf-8"))
    plane = _descriptions((SITE / "story/into-the-plane.qmd").read_text("utf-8"))
    assert landing["demo-2d-text"] == plane["demo-2d-text"]


def _plane_cell() -> str:
    (cell,) = _plot_cells(SITE / "story/into-the-plane.qmd")
    return cell


def test_the_plane_cell_uses_the_demos_truth() -> None:
    """Only this catches the page's cell left at the old truth."""
    cell = _plane_cell()

    assert f"masses=np.array([{demo_2d.TRUE_M:.2f}])" in cell
    assert f"positions=np.array([[{demo_2d.TRUE_X}, {demo_2d.TRUE_Y}]])" in cell
    assert "mass_range=(0.005, 0.15)" in cell
    assert "n_observations=25, seed=42" in cell
    assert "n_particles=500" in cell


def test_the_plane_cells_knot_is_offset_by_about_its_width() -> None:
    """The cell's run, from its literals with the demo's mass: the cloud's
    centre lies about one spread from the true position (0.94 at M = 0.10), and
    the page says so rather than calling it a knot at the truth."""
    clock_array = ClockArray(
        positions=np.array(
            [
                [-4.0, 0.0],
                [-2.0, 3.0],
                [1.0, 4.0],
                [4.0, 2.0],
                [5.0, -1.0],
                [2.0, -4.0],
                [-1.0, -3.0],
                [-3.0, -1.5],
            ]
        ),
        track_offset=3.0,
    )
    noise = NoiseConfig(observation_std=0.005)
    truth = MassConfig(
        positions=np.array([[demo_2d.TRUE_X, demo_2d.TRUE_Y]]),
        masses=np.array([demo_2d.TRUE_M]),
    )
    sim = simulate(
        SimulationConfig(
            clock_array=clock_array,
            ground_truth=truth,
            noise=noise,
            n_observations=25,
            seed=42,
        )
    )
    result = infer(
        sim.observations,
        InferenceConfig(
            clock_array=clock_array,
            noise=noise,
            prior=PriorConfig(position_range=(-8.0, 8.0), mass_range=(0.005, 0.15)),
            n_particles=500,
            n_masses=1,
            ess_target=0.8,
            rejuvenation_steps=2,
            proposal_scale=2.38,
            seed=42,
        ),
    )
    offset = np.hypot(*(result.posterior_mean[:2] - truth.positions[0]))
    ratio = offset / np.mean(result.posterior_std[:2])
    page = " ".join((SITE / "story/into-the-plane.qmd").read_text("utf-8").split())

    assert 0.5 < ratio < 1.5
    assert (
        "to a small knot beside the true position, offset from it by about its own "
        "width"
    ) in page
    assert "tight knot at the truth" not in page


def test_the_model_caption_counts_the_archives_observations() -> None:
    meta = load_replay(ARCHIVES / "demo_model_comparison.npz").meta
    (embed,) = REPLAY_EMBEDS["story/how-many-masses.qmd"]
    assert f"as {meta['n_observations']} observations accumulate" in embed.caption
    assert f"Truth is K={meta['true_k']}" in embed.caption


def test_the_1d_page_says_how_many_dots_are_shown() -> None:
    """Only this catches the page claiming the clip shows the whole cloud."""
    meta = load_replay(ARCHIVES / "demo_1d.npz").meta
    page = " ".join(
        (SITE / "story/the-search-in-one-dimension.qmd").read_text("utf-8").split()
    )
    assert meta["n_particles"] == 1000
    assert meta["cloud_size"] == 500
    assert (
        "one thousand of them in the run animated above, which shows a sample of "
        "five hundred"
    ) in page


_RETIRED = ("demo_density.png",)
_ONE_MENTION = ["line 1: demo_density.png"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("![a](../assets/demo_density.png)", _ONE_MENTION, id="md_image"),
        pytest.param("[a](assets/demo_density.png)", _ONE_MENTION, id="md_link"),
        pytest.param(
            "[a]: ../site/assets/demo_density.png", _ONE_MENTION, id="reference"
        ),
        pytest.param(
            '<img src="../assets/demo_density.png" alt="">', _ONE_MENTION, id="html_img"
        ),
        pytest.param(
            "[a](https://github.com/example/clocks/blob/main/assets/demo_density.png)",
            _ONE_MENTION,
            id="github_url",
        ),
        pytest.param(
            "![a](../assets/demo_density.png?v=2)", _ONE_MENTION, id="with_query"
        ),
        pytest.param(
            "![a](../assets/%64emo_density.png)", _ONE_MENTION, id="encoded_basename"
        ),
        pytest.param(
            "![a](<../assets/demo_density.png>)", _ONE_MENTION, id="angle_brackets"
        ),
        pytest.param(
            '<img src="../assets/demo_density&#46;png" alt="">',
            _ONE_MENTION,
            id="html_entity",
        ),
        pytest.param(
            '<a href="../assets/demo_density&#46;png">a</a>',
            _ONE_MENTION,
            id="href_entity",
        ),
        pytest.param(
            "![retired][density]\n\n  [density]: ../assets/demo_density.png",
            ["line 3: demo_density.png"],
            id="indented_reference",
        ),
        pytest.param(
            '<picture><source srcset="../assets/demo_density.png">'
            '<img src="../assets/kept.gif" alt=""></picture>',
            _ONE_MENTION,
            id="srcset",
        ),
        pytest.param("`assets/demo_density.png`", _ONE_MENTION, id="inline_code"),
        pytest.param("see demo_density.png", _ONE_MENTION, id="bare_mention"),
        pytest.param(
            "![a](../assets/demo_density\\.png)", _ONE_MENTION, id="backslash_escape"
        ),
        pytest.param(
            '<img src="../assets/demo_density.&#9;png">', _ONE_MENTION, id="tab_entity"
        ),
        pytest.param(
            "![a](../assets/output/demo_density.png)",
            _ONE_MENTION,
            id="output_under_assets",
        ),
        pytest.param(
            "![a](../assets/demo&#37;5Fdensity.png)", _ONE_MENTION, id="entity_percent"
        ),
        pytest.param(
            "![a](../assets/demo_density%5C.png)", _ONE_MENTION, id="percent_backslash"
        ),
        pytest.param(
            "output/demo_density.png ![a](../assets/demo_density.png)",
            _ONE_MENTION,
            id="output_then_link",
        ),
        pytest.param(
            '<img src="../assets/demo_density.&#13;png">', _ONE_MENTION, id="cr_entity"
        ),
        pytest.param(
            "myoutput/demo_density.png", _ONE_MENTION, id="word_before_output"
        ),
        pytest.param(".output/demo_density.png", _ONE_MENTION, id="dot_before_output"),
        pytest.param(
            "x-output/demo_density.png", _ONE_MENTION, id="dash_before_output"
        ),
    ],
)
def test_retired_asset_mentions_are_found(text: str, expected: list[str]) -> None:
    assert _retired_asset_mentions(text, _RETIRED) == expected


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(
            "uv run demo-density             # → output/demo_density.png",
            id="command_output",
        ),
        pytest.param("[a](output/demo_density.png)", id="output_link"),
        pytest.param("![a](../assets/kept.gif)", id="other_asset"),
        pytest.param("The density figure is drawn at build time.", id="no_mention"),
    ],
)
def test_allowed_mentions(text: str) -> None:
    assert _retired_asset_mentions(text, _RETIRED) == []


_MENTION_FILES = [REPO / "README.md"] + sorted(
    path
    for path in SITE.rglob("*.qmd")
    if not _EXCLUDED_DIRS & set(path.relative_to(SITE).parts)
)


@pytest.mark.parametrize(
    "path",
    [
        pytest.param(path, id=path.relative_to(REPO).as_posix())
        for path in _MENTION_FILES
    ],
)
def test_no_page_mentions_a_retired_asset(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert _retired_asset_mentions(text, RETIRED_ASSETS) == []


@pytest.mark.parametrize(
    "path",
    [
        pytest.param(path, id=path.relative_to(REPO).as_posix())
        for path in _MENTION_FILES
        if path.suffix == ".qmd"
    ],
)
def test_no_page_has_unpinned_video_markup(path: Path) -> None:
    page = path.relative_to(SITE).as_posix()
    text = path.read_text(encoding="utf-8")
    assert _replay_embed_violations(text, page, REPLAY_EMBEDS.get(page, ())) == []


@pytest.mark.parametrize("module", LIBRARY_DRAWING_MODULES)
def test_library_drawing_code_has_no_unstyled_colour(module: str) -> None:
    text = (REPO / module).read_text(encoding="utf-8")
    assert _style_violations(text) == []


def test_every_figure_builder_module_is_scanned() -> None:
    defining = {
        path.relative_to(REPO).as_posix()
        for path in (REPO / "src" / "clocks").rglob("*.py")
        if any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.endswith("_figure")
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        )
    }
    assert defining <= set(LIBRARY_DRAWING_MODULES)


def _clip_modules() -> list[Path]:
    return sorted((REPO / "src" / "clocks").glob("_clip*.py"))


def test_every_clip_module_is_scanned() -> None:
    clip_modules = {path.relative_to(REPO).as_posix() for path in _clip_modules()}
    assert clip_modules
    assert clip_modules <= set(LIBRARY_DRAWING_MODULES)


# What a renderer may import from the package: never the forward model.
_RENDERER_IMPORTS = frozenset(
    {"clocks._replay", "clocks._style", "clocks._figures", "clocks._video"}
)


def _package_imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"{path.name}: relative import"
            found.add(node.module or "")
    return {name for name in found if name == "clocks" or name.startswith("clocks.")}


@pytest.mark.parametrize(
    "path",
    [
        *_clip_modules(),
        REPO / "src" / "clocks" / "_video.py",
        REPO / "src" / "clocks" / "_site_media.py",
    ],
    ids=lambda path: path.name,
)
def test_clip_modules_import_no_model_code(path: Path) -> None:
    imported = _package_imports(path)
    stray = {
        name
        for name in imported
        if name not in _RENDERER_IMPORTS and not name.startswith("clocks._clip")
    }
    assert stray == set()


def _draw_density_figure(colors):
    from clocks._figures import density_figure
    from clocks._replay import load_replay

    return density_figure(
        load_replay(REPO / "assets" / "replays" / "demo_density.npz"), colors
    )


def _draw_range_study_figure(colors):
    from clocks._echo_study import load_study, range_study_figure

    return range_study_figure(
        load_study(REPO / "assets" / "echolocation_range_study.json"), colors
    )


_FIGURE_DRAWERS = {
    "density_figure": _draw_density_figure,
    "range_study_figure": _draw_range_study_figure,
}


@pytest.mark.parametrize("name", sorted(FIGURE_FUNCTION_HEIGHTS))
def test_figure_function_heights_are_the_drawn_ones(name: str) -> None:
    import matplotlib.pyplot as plt

    from clocks._style import style

    with style("light") as colors:
        fig = _FIGURE_DRAWERS[name](colors)
    try:
        assert round(fig.get_figheight() * 100) == FIGURE_FUNCTION_HEIGHTS[name]
        assert round(fig.get_figwidth() * 100) == 750
    finally:
        plt.close(fig)


BUILDER_CELL_PROGRAMS: dict[str, str] = {
    "density_figure": """\
from pathlib import Path

from clocks.viz import THEMES, density_figure, load_replay, show, style

# A cell runs in its page's directory: the archive is two levels up.
replay = load_replay(Path("../../assets/replays/demo_density.npz"))
for theme in THEMES:
    with style(theme) as colors:
        fig = density_figure(replay, colors)
        show(fig)
""",
    "range_study_figure": """\
import json
from pathlib import Path

from clocks.viz import THEMES, range_study_figure, show, style

# A cell runs in its page's directory: the study is two levels up.
study = json.loads(Path("../../assets/echolocation_range_study.json").read_text())
for theme in THEMES:
    with style(theme) as colors:
        fig = range_study_figure(study, colors)
        show(fig)
""",
}


def _builder_cell(builder: str) -> tuple[Path, str]:
    """Return the page and the plot cell whose builders include *builder*."""
    found = [
        (SITE / page, cell)
        for page in PLOT_PAGES
        for cell in _plot_cells(SITE / page)
        if builder in _facade_builders(cell)
    ]
    assert len(found) == 1, f"{builder} is drawn by {len(found)} cells"
    return found[0]


_OPTION_LINES = (
    "#| code-fold: true",
    "#| renderings: [light, dark]",
    "#| fig-cap:",
)
_CAPTION_LINE = re.compile(r'^#\|   - ".*"$')


def _comment_lines(text: str) -> list[str]:
    """Return the stripped lines of ``text`` that start with ``#``."""
    return [line.strip() for line in text.splitlines() if line.strip().startswith("#")]


def _builder_cell_option_violations(cell: str, program: str) -> list[str]:
    """Return how a builder cell's header and comments depart from the pin.

    The first five lines must be, by position, ``#| code-fold: true``,
    ``#| renderings: [light, dark]``, ``#| fig-cap:`` and two identical caption
    entries. Every other comment line of the cell, whatever follows its ``#``,
    must be one of ``program``'s comment lines, in order, so no spelling of an
    option marker can be missed. A trailing comment on a code line is not a
    comment line here, and cannot be an option: Quarto reads options only from
    the whole lines at the very top of a cell (its pattern is ``^#\\s*\\| ?``).
    """
    lines = cell.splitlines()
    head = [*lines[:5], *[""] * (5 - len(lines[:5]))]
    messages = [
        f"line {number} is not {text!r}"
        for number, text in enumerate(_OPTION_LINES, start=1)
        if head[number - 1] != text
    ]
    captions = head[3:5]
    messages += [
        f"line {number} is not a caption entry"
        for number, line in enumerate(captions, start=4)
        if not _CAPTION_LINE.match(line)
    ]
    if all(_CAPTION_LINE.match(line) for line in captions) and (
        captions[0] != captions[1]
    ):
        messages.append("the two captions differ")
    actual = _comment_lines("\n".join(lines[5:]))
    expected = _comment_lines(program)
    messages += [
        f"unexpected comment line: {line!r}" for line in actual if line not in expected
    ]
    messages += [
        f"missing comment line: {line!r}" for line in expected if line not in actual
    ]
    if not messages and actual != expected:
        messages.append("the comment lines are out of order")
    return messages


_CANONICAL_OPTIONS = (
    "#| code-fold: true\n"
    "#| renderings: [light, dark]\n"
    "#| fig-cap:\n"
    '#|   - "A caption."\n'
    '#|   - "A caption."\n'
)
_SMALL_PROGRAM = "# A note.\nx = 1\n"
_CANONICAL_CELL = _CANONICAL_OPTIONS + _SMALL_PROGRAM


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        pytest.param(_CANONICAL_CELL, [], id="canonical"),
        pytest.param(
            _CANONICAL_CELL.replace("#| fig-cap:", "#| classes: [d-none]\n#| fig-cap:"),
            [
                "line 3 is not '#| fig-cap:'",
                "line 4 is not a caption entry",
                "unexpected comment line: '#|   - \"A caption.\"'",
            ],
            id="extra_option",
        ),
        pytest.param(
            _CANONICAL_CELL + "#| eval: false\n",
            ["unexpected comment line: '#| eval: false'"],
            id="eval_false",
        ),
        pytest.param(
            _CANONICAL_CELL.replace(
                "#| fig-cap:", "#| renderings: [light, dark]\n#| fig-cap:"
            ),
            [
                "line 3 is not '#| fig-cap:'",
                "line 4 is not a caption entry",
                "unexpected comment line: '#|   - \"A caption.\"'",
            ],
            id="duplicate_renderings",
        ),
        pytest.param(
            _CANONICAL_CELL.replace('#|   - "A caption."\n', '#|   - "Another."\n', 1),
            ["the two captions differ"],
            id="captions_differ",
        ),
        pytest.param(
            _CANONICAL_CELL.replace('#|   - "A caption."\n', "", 1),
            [
                "line 5 is not a caption entry",
                "missing comment line: '# A note.'",
            ],
            id="one_caption",
        ),
        pytest.param(
            _CANONICAL_OPTIONS + "import json\n#| eval: false\n" + _SMALL_PROGRAM,
            ["unexpected comment line: '#| eval: false'"],
            id="option_after_code",
        ),
        pytest.param(
            "#| renderings: [light, dark]\n#| code-fold: true\n"
            + _CANONICAL_CELL.split("\n", 2)[2],
            [
                "line 1 is not '#| code-fold: true'",
                "line 2 is not '#| renderings: [light, dark]'",
            ],
            id="reordered",
        ),
        pytest.param(
            _CANONICAL_OPTIONS + "# | classes: [d-none]\n" + _SMALL_PROGRAM,
            ["unexpected comment line: '# | classes: [d-none]'"],
            id="spaced_marker",
        ),
        pytest.param(
            _CANONICAL_CELL + "# | classes: [d-none]",
            ["unexpected comment line: '# | classes: [d-none]'"],
            id="spaced_marker_at_end",
        ),
        pytest.param(
            _CANONICAL_CELL.replace(
                "#| code-fold: true\n", '#|   - "x"\n#| code-fold: true\n'
            ),
            [
                "line 1 is not '#| code-fold: true'",
                "line 2 is not '#| renderings: [light, dark]'",
                "line 3 is not '#| fig-cap:'",
                "line 4 is not a caption entry",
                "unexpected comment line: '#|   - \"A caption.\"'",
            ],
            id="caption_before_header",
        ),
        pytest.param(
            _CANONICAL_CELL + "# note\n",
            ["unexpected comment line: '# note'"],
            id="extra_plain_comment",
        ),
        pytest.param(
            _CANONICAL_CELL.replace("# A note.\n", ""),
            ["missing comment line: '# A note.'"],
            id="program_comment_removed",
        ),
        pytest.param(
            _CANONICAL_CELL.replace("x = 1", "x = 1  # a trailing note"),
            [],
            id="trailing_comment_is_not_a_comment_line",
        ),
    ],
)
def test_builder_cell_options_are_exactly_the_three(
    cell: str, expected: list[str]
) -> None:
    assert _builder_cell_option_violations(cell, _SMALL_PROGRAM) == expected


@pytest.mark.parametrize(
    "builder",
    [
        pytest.param(name, id=name.removesuffix("_figure"))
        for name in BUILDER_CELL_PROGRAMS
    ],
)
def test_builder_cell_is_exactly_its_pinned_program(builder: str) -> None:
    """Each cell draws committed data that the Reproducibility page names.

    So the whole cell is pinned, options included: any change to a cell is a
    deliberate change to this test. The program is compared by syntax tree, so
    spacing inside a statement does not matter and every statement, name,
    literal and argument does; the option block and every comment line are
    compared as text, by ``_builder_cell_option_violations``.
    """
    page, cell = _builder_cell(builder)
    program = BUILDER_CELL_PROGRAMS[builder]
    assert _builder_cell_option_violations(cell, program) == []
    assert ast.dump(ast.parse(cell)) == ast.dump(ast.parse(program))

    literals = [
        node.value
        for node in ast.walk(ast.parse(program))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    assert len(literals) == 1
    resolved = (page.parent / literals[0]).resolve()
    assert resolved.is_file()
    assert resolved.is_relative_to((REPO / "assets").resolve())


def test_every_builder_cell_has_a_pinned_program() -> None:
    used: list[str] = []
    for page in PLOT_PAGES:
        for cell in _plot_cells(SITE / page):
            builders = _facade_builders(cell)
            if not builders:
                continue
            assert len(builders) == 1, f"{page}: a cell calls {builders}"
            assert builders[0] in BUILDER_CELL_PROGRAMS, f"{page}: {builders[0]}"
            used.append(builders[0])
    assert sorted(used) == sorted(BUILDER_CELL_PROGRAMS)
