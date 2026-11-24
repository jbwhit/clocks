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
"""

import ast
import html
import importlib.util
import re
import string
import urllib.parse
from collections.abc import Sequence
from pathlib import Path

import pytest

import clocks.viz

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
)
LIBRARY_DRAWING_MODULES: tuple[str, ...] = (
    "src/clocks/_figures.py",
    "src/clocks/_demos/demo_density.py",
    "src/clocks/_echo_study.py",
    "src/clocks/_clip_parts.py",
    "src/clocks/_clip_maps.py",
    "src/clocks/_clip_models.py",
    "src/clocks/_clip_tracks.py",
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


@pytest.mark.parametrize("page", PLOT_PAGES)
def test_manifest_matches_the_source(page: str) -> None:
    figures = check_site_figures.EXPECTED_FIGURES[_manifest_page(page)]
    assert all(fig.clip is None for fig in figures), page  # clips: pull request 3
    from_manifest = tuple((fig.kind, fig.height) for fig in figures)
    from_source = tuple(
        ("plot", _cell_height(cell)) for cell in _plot_cells(SITE / page)
    )
    assert from_manifest == from_source


def test_the_manifest_lists_exactly_the_plot_pages() -> None:
    assert set(check_site_figures.EXPECTED_FIGURES) == {
        _manifest_page(page) for page in PLOT_PAGES
    }
    assert len(PLOT_PAGES) == 9


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
            '<img src="../assets/demo_2d.gif" alt=""></picture>',
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
        pytest.param("![a](../assets/demo_2d.gif)", id="other_asset"),
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
    [*_clip_modules(), REPO / "src" / "clocks" / "_video.py"],
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
            _CANONICAL_CELL.replace("x = 1", "x = 1\n# | classes: [d-none]"),
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

    So the whole cell is pinned, options included, compared by syntax tree
    (comments and spacing do not matter; every statement, name, literal and
    argument does): any change to a cell is a deliberate change to this test.
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
