# Figure Style for Build-Time Plots: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the nine plots the site draws at build time one shared style, in a light and a dark rendering each, with colours that carry fixed meanings.

**Architecture:** A private module `src/clocks/_style.py` holds the palette, a context manager that applies a theme's rcParams, and a display helper that emits a figure at its authored size. Each page's plotting cell loops over the two themes and Quarto shows the rendering that matches the reader's theme. A source test and a rendered-site script hold the convention in place.

**Tech Stack:** Python 3.12+, matplotlib, pytest, Quarto 1.9.32+, uv, ruff.

**Status:** Approved for execution. Codex xhigh round 3, on `07711b3`: "SOUND ENOUGH TO IMPLEMENT", with two minor corrections that this revision folds in.

**Spec:** `docs/plans/2026-11-18-figure-style-design.md`. Read it before any task. Where this plan and the spec disagree, the spec wins and the disagreement is a plan bug to report.

## Global Constraints

- Python only through uv: `uv run pytest`, `uv run ruff format .`, `uv run ruff check .`. Never bare `python` or `pip`.
- Themes are exactly `("light", "dark")`, in that order, everywhere.
- Every plot is 7.5 inches wide at 100 dpi, emitted as a 1500-pixel-wide PNG declared 750 CSS pixels wide. Default height 4.2 inches (420 CSS pixels).
- Page cells never call `plt.show()`, never pass `cmap`, and never use a quoted matplotlib colour name or `tab:` colour.
- Data, scenarios, seeds, printed output and prose do not change. The only content changes are the four listed in the spec under "Declared changes to what a plot shows".
- The committed GIFs and PNGs under `assets/` and `site/assets/` are not touched. `tests/test_generated_assets.py` must stay green throughout.
- Quarto 1.9.32 or later is required to render the site.
- Before every commit run the full local gate: `uv run ruff format --check .`, `uv run ruff check .`, `uv run pytest`, and `uv run pytest -m slow` (CI runs the slow tests as a separate step; the default run skips them). Run the checks and the commit in separate commands, and read the output before committing.
- Every commit ends with a `Co-Authored-By:` trailer naming the model that wrote it. Commit messages are passed with a quoted heredoc (`<<'MSG'`).
- Commit after each task and push; do not open a pull request (the coordinator does).

## Review Focus

Conditions the spec implies but does not spell out a test for, most likely to bite first. Each has a test in the task that owns the code.

1. **A bundled font file is missing** (a broken build, a stray `.gitignore`): `style()` must raise `FileNotFoundError` naming the missing path, not fall back silently to DejaVu. Task 2.
2. **`style()` used inside `style()`**: leaving the inner block must restore the outer theme, not the pre-style defaults. Task 2.
3. **`show()` called where IPython is absent** (a plain script): it must raise `ImportError` saying that `show()` needs IPython, not an unexplained module error. Task 2.
4. **`show()` on a figure that is not the default size**: a 5 by 3 inch figure at 120 dpi must come out 1200 by 720 pixels declared 600 by 360. Task 2.
5. **A theme name with the wrong case** (`"Light"`): rejected with the same `ValueError` as any unknown theme. Task 1.

---

### Task 1: Palette

**Files:**
- Create: `src/clocks/_style.py`
- Test: `tests/test_style.py`

**Interfaces:**
- Produces: `THEMES: tuple[str, str]`; `Palette` (frozen dataclass with `str` fields `background, ink, muted, rule, grid, truth, belief, belief_alt, threshold` and `ordered: tuple[str, str, str]`); `palette(theme: str) -> Palette`.

- [ ] **Step 1: Write the failing tests** in `tests/test_style.py`. Put a WCAG helper at the top of the test file:

```python
def _luminance(hex_color: str) -> float:
    channels = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(a: str, b: str) -> float:
    lighter, darker = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)
```

Tests, each parametrized over `THEMES` where it says "each theme":

- `test_palette_colours_are_six_digit_hex`: for each theme, every `str` field and each `ordered` entry matches `^#[0-9a-f]{6}$`, and `len(ordered) == 3`.
- `test_themes_differ_in_every_role`: for every field name, `getattr(palette("light"), name) != getattr(palette("dark"), name)`.
- `test_semantic_hues_are_distinct`: for each theme, `len({p.truth, p.belief, p.belief_alt}) == 3`.
- `test_data_colours_clear_three_to_one`: for each theme, `_contrast(colour, p.background) >= 3.0` for `truth`, `belief`, `belief_alt`, `muted`, `threshold` and each `ordered` entry.
- `test_ink_clears_seven_to_one`: for each theme, `_contrast(p.ink, p.background) >= 7.0`.
- `test_ordered_runs_from_least_to_most_contrast`: for each theme, the three contrasts are strictly increasing.
- `test_palette_rejects_unknown_theme`: parametrized over `"sepia"` and `"Light"`; `pytest.raises(ValueError)` and the message contains the bad value, `"light"` and `"dark"`.

- [ ] **Step 2: Run them and confirm they fail** with `ModuleNotFoundError: clocks._style`.

Run: `uv run pytest tests/test_style.py -q`

- [ ] **Step 3: Implement** `THEMES`, `Palette` and `palette()` in `src/clocks/_style.py`, with the hex values from the spec's "Colour roles" table copied exactly. The unknown-theme message is `f"unknown theme {theme!r}; expected one of {THEMES}"`.

- [ ] **Step 4: Run the tests and confirm they pass.** Then run the full gate.

- [ ] **Step 5: Commit and push** with message `Add the figure palette with fixed colour roles`.

---

### Task 2: Style context, bundled font and display helper

**Files:**
- Create: `src/clocks/_fonts/SourceSans3-Regular.ttf`, `src/clocks/_fonts/SourceSans3-Semibold.ttf`, `src/clocks/_fonts/LICENSE.md`
- Modify: `src/clocks/_style.py`, `src/clocks/viz.py`, `pyproject.toml`, `scripts/check_wheel_entrypoints.py`
- Test: `tests/test_style.py`

**Interfaces:**
- Consumes: `THEMES`, `Palette`, `palette()` from Task 1.
- Produces: `style(theme: str) -> ContextManager[Palette]`; `show(fig: matplotlib.figure.Figure) -> None`. `clocks.viz` re-exports `THEMES`, `Palette`, `palette`, `style`, `show` and lists them in `__all__`.

- [ ] **Step 1: Add the font files.** Download from the `3.052R` tag of `adobe-fonts/source-sans`:

```bash
base=https://raw.githubusercontent.com/adobe-fonts/source-sans/3.052R
mkdir -p src/clocks/_fonts
curl -sL -o src/clocks/_fonts/SourceSans3-Regular.ttf  $base/TTF/SourceSans3-Regular.ttf
curl -sL -o src/clocks/_fonts/SourceSans3-Semibold.ttf $base/TTF/SourceSans3-Semibold.ttf
curl -sL -o src/clocks/_fonts/LICENSE.md               $base/LICENSE.md
shasum -a 256 src/clocks/_fonts/*
```

The checksums must be exactly:

```
89ad2c4f66dd29127527493e729c31e731f111cf10faf5774c3db9275ed0c22c  LICENSE.md
4644c81b86ec9caaa76b634889968ed3c4f4f52f054855933acc7c2b21e53b0f  SourceSans3-Regular.ttf
a3f4f8dcf343a8f24dc61951de93f3ba1558b15cd250ba24af8a40e957081b7d  SourceSans3-Semibold.ttf
```

If any differs, stop and report; do not commit the files.

- [ ] **Step 2: Write the failing tests**, added to `tests/test_style.py`:

Every test in this step that changes rcParams runs inside its own `with matplotlib.rc_context():` block, so a test that fails cannot leak settings into the next. The colour cycle is read with `plt.rcParams["axes.prop_cycle"].by_key()["color"]`.

- `test_style_applies_theme_and_restores` (each theme): set `plt.rcParams` to sentinel values first (`figure.facecolor="#123456"`, `axes.edgecolor="#654321"`, `text.color="#abcdef"`, `font.family=["monospace"]`, and a colour cycle of `["#010203"]`). Inside `with style(theme) as colors:` assert `colors == palette(theme)`, `rcParams["figure.facecolor"] == colors.background`, `rcParams["axes.edgecolor"] == colors.rule`, `rcParams["text.color"] == colors.ink`, `rcParams["font.family"] == ["Source Sans 3"]`, `rcParams["figure.figsize"] == [7.5, 4.2]`, `rcParams["figure.autolayout"] is False`, and the colour cycle is exactly `[colors.ink]`. After the block all five sentinels are back.
- `test_style_restores_when_the_block_raises` (each theme): same sentinels, raise `RuntimeError` inside the block, catch it outside, assert the sentinels are back.
- `test_nested_style_restores_the_outer_theme`: inside `style("light")` enter `style("dark")`; after the inner block `rcParams["figure.facecolor"] == palette("light").background`.
- `test_style_rejects_unknown_theme`: parametrized over `"sepia"` and `"Light"`; `ValueError` whose message contains the bad value, `"light"` and `"dark"`; and rcParams are unchanged afterwards.
- `test_bundled_fonts_are_registered_once`: after `with style("light"): pass`, collect `font_manager.fontManager.ttflist` entries whose `fname` lies under `Path(clocks._style.__file__).parent / "_fonts"`. Assert there are exactly two, both `name == "Source Sans 3"`, with weights `{400, 600}`. Enter `style("dark")` and assert the count is still two. Assert `font_manager.findfont(FontProperties(family="Source Sans 3", weight=w), fallback_to_default=False)` succeeds for `w` in `(400, 600)`.
- `test_missing_font_file_is_reported`: monkeypatch the module's font directory constant to an empty `tmp_path` and reset its registered flag; entering `style("light")` raises `FileNotFoundError` whose message contains `SourceSans3-Regular.ttf`.
- `test_a_partly_missing_font_set_registers_nothing`: point the font directory at a `tmp_path` holding a copy of the Regular file only, reset the flag, and record `len(font_manager.fontManager.ttflist)`. Entering `style("light")` raises `FileNotFoundError` naming `SourceSans3-Semibold.ttf`, and the list length is unchanged.
- `test_show_emits_the_authored_canvas`: replace `IPython.display.display` with a recorder (monkeypatch). With `plt.rcParams["savefig.bbox"] = "tight"` and `plt.rcParams["savefig.pad_inches"] = 0.5` set, inside `style("light")` create `fig, ax = plt.subplots()`, plot one line, call `show(fig)`. Assert one recorded call; its argument is an `IPython.display.Image` with `retina` true whose PNG bytes decode (use `matplotlib.image.imread(io.BytesIO(...))`) to shape `(840, 1500, 4)` or `(840, 1500, 3)`, with no pixel whose alpha is below 1 when an alpha channel exists; `image.width == 750` and `image.height == 420`; and `plt.fignum_exists(fig.number)` is false.
- `test_show_scales_with_figure_size_and_dpi`: with the same two `savefig` settings, a `plt.figure(figsize=(5, 3), dpi=120)` figure holding one axes with a plotted line gives a 1200 by 720 PNG declared 600 by 360. (The padding makes a tight crop change the size even for a sparse figure, so the test cannot pass if the crop is applied.)
- `test_show_explains_a_missing_ipython`: monkeypatch `sys.modules["IPython.display"]` to `None` (which makes the import raise); `show(fig)` raises `ImportError` matching `clocks\.viz\.show needs IPython`. (The raw import error already contains the word "IPython", so matching less would pass without the translation.)

- [ ] **Step 3: Run them and confirm they fail** on the missing names.

Run: `uv run pytest tests/test_style.py -q`

- [ ] **Step 4: Implement in `src/clocks/_style.py`:**

- `_FONT_DIR: Path` (the `_fonts` directory beside the module) and `_FONT_FILES = ("SourceSans3-Regular.ttf", "SourceSans3-Semibold.ttf")`.
- `_register_fonts() -> None`: on first call, check that every file exists, raising `FileNotFoundError(f"bundled font missing: {path}")` for the first that does not, before adding any; then add each with `font_manager.fontManager.addfont`; do nothing on later calls. Keep the "already registered" state in a module-level flag the test can reset.
- `style(theme)`: validate the theme through `palette(theme)` before touching anything, call `_register_fonts()`, then `with matplotlib.rc_context(rc): yield colors`, where `rc` is the spec's `BASE` block plus the per-theme keys listed under it. The colour cycle is `cycler(color=[colors.ink])`.
- `show(fig)`: import `Image` and `display` from `IPython.display` inside the function, converting `ImportError` to `ImportError("clocks.viz.show needs IPython; it is meant for Jupyter and Quarto cells")`. Save to a `BytesIO` inside `rc_context({"savefig.bbox": None})` with `format="png"`, `dpi=2 * fig.dpi`, `bbox_inches=None`, `transparent=False`, `facecolor=fig.get_facecolor()`. Close the figure with `plt.close(fig)`. Display `Image(data=..., retina=True)`.

Nothing in the module calls `matplotlib.use`.

- [ ] **Step 5: Re-export** the five names from `src/clocks/viz.py` and add them to `__all__`.

- [ ] **Step 6: Packaging.** In `pyproject.toml` set `license = "MIT AND OFL-1.1-RFN"` and `license-files = ["LICENSE", "src/clocks/_fonts/LICENSE.md"]`. Immediately after that edit, before any other `uv` command, run `uv lock --check`: it must report the lockfile current. (A later `uv run` would quietly refresh a stale lock and hide the answer.) A licence change alters no dependency input, so if it reports stale, find out why before refreshing anything, and report it. Extend `scripts/check_wheel_entrypoints.py` so that, for the wheel it is given, it also fails unless the archive contains `clocks/_fonts/SourceSans3-Regular.ttf`, `clocks/_fonts/SourceSans3-Semibold.ttf` and `clocks/_fonts/LICENSE.md`, and unless running `python -c "from clocks.viz import style; cm = style('light'); cm.__enter__(); cm.__exit__(None, None, None)"` in the installed environment exits 0. It also fails unless the wheel's `.dist-info/licenses/` holds both `LICENSE` and `src/clocks/_fonts/LICENSE.md`, and unless `METADATA` carries `License-Expression: MIT AND OFL-1.1-RFN` and a `License-File:` line for each of the two. Read the script first and follow its existing structure and error style: it already builds a temporary environment and has a `_run` helper and a `scripts_directory / python_name` path, which the new command uses. CI already calls the script and needs no change.

- [ ] **Step 7: Verify.**

Run: `uv run pytest tests/test_style.py -q` — all pass.
Run: `uv build && uv run python scripts/check_wheel_entrypoints.py dist/*.whl python3` — exits 0. Then delete nothing from `dist/`; it is ignored by git.
Run the full gate.

- [ ] **Step 8: Commit and push** with message `Add the style context, bundled Source Sans 3 and a fixed-size figure display`. The commit body records the release tag `3.052R` and the three SHA-256 values.

---

### Task 3: Restyle the nine plotting cells

**Files:**
- Create: `tests/test_site_figure_style.py`
- Modify: the nine `.qmd` files in the spec's scope table

**Interfaces:**
- Consumes: `from clocks.viz import THEMES, style, show`; the `Palette` fields.

- [ ] **Step 1: Write the source contract test** `tests/test_site_figure_style.py`.

```python
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
```

Helpers: `_python_cells(path) -> list[str]` returns the text of each ` ```{python} ` fenced cell; `_plot_cells(path)` returns those containing `plt.subplots(` or `plt.figure(`.

Tests:

- `test_plot_pages_are_exactly_the_known_nine`: the sorted relative paths of every `.qmd` under `site/` (excluding `_output` and `.quarto`) that has a plot cell equal `sorted(PLOT_PAGES)`.
- Parametrized over `PLOT_PAGES`, with exactly one plot cell asserted per page:
  - `test_cell_declares_both_renderings`: the cell contains the line `#| renderings: [light, dark]`.
  - `test_cell_gives_one_identical_caption_per_rendering`: the cell has `#| fig-cap:` followed by exactly two `#|   - "..."` lines whose quoted strings are equal and non-empty.
  - `test_cell_draws_through_the_style_and_show`: the cell contains `for theme in THEMES:`, `with style(theme) as`, and `show(fig)`.
  - `test_cell_has_no_unstyled_colour_or_display`: parses the plotting cell and asserts `_style_violations(cell)` is empty. The helper walks the cell's syntax tree and reports calls to `plt.show`, a `cmap` keyword, a string-literal colour (alone or in a list or tuple) in any colour keyword, a string `fmt` keyword, and positional string arguments to plot-family calls beyond the two column names allowed when `data=` is passed.

- [ ] **Step 2: Run it.** `test_plot_pages_are_exactly_the_known_nine` passes already. The rendering, caption and style-and-show tests fail for all nine pages. The colour-and-display test, in the regular-expression form this step first used, failed for seven; `clocks-as-gravimeters` and `how-many-masses` used no named colours and passed it from the start. (The syntax-tree form that replaced it fails six of the original cells: `the-search-in-one-dimension` passes it, because that cell passed its `tab:` colours through a variable, which a keyword check does not see.)

- [ ] **Step 3: Convert each cell**, following the spec's "Two renderings per plot" pattern, its "Per-plot colour assignments" table, and its four "Declared changes". For every cell:

- Add `THEMES, style, show` to the cell's imports via `from clocks.viz import THEMES, show, style`.
- Replace `#| fig-cap: "..."` with the two-entry list form.
- Keep all simulation, inference and `print` calls above the theme loop, unchanged. Several cells also compute what they plot while drawing it (the `clock_rates` calls on the gravimeter, one-clock and two-mass pages, the point-mass and Gaussian rates on `beyond-point-masses`, the per-observation series on `how-many-masses`). Move that computation above the loop into plain arrays, so the loop only draws.
- Draw inside `for theme in THEMES:` / `with style(theme) as colors:`, give every data artist an explicit colour from `colors`, and end the block with `show(fig)`. Remove `plt.close(fig)` and the trailing bare `fig`.
- Heights other than the default: `the-search-in-one-dimension` and `the-particle-filter` (two stacked panels) use `figsize=(7.5, 5.4)`; `into-the-plane` (equal aspect) uses `figsize=(7.5, 5.4)`. All others use the default.
- A label at the end of a curve is placed with `ax.annotate` at the curve's last data point, offset 6 points right, vertically centred, in the series colour; widen the x-limit enough that the label is not clipped.
- A label on a horizontal line (a truth value, the ESS target, the noise level) sits just above the line at its right end: anchored at the right-hand data limit on the line, offset 3 points up and 2 points left, right-aligned and bottom-aligned, in the line's colour. It is not centred on the line, which would run through the text.

Per-page details:

| Page | Detail |
|---|---|
| `clocks-as-gravimeters` | Read the lower y-limit after drawing the three curves and before adding the reference line. The upper limit is `1 + 0.05 * (max_rate - min_rate)` across all three curves. The line at 1 is solid, in `colors.rule`. |
| `one-clock-is-not-enough` | The page's first cell only prints and stays as it is. In the plotting cell the two series keep their joining lines, with circle and square markers, and the legend shows those markers. |
| `the-search-in-one-dimension` | Each panel's truth label follows the horizontal-line rule above: anchored at the axes' right x-limit on the truth value, in data coordinates. `axhline` spans axes fractions, so its own x-data cannot be used as the anchor. |
| `into-the-plane` | The weights are no longer passed to the scatter. The ring and the crosshair are one legend entry. Remove the existing `fontsize=8` on the legend so the style's size applies. |
| `two-hidden-masses` | The legend entries show the circle and square markers. Today the labels belong to the dense, markerless curves. |
| `how-many-masses` | The legend is replaced by three end labels; the K = 2 label reads `K = 2 (true model)`. |
| `beyond-point-masses` | Keeps its legend. Widths 0.5 and 2.0 take `ordered[0]` and `ordered[2]`. |
| `gravitational-echolocation` | Both lines are labelled directly. Read the log y-limits after drawing the signal and the noise line, shade from the lower limit to the noise level in `colors.threshold` at alpha 0.12, then set the limits back. |
| `the-particle-filter` | The ESS axis runs from 0 to `pf.n_particles`. The target is labelled on its line. The stage plot keeps `where="mid"` and gets integer ticks. |

Caption wording that must change (both list entries, identically):

- `into-the-plane`: "The surviving hypotheses after 25 observations. Dots are particles; the ring and crosshair mark the true mass position."
- `two-hidden-masses`: replace "at the five clock positions (dots)" with "at the five marked clock positions" in the existing caption; nothing else.
- `the-search-in-one-dimension`: replace "Dashed lines mark the truth" with "Solid lines mark the truth"; nothing else.
- `gravitational-echolocation`: "The differential signal an exterior mass leaves on the head, vs range. The dashed line is the noise of a single observation, and the shaded region lies below it; the sense must work in the shrinking gap above."
- `how-many-masses`, `clocks-as-gravimeters`, `one-clock-is-not-enough`, `beyond-point-masses`, `the-particle-filter`: wording unchanged.

- [ ] **Step 4: Run the tests** and confirm all pass. `uv run pytest tests/test_site_figure_style.py tests/test_generated_assets.py tests/test_site_prose_contract.py -q`

- [ ] **Step 5: Render and look.**

Run: `cd site && uv run --frozen quarto render && cd ..` — exits 0.
Run: `uv run python scripts/check_site_links.py` — both contracts OK.

For each of the nine pages, open the two PNGs Quarto wrote under `site/_output/<section>/<page>_files/figure-html/` and check against the cell source: the right series has the right colour, labels are not clipped or overlapping, the dark rendering has a dark face. Fix what is wrong before going on.

- [ ] **Step 6: Run the full gate, then commit and push** with message `Draw the nine build-time plots in the shared style, light and dark`.

---

### Task 4: Rendered-site check, deploy gate and documentation

**Files:**
- Create: `scripts/check_site_figures.py`, `tests/test_check_site_figures.py`
- Modify: `.github/workflows/site.yml`, `site/reproduce/getting-started.qmd`

**Interfaces:**
- Produces: `scripts/check_site_figures.py` with `figure_failures(output_dir: Path) -> list[str]` and a `main()` that prints the failures and exits 1 if there are any, otherwise prints `Figure contract OK: 9 pages checked.` and exits 0.

- [ ] **Step 1: Write the failing tests** `tests/test_check_site_figures.py`, building small HTML files under `tmp_path` in the shape Quarto emits (a `div` with class `cell-output cell-output-display light-content` containing a `figure` with an `img` carrying `width`, `height` and `src`, and a `figcaption`; the same with `dark-content`). Import the script the way `tests/test_site_prose_contract.py` imports `check_site_links`.

Each test monkeypatches the script's expected-pages constant down to the one page it builds, so the other eight being absent from `tmp_path` is not what it measures. Each defect fixture differs from the well-formed page in exactly one respect; in particular the image-count fixtures keep both captions. The image files the fixture references are created as small distinct files.

- `test_a_well_formed_page_passes`: one light and one dark image, 750 by 420, different existing `src` files, equal captions (include inline `<code>` inside the caption, as the two-mass caption has): `figure_failures` returns an empty list.
- One test per defect, each asserting exactly one failure whose message names the page and states the reason: the dark image missing; two light images; a width of 597; a height other than the page's expected height; the same file referenced by both wrappers, written as `plot.png` in one and `./plot.png` in the other, so that comparing the raw strings would miss it; a `src` that does not exist on disk; no `figcaption` in the dark wrapper; two `figcaption`s in one wrapper; captions that differ only in the text inside a nested `<code>` element, so that a parser dropping nested text would miss it.
- `test_a_missing_page_is_a_failure`: with the constant left at one expected page and no file written, the page is reported as missing.

- [ ] **Step 2: Run them and confirm they fail.**

- [ ] **Step 3: Implement** with the standard library's `html.parser`, as `scripts/check_site_links.py` does. The expected pages and heights are a module constant: the nine pages, height 420 except `story/the-search-in-one-dimension.html`, `story/into-the-plane.html` and `method/the-particle-filter.html` at 540. Count only images inside a `light-content` or `dark-content` wrapper, so the committed GIFs and PNGs on the same pages are ignored. A caption's text is everything inside its `figcaption`, including text within nested tags, with whitespace collapsed. Each `src` is resolved relative to the page and must exist; the two must be different files.

- [ ] **Step 4: Run the tests** and confirm they pass. Then render the site in this worktree (`site/_output` is ignored by git, so no earlier render is present) and run both checks from the repository root:

Run: `cd site && uv run --frozen quarto render && cd ..` — exits 0.
Run: `uv run python scripts/check_site_links.py` — both contracts OK.
Run: `uv run python scripts/check_site_figures.py` — prints `Figure contract OK: 9 pages checked.`
Run: `quarto --version` and note the version in the commit message.

- [ ] **Step 5: Gate the deployment.** In `.github/workflows/site.yml`, after the "Render site" step and before the upload, add a step named "Check rendered site" that runs `uv run --frozen python scripts/check_site_links.py` and then `uv run --frozen python scripts/check_site_figures.py` from the repository root.

- [ ] **Step 6: Document the requirement.** In `site/reproduce/getting-started.qmd`, add a third bullet under "Prerequisites": Quarto 1.9.32 or later, needed only to render the site, with a link to `https://quarto.org/docs/get-started/`.

- [ ] **Step 7: Phone and contact-sheet checks.** These produce evidence for the pull request, not test code.

- Put every output under the scratch directory the coordinator names in the dispatch (if none is named, a new directory under the system temporary directory). The task's scratch outputs are not committed; afterwards the coordinator commits one full-width sheet per theme under `docs/plans/2026-11-18-figure-style/` (see below).
- Load each of the nine pages in a 390-pixel-wide iframe served from `site/_output` (headless Chrome will not lay out a window narrower than about 500 pixels, so a `--window-size=390` screenshot crops instead of reflowing). Do this once in each theme: the page's theme switch calls `window.quartoToggleColorScheme()`, and the dark theme is active when `document.body` has the class `quarto-dark`. In each theme, for each page, record `document.documentElement.scrollWidth`, and for the plot rendering that is visible in that theme its rendered width and its container's width. The scroll width must be 390, and the visible plot's width must be greater than zero and no wider than its container; a hidden rendering reports zero width and must not be the one measured.
- Build four contact sheets with ImageMagick (`magick montage`): the nine plots at 750 pixels wide (their displayed size) in the light theme, the same in the dark theme, and each of those again with every plot resized to 390 pixels wide.
- Return to the coordinator the four sheet paths and the per-page phone measurements. The coordinator inspects the sheets against each cell's source and, because the command-line client cannot attach images to the pull request, commits the two full-width (750-pixel) sheets, one per theme, under `docs/plans/2026-11-18-figure-style/` as the record that accompanies it.

- [ ] **Step 8: Run the full gate, re-render, re-run both site checks, then commit and push** with message `Check the rendered figures and gate the deployment on the site checks`.

---

### Task 5: Mutation checks

No files change. This task proves the new tests can fail. Work in the task's own worktree; set `PYTHONDONTWRITEBYTECODE=1`; delete every `__pycache__` before each mutant; confirm `git status --porcelain` is empty before and after each; apply each mutant and restore it in a `try`/`finally`; run the whole suite including the slow tests (`uv run pytest -q -m ''`) for each mutant, in the foreground with a 600000 ms timeout.

| Mutation | Tests that must fail, and no others |
|---|---|
| `palette("dark")` returns the light palette | `test_themes_differ_in_every_role` |
| Light `ordered[0]` changed to `#a9cbe6` | `test_data_colours_clear_three_to_one[light]` |
| `style()` yields without `rc_context` (sets rcParams directly, never restores) | `test_style_applies_theme_and_restores[light]`, `[dark]`, `test_style_restores_when_the_block_raises[light]`, `[dark]`, `test_nested_style_restores_the_outer_theme` |
| Unknown-theme message names only `'light'`, not both themes | `test_palette_rejects_unknown_theme[sepia]`, `[Light]`, `test_style_rejects_unknown_theme[sepia]`, `[Light]` |
| `show()` passes `bbox_inches="tight"` | `test_show_emits_the_authored_canvas`, `test_show_scales_with_figure_size_and_dpi` |
| `_register_fonts()` registers on every call | `test_bundled_fonts_are_registered_once` |
| Remove `#| renderings: [light, dark]` from `story/one-clock-is-not-enough.qmd` | `test_cell_declares_both_renderings[story/one-clock-is-not-enough.qmd]` |
| Change one of the two captions in `story/how-many-masses.qmd` | `test_cell_gives_one_identical_caption_per_rendering[story/how-many-masses.qmd]` |
| Use `color="tab:blue"` for one line in `story/clocks-as-gravimeters.qmd` | `test_cell_has_no_unstyled_colour_or_display[story/clocks-as-gravimeters.qmd]` |
| Add `cmap = "viridis"` to the scatter in `story/into-the-plane.qmd` | `test_cell_has_no_unstyled_colour_or_display[story/into-the-plane.qmd]` |

- [ ] **Step 1: Run each mutation** and record, for each, the exact set of failing test IDs.
- [ ] **Step 2: Report** the table of results. Where the failing set differs from the expected set, say which tests were extra or missing; do not change tests or code to make the table match. A difference is a finding for the coordinator.
