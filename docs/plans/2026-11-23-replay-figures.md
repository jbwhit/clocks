# The Two Static Figures (Pull Request 2): Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Draw the density figure and the echolocation range-study figure from committed data when the site is built, in both themes; retire their two committed PNGs; switch `demo-density` to the recorded run; and teach the rendered-site check and the source guard that a page can hold several plots.

**Architecture:** `density_figure(replay, colors)` lives in a new private module `src/clocks/_figures.py`, beside two small trace helpers that pull request 3's clips will reuse. `range_study_figure(study, colors)` lives in `src/clocks/_echo_study.py` next to `summarize`, and `write_summary_figure` becomes a thin wrapper over it. Each returns a Matplotlib figure drawn with the shared palette; an executed page cell calls it once per theme, and the demo command and the scan script call it once for a PNG. `scripts/check_site_figures.py` gains an ordered per-page manifest of figures, and `tests/test_site_figure_style.py` checks the page source against the same manifest. Small follow-ups owed from pull request 1's review ride along in two tasks of their own.

**Tech Stack:** Python 3.12+, NumPy, Matplotlib 3.10, pytest, uv, ruff, Quarto (1.11.9 on the runner), GitHub Actions.

**Status:** Draft, not yet reviewed.

**Spec:** `docs/plans/2026-11-19-replay-and-animation-design.md`: section 6, the parts of section 9 named below, section 10, item 2 of section 11, the "Figures" and "Site" parts of "Testing", the mutation rows for figures, the rendered-site check and the source guard, and "Checks by eye". Read them before any task. Sections 1 to 3 describe the archive the density figure reads. Where this plan and the spec disagree, the spec wins and the disagreement is a plan bug to report, except for the departures listed under "Decisions this plan makes", each of which says why. The plan for pull request 1, `docs/plans/2026-11-19-replay-data.md`, shows the conventions this one follows.

**Terms.** A *theme wrapper* is an element of the rendered HTML whose class list holds `light-content` or `dark-content`; Quarto emits one per rendering of a cell, and the site's stylesheet shows only the current theme's. A *plot cell* is an executed Python cell on a page that draws a figure. The *manifest* is the per-page list of figures, in document order, that the rendered-site check and the source guard both read. A *trace* is a panel that plots one parameter's posterior mean against the number of observations seen. *Retired* files are committed media this track removes.

## Global Constraints

- Python only through uv: `uv run pytest`, `uv run ruff format .`, `uv run ruff check .`, `uv run python ...`. Never bare `python` or `pip`.
- **Every task runs the same cycle**, and its steps say only what is particular to it: write the tests of Step 1; run them and see them fail for the stated reason; implement; run them and see them pass; run the full local gate (`uv run ruff format --check .`, `uv run ruff check .`, `uv run pytest -q`, `uv run pytest -q -m slow`) and read its output; then, in a separate command, commit with the task's message and `git push origin claude-replay-figures`. Do not open a pull request (the coordinator does). Begin every git step with an explicit `cd` to the worktree.
- **Every task that changes a page also renders the site and runs both checks**, after the gate and before the commit: `cd site && uv run --frozen quarto render && cd ..`, then `uv run --frozen python scripts/check_site_links.py` and `uv run --frozen python scripts/check_site_figures.py`. The runner uses Quarto 1.11.9 (`.github/workflows/site-check.yml`); state the local `quarto --version` in the commit body.
- Commit messages are passed with a quoted heredoc (`<<'MSG'`) and end with a `Co-Authored-By:` trailer naming the model that wrote the commit.
- **The repository is public.** No committed file may contain an absolute local path, a person's name or an email address. This includes the contact-sheet script and anything printed into a commit body.
- The archives are not re-recorded. At the end, `git diff --stat main -- assets/replays` is empty. Under `assets/` and `site/assets/` the only changes are the four deletions of Tasks 4 and 6.
- No library drawing code contains a literal colour: every colour is a field of the `Palette` passed in. Role colours: density, the true profile `truth`, the median profile and its band `belief`, clocks `ink`; traces, the mean and its band `belief`, the truth `truth`; range study, each run and the median error `ink`, the median posterior standard deviation `belief`, the pass limits `threshold`, always dashed.
- Both figures are 750 by 540 CSS pixels: `figsize=(7.5, 5.4)` at the style's 100 dpi, which `show` renders at twice that. Files written by the command and the scan script are saved at `dpi=200` (1500 by 1080 pixels), the same pixels the site shows.
- A figure function draws with the rcParams in force, so callers call it inside `with style(theme) as colors:` and pass that `colors`. It never calls `plt.show`, never saves, and never closes the figure it returns.
- **Renderers never call the clock-rate forward model** (spec, section 1). `_figures.py` evaluates the density parameterisation A·exp(−(x−μ)²/2σ²) itself.
- Out of scope: clips, video, the encoder, `_video.py`, the pre-render script, every demo command other than `demo-density`, the 2D truth change, the version bump, removing `animate_*`, `plot_*` or `create_*_dashboard` names, the README GIF and its paragraph, the Architecture page's module list, and the "run exactly once" protocol sentences. Pull request 3 does all of these.
- `demo-density` keeps `--output PATH` (default `output/demo_density.png`), `--observations` and `--particles`. After Task 4 it writes exactly one file, the PNG at `--output`: the density figure of the run `record(n_observations=..., n_particles=...)` returns, light theme, 1500 by 1080 pixels. It writes no archive; `--output-dir`, `--theme` and the `.npz` output are pull request 3's.
- The palette is not edited. Neither figure draws with `ordered`, so the two collisions noted in the figure-style review (`ordered[1]` is the hex of `belief`; light `ink` against `ordered[2]` on one axes) cannot arise in them.

## Review Focus

Conditions the spec implies but does not spell out a test for, most likely to bite first. Each has a test in the task that owns the code.

1. **A one-observation, sixteen-particle run.** `tests/test_cli.py` already runs `demo-density --observations 1 --particles 16`. The density figure must draw it: traces from 0 to 1, limits from a single row, percentiles over sixteen dots. Task 3.
2. **A trace whose rows after the prior have no spread and sit on the truth**, as in a one-particle run. The smallest interval holding them has zero width, and Matplotlib warns on equal limits. `trace_limits` pads by 4% of the prior range instead. Task 3.
3. **`--output` naming a directory that does not exist yet.** Today's command creates it; the new one must too. Task 4.
4. **The wrong archive passed to `density_figure`.** Every replay has a `cloud`, so another demo's archive would draw a meaningless profile without complaint. It raises `ValueError` naming the demo. Task 3.
5. **A study with an error of exactly zero.** The scan script's development studies, and `tests/test_echo_study.py`'s fake records (`mass_error=0.0`), can hold one. A log axis cannot show it; `write_summary_figure` must still write its file without raising. Task 5 keeps the existing fake-record test passing through the new wrapper.

## Decisions this plan makes

Each is something the spec leaves open or states differently; the reason follows.

- **Where the figures live.** `density_figure` and the trace helpers go in a new `src/clocks/_figures.py`. `range_study_figure` stays in `_echo_study.py`: `_figures.py` importing `summarize` while `write_summary_figure` imports `range_study_figure` would be an import cycle.
- **`clocks.viz` exports `density_figure`, `range_study_figure` and `load_replay` now.** Section 7 lists them as additions in pull request 3's upgrade note, but section 10 defines a plot cell partly by its call to "a `clocks.viz` function named `*_figure`", so the page cells need them in this pull request. Adding names breaks nothing; the note in pull request 3 stays true.
- **The density traces follow section 5's trace conventions**: mean, a one-standard-deviation band, the truth as a solid line, x from 0 to T, a header with the name, the truth and the final estimate, and y-limits from rows k ≥ 1 and the truth. Section 6 says only "traces for μ, σ and A"; one convention for every trace on the site is the simpler reading. "Widened 8%" is read as 4% of the interval's width added on each side.
- **The density profile is evaluated on 321 points across μ's prior range**, −8 to 8, a step of 0.05 that contains the five clocks and the true centre exactly. Section 6 says "along the track" without an extent.
- **Pass counts are computed from the limits**: a run counts when its position error and its mass error are both within the study's tolerances, which is what the label "runs within both limits" says. The records' own `passed` flags agree on every committed record (measured: 12, 12, 11, 8, 3, 0 either way).
- **The manifest's shape** is section 10's, `Fig(kind, height, clip)` as a `NamedTuple`, keyed by output page; in this pull request every entry is `Fig("plot", height)`. Pull request 3 inserts `Fig("clip", height, name)` entries and adds the clip branch; nothing written here is rewritten.
- **Library modules under the colour check**: every module that defines a function whose name ends in `_figure`, plus `src/clocks/_demos/demo_density.py`. A test finds those modules by syntax tree, so a new figure module cannot be left out silently. The old panel and animation modules are pull request 3's to delete and are not scanned.
- **Provenance prose in this pull request**: the section retitle and the calibration-report link label (both describe build-time figures, which exist from this pull request on), the density and range-study sentences, and the CI paragraph's list of committed exceptions. The GIF and clip prose, the echolocation-animation sentences and "run exactly once" are pull request 3's.
- **The calibration report gains one dated sentence per retired PNG**, in the task that retires it. Section 3 says the report gains a dated note without assigning it to a pull request.
- **Story-page provenance vocabulary gains "committed replay" and "replay file" now**, since this pull request is the first to put replay data behind a story page.
- **The prose says what the gate does**: computed inputs are compared to within 1e-12, not exactly. Section 9's draft sentence says "compare its inputs exactly", which the merged gate no longer does (Task 8 corrects the spec's own wording).
- **Contact sheets go under `docs/plans/2026-11-23-replay-figures/`**, matching this plan's date, not the spec's `2026-11-19-replay-and-animation/`.
- **`actions/upload-artifact` moves to `@v7`.** Its published `action.yml` runs on `node20` at `v4` and `v5` and on `node24` at `v6` and `v7` (read with `gh api`, below); `v7.0.2` was released on 2026-10-07. The spec's YAML sample shows `@v4`.
- **Owed item 6 needs no change.** On `main` at `89f7fb5`, `scripts/record_replays.py` has no `_compare_one`; `_read_baseline` and `_record_and_compare` catch every `Exception` per demo and report it (read). Only the `meta` field naming is planned (Task 9).

## Measurements

Run on 2026-11-23 in the worktree at `89f7fb5`, macOS arm64, Python 3.14.0, Matplotlib 3.10.8, through `uv run`. Probe scripts were in a session scratch directory and are not kept.

- **P1. The committed density archive.** `load_replay(Path("assets/replays/demo_density.npz"))`: 60 observations, 2,000 particles, `cloud` (61, 500, 3) float32, truth (1.5, 2.0, 0.010), clocks at −6, −3, 0, 3, 6, prior ranges [−8, 8], [0.1, 5.0], [0.001, 0.03]. Final mean (1.50232, 2.13582, 0.00953) ± (0.08289, 0.11569, 0.00046).
- **P2. Profiles of the last row**, on `np.linspace(-8.0, 8.0, 321)`, from `cloud[-1]` cast to float64, `np.median` and `np.percentile(..., [16, 84])`: at x = −6.0 (index 40) 0.0000196, 0.0000095, 0.0000349; at x = 0.0 (index 160) 0.0074162, 0.0071591, 0.0076912; at x = 1.5 (index 190) 0.0095141, 0.0090899, 0.0100017 (median, 16th, 84th). The true profile peaks at 0.0100000 and the 84th percentile at 0.0100037, so the profile panel's top is 1.08 × 0.0100037 = 0.0108040.
- **P3. Trace limits and headers on the committed archive**, by the rule of Task 3: μ (0.93006, 4.35585), σ (1.49706, 4.73440), A (0.0045146, 0.0123533); headers `μ   true 1.50   1.50 ± 0.08`, `σ   true 2.00   2.14 ± 0.12`, `A   true 0.0100   0.0095 ± 0.0005`.
- **P4. The committed range study.** `assets/echolocation_range_study.json`: 72 records, ranges 2.0, 2.6, 3.5, 4.6, 6.1, 8.0, tolerances `position_error_max` 1.0 and `mass_error_max` 0.04. Pass counts from both limits, and from the records' `passed`: 12, 12, 11, 8, 3, 0. From the position limit alone the same; from the mass limit alone 12, 12, 12, 11, 11, 12. Every error is positive (smallest 0.0105 and 0.000204).
- **P5. The command against `record()`** at 3 observations and 64 particles: `record()` ends at (1.768, 2.495, 0.0101) ± (0.424, 0.950, 0.0061) with ESS 53.2; today's shared-generator loop ends at (1.182, 1.132, 0.0190).
- **P6. Guards.** `_style_violations('ax.plot(x, "rate", "r-", data=frame)')` returns `[]`; `_resolve_source("demo_files/a%00b.png", ...)` raises `ValueError: lstat: embedded null character in path` (both as the spec's M14).
- **P7. Colour walk.** On a scratch figure drawn under `style(theme)` with role colours, every visible artist's colour was a palette colour once edges of zero line width were ignored; without that rule the figure patch's white edge (`figure.edgecolor`, width 0) appeared. `fill_between` gives one `FillBetweenPolyCollection` whose single path holds each x twice, once per edge.
- **P8. One-particle recordings.** `demo_1d.record(n_observations=2, n_particles=1)` and the same for `demo_multi_mass` validate; every `predicted_std` is 0; the dots' readings differ from `predicted_mean` by at most 6.4e-11 and 1.4e-9 (float32 rounding of the dots); `_assert_predictions_match_dots` fails both with "inf s.d./sqrt(m)".
- **P9. Rows chosen at run time.** `demo_2d`'s largest |x–y correlation| after the prior is at row 5 (−0.326; row 1 is −0.287); flipping that row's sign is caught ("row 5: dots' correlation differs from cov's by 0.712"). Echolocation's smallest fraction of dots in view is row 0 (0.004, bound 0.0169); setting its weight to that fraction plus 0.036 is caught.
- **P10. `actions/upload-artifact`.** `gh api repos/actions/upload-artifact/contents/action.yml?ref=<tag>`: `using: 'node20'` at `v4` and `v5`, `using: 'node24'` at `v6` and `v7`. The inputs this workflow uses (`name`, `path`, `if-no-files-found`) are unchanged at `v7`.

---

### Task 1: The rendered-site check learns a manifest of several plots per page

**Files:**
- Modify: `scripts/check_site_figures.py`
- Test: `tests/test_check_site_figures.py`

**Interfaces:**
- Produces, in `scripts/check_site_figures.py`:
  - `class Fig(NamedTuple)` with `kind: str`, `height: int`, `clip: str | None = None`. A `NamedTuple`, not a dataclass: the tests import the script by path without registering it (spec, M14).
  - `EXPECTED_FIGURES: dict[str, tuple[Fig, ...]]`, replacing `EXPECTED_PAGES`, keyed by output page, each tuple in document order. Its value after this task, the site as it is today:

```python
EXPECTED_FIGURES: dict[str, tuple[Fig, ...]] = {
    "story/clocks-as-gravimeters.html": (Fig("plot", 420),),
    "story/one-clock-is-not-enough.html": (Fig("plot", 420),),
    "story/the-search-in-one-dimension.html": (Fig("plot", 540),),
    "story/into-the-plane.html": (Fig("plot", 540),),
    "story/two-hidden-masses.html": (Fig("plot", 420),),
    "story/how-many-masses.html": (Fig("plot", 420),),
    "story/beyond-point-masses.html": (Fig("plot", 420),),
    "story/gravitational-echolocation.html": (Fig("plot", 420),),
    "method/the-particle-filter.html": (Fig("plot", 540),),
}
```

  - `figure_failures(output_dir: Path) -> list[str]`, signature unchanged.

- [ ] **Step 1: Write the failing tests.** The autouse fixture now sets `EXPECTED_FIGURES` to `{PAGE: (Fig("plot", 420),)}`; `REAL_EXPECTED_FIGURES` captures the real value before it, as `REAL_EXPECTED_PAGES` does today. Add two builders beside `_build`: `_pair(n: int, height: int) -> tuple[str, str]`, the light and dark wrappers of figure `n`, whose images are `demo_files/light{n}.png` and `demo_files/dark{n}.png` with the file name as content; and `_page(tmp_path, wrappers: Sequence[str], page: str = PAGE) -> Path`, which writes those files and a page holding the wrappers in the order given.

Existing tests keep their reasons, which stay substrings of the new messages. Every per-wrapper message is now labelled `f"{page}: figure {i} {theme}-content"` with `i` counted from 1, so three existing assertions change:

- `test_a_well_formed_tall_page_passes`: the two expected failures become `f"{PAGE}: figure 1 {theme}-content: height 540, expected 420"`; the two `EXPECTED_PAGES` patches become `EXPECTED_FIGURES` patches with `(Fig("plot", 540),)` and `(Fig("plot", 420),)`.
- `test_a_dark_caption_in_a_second_dark_wrapper`: the reason becomes `expected 1 figure(s), 2 theme wrappers; found 3`.
- `test_expected_pages_match_the_plot_pages_and_their_heights` becomes `test_the_manifest_lists_the_nine_plot_pages`: the key set equals the nine `PLOT_PAGES` as `.html`, every `Fig` has kind `"plot"` and `clip is None`, and each page's single height is 540 if its only plot cell contains `figsize=(7.5, 5.4)` and 420 otherwise. (Task 2 moves this comparison to the source guard.)

New tests:

- `test_a_page_with_two_plots_passes`: manifest `(Fig("plot", 540), Fig("plot", 420))`; `_page(tmp_path, [*_pair(1, 540), *_pair(2, 420)])` gives `[]`.
- `test_two_plots_with_their_heights_swapped`: same manifest, pairs built at 420 then 540; the failures are exactly, in this order, `f"{PAGE}: figure 1 light-content: height 420, expected 540"`, the same for `figure 1 dark-content`, then `figure 2 light-content: height 540, expected 420` and `figure 2 dark-content: height 540, expected 420`.
- `test_wrappers_out_of_order`: same manifest; wrappers in the order light 1, light 2, dark 1, dark 2. The failures are exactly `f"{PAGE}: figure 1: expected light-content then dark-content, found light-content then light-content"` and `f"{PAGE}: figure 2: expected light-content then dark-content, found dark-content then dark-content"`.
- `test_a_missing_second_figure`: same manifest, only `_pair(1, 540)`; the one failure is `f"{PAGE}: expected 2 figure(s), 4 theme wrappers; found 2"`.
- `test_identical_light_and_dark_content`: `_build(tmp_path)`, then `demo_files/dark.png` rewritten with `b"light.png"`; one failure containing `figure 1: light and dark images have the same content`.
- `test_an_encoded_nul_is_a_failure`: `_build(tmp_path, dark=[("demo_files/dark%00.png", 750)])`; one failure containing `image path contains a NUL`. (Today this raises `ValueError`: P6.)
- `test_a_wrapper_on_an_unlisted_page`: `_build(tmp_path)`, plus `story/other.html` holding one light wrapper from `_pair(1, 420)`; `figure_failures` returns exactly `["story/other.html: not in the figure manifest, but has 1 theme wrapper(s)"]`.
- `test_a_listed_page_with_no_figures_passes`: manifest `{PAGE: ()}` and a page with no wrappers gives `[]`. (The shape pull request 3 needs for a page whose only figure is a clip is not this; the test pins that an empty tuple means "none".)

- [ ] **Step 2: Run** `uv run pytest tests/test_check_site_figures.py -q`; the new tests fail (no `Fig`, no `EXPECTED_FIGURES`), and `test_an_encoded_nul_is_a_failure` errors with the `ValueError` of P6.

- [ ] **Step 3: Implement.**

- `figure_failures` checks each page in `EXPECTED_FIGURES` as today (a missing page is still `f"{page}: page is missing from {output_dir}"`), then every `*.html` under `output_dir` outside `site_libs/` whose relative path is not a key: any theme wrapper there is the unlisted-page failure above.
- `_page_failures(page, path, figures, output_dir)` keeps the both-classes and nesting checks first. Then, with `n = len(figures)`, a wrapper count other than `2 * n` is the single count failure and nothing more is checked on that page. Otherwise wrappers are taken in document order in pairs: pair `i` must be light then dark, else the order failure for that pair and no further checks on it. A pair whose `Fig.kind` is not `"plot"` raises `ValueError(f"{page}: figure {i}: no check for kind {kind!r}")`; pull request 3 replaces that line with the clip checks.
- Per plot pair, today's checks with the new label, plus: after both files resolve and are different paths, `light.read_bytes() == dark.read_bytes()` is the content failure.
- `_resolve_source`: after `unquote`, a path containing `"\x00"` returns `(None, f"image path contains a NUL: {src!r}")`, before anything touches the file system.
- `main` prints `f"Figure contract OK: {len(EXPECTED_FIGURES)} pages, {n} figures checked."` with `n` the total number of `Fig` entries. Update the module docstring: several plots per page, in manifest order, and unlisted pages.

- [ ] **Step 4: Tests pass; full gate; commit and push** as `Check each page's plots against an ordered manifest; report an encoded NUL`.

---

### Task 2: The source guard learns several plot cells, `data=` by position and retired assets

**Files:**
- Modify: `tests/test_site_figure_style.py`, `tests/test_check_site_figures.py`

**Interfaces:**
- Consumes: `Fig`, `EXPECTED_FIGURES` from Task 1, imported by path as `tests/test_check_site_figures.py` does.
- Produces, in `tests/test_site_figure_style.py`:
  - `_is_plot_cell(cell: str) -> bool` and `_plot_cells(path: Path) -> list[str]` (all plot cells of a page, in order).
  - `FIGURE_FUNCTION_HEIGHTS: dict[str, int]`, empty here; Tasks 3 and 5 add `"density_figure": 540` and `"range_study_figure": 540`.
  - `_cell_height(cell: str) -> int`: the height of a cell calling a name in `FIGURE_FUNCTION_HEIGHTS`; else 540 if the cell contains `figsize=(7.5, 5.4)`; else 420.
  - `RETIRED_ASSETS: tuple[str, ...] = ()`; Tasks 4 and 6 add `"demo_density.png"` and `"echolocation_range_study.png"`. `_retired_asset_links(text: str, retired: Sequence[str]) -> list[str]` returns each link or embed target in `text` that names a retired file under an `assets/` directory.
  - `LIBRARY_DRAWING_MODULES: tuple[str, ...] = ()`, paths from the repository root; Tasks 3, 4 and 5 add `src/clocks/_figures.py`, `src/clocks/_demos/demo_density.py`, `src/clocks/_echo_study.py`.

- [ ] **Step 1: Write the failing tests.**

Plot cells:

- `test_a_cell_is_a_plot_cell`, parametrized with IDs: `renderings` `"#| renderings: [light, dark]\nx = 1"` → True; `subplots` `"fig, ax = plt.subplots()"` → True; `figure` `"fig = plt.figure()"` → True; `show` `"show(fig)"` → True; `figure_function` `"fig = density_figure(replay, colors)"` → True; `not_a_plot` `"table = Markdown(text)\nprint(table)"` → False; `a_method_named_figure` `"ax.figure.canvas.draw()"` → False.
- The three cell-contract tests and `test_cell_has_no_unstyled_colour_or_display` are parametrized over every plot cell, not a page's only one: `PLOT_CELLS = [pytest.param(page, index, id=f"{page}#{index + 1}") for page in PLOT_PAGES for index in range(len(_plot_cells(SITE / page)))]`. `_only_plot_cell` is deleted.
- `test_manifest_matches_the_source` (each of `PLOT_PAGES`): `tuple((fig.kind, fig.height) for fig in EXPECTED_FIGURES[page.replace(".qmd", ".html")])` equals `tuple(("plot", _cell_height(cell)) for cell in _plot_cells(SITE / page))`.
- `test_the_manifest_lists_exactly_the_plot_pages`: the manifest's key set equals the nine `PLOT_PAGES` as `.html`, and `len(PLOT_PAGES) == 9`. Remove `test_the_manifest_lists_the_nine_plot_pages` from `tests/test_check_site_figures.py`, and that file's import of `test_site_figure_style` if nothing else uses it.

`data=` by position: add to `test_style_violations_reports_unstyled_calls` the case `('ax.plot(x, "rate", "r-", data=frame)', "positional format string")`, with the ID `data_third_argument` (give the list's cases IDs as `pytest.param`s if that is needed to name one). The existing allowed case `ax.plot("step", "rate", data=frame, color=colors.ink)` stays allowed.

Retired assets, against `retired = ("demo_density.png",)`:

- `test_retired_asset_links_are_found`, IDs and inputs, each giving a one-element list holding the target as written: `md_image` `"![a](../assets/demo_density.png)"`; `md_link` `"[a](assets/demo_density.png)"`; `reference` `"[a]: ../site/assets/demo_density.png"` (a line of its own); `html_img` `'<img src="../assets/demo_density.png" alt="">'`; `github_url` `"[a](https://github.com/jbwhit/clocks/blob/main/assets/demo_density.png)"`; `with_query` `"![a](../assets/demo_density.png?v=2)"`.
- `test_names_that_are_not_links_are_allowed`, IDs `command_output` `"uv run demo-density             # → output/demo_density.png"`, `inline_code` `` "`assets/demo_density.png`" ``, `output_link` `"[a](output/demo_density.png)"`, `other_asset` `"![a](../assets/demo_2d.gif)"`: each gives `[]`.
- `test_no_page_links_to_a_retired_asset` (each of `README.md` and every `.qmd` under `site/` outside `_output` and `.quarto`, ID the path): `_retired_asset_links(text, RETIRED_ASSETS) == []`.

Library drawing code:

- `test_library_drawing_code_has_no_unstyled_colour` (each of `LIBRARY_DRAWING_MODULES`): `_style_violations` on the module's text is `[]`. Empty here; Task 3 gives it its first case.
- `test_every_figure_builder_module_is_scanned`: every `.py` under `src/clocks/` defining a function whose name ends in `_figure` is in `LIBRARY_DRAWING_MODULES`. Here that is `src/clocks/_echo_study.py` (`write_summary_figure`), which still draws with literal colours, so this test is written now and marked `xfail(strict=True, reason="Task 5 replaces the range-study drawing")`; Task 5 removes the mark.

- [ ] **Step 2: Run** `uv run pytest tests/test_site_figure_style.py tests/test_check_site_figures.py -q`. `test_a_cell_is_a_plot_cell[show]`, `[renderings]` and `[figure_function]`, the `data_third_argument` case and the retired-asset tests fail; the strict `xfail` is reported as expected.

- [ ] **Step 3: Implement.**

- `_is_plot_cell`: true when a line of the cell starts with `#| renderings:`, or its syntax tree holds a call to `plt.subplots`, `plt.figure`, a bare `show`, or any name or attribute ending in `_figure` that is called (`ax.figure.canvas.draw()` calls `draw`, not `figure`).
- `_style_violations`: with `data=`, a string at any positional index from 2 on is a format string; without it, any positional string is. Update the comment.
- `_retired_asset_links`: collect targets from Markdown links and images (`](target`), reference definitions (`[label]: target` at the start of a line) and HTML `src=`/`href=` attributes; for each, take `urlsplit(target).path` and report it when its last two `PurePosixPath` parts are `("assets", name)` for a retired `name`.
- Update the module docstring: plot cells, the manifest, retired assets, library modules.

- [ ] **Step 4: Tests pass; full gate; commit and push** as `Guard every plot cell on a page, data= by position, and links to retired assets`.

---

### Task 3: `density_figure` and the trace helpers

**Files:**
- Create: `src/clocks/_figures.py`, `tests/_figure_artists.py`, `tests/test_figures.py`
- Modify: `src/clocks/viz.py`, `tests/test_site_figure_style.py`

**Interfaces:**
- Consumes: `Replay`, `load_replay`, `validate_replay` from `clocks._replay`; `Palette`, `style`, `THEMES` from `clocks._style`.
- Produces, in `src/clocks/_figures.py`:
  - `PROFILE_POINTS = 321`.
  - `trace_limits(mean: NDArray[np.float64], sd: NDArray[np.float64], truth: float, prior_range: Sequence[float]) -> tuple[float, float]`. `mean` and `sd` are one parameter's column over rows 0 to T; row 0 is left out. With `lo = min(truth, min(mean[1:] - sd[1:]))` and `hi = max(truth, max(mean[1:] + sd[1:]))`, the pad is `0.04 * (hi - lo)`, or `0.04 * (prior_range[1] - prior_range[0])` when `hi == lo`; the result is `(max(lo - pad, prior_range[0]), min(hi + pad, prior_range[1]))`.
  - `trace_header(name: str, truth: float, mean: float, sd: float, prior_range: Sequence[float]) -> str`: `f"{name}   true {truth:.{d}f}   {mean:.{d}f} ± {sd:.{d}f}"` (three spaces between parts) with `d = max(2, 2 - math.floor(math.log10(prior_range[1] - prior_range[0])))`: 2 for μ and σ, 4 for A.
  - `density_figure(replay: Replay, colors: Palette) -> Figure`.
- Produces, in `tests/_figure_artists.py` (a helper module, not collected): `palette_hexes(colors: Palette) -> set[str]` (every field, `ordered` included, as lower-case `#rrggbb`); `visible_colours(fig: Figure) -> set[str]` (below); `by_gid(container, gid: str) -> list[Artist]` (every artist under `container`, through `findobj`, with that gid); `band_edges(collection, x: NDArray) -> tuple[NDArray, NDArray]` (for each x, the smallest and largest y among the band's path vertices at that x); `png_size(path: Path) -> tuple[int, int]` (width and height from the IHDR bytes 16 to 24).
- `clocks.viz` gains `density_figure` and `load_replay` in its imports and `__all__`.

**The density figure's artists.** The tests find artists by gid, so these are part of the interface. Axes gids, in `fig.axes` order: `"profile"` (the top row, full width), then `"trace-μ"`, `"trace-σ"`, `"trace-A"` (one row of three below, named from `meta["param_names"]`). In `"profile"`: a `Line2D` `"true-profile"`; a `Line2D` `"median-profile"`; the `fill_between` collection `"profile-band"`; a `Line2D` `"clocks"` with `linestyle="none"`, `marker="s"`, at y = 0, `clip_on=False`; and one direct label each, `"true profile"` (colour `truth`), `"pointwise median and 68% band"` (`belief`), `"clocks"` (`ink`), placed where they do not collide (the contact sheet checks). x-limits (−8, 8), the μ prior range; y-limits from 0 to 1.08 times the larger of the true profile's and the band's maxima. In each trace: a `Line2D` `"trace-mean"` over `np.arange(T + 1)`; the `fill_between` collection `"trace-band"` from mean − sd to mean + sd; a `Line2D` `"trace-truth"` from `[0, T]` at the truth (not `axhline`, so its data are the truth); the title from `trace_header` at row T; x-limits (0, T); y-limits from `trace_limits`; x label "observations". Gridspec height ratios and spacing are the implementer's, inside `figsize=(7.5, 5.4)`.

- [ ] **Step 1: Write the failing tests** in `tests/test_figures.py`.

A literal replay that `validate_replay` accepts (checked while planning), with T = 2, four particles and four dots:

```python
F8 = np.float64
CONTROLS = {"ess_target": 0.8, "rejuvenation_steps": 2, "proposal_scale": 2.38,
            "resampling": "systematic"}
CREATED = {"python": "3.14.0", "numpy": "2.4.2", "scipy": "1.17.1", "platform": "test"}


def density_replay() -> Replay:
    meta = {"format": "clocks-replay", "version": 1, "demo": "demo_density",
            "kind": "filter", "seed": 7, "n_observations": 2, "n_particles": 4,
            "noise_std": 0.005, "track_offset": 1.0, "param_names": ["μ", "σ", "A"],
            "prior_ranges": [[-8.0, 8.0], [0.1, 5.0], [0.001, 0.03]],
            "controls": CONTROLS, "cloud_size": 4, "created_with": CREATED}
    sds = (np.array([4.6, 1.4, 0.0084]), np.array([0.6, 0.5, 0.002]),
           np.array([0.1, 0.15, 0.0006]))
    arrays = {
        "clock_positions": np.array([[-6.0], [-3.0], [0.0], [3.0], [6.0]], F8),
        "true_params": np.array([1.5, 2.0, 0.010], F8),
        "true_rates": np.array([0.999, 0.996, 0.992, 0.991, 0.997], F8),
        "observations": np.array([[0.998, 0.995, 0.993, 0.990, 0.996],
                                  [1.000, 0.997, 0.991, 0.992, 0.998]], F8),
        "mean": np.array([[0.0, 2.55, 0.0155], [1.2, 2.3, 0.011],
                          [1.45, 2.1, 0.0102]], F8),
        "cov": np.stack([np.diag(s**2) for s in sds]).astype(F8),
        "ess": np.array([4.0, 3.5, 3.8], F8),
        "log_evidence": np.array([0.0, 5.0, 11.0], F8),
        "tempering_stages": np.array([2, 1], np.int64),
        "mh_proposals": np.array([8, 8], np.int64),
        "mh_acceptances": np.array([3, 4], np.int64),
        "cloud": np.array([
            [[-5.0, 0.5, 0.004], [-1.0, 2.0, 0.012], [2.0, 3.5, 0.020], [6.0, 4.5, 0.028]],
            [[0.5, 1.8, 0.009], [1.0, 2.2, 0.011], [1.5, 2.6, 0.012], [2.0, 2.9, 0.013]],
            [[1.3, 1.9, 0.0098], [1.4, 2.0, 0.0100], [1.5, 2.2, 0.0104], [1.6, 2.3, 0.0106]],
        ], np.float32),
    }
    return Replay(meta, arrays)
```

A copy with one value changed is `Replay(meta, {**arrays, key: edited})`, never an in-place edit of a shared array. "Drawn in a theme" means inside `with style(theme) as colors:`, with the figure closed in a `finally`. Profiles in the tests are computed by an explicit loop, `np.array([a * np.exp(-((x - mu) ** 2) / (2 * s**2)) for mu, s, a in dots])`, with `dots = cloud[-1].astype(np.float64)` and `x = np.linspace(-8.0, 8.0, 321)`. Standard deviations are `np.sqrt(np.diagonal(cov, axis1=1, axis2=2))`. Tests marked "(each theme)" are parametrized over `THEMES`.

The helper module:

- `test_visible_colours_finds_a_literal_colour`: a figure drawn in the light theme with one `ax.plot([0, 1], [0, 1], color="#ff0000")`: `"#ff0000" in visible_colours(fig)`, and `visible_colours(fig) - palette_hexes(colors) == {"#ff0000"}`. (`visible_colours` walks `fig.findobj()`; for each visible artist it takes `get_color`, `get_facecolor`, `get_markerfacecolor`, `get_markeredgecolor` and, only when the artist's line width is above 0, `get_edgecolor`, where each exists; every colour with alpha above 0 is added as `to_hex(..., keep_alpha=False)`. Without the line-width rule the figure patch's white edge is reported: P7.)

The figure:

- `test_density_figure_size_and_axes` (each theme): `tuple(fig.get_size_inches()) == (7.5, 5.4)`; `[ax.get_gid() for ax in fig.axes] == ["profile", "trace-μ", "trace-σ", "trace-A"]`.
- `test_true_profile` (each theme): `"true-profile"`'s x data equal `np.linspace(-8.0, 8.0, 321)` exactly, its y data equal `0.010 * np.exp(-((x - 1.5) ** 2) / (2 * 2.0**2))` with `rtol=1e-12`, and its colour is `colors.truth`.
- `test_median_and_band_are_pointwise_percentiles_of_the_last_row` (each theme): `"median-profile"`'s y data equal `np.median(profiles, axis=0)`, and `band_edges(band, x)` equal `np.percentile(profiles, 16, axis=0)` and `np.percentile(profiles, 84, axis=0)`, all with `rtol=1e-12`; the line's colour and the band's face colour (alpha dropped) are `colors.belief`.
- `test_only_the_last_cloud_row_feeds_the_profile`: with `cloud[1]` replaced by `cloud[0]`, the median line and band edges are `np.array_equal` to the original's; with 1.0 added to μ in every dot of `cloud[-1]`, the median line moves by more than 1e-4 somewhere.
- `test_changing_the_truth_moves_only_the_truth`: with `true_params[0] = 0.0`, `"true-profile"` moves, `"median-profile"` does not, `"trace-truth"` in `"trace-μ"` moves to `[0.0, 0.0]` and the σ and A truth lines do not.
- `test_clocks_are_ink_squares_on_the_axis` (each theme): `"clocks"` has x data `[-6.0, -3.0, 0.0, 3.0, 6.0]`, y data all 0, marker `"s"`, line style `"None"`, colour `colors.ink`.
- `test_profile_limits`: x-limits `(-8.0, 8.0)`; y-limits `(0.0, 1.08 * max(true.max(), upper.max()))`, `rel=1e-12`.
- `test_traces_follow_the_archive` (each theme): for each column j, in `"trace-<name>"`: `"trace-mean"` has x data `[0, 1, 2]` and y data `mean[:, j]`; `band_edges` are `mean[:, j] ∓ sd[:, j]` (`rtol=1e-12`); `"trace-truth"` has x data `[0, 2]` and y data `[t_j, t_j]`; colours `belief`, `belief`, `truth`; x-limits `(0, 2)`; y-limits, `rel=1e-9`, μ `(0.552, 1.848)`, σ `(1.76, 2.84)`, A `(0.00884, 0.01316)`.
- `test_changing_one_row_moves_only_that_point`: with `mean[1, 0] += 0.1`, `"trace-mean"` in `"trace-μ"` differs from the original's at index 1 by 0.1 (`rel=1e-12`) and nowhere else.
- `test_trace_headers`: the three titles are `"μ   true 1.50   1.45 ± 0.10"`, `"σ   true 2.00   2.10 ± 0.15"`, `"A   true 0.0100   0.0102 ± 0.0006"`.
- `test_trace_limits`, IDs and values (`pytest.approx`, `rel=1e-12`):
  - `prior_row_left_out`: `trace_limits(np.array([0.0, 1.2, 1.45]), np.array([4.6, 0.6, 0.1]), 1.5, (-8.0, 8.0)) == (0.552, 1.848)`; with row 0 included it would be (−4.968, 4.968).
  - `clipped_to_prior`: `trace_limits(np.array([0.0, 7.9]), np.array([1.0, 0.5]), 7.5, (-8.0, 8.0)) == (7.36, 8.0)`.
  - `zero_width`: `trace_limits(np.array([0.0, 1.5, 1.5]), np.array([3.0, 0.0, 0.0]), 1.5, (-8.0, 8.0)) == (0.86, 2.14)`.
- `test_density_figure_draws_a_one_observation_run`: the fixture cut to its first observation (`n_observations` 1; `mean`, `cov`, `ess`, `log_evidence`, `cloud` to two rows; the integer arrays and `observations` to one), which `validate_replay` accepts; drawn with `warnings.simplefilter("error")`, each `"trace-mean"` has x data `[0, 1]` and x-limits `(0, 1)`.
- `test_density_figure_refuses_another_demo`: the fixture with `meta["demo"] = "demo_1d"` raises `ValueError` matching `density_figure needs a demo_density replay, got 'demo_1d'`.
- `test_density_figure_on_the_committed_archive` (P2, P3): drawn from `load_replay(ROOT / "assets/replays/demo_density.npz")`: `"median-profile"` at indices 40, 160, 190 is `0.0000196`, `0.0074162`, `0.0095141`, and the band's lower and upper edges there are `0.0000095`, `0.0071591`, `0.0090899` and `0.0000349`, `0.0076912`, `0.0100017`, each `abs=5e-8`; the profile's top y-limit is `0.0108040` (`abs=5e-8`); the titles are `"μ   true 1.50   1.50 ± 0.08"`, `"σ   true 2.00   2.14 ± 0.12"`, `"A   true 0.0100   0.0095 ± 0.0005"`.
- `test_every_visible_colour_is_in_the_palette` (each theme), on the committed archive: `visible_colours(fig) <= palette_hexes(colors)`.
- `test_figures_module_imports_no_forward_model`: the syntax tree of `src/clocks/_figures.py` imports, from `clocks`, only `clocks._replay` and `clocks._style`.
- `test_viz_exports_the_density_figure`: `clocks.viz.density_figure is clocks._figures.density_figure`; `clocks.viz.load_replay is clocks._replay.load_replay`; both names are in `clocks.viz.__all__`.

In `tests/test_site_figure_style.py`: `FIGURE_FUNCTION_HEIGHTS["density_figure"] = 540`; `LIBRARY_DRAWING_MODULES` gains `"src/clocks/_figures.py"`; and a new `test_figure_function_heights_are_the_drawn_ones` (each name in `FIGURE_FUNCTION_HEIGHTS`) draws the committed data in the light theme through a name-to-drawer map in the test and asserts `round(fig.get_figheight() * 100) == FIGURE_FUNCTION_HEIGHTS[name]` and `round(fig.get_figwidth() * 100) == 750`.

- [ ] **Step 2: Run** `uv run pytest tests/test_figures.py tests/test_site_figure_style.py -q`; they fail on `ModuleNotFoundError: clocks._figures`.

- [ ] **Step 3: Implement.**

- `density_figure` raises the `ValueError` above when `replay.meta["demo"] != "demo_density"`, then draws as "The density figure's artists" says. The profiles are computed in float64 from `cloud[-1].astype(np.float64)`; the band is `np.percentile(profiles, [16, 84], axis=0)` (NumPy's default linear method) and the median `np.median(profiles, axis=0)`. Every artist is drawn with keyword styling only (`color=`, `marker=`, `linestyle=`), since the source guard reads this module.
- `src/clocks/_figures.py` imports from `clocks` only `clocks._replay` (for `Replay`) and `clocks._style` (for `Palette`).
- `clocks.viz`: import `density_figure` from `clocks._figures` and `load_replay` from `clocks._replay`; add both to `__all__`; name `_figures` in the module docstring.

- [ ] **Step 4: Tests pass; full gate; commit and push** as `Draw the density figure from a replay, with shared trace helpers`.

---

### Task 4: `demo-density` draws the recorded run, and the density figure replaces its PNG

One change, as section 11 requires: the command's switch to `record()` and the new figure, the page cell, the removal of the PNG and its hash row, and the provenance sentences that describe them.

**Files:**
- Modify: `src/clocks/_demos/demo_density.py`, `site/story/beyond-point-masses.qmd`, `site/reproduce/reproducibility.qmd`, `docs/2026-08-16-development-calibration.md`, `scripts/check_site_figures.py`
- Delete: `assets/demo_density.png`, `site/assets/demo_density.png`
- Test: `tests/test_density_demo.py`, `tests/test_generated_assets.py`, `tests/test_site_figure_style.py`

**Interfaces:**
- Consumes: `record()` (already in `demo_density.py` from pull request 1), `density_figure` (Task 3), `style` from `clocks._style`; `RETIRED_ASSETS`, `LIBRARY_DRAWING_MODULES` (Task 2); `EXPECTED_FIGURES` (Task 1).
- Produces: `demo_density._run_demo(args)` as described below; `_save_summary` is deleted. `build_parser`, `main`, `build_density_filter`, `scenario` and `record` keep their signatures. In `tests/test_generated_assets.py`, `RETIRED_VISUALS: dict[str, str]` (file name to the SHA-256 it was released with), which Task 6 extends.

- [ ] **Step 1: Write the failing tests.**

In `tests/test_density_demo.py` (each reads the PNG size with `png_size` from `tests/_figure_artists.py`):

- `test_density_command_prints_and_draws_the_recorded_run`: `main(["--output", str(tmp_path / "new" / "density.png"), "--observations", "3", "--particles", "64"]) == 0`. With `expected = demo_density.record(n_observations=3, n_particles=64)`, `m = expected.arrays["mean"][-1]`, `s` its standard deviations and `ess = expected.arrays["ess"][-1]`, standard output holds the line `Final estimate:` followed by exactly these four lines: `f"  mu    = {m[0]:.3f} ± {s[0]:.3f}  (true: 1.5)"`, `f"  sigma = {m[1]:.3f} ± {s[1]:.3f}  (true: 2.0)"`, `f"  A     = {m[2]:.3f} ± {s[2]:.3f}  (true: 0.01)"`, `f"  ESS   = {ess:.0f} / 64"`; and `png_size(tmp_path / "new" / "density.png") == (1500, 1080)`. (P5: today's command prints μ = 1.182 here, `record()` 1.768.)
- `test_density_command_draws_the_density_figure_in_the_light_theme`: with `demo_density.density_figure` wrapped by a spy that records its arguments and calls through, one run at 2 observations and 16 particles calls it once, with a replay whose `meta["n_observations"] == 2` and `meta["n_particles"] == 16` and with `colors == palette("light")`.
- `test_density_command_writes_only_its_png`: with the working directory changed to `tmp_path` (`monkeypatch.chdir`), `main(["--output", "out/d.png", "--observations", "1", "--particles", "16"])` leaves exactly one file under `tmp_path`, `out/d.png`.

In `tests/test_generated_assets.py`:

- Remove the `"demo_density.png"` row from `EXPECTED_VISUAL_HASHES`.
- `RETIRED_VISUALS = {"demo_density.png": "a7a6e6e9628640ac08e89152b8471d37db981c264380ba567634032a0f59dc7a"}`.
- `test_retired_visuals_are_gone_but_recorded` (each of `RETIRED_VISUALS`): neither `ASSETS / name` nor `SITE_ASSETS / name` exists; the calibration report still contains the hash; and the report matches `rf"`{re.escape(name)}` was removed from both trees on \d{{4}}-\d{{2}}-\d{{2}};"`.
- `test_density_provenance_names_its_run`: with `meta = load_replay(ROOT / "assets/replays/demo_density.npz").meta`, the whitespace-normalised Reproducibility page contains `"**Replay data.** The density figure on [Beyond Point Masses](../story/beyond-point-masses.qmd) is drawn when the site is built, from `assets/replays/demo_density.npz`, the recorder's archive of the demo's packaged run: "` + `f"{meta['n_observations']} observations, {meta['n_particles']:,} particles and seed {meta['seed']}. "`; and does not contain `"density PNG"`.
- `test_provenance_section_describes_figures_drawn_at_build_time`: the raw page contains `"## Where the figures come from {#sec-figures}"`; the normalised page contains `"[development calibration report](https://github.com/jbwhit/clocks/blob/main/docs/2026-08-16-development-calibration.md) is the record of the August 2026 release"`.
- `PROVENANCE_VOCABULARY` gains `"committed replay"` and `"replay file"`.

In `tests/test_site_figure_style.py`: `RETIRED_ASSETS = ("demo_density.png",)`; `LIBRARY_DRAWING_MODULES` gains `"src/clocks/_demos/demo_density.py"`. In `scripts/check_site_figures.py`, `"story/beyond-point-masses.html": (Fig("plot", 540), Fig("plot", 420))`.

- [ ] **Step 2: Run** `uv run pytest tests/test_density_demo.py tests/test_generated_assets.py tests/test_site_figure_style.py tests/test_check_site_figures.py -q`. The command tests fail (today's run, a 2250 by 600 PNG); the retired-asset, provenance and manifest tests fail.

- [ ] **Step 3: Implement.**

`demo_density._run_demo(args)`: `replay = record(n_observations=args.observations, n_particles=args.particles)`; print the truth and the true rates from `replay.arrays` in today's format (`True density: mu=1.5, sigma=2.0, amplitude=0.01`, then `True rates: [...]`); for k = 20, 40, … up to the last observation, print today's `After k observations:` block from row k of `mean`, the standard deviations of `cov` and `ess`; then `Final estimate:` from the last row; then, inside `with style("light") as colors:`, `fig = density_figure(replay, colors)`, `args.output.parent.mkdir(parents=True, exist_ok=True)`, `fig.savefig(args.output, dpi=200)`, `plt.close(fig)`; then `Saved: <path>`. `_format_estimate` takes the mean, the standard deviation and the truth as floats. Delete `_save_summary` and every import only it used. Imports stay after `matplotlib.use("Agg")` (`tests/test_cli.py::test_demo_selects_agg_before_importing_pyplot`). Update `record()`'s docstring: the command now runs exactly this.

`site/story/beyond-point-masses.qmd`: replace the Markdown image on line 17 with this cell, in the same place:

````markdown
```{python}
#| code-fold: true
#| renderings: [light, dark]
#| fig-cap:
#|   - "The density demo, one simulated run. Top: the true profile and, from 500 particles sampled from the final posterior, the pointwise median profile with its central 68% band; squares mark the clocks. Below: each parameter's posterior mean after each observation, with a band of one standard deviation, and its true value."
#|   - "The density demo, one simulated run. Top: the true profile and, from 500 particles sampled from the final posterior, the pointwise median profile with its central 68% band; squares mark the clocks. Below: each parameter's posterior mean after each observation, with a band of one standard deviation, and its true value."
from pathlib import Path

from clocks.viz import THEMES, density_figure, load_replay, show, style

# A cell runs in its page's directory: the archive is two levels up.
replay = load_replay(Path("../../assets/replays/demo_density.npz"))
for theme in THEMES:
    with style(theme) as colors:
        fig = density_figure(replay, colors)
        show(fig)
```
````

`git rm assets/demo_density.png site/assets/demo_density.png`.

`docs/2026-08-16-development-calibration.md`: after the hash table, a paragraph whose first sentence is "`demo_density.png` was removed from both trees on <today's date, YYYY-MM-DD>; the density figure is now drawn when the site is built, from `assets/replays/demo_density.npz`, and its old bytes remain in git history under the hash above." The table is not edited.

`site/reproduce/reproducibility.qmd` (line numbers as on `main`):

- Lines 38 to 40 become "…regenerated from source at deploy time. The demo GIFs and the echolocation range-study PNG are the exceptions: they are committed files, described in the next section."
- Line 46 becomes `## Where the figures come from {#sec-figures}`, and lines 48 to 52 become: "The story pages keep commentary on these figures' origin to a short caveat. This section summarizes where each came from and what it does and does not establish. The [development calibration report](https://github.com/jbwhit/clocks/blob/main/docs/2026-08-16-development-calibration.md) is the record of the August 2026 release: the generating commit and the hashes of the files that release shipped, not of every figure now on this site."
- Lines 54 to 62 become two paragraphs. **Demo animations.** Today's paragraph with the density figure taken out: "Each committed GIF is the packaged default output of its `uv run demo-*` command, …", "Each animation depicts one named finite realization …", the rest unchanged. Then: "**Replay data.** The density figure on [Beyond Point Masses](../story/beyond-point-masses.qmd) is drawn when the site is built, from `assets/replays/demo_density.npz`, the recorder's archive of the demo's packaged run: 60 observations, 2,000 particles and seed 42. The archive holds the configuration, seed and observations, a summary of the cloud before any observation and after each, and a 500-particle sample for display. Tests run the demo again and compare its inputs, to within 1e-12 where they are computed, and its summaries within stated bounds; they do not require the same random path. `uv run demo-density` repeats this run and draws the same figure. The archive is one finite realization: it establishes no population recovery rate."
- Lines 64 to 66 (the density prior) are unchanged.

- [ ] **Step 4: Tests pass; full gate; render the site and run both checks.** `check_site_figures.py` prints `Figure contract OK: 9 pages, 10 figures checked.` Open the rendered `story/beyond-point-masses.html` in a browser, switch theme, and confirm the figure shows in both; confirm `site/_output/assets/` holds no `demo_density.png`.

- [ ] **Step 5: Commit and push** as `Draw the density figure from its replay; demo-density runs the recorded scenario`. The body states that the command's run changed (P5) and that the archive was not re-recorded.

---

### Task 5: `range_study_figure`, with `write_summary_figure` as its wrapper

**Files:**
- Modify: `src/clocks/_echo_study.py`, `src/clocks/viz.py`, `tests/test_site_figure_style.py`
- Test: `tests/test_echo_study.py`

**Interfaces:**
- Consumes: `summarize` (same module); `Palette`, `style`, `THEMES`; the helpers in `tests/_figure_artists.py`.
- Produces: `range_study_figure(study: Mapping[str, Any], colors: Palette) -> Figure` in `clocks._echo_study`, exported from `clocks.viz`; `write_summary_figure(study: dict, png_path: Path) -> None`, same signature, now `with style("light") as colors:` → `range_study_figure` → `savefig(png_path, dpi=200)` → close. `scripts/scan_echolocation_range.py` and the monkeypatch in `tests/test_scenarios.py` are untouched.

**The range-study figure's artists.** Two stacked axes sharing x, both log-scaled in y, gids `"position-error"` (top) and `"mass-error"`. In each: a `Line2D` `"runs"`, one point per record in record order, `linestyle="none"`, `marker="o"`, colour `ink`, alpha 0.35; a `Line2D` `"median-error"` over the ranges `summarize` returns, `marker="o"`, colour `ink`; a `Line2D` `"median-spread"`, `marker="s"`, solid, colour `belief`; and an `axhline` `"limit"` at the study's tolerance (`position_error_max`, `mass_error_max`), dashed, colour `threshold`, labelled `f"limit {tolerance}"` in `threshold` (so `limit 1.0`, `limit 0.04`). In the top axes only: the direct labels `"median error"` (`ink`) and `"median posterior s.d."` (`belief`), one each in the whole figure; one `Text` per range with gid `"pass-count"`, x at the range in data coordinates and y just above the axes, reading the count of runs at that range whose position error and mass error are both within the tolerances; and one `Text` `"runs within both limits"` (gid `"pass-count-label"`). Y labels "position error" and "mass error", x label "range (circumradii)" on the bottom axes, no title on either axes and no figure title. `figsize=(7.5, 5.4)`.

- [ ] **Step 1: Write the failing tests** in `tests/test_echo_study.py`, on a literal study (a plain dict; it does not go through `save_study`):

```python
TOLERANCES = {"position_error_max": 1.0, "mass_error_max": 0.04, "far_std_factor_min": 20.0}


def _run(range_r, position_error, mass_error, pos_std, mass_std, passed):
    return {"range_r": range_r, "position_error": position_error,
            "mass_error": mass_error, "pos_std": pos_std, "mass_std": mass_std,
            "passed": passed}


def literal_study() -> dict:
    return {"tolerances": dict(TOLERANCES), "ranges": [2.0, 4.0], "results": [
        _run(2.0, 0.05, 0.002, 0.04, 0.002, True),
        _run(2.0, 0.10, 0.050, 0.06, 0.003, False),  # inside position, outside mass
        _run(2.0, 0.20, 0.004, 0.08, 0.004, True),
        _run(4.0, 1.50, 0.010, 1.20, 0.020, False),  # outside position
        _run(4.0, 0.80, 0.020, 0.90, 0.030, True),
        _run(4.0, 2.50, 0.030, 1.60, 0.025, False),
    ]}
```

By hand: medians at 2.0 and 4.0 are position error 0.10 and 1.50, mass error 0.004 and 0.020, position s.d. 0.06 and 1.20, mass s.d. 0.003 and 0.025; runs within both limits 2 and 1 (the position limit alone gives 3 and 1, the mass limit alone 2 and 3).

- `test_range_study_figure_size_and_axes` (each theme): size `(7.5, 5.4)`; gids `["position-error", "mass-error"]`; both y scales `"log"`; the two axes share x; both titles `""` and `fig._suptitle is None`; labels as above.
- `test_runs_are_faint_ink_dots` (each theme): in each axes, `"runs"` has x data `[2.0, 2.0, 2.0, 4.0, 4.0, 4.0]` and y data the six errors of that panel in record order; marker `"o"`, line style `"None"`, colour `colors.ink`, alpha below 1.
- `test_medians_and_spreads_come_from_summarize` (each theme): `"median-error"` and `"median-spread"` have x data `[2.0, 4.0]` and y data `[0.10, 1.50]`, `[0.06, 1.20]` (top) and `[0.004, 0.020]`, `[0.003, 0.025]` (bottom), each also equal to the matching `summarize` field; colours `ink` and `belief`; markers `"o"` and `"s"`; the spread line's style `"-"`.
- `test_limits_are_the_studys_tolerances` (each theme): `"limit"` y data `[1.0, 1.0]` and `[0.04, 0.04]`, style `"--"`, colour `colors.threshold`; texts `"limit 1.0"` and `"limit 0.04"` exist.
- `test_limits_follow_the_study`: with `position_error_max` 2.0, the top `"limit"` is at `[2.0, 2.0]`, `"limit 2.0"` exists, and the pass counts read `["2", "2"]`.
- `test_pass_counts_use_both_limits`: the `"pass-count"` texts, as `(x, text)` pairs in x order, are `[(2.0, "2"), (4.0, "1")]`; exactly one text in the figure reads `"runs within both limits"`.
- `test_the_lines_are_labelled_once`: exactly one text in the figure reads `"median posterior s.d."`, and exactly one `"median error"`.
- `test_changing_one_run_moves_its_artists`: with the first record's position error 0.5, the top `"runs"` y data start with 0.5 and the top `"median-error"` y data are `[0.20, 1.50]`; the bottom axes' `"runs"`, `"median-error"` and `"median-spread"` data are `np.array_equal` to the original's.
- `test_range_study_on_the_committed_study` (P4): from `load_study(ROOT / "assets/echolocation_range_study.json")`, the `"pass-count"` texts are `["12", "12", "11", "8", "3", "0"]` at x `[2.0, 2.6, 3.5, 4.6, 6.1, 8.0]`; the limits are at 1.0 and 0.04; the `"median-error"` y data equal `summarize`'s `med_position_error` and `med_mass_error`, exactly.
- `test_every_visible_colour_is_in_the_palette` (each theme), on the committed study: `visible_colours(fig) <= palette_hexes(colors)`.
- `test_write_summary_figure_is_the_light_range_study`: written from the literal study, `png_size(path) == (1500, 1080)`, and the pixel at (0, 0) of `matplotlib.image.imread(path)` equals `to_rgb(palette("light").background)` within `1 / 255`.
- `test_write_summary_figure_creates_png` (existing, fake records with `mass_error=0.0`) still passes unchanged.
- `test_viz_exports_the_range_study_figure`: `clocks.viz.range_study_figure is clocks._echo_study.range_study_figure`, and the name is in `__all__`.

In `tests/test_site_figure_style.py`: `FIGURE_FUNCTION_HEIGHTS["range_study_figure"] = 540`, with its drawer reading the committed study; `LIBRARY_DRAWING_MODULES` gains `"src/clocks/_echo_study.py"`; remove the `xfail` mark from `test_every_figure_builder_module_is_scanned`.

- [ ] **Step 2: Run** `uv run pytest tests/test_echo_study.py tests/test_site_figure_style.py -q`; the new tests fail on the missing name, and the library colour check fails on today's `write_summary_figure` (`"o"`, `color="steelblue"`).

- [ ] **Step 3: Implement** as "The range-study figure's artists" says. Ranges, medians and spreads come from `summarize(study["results"])`; the counts are computed per range from the records and `study["tolerances"]`, never from `passed`. Keyword styling only. In `clocks.viz`, import and export `range_study_figure`.

- [ ] **Step 4: Tests pass; full gate; commit and push** as `Draw the echolocation range study in the figure style; keep write_summary_figure as a wrapper`.

---

### Task 6: The range-study figure replaces its PNG

**Files:**
- Modify: `site/story/gravitational-echolocation.qmd`, `site/reproduce/reproducibility.qmd`, `docs/2026-08-16-development-calibration.md`, `scripts/check_site_figures.py`
- Delete: `assets/echolocation_range_study.png`, `site/assets/echolocation_range_study.png`
- Test: `tests/test_generated_assets.py`, `tests/test_site_figure_style.py`

**Interfaces:**
- Consumes: `range_study_figure` (Task 5); `RETIRED_VISUALS` (Task 4); `RETIRED_ASSETS` (Task 2).

- [ ] **Step 1: Write the failing tests.**

- `tests/test_generated_assets.py`: remove the `"echolocation_range_study.png"` row from `EXPECTED_VISUAL_HASHES`; add `"echolocation_range_study.png": "6d62a1c88c78837299bb434e66574d6930eef24c6cf9805e42ec66e141dd86bc"` to `RETIRED_VISUALS`. New `test_range_study_provenance_names_its_source`: the normalised Reproducibility page contains `"The figure is drawn when the site is built, from the certified block-400 records in `assets/echolocation_range_study.json`. The frozen cell passed "` and does not contain `"range-study PNG"`. The existing pinned sentence beginning "The frozen cell passed" is unchanged and still asserted.
- `tests/test_site_figure_style.py`: `RETIRED_ASSETS` gains `"echolocation_range_study.png"`.
- `scripts/check_site_figures.py`: `"story/gravitational-echolocation.html": (Fig("plot", 420), Fig("plot", 540))`.

- [ ] **Step 2: Run** `uv run pytest tests/test_generated_assets.py tests/test_site_figure_style.py tests/test_check_site_figures.py -q`; the retired-asset, provenance and manifest tests fail.

- [ ] **Step 3: Implement.**

`site/story/gravitational-echolocation.qmd`: replace the Markdown image on line 116 with this cell, in the same place. The prose around it, including the pinned pass-count sentence, is unchanged.

````markdown
```{python}
#| code-fold: true
#| renderings: [light, dark]
#| fig-cap:
#|   - "Resolution against range, twelve runs at each of six ranges. Top: position error; bottom: mass error. Dots are individual runs and circles their median; squares are the filter's own uncertainty, the median posterior standard deviation; dashed lines are the pass limits. The counts along the top are the runs within both limits."
#|   - "Resolution against range, twelve runs at each of six ranges. Top: position error; bottom: mass error. Dots are individual runs and circles their median; squares are the filter's own uncertainty, the median posterior standard deviation; dashed lines are the pass limits. The counts along the top are the runs within both limits."
import json
from pathlib import Path

from clocks.viz import THEMES, range_study_figure, show, style

# A cell runs in its page's directory: the study is two levels up.
study = json.loads(Path("../../assets/echolocation_range_study.json").read_text())
for theme in THEMES:
    with style(theme) as colors:
        fig = range_study_figure(study, colors)
        show(fig)
```
````

`git rm assets/echolocation_range_study.png site/assets/echolocation_range_study.png`.

`docs/2026-08-16-development-calibration.md`: add to Task 4's paragraph "`echolocation_range_study.png` was removed from both trees on <today's date>; the range-study figure is now drawn when the site is built, from `echolocation_range_study.json`, whose row above is unchanged."

`site/reproduce/reproducibility.qmd`: the CI bullet's last sentence becomes "The demo GIFs are the exceptions: they are committed files, described in the next section."; the sentence "The displayed study was regenerated directly from the certified block-400 records." becomes "The figure is drawn when the site is built, from the certified block-400 records in `assets/echolocation_range_study.json`."

- [ ] **Step 4: Tests pass; full gate; render the site and run both checks.** `check_site_figures.py` prints `Figure contract OK: 9 pages, 11 figures checked.` Confirm in the rendered `story/gravitational-echolocation.html` that the counts along the top read 12, 12, 11, 8, 3, 0, in both themes.

- [ ] **Step 5: Commit and push** as `Draw the range-study figure at build time and retire its PNG`.
