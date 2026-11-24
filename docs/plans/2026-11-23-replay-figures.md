# The Two Static Figures (Pull Request 2): Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Draw the density figure and the echolocation range-study figure from committed data when the site is built, in both themes; retire their two committed PNGs; switch `demo-density` to the recorded run; and teach the rendered-site check and the source guard that a page can hold several plots.

**Architecture:** `density_figure(replay, colors)` lives in a new private module `src/clocks/_figures.py`, beside two small trace helpers that pull request 3's clips will reuse. `range_study_figure(study, colors)` lives in `src/clocks/_echo_study.py` next to `summarize`, and `write_summary_figure` becomes a thin wrapper over it. Each returns a Matplotlib figure drawn with the shared palette; an executed page cell calls it once per theme, and the demo command and the scan script call it once for a PNG. `scripts/check_site_figures.py` gains an ordered per-page manifest of figures, and `tests/test_site_figure_style.py` checks the page source against the same manifest. Small follow-ups owed from pull request 1's review ride along in two tasks of their own.

**Tech Stack:** Python 3.12+, NumPy, Matplotlib 3.10, pytest, uv, ruff, Quarto (1.11.9 on the runner), GitHub Actions.

**Status:** Draft, not yet reviewed.

**Review:** Codex xhigh round 1 at `0bdb978`: NOT YET; revised in `18d534f`. Codex xhigh round 2 at `7c3894b`: NOT YET; revised in `cff34e2`. Codex xhigh round 3 at `6641ca4`: NOT YET (no Blocking); revised in `30e6d79`. Codex xhigh round 4 at `e10be28`: NOT YET (one Important); Gemini (agy-review) at `e10be28`: SOUND ENOUGH TO IMPLEMENT; revised in `c5af253`.

**Spec:** `docs/plans/2026-11-19-replay-and-animation-design.md`: section 6, the parts of section 9 named below, section 10, item 2 of section 11, the "Figures" and "Site" parts of "Testing", the mutation rows for figures, the rendered-site check and the source guard, and "Checks by eye". Read them before any task. Sections 1 to 3 describe the archive the density figure reads. Where this plan and the spec disagree, the spec wins and the disagreement is a plan bug to report, except for the departures listed under "Decisions this plan makes", each of which says why. The plan for pull request 1, `docs/plans/2026-11-19-replay-data.md`, shows the conventions this one follows.

**Terms.** A *theme wrapper* is an element of the rendered HTML whose class list holds `light-content` or `dark-content`; Quarto emits one per rendering of a cell, and the site's stylesheet shows only the current theme's. A *plot cell* is an executed Python cell on a page that draws a figure. The *manifest* is the per-page list of figures, in document order, that the rendered-site check and the source guard both read. A *trace* is a panel that plots one parameter's posterior mean against the number of observations seen. *Retired* files are committed media this track removes.

## Global Constraints

- Python only through uv: `uv run pytest`, `uv run ruff format .`, `uv run ruff check .`, `uv run python ...`. Never bare `python` or `pip`.
- **Every task runs the same cycle**, and its steps say only what is particular to it: write the tests of Step 1; run them and see them fail for the stated reason; implement; run them and see them pass; run the full local gate (`uv run ruff format --check .`, `uv run ruff check .`, `uv run pytest -q`, `uv run pytest -q -m slow`) and read its output; then, in a separate command, commit with the task's message and `git push origin claude-replay-figures`. Do not open a pull request (the coordinator does). Begin every git step with an explicit `cd` to the worktree.
- **Every task that changes a page also renders the site and runs both checks**, after the gate and before the commit: `cd site && uv run --frozen quarto render && cd ..`, then `uv run --frozen python scripts/check_site_links.py` and `uv run --frozen python scripts/check_site_figures.py`. The runner uses Quarto 1.11.9 (`.github/workflows/site-check.yml`); state the local `quarto --version` in the commit body.
- Commit messages are passed with a quoted heredoc (`<<'MSG'`) and end with a `Co-Authored-By:` trailer naming the model that wrote the commit.
- Implementers of Tasks 1 to 9 never use `git checkout`, `git restore` or `git stash` to undo anything: a wrong edit is undone by editing, and a deleted file that must come back is reported to the coordinator. Task 10's mutants are run by the coordinator's harness in a separate worktree.
- **A red step fails on behaviour, not on a missing name.** Where a task's tests import names that do not exist yet, its Step 2 first adds those names with the signatures of its Interfaces block and bodies that do nothing useful (as each task says), so the run shows the assertions failing.
- **The repository is public.** No committed file may contain an absolute local path, a person's name or an email address. This includes the contact-sheet script and anything printed into a commit body.
- The archives are not re-recorded. At the end, `git diff --stat main -- assets/replays` is empty. Under `assets/` and `site/assets/` the only changes are the four deletions of Tasks 4 and 6.
- No library drawing code contains a literal colour: every colour is a field of the `Palette` passed in. Role colours: density, the true profile `truth`, the median profile and its band `belief`, clocks `ink`; traces, the mean and its band `belief`, the truth `truth`; range study, each run and the median error `ink`, the median posterior standard deviation `belief`, the pass limits `threshold`, always dashed.
- Both figures are 750 by 540 CSS pixels: `figsize=(7.5, 5.4)` at the style's 100 dpi, which `show` renders at twice that. **Both static-figure PNGs (the one `demo-density` writes and the one `write_summary_figure` writes) go through one function, `save_png(fig, path)` in `clocks._style`** (Task 4); the archive, study JSON and animation writers are unchanged. It sits beside `show`, with the same crop and opacity controls: `dpi=200`, `format="png"`, `bbox_inches=None` inside `matplotlib.rc_context({"savefig.bbox": None})`, `transparent=False`, `facecolor=fig.get_facecolor()`. It writes 1500 by 1080 opaque pixels, the pixels the site shows, as PNG bytes at exactly the path given, whatever its suffix and whatever the caller's `savefig.*` settings (P15: a plain `savefig` under `savefig.bbox="tight"` and `savefig.transparent=True` wrote 1523 by 1103 transparent pixels; P16: without `facecolor=fig.get_facecolor()`, a caller's `savefig.facecolor="magenta"` paints the background). For `demo-density` this is a small change: today an extensionless `--output out/d` writes `out/d.png` (Matplotlib appends the suffix it infers; P12), and from Task 4 it writes `out/d`. No caller of `write_summary_figure` depends on suffix inference: the scan script passes `output/echolocation_range_study.png` and the tests `study.png` (read). `save_png` is not exported from `clocks.viz`: the page cells use `show`, and the contact-sheet script imports it from `clocks._style`. Pull request 3's demo commands reuse it.
- A figure function draws with the rcParams in force, so callers call it inside `with style(theme) as colors:` and pass that `colors`. It never calls `plt.show`, never saves, and never closes the figure it returns.
- **Renderers never call the clock-rate forward model** (spec, section 1). `_figures.py` evaluates the density parameterisation A·exp(−(x−μ)²/2σ²) itself.
- Out of scope: clips, video, the encoder, `_video.py`, the pre-render script, every demo command other than `demo-density`, the 2D truth change, the version bump, removing `animate_*`, `plot_*` or `create_*_dashboard` names, the README GIF and its paragraph, the Architecture page's module list, and the "run exactly once" protocol sentences. Pull request 3 does all of these.
- `demo-density` keeps `--output PATH` (default `output/demo_density.png`), `--observations` and `--particles`. After Task 4 it writes exactly one file, PNG bytes at exactly `--output`: the density figure of the run `record(n_observations=..., n_particles=...)` returns, light theme, 1500 by 1080 pixels. It writes no archive; `--output-dir`, `--theme` and the `.npz` output are pull request 3's.
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
- **A plot cell is recognised by what it calls from the façade.** Section 10 names "a `clocks.viz` function named `*_figure`". The guard resolves builders **from imports in the same cell**, which is how every page cell is written: `from clocks.viz import <name>` (with any `as` alias), `import clocks.viz as <alias>` then `<alias>.<name>(...)`, and `import clocks.viz` then `clocks.viz.<name>(...)`. An unrelated `helper_figure()` or `other.helper_figure()` is not a plot producer. The same resolution gives the height lookup. Inside one `with style(<loop variable>)` block in the body (not the `else`) of `for <loop variable> in THEMES`, a builder call must be the whole right-hand side of an assignment to a plain name, `<name> = <builder>(...)`, and the very next statement in that block must be exactly `show(<name>)`. This canonical form is required by construction, rather than by tracking which figure a name holds.
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
- **P11. Warnings.** A scratch figure built like Task 3's (gridspec 2 by 3, the profile, band, clocks and a label; three traces with Greek-letter headers and explicit limits) from the literal fixture of Task 3, full and cut to one observation, created, drawn (`fig.canvas.draw()`) and saved to memory at `dpi=200` inside `warnings.simplefilter("error")` and `style(theme)`: no warning in either theme. The bundled Source Sans 3 has the μ, σ and ± glyphs.
- **P12. Matplotlib 3.10.8 behaviour.** `matplotlib.image.imread` of a saved PNG returns shape (240, 320, 4): RGBA. `fig.savefig(dir / "noext")` writes `noext.png`; `fig.savefig(dir / "forced", format="png")` writes `forced`, starting with the PNG signature `b"\x89PNG\r\n\x1a\n"`. `Figure.get_suptitle()` exists and returns `""` with no title.
- **P13. Mutant values, recomputed.** With row 0 included, `trace_limits` gives (−1.376, 8.0) for `clipped_to_prior`, (−3.24, 3.24) for `zero_width` and (−4.968, 4.968) for `prior_row_left_out`. On Task 5's literal study (round 2's version, with one disagreeing flag): at the default tolerances, both limits 2 and 1, position alone 3 and 1, mass alone 2 and 3, strict `<` 2 and 0, flags 1 and 1; at `position_error_max` 2.0 and `mass_error_max` 0.035, both 2 and 1, position alone 3 and 2, mass alone 2 and 2, strict 2 and 1, flags 1 and 1, a mass limit fixed at 0.04 2 and 2, a position limit fixed at 1.0 2 and 0.
- **P14. Two shell and helper probes.** Today's `_assert_predictions_match_dots` on the mixed row of Task 8 raises `row 0, clock 0: the dots' mean reading is nan s.d./sqrt(m) from predicted_mean`. Task 9's revised summary line, run with `bash -e`, a `tail` that exits 1 first on `PATH` and a non-empty report, prints the warning and creates no summary file.
- **P15. Saving, entities, JPEG metadata, archives.** A 7.5 by 5.4 figure made in `style("light")`, saved under `matplotlib.rc_context({"savefig.bbox": "tight", "savefig.transparent": True})`: plain `savefig(path, dpi=200, format="png")` wrote 1523 by 1103 pixels with a transparent white corner; with `bbox_inches=None` inside `rc_context({"savefig.bbox": None})`, `transparent=False` and `facecolor=fig.get_facecolor()` it wrote 1500 by 1080, opaque, corner (249, 245, 215), the light background. `urlsplit("../assets/demo_density&#46;png").path` is `../assets/demo_density&`; after `html.unescape` and `unquote` it is `../assets/demo_density.png`. Pillow 12.1.1, `Image.new("RGB", ...).save(path, quality=85)` as JPEG: `Image.open(path).info` has exactly the keys `jfif`, `jfif_density`, `jfif_unit`, `jfif_version`, and `getexif()` is empty. The committed `demo_1d`, `demo_multi_mass` and echolocation archives have no `predicted_std` entry equal to 0.
- **P16. Round-3 probes.** Under `rc_context({"savefig.bbox": "tight", "savefig.transparent": True, "savefig.facecolor": "magenta"})`, a light-style figure saved with `show`'s controls wrote 1500 by 1080 with corner (249, 245, 215, 255); without only `facecolor=fig.get_facecolor()` it wrote 1500 by 1080 with corner (255, 0, 255, 255). A plain Pillow JPEG save has `applist == [("APP0", b"JFIF\x00…")]`, one segment. Today's `_assert_predictions_match_dots` under `warnings.simplefilter("error")` on a one-particle `demo_1d` recording raises `RuntimeWarning: divide by zero encountered in divide`; without the filter, the off-dot case reports `… is inf s.d./sqrt(m) …` and the spread case `… is nan s.d./sqrt(m) …`. Pass counts on Task 5's literal study at (position, mass) limits (2.0, 0.04): both 2 and 2, position alone 3 and 2, strict 2 and 1, flags 1 and 1, default tolerances 2 and 1; at (1.0, 0.003): both 1 and 0, position alone 3 and 1, strict 1 and 0, flags 1 and 1, default 2 and 1, mass limit fixed at 0.04 2 and 1.

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
- `test_identical_light_and_dark_content`: `site = _build(tmp_path)`, then `(tmp_path / "story/demo_files/dark.png").write_bytes(b"light.png")` (the file the page's dark image names; `_build` returns `tmp_path` and writes under `story/demo_files/`); one failure containing `figure 1: light and dark images have the same content`.
- `test_an_encoded_nul_is_a_failure`: `_build(tmp_path, dark=[("demo_files/dark%00.png", 750)])`; one failure containing `image path contains a NUL`. (Today this raises `ValueError`: P6.)
- `test_a_wrapper_on_an_unlisted_page`: `_build(tmp_path)`, plus `story/other.html` holding one light wrapper from `_pair(1, 420)`; `figure_failures` returns exactly `["story/other.html: not in the figure manifest, but has 1 theme wrapper(s)"]`.
- `test_a_listed_page_with_no_figures_passes`: manifest `{PAGE: ()}` and a page with no wrappers gives `[]`.
- `test_a_listed_page_with_no_figures_must_have_no_wrappers`: manifest `{PAGE: ()}` and a page holding `_pair(1, 420)`; the one failure is `f"{PAGE}: expected 0 figure(s), 0 theme wrappers; found 2"`. (A checker that skipped empty entries would pass the previous test and fail this one.)

- [ ] **Step 2: Stage the names, then run.** Add `Fig` and `EXPECTED_FIGURES` (the value above) to the script and point `figure_failures` at it, each page checked with today's one-wrapper-per-theme logic at its first `Fig`'s height, and a page whose tuple is empty skipped. Run `uv run pytest tests/test_check_site_figures.py -q`: the two-plot, order, count, content, unlisted-page and `must_have_no_wrappers` tests fail on their assertions (`test_a_listed_page_with_no_figures_passes` passes, the shim skipping the page), the relabelled existing tests fail on the old labels, and `test_an_encoded_nul_is_a_failure` errors with the `ValueError` of P6.

- [ ] **Step 3: Implement.**

- `figure_failures` checks each page in `EXPECTED_FIGURES` as today (a missing page is still `f"{page}: page is missing from {output_dir}"`), then every `*.html` under `output_dir` outside `site_libs/` whose relative path is not a key: any theme wrapper there is the unlisted-page failure above.
- `_page_failures(page, path, figures, output_dir)` keeps the both-classes and nesting checks first. Then, with `n = len(figures)`, a wrapper count other than `2 * n` is the single count failure and nothing more is checked on that page. Otherwise wrappers are taken in document order in pairs: pair `i` must be light then dark, else the order failure for that pair and no further checks on it. A pair whose `Fig.kind` is not `"plot"` raises `ValueError(f"{page}: figure {i}: no check for kind {kind!r}")`; pull request 3 replaces that line with the clip checks.
- Per plot pair, today's checks with the new label, plus: after both files resolve and are different paths, `light.read_bytes() == dark.read_bytes()` is the content failure.
- `_resolve_source`: after `unquote`, a path containing `"\x00"` returns `(None, f"image path contains a NUL: {src!r}")`, before anything touches the file system.
- `main` prints `f"Figure contract OK: {len(EXPECTED_FIGURES)} pages, {n} figures checked."` with `n` the total number of `Fig` entries. Update the module docstring: several plots per page, in manifest order, and unlisted pages.

- [ ] **Step 4: Tests pass; full gate; render the site and run both checks** (the checker changed): `check_site_figures.py` prints `Figure contract OK: 9 pages, 9 figures checked.`, and with the six unlisted pages scanned, nothing else.

- [ ] **Step 5: Commit and push** as `Check each page's plots against an ordered manifest; report an encoded NUL`.

---

### Task 2: The source guard learns several plot cells, `data=` by position and retired assets

**Files:**
- Modify: `tests/test_site_figure_style.py`, `tests/test_check_site_figures.py`

**Interfaces:**
- Consumes: `Fig`, `EXPECTED_FIGURES` from Task 1, imported by path as `tests/test_check_site_figures.py` does.
- Produces, in `tests/test_site_figure_style.py`:
  - `_facade_builders(cell: str) -> list[str]`: the `clocks.viz` names ending in `_figure` that the cell calls, as their names in `clocks.viz` and in call order, resolved from imports in the same cell. A call counts when it names something the cell imported with `from clocks.viz import <name>` (an `as` alias resolves to `<name>`), is `<alias>.<name>(...)` after `import clocks.viz as <alias>`, or is `clocks.viz.<name>(...)` after `import clocks.viz`. Any other `*_figure` call does not count.
  - `_style_block_violations(cell: str) -> list[str]`: for a cell with façade builders. A *style block* is a `with` statement whose item is `style(<v>)` and which is a statement of the `body` (not the `orelse`) of a `for <v> in THEMES` loop, `<v>` being that loop's target name. One message per builder call, and per `show` call, not inside a style block; and, for each builder call that is inside one, one message unless its statement is `<name> = <builder>(...)` (an `Assign` with one plain `Name` target whose value is the call) and the next statement of the same block is exactly the expression `show(<name>)`. A statement in between, `show` of another name, `show(<builder>(...))`, or a builder call that is not such an assignment is a violation. Messages: `f"{name}() is not inside a style(theme) block of a THEMES loop"`, `"show() is not inside a style(theme) block of a THEMES loop"`, `f"{name}() is not `<name> = {name}(...)` followed immediately by `show(<name>)`"`, in that order.
  - `_is_plot_cell(cell: str) -> bool` and `_plot_cells(path: Path) -> list[str]` (all plot cells of a page, in order).
  - `FIGURE_FUNCTION_HEIGHTS: dict[str, int]`, empty here; Tasks 3 and 5 add `"density_figure": 540` and `"range_study_figure": 540`. Task 3's `test_figure_function_heights_are_the_drawn_ones` is what ties this mapping, which the source guard uses, to the figures' real sizes.
  - `_cell_height(cell: str) -> int`: `FIGURE_FUNCTION_HEIGHTS[name]` for the first name `_facade_builders` returns, if any; else 540 if the cell contains `figsize=(7.5, 5.4)`; else 420.
  - `RETIRED_ASSETS: tuple[str, ...] = ()`; Tasks 4 and 6 add `"demo_density.png"` and `"echolocation_range_study.png"`. `_retired_asset_links(text: str, retired: Sequence[str]) -> list[str]` returns each link or embed destination in `text` that names a retired file under an `assets/` directory, as written (HTML attribute values after `html.unescape`, Markdown destinations without angle brackets).
  - `LIBRARY_DRAWING_MODULES: tuple[str, ...] = ()`, paths from the repository root; Tasks 3, 4 and 5 add `src/clocks/_figures.py`, `src/clocks/_demos/demo_density.py`, `src/clocks/_echo_study.py`.

- [ ] **Step 1: Write the failing tests.**

Plot cells:

- `test_a_cell_is_a_plot_cell`, parametrized with IDs: `renderings` `"#| renderings: [light, dark]\nx = 1"` → True; `subplots` `"fig, ax = plt.subplots()"` → True; `figure` `"fig = plt.figure()"` → True; `show` `"show(fig)"` → True; `figure_function` `"from clocks.viz import density_figure\nfig = density_figure(replay, colors)"` → True; `aliased` `"from clocks.viz import density_figure as draw\nfig = draw(replay, colors)"` → True; `viz_module` `"import clocks.viz as viz\nfig = viz.range_study_figure(study, colors)"` → True; `qualified` `"import clocks.viz\nfig = clocks.viz.density_figure(replay, colors)"` → True; `helper_figure` `"def helper_figure():\n    return 1\n\nhelper_figure()"` → False; `attribute_helper` `"import other\nfig = other.helper_figure()"` → False; `unimported_figure` `"fig = density_figure(replay, colors)"` → False; `not_a_plot` `"table = Markdown(text)\nprint(table)"` → False; `a_method_named_figure` `"ax.figure.canvas.draw()"` → False.
- `test_facade_builders_are_resolved`: `_facade_builders("from clocks.viz import density_figure as draw\nimport clocks.viz as viz\nimport clocks.viz\ndraw(r, c)\nviz.range_study_figure(s, c)\nclocks.viz.density_figure(r, c)\nhelper_figure()\nother.helper_figure()")` is `["density_figure", "range_study_figure", "density_figure"]`.
- `test_style_block_violations`, synthetic cells, each starting `from clocks.viz import THEMES, density_figure, show, style`, IDs and expected lists:
  - `canonical`, the page form (`for theme in THEMES:` / `with style(theme) as colors:` / `fig = density_figure(replay, colors)` / `show(fig)`): `[]`;
  - `fixed_theme`, the same with `style("light")`: `["density_figure() is not inside a style(theme) block of a THEMES loop", "show() is not inside a style(theme) block of a THEMES loop"]`;
  - `other_variable`, the same with `style(other_theme)`, `other_theme` assigned earlier in the cell: the same two messages;
  - `loop_else`, `for theme in THEMES:` / `pass` / `else:` / the canonical `with` block: the same two messages (a `with` in the loop's `else` runs once);
  - `comment_only`, a cell whose comment reads `# for theme in THEMES: with style(theme) as colors: show(fig)` and whose code is `fig = density_figure(replay, colors)` then `show(fig)` at top level: the same two messages;
  - `show_outside`, `show(fig)` dedented to the loop body after the `with`: `["show() is not inside a style(theme) block of a THEMES loop", "density_figure() is not `<name> = density_figure(...)` followed immediately by `show(<name>)`"]`;
  - `builder_outside`, `fig = density_figure(replay, colors)` before the `with`, `show(fig)` inside: `["density_figure() is not inside a style(theme) block of a THEMES loop"]`;
  - `two_blocks`, builder in one `with style(theme)` block and `show(fig)` in a second one in the same loop: `["density_figure() is not `<name> = density_figure(...)` followed immediately by `show(<name>)`"]`;
  - `commented_out_show`, the canonical cell with `show(fig)` replaced by the comment `# show(fig)`: `["density_figure() is not `<name> = density_figure(...)` followed immediately by `show(<name>)`"]`;
  - `show_of_another_figure`, the canonical cell with `other = plt.figure()` added to the block and `show(fig)` replaced by `show(other)`: `["density_figure() is not `<name> = density_figure(...)` followed immediately by `show(<name>)`"]`;
  - `reassigned_before_show`, the canonical cell with `fig = plt.figure(figsize=(7.5, 5.4))` inserted between `fig = density_figure(replay, colors)` and `show(fig)`: `["density_figure() is not `<name> = density_figure(...)` followed immediately by `show(<name>)`"]` (it would display the replacement figure);
  - `statement_between`, the canonical cell with `print(theme)` inserted between the two lines: `["density_figure() is not `<name> = density_figure(...)` followed immediately by `show(<name>)`"]` (harmless, and still rejected: the rule is "immediately");
  - `show_of_the_call`, the block holding only `show(density_figure(replay, colors))`: `["density_figure() is not `<name> = density_figure(...)` followed immediately by `show(<name>)`"]`.
- `test_builder_calls_and_show_are_inside_the_style_block` (each plot cell for which `_facade_builders` is not empty; none until Task 4): `_style_block_violations(cell) == []`. The two page cells of Tasks 4 and 6 are already in the canonical form (`fig = density_figure(replay, colors)` then `show(fig)`; `fig = range_study_figure(study, colors)` then `show(fig)`). The existing text checks (`for theme in THEMES:`, `with style(theme) as`, `show(fig)`) stay.
- The three cell-contract tests and `test_cell_has_no_unstyled_colour_or_display` are parametrized over every plot cell, not a page's only one: `PLOT_CELLS = [pytest.param(page, index, id=f"{page}#{index + 1}") for page in PLOT_PAGES for index in range(len(_plot_cells(SITE / page)))]`. `_only_plot_cell` is deleted.
- `test_manifest_matches_the_source` (each of `PLOT_PAGES`): `tuple((fig.kind, fig.height) for fig in EXPECTED_FIGURES[page.replace(".qmd", ".html")])` equals `tuple(("plot", _cell_height(cell)) for cell in _plot_cells(SITE / page))`.
- `test_the_manifest_lists_exactly_the_plot_pages`: the manifest's key set equals the nine `PLOT_PAGES` as `.html`, and `len(PLOT_PAGES) == 9`. Remove `test_the_manifest_lists_the_nine_plot_pages` from `tests/test_check_site_figures.py`, and that file's import of `test_site_figure_style` if nothing else uses it.

`data=` by position: add to `test_style_violations_reports_unstyled_calls` the case `('ax.plot(x, "rate", "r-", data=frame)', "positional format string")`, with the ID `data_third_argument` (give the list's cases IDs as `pytest.param`s if that is needed to name one). The existing allowed case `ax.plot("step", "rate", data=frame, color=colors.ink)` stays allowed.

Retired assets, against `retired = ("demo_density.png",)`:

- `test_retired_asset_links_are_found`, IDs and inputs, each giving a one-element list holding the target as written: `md_image` `"![a](../assets/demo_density.png)"`; `md_link` `"[a](assets/demo_density.png)"`; `reference` `"[a]: ../site/assets/demo_density.png"` (a line of its own); `html_img` `'<img src="../assets/demo_density.png" alt="">'`; `github_url` `"[a](https://github.com/example/clocks/blob/main/assets/demo_density.png)"`; `with_query` `"![a](../assets/demo_density.png?v=2)"`; `encoded_basename` `"![a](../assets/%64emo_density.png)"`; `angle_brackets` `"![a](<../assets/demo_density.png>)"`, which gives `["../assets/demo_density.png"]` (a destination is reported as written, without its angle brackets); `html_entity` `'<img src="../assets/demo_density&#46;png" alt="">'`, which gives `["../assets/demo_density.png"]` (a browser decodes the entity; `urlsplit` on the raw value would read `#46;png` as a fragment: P15); `href_entity` `'<a href="../assets/demo_density&#46;png">a</a>'`, which gives the same.
- `test_names_that_are_not_links_are_allowed`, IDs `command_output` `"uv run demo-density             # → output/demo_density.png"`, `inline_code` `` "`assets/demo_density.png`" ``, `output_link` `"[a](output/demo_density.png)"`, `other_asset` `"![a](../assets/demo_2d.gif)"`: each gives `[]`.
- `test_no_page_links_to_a_retired_asset` (each of `README.md` and every `.qmd` under `site/` outside `_output` and `.quarto`, ID the path relative to the repository root, such as `site/story/beyond-point-masses.qmd`): `_retired_asset_links(text, RETIRED_ASSETS) == []`.

Library drawing code:

- `test_library_drawing_code_has_no_unstyled_colour` (each of `LIBRARY_DRAWING_MODULES`): `_style_violations` on the module's text is `[]`. Empty here; Task 3 gives it its first case.
- `test_every_figure_builder_module_is_scanned`: every `.py` under `src/clocks/` defining a function whose name ends in `_figure` is in `LIBRARY_DRAWING_MODULES`. Here that is `src/clocks/_echo_study.py` (`write_summary_figure`), which still draws with literal colours, so this test is written now and marked `xfail(strict=True, reason="Task 5 replaces the range-study drawing")`; Task 5 removes the mark.

- [ ] **Step 2: Stage the helpers, then run.** Add `_is_plot_cell` with today's text rule (`"plt.subplots(" in cell or "plt.figure(" in cell`), `_facade_builders` and `_style_block_violations` returning `[]`, `_cell_height` without the builder lookup and `_retired_asset_links` returning `[]`. Run `uv run pytest tests/test_site_figure_style.py tests/test_check_site_figures.py -q`: `test_a_cell_is_a_plot_cell[show]`, `[renderings]`, `[figure_function]`, `[aliased]`, `[viz_module]` and `[qualified]`, `test_facade_builders_are_resolved`, the twelve negative `test_style_block_violations` cases, the `data_third_argument` case and the ten `test_retired_asset_links_are_found` cases fail on their assertions; the strict `xfail` is reported as expected.

- [ ] **Step 3: Implement.**

- `_facade_builders` walks the cell's syntax tree: `ImportFrom` nodes with module `clocks.viz` give a local-name-to-`viz`-name map for names ending in `_figure`; `Import` nodes for `clocks.viz` give the module's spellings (the `as` name, or the dotted `clocks.viz` when there is none); then each `Call` whose `func` is a mapped `Name`, or an `Attribute` ending in `_figure` whose value spells the module, is collected.
- `_style_block_violations` finds, for each builder or `show` call, its innermost enclosing `With`, and requires that `With` to be an element of an enclosing `For`'s `body` list (not `orelse`), through parent links on the syntax tree. `show` calls are found as `ast.Call` nodes, so a comment never counts. The canonical-form check reads the builder's statement and the next element of the same `With.body` list.
- `_is_plot_cell`: true when a line of the cell starts with `#| renderings:`, or its syntax tree holds a call to `plt.subplots`, `plt.figure` or a bare `show`, or `_facade_builders(cell)` is not empty.
- `_style_violations`: with `data=`, a string at any positional index from 2 on is a format string; without it, any positional string is. Update the comment.
- `_retired_asset_links`: collect destinations from Markdown links and images (`](destination`, the destination with any surrounding `<` `>` removed), reference definitions (`[label]: destination` at the start of a line) and HTML `src=`/`href=` attributes, whose values are decoded with `html.unescape` (simpler than running `html.parser` over Markdown text, and it decodes the same entities); for each, take `unquote(urlsplit(destination).path)` and report the destination when that path's last two `PurePosixPath` parts are `("assets", name)` for a retired `name`.
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
- Produces, in `tests/_figure_artists.py` (a helper module, not collected): `palette_hexes(colors: Palette) -> set[str]` (every field, `ordered` included, as lower-case `#rrggbb`); `visible_colours(fig: Figure) -> set[str]` (below); `one(container, gid: str) -> Artist` (asserts that exactly one artist under `container`, through `findobj`, carries that gid, and returns it; tests call it on the owning axes); `gid_census(fig: Figure) -> dict[str, list[str]]` (each artist gid in the figure, other than the axes' own, mapped to the gids of the axes holding an artist with it, in `fig.axes` order, one entry per artist), which catches a duplicate drawn on another axes; `band_edges(collection, x: NDArray) -> tuple[NDArray, NDArray]` (for each x, the smallest and largest y among the band's path vertices at that x); `png_size(path: Path) -> tuple[int, int]` (width and height from the IHDR bytes 16 to 24).
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

- `test_density_figure_size_and_axes` (each theme): `tuple(fig.get_size_inches()) == (7.5, 5.4)`; `[ax.get_gid() for ax in fig.axes] == ["profile", "trace-μ", "trace-σ", "trace-A"]`; and `gid_census(fig)` equals `{"true-profile": ["profile"], "median-profile": ["profile"], "profile-band": ["profile"], "clocks": ["profile"], "trace-mean": TRACES, "trace-band": TRACES, "trace-truth": TRACES}` with `TRACES = ["trace-μ", "trace-σ", "trace-A"]`: every expected artist exists once, on its own axes.
Artists are fetched with `one(axes, gid)` on the axes that owns them.

- `test_true_profile` (each theme): `"true-profile"`'s x data equal `np.linspace(-8.0, 8.0, 321)` exactly, its y data equal `0.010 * np.exp(-((x - 1.5) ** 2) / (2 * 2.0**2))` with `rtol=1e-12`, and its colour is `colors.truth`.
- `test_median_and_band_are_pointwise_percentiles_of_the_last_row` (each theme): `"median-profile"`'s x data equal `x` exactly and its y data `np.median(profiles, axis=0)`, and `band_edges(band, x)` equal `np.percentile(profiles, 16, axis=0)` and `np.percentile(profiles, 84, axis=0)`, all with `rtol=1e-12`; the line's colour and the band's face colour (alpha dropped) are `colors.belief`; the band is visible and its face colour's alpha is above 0 (the palette walk skips a transparent artist, so only this assertion catches one).
- `test_only_the_last_cloud_row_feeds_the_profile`: with `cloud[1]` replaced by `cloud[0]`, the median line and band edges are `np.array_equal` to the original's; with 1.0 added to μ in every dot of `cloud[-1]`, the median line moves by more than 1e-4 somewhere.
- `test_changing_the_truth_moves_only_the_truth`: with `true_params[0] = 0.0`, `"true-profile"` moves, `"median-profile"` does not, `"trace-truth"` in `"trace-μ"` moves to `[0.0, 0.0]` and the σ and A truth lines do not.
- `test_clocks_are_ink_squares_on_the_axis` (each theme): `"clocks"` has x data `[-6.0, -3.0, 0.0, 3.0, 6.0]`, y data all 0, marker `"s"`, line style `"None"`, colour `colors.ink`; it is visible, `get_clip_on()` is False (a square clipped at y = 0 would show half), its alpha is `None` or 1, and its marker size is above 0.
- `test_clocks_follow_the_archive`: with `clock_positions[2, 0] = 0.5`, `"clocks"` has x data `[-6.0, -3.0, 0.5, 3.0, 6.0]`.
- `test_density_labels` (each theme): the `"profile"` axes holds exactly one visible `Text` reading each of `"true profile"`, `"pointwise median and 68% band"` and `"clocks"`, coloured `truth`, `belief` and `ink`.
- `test_profile_limits`: x-limits `(-8.0, 8.0)`; y-limits `(0.0, 1.08 * max(true.max(), upper.max()))`, `rel=1e-12`, where `true` and `upper` are the test's own true profile and 84th percentile, not read from the drawn artists.
- `test_traces_follow_the_archive` (each theme): for each column j, in `"trace-<name>"`: `"trace-mean"` has x data `[0, 1, 2]` and y data `mean[:, j]`; `band_edges` are `mean[:, j] ∓ sd[:, j]` (`rtol=1e-12`); `"trace-truth"` has x data `[0, 2]`, y data `[t_j, t_j]` and line style `"-"`; colours `belief`, `belief`, `truth`; the band is visible with face alpha above 0; x-limits `(0, 2)`; y-limits, `rel=1e-9`, μ `(0.552, 1.848)`, σ `(1.76, 2.84)`, A `(0.00884, 0.01316)`.
- `test_changing_one_row_moves_only_that_point`: with `mean[1, 0] += 0.1`, `"trace-mean"` in `"trace-μ"` differs from the original's at index 1 by 0.1 (`rel=1e-12`) and nowhere else.
- `test_trace_headers`: the three titles are `"μ   true 1.50   1.45 ± 0.10"`, `"σ   true 2.00   2.10 ± 0.15"`, `"A   true 0.0100   0.0102 ± 0.0006"`.
- `test_trace_limits`, IDs and values (`pytest.approx`, `rel=1e-12`):
  - `prior_row_left_out`: `trace_limits(np.array([0.0, 1.2, 1.45]), np.array([4.6, 0.6, 0.1]), 1.5, (-8.0, 8.0)) == (0.552, 1.848)`; with row 0 included it would be (−4.968, 4.968).
  - `clipped_to_prior`: `trace_limits(np.array([0.0, 7.9]), np.array([1.0, 0.5]), 7.5, (-8.0, 8.0)) == (7.36, 8.0)`.
  - `zero_width`: `trace_limits(np.array([0.0, 1.5, 1.5]), np.array([3.0, 0.0, 0.0]), 1.5, (-8.0, 8.0)) == (0.86, 2.14)`.
- `test_density_figure_draws_a_one_observation_run`: the fixture cut to its first observation (`n_observations` 1; `mean`, `cov`, `ess`, `log_evidence`, `cloud` to two rows; the integer arrays and `observations` to one), which `validate_replay` accepts; created and drawn (`fig.canvas.draw()`) under `warnings.simplefilter("error")`, each `"trace-mean"` has x data `[0, 1]` and x-limits `(0, 1)`. (The full fixture and the cut one, drawn under the same filter, are clean here: P11.)
- `test_density_figure_refuses_another_demo`: a valid replay of another demo, `demo_1d.record(n_observations=1, n_particles=16)` (renaming the density fixture would make it invalid), raises `ValueError` matching `density_figure needs a demo_density replay, got 'demo_1d'`.
- `test_density_figure_on_the_committed_archive` (P2, P3): drawn from `load_replay(ROOT / "assets/replays/demo_density.npz")`: `"median-profile"` at indices 40, 160, 190 is `0.0000196`, `0.0074162`, `0.0095141`, and the band's lower and upper edges there are `0.0000095`, `0.0071591`, `0.0090899` and `0.0000349`, `0.0076912`, `0.0100017`, each `abs=5e-8`; the profile's top y-limit is `0.0108040` (`abs=5e-8`); the titles are `"μ   true 1.50   1.50 ± 0.08"`, `"σ   true 2.00   2.14 ± 0.12"`, `"A   true 0.0100   0.0095 ± 0.0005"`.
- `test_every_visible_colour_is_in_the_palette` (each theme), on the committed archive: `visible_colours(fig) <= palette_hexes(colors)`.
- `test_figures_module_imports_no_forward_model`: the syntax tree of `src/clocks/_figures.py` imports, from `clocks`, only `clocks._replay` and `clocks._style`.
- `test_viz_exports_the_density_figure`: `clocks.viz.density_figure is clocks._figures.density_figure`; `clocks.viz.load_replay is clocks._replay.load_replay`; both names are in `clocks.viz.__all__`.

In `tests/test_site_figure_style.py`: `FIGURE_FUNCTION_HEIGHTS["density_figure"] = 540`; `LIBRARY_DRAWING_MODULES` gains `"src/clocks/_figures.py"`; and a new `test_figure_function_heights_are_the_drawn_ones` (each name in `FIGURE_FUNCTION_HEIGHTS`) draws the committed data in the light theme through a name-to-drawer map in the test and asserts `round(fig.get_figheight() * 100) == FIGURE_FUNCTION_HEIGHTS[name]` and `round(fig.get_figwidth() * 100) == 750`.

- [ ] **Step 2: Stage the names, then run.** Create `src/clocks/_figures.py` with `PROFILE_POINTS`, `trace_limits` and `trace_header` raising `NotImplementedError`, and `density_figure` returning an empty `plt.figure(figsize=(7.5, 5.4))`; export it and `load_replay` from `clocks.viz`. Run `uv run pytest tests/test_figures.py tests/test_site_figure_style.py -q`: the artist tests fail in `one(...)` on missing gids, the helper tests on `NotImplementedError`, and the wrong-demo test because nothing is raised; `test_visible_colours_finds_a_literal_colour`, `test_figures_module_imports_no_forward_model`, `test_viz_exports_the_density_figure` and the size half of the size test already pass.

- [ ] **Step 3: Implement.**

- `density_figure` raises the `ValueError` above when `replay.meta["demo"] != "demo_density"`, then draws as "The density figure's artists" says. The profiles are computed in float64 from `cloud[-1].astype(np.float64)`; the band is `np.percentile(profiles, [16, 84], axis=0)` (NumPy's default linear method) and the median `np.median(profiles, axis=0)`. Every artist is drawn with keyword styling only (`color=`, `marker=`, `linestyle=`), since the source guard reads this module.
- `src/clocks/_figures.py` imports from `clocks` only `clocks._replay` (for `Replay`) and `clocks._style` (for `Palette`).
- `clocks.viz`: import `density_figure` from `clocks._figures` and `load_replay` from `clocks._replay`; add both to `__all__`; name `_figures` in the module docstring.

- [ ] **Step 4: Tests pass; full gate; commit and push** as `Draw the density figure from a replay, with shared trace helpers`.

---

### Task 4: `demo-density` draws the recorded run, and the density figure replaces its PNG

One change, as section 11 requires: the command's switch to `record()` and the new figure, the page cell, the removal of the PNG and its hash row, and the provenance sentences that describe them.

**Files:**
- Modify: `src/clocks/_style.py`, `src/clocks/_demos/demo_density.py`, `site/story/beyond-point-masses.qmd`, `site/reproduce/reproducibility.qmd`, `docs/2026-08-16-development-calibration.md`, `scripts/check_site_figures.py`
- Delete: `assets/demo_density.png`, `site/assets/demo_density.png`
- Test: `tests/test_style.py`, `tests/test_density_demo.py`, `tests/test_generated_assets.py`, `tests/test_site_figure_style.py`

**Interfaces:**
- Consumes: `record()` (already in `demo_density.py` from pull request 1), `density_figure` (Task 3), `style` from `clocks._style`; `RETIRED_ASSETS`, `LIBRARY_DRAWING_MODULES` (Task 2); `EXPECTED_FIGURES` (Task 1).
- Produces: `save_png(fig: Figure, path: Path) -> None` in `clocks._style`, beside `show`, with the settings of Global Constraints; it does not close the figure (callers do). `demo_density._run_demo(args)` as described below; `_save_summary` is deleted. `build_parser`, `main`, `build_density_filter`, `scenario` and `record` keep their signatures. In `tests/test_generated_assets.py`, `RETIRED_VISUALS: dict[str, str]` (file name to the SHA-256 it was released with), which Task 6 extends.

- [ ] **Step 1: Write the failing tests.**

In `tests/test_style.py`:

- `test_save_png_writes_the_authored_canvas` (IDs `plain`, and `hostile` under `matplotlib.rc_context({"savefig.bbox": "tight", "savefig.transparent": True, "savefig.facecolor": "magenta"})`): a figure made inside `with style("light"):` with `plt.subplots(figsize=(7.5, 5.4))` and one line; that block is then exited, and `save_png(fig, tmp_path / "out.bin")` is called after it, with the case's `rc_context` (for `hostile`) still active, so the caller's magenta face colour is in force at the save; leaves exactly one file, `out.bin`, starting with the PNG signature, `png_size` (1500, 1080), and with `pixel = imread(path, format="png")[0, 0]`, `pixel[:3]` equal to `to_rgb(palette("light").background)` (`rtol=0, atol=1/255`) and `pixel[3] == 1.0` (P15).

In `tests/test_density_demo.py` (each reads the PNG size with `png_size` from `tests/_figure_artists.py`):

- `test_density_command_prints_and_draws_the_recorded_run`: `main(["--output", str(tmp_path / "new" / "density.png"), "--observations", "3", "--particles", "64"]) == 0`. With `expected = demo_density.record(n_observations=3, n_particles=64)`, `m = expected.arrays["mean"][-1]`, `s` its standard deviations and `ess = expected.arrays["ess"][-1]`, standard output holds the line `Final estimate:` followed by exactly these four lines: `f"  mu    = {m[0]:.3f} ± {s[0]:.3f}  (true: 1.5)"`, `f"  sigma = {m[1]:.3f} ± {s[1]:.3f}  (true: 2.0)"`, `f"  A     = {m[2]:.3f} ± {s[2]:.3f}  (true: 0.01)"`, `f"  ESS   = {ess:.0f} / 64"`; and `png_size(tmp_path / "new" / "density.png") == (1500, 1080)`. (P5: today's command prints μ = 1.182 here, `record()` 1.768.)
- `test_density_command_prints_progress_from_the_replay`: `demo_density.record` replaced by a stub that records its keywords and returns the committed archive (`load_replay(ROOT / "assets/replays/demo_density.npz")`, 60 observations) with `true_rates[0]` changed to `0.5` (`Replay(meta, {**arrays, "true_rates": edited})`), a value no scenario computes. `main(["--output", str(tmp_path / "d.png"), "--observations", "40", "--particles", "16"])` calls it once with `n_observations=40, n_particles=16`; standard output starts with `True density: mu=1.5, sigma=2.0, amplitude=0.01` and the line `f"True rates: {edited}"`, the stub's edited rates, so the printed rates can only come from the replay; and for k = 20, 40 and 60 holds `After {k} observations:` followed by the four lines of that row in the format above, the ESS line over `meta["n_particles"]` (2,000). The progress is printed from the replay the command drew, not from its arguments.
- `test_density_command_saves_the_figure_it_drew`: `demo_density.density_figure` is wrapped by a spy that records its arguments, the figure it returns and `to_hex(matplotlib.rcParams["figure.facecolor"])` at the moment of the call, and calls through; `demo_density.save_png` by a spy that records its arguments and calls through. One run, `--output <tmp_path>/d.png --observations 2 --particles 16`, calls the builder once, with a replay whose `meta["n_observations"] == 2` and `meta["n_particles"] == 16`, with `colors == palette("light")`, and with the light style active (the recorded face colour is `palette("light").background`); and calls `save_png` exactly once, with the figure the builder returned (`is`) and the path `tmp_path / "d.png"`. The file checks of the previous tests are output smoke tests; this one proves which figure is written, and how.
- `test_density_command_writes_png_bytes_at_the_path_given` (IDs `no_suffix` `out/d`, `misleading_suffix` `out/d.jpg`): with the working directory changed to `tmp_path` (`monkeypatch.chdir`), `main(["--output", <path>, "--observations", "1", "--particles", "16"])` leaves exactly one file under `tmp_path`, at `<path>`, whose first eight bytes are `b"\x89PNG\r\n\x1a\n"`. (Today the first writes `out/d.png` and the second a JPEG: P12. A deliberate small change, stated in the commit body.)
- `test_density_command_writes_only_its_png`: the same with `--output out/d.png` leaves exactly `out/d.png`. This is a preservation test: it passes before the change too, and guards against the command starting to write an archive or a second file.

In `tests/test_generated_assets.py`:

- Remove the `"demo_density.png"` row from `EXPECTED_VISUAL_HASHES`.
- `RETIRED_VISUALS = {"demo_density.png": "a7a6e6e9628640ac08e89152b8471d37db981c264380ba567634032a0f59dc7a"}`.
- `test_retired_visuals_are_gone_but_recorded` (each of `RETIRED_VISUALS`): neither `ASSETS / name` nor `SITE_ASSETS / name` exists; the calibration report still has the table row ``f"| `{name}` | `{expected_hash}` |"`` as one of its lines, so the hash stays pinned to its file name; and the report matches `rf"`{re.escape(name)}` was removed from both trees on \d{{4}}-\d{{2}}-\d{{2}};"`.
- `test_density_provenance_names_its_run`: with `meta = load_replay(ROOT / "assets/replays/demo_density.npz").meta`, the whitespace-normalised Reproducibility page contains `"**Replay data.** The density figure on [Beyond Point Masses](../story/beyond-point-masses.qmd) is drawn when the site is built, from `assets/replays/demo_density.npz`, the recorder's archive of the demo's packaged run: "` + `f"{meta['n_observations']} observations, {meta['n_particles']:,} particles and seed {meta['seed']}. "`; and does not contain `"density PNG"`.
- `test_provenance_section_describes_figures_drawn_at_build_time`: the raw page contains `"## Where the figures come from {#sec-figures}"`; the normalised page contains `"[development calibration report](https://github.com/jbwhit/clocks/blob/main/docs/2026-08-16-development-calibration.md) is the record of the August 2026 release"`.
- `PROVENANCE_VOCABULARY` gains `"committed replay"` and `"replay file"`.

In `tests/test_site_figure_style.py`: `RETIRED_ASSETS = ("demo_density.png",)`; `LIBRARY_DRAWING_MODULES` gains `"src/clocks/_demos/demo_density.py"`. In `scripts/check_site_figures.py`, `"story/beyond-point-masses.html": (Fig("plot", 540), Fig("plot", 420))`.

- [ ] **Step 2: Stage the names, then run.** Add `save_png` to `clocks._style` with the body `fig.savefig(path, format="png")` only, and add `from clocks._figures import density_figure` and `from clocks._style import save_png, style` to `demo_density.py`, after `matplotlib.use("Agg")`; `_run_demo` stays on its old path. Run `uv run pytest tests/test_style.py tests/test_density_demo.py tests/test_generated_assets.py tests/test_site_figure_style.py tests/test_check_site_figures.py -q`. Both `save_png` cases fail on size (750 by 540) and the hostile one on transparency; the command tests fail on their assertions (today's run and a 2250 by 600 PNG; the stubbed `record` never called; the spies never called; `out/d.png` and a JPEG written), except the preservation test, which passes; the retired-asset, provenance and manifest tests fail.

- [ ] **Step 3: Implement.**

`demo_density._run_demo(args)`: `replay = record(n_observations=args.observations, n_particles=args.particles)`; print the truth and the true rates from `replay.arrays` in today's format (`True density: mu=1.5, sigma=2.0, amplitude=0.01`, then `True rates: [...]`); for k = 20, 40, … up to the last observation, print today's `After k observations:` block from row k of `mean`, the standard deviations of `cov` and `ess`; then `Final estimate:` from the last row; then, inside `with style("light") as colors:`, `fig = density_figure(replay, colors)`, `args.output.parent.mkdir(parents=True, exist_ok=True)`, `save_png(fig, args.output)`, `plt.close(fig)`; then `Saved: <path>`. The progress loop runs over the replay's own `meta["n_observations"]`, and the ESS lines divide by `meta["n_particles"]`. `save_png` is implemented with the settings of Global Constraints. `_format_estimate` takes the mean, the standard deviation and the truth as floats. Delete `_save_summary` and every import only it used. Imports stay after `matplotlib.use("Agg")` (`tests/test_cli.py::test_demo_selects_agg_before_importing_pyplot`). Update `record()`'s docstring: the command now runs exactly this.

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

`docs/2026-08-16-development-calibration.md`: after the hash table, a paragraph whose first sentence is "`demo_density.png` was removed from both trees on YYYY-MM-DD (today's date); the density figure is now drawn when the site is built, from `assets/replays/demo_density.npz`, and its old bytes remain in git history under the hash above." The table is not edited.

`site/reproduce/reproducibility.qmd` (line numbers as on `main`):

- Lines 38 to 40 become "…regenerated from source at deploy time. The demo GIFs and the echolocation range-study PNG are the exceptions: they are committed files, described in the next section."
- Line 46 becomes `## Where the figures come from {#sec-figures}`, and lines 48 to 52 become: "The story pages keep commentary on these figures' origin to a short caveat. This section summarizes where each came from and what it does and does not establish. The [development calibration report](https://github.com/jbwhit/clocks/blob/main/docs/2026-08-16-development-calibration.md) is the record of the August 2026 release: the generating commit and the hashes of the files that release shipped, not of every figure now on this site."
- Lines 54 to 62 become two paragraphs. **Demo animations.** Today's paragraph with the density figure taken out: "Each committed GIF is the packaged default output of its `uv run demo-*` command, …", "Each animation depicts one named finite realization …", the rest unchanged. Then: "**Replay data.** The density figure on [Beyond Point Masses](../story/beyond-point-masses.qmd) is drawn when the site is built, from `assets/replays/demo_density.npz`, the recorder's archive of the demo's packaged run: 60 observations, 2,000 particles and seed 42. The archive holds the configuration, seed and observations, a summary of the cloud before any observation and after each, and a 500-particle sample for display. Tests run the demo again; they compare its inputs, the computed ones (clock positions, true rates and readings) at `rtol=1e-12, atol=1e-12` and the rest exactly, and its summaries within stated bounds; they do not require the same random path. `uv run demo-density` records the same configured scenario and draws its result. The archive is one finite realization: it establishes no population recovery rate."
- Lines 64 to 66 (the density prior) are unchanged.

- [ ] **Step 4: Tests pass; full gate; render the site and run both checks.** `check_site_figures.py` prints `Figure contract OK: 9 pages, 10 figures checked.` Open the rendered `story/beyond-point-masses.html` in a browser, switch theme, and confirm the figure shows in both; confirm `site/_output/assets/` holds no `demo_density.png`.

- [ ] **Step 5: Commit and push** as `Draw the density figure from its replay; demo-density runs the recorded scenario`. The body states that the command's run changed (P5), that `--output` now gets PNG bytes at exactly the path given (an extensionless path no longer gains `.png`), and that the archive was not re-recorded.

---

### Task 5: `range_study_figure`, with `write_summary_figure` as its wrapper

**Files:**
- Modify: `src/clocks/_echo_study.py`, `src/clocks/viz.py`, `tests/test_site_figure_style.py`
- Test: `tests/test_echo_study.py`

**Interfaces:**
- Consumes: `summarize` (same module); `Palette`, `style`, `THEMES`, and `save_png` (Task 4) from `clocks._style`; the helpers in `tests/_figure_artists.py`.
- Produces: `range_study_figure(study: Mapping[str, Any], colors: Palette) -> Figure` in `clocks._echo_study`, exported from `clocks.viz`; `write_summary_figure(study: dict, png_path: Path) -> None`, same signature, now `with style("light") as colors:` → `range_study_figure(study, colors)` → `save_png(fig, png_path)` → close, both called through module globals (`save_png` imported from `clocks._style`) so a test can spy on them. `scripts/scan_echolocation_range.py` and the monkeypatch in `tests/test_scenarios.py` are untouched.

**The range-study figure's artists.** Two stacked axes sharing x, both log-scaled in y, gids `"position-error"` (top) and `"mass-error"`. In each: a `Line2D` `"runs"`, one point per record in record order, `linestyle="none"`, `marker="o"`, colour `ink`, alpha 0.35; a `Line2D` `"median-error"` over the ranges `summarize` returns, solid, `marker="o"`, colour `ink`; a `Line2D` `"median-spread"`, `marker="s"`, solid, colour `belief`; and an `axhline` `"limit"` at the study's tolerance (`position_error_max`, `mass_error_max`), dashed, colour `threshold`, labelled `f"limit {tolerance}"` in `threshold` (so `limit 1.0`, `limit 0.04`). In the top axes only: the direct labels `"median error"` (`ink`) and `"median posterior s.d."` (`belief`), one each in the whole figure; one `Text` per range with gid `"pass-count"`, x at the range in data coordinates and y just above the axes, reading the count of runs at that range whose position error and mass error are both within the tolerances; and one `Text` `"runs within both limits"` (gid `"pass-count-label"`). Y labels "position error" and "mass error", x label "range (circumradii)" on the bottom axes, no title on either axes and no figure title. `figsize=(7.5, 5.4)`.

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
        _run(2.0, 0.20, 0.004, 0.08, 0.004, False),  # inside both; its flag disagrees
        _run(4.0, 1.50, 0.010, 1.20, 0.020, False),  # outside position
        _run(4.0, 1.00, 0.040, 0.90, 0.030, True),   # exactly on both limits
        _run(4.0, 2.50, 0.030, 1.60, 0.025, False),
    ]}
```

By hand: medians at 2.0 and 4.0 are position error 0.10 and 1.50, mass error 0.004 and 0.030, position s.d. 0.06 and 1.20, mass s.d. 0.003 and 0.025; runs within both limits 2 and 1. The position limit alone gives 3 and 1, the mass limit alone 2 and 3, strict `<` comparisons 2 and 0 (the run on both limits drops out), and the records' `passed` flags 1 and 1 (the third run's flag disagrees with its errors). At `position_error_max` 2.0 and `mass_error_max` 0.035: both limits 2 and 1; position alone 3 and 2; mass alone 2 and 2; flags 1 and 1; a mass limit fixed at 0.04 gives 2 and 2 (all recomputed, P13). As in Task 3, artists are fetched with `one(axes, gid)` on the axes that owns them, and texts are looked up on their owning axes.

- `test_range_study_figure_size_and_axes` (each theme): size `(7.5, 5.4)`; gids `["position-error", "mass-error"]`; both y scales `"log"`; the two axes share x; both titles `""` and `fig.get_suptitle() == ""` (P12); labels as above; and `gid_census(fig)` equals `{"runs": BOTH, "median-error": BOTH, "median-spread": BOTH, "limit": BOTH, "pass-count": ["position-error", "position-error"], "pass-count-label": ["position-error"]}` with `BOTH = ["position-error", "mass-error"]`.
- `test_runs_are_faint_ink_dots` (each theme): in each axes, `"runs"` has x data `[2.0, 2.0, 2.0, 4.0, 4.0, 4.0]` and y data the six errors of that panel in record order; marker `"o"`, line style `"None"`, colour `colors.ink`, visible, alpha above 0 and below 1.
- `test_medians_and_spreads_come_from_summarize` (each theme): `"median-error"` and `"median-spread"` have x data `[2.0, 4.0]` and y data `[0.10, 1.50]`, `[0.06, 1.20]` (top) and `[0.004, 0.030]`, `[0.003, 0.025]` (bottom), each also equal to the matching `summarize` field; colours `ink` and `belief`; markers `"o"` and `"s"`; both lines' style `"-"` (a median line with its line style disabled would show only circles).
- `test_limits_are_the_studys_tolerances` (each theme): `"limit"` y data `[1.0, 1.0]` and `[0.04, 0.04]`, style `"--"`, colour `colors.threshold`.
- `test_limits_follow_the_study`: with `position_error_max` 2.0 and `mass_error_max` 0.035, the top `"limit"` is at `[2.0, 2.0]` and the bottom one at `[0.035, 0.035]`, the texts `"limit 2.0"` (top) and `"limit 0.035"` (bottom) exist, and the pass counts read `["2", "1"]`. (These counts equal the default tolerances' counts, so this test pins the drawn limits; the next one pins the counts.)
- `test_pass_counts_follow_each_tolerance` (IDs `position_changed`: `position_error_max` 2.0 with the default mass limit, counts `["2", "2"]`; `mass_changed`: `mass_error_max` 0.003 with the default position limit, counts `["1", "0"]`; recounted, P16): the `"pass-count"` texts in x order. Counts computed with the default tolerances would read `["2", "1"]` in both.
- `test_pass_counts_use_both_limits`: the `"pass-count"` texts, as `(x, text)` pairs in x order, are `[(2.0, "2"), (4.0, "1")]` (the flags would give `"1"` at 2.0); exactly one text in the figure reads `"runs within both limits"`.
- `test_labels_sit_on_their_axes` (each theme): the top axes holds exactly one visible `Text` reading each of `"median error"` (`ink`), `"median posterior s.d."` (`belief`), `"limit 1.0"` (`threshold`) and `"runs within both limits"`; the bottom axes holds exactly one reading `"limit 0.04"` (`threshold`) and none reading `"median error"` or `"median posterior s.d."`, so each line is labelled once in the figure.
- `test_changing_one_run_moves_its_artists`: with the first record's position error 0.5, the top `"runs"` y data start with 0.5 and the top `"median-error"` y data are `[0.20, 1.50]`; the bottom axes' `"runs"`, `"median-error"` and `"median-spread"` data are `np.array_equal` to the original's.
- `test_range_study_on_the_committed_study` (P4): from `load_study(ROOT / "assets/echolocation_range_study.json")`, the `"pass-count"` texts are `["12", "12", "11", "8", "3", "0"]` at x `[2.0, 2.6, 3.5, 4.6, 6.1, 8.0]`; the limits are at 1.0 and 0.04; the `"median-error"` y data equal `summarize`'s `med_position_error` and `med_mass_error`, exactly.
- `test_every_visible_colour_is_in_the_palette` (each theme), on the committed study: `visible_colours(fig) <= palette_hexes(colors)`.
Both wrapper tests below run inside `matplotlib.rc_context({"figure.facecolor": "magenta", "savefig.facecolor": "magenta"})`, so a figure built outside the light style, or saved with the caller's face colour, is visibly wrong whatever the session's rcParams.

- `test_write_summary_figure_saves_the_light_range_study`: with `clocks._echo_study.range_study_figure` wrapped by a spy that records its arguments, the figure it returns and `to_hex(matplotlib.rcParams["figure.facecolor"])` at the moment of the call, and `clocks._echo_study.save_png` by a recording spy, both calling through, `write_summary_figure(study, path)` calls the builder once with that very `study` object, `palette("light")` and the light style active (the recorded face colour is `palette("light").background`), and `save_png` once, with the figure the builder returned (`is`) and `path`.
- `test_write_summary_figure_output` (smoke test of the file): written from the literal study, `png_size(path) == (1500, 1080)`; with `pixel = matplotlib.image.imread(path)[0, 0]`, an RGBA value (P12), `pixel[:3]` equals `to_rgb(palette("light").background)` with `rtol=0, atol=1/255` and `pixel[3] == 1.0`.
- `test_write_summary_figure_creates_png` (existing, fake records with `mass_error=0.0`) still passes unchanged.
- `test_viz_exports_the_range_study_figure`: `clocks.viz.range_study_figure is clocks._echo_study.range_study_figure`, and the name is in `__all__`.

In `tests/test_site_figure_style.py`: `FIGURE_FUNCTION_HEIGHTS["range_study_figure"] = 540`, with its drawer reading the committed study; `LIBRARY_DRAWING_MODULES` gains `"src/clocks/_echo_study.py"`; remove the `xfail` mark from `test_every_figure_builder_module_is_scanned`.

- [ ] **Step 2: Stage the name, then run.** Add `range_study_figure` returning an empty `plt.figure(figsize=(7.5, 5.4))`, exported from `clocks.viz`, and add `from clocks._style import save_png, style` to `_echo_study.py`; leave `write_summary_figure` as it is. Run `uv run pytest tests/test_echo_study.py tests/test_site_figure_style.py -q`: the artist tests fail in `one(...)` on missing gids, the wrapper tests because the old wrapper never calls the builder and saves at 150 dpi, and the library colour check on today's drawing (`"o"`, `color="steelblue"`).

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

`docs/2026-08-16-development-calibration.md`: add to Task 4's paragraph "`echolocation_range_study.png` was removed from both trees on YYYY-MM-DD (today's date); the range-study figure is now drawn when the site is built, from `echolocation_range_study.json`, whose row above is unchanged, and the PNG's old bytes remain in git history under the hash above."

`site/reproduce/reproducibility.qmd`: the CI bullet's last sentence becomes "The demo GIFs are the exceptions: they are committed files, described in the next section."; the sentence "The displayed study was regenerated directly from the certified block-400 records." becomes "The figure is drawn when the site is built, from the certified block-400 records in `assets/echolocation_range_study.json`."

- [ ] **Step 4: Tests pass; full gate; render the site and run both checks.** `check_site_figures.py` prints `Figure contract OK: 9 pages, 11 figures checked.` Confirm in the rendered `story/gravitational-echolocation.html` that the counts along the top read 12, 12, 11, 8, 3, 0, in both themes.

- [ ] **Step 5: Commit and push** as `Draw the range-study figure at build time and retire its PNG`.

---

### Task 7: The site check runs when the figure data change; contact sheets

**Files:**
- Modify: `.github/workflows/site-check.yml`, `site/reproduce/reproducibility.qmd`, and any figure code Step 5 has to fix (`src/clocks/_figures.py`, `src/clocks/_echo_study.py`)
- Create: `tests/test_site_check_workflow.py`; under `docs/plans/2026-11-19-replay-and-animation/` (the directory the spec's "Checks by eye" names), `contact_sheets.py`, `density.jpg` and `range-study.jpg`
- Test: `tests/test_site_check_workflow.py`, `tests/test_generated_assets.py`

**Interfaces:**
- Consumes: `density_figure`, `range_study_figure`, `load_replay`, `style`, `THEMES` from `clocks.viz`; `save_png` from `clocks._style`.

- [ ] **Step 1: Write the failing tests.**

In `tests/test_site_check_workflow.py`, reading workflows as text (no YAML library is a declared dependency): `_pull_request_paths(text: str) -> list[str]` returns the entries of the `paths:` list directly under the top-level `on:` → `pull_request:` key: the lines after that block's `    paths:` that are `      - "<entry>"`, comments skipped, up to the first line indented four spaces or less.

- `test_the_path_parser_reads_only_the_pull_request_list`: on this fixed fixture, `_pull_request_paths(FIXTURE) == ["site/**", "src/clocks/**"]`; the `push.paths` decoy is not returned.

```python
FIXTURE = """name: Fixture
on:
  push:
    branches: [main]
    paths:
      - "assets/replays/**"
  pull_request:
    branches: [main]
    paths:
      - "site/**"
      # a comment inside the list
      - "src/clocks/**"

concurrency:
  group: fixture
"""
```

- `test_site_check_runs_when_the_figure_data_change` (IDs the two paths): `_pull_request_paths(<site-check.yml text>).count(path) == 1` for `assets/replays/**` and for `assets/echolocation_range_study.json`.

In `tests/test_generated_assets.py`: `test_site_check_description_names_the_figure_data`: the whitespace-normalised Reproducibility page contains `"touch the site, the library, or the data the site's figures are drawn from"`.

- [ ] **Step 2: Run them.** The parser test passes once the helper is written (it needs nothing from the workflow); the path test fails on both paths; the description test fails.

- [ ] **Step 3: Add the two paths** under `pull_request.paths`, after `"src/clocks/**"`, with a comment: the site draws its density and range-study figures from these files. In the Reproducibility page's `site-check.yml` bullet, "on non-draft pull requests targeting `main` that touch the site or the library" becomes "on non-draft pull requests targeting `main` that touch the site, the library, or the data the site's figures are drawn from".

- [ ] **Step 4: Contact sheets.** `contact_sheets.py` takes `--out DIR`; for each figure and theme it draws through `clocks.viz` inside `style(theme)`, writes the rendering with `save_png` to a temporary file (1500 by 1080), and composes one sheet per figure with Pillow (already installed, as a Matplotlib dependency): the light and dark renderings at full width in the top row, and the same two resampled to 780 pixels wide (390 CSS pixels at twice density, what a phone shows) in the bottom row, each labelled with its theme and width. It writes `density.jpg` and `range-study.jpg` with `image.save(path, quality=85)`, passing no `exif` and no `comment`, and prints their paths relative to the repository. It holds no absolute path and must pass `ruff format --check` and `ruff check` (the directory is not excluded). Run `uv run python docs/plans/2026-11-19-replay-and-animation/contact_sheets.py --out docs/plans/2026-11-19-replay-and-animation`. The metadata check (run in Step 6, on the final sheets) is where a path or a name could be stored: `Image.open(path).getexif()` is empty, `set(Image.open(path).info) == {"jfif", "jfif_density", "jfif_unit", "jfif_version"}`, and `[marker for marker, _ in Image.open(path).applist] == ["APP0"]` with that segment's data starting `b"JFIF\x00"` (what Pillow 12.1.1 writes for a plain save: P15, P16), so no `comment`, `exif`, `xmp` or `icc_profile` key and no other application segment is present. The compressed image data is not searched; it holds arbitrary byte sequences.

- [ ] **Step 5: Look at both sheets** and report, per sheet, each item as seen or not seen: no text overlaps other text or a data mark at either width; at 390 pixels the trace headers, the pass counts and the limit labels can be read; the density band is visible in both themes; the faint run dots are visible on the dark background; the dashed limits are told apart from the grid. A failed item is fixed in the figure code, Tasks 3 and 5's tests are re-run, and the sheets are redrawn; say what moved.

- [ ] **Step 6: Check the final sheets, gate and render the final tree.** On the sheets as they stand after any Step 5 redraw, check the metadata described in Step 4. Tests pass; full gate; then, whether or not Step 5 changed anything, render the site and run both checks on this tree (`Figure contract OK: 9 pages, 11 figures checked.`).

- [ ] **Step 7: Phone width on the rendered site**, PR #21's procedure (`docs/plans/2026-11-18-figure-style.md`, Task 4 Step 7), on the render of Step 6. Serve `site/_output` and load `story/beyond-point-masses.html` and `story/gravitational-echolocation.html` each in a 390-pixel-wide `<iframe>` in headless Chrome (a window narrower than about 500 pixels is not laid out; a `--window-size=390` screenshot crops instead of reflowing). Do it once per theme: the page's switch calls `window.quartoToggleColorScheme()`, and the dark theme is active when `document.body` has the class `quarto-dark`. In each theme, for each page, record `document.documentElement.scrollWidth`, and for each plot rendering visible in that theme (two per page) its rendered image width and its container's width. Expected: the scroll width is 390, and every visible plot's width satisfies `0 < image_width <= container_width`; a hidden rendering reports zero width and must not be the one measured. The numbers go to the coordinator for the pull request's description, not into a file.

- [ ] **Step 8: Commit and push** the workflow, the Reproducibility page, the tests, the script, the two sheets and any figure code Step 5 changed, as `Run the site check when the figure data change; add contact sheets of both figures`. The body lists Step 5's findings and any figure change.

---

### Task 8: Follow-ups owed to the replay tests and documents

Items 1, 2 and 5 of pull request 1's deferred list. No archive is re-recorded: every test below reads the committed `demo_2d` and echolocation archives, or records a one-particle run in memory.

**Files:**
- Modify: `tests/test_replay.py`, `tests/test_replays.py`, `docs/plans/2026-11-19-replay-and-animation-design.md`, `docs/plans/2026-11-19-replay-data.md`

- [ ] **Step 1: The thinning tie.** In `tests/test_replay.py`, `PHI = (math.sqrt(5.0) - 1.0) / 2.0`, the call `_replay.THINNING_OFFSET` makes, so the tie case stays exact wherever `5**0.5` and `math.sqrt(5.0)` round differently. Nothing fails before or after on this machine; the change is for exactness, not a new check.

- [ ] **Step 2: Write the failing tests** in `tests/test_replays.py`.

Zero spread (spec, section 3's prediction invariant; P8). `ZERO_SPREAD_ATOL = 1e-7`, with a comment: where `predicted_std` is 0, as in a one-particle run, the dots' mean reading must be within this of `predicted_mean` and their standard deviation at most this; rounding a dot to float32 moves its reading by 6.4e-11 (`demo_1d`) and 1.4e-9 (`demo_multi_mass`) in one-particle recordings.

- `test_a_one_particle_recording_passes_the_prediction_check` (IDs `demo_1d`, `demo_multi_mass`): `replay = module.record(n_observations=2, n_particles=1)`; under `warnings.simplefilter("error")`, `_assert_predictions_match_dots(predicted_mean, predicted_std, _readings_of_dots(cloud, clock_positions, track_offset, centred=False, first_row=0), 0)` does not raise. Every row there has zero spread at every clock (P8), so this also covers a row with no positive-spread clock.
- `test_a_one_particle_prediction_off_the_dot_is_caught`: the `demo_1d` recording above with `predicted_mean[0, 0] += 1e-6` raises `AssertionError` matching `row 0, clock 0: the dots' mean reading differs from predicted_mean by 1e-06 where predicted_std is 0`.
- `test_a_spread_where_none_is_predicted_is_caught`: on plain arrays, `predicted_mean = np.array([[1.0]])`, `predicted_std = np.array([[0.0]])`, dot readings `np.array([[[0.99], [1.01]]])`: raises matching `row 0, clock 0: the dots' s.d. is 0.01 where predicted_std is 0`.
- `test_a_mixed_row_names_the_original_clock`: on plain arrays, `predicted_mean = np.array([[1.0, 1.0]])`, `predicted_std = np.array([[0.0, 0.01]])`, dot readings `np.array([[[1.0, 1.09], [1.0, 1.11]]])`: clock 0 has zero spread and agrees; clock 1 is 10 predicted s.d. off, 14.1 in units of s.d./√m against 6. Raises matching `row 0, clock 1: the dots' mean reading is`; a helper that renumbered the positive-spread clocks would say clock 0.

Prior cases declared on their own. `PRIOR_CHECKED_DEMOS = ("demo_1d", "demo_multi_mass")`, with the comment that their panels draw the prior's predicted bars. `_SHIFTS` builds its `prior` cases from this tuple, not by filtering `PREDICTING_DEMOS` through `FIRST_CHECKED_ROW`.

- `test_prior_cases_and_the_skip_map_partition_the_predicting_demos`: `set(PRIOR_CHECKED_DEMOS).isdisjoint(FIRST_CHECKED_ROW)` and `set(PRIOR_CHECKED_DEMOS) | set(FIRST_CHECKED_ROW) == set(PREDICTING_DEMOS)`.

Rows chosen at run time (P9):

- `test_a_tilted_covariance_is_caught`: from `demo_2d`'s `cov`, the x–y correlation of every row; `row` is the index, after the prior, of the largest absolute value. First assert that value exceeds 0.2 (the spec: a reversed tilt is caught only where |r| exceeds about 0.2). Flip the sign of both off-diagonal entries of that row; `_assert_dots_share_orientation` raises matching `f"row {row}:"`. (Row 5 today, |r| 0.326.)
- `test_a_weight_in_view_off_by_its_sampling_error_is_caught`: from echolocation's `cloud` and `view_bounds`, the fraction `p` of dots inside the view in every row; `row = int(np.argmin(p))`; set `weight_in_view[row] = p[row] + 0.036`. First assert, with `bound = max(0.01, 6 * sqrt(p (1 − p) / m))` at that row, that `bound < 0.036 < 0.05`: outside six sampling errors, inside the old fixed tolerance. Then the helper raises matching `f"row {row}:"`. (Row 0 today, `p` 0.004, bound 0.0169.)

- [ ] **Step 3: Run** `uv run pytest tests/test_replays.py -q`; what fails, measured with today's helper (P16): both `test_a_one_particle_recording_passes_the_prediction_check` cases, on `RuntimeWarning: divide by zero encountered in divide`, which their `warnings.simplefilter("error")` raises before any assertion; `test_a_one_particle_prediction_off_the_dot_is_caught`, whose `match` fails because today's message reads `row 0, clock 0: the dots' mean reading is inf s.d./sqrt(m) from predicted_mean`; `test_a_spread_where_none_is_predicted_is_caught`, whose `match` fails on `… is nan s.d./sqrt(m) …` (the mean check fires first, on 0/0); and `test_a_mixed_row_names_the_original_clock`, on the same `nan` message naming clock 0 (P14). The rest pass, being rewrites of passing tests.

- [ ] **Step 4: Implement** in `_assert_predictions_match_dots` (the committed archives have no zero entry in `predicted_std`, P15, so their existing tests are unaffected): per row, split the clocks by `predicted_std == 0`. The zero-spread clocks are checked against `ZERO_SPREAD_ATOL` with the two messages above; the others exactly as today. Reductions such as `argmax` run only over a non-empty set, so a row with no positive-spread clock skips them, and every message names the clock by its original column index (index into the full row, not into the subset). No division by a zero spread. In `assert_computed_input_matches`' docstring, "their last bit can differ" becomes "their low-order bits can differ".

- [ ] **Step 5: Correct the two documents** that still describe clock positions as compared exactly.

- Design, section 3, gate item 1 becomes: "**Schema and inputs.** Key set, shapes and dtypes; `meta` except `created_with`; `true_params`, `model_ks` and `model_prior`, exactly. The inputs computed in floating point (`clock_positions`, `true_rates`, `observations`, `filter_observations`) are compared with `rtol=1e-12` and `atol=1e-12`: a library's low-order bits are not the scenario, and centred and contrast values near zero need the absolute term. The two-mass 2D demo's clocks are drawn from a seeded generator, and the x86-64 runner rounds their scaling differently from the arm64 machine that recorded the archive (pull request 1, `a108783`)."
- Design, section 3, first committed-archive invariant: "`meta` and the geometry equal the demo's `scenario()`" becomes "`meta` equals the demo's `scenario()`, and its clock positions match the scenario's at `rtol=1e-12, atol=1e-12`".
- Plan for pull request 1: in Global Constraints, `clock_positions` joins the computed inputs; in Task 9, `test_archive_matches_its_scenario` reads "`true_params` is `np.array_equal` to the scenario's; `clock_positions` and `true_rates` match with `rtol=1e-12, atol=1e-12`", and `test_gate_inputs` moves `clock_positions` from the exact list to the `rtol=1e-12, atol=1e-12` list.
- Design, section 9, the **Replay data** bullet: "Tests run each demo again, compare its inputs exactly and compare its summaries within stated bounds;" becomes "Tests run each demo again; they compare its inputs, the computed ones (clock positions, true rates and readings) at `rtol=1e-12, atol=1e-12` and the rest exactly, and its summaries within stated bounds;". Task 4's prose already says this.
- Each document's status paragraph gains one sentence: "Corrected after merge: clock positions are compared at `rtol=1e-12, atol=1e-12`, not exactly (`a108783`)."
- Then `rg -n "clock_positions|inputs exactly|exactly" docs/plans/2026-11-19-replay-and-animation-design.md docs/plans/2026-11-19-replay-data.md` and read every hit; no description of the **committed-archive gate comparisons** may still call clock positions or computed inputs exact. The same-run recorder tests in `tests/test_demo_records.py` (for example the scenario-clocks test) are a separate contract, compare within one process, and stay exact.

- [ ] **Step 6: Tests pass; full gate; commit and push** as `Settle the replay tests' owed follow-ups: zero spread, prior cases, rows chosen at run time`.

---

### Task 9: Follow-ups owed to the replay report

Items 3, 4 and the `meta` part of 6 of pull request 1's deferred list.

**Files:**
- Modify: `src/clocks/_replay.py`, `.github/workflows/replay-report.yml`
- Test: `tests/test_replay.py`, `tests/test_record_replays_script.py`, `tests/test_replay_report_workflow.py`

**Interfaces:**
- Produces: `first_difference` returns `Difference(f"meta.{field}", None, math.nan)` for a `meta` difference, `field` being the first key in `sorted(set(reference.meta) | set(other.meta))`, `created_with` excluded, whose values differ or which one side lacks. The report line becomes, for example, `demo_1d: differs at meta.seed`.

- [ ] **Step 1: Write the failing tests.**

- `tests/test_replay.py`: `test_meta_difference_is_reported_first` expects `("meta.seed", None)`. New `test_meta_difference_names_the_first_field_in_sorted_order`: `seed` and `noise_std` both changed → `"meta.noise_std"`. It first asserts that `seed` comes before `noise_std` in the fixture's insertion order (it does, in `BASE_META`), so a loop over insertion order would answer `"meta.seed"` and fail. New `test_a_meta_key_missing_on_one_side_is_named` (IDs `other_lacks`, `reference_lacks`): `cloud_size` deleted from `other.meta`, or from `reference.meta`, → `"meta.cloud_size"` (a loop over the reference's keys alone would miss the second).
- `tests/test_record_replays_script.py`: the two expected lines in `test_baselines_are_all_read_before_any_demo_is_recorded` become `"demo_1d: differs at meta.seed"` and `"demo_2d: differs at meta.seed"`; the line in `test_a_difference_that_is_not_numeric_omits_the_size` becomes `"demo_1d: differs at meta.seed"`.
- `tests/test_replay_report_workflow.py`: `_run_last_step` gains `path_prefix: Path | None = None`, put before `/usr/bin:/bin` in `PATH`. New `test_a_failed_tail_leaves_the_summary_untouched`: a directory holding an executable `tail` that is `#!/bin/sh\nexit 1\n`, and a report `"# report\n"` with upload `success`: standard output contains `the report for sample 2 could not be added to the summary`, and the summary file, seeded beforehand with `"previous summary\n"`, still holds exactly those bytes: nothing is appended and nothing truncated, so no opening fence is left behind. (`_run_last_step` gains a way to seed the summary file.) New `test_the_upload_runs_on_node_24`: in the step whose lines include `id: upload` (found with `_steps()`), the line whose stripped form starts with `uses:` is exactly `uses: actions/upload-artifact@v7` (P10). A mention elsewhere, such as a comment, neither satisfies nor fails it.

- [ ] **Step 2: Run** those three files; the new and changed tests fail on their assertions (`meta` without a field; no warning, because `[ -z "$(tail ...)" ]` sees the failed `tail` as an empty last byte, and the fence and report already appended; `@v4`).

- [ ] **Step 3: Implement.**

- `first_difference` as in Interfaces. The report's text sanitizer is unchanged; `meta` keys are validated, so a field name is plain ASCII.
- The summary line of the last step becomes:

```bash
last=$(tail -c1 replay-report.md) && { echo '```text' && cat replay-report.md && { [ -z "$last" ] || echo; } && echo '```'; } >> "$GITHUB_STEP_SUMMARY" || echo "::warning title=Replay report::the report for sample $SAMPLE could not be added to the summary"
```

  An assignment's status is its command substitution's, so a failed `tail` ends the chain before the group runs, and the group's `>>` redirection, which opens the summary file, is never reached: nothing is appended. (Codex ran the round-1 form, with the assignment after `cat`, and found it warns but leaves an open fence.)
- `uses: actions/upload-artifact@v7`. Its inputs `name`, `path` and `if-no-files-found` are unchanged (P10).

- [ ] **Step 4: Tests pass; full gate; commit and push** as `Name the first differing meta field; stop swallowing a failed tail; upload on Node 24`. The coordinator confirms on the first ready run that each report job's upload step succeeded on `@v7`.

---

### Task 10: Mutation checks

No files change. This task proves the new tests can fail. **The coordinator runs it, with a mutation harness in a separate worktree** that no reviewer or implementer is using; nothing in Tasks 1 to 9 applies or undoes a mutant. The harness sets `PYTHONDONTWRITEBYTECODE=1`, deletes every `__pycache__` before each mutant, confirms `git status --porcelain` is empty before and after each, applies each mutant and restores it in a `try`/`finally`, and runs **the whole suite, default and slow tests together (`uv run pytest -q -m ''`), for every mutant**, workflow and prose mutants included, in the foreground with a 600000 ms timeout.

| Mutation (what it replaces) | Tests that must fail, and no others |
|---|---|
| `_resolve_source`: the NUL check deleted | `test_an_encoded_nul_is_a_failure` |
| plot pairs: the content comparison deleted | `test_identical_light_and_dark_content` |
| `figure_failures`: the scan of unlisted pages deleted | `test_a_wrapper_on_an_unlisted_page` |
| `_page_failures`: an empty tuple skips the page | `test_a_listed_page_with_no_figures_must_have_no_wrappers` |
| plot pairs formed as the i-th light wrapper with the i-th dark one, not in document order | `test_wrappers_out_of_order` |
| `_style_violations`: the positional rule replaced by `main`'s count, `len(strings) > (2 if has_data else 0)` | `test_style_violations_reports_unstyled_calls[data_third_argument]` |
| `_is_plot_cell`: its body replaced by `"plt.subplots(" in cell or "plt.figure(" in cell` | `test_a_cell_is_a_plot_cell[renderings]`, `[show]`, `[figure_function]`, `[aliased]`, `[viz_module]`, `[qualified]`; `test_manifest_matches_the_source[story/beyond-point-masses.qmd]`, `[story/gravitational-echolocation.qmd]`. Report collection changes apart from failures: the two page cases of `test_builder_calls_and_show_are_inside_the_style_block` disappear; on Beyond Point Masses the old plot cell moves from `#2` to `#1`, so the contract IDs `…[story/beyond-point-masses.qmd#1]` are reused for a different cell and `#2` disappears; on Gravitational Echolocation the contract IDs `#2` disappear. List the missing IDs and the reused IDs |
| `_facade_builders`: import resolution replaced by round 1's matcher, any `Call` whose function is a `Name` or `Attribute` named `*_figure` | `test_a_cell_is_a_plot_cell[helper_figure]`, `[attribute_helper]`, `[unimported_figure]`, `[aliased]` (`draw` does not end in `_figure`); `test_facade_builders_are_resolved` |
| `_style_block_violations`: the `style(<loop variable>)` condition replaced by "any call to `style`" | `test_style_block_violations[fixed_theme]`, `[other_variable]` |
| `_style_block_violations`: the condition replaced by "`style` called with any name" | `test_style_block_violations[other_variable]` |
| `_style_block_violations`: a `with` in the loop's `orelse` accepted as well as one in its `body` | `test_style_block_violations[loop_else]` |
| `_style_block_violations`: the canonical-form check deleted | `test_style_block_violations[show_outside]`, `[two_blocks]`, `[commented_out_show]`, `[show_of_another_figure]`, `[reassigned_before_show]`, `[statement_between]`, `[show_of_the_call]` |
| `_style_block_violations`: "the next statement is `show(<name>)`" relaxed to "some later statement of the block is `show(<name>)`" | `test_style_block_violations[reassigned_before_show]`, `[statement_between]` |
| `_retired_asset_links` returns `[]` | `test_retired_asset_links_are_found`, all ten IDs |
| `_retired_asset_links`: `unquote` deleted | `test_retired_asset_links_are_found[encoded_basename]` |
| `_retired_asset_links`: `html.unescape` deleted | `test_retired_asset_links_are_found[html_entity]`, `[href_entity]` |
| `site/story/beyond-point-masses.qmd`: the old `![…](../assets/demo_density.png)` line put back beside the new cell | `test_no_page_links_to_a_retired_asset[site/story/beyond-point-masses.qmd]` |
| the density cell's `#\| renderings: [light, dark]` line deleted (an unthemed plot cell the old detection could not see) | `test_cell_declares_both_renderings[story/beyond-point-masses.qmd#1]` |
| the density cell's `show(fig)` line dedented four spaces, so it ends the `for` body after the `with` block | `test_builder_calls_and_show_are_inside_the_style_block[story/beyond-point-masses.qmd#1]` |
| `density_figure`: band percentiles 16 and 84 replaced by 5 and 95 | `test_median_and_band_are_pointwise_percentiles_of_the_last_row[light]`, `[dark]`; `test_density_figure_on_the_committed_archive`; `test_profile_limits` |
| `density_figure`: the profile band's `alpha` replaced by 0 | `test_median_and_band_are_pointwise_percentiles_of_the_last_row[light]`, `[dark]` |
| `density_figure`: a second `"true-profile"` line drawn on the `"trace-μ"` axes | `test_density_figure_size_and_axes[light]`, `[dark]` |
| `density_figure`: the clocks' x data replaced by the literal `[-6.0, -3.0, 0.0, 3.0, 6.0]` | `test_clocks_follow_the_archive` |
| `density_figure`: the clocks drawn with `clip_on=True` | `test_clocks_are_ink_squares_on_the_axis[light]`, `[dark]` |
| `density_figure`: the `"pointwise median and 68% band"` label deleted | `test_density_labels[light]`, `[dark]` |
| `trace_limits`: `[1:]` removed, so row 0 is included | `test_trace_limits[prior_row_left_out]`, `[clipped_to_prior]`, `[zero_width]` (P13); `test_traces_follow_the_archive[light]`, `[dark]` |
| the trace's truth line drawn at `mean[-1, j]` instead of the truth | `test_traces_follow_the_archive[light]`, `[dark]`; `test_changing_the_truth_moves_only_the_truth` |
| `trace_header` called with row 0 instead of row T | `test_trace_headers`; `test_density_figure_on_the_committed_archive` |
| `density_figure`: the median line's `color=colors.belief` replaced by `color="steelblue"` | `test_median_and_band_are_pointwise_percentiles_of_the_last_row[light]`, `[dark]`; `test_every_visible_colour_is_in_the_palette` in `tests/test_figures.py`, `[light]`, `[dark]`; `test_library_drawing_code_has_no_unstyled_colour[src/clocks/_figures.py]` |
| `range_study_figure`: counts from the position limit alone | `test_pass_counts_use_both_limits`; `test_limits_follow_the_study` (P13); `test_pass_counts_follow_each_tolerance[position_changed]`, `[mass_changed]` (P16) |
| `range_study_figure`: counts with strict `<` | `test_pass_counts_use_both_limits`; `test_pass_counts_follow_each_tolerance[position_changed]` (P16) |
| `range_study_figure`: counts from the records' `passed` flags | `test_pass_counts_use_both_limits`; `test_limits_follow_the_study`; `test_pass_counts_follow_each_tolerance[position_changed]`, `[mass_changed]` |
| `range_study_figure`: `study["tolerances"]["mass_error_max"]` replaced by `ECHO_PASS_MASS_TOL` (line, label and counts) | `test_limits_follow_the_study`; `test_pass_counts_follow_each_tolerance[mass_changed]` (P16) |
| `range_study_figure`: the position limit line drawn at `ECHO_PASS_POS_TOL`, its label and the counts still from the study | `test_limits_follow_the_study` |
| `range_study_figure`: the counts computed with `ECHO_PASS_POS_TOL` and `ECHO_PASS_MASS_TOL`, the lines and labels still from the study | `test_pass_counts_follow_each_tolerance[position_changed]`, `[mass_changed]` |
| `range_study_figure`: the spread line drawn from the median errors | `test_medians_and_spreads_come_from_summarize[light]`, `[dark]` |
| `range_study_figure`: the median-error line drawn with `linestyle="none"` | `test_medians_and_spreads_come_from_summarize[light]`, `[dark]` |
| `write_summary_figure`: `fig = range_study_figure(study, palette("light"))` moved before the `with style("light")` block, `save_png` left inside | `test_write_summary_figure_saves_the_light_range_study`, `test_write_summary_figure_output` |
| `write_summary_figure`: `save_png(plt.figure(figsize=(7.5, 5.4)), png_path)` inside the style block instead of the built figure | `test_write_summary_figure_saves_the_light_range_study` |
| `save_png`: `dpi=200` replaced by `dpi=100` | `test_save_png_writes_the_authored_canvas[plain]`, `[hostile]`; `test_density_command_prints_and_draws_the_recorded_run`; `test_write_summary_figure_output` |
| `save_png`: reduced to `fig.savefig(path, dpi=200, format="png")` | `test_save_png_writes_the_authored_canvas[hostile]` |
| `save_png`: only `facecolor=fig.get_facecolor()` deleted | `test_save_png_writes_the_authored_canvas[hostile]` (a magenta corner, P16); the wrapper saves inside `style("light")`, whose `savefig.facecolor` is the background, so its tests pass |
| `save_png`: `format="png"` deleted | `test_save_png_writes_the_authored_canvas[plain]`, `[hostile]`; `test_density_command_writes_png_bytes_at_the_path_given[no_suffix]`, `[misleading_suffix]` |
| `demo_density._run_demo`: `n_particles=N_PARTICLES` instead of `args.particles` | `test_density_command_prints_and_draws_the_recorded_run`, `test_density_command_prints_progress_from_the_replay`, `test_density_command_saves_the_figure_it_drew` |
| `demo_density._run_demo`: the progress loop deleted | `test_density_command_prints_progress_from_the_replay` |
| `demo_density._run_demo`: `save_png(fig, …)` replaced by `fig.savefig(args.output, dpi=200, format="png")` | `test_density_command_saves_the_figure_it_drew` |
| `demo_density._run_demo`: `save_png(plt.figure(figsize=(7.5, 5.4)), args.output)` instead of the built figure | `test_density_command_saves_the_figure_it_drew` |
| `site/assets/demo_density.png` restored from `main` | `test_retired_visuals_are_gone_but_recorded[demo_density.png]` |
| the calibration report's `demo_density.png` row given the range PNG's hash | `test_retired_visuals_are_gone_but_recorded[demo_density.png]` |
| the Reproducibility page keeps "and the density PNG" in the CI paragraph | `test_density_provenance_names_its_run` |
| the Reproducibility page's site-check bullet reverted to "the site or the library" | `test_site_check_description_names_the_figure_data` |
| `site-check.yml`: the `"assets/replays/**"` line deleted | `test_site_check_runs_when_the_figure_data_change[assets/replays/**]` |
| `site-check.yml`: a `push:` trigger added whose `paths:` list holds `"assets/replays/**"`, and that line deleted from `pull_request.paths` | `test_site_check_runs_when_the_figure_data_change[assets/replays/**]` |
| `FIRST_CHECKED_ROW` gains `"demo_1d": 1` | `test_a_shifted_prediction_is_caught[prior-demo_1d]`, `test_prior_cases_and_the_skip_map_partition_the_predicting_demos` |
| `_assert_predictions_match_dots` restored to `main`'s body (no zero-spread branch) | `test_a_one_particle_recording_passes_the_prediction_check[demo_1d]`, `[demo_multi_mass]`; `test_a_one_particle_prediction_off_the_dot_is_caught`; `test_a_spread_where_none_is_predicted_is_caught`; `test_a_mixed_row_names_the_original_clock` (P14). The committed archives have no zero spread (P15) and add no failure |
| `ZERO_SPREAD_ATOL = 1.0` | `test_a_one_particle_prediction_off_the_dot_is_caught`, `test_a_spread_where_none_is_predicted_is_caught` |
| the positive-spread check reports the clock's index within the positive-spread subset | `test_a_mixed_row_names_the_original_clock` |
| `first_difference` returns `"meta"` without the field | `test_meta_difference_is_reported_first`, `test_meta_difference_names_the_first_field_in_sorted_order`, `test_a_meta_key_missing_on_one_side_is_named[other_lacks]`, `[reference_lacks]`, `test_baselines_are_all_read_before_any_demo_is_recorded`, `test_a_difference_that_is_not_numeric_omits_the_size` |
| `first_difference` takes the first differing field in insertion order, not sorted | `test_meta_difference_names_the_first_field_in_sorted_order` |
| `first_difference` looks only at the reference's keys | `test_a_meta_key_missing_on_one_side_is_named[reference_lacks]` |
| the summary step in round 1's form, `last=$(...)` after `cat` inside the appended group | `test_a_failed_tail_leaves_the_summary_untouched` |
| the summary step in `main`'s form, `[ -z "$(tail -c1 replay-report.md)" ]` | `test_a_failed_tail_leaves_the_summary_untouched` |
| the upload step's `uses:` back to `actions/upload-artifact@v4` | `test_the_upload_runs_on_node_24` |

Several expectations rest on measurement or arithmetic, not on a run of this code: the position-only, strict and flag counts are indistinguishable from the true ones on the committed study (P4), so only the literal study catches them (P13); the strict-`<` count differs only at the default tolerances and at `position_changed`, where a literal run sits on the mass limit of 0.04; the `steelblue` mutant is caught by the role assertion, the palette walk and the source guard, three separate defences; `test_profile_limits` fails on the percentile mutant because its expected top edge comes from the test's own 84th percentile, not from the drawn band; a transparent band passes the palette walk (which skips alpha 0) and is caught only by the opacity assertion. Where a mutant does not depend on the theme, both theme IDs are named. Two changes have no mutant: the `PHI` spelling (Task 8 Step 1), which changes no result on this machine, and the run-time row choice in the two "is caught" tests, a rewrite of passing tests whose preconditions fail loudly if the archives change.

- [ ] **Step 1: Run each mutation** and record, for each, the exact set of failing test IDs.
- [ ] **Step 2: Report** the table of results, with test IDs that stopped being collected listed apart from failures. Where the failing set differs from the expected set, say which tests were extra or missing; do not change tests or code to make the table match. A difference is a finding for the coordinator.

---

### Hand-over

The coordinator opens the pull request as a draft and, in its description, gives: both contact sheets with Task 7's findings; the phone-width numbers of Task 7 Step 7; the local and runner Quarto versions; the mutation table of Task 10; and the command's changed run (P5). After review converges and the pull request is marked ready, the coordinator confirms on that head that CI's default and slow steps, the site check and all three replay-report jobs succeed, and that each report job's upload ran on `@v7`.
