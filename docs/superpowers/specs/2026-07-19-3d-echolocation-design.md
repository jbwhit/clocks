# 3D Gravitational Echolocation — Demo, Range Study, and Site Page

**Date:** 2026-07-19
**Status:** Draft — awaiting Codex xhigh review
**Origin:** `docs/someday-maybe.md` § "3D and exterior masses: gravitational
echolocation"

## Motivation

Every existing demo and story page keeps the hidden mass *inside* (or level
with) the clock array. This project pushes it outside, in 3D: a fixed lattice
of clocks — the sci-fi framing is a person born with atomic clocks embedded
in their head — senses the mass distribution of its surroundings the way
echolocation senses surfaces. The library's physics and inference are already
dimension-agnostic; what's missing is a 3D scenario, 3D visualization, a
quantitative answer to "how far can this sense reach?", and a site page that
tells the story.

## Scope (this round)

1. A 3D exterior-mass demo producing a rotating-camera animated GIF.
2. A resolution-vs-range study quantifying how inference degrades with
   distance, with a pre-generated summary figure.
3. A new site story page ("Gravitational Echolocation") using both, framed
   with a sci-fi hook and the site's neutral scientific voice for the body.

Decisions made during brainstorming:

- **Presentation:** animated GIF with a slowly rotating camera — consistent
  with the site's deliberate static-GIF scope choice
  (`2026-06-09-deep-clean-and-website-design.md`). No interactive 3D, no
  Pyodide.
- **Study scope:** resolution vs range only. Lattice-geometry comparisons and
  angular-resolution measurement are deferred (remain in someday-maybe).
- **Framing:** sci-fi hook, science body. The page opens with the premise,
  runs the analysis in the site's existing voice, and closes by answering
  whether the sense is physically coherent.
- **Dashboard layout:** "hero" layout — large rotating 3D scene (~2/3 width)
  with a stacked side column of diagnostics (selected via visual mockup
  comparison).

## Design

### 1. Scenario — `run_echolocation_3d` in `src/clocks/_scenarios.py`

The shared scenario used by demo, study, and acceptance test, alongside
`run_multi_mass_2d`:

- **The head:** a 3×3×3 cubic lattice of 27 clocks, spacing 1.0, centered on
  the origin (half-width 1.0, so the "head" has radius ≈ 1 simulation unit to
  face centers). Fixed geometry everywhere — it is the character in the
  story.
- **The world:** a single exterior point mass (K = 1) at distance `range_r`
  from the lattice center (units of head half-width ≈ head-radii), along a
  fixed, documented, off-axis direction so no projection or panel hides it.
  `range_r` is the scenario's main parameter: the study sweeps it, the demo
  fixes one mid-range value.
- **Inference state:** (x, y, z, M) — single-mass filter, shipped annealed
  jitter defaults (`jitter_tau=15.0`, mode `"annealed"`). The prior position
  box extends beyond the maximum swept range in every axis — deliberately
  mostly *outside* the head. Particle count, noise floor (`observation_std`),
  and observation count are tuning parameters fixed during implementation
  against the acceptance criteria below, then frozen in the scenario.
- **Return type:** the same `RunResult` shape as the 2D scenario (errors,
  coverage, residuals), so scan-harness reporting and acceptance-test
  patterns carry over.

**Physics note (also the page's central explanation):** an exterior mass
presents the lattice with a nearly uniform potential offset plus a small
differential gradient across the clocks. Only the differential part localizes
the mass, and it falls off much faster with range than the offset — this is
the mechanism the study measures, closely related to the known mass-distance
degeneracy.

### 2. Demo and 3D visualization

- **New module `src/clocks/_panels3d.py`**, exposed through the `viz.py`
  facade. `_panels.py` (366 lines) keeps its 1D/2D focus; 3D scene
  composition, camera control, and depth-handled scatter are a separate
  concern.
- **Hero scene panel** (single `projection='3d'` axes): the 27-clock lattice
  (small markers plus a translucent wireframe cube so the head reads as an
  object), the true exterior mass (star marker), and the particle cloud
  (alpha scatter, subsampled if needed for GIF size/render time). Camera:
  fixed elevation, azimuth advancing a constant step per frame — at most one
  slow orbit over the full animation.
- **Side column** (top to bottom): convergence history (position error with
  uncertainty band), mass-estimate histogram vs truth, observed clock rates.
  Reuse existing panel functions unchanged where possible.
- **Animation** reuses `_animate.py` machinery, including the
  precomputed-filter-states pattern (frame-0 fix); new parts are the 2:1
  GridSpec and per-frame camera azimuth.
- **Entry point:** `scripts/demo_echolocation_3d.py`, wired as
  `uv run demo-echolocation-3d` via `_cli.py` + `pyproject.toml`, writing
  `output/demo_echolocation_3d.gif`, copied to `assets/` and the site like
  the other demos.
- **Demo seed policy:** the demo's single seed may be curated for visual
  clarity; the page and README must disclose that. The study carries the
  quantitative argument.

### 3. Resolution-vs-range study

- **Harness:** `scripts/scan_echolocation_range.py`, modeled on
  `scan_multi_mass_2d.py` — multiprocessing `Pool`, CLI flags (`--ranges`,
  `--seeds`, `--workers`), per-cell table on stdout.
- **Sweep:** `range_r` over ~2–12 head-radii (log-ish spacing, ~7 values) ×
  12 seeds per range (seeds 0–11, fixed and documented; not cherry-picked)
  on the shared scenario. ~84 runs, comparable to the annealed-jitter tuning
  grid.
- **Metrics per run:** final position error, final mass error, final
  posterior std (the filter's claimed uncertainty). The story is all three
  vs range: where error grows, whether the filter *knows* it is uncertain
  (honest posterior widening) or is confidently wrong, and where the
  mass-distance degeneracy takes over.
- **Outputs:** stdout table; JSON results file in `output/`; summary figure
  `output/echolocation_range_study.png` (median + per-seed points, position
  and mass/uncertainty as two aligned subplots), generated *from the JSON*
  so it can be restyled without re-sweeping; PNG copied to `assets/` and the
  site.
- **Why pre-generated:** the Quarto site fully recomputes all cells per
  build and assumes fast cells; an 84-run sweep cannot live in a page cell.
  The page embeds the PNG; the reproduce section documents one-command
  regeneration — consistent with how GIFs are handled.

### 4. Site page — `site/story/gravitational-echolocation.qmd`

Added to the Part 1 sidebar after "Beyond Point Masses" as the story's
capstone (every earlier page keeps the mass inside the array; this one
pushes it outside). Structure:

1. **Hook** (2–3 paragraphs): the born-with-a-clock-lattice premise, then
   the pivot to the site's neutral voice: this is a well-posed question and
   we have the machinery to answer it.
2. **The physics of an exterior mass:** offset-vs-gradient explanation, with
   one fast in-page code cell plotting the differential-signal falloff (pure
   forward-model evaluation, no inference — fits the fast-cell constraint).
3. **The demo:** the rotating hero GIF at one mid-range distance, narrated
   in the style of the other story pages.
4. **The study:** the pre-generated range-study figure; interpretation of
   error growth, posterior honesty, and degeneracy onset; a plain-language
   answer: at this noise floor, the sense has a usable range of ~N
   head-radii (N filled in from the actual sweep).
5. **Verdict:** is the sci-fi sense physically coherent? Answered with the
   study's numbers, tying back to the mass-distance degeneracy thread.

**Supporting updates:** README demo entry; `docs/someday-maybe.md` marks the
item shipped with a pointer to this spec; reproduce pages list
`demo-echolocation-3d` and the scan command; sidebar entry in `_quarto.yml`.

### 5. Testing and error handling

- **Unit tests (fast):**
  - Scenario: lattice builder yields 27 clocks at expected positions; mass
    at requested range along the documented direction; `RunResult` fields
    populate.
  - Viz: smoke tests for `_panels3d` mirroring `test_viz.py` — dashboard
    builds with expected axes; a single frame renders without error (no GIF
    encoding in fast tests).
  - Scan: pure helpers (results→JSON round-trip, median/summary computation)
    tested without running inference.
- **Acceptance test (slow marker):**
  `tests/test_acceptance_echolocation_3d.py`, following
  `test_acceptance_multi_mass_2d.py`; pins the qualitative result on fixed
  seeds:
  - at close range (~2 head-radii) the filter localizes within tolerance on
    ≥ 10/12 seeds;
  - at far range the posterior std is materially larger than at close range
    (honest-uncertainty property).
  Exact thresholds are finalized from the first real sweep during
  implementation, then frozen. Run via `uv run pytest -m slow`.
- **Fail-fast validation:** scenario rejects `range_r` inside the lattice
  (exterior means exterior) and a prior box that does not contain the true
  mass position; scan CLI validates its arguments. No new validation inside
  inner library code (house rule: validate at boundaries).
- **No core changes expected:** physics and inference are already
  dimension-agnostic; the work is additive (new scenario, new viz module,
  new scripts, new page, new tests). Any genuine 3D core bug found during
  implementation gets its own fix + regression test, not a demo-side
  workaround.

## Acceptance criteria

1. `uv run demo-echolocation-3d` produces a rotating-camera GIF in which the
   particle cloud visibly converges on the exterior mass; file size in line
   with existing demo GIFs.
2. `uv run scripts/scan_echolocation_range.py` completes the default sweep
   and writes table + JSON + PNG; results show a clear error-vs-range trend.
3. New site page renders in the Quarto build with GIF, falloff cell, and
   study figure; sidebar and cross-links updated.
4. Fast test suite passes; slow acceptance test passes on the frozen
   thresholds; `uv run ruff format --check .` and `uv run ruff check .`
   clean.
5. Existing demos and tests unaffected.

## Out of scope (deferred, tracked in someday-maybe)

- Lattice-geometry comparisons (field of view per clock).
- Angular-resolution measurement.
- Interactive/Pyodide in-browser inference.
- Multiple exterior masses or moving masses.
