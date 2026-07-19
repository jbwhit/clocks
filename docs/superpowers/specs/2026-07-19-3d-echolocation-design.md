# 3D Gravitational Echolocation — Demo, Range Study, and Site Page

**Date:** 2026-07-19
**Status:** Draft — Codex xhigh round 1 returned NEEDS REVISION; round-1
fixes applied; round 2 returned NEEDS REVISION (1 major, 2 minor); round-2
fixes applied, awaiting round 3.
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

Decisions made during brainstorming and review:

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
- **Measurement model:** differential (mean-centered) clock rates, not
  absolute rates (Codex round 1, blocker). A self-contained head has no
  external time reference; only relative rates among its own clocks are
  observable. See §1a.

## Design

### 1. Scenario — `run_echolocation_3d` in `src/clocks/_scenarios.py`

The shared scenario used by demo, study, and acceptance test, alongside
`run_multi_mass_2d`:

- **The head:** a 3×3×3 cubic lattice of 27 clocks, spacing 1.0, centered on
  the origin (`track_offset=0`). Positions span ±1.0 per axis; the
  **circumradius is `R_head = sqrt(3)` ≈ 1.73** (center to corner clocks).
  Range is measured in units of `R_head`. Fixed geometry everywhere — it is
  the character in the story.
- **The world:** a single exterior point mass (K = 1) at distance
  `range_r × R_head` from the lattice center, along a fixed, documented,
  off-axis unit direction (not axis- or diagonal-aligned, so no projection
  or symmetry hides it). `range_r` is the scenario's main parameter: the
  study sweeps it, the demo fixes one mid-range value.
- **Physical validity (fail fast):** the forward model clamps
  `1 + 2Φ ≤ 0` (black-hole regime), and the weak-field story requires
  staying far from it. The scenario computes the minimum clock–mass
  distance `d_min` (≥ `range_r × R_head − R_head`) and raises unless
  `d_min ≥ 10 × M_true` (i.e. `2M/r ≤ 0.2` at the nearest clock). With the
  starting defaults below (`M_true = 0.15`, minimum swept range
  `2 × R_head`): `d_min ≥ R_head ≈ 1.73 > 10 × 0.15 = 1.5`, ~15% margin.
  The constraint is validated at construction, not assumed.
- **Starting parameter defaults** (frozen during the tuning phase, §3a):
  `M_true = 0.15`, `observation_std = 0.005`, `n_observations = 80`,
  `n_particles = 6000`. Scales are informed by a review-time feasibility
  probe (Codex round 1: `M = 0.2` recovered at small ranges, degenerate at
  large ranges under absolute rates); the centered model (§1a) is strictly
  less informative and the probe's close-range mass violated the weak-field
  constraint, so tuning may adjust these — subject to the constraint above,
  on tuning seeds only, never on certification seeds.
- **Inference state:** (x, y, z, M) — single-mass filter, shipped annealed
  jitter defaults (`jitter_tau=15.0`, mode `"annealed"`,
  `jitter_std=0.02` floor).
- **Filter construction — raw `ParticleFilter`, not `infer()`:** the public
  `InferenceConfig` cannot express a centered measurement model or a
  per-axis prior box, and it deliberately stays that way (no core API
  changes). The scenario constructs `ParticleFilter` directly — the
  precedented path (`ModelComparison` does exactly this, and the README
  documents `build_particle_filter` for observation-by-observation use) —
  with:
  - a prior sampler over a position box that contains the full swept range
    in every axis (deliberately mostly *outside* the head) and a mass range
    `[M_lo, M_hi] = [0.05, 2.0]`;
  - a custom `log_prior` enforcing the position box **and the full mass
    range `[M_lo, M_hi]`** (unlike the public API's masses-positive-only
    prior), so the mass–distance degeneracy at far range shows up as
    posterior widening *within* the prior box rather than runaway mass
    growth;
  - per-parameter `support_bounds` **identical to the log-prior support**
    (position box; mass `[M_lo, M_hi]` — not the API's
    `[nextafter(0,1), inf)` convention), since reflected annealed jitter
    moves particles anywhere inside `support_bounds` and the filter
    requires reflection to land inside the log-prior's support;
  - the centered forward model of §1a. No `constraint_fn` (K = 1).
- **Return type — `EchoRunResult` (new TypedDict):** the multi-mass-2D
  `RunResult` fields don't decompose position vs mass, which is the whole
  point here. Fields: `seed`, `range_r`, `position_error` (Euclidean),
  `mass_error` (absolute), `pos_std` (Euclidean norm of the three position
  stds), `mass_std`, `covered_3sigma` (per-parameter, as in `RunResult`),
  `residual_over_noise` (computed on *centered* rates), `passed` (gate for
  the acceptance test, §5).

### 1a. Measurement model — differential rates

An exterior mass presents the lattice with a nearly uniform potential
offset plus a small differential signal across the clocks. The existing
pipeline observes **absolute** dilation factors, in which the uniform
offset (∝ M/R) is measurable and would let the filter infer range from
offset + gradient — physically equivalent to assuming the head is
calibrated against a distant reference clock, which contradicts the
premise. The scenario therefore centers both data and predictions:

- **Observations:** simulate absolute rates via `simulate()` as usual, then
  subtract each observation's across-clock mean before feeding the filter
  (`Observation(rates=rates - rates.mean())`).
- **Predictions:** the scenario's `forward_model_batch` wraps
  `clock_rates_batch` and subtracts each particle's across-clock mean
  (row-wise). The scalar `forward_model` does the same.
- **Likelihood approximation (documented in code and on the page):**
  centering noisy iid Gaussian rates yields noise with covariance
  `σ²(I − 11ᵀ/N)` — variance `σ²(1 − 1/N)` and pairwise correlation
  `−1/(N−1)`. The scenario keeps the existing iid Gaussian likelihood at
  std `σ`. This is benign: for centered residuals the iid quadratic form
  equals the projected-Gaussian quadratic form up to a
  parameter-independent constant, so particle *weights* are unaffected;
  only the absolute likelihood normalization (and hence log-evidence,
  unused here) differs. Noted in code, not modeled. No changes to
  `noise.py`.
- **Physics consequence (the page's central explanation):** after
  centering, the leading observable is the potential *gradient* across the
  head (differential spread ∼ 2·M·a/R² for lattice half-width a), and
  range–mass separation rests on the far weaker curvature term (∼ M/R³).
  This is the honest version of the mass–distance degeneracy: direction is
  cheap, range is expensive.

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
  uncertainty band — reuses the existing convergence panel), mass-estimate
  histogram vs truth (reuses `plot_mass_histogram`), and a **new
  rates-by-clock-index panel** in `_panels3d`: centered observed vs
  predicted rate per clock index. The existing `plot_clock_rates` plots
  against `positions[:, 0]` and would collapse the lattice's 9 clocks per
  x-plane onto shared abscissae; it is not reused for this panel.
- **Animation** reuses `_animate.py` machinery, including the
  precomputed-filter-states pattern (frame-0 fix); new parts are the 2:1
  GridSpec and per-frame camera azimuth. The demo drives the scenario's
  raw `ParticleFilter` observation-by-observation (the documented custom-
  animation path).
- **Entry point:** `scripts/demo_echolocation_3d.py`, wired as
  `uv run demo-echolocation-3d` via `_cli.py` + `pyproject.toml`, writing
  `output/demo_echolocation_3d.gif`.
- **Demo seed policy:** the demo's single seed may be curated for visual
  clarity; the page and README must disclose that. The study carries the
  quantitative argument.

### 3. Resolution-vs-range study

- **Harness:** `scripts/scan_echolocation_range.py`, modeled on
  `scan_multi_mass_2d.py` — multiprocessing `Pool`, CLI flags (`--ranges`,
  `--seed-block`, `--workers`, `--per-run`), per-cell table on stdout.
- **Sweep:** `range_r` over ~2–8 circumradii (log-ish spacing, ~6 values;
  defaults frozen during tuning) × 12 seeds per range. ~72 runs per sweep,
  comparable to the annealed-jitter tuning grid.
- **SNR sanity gate (before any tuning):** the harness prints, for each
  swept range, the noise-free centered signal magnitude
  (`max |centered rates|`) against `observation_std`. If the far end of the
  sweep is more than ~10× below the noise floor per observation, the sweep
  bounds are adjusted before burning any runs. This table also feeds the
  page's falloff figure.
- **Metrics per run** (from `EchoRunResult`): final position error, final
  mass error, final posterior stds. The story is all of these vs range:
  where error grows, whether the filter *knows* it is uncertain (honest
  posterior widening) or is confidently wrong, and where the mass–distance
  degeneracy takes over.
- **Outputs:** stdout table; JSON results file in `output/`; summary figure
  `output/echolocation_range_study.png` (median + per-seed points, position
  and mass/uncertainty as two aligned subplots), generated *from the JSON*
  so it can be restyled without re-sweeping.

### 3a. Seed protocol (tuning vs certification)

Copied from the annealed-jitter design's protocol, which exists precisely
to avoid certifying on the data used to choose the setup:

- **Tuning seeds 0–11** (per range): used to freeze scenario parameters
  (`M_true`, noise, observation count, particle count, sweep bounds) and to
  derive the acceptance-test thresholds. May be run as often as needed.
- **Certification seeds 300–311** (per range): run **exactly once** after
  everything is frozen. The published study figure, the page's quantitative
  claims ("usable range ≈ N circumradii"), and the acceptance test's
  recorded pass counts all come from this certification sweep. If the
  certification run fails its gates, the block is burned: the failure is
  diagnosed and documented in the spec, parameters are re-frozen on tuning
  seeds, and the next block (400–411) is used. (300s chosen to avoid
  collision with the multi-mass-2D convention: 0–11 tuning, 100s burned,
  200s certification.)
- **Operationally:** the harness takes `--seed-block N` (seeds N…N+11;
  default 0 = the tuning block; certification is `--seed-block 300`). The
  JSON output records the seed block used, and any burned block is recorded
  in this spec's status history — so which seeds produced which artifact is
  never ambient knowledge.

### 4. Site page — `site/story/gravitational-echolocation.qmd`

Added to the Part 1 sidebar after "Beyond Point Masses" as the story's
capstone (every earlier page keeps the mass inside the array; this one
pushes it outside). Structure:

1. **Hook** (2–3 paragraphs): the born-with-a-clock-lattice premise, then
   the pivot to the site's neutral voice: this is a well-posed question and
   we have the machinery to answer it. The differential measurement model
   is introduced *here*, as part of the premise: the head has no outside
   reference; it can only compare its own clocks to each other.
2. **The physics of an exterior mass:** offset-vs-gradient explanation and
   why centering removes the offset, with one fast in-page code cell
   plotting the centered differential-signal falloff vs range against the
   noise floor (pure forward-model evaluation, no inference — fits the
   fast-cell constraint; same computation as the harness's SNR gate).
3. **The demo:** the rotating hero GIF at one mid-range distance, narrated
   in the style of the other story pages.
4. **The study:** the certification-sweep figure; interpretation of error
   growth, posterior honesty, and degeneracy onset; a plain-language
   answer: at this noise floor, the sense has a usable range of ~N
   circumradii (N filled in from the certification sweep).
5. **Verdict:** is the sci-fi sense physically coherent? Answered with the
   study's numbers, tying back to the mass-distance degeneracy thread.

**Supporting updates:** README demo entry (with the differential-model
one-liner); `docs/someday-maybe.md` marks the item shipped with a pointer to
this spec; reproduce pages list `demo-echolocation-3d` and the scan command;
sidebar entry in `_quarto.yml`.

**Asset policy** (matching the annealed-jitter artifact policy): artifacts
are generated into `output/`, copied byte-identically to `assets/` (README)
and `site/assets/` (site), and the reproduce page documents the one-command
regeneration for each. The Quarto site fully recomputes cells per build and
assumes fast cells, so the sweep figure and GIF are pre-generated; only the
falloff cell (pure forward model) runs in-page.

### 5. Testing and error handling

- **Unit tests (fast):**
  - Scenario: lattice builder yields 27 clocks at expected positions with
    circumradius `sqrt(3)`; mass at requested range along the documented
    direction; the weak-field validation rejects a too-close/too-heavy
    configuration; `EchoRunResult` fields populate.
  - Measurement model: centered forward model output has zero row-mean;
    centered observations preserve the differential signal (centering a
    constant-offset rate vector yields zeros).
  - **3D core coverage (new):** `clock_rates_batch` agrees with looped
    `clock_rates` on random 3D single-mass configurations (batch
    equivalence), and a fast small-N 3D API recovery test (close-range
    mass, loose tolerance) pins that `(x, y, z, M)` inference works
    end-to-end at all.
  - Viz: smoke tests for `_panels3d` mirroring `test_viz.py` — dashboard
    builds with expected axes; a single frame renders without error (no GIF
    encoding in fast tests); rates panel indexes clocks 0–26.
  - Scan: pure helpers (results→JSON round-trip, median/summary
    computation, SNR table) tested without running inference.
- **Acceptance test (slow marker):**
  `tests/test_acceptance_echolocation_3d.py`, following
  `test_acceptance_multi_mass_2d.py`; pins the qualitative result on the
  certification seeds:
  - at the closest swept range the filter localizes within tolerance on
    ≥ 10/12 seeds;
  - at the farthest swept range the position posterior std is materially
    larger than at the closest range (honest-uncertainty property).
  Thresholds are frozen from the tuning sweep (§3a) before the
  certification run and never adjusted afterward. Run via
  `uv run pytest -m slow`.
- **Fail-fast validation:** scenario rejects `range_r < 2` (in
  circumradii; exterior means exterior, with clearance) and any
  configuration violating the weak-field constraint (§1); scan CLI
  validates its arguments. No new validation inside inner library code
  (house rule: validate at boundaries).
- **No core library changes:** physics, inference, noise, and the public
  API are untouched; the work is additive (new scenario, new viz module,
  new scripts, new page, new tests) and the scenario composes existing
  `ParticleFilter` hooks (`forward_model_batch`, `log_prior`,
  `support_bounds`). Any genuine 3D core bug found during implementation
  gets its own fix + regression test, not a demo-side workaround.

## Acceptance criteria

1. `uv run demo-echolocation-3d` produces a rotating-camera GIF in which the
   particle cloud visibly converges on the exterior mass; file size in line
   with existing demo GIFs.
2. `uv run scripts/scan_echolocation_range.py` completes the default sweep
   and writes table + JSON + PNG; the certification sweep shows a clear
   error-vs-range trend with honest posterior widening.
3. New site page renders in the Quarto build with GIF, falloff cell, and
   study figure; sidebar and cross-links updated.
4. Fast test suite passes; slow acceptance test passes on the frozen
   thresholds against the certification seeds; `uv run ruff format --check .`
   and `uv run ruff check .` clean.
5. Existing demos and tests unaffected.

## Out of scope (deferred, tracked in someday-maybe)

- Lattice-geometry comparisons (field of view per clock).
- Angular-resolution measurement.
- Interactive/Pyodide in-browser inference.
- Multiple exterior masses or moving masses.
- Exact centered-noise likelihood (projected covariance) — the iid
  approximation is documented instead.
- Exterior-only prior support carve-outs (the box prior includes the head
  interior; the data excludes it).
