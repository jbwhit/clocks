# Deep Clean and Website Design

**Date:** 2026-06-09
**Status:** Approved
**Goal:** Bring the repo from "polished tech demo" to publication quality, then ship a
GitHub Pages website in the style of `habitable-zone-black-holes` and `discoverneptune`.

## Context

A full review (2026-06-09) found the codebase healthy — 102 tests passing, ruff clean,
physics correct — with a short list of correctness, completeness, and hygiene issues.
The user chose a deep-clean-first sequencing: fix everything, consolidate `viz.py`,
then build the site. The site follows the static Quarto approach of the two reference
repos (no Pyodide/interactive scope).

## Phase 1 — Correctness fixes

Each lands as its own commit with tests.

1. **Log-evidence accumulation fix** (`src/clocks/inference.py`, `ParticleFilter.update`).
   `log_weights` starts from the normalized previous weights, so the per-step marginal
   likelihood estimate is `max_lw + log(sum(exp(log_weights)))`. The current code
   subtracts an extra `log(n_particles)`, biasing absolute log-evidence by
   `-T*log(N)`. The bias cancels in model comparison only because all filters share the
   same particle count. Fix: remove the extra term.
   Tests: (a) accumulated increment equals directly computed `log(sum(w_i * L_i))` on a
   small problem; (b) model comparison still identifies K=2 on the standard scenario.
2. **Document `jitter_std` semantics.** In `"fixed"` mode it is an absolute std; in
   `"covariance"` mode it scales the weighted posterior covariance (0.02 = 2% of the
   cloud's own scale). Update docstrings on `ParticleFilter` and `InferenceConfig`.
3. **Fail fast on empty observations.** `infer()` raises `ValueError` for an empty
   observation list in both fixed-K and model-comparison modes (currently: silent
   prior return vs `NameError`). Tests for both modes.
4. **NaN guard.** If all particles have zero weight after reweighting (e.g., every
   log-prior is `-inf`), raise `RuntimeError` with a message naming the cause instead
   of propagating NaN.

Explicitly out of scope: adding `converged()` output to `InferenceResult` (YAGNI — the
method is public on the filter itself).

## Phase 2 — viz consolidation and demo-script port

All five `animate_*` functions duplicate the 2×2 dashboard scaffolding (drive filter →
clear panels → redraw → `FuncAnimation` → save). Differences are only per-panel
renderers and labels.

1. **Extract one private animation driver** owning the `FuncAnimation`/save lifecycle;
   it takes per-frame panel-renderer callables. Each public `animate_*` becomes panel
   wiring (~30 lines).
2. **Public signatures unchanged.** All 15 viz exports in `__init__.py` keep working;
   call sites in tests and demos untouched.
3. **Split only if needed.** If consolidated `viz.py` still exceeds ~500 lines, split
   into `viz.py` (static primitives, re-exporting) plus private `_animate.py`. The
   import path `clocks.viz.animate_inference` stays valid either way.
4. **Expose `build_particle_filter(config: InferenceConfig) -> ParticleFilter`** in the
   public API (promote existing `_build_particle_filter`). Needed because animators
   drive the filter frame-by-frame and cannot consume `infer()`'s summarized result.
   All five demos need it (rule of three satisfied).
5. **Port all demo scripts** to `simulate(...)` + `build_particle_filter(...)`
   (`ModelComparison` for the comparison demo). Scripts shrink to config + animate
   call and become living documentation of the public API.
6. **`demo-density` gets a figure:** static 3-panel PNG (true vs inferred density
   profile, clock rates, convergence) for parity with the other demos.

## Phase 3 — Repo hygiene

1. **README refresh:** remove the hardcoded test count (stale twice already); add
   `api.py`, `config.py`, `results.py`, `test_api.py` to the project-structure
   listing; document `build_particle_filter`; add a site link once live.
2. **LICENSE:** MIT.
3. **Untracked files:**
   - Absorb `next.md` and the unimplemented `gemini-convo.md` ideas (MCMC
     rejuvenation, special-relativity velocity term, interactivity) into a committed
     `docs/someday-maybe.md`; delete both scratch files.
   - Commit `docs/superpowers/specs/2026-03-13-library-api-design-gemini-review.md`.
   - Add `.gemini/` to `.gitignore`.
4. **CI:** `ci.yml` unchanged; the Pages deploy workflow arrives in Phase 4.

## Phase 4 — Website

**Working title:** "GPS in Reverse" — subtitle: "Finding hidden masses with an array
of ticking clocks."

**Plumbing:** `site/` directory (avoids colliding with `docs/superpowers`).
`_quarto.yml` mirrors `habitable-zone-black-holes/site/_quarto.yml`: floating sidebar,
search, repo-actions, `page-navigation`, light/dark `custom.scss`/`custom-dark.scss`
adapted from blackholes, gruvbox highlight styles, `code-fold: true`, `code-copy`,
fonts include. Deploy via `.github/workflows/site.yml` to GitHub Pages, mirroring the
blackholes workflow.

**Figures policy:** pages import the `clocks` library and execute small fast cells at
render time (forward-model curves, potential wells, posterior snapshots — sub-second).
The five animated GIFs are pre-rendered assets copied into `site/assets/`, never
regenerated at build.

**Pages** (~11 total):

- `index.qmd` — hook: GPS satellites correct for general relativity to locate you;
  run that backwards and a clock array becomes a gravity detector. Hero GIF (2D demo)
  and a "read the idea in N pages" list, like blackholes.
- **Part 1 — The Story**
  1. Clocks as Gravimeters — time dilation, sqrt(1 + 2*Phi); real optical lattice
     clocks resolving centimeter height differences.
  2. One Clock Is Not Enough — the mass–distance degeneracy; why localization needs
     an array.
  3. The Search in One Dimension — 1D demo; the particle filter as detective.
  4. Into the Plane — 2D demo.
  5. Two Hidden Masses — multi-mass inference; label-switching/co-location degeneracy
     discussed honestly (the x2 = 4.5 story from git history).
  6. How Many Masses? — Bayesian model comparison, Occam's razor, the evidence
     (made trustworthy by the Phase 1 fix).
  7. Beyond Point Masses — Gaussian density profile, using the new Phase 2 figure.
- **Part 2 — Under the Hood**
  - The Particle Filter — SMC mechanics: weights, resampling, jitter, evidence.
  - Units and Scales — what M = 0.8 means when G = c = 1; mapping the toy to reality.
- **Part 3 — Reproduce**
  - Getting Started — uv sync, run the demos, use the library API.
  - Reproducibility — seeds, test suite, CI.

## Sequencing and verification

Phases land in order 1 → 2 → 3 → 4; each phase is a set of small commits, pushed as
completed. Verification gates: full test suite plus ruff after every commit; for
Phase 2, regenerate at least one GIF and visually confirm parity; for Phase 4,
`quarto render` must succeed locally and the deployed site must serve all pages and
GIFs.
