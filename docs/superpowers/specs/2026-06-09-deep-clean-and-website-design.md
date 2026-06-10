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
   Tests: (a) deterministic exact-evidence test with resampling disabled
   (`resample_threshold=0`) and at least two observations, asserting each per-step
   increment equals `log(sum(prev_weights * likelihoods))` computed directly — the
   second step pins the nonuniform-previous-weights case the bias claim is actually
   about; (b) model comparison still identifies K=2 on the standard scenario.
2. **Document `jitter_std` semantics.** In `"fixed"` mode it is an absolute std; in
   `"covariance"` mode it scales the weighted posterior covariance (0.02 = 2% of the
   cloud's own scale). Update docstrings on `ParticleFilter` and `InferenceConfig`.
3. **Fail fast on empty observations.** `infer()` raises `ValueError` for an empty
   observation list, guarded at the top of `infer()` before mode dispatch (currently:
   silent prior return in fixed-K mode vs `UnboundLocalError` in model-comparison
   mode). Tests for both modes.
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
   it takes per-frame panel-renderer callables. The driver must be generic over figure
   and axes setup and support per-frame cleanup hooks, because the variants differ:
   `animate_model_comparison` uses a 1x2 layout (not the 2x2 dashboard) and the 2D
   animations manage colorbar lifecycle. If the comparison animation does not fit the
   shared driver cleanly, it keeps a small separate driver rather than forcing the
   abstraction. Each public `animate_*` becomes panel wiring (~30 lines).
   While in there: `animate_model_comparison` currently assumes contiguous
   `K=1..k_max`; change it to iterate `sorted(model_comparison.filters)` and add a
   test with `k_values=(2, 3)`.
2. **Public signatures unchanged.** All 15 viz exports in `__init__.py` keep working;
   call sites in tests and demos untouched.
3. **Split only if needed.** If consolidated `viz.py` still exceeds ~500 lines, split
   into `viz.py` (static primitives, re-exporting) plus private `_animate.py`. The
   import path `clocks.viz.animate_inference` stays valid either way.
4. **Expose `build_particle_filter(config: InferenceConfig) -> ParticleFilter`** in the
   public API (promote existing `_build_particle_filter`). Needed because animators
   drive the filter frame-by-frame and cannot consume `infer()`'s summarized result.
   The four fixed-K point-mass demos need it (rule of three satisfied).
5. **Port the demo scripts** to the public API, each to the entry point that fits:
   the four fixed-K point-mass animation demos (`demo_1d`, `demo_2d`,
   `demo_multi_mass`, `demo_multi_mass_2d`) use `simulate(...)` +
   `build_particle_filter(...)`; `demo_model_comparison` uses `simulate(...)` +
   `ModelComparison` (already public); `demo_density` keeps its custom Gaussian
   density forward model (it cannot use `InferenceConfig`'s point-mass builder) and
   uses `simulate`-style data generation only where it applies. Scripts shrink to
   config + animate call and become living documentation of the public API.
6. **`demo-density` gets a figure:** static 3-panel PNG (true vs inferred density
   profile, clock rates, convergence) for parity with the other demos.

## Phase 3 — Repo hygiene

1. **README refresh:** remove the hardcoded test count (stale twice already); add
   `api.py`, `config.py`, `results.py`, `test_api.py` to the project-structure
   listing; document `build_particle_filter`; add a site link once live.
2. **LICENSE and package metadata:** add MIT `LICENSE`; complete PEP 621 metadata in
   `pyproject.toml` — `readme = "README.md"`, `license = "MIT"`, and `[project.urls]`
   with the repo and (once live) site URLs.
3. **Untracked files:**
   - Absorb `next.md` and the unimplemented `gemini-convo.md` ideas (MCMC
     rejuvenation, special-relativity velocity term, interactivity) into a committed
     `docs/someday-maybe.md`; delete both scratch files.
   - Commit `docs/superpowers/specs/2026-03-13-library-api-design-gemini-review.md`.
   - Add `.gemini/` to `.gitignore`.
4. **CI:** `ci.yml` unchanged; the Pages deploy workflow arrives in Phase 4.
5. **Lockfile:** the working tree carries an uncommitted `uv.lock` delta (an
   `[options] exclude-newer` block injected by the user's global uv supply-chain
   policy; it will reappear on every uv invocation). Commit it deliberately together
   with the Phase 4 Jupyter dependency addition rather than letting it ride along in
   an unrelated commit.

## Phase 4 — Website

**Working title:** "GPS in Reverse" — subtitle: "Finding hidden masses with an array
of ticking clocks."

**Plumbing:** `site/` directory (avoids colliding with `docs/superpowers`).
`_quarto.yml` mirrors `habitable-zone-black-holes/site/_quarto.yml`: floating sidebar,
search, repo-actions, `page-navigation`, light/dark `custom.scss`/`custom-dark.scss`
adapted from blackholes, gruvbox highlight styles, `code-fold: true`, `code-copy`,
fonts include. Deploy via `.github/workflows/site.yml` to GitHub Pages, mirroring the
blackholes workflow; the repo's GitHub Pages setting must be switched to "deploy from
GitHub Actions".

**Dependencies:** executed Quarto cells need a Jupyter kernel — add `jupyter` (or
`ipykernel`) to the dev dependency group and commit the regenerated `uv.lock` (which
also absorbs the pending `[options]` block, see Phase 3).

**Concrete file deliverables** (mirroring blackholes): `site/_quarto.yml`,
`site/index.qmd`, the Part 1/2/3 pages below, `site/custom.scss`,
`site/custom-dark.scss`, `site/styles.css`, `site/includes/fonts.html`,
`site/assets/favicon.ico`, the five demo GIFs copied to `site/assets/`, and
`.github/workflows/site.yml`. Render globs in `_quarto.yml` must cover every page
directory used.

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
