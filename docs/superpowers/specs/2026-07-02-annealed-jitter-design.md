# Annealed Jitter for the Particle Filter

**Date:** 2026-07-02
**Status:** Draft — pending Codex xhigh review.
**Goal:** Fix the multi-mass-2D premature-collapse failure (at best 7/12 seeds
recover truth under tested fixed jitters) by annealing the post-resampling
jitter from prior scale down to a floor, and make the annealed mode the
library default. Acceptance: ≥ 10/12 seeds pass on the multi-mass-2D scan.

## Context

The 6-parameter multi-mass 2D problem (x1, y1, x2, y2, M1, M2) does not
reliably recover truth under fixed jitter. A 36-run scan (2026-06, during
PR #4 verification) found jitter_std 0.02/0.05/0.10 pass 1/12, 5/12, and
7/12 seeds respectively; the demo ships 0.05 with seed 11. The failure is
premature collapse: the cloud confidently locks onto a mode whose residuals
are ~7× the observation noise. A fixed jitter large enough to escape wrong
modes late is too disruptive to converge; a small one freezes early.

The June scan harness was ad-hoc and never committed, and its pass
criterion was never written down. This work therefore includes building the
harness (script + slow test) and defining the pass rule, not just the new
jitter mode.

Scope boundary (from `docs/someday-maybe.md`): annealed/floored jitter
only. Likelihood tempering and MCMC resample-move (rejuvenation) stay out
of scope. Combining annealing with the `covariance` jitter mode is also out
of scope.

## Design

### 1. Mechanism

New jitter mode `"annealed"` alongside `"fixed"` and `"covariance"`. At
each resample, the per-parameter jitter standard deviation is

```
sigma_t = floor + (init − floor) · exp(−t / tau)
```

- `t` — the 0-based index of the observation being processed when the
  resample fires (i.e., the state's `observations_seen` value before this
  update increments it). A resample during the very first observation uses
  t = 0, so sigma starts exactly at the initial cloud scale.
- `init` — per-parameter std of the **initial** particle cloud, captured
  once at construction. The initial cloud is a prior sample, so this is
  prior scale without plumbing prior bounds into `ParticleFilter`.
- Clamp: `init = max(init, floor)` per parameter, so the schedule never
  anneals upward (guards degenerate/tight priors and floor > prior-scale
  configurations).
- `floor` — the existing `jitter_std` parameter. It keeps its "absolute
  standard deviation" meaning; it is now the late-run asymptote.
- `tau` — new knob `jitter_tau`, in units of observations; must be > 0.

The jitter draw is per-parameter isotropic Gaussian (a vector of stds
broadcast over particles), i.e. the `fixed` branch generalized to a
scheduled vector scale. The `fixed` and `covariance` branches are
unchanged.

### 2. API and plumbing

- `ParticleFilter.__init__`: accept `jitter="annealed"` (new **default**)
  and `jitter_tau: float` (default = the scan-selected value, see §3).
  Validate `jitter_tau > 0` unconditionally (fail fast, regardless of
  mode). Capture the
  clamped `init` vector after drawing the initial cloud.
- `ParticleFilter._resample`: add the annealed branch computing `sigma_t`
  from the current state's `observations_seen`.
- `InferenceConfig`: `jitter` default flips to `"annealed"`; new field
  `jitter_tau: float` with the same default; `__post_init__` validates
  `jitter_tau > 0`. Docstring updated: `jitter_std` is the floor in
  annealed mode.
- `build_particle_filter` and `ModelComparison.__init__` pass `jitter_tau`
  through. `ModelComparison`'s own `jitter` default flips to `"annealed"`.

Defaults for `jitter_tau` (and the demo's `jitter_std` floor) are chosen
empirically by the scan; the spec placeholder assumption is tau ≈ 15 and
floor ≈ 0.02, to be replaced by the winning cell before merge.

### 3. Scan harness

- `scripts/_multi_mass_2d_scenario.py` — shared scenario module holding
  the truth constants and the rejection-sampling clock placement currently
  in `scripts/demo_multi_mass_2d.py`. Both the demo and the scan import it
  (same directory, so a plain import works under `uv run scripts/...`).
- `scripts/scan_multi_mass_2d.py` — grid over (tau, floor) × 12 seeds
  (0–11), multiprocessing across runs, printing a per-cell pass table.
  Indicative grid (overridable via CLI): tau ∈ {5, 10, 15, 25, 40},
  floor ∈ {0.01, 0.02, 0.05}.
  Seed `s` drives clock placement, simulation noise, and filter rng
  together, exactly as the demo's single `SEED` does today. Includes a
  fixed-jitter baseline mode to reproduce the June numbers.
- **Pass rule** (one seed): every one of the 6 parameters has truth within
  `mean ± 3·posterior_std`, **and** the posterior stds are capped at 0.5
  for position components and 0.1 for masses. The caps exclude
  "prior-wide cloud passes by being vague"; the 3σ window excludes
  "confident collapse on a wrong mode". June's passing runs had stds
  ~0.09 (positions) / ~0.05 (masses), comfortably inside the caps.
- `tests/test_acceptance_multi_mass_2d.py` — `@pytest.mark.slow` test
  running the 12 seeds at shipped defaults through the same runner
  function the script uses, asserting ≥ 10/12 passes. The `slow` marker
  is registered in `pyproject.toml` and excluded by default
  (`addopts = "-m 'not slow'"`); run manually with
  `uv run pytest -m slow`. Regular CI stays fast; no scheduled CI job.

### 4. Demos and site

- The default flip means all five demos pick up annealed jitter with no
  per-script changes; `demo_multi_mass_2d.py` additionally sets
  `jitter_std` to the winning floor.
- Regenerate all five GIFs into `assets/` and `site/assets/`, keeping the
  committed artifacts reproducible by the current scripts (the repo's
  standing policy).
- Site text (`site/method/the-particle-filter.qmd`): add an **annealed**
  bullet to the jitter-modes list, state the new default, and rewrite the
  particle-impoverishment failure note (which currently says the freeze
  "motivated the demos shipping with fixed jitter") to describe the
  annealed fix and the scan result. Check `site/story/two-hidden-masses.qmd`
  and `into-the-plane.qmd` for stale jitter phrasing. (The "at best 7/12
  seeds" scan numbers live in `docs/someday-maybe.md`, not on the site.)
- `docs/someday-maybe.md`: mark the adaptive/annealed jitter item done
  with a pointer to the harness and the shipped defaults; the "existing
  36-run seed-scan harness" phrasing becomes true.

### 5. Testing and failure handling

Fast unit tests (regular CI):

- Schedule endpoints: at t = 0 the effective std is the (clamped) initial
  cloud scale; as t → ∞ it approaches `floor`.
- Upward-anneal clamp: `init < floor` yields a constant-`floor` schedule.
- Validation: `jitter_tau <= 0` raises `ValueError` (both
  `ParticleFilter` and `InferenceConfig`).
- Default mode is `"annealed"` in `ParticleFilter`, `InferenceConfig`,
  and `ModelComparison`; `jitter_tau` plumbs through
  `build_particle_filter` and `ModelComparison`.
- Existing tests audited for baked-in `"fixed"`-default assumptions and
  updated deliberately (not silently).

Acceptance (slow): ≥ 10/12 seeds pass at shipped defaults.

Failure handling: if no (tau, floor) cell reaches 10/12, stop and reassess
rather than shipping a default that misses the bar. The recorded fallback
is a hybrid schedule with a posterior-std-scaled lower bound (approach C
from the design discussion); the same harness evaluates it.

## Out of scope

- Likelihood tempering; MCMC resample-move / rejuvenation.
- Annealing combined with `covariance`-mode jitter.
- In-browser interactivity, 3D demos, and other someday-maybe items.
