# Annealed Jitter for the Particle Filter

**Date:** 2026-07-02
**Status:** Draft — Codex xhigh round 1 (NEEDS REVISION: 3 Critical, 6
Important, 3 Minor) applied: pass rule replaced with the absolute-error
rule that exactly reproduces the June baseline (verified by rerunning the
36-run scan), tuning/holdout seed split added, post-jitter support-repair
policy added, animation frame-0 double-processing fix pulled into scope,
scenario module moved into the package, validation tightened, default-flip
behavioral regressions and full-site re-render added. Pending round 2.
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

The jitter draw is an axis-aligned diagonal Gaussian (a vector of
per-parameter stds broadcast over particles), i.e. the `fixed` branch
generalized to a scheduled vector scale. The `fixed` and `covariance`
branches are unchanged.

**Post-jitter support policy.** Today `log_prior` is evaluated *before*
resampling, so particles jittered out of support (out-of-range positions,
non-positive masses) enter the public state with uniform weight and are
only killed by the next observation — and after the final observation
there is no next one. Prior-scale annealed jitter makes this common
instead of rare, so `_resample` gains a repair step (all jitter modes,
when `log_prior` is provided): after jittering, particles with
`log_prior = −inf` are re-jittered from their resampled parent, up to 10
retries; any still-invalid particle falls back to its parent value
unjittered (parents are always in support, since invalid particles carry
zero weight into resampling and are never selected). `constraint_fn`
still runs afterwards. The existing `log_prior` support definition
(positions in range, mass > 0 — the mass upper bound deliberately
unenforced) is unchanged, preserving baseline comparability.

### 2. API and plumbing

- `ParticleFilter.__init__`: accept `jitter="annealed"` (new **default**)
  and `jitter_tau: float` (default = the scan-selected value, see §3).
  Validate unconditionally (fail fast, regardless of mode): `jitter_tau`
  finite and > 0, `jitter_std` finite and ≥ 0 — NaN passes a naive `<= 0`
  check, an infinite tau never decays, and a negative `jitter_std` makes
  the annealed asymptote negative and blows up inside `rng.normal`.
  Capture the
  clamped `init` vector after drawing the initial cloud.
- `ParticleFilter._resample`: add the annealed branch computing `sigma_t`
  from the current state's `observations_seen`.
- `InferenceConfig`: `jitter` default flips to `"annealed"`; new field
  `jitter_tau: float` with the same default; `__post_init__` applies the
  same finiteness/sign validation to `jitter_tau` and `jitter_std`
  (currently unvalidated). Docstring updated: `jitter_std` is the floor
  in annealed mode.
- `build_particle_filter` and `ModelComparison.__init__` pass `jitter_tau`
  through. `ModelComparison`'s own `jitter` default flips to `"annealed"`.

Defaults for `jitter_tau` (and the demo's `jitter_std` floor) are chosen
empirically by the scan using the winner-selection rule in §3; the spec
placeholder assumption is tau ≈ 15 and floor ≈ 0.02, to be replaced by
the winning cell before merge. The shipped defaults are recorded in this
spec's status line and in the site text when chosen.

### 3. Scan harness

- `clocks._scenarios` — shared scenario module (in the installed package,
  not `scripts/`) holding the truth constants, the rejection-sampling
  clock placement currently in `scripts/demo_multi_mass_2d.py`, and the
  single-run runner + pass-rule evaluation used by scan and test. It must
  live in the package because the demo console-scripts launch via
  `runpy.run_path` (`clocks._cli`), which does not put `scripts/` on
  `sys.path`, and pytest imports from the repo root have the same
  problem.
- `scripts/scan_multi_mass_2d.py` — grid over (tau, floor) × 12 seeds
  (0–11), multiprocessing across runs, printing a per-cell pass table.
  Indicative grid (overridable via CLI): tau ∈ {5, 10, 15, 25, 40},
  floor ∈ {0.01, 0.02, 0.05}.
  Seed `s` drives clock placement, simulation noise, and filter rng
  together, exactly as the demo's single `SEED` does today. Includes a
  fixed-jitter baseline mode to reproduce the June numbers.
- **Pass rule** (one seed): absolute posterior-mean error within 0.5 sim
  units for every position component and within 0.1 for every mass. This
  rule reproduces the June baseline **exactly** (verified 2026-07-02:
  jitter_std 0.02/0.05/0.10 → 1/12, 5/12, 7/12), so before/after numbers
  are directly comparable. An earlier draft used "truth within mean ± 3σ
  plus posterior-std caps"; it was dropped because it cannot reproduce
  the baseline (it scores 0.10 as 0/12 — every seed's mass std lands at
  0.101–0.108, just over the cap) and because jitter directly inflates
  `posterior_std`, making the rule self-referential.
- **Diagnostics** reported per run (not gating): 3σ coverage (is truth
  within mean ± 3·posterior_std per parameter), max posterior std, and
  predictive residual `max |rates(posterior_mean) − true_rates| /
  noise_std` — the June failure signature was residuals ~7× noise.
- **Tuning vs holdout seeds.** The grid is tuned on seeds 0–11; the
  winning cell is then validated on fresh holdout seeds 100–111.
  Acceptance = ≥ 10/12 on the **holdout** seeds (selection on the same
  seeds that certify the winner would bias the estimate). Winner
  selection on the tuning seeds: highest pass count, ties broken by
  lowest median (over seeds) max-parameter absolute error, remaining
  ties by smaller tau (less artificial diffusion).
- `tests/test_acceptance_multi_mass_2d.py` — `@pytest.mark.slow` test
  running the 12 **holdout** seeds at shipped defaults through the same
  runner function the script uses, asserting ≥ 10/12 passes. This is a
  deterministic regression pin (same seeds, same code ⇒ same result),
  not a population reliability estimate. The `slow` marker is registered
  in `pyproject.toml` and excluded by default (`addopts = "-m 'not
  slow'"`); run manually with `uv run pytest -m slow`. Regular CI stays
  fast; accepted tradeoff: the acceptance scan does not guard future
  merges automatically — rerun it when touching inference defaults.

### 4. Demos, animation fix, and site

- **Animation off-by-one fix** (pre-existing bug, load-bearing here):
  `_animate_filter_dashboard` and the model-comparison animator hand
  `FuncAnimation` a state-mutating callback with no `init_func`, so
  matplotlib processes frame 0 twice — the committed GIFs ran 81 filter
  updates for 80 observations (verified against the committed estimates).
  Because the anneal schedule keys off `observations_seen`, this would
  also shift the schedule. Fix: precompute the filter states/estimates by
  driving the filter through the observations once, then let the
  animation render from the stored sequence; assert afterwards that
  `observations_seen == len(observations)`. GIF paths then match the
  `infer()`/scan path exactly.
- The default flip means all demos pick up annealed jitter with no
  per-script changes; `demo_multi_mass_2d.py` additionally sets
  `jitter_std` to the winning floor. `demo_density.py` uses the raw
  `ParticleFilter` default, so its PNG changes too.
- Regenerate **all** committed demo artifacts (five GIFs + the density
  PNG): scripts write to `output/`, then copy to `assets/` and
  `site/assets/` with a byte-equality check between the two copies,
  keeping the committed artifacts reproducible by the current scripts
  (the repo's standing policy).
- Site text (`site/method/the-particle-filter.qmd`): add an **annealed**
  bullet to the jitter-modes list, state the new default, and rewrite the
  particle-impoverishment failure note (which currently says the freeze
  "motivated the demos shipping with fixed jitter") to describe the
  annealed fix and the scan result. Check `site/story/two-hidden-masses.qmd`
  and `into-the-plane.qmd` for stale jitter phrasing. (The "at best 7/12
  seeds" scan numbers live in `docs/someday-maybe.md`, not on the site.)
- Many site pages **execute** `InferenceConfig`-based examples at render
  time, so the default flip changes their printed outputs: re-render the
  full site and review the executed cell outputs, not just the prose.
- `docs/someday-maybe.md`: mark the adaptive/annealed jitter item done
  with a pointer to the harness and the shipped defaults; the "existing
  36-run seed-scan harness" phrasing becomes true.

### 5. Testing and failure handling

Fast unit tests (regular CI):

- Schedule endpoints: at t = 0 the effective std is the (clamped) initial
  cloud scale; as t → ∞ it approaches `floor`.
- Upward-anneal clamp: `init < floor` yields a constant-`floor` schedule.
- Validation: non-finite or non-positive `jitter_tau`, and non-finite or
  negative `jitter_std`, raise `ValueError` (both `ParticleFilter` and
  `InferenceConfig`).
- Post-jitter support repair: after a resample with large jitter and a
  `log_prior`, every particle in the public state is in support; a
  particle whose every retry fails falls back to its parent.
- Animation: after generating an animation, the filter has seen exactly
  `len(observations)` observations (pins the frame-0 double-processing
  fix).
- Default mode is `"annealed"` in `ParticleFilter`, `InferenceConfig`,
  and `ModelComparison`; `jitter_tau` plumbs through
  `build_particle_filter` and `ModelComparison`.
- **Default-flip behavioral regressions** (the flip is otherwise only
  validated on the multi-mass-2D problem): existing scenario tests
  (single-mass 1D/2D recovery, multi-mass 1D, model comparison selecting
  the true K) must pass under the new defaults with their thresholds
  unchanged; add a model-comparison correct-K regression if none exists.
  If model comparison degrades under annealed jitter (its evidence
  accumulates through a K-dependent artificial transition), the recorded
  fallback is `ModelComparison` keeping `jitter="fixed"` as its default
  while the rest of the library flips.
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
