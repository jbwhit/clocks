# Clone-Freeze Diagnosis and Support-Repair Amendment

**Date:** 2026-07-03
**Status:** Root cause confirmed by two independent instrumented
reproductions (Claude + Codex xhigh); remedy agreed (Codex xhigh:
"AGREED DESIGN", one round). Amends the support-repair policy of
[2026-07-02-annealed-jitter-design.md](2026-07-02-annealed-jitter-design.md).

## The decision-gate failure

The annealed-jitter tuning winner (tau=5, floor=0.02) scored 10/12 on
tuning seeds 0–11 but only 7/12 on holdout seeds 100–111. The five
failing seeds (101, 105, 106, 107, 111) all reported weighted posterior
std ≈ 0 and bit-identical errors across floors 0.01/0.02/0.05 — the
floor never influenced those runs.

## Root cause: clone-freeze

Confirmed on seed 101 (instrumented; both reviewers reproduced
independently):

1. t=0: the first reweight is razor-sharp (noise_std=0.005, 10 clocks);
   one prior particle dominates and systematic resampling makes nearly
   all 4000 parents clones of it.
2. The annealed jitter at t=0 uses sigma = init (full prior scale), so
   ~73% of 6-D proposals leave prior support.
3. Reject-and-stay reverts every invalid proposal to its parent →
   ~2942 exact clones + ~1058 scattered valid draws.
4. t=1: the clone value out-weighs the scattered prior-scale draws.
   Identical clones share weight equally, so ESS = clone count = 2942
   (exactly: 1 / (2942 · (1/2942)²)), above the resample threshold
   (0.5 × 4000 = 2000).
5. ESS never falls again; no resample ever fires; no jitter is ever
   applied; the filter reports ~zero posterior std at a wrong mode
   (residuals 3.6–6.7× noise) for the remaining 79 observations.

**Mass reversion to a single dominant parent creates a clone-majority
cloud whose ESS (weight diversity) stays high despite zero state
diversity, permanently disabling the ESS-triggered resample→jitter
cycle.** Larger tau extends the prior-scale phase and raises the freeze
probability — this fully explains the tuning grid's collapse above
tau≈5.

Correction to an earlier draft of this diagnosis: the fixed-jitter
baseline was NOT degraded by the repair. June's floors were
0.02/0.05/0.10 (1/12, 5/12, 7/12); the scan's baseline used
0.01/0.02/0.05. At the overlapping floors the post-repair numbers match
June exactly (0.02 → 1/12, 0.05 → 5/12), and repair-disabled runs
reproduce them bit-for-bit — fixed jitter at these floors almost never
proposes out of support. Baseline comparisons should use June's floors
{0.02, 0.05, 0.10}.

## Agreed remedy (Codex xhigh, 2026-07-03)

**A. Bounds-aware reflection for the diagonal Gaussian modes ("fixed"
and "annealed"):**

- `ParticleFilter` gains optional `support_bounds: (lower, upper)` —
  per-parameter arrays. `build_particle_filter` and `ModelComparison`
  construct them from `PriorConfig`: position coordinates get
  `position_range`; masses get lower = smallest positive float
  (`np.nextafter(0.0, 1.0)`), upper = `+inf`. Masses are NOT reflected
  at `mass_range` — the approved support stays "mass > 0, no upper
  bound".
- Repeated triangular-wave reflection into [lower, upper] (a single
  bounce or clipping mishandles overshoots larger than the box width);
  one-sided reflection when only one bound is finite.
- Order: jitter → reflect → `constraint_fn` → final `log_prior`
  validation. If any particle is still invalid after reflection, raise
  `RuntimeError` (the supplied bounds contradict `log_prior`) — never
  silently revert.
- Rationale: the diagonal reflected kernel is symmetric and preserves a
  uniform hard-box distribution without boundary atoms, and reflection
  cannot create clones.

**B. Clone-aware resample backstop for the remaining reject-and-stay
paths** (covariance mode, or a raw `ParticleFilter` with `log_prior`
but no `support_bounds`):

- Keep one-shot reject-and-stay there (coordinate-wise reflection is
  not symmetric for correlated covariance kernels).
- After any repair that actually reverted proposals, also compute the
  state-collapsed ESS (group weights by unique particle value):
  `ESS_state = 1 / Σ_g (Σ_{i in g} w_i)²`. Trigger resampling when
  either ordinary ESS or ESS_state crosses the existing threshold. If
  ESS_state < 2, the covariance branch uses its isotropic fallback.

**C. Rejected alternative:** roughening (jitter every update) moves
weighted particles without reweighting or an MH correction. The spec's
recorded fallback (posterior-std-scaled floor) does not address the
failure at all — frozen runs never reach the jitter code.

## Prototype evidence (read-only, pre-implementation)

Reflection monkeypatch, both reviewers independently:

- annealed tau=5 floor=0.02 — tuning 10/12, burned holdout 11/12,
  frozen runs: none.
- annealed tau=15 floor=0.02 — tuning 12/12 (the tau≥15 collapse was
  entirely the freeze).
- fixed 0.05 — 4/12 tuning, 2/12 holdout (reflection barely affects the
  baseline; the annealed gain is real).

## Retry protocol (preregistered)

- Tuning stays on seeds 0–11. Seeds 100–111 are burned (diagnostics and
  regressions only — e.g. a seed-101 no-freeze regression test).
- After implementing the remedy: rerun the full tuning grid, select the
  winner by the existing total order, freeze code + grid + winner + pass
  rule, then run seeds 200–211 exactly once. If 200–211 fails, stop and
  burn that set too.
- The scan script and slow acceptance test move their certification
  seeds to 200–211. Baseline runs use floors {0.02, 0.05, 0.10}.
