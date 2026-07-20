# Someday / Maybe

Ideas considered and deliberately not implemented yet. (Sources: scratch
notes and an external Gemini review, 2026-02; updated 2026-06.)

## Inference

- **MCMC rejuvenation step.** Add a Metropolis-Hastings accept/reject after
  the post-resampling jitter, turning the filter into a rigorous SMC sampler
  that exactly preserves the posterior. Today's jitter slightly distorts it.
- **Adaptive or annealed jitter.** Shipped 2026-07-03 — spec
  `docs/superpowers/specs/2026-07-02-annealed-jitter-design.md` (amended
  per `docs/superpowers/specs/2026-07-03-clone-freeze-diagnosis.md`).
  Harness: `scripts/scan_multi_mass_2d.py`; acceptance test:
  `tests/test_acceptance_multi_mass_2d.py` (slow marker, run via
  `uv run pytest -m slow`). Shipped defaults: `jitter_tau=15.0`
  (library default, all inference entry points), demo/runner floor
  `jitter_std=0.02`, mode `"annealed"`. Measured: post-reflection
  fixed-jitter baseline at the shipped floor recovered only 1/12 tuning
  seeds (0.02 → 1/12, 0.05 → 4/12, 0.10 → 7/12); annealed jitter
  (tau=15, floor=0.02) recovered 12/12 tuning seeds (0-11) and 11/12 on
  a fresh certification holdout (seeds 200-211, run exactly once).
- **Neural-net amortized inference.** Train a network on simulated
  (clock rates → mass parameters) pairs and compare its speed/accuracy
  against the particle filter.

## Physics

- **Special-relativity velocity term.** Give masses or clocks velocities and
  use the combined dilation factor sqrt(1 + 2*Phi - v^2) (c = 1), so SR and
  GR effects compete the way they do for real GPS satellites.
- **Mass placement scenarios.** Masses inside vs outside a ring of clocks in
  2D; which geometries make the inverse problem ill-posed?
- **3D and exterior masses: gravitational echolocation.** Shipped
  2026-07-19 — spec
  `docs/superpowers/specs/2026-07-19-3d-echolocation-design.md`.
  Demo: `uv run demo-echolocation-3d`; study:
  `scripts/scan_echolocation_range.py`; site page
  `site/story/gravitational-echolocation.qmd`. Deferred follow-ons still
  open: lattice-geometry comparisons (field of view per clock),
  angular-resolution measurement, multiple/moving exterior masses.

## Deployment

- **In-browser interactivity.** The inference engine is pure numpy/scipy,
  which Pyodide supports — a live-slider demo could run client-side on the
  website without a server. (The website ships static GIFs by deliberate
  scope choice; see docs/superpowers/specs/2026-06-09-deep-clean-and-website-design.md.)
