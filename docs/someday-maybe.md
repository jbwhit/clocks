# Someday / Maybe

Ideas considered and deliberately not implemented yet. (Sources: scratch
notes and an external Gemini review, 2026-02; updated 2026-06.)

## Inference

- **Richer invariant proposals.** Compare the current symmetric Gaussian
  random-walk Metropolis-Hastings move with target-preserving alternatives
  for curved or multimodal posteriors. Any replacement must retain an explicit
  acceptance rule and preserve each tempered target.
- **Population-level reliability study.** The corrected SMC controls and
  tolerances are frozen from the development seeds, and the reserved 400–411
  block has now been run exactly once. The protected results provide regression
  and calibration evidence for those particular
  cases, not a reliability estimate over a declared population. A later study
  could preregister such a population and attach confidence intervals to
  failure rates.
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
  website without a server. (The website ships pre-rendered video drawn
  from committed replay data, with native controls and no script; see
  docs/plans/2026-11-19-replay-and-animation-design.md.)
