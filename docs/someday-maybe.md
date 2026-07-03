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
- **3D and exterior masses: gravitational echolocation.** The library is
  already dimension-agnostic, but no demo or page exercises 3D, and none
  places the mass *outside* the clock volume. The motivating premise is a
  hard sci-fi idea: a person born with a 3D lattice of atomic clocks
  embedded in their head, sensing the mass distribution of their
  surroundings the way echolocation senses surfaces. The physics questions
  it raises are real and well-posed for this codebase: how does inference
  quality fall off with distance once the mass is outside the lattice
  (exterior masses only show the lattice a far-field gradient — closely
  related to the mass-distance degeneracy)? What lattice geometry maximizes
  "field of view" per clock? What is the angular/range resolution of an
  N-clock head at a given noise floor? A 3D demo plus a resolution-vs-range
  study would make a strong site page — and would quantify whether the
  sci-fi sense is physically coherent.

## Deployment

- **In-browser interactivity.** The inference engine is pure numpy/scipy,
  which Pyodide supports — a live-slider demo could run client-side on the
  website without a server. (The website ships static GIFs by deliberate
  scope choice; see docs/superpowers/specs/2026-06-09-deep-clean-and-website-design.md.)
