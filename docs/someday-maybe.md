# Someday / Maybe

Ideas considered and deliberately not implemented yet. (Sources: scratch
notes and an external Gemini review, 2026-02; updated 2026-06.)

## Inference

- **MCMC rejuvenation step.** Add a Metropolis-Hastings accept/reject after
  the post-resampling jitter, turning the filter into a rigorous SMC sampler
  that exactly preserves the posterior. Today's jitter slightly distorts it.
- **Adaptive or annealed jitter.** The 6-parameter multi-mass 2D problem
  recovers truth on a minority of seeds under any tested fixed jitter
  (scan 2026-06: jitter_std 0.02/0.05/0.10 → 1/12, 5/12, 7/12 passes; the
  demo ships 0.05, the best tested was 0.10 at ~58%).
  Premature collapse onto non-fitting modes is the failure; a jitter that
  scales with the posterior std or anneals over observations would attack
  it directly, where a larger constant only buys a noise floor.
- **Neural-net amortized inference.** Train a network on simulated
  (clock rates → mass parameters) pairs and compare its speed/accuracy
  against the particle filter.

## Physics

- **Special-relativity velocity term.** Give masses or clocks velocities and
  use the combined dilation factor sqrt(1 + 2*Phi - v^2) (c = 1), so SR and
  GR effects compete the way they do for real GPS satellites.
- **Mass placement scenarios.** Masses inside vs outside a ring of clocks in
  2D; which geometries make the inverse problem ill-posed?

## Deployment

- **In-browser interactivity.** The inference engine is pure numpy/scipy,
  which Pyodide supports — a live-slider demo could run client-side on the
  website without a server. (The website ships static GIFs by deliberate
  scope choice; see docs/superpowers/specs/2026-06-09-deep-clean-and-website-design.md.)
