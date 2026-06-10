# Someday / Maybe

Ideas considered and deliberately not implemented yet. (Sources: scratch
notes and an external Gemini review, 2026-02; updated 2026-06.)

## Inference

- **MCMC rejuvenation step.** Add a Metropolis-Hastings accept/reject after
  the post-resampling jitter, turning the filter into a rigorous SMC sampler
  that exactly preserves the posterior. Today's jitter slightly distorts it.
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
