# Someday / Maybe

Ideas considered and deliberately not implemented yet. (Sources: scratch
notes and an external Gemini review, 2026-02; updated 2026-06.)

## Inference

- **MCMC rejuvenation step.** Add a Metropolis-Hastings accept/reject after
  the post-resampling jitter, turning the filter into a rigorous SMC sampler
  that exactly preserves the posterior. Today's jitter slightly distorts it.
- **Adaptive or annealed jitter.** The 6-parameter multi-mass 2D problem
  does not reliably recover truth under tested fixed jitters — at best
  7/12 seeds (scan 2026-06: jitter_std 0.02/0.05/0.10 → 1/12, 5/12, 7/12
  passes; the demo ships 0.05).
  Premature collapse onto non-fitting modes is the failure; a jitter that
  scales with the posterior std or anneals over observations would attack
  it directly, where a larger constant only buys a noise floor.
  Sizing (2026-06): small-to-medium — ~50-100 lines (new jitter mode in
  `_resample` + `InferenceConfig` plumbing) with validation via the
  existing 36-run seed-scan harness. Likely design: anneal from
  prior-scale jitter toward a floor over the first observations (pure
  std-scaling alone re-creates the covariance-mode freeze; needs a floor).
  Scope boundary: annealed/floored jitter only, acceptance >= 10/12 on the
  multi-mass-2d scan; likelihood tempering and MCMC resample-move stay
  out of scope (see the rejuvenation item above).
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
