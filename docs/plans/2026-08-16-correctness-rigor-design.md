# Correctness and Rigor Remediation Design

**Date:** 2026-08-16

**Status:** Approved

## Objective

Correct the repository's known coding, physics, inference, packaging, and
prose defects. The result should make mathematically defensible claims, reject
inputs outside the stated physical model, and run the documented commands from
an installed wheel.

Version 0.1 has no backward-compatibility requirement. Misleading parameters,
behaviors, and internal modules will be removed rather than retained behind
aliases.

## Scope

This correction pass addresses six verified defects:

1. Forward models silently clamp invalid time-dilation radicands and several
   published scenarios operate outside the weak-field approximation.
2. Unconditional post-resampling jitter changes the target distribution, so
   posterior and evidence claims are not valid Bayesian SMC claims.
3. Residual resampling clips residual draws against the number of residual
   draws instead of the number of source particles, biasing the result.
4. Installed-wheel console commands depend on repository-only `scripts/`
   files and fail outside a checkout.
5. Public data and physics boundaries permit malformed shapes, broadcasting,
   non-finite values, and invalid physical parameters.
6. The site incorrectly says a symmetric ring makes mirror-image source
   locations indistinguishable even though clock channels are labeled.

Unrelated presentation improvements remain future work unless a prose or asset
change is necessary to stop making an incorrect claim.

## Physics Contract

The project will continue to use the pedagogical potential model

\[
\Phi_i = -\sum_j \frac{M_j}{r_{ij}}
\]

and rate map

\[
\frac{d\tau_i}{dt} = \sqrt{1 + 2\Phi_i}
\]

in units where `G = c = 1`. It will describe this as a weak-field surrogate,
not an exact strong-field solution or a superposition law from general
relativity.

The operational validity policy is

\[
\max_i |2\Phi_i| \le 0.1.
\]

This bound is a documented modeling policy. It keeps every radicand at least
0.9 for nonnegative masses, but the bound is about the approximation's stated
domain rather than merely avoiding a square-root error.

Direct scalar and batch physics calls are strict:

- positions, distances, potentials, masses, offsets, and density parameters
  must be finite and have their documented ranks and compatible shapes;
- masses and density amplitudes must be nonnegative;
- any positive point mass must have positive distance from every evaluated
  clock;
- potentials passed to the rate map must be nonpositive and within the
  weak-field bound;
- the Gaussian line-density model requires a positive width, positive track
  offset, positive integration limit, and at least two quadrature points; and
- violations raise a specific `ValueError` subclass instead of being clipped,
  broadcast, or converted into a plausible rate.

The numerical distance and radicand floors will be deleted.

Inference needs different error handling from direct evaluation. A candidate
particle outside prior support, ordering support, or the weak-field domain is
an impossible hypothesis and receives log density `-inf`; it must not abort a
whole batch. The SMC target evaluator will therefore apply the support mask
before calling strict forward models and evaluate predictions only for valid
rows. Ground truths and public direct calls remain loud failures.

All shipped ground truths, priors, examples, tests, studies, and generated
assets will be audited. Ground truths will be retuned into the validity domain.
Rectangular inference ranges may include invalid combinations because the
actual prior is the normalized rectangular prior conditioned on ordering and
weak-field validity. Rejection sampling will draw from that conditional prior,
with a bounded-attempt failure explaining when a requested prior has no useful
valid volume.

## Rigorous Sequential Monte Carlo

`ParticleFilter.update(observation)` remains the streaming interface, but its
implementation becomes adaptive tempered resample-move SMC for static
parameters.

For an incoming observation, define intermediate targets

\[
\pi_{t,\beta}(\theta) \propto
p(\theta)\prod_{s < t}p(y_s\mid\theta)
p(y_t\mid\theta)^\beta, \qquad 0 \le \beta \le 1.
\]

One public update advances `beta` from zero to one. At each stage the filter
chooses the largest likelihood increment that keeps effective sample size at
the configured target, using bisection when the full remaining increment is
too sharp. The incremental normalizing-constant estimate is accumulated before
any resampling. If ESS reaches the threshold, particles are resampled and
receive equal weights.

Every resample is followed by random-walk Metropolis-Hastings rejuvenation
against the exact current intermediate target. The proposal covariance is
computed and frozen for that stage, regularized against the initial prior
scale, and shared by every particle. This makes the Gaussian proposal
symmetric. Proposals outside support are rejected and remain at their parent;
there is no reflection, sorting, repair sampling, or unconditional mutation.

For multiple masses, the canonical support is strict ordering by the first
spatial coordinate. Initial draws are sorted together with their associated
masses, which represents the normalized ordered version of the exchangeable
prior. MH proposals that cross the ordering boundary are rejected. Sorting a
proposal after it is drawn is forbidden because it would obscure proposal
symmetry.

The existing `jitter`, `jitter_std`, `jitter_tau`, `constraint_fn`, and
`support_bounds` interfaces will be removed. Their replacements are explicit
SMC controls:

- `ess_target` in `(0, 1)`, default `0.8`;
- `rejuvenation_steps`, a positive integer, default `2`; and
- `proposal_scale`, a positive finite multiplier, default `2.38` and applied
  as `proposal_scale / sqrt(n_parameters)`.

The resampling method remains configurable. Systematic, stratified, and
residual implementations all accept a source-weight vector and a separate
draw count, and clip indices against `len(weights) - 1`. Residual resampling's
deterministic and stochastic portions will be tested independently.

The initial sampler must draw from the actual normalized prior represented by
`log_prior_density`. The density is needed up to an additive constant for MH;
the initial sample establishes the normalized prior measure for evidence. The
API-built clock models use a uniform prior conditioned on bounds, ordering,
and physical validity. Documentation will state that custom samplers and prior
densities must describe the same distribution or posterior and evidence
results are undefined.

### Efficient target evaluation

The observation model is static with independent Gaussian noise of fixed
standard deviation. Completed observations can therefore be represented by
three sufficient statistics: the count `n`, component-wise sum `sum_y`, and
scalar sum of squares `sum_y2`. For a prediction vector `mu(theta)`, the
cumulative Gaussian log likelihood is

\[
-nC\log(\sigma\sqrt{2\pi})
-\frac{\sum\|y\|^2 - 2\mu^T\sum y + n\|\mu\|^2}{2\sigma^2}.
\]

During tempering, the current observation contributes fractionally by `beta`.
This lets each MH stage evaluate the complete target in `O(N * C)` without
replaying every earlier observation. Sufficient statistics are committed only
after `beta` reaches one. One history entry is still emitted per public
observation, not per internal tempering stage.

The filter will expose diagnostic information needed to assess correctness:
the cumulative log evidence, tempering-stage count for the most recent update,
and MH proposal/acceptance counts. Diagnostics do not change the target.

## Public Data Contracts

Public dataclasses will coerce numeric inputs to defensive `float64` arrays,
validate them, and store read-only copies so their invariants cannot be broken
after construction.

- `MassConfig.positions`: nonempty 2-D array, with the existing 1-D shorthand
  interpreted as one spatial dimension.
- `MassConfig.masses`: nonempty 1-D array, with scalar shorthand retained;
  length must match positions and values must be nonnegative.
- `ClockArray.positions`: nonempty 2-D array, with 1-D shorthand retained;
  `track_offset` must be finite and nonnegative.
- `Observation.rates`: nonempty 1-D finite array; `time` must be finite.
- `ParticleState`: 2-D particles, matching 1-D nonnegative normalized weights,
  finite values, and a nonnegative observation count.

Configuration objects will reject non-finite noise, ranges, thresholds,
proposal controls, counts, and seeds where applicable. `PriorConfig.mass_range`
becomes real support rather than initialization advice. `infer()` validates
every observation's channel count against the clock array. `simulate()` and
all physics entry points validate spatial dimensional agreement.

Batch functions will require their documented exact ranks. They will not
accept shapes that merely happen to broadcast.

## Packaging and Commands

Executable demo implementations will move into packaged private modules under
`src/clocks/_demos/`, one module per command. `pyproject.toml` console scripts
will point directly at each packaged module's `main()` function.

Repository files under `scripts/demo_*.py` will become thin wrappers importing
and invoking those packaged functions, preserving convenient source-tree
execution without making the wheel depend on repository layout. The current
`clocks._cli` run-path/import fallback will be deleted.

A wheel smoke test will build the wheel, install it into a fresh temporary
environment, invoke every console command with a fast `--help` path, and
confirm that command dispatch reaches packaged code. Demo `main()` functions
will use `argparse`; the default no-argument behavior continues to generate the
documented output.

## Prose and Generated Evidence

The method pages will describe adaptive tempering, target-preserving MH moves,
conditional prior support, and evidence accumulation. All claims that jitter
"slightly" distorts the posterior, that mass ranges apply only initially, or
that the old clone-repair machinery is rigorous will be removed.

The units page will no longer defend deliberately strong-field scenarios. It
will state the `|2 Phi| <= 0.1` policy, explain that the chosen formula is a
weak-field surrogate, and distinguish numerical visibility from physical
fidelity.

The 2-D geometry page will say that a symmetric clock layout permutes the
readings under reflection. With labeled channels that is generally a different
observation vector; symmetry becomes a degeneracy only if channel identity is
discarded or the data themselves respect the permutation.

The architecture and reproducibility pages will be updated for packaged demos,
the new inference controls, and the stronger verification suite.

All committed GIF/PNG/JSON evidence affected by changed physics or inference
will be regenerated from fixed seeds after tests pass. Generated assets are
evidence: their producing command, configuration, and source commit must be
recorded, and no hand-edited asset may substitute for regeneration.

## Testing Strategy

Implementation proceeds test-first in correction-sized commits.

1. Physics unit tests prove invalid weak-field states raise, scalar and batch
   paths agree on valid states, no floor remains, and density-domain checks are
   enforced.
2. Contract tests cover every rejected rank, incompatible dimension,
   non-finite value, negative physical parameter, wrong observation length,
   and mutation attempt.
3. Resampling tests use deterministic draws and repeated empirical counts to
   detect the residual-index bug, including the case where the residual draw
   count is smaller than the source population.
4. SMC tests compare a one-dimensional Gaussian conjugate problem against its
   analytic posterior and marginal likelihood. Tests force multiple tempering
   stages and resample-move cycles, check MH detailed-balance consequences via
   moments, and verify evidence remains accurate after rejuvenation.
5. Prior tests show all initial and moved particles remain within mass,
   position, ordering, and physical support, and that impossible conditional
   priors fail explicitly.
6. End-to-end fixed-K and model-comparison tests use weak-field scenarios and
   fixed statistical tolerances chosen before looking at final seeds.
7. Packaging tests build and install the wheel in isolation and exercise all
   seven entry points.
8. Documentation tests render all Quarto pages and run the site link contract.
9. The full default suite, slow acceptance suite, Ruff checks, locked build,
   wheel smoke test, and site render must all pass before completion.

Where a regression could survive an output-only assertion, the test will also
contain a focused structural or statistical assertion. In particular, the
residual-resampling and evidence tests must fail under the exact old defects.

## Migration and Non-Goals

No compatibility shims will be provided for jitter-era configuration or the
repository-dependent CLI dispatcher. Examples and documentation migrate in the
same change so there is one supported behavior.

This work does not introduce exact Schwarzschild multi-body physics, dynamic
state tracking, non-Gaussian observation models, neural inference, or new site
features. Those require separate scientific and product designs.

## Completion Criteria

The correction is complete when:

- no public forward model silently clamps, broadcasts, or accepts a state
  outside the documented domain;
- posterior updates and log evidence arise from a target-preserving tempered
  resample-move SMC algorithm;
- empirical residual-resampling frequencies match their input weights;
- all configured prior ranges are true support;
- every documented console command works from the built wheel;
- all prose claims match the implemented physics and inference;
- affected generated evidence has been reproducibly regenerated; and
- the complete verification matrix passes from a clean worktree.
