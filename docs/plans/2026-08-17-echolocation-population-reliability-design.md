# Echolocation Population Reliability and Identifiability Design

**Date:** 2026-08-17

**Status:** Approved for implementation

## Purpose

The existing echolocation certification is fixed-case regression evidence. It
uses one exterior direction, one mass, six fixed ranges, and twelve reserved
noise/inference seeds. It does not estimate reliability over a declared
population, and its falloff from 12/12 close-range recoveries to 0/12 at the
farthest certified range does not by itself distinguish physical information
loss from inference failure.

This work adds a preregistered population study for the current 27-clock cubic
head. It pairs empirical SMC recovery with an analytic local Fisher-information
diagnostic so that range-dependent failures can be interpreted without tuning
the inference algorithm after seeing the study results.

## Goals

1. Estimate the existing binary recovery probability across a declared
   exterior-mass population, stratified by range, with confidence intervals.
2. Measure angular, radial, mass, uncertainty, residual, and computational
   behavior continuously rather than reducing every result to pass/fail.
3. Compute an analytic, dimensionless local sensitivity diagnostic at every
   truth and relate empirical failures to weakly identified parameter
   combinations.
4. Freeze every population case and random stream before the full inference
   run.
5. Publish a canonical, semantically validated artifact and an honest report
   without a post-hoc acceptance threshold or SMC retuning.

## Non-goals

- Comparing alternative clock-head geometries.
- Selecting new SMC controls or changing the certified controls.
- Replacing the existing development or certification artifacts.
- Claiming global identifiability from a local Fisher calculation.
- Performing full simulation-based calibration or posterior-rank testing.
- Declaring the system acceptable or unacceptable using a threshold chosen
  after the earlier 46/72 fixed-case result.

## Scientific population

The study contains 384 cases: six equal-width strata on the log-range interval
from 2 to 8 head circumradii, with 64 independently generated cases in each
stratum. Equal allocation makes the six range estimates equally precise and
the equal-weighted stratum mean correspond to the declared log-uniform range
population.

Within each stratum:

- direction is uniform on the unit sphere;
- log-range is uniform within the stratum bounds;
- log-mass is uniform from log(0.02) to log(0.08);
- the clock head is the existing 3 x 3 x 3 cubic lattice;
- there are 80 observations;
- observation noise standard deviation is 0.001; and
- the filter uses the already certified controls: 6,000 particles,
  `ess_target=0.9`, one rejuvenation step, and proposal scale 1.5.

The upper mass bound is globally safe under the project's strict weak-field
policy at the closest declared exterior range. Every generated truth is still
validated against the actual clock geometry before the manifest is accepted.

## Endpoints and estimands

The primary endpoint preserves the current definition:

```text
position_error <= 1.0 and mass_error <= 0.04
```

The primary results are the six range-stratum success probabilities with
two-sided 95% Wilson intervals. There is deliberately no pass/fail target for
the study itself.

The overall log-uniform-population success estimate is the equal-weighted mean
of the six stratum estimates. Because allocation is fixed by stratum, the
report must not describe all 384 observations as one homogeneous binomial
sample. Its uncertainty interval is produced by a preregistered stratified
bootstrap that resamples within each stratum using a separate recorded
analysis seed.

Secondary empirical metrics are:

- absolute Cartesian position error;
- absolute mass error;
- angular separation between true and estimated directions;
- absolute log-range error;
- absolute log-mass error;
- posterior Cartesian-position and mass standard deviations;
- marginal central 95% interval bounds and truth-coverage flags for x, y, z,
  and mass;
- residual-to-noise ratio;
- forward-model evaluation count; and
- SNR at the true scenario.

Coverage is a frequentist diagnostic over this declared population. Because
the population distribution is not identical to the filter's rectangular
prior, coverage is not described as a Bayesian calibration theorem.

## Architecture

### Population-study package module

A new private package module, `clocks._reliability`, owns:

- immutable study constants and typed records;
- deterministic case-manifest generation;
- manifest and result semantic validation;
- analytic contrast-space Jacobians and Fisher diagnostics;
- Wilson intervals and stratified-bootstrap summaries;
- canonical JSON encoding, atomic checkpointing, resumption, and finalization;
- publication summary tables; and
- reliability/identifiability figures.

The module lives in the package rather than under `scripts/` so tests and
Quarto can import the same implementation. It must not import pyplot before a
caller has selected a headless backend.

### Scenario integration

`clocks._scenarios` gains a generic internal echolocation case runner accepting
an arbitrary valid truth position, mass, observation seed, and inference seed.
The existing fixed-direction `run_echolocation_3d` behavior remains unchanged
and delegates to the generic path. Current certification constants and
artifacts are not rewritten.

The generic path records posterior marginal central intervals from the final
weighted particle state. It must use an explicitly tested weighted-quantile
routine and must not assume terminal weights are uniform.

### Thin CLI

`scripts/run_echolocation_population.py` provides three explicit operations:

1. generate the frozen manifest;
2. run or resume the exact manifest; and
3. finalize or re-render a complete result artifact.

Population-shaping overrides are allowed only during small development tests.
The release manifest command uses the package constants. Once a manifest is
marked frozen, the run command rejects overrides to case count, distributions,
head, observation model, tolerances, or SMC controls.

## Randomness and preregistration

The release manifest uses the transparent date-derived master seed `20260817`.
The root `numpy.random.SeedSequence` spawns one child sequence per case. Each
case child then spawns independent parameter-generation, observation-noise,
and inference streams. The analysis bootstrap uses a separate child sequence.

The committed manifest stores the realized truth and the integer observation
and inference seeds for every case. Reproduction therefore does not depend on
regenerating values from a future NumPy implementation. The document also
records the generator algorithm, master seed, spawn policy, package controls,
population bounds, and canonical content hash.

Case IDs are stable and encode only study version, stratum, and ordinal; they
do not depend on result values. Directions are generated by normalizing three
independent standard normal variates. An exactly zero draw is rejected and
redrawn, although it has probability zero in the mathematical model.

## Analytic local-identifiability diagnostic

For clock position `c_i`, true mass position `p`, true mass `M`, distance
`r_i = ||c_i - p||`, and strict weak-field rate

```text
f_i = sqrt(1 - 2 M / r_i),
```

the exact derivatives are

```text
d f_i / d p = -M (c_i - p) / (f_i r_i^3)
d f_i / d M = -1 / (f_i r_i).
```

The existing `(C - 1) x C` orthonormal contrast matrix `Q` removes the common
clock mode. The Cartesian Jacobian is transformed by `Q` before information is
calculated.

Let `rho = ||p||`, `u = p / rho`, and let `t1, t2` be any deterministic
orthonormal tangent basis perpendicular to `u`. The local dimensionless
coordinates are two angular displacements, log-range, and log-mass. Their
Jacobian columns are

```text
J_p @ (rho * t1)
J_p @ (rho * t2)
J_p @ p
J_M * M.
```

For `N` independent observations with contrast noise standard deviation
`sigma`, the whitened local sensitivity matrix is

```text
A = sqrt(N) / sigma * J_dimensionless.
```

The case record contains:

- all four singular values of `A`, in descending order;
- the condition number;
- numerical rank under the documented standard SVD machine tolerance;
- the weakest right-singular vector's squared loadings, combined into angular,
  log-range, and log-mass contributions; and
- local Cramer-Rao standard deviations when the matrix is numerically full
  rank and the values are representable.

Singular values must be invariant to the arbitrary rotation of `t1, t2` inside
the tangent plane. Weak-vector angular contributions are combined across both
angular coordinates for the same reason. Fisher values are local linear
diagnostics, not claims of global uniqueness or guaranteed posterior width.

## Manifest and result artifacts

### Frozen manifest

`docs/reliability/echolocation_population_v1_manifest.json` contains:

- schema and study versions;
- study identity and status;
- master/analysis seeds and generator metadata;
- head, inference, population, endpoint, and interval specifications;
- exact stratum edges and allocation;
- 384 case definitions; and
- a canonical hash over the semantic manifest payload.

Before the full run, the manifest generator and validators are tested, the
release manifest is generated, and the implementation plus manifest is
committed and pushed. An independent exact-SHA review is required before
population inference begins.

### Checkpoint

The untracked checkpoint under `output/` contains the manifest hash and a map
of completed case IDs to result records. The parent process, not workers,
publishes checkpoints using same-directory temporary files, flush/fsync, and
`os.replace`.

On resume, every existing record is validated before missing cases are
scheduled. A duplicate, foreign, malformed, or manifest-mismatched record
aborts without overwriting the checkpoint. Worker exceptions leave the last
valid checkpoint intact and are never converted into skipped cases.

### Final artifact

`docs/reliability/echolocation_population_v1_results.json` is published only
when all 384 exact case IDs are present. Finalization recomputes every
reconstructible truth, error, pass, SNR, and Fisher field; validates marginal
interval ordering and coverage booleans; rejects nonfinite or extra data; and
canonicalizes ordering and numeric JSON types.

The artifact contains summaries for convenience, but the validator and tests
recompute all published summaries from case records rather than trusting
stored aggregates.

## Analysis and reporting

The preregistered report includes:

1. the population and freeze protocol;
2. a six-row reliability table with Wilson intervals;
3. the equal-weighted overall estimate and stratified-bootstrap interval;
4. median and 10th/90th percentiles by range stratum for continuous metrics;
5. marginal interval-coverage rates with Wilson intervals;
6. smallest singular value versus empirical error;
7. weakest-direction angular/range/mass composition by stratum;
8. SNR and forward-evaluation distributions; and
9. explicit limitations.

The figures are publication-quality, colorblind-safe, and generated from the
tracked final artifact:

- success probability with Wilson intervals by range stratum;
- empirical angular/radial/mass resolution distributions;
- empirical error against the smallest Fisher singular value; and
- weakest-direction component composition across range.

The existing gravitational-echolocation story links to the population report
and explains the distinction among fixed-case certification, empirical
population reliability, and local Fisher information. It must not generalize
beyond the declared head, population, noise level, observation count, or SMC
controls.

## Error handling and audit rules

- Validate the complete manifest before starting any worker.
- Reject non-unit directions, out-of-stratum ranges, out-of-population masses,
  physical-domain violations, duplicated seeds/case IDs, and coupled random
  streams.
- Reject complex, nonfinite, wrong-rank, and wrong-shape inputs without warning
  leakage.
- Never silently clamp a physical state, metric, interval, probability, or
  Fisher value.
- A partial checkpoint is resumable evidence, not a final study.
- No results may alter distributions, case allocation, controls, metrics,
  intervals, figures, or report structure under study version 1.
- A genuine implementation/specification defect invalidates version 1. The
  reason is recorded and a new version is created; evidence is never silently
  replaced under the same identity.
- Development smoke cases use a separate explicit study identity and can never
  be finalized or archived as the release population.

## Testing strategy

Implementation follows strict test-driven development.

### Population and seed tests

- deterministic manifest bytes for the release seed;
- exact six-by-64 allocation and stable case ordering;
- unit directions, bounds, log-stratum membership, and physical validity;
- distinct parameter, observation, inference, and analysis streams;
- mutation tests for every manifest semantic class; and
- source-array and decoded-JSON mutation resistance where public snapshots are
  exposed.

### Mathematical tests

- analytic Cartesian/contrast Jacobians against central finite differences at
  multiple valid off-axis truths;
- tangent-basis orthonormality and singular-value/bundled-angular-loading
  invariance under tangent-plane rotation;
- Fisher scaling by `sqrt(N) / sigma`;
- positive-semidefinite information and known symmetry/degeneracy cases;
- dimensionless coordinate ordering and weakest-vector loading sums; and
- Wilson and stratified-bootstrap calculations against hand-computed examples.

### Runner and artifact tests

- weighted quantiles against exact hand-computed distributions;
- arbitrary-truth scenario smoke tests with deliberately distinct noise and
  inference seeds;
- injected fake runners for complete, partial, resumed, failed, duplicate, and
  foreign result sets;
- sentinel-file tests proving validation failures preserve prior evidence;
- atomic publication tests;
- independent recomputation of every derived result and summary field; and
- refusal to finalize 383 cases, development cases, or a mismatched manifest.

The default test suite performs no 384-case inference. A tiny real scenario
smoke test uses minimal observations and particles. The release population run
is a deliberate protected command executed only after the manifest freeze
review.

## Verification and review lifecycle

1. Implement and test the generic case runner, math, manifest, runner, summary,
   and documentation machinery.
2. Generate the exact release manifest from the frozen constants.
3. Run the full default/Ruff/site/build gates.
4. Commit and push the implementation plus manifest freeze.
5. Obtain independent and AGY review at that exact pushed SHA.
6. Run or resume the 384 exact cases without changing the frozen code or
   manifest.
7. Finalize the canonical artifact, figures, and report.
8. Re-run full verification and independent/AGY review before merge.

The protected run may be resumed after interruption because every case and
random stream is already fixed. It may not be widened, narrowed, tuned, or
filtered after results are observed.
