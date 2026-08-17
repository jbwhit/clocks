# Echolocation Population Reliability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build, preregister, execute, validate, and publish a 384-case population reliability study for the current 27-clock echolocation head, paired with analytic local-identifiability diagnostics.

**Architecture:** A new private `clocks._reliability` module owns the population specification, analytic Jacobian/Fisher calculations, canonical manifests/results, summaries, checkpoints, and figures. `clocks._scenarios` gains a generic arbitrary-truth runner while its certified fixed-scenario interface stays unchanged. A thin script exposes manifest, run/resume, finalize, and figure operations; the release manifest is frozen and reviewed before the protected inference run.

**Tech Stack:** Python 3.12+, NumPy, SciPy, Matplotlib, argparse, multiprocessing, canonical JSON, pytest, Ruff, Quarto, uv.

---

## Execution rules

- Use @superpowers:test-driven-development for every production change.
- Use @superpowers:systematic-debugging for any unexpected failure.
- Use @scientific-visualization when implementing the release figures.
- Use @superpowers:verification-before-completion before every commit/push or success claim.
- Use @superpowers:requesting-code-review after every task and at both freeze gates.
- Do not run the 384-case release study before Task 9 has been committed, pushed, and independently approved at its exact SHA.
- Do not alter the release manifest, study constants, metrics, analysis, or controls after the first protected result is observed. A real defect requires an explicit version bump and invalidation record.
- Preserve the existing fixed-case development/certification artifacts byte-for-byte.
- Add the actual authoring model's `Co-Authored-By` trailer to every commit, per the global instructions.

## Task 1: Add exact local-identifiability mathematics

**Files:**

- Create: `src/clocks/_reliability.py`
- Create: `tests/test_reliability_math.py`
- Reference: `src/clocks/_scenarios.py:185-315`
- Reference: `src/clocks/physics.py:57-149`

### Step 1: Write failing tangent-basis and Jacobian tests

Add tests that construct the existing head and several off-axis valid truths.
The core finite-difference test should have this shape:

```python
def test_analytic_contrast_jacobian_matches_central_difference() -> None:
    head = build_head_lattice()
    position = np.array([2.4, -3.1, 4.2])
    mass = 0.05
    actual_position, actual_mass = contrast_jacobian(position, mass, head)

    def contrasted(pos: np.ndarray, m: float) -> np.ndarray:
        rates = clock_rates(
            MassConfig(pos.reshape(1, 3), np.array([m])), head
        )
        return contrast_matrix(len(head.positions)) @ rates

    position_fd = np.column_stack(
        [
            (contrasted(position + eps * axis, mass)
             - contrasted(position - eps * axis, mass)) / (2.0 * eps)
            for axis in np.eye(3)
        ]
    )
    mass_fd = (
        contrasted(position, mass + mass_eps)
        - contrasted(position, mass - mass_eps)
    ) / (2.0 * mass_eps)
    np.testing.assert_allclose(actual_position, position_fd, rtol=2e-6, atol=1e-10)
    np.testing.assert_allclose(actual_mass, mass_fd, rtol=2e-6, atol=1e-10)
```

Also require:

- deterministic orthonormal tangent vectors perpendicular to the direction;
- rejection of zero/nonfinite/wrong-shape positions and invalid mass/noise/count;
- warning-free rejection of complex inputs;
- exact output shapes `(26, 3)` and `(26,)`; and
- no finite-difference code in production.

### Step 2: Run the focused RED tests

Run:

```bash
uv run pytest tests/test_reliability_math.py -q -W error
```

Expected: collection or import failure because `clocks._reliability` and its
symbols do not exist.

### Step 3: Implement the exact contrast Jacobian

In `src/clocks/_reliability.py`, add validated helpers equivalent to:

```python
def contrast_jacobian(
    position: object,
    mass: object,
    clock_array: ClockArray,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    p = finite_float_array("position", position, ndim=1)
    m = finite_float("mass", mass)
    if p.shape != (3,) or m <= 0.0:
        raise ValueError(...)
    diff = clock_array.positions - p
    distance = np.sqrt(np.sum(diff**2, axis=1))
    rates = clock_rates(MassConfig(p.reshape(1, 3), np.array([m])), clock_array)
    d_position = -m * diff / (rates[:, None] * distance[:, None] ** 3)
    d_mass = -1.0 / (rates * distance)
    q = contrast_matrix(len(clock_array.positions))
    return q @ d_position, q @ d_mass
```

Use the project's strict validators and narrow `np.errstate` only around
arithmetic whose nonfinite result is explicitly checked. Do not clamp a
singularity or strong-field state.

### Step 4: Add failing Fisher-invariance and scaling tests

Test `local_identifiability` for:

- singular values invariant under a supplied 2-D tangent-plane rotation;
- angular loading equal to the sum of both angular squared loadings;
- all loading groups nonnegative and summing to one;
- singular values scaling by `sqrt(N)` and inversely with `sigma`;
- rank tolerance equal to `eps * max(A.shape) * s_max`;
- descending finite nonnegative singular values;
- a finite condition number only at numerical rank four, otherwise `None`;
- Cramer-Rao standard deviations present only at numerical rank four; and
- information `A.T @ A` symmetric positive semidefinite.

### Step 5: Implement dimensionless Fisher diagnostics

Add frozen/typed `IdentifiabilityResult` data and:

```python
def local_identifiability(
    position: object,
    mass: object,
    clock_array: ClockArray,
    *,
    n_observations: int,
    noise_std: float,
) -> IdentifiabilityResult:
    ...
```

Build columns in the exact order
`(angular_1, angular_2, log_range, log_mass)`, compute
`A = sqrt(n_observations) / noise_std * J`, and use `np.linalg.svd` without a
scientific cutoff. Square the weakest right-singular vector and publish
combined angular, log-range, and log-mass loadings. Make array snapshots deeply
immutable.

### Step 6: Run focused and regression tests

Run:

```bash
uv run pytest tests/test_reliability_math.py tests/test_physics.py tests/test_scenarios.py -q -W error
uv run ruff format --check src/clocks/_reliability.py tests/test_reliability_math.py
uv run ruff check src/clocks/_reliability.py tests/test_reliability_math.py
```

Expected: all pass.

### Step 7: Commit

```bash
git add src/clocks/_reliability.py tests/test_reliability_math.py
git commit -m "feat: add echolocation identifiability diagnostics"
git push
```

## Task 2: Generalize echolocation execution to arbitrary frozen truths

**Files:**

- Modify: `src/clocks/_scenarios.py:185-465`
- Modify: `tests/test_scenarios.py`
- Modify: `tests/test_acceptance_echolocation_3d.py`
- Test: `tests/test_reliability_math.py`

### Step 1: Write failing arbitrary-truth and seed-separation tests

Add a small real test using an off-axis truth not equal to `ECHO_DIRECTION`,
minimal particles/observations, and deliberately distinct observation and
inference seeds. Require a result containing immutable mean/std arrays,
central marginal 95% bounds, existing errors, angular/log errors, residual,
evaluation count, and seed provenance.

Add a spy test proving:

```python
run_echolocation_case(
    truth_position=...,
    truth_mass=...,
    observation_seed=101,
    inference_seed=202,
    ...,
)
```

passes 101 only to simulation and 202 only to the filter.

### Step 2: Write failing weighted-quantile tests

Cover exact hand calculations, unsorted particles, repeated values, nonuniform
weights, endpoint quantiles, zero/negative/nonfinite weights, complex data,
and shape mismatch. The public result must not assume terminal SMC weights are
uniform.

### Step 3: Run RED tests

```bash
uv run pytest tests/test_scenarios.py tests/test_acceptance_echolocation_3d.py \
  -k "arbitrary or quantile or distinct_seed or fixed_echo" -q -W error
```

Expected: failures because the generic runner and weighted quantile helper are
absent.

### Step 4: Implement generic observations, filter execution, and summaries

Refactor without changing the certified wrapper:

```python
def run_echolocation_case(
    *,
    truth_position: NDArray[np.floating],
    truth_mass: float,
    observation_seed: int,
    inference_seed: int,
    n_particles: int = ECHO_N_PARTICLES,
    n_observations: int = ECHO_N_OBSERVATIONS,
    noise_std: float = ECHO_NOISE_STD,
    ess_target: float = ECHO_ESS_TARGET,
    rejuvenation_steps: int = ECHO_REJUVENATION_STEPS,
    proposal_scale: float = ECHO_PROPOSAL_SCALE,
) -> EcholocationCaseResult:
    ...
```

Validate `truth_position` against the exterior minimum and strict physical
support. Use `np.linalg.norm(position) / ECHO_R_HEAD` for range and keep the
existing rectangular inference prior. Compute weighted marginal 0.025 and
0.975 quantiles from the final `ParticleState` before discarding particles.

Make `run_echolocation_3d(seed, range_r, ...)` call this path with the existing
fixed truth, mass, and the same seed for both streams so its historical
deterministic contract is unchanged. Convert the generic result back to the
existing `EchoRunResult` exact field set.

### Step 5: Prove the certified behavior is unchanged

Run:

```bash
uv run pytest tests/test_scenarios.py tests/test_acceptance_echolocation_3d.py \
  tests/test_echo_study.py -m "not slow" -q -W error
```

Expected: all current fixed-scenario and new generic tests pass. Do not run the
slow certification replay solely for this task.

### Step 6: Commit

```bash
git add src/clocks/_scenarios.py tests/test_scenarios.py \
  tests/test_acceptance_echolocation_3d.py tests/test_reliability_math.py
git commit -m "feat: run arbitrary echolocation truths"
git push
```

## Task 3: Generate and semantically validate population manifests

**Files:**

- Modify: `src/clocks/_reliability.py`
- Create: `tests/test_reliability_manifest.py`
- Reference: `src/clocks/_calibration.py:80-174`

### Step 1: Write failing release-population tests

Define literal expected constants in tests, independent of production:

```python
EXPECTED_MASTER_SEED = 20260817
EXPECTED_N_STRATA = 6
EXPECTED_CASES_PER_STRATUM = 64
EXPECTED_RANGE = (2.0, 8.0)
EXPECTED_MASS = (0.02, 0.08)
```

Require `generate_release_manifest()` to produce:

- 384 uniquely identified cases;
- exactly 64 cases in every stratum;
- log-spaced edges including exact endpoints 2 and 8;
- unit directions and positions equal to
  `direction * range_r * ECHO_R_HEAD`;
- range/mass values in their declared half-open strata/population, with the
  final upper boundary handled explicitly;
- valid weak-field truths for the actual head;
- distinct parameter-generation/observation-noise/inference seeds within each
  case and no duplicate stream seeds across the release manifest;
- immutable returned snapshots; and
- canonical identical bytes across two invocations.

### Step 2: Write semantic mutation tests before implementation

Parametrize mutations over every semantic class: schema/study version, status,
master seed, generator, head geometry, distributions, controls, endpoint,
interval method, stratum edges/allocation, case ID/order, direction norm,
position reconstruction, range/mass bounds, physical validity, seeds, duplicate
cases, missing/extra cases, bool-as-number, nonfinite, complex before encoding,
and manifest hash.

Each invalid manifest must raise `ValueError` with a field-specific message.

### Step 3: Run RED tests

```bash
uv run pytest tests/test_reliability_manifest.py -q -W error
```

Expected: missing manifest symbols.

### Step 4: Implement constants, typed cases, generator, and canonical hash

Add constants including:

```python
RELIABILITY_SCHEMA_VERSION = 1
RELIABILITY_STUDY_VERSION = 1
RELIABILITY_MASTER_SEED = 20260817
RELIABILITY_N_STRATA = 6
RELIABILITY_CASES_PER_STRATUM = 64
RELIABILITY_RANGE_BOUNDS = (2.0, 8.0)
RELIABILITY_MASS_BOUNDS = (0.02, 0.08)
```

Use one root `SeedSequence`; spawn one analysis child and 384 case children;
spawn independent parameter, observation, and inference children inside each
case. Store realized directions/ranges/masses and all three integer stream
seeds. Hash the canonical semantic payload with the hash field omitted. Do not
include a wall-clock timestamp or other environment-dependent metadata.

Implement strict `validate_manifest`, `encode_manifest`, `load_manifest`, and
same-directory atomic `write_manifest`. Never use generic truthiness for JSON
numbers; reject booleans explicitly.

### Step 5: Run focused tests and inspect deterministic bytes

```bash
uv run pytest tests/test_reliability_manifest.py tests/test_reliability_math.py -q -W error
uv run ruff format --check src/clocks/_reliability.py tests/test_reliability_manifest.py
uv run ruff check src/clocks/_reliability.py tests/test_reliability_manifest.py
```

Expected: all pass. Do not create the release manifest in `docs/reliability/`
yet; that is the freeze task.

### Step 6: Commit

```bash
git add src/clocks/_reliability.py tests/test_reliability_manifest.py
git commit -m "feat: preregister echolocation populations"
git push
```

## Task 4: Add result semantics and preregistered statistical summaries

**Files:**

- Modify: `src/clocks/_reliability.py`
- Create: `tests/test_reliability_statistics.py`
- Modify: `tests/test_reliability_manifest.py`

### Step 1: Write failing Wilson-interval tests

Test hand-computed cases `(0, 64)`, `(32, 64)`, `(64, 64)`, invalid counts,
and monotonic bounds. Use the literal 95% normal quantile
`1.959963984540054`; do not call a configurable confidence-level selector in
release summaries.

### Step 2: Write failing stratified-bootstrap tests

Require resampling within, never across, strata. Use a tiny two-stratum fixture
and a fixed analysis seed. Assert deterministic sample statistics and interval
bounds using 10,000 replicates and `np.quantile(..., method="linear")`.
All-success and all-failure fixtures must return degenerate overall intervals
at one and zero respectively.

### Step 3: Write failing result semantic tests

Build a valid synthetic record and independently recompute:

- truth position/range/direction/mass from the manifest case;
- position, mass, angular, log-range, and log-mass errors from mean/truth;
- binary pass from literal tolerances;
- coverage booleans from stored marginal interval bounds and truth;
- SNR from the strict forward model;
- Fisher diagnostics from truth; and
- exact seed/control provenance.

Mutation tests must corrupt each stored derived field and prove validation
fails. Interval bounds are validated for type/order/coverage; their numerical
quantiles cannot be reconstructed without the discarded particle cloud and are
therefore trusted only through the tested runner.

### Step 4: Implement statistics and result validation

Add:

```python
def wilson_interval(successes: int, total: int) -> tuple[float, float]: ...
def summarize_population(manifest: Mapping, records: Sequence[Mapping]) -> dict: ...
def validate_result_record(case: Mapping, record: Mapping) -> None: ...
def validate_complete_results(manifest: Mapping, document: Mapping) -> None: ...
```

Summaries contain six Wilson rows, equal-weighted overall point estimate,
stratified-bootstrap interval, continuous median/q10/q90 rows, marginal
coverage Wilson rows, and no pass/fail verdict.

### Step 5: Run GREEN tests

```bash
uv run pytest tests/test_reliability_statistics.py \
  tests/test_reliability_manifest.py tests/test_reliability_math.py -q -W error
```

Expected: all pass.

### Step 6: Commit

```bash
git add src/clocks/_reliability.py tests/test_reliability_statistics.py \
  tests/test_reliability_manifest.py
git commit -m "feat: summarize population reliability"
git push
```

## Task 5: Implement atomic checkpoint, resume, and finalization

**Files:**

- Modify: `src/clocks/_reliability.py`
- Create: `tests/test_reliability_runner.py`

### Step 1: Write failing injected-runner tests

Use small development manifests and an injected deterministic fake runner.
Cover:

- empty start schedules every case once;
- a valid partial checkpoint schedules only missing IDs;
- results are canonicalized by manifest case order despite unordered completion;
- worker failure preserves the last complete atomic checkpoint;
- rerunning a complete checkpoint performs no inference;
- duplicate, foreign, corrupt, or wrong-manifest checkpoint records abort before
  a worker is called; and
- a sentinel destination remains byte-identical on validation failure.

### Step 2: Write failing finalization refusal tests

Require refusal for 383/384 release cases, development identity, missing/extra
case, mismatched manifest hash, corrupt summary, and noncanonical fields.
Require complete valid finalization to be atomic and byte-stable.

### Step 3: Run RED tests

```bash
uv run pytest tests/test_reliability_runner.py -q -W error
```

Expected: missing runner/checkpoint/finalizer functions.

### Step 4: Implement the case worker

Add a top-level pickleable worker that:

1. reads one frozen case;
2. calls `run_echolocation_case` with exact truth/seeds/controls;
3. computes local identifiability from the same truth;
4. constructs all empirical and provenance fields; and
5. validates the record before returning it.

Do not let workers write shared files.

### Step 5: Implement parent-owned checkpointing and resume

Implement a parent orchestrator with injectable runner/executor for tests. The
production path may use `multiprocessing.Pool.imap_unordered`, but the parent
must validate each record and atomically replace the checkpoint after every
accepted case. A worker exception propagates after already accepted records
have been checkpointed.

### Step 6: Implement final result encoding

Finalization validates exact completeness, recomputes summaries, and writes
canonical JSON atomically. Store the manifest content hash and a result payload
hash. Loading must verify both before exposing data.

### Step 7: Run focused and full tests

```bash
uv run pytest tests/test_reliability_runner.py tests/test_reliability_statistics.py \
  tests/test_reliability_manifest.py -q -W error
uv run pytest -q
uv run ruff format --check .
uv run ruff check .
git diff --check
```

Expected: all pass; the default suite runs no release inference.

### Step 8: Commit

```bash
git add src/clocks/_reliability.py tests/test_reliability_runner.py \
  tests/test_reliability_statistics.py tests/test_reliability_manifest.py
git commit -m "feat: checkpoint population studies atomically"
git push
```

## Task 6: Add the thin population-study CLI

**Files:**

- Create: `scripts/run_echolocation_population.py`
- Modify: `tests/test_cli.py`
- Modify: `README.md`

### Step 1: Write failing parser/help tests

Require subcommands:

```text
manifest --output PATH [--development-cases-per-stratum N]
        [--development-n-particles N] [--development-n-observations N]
run --manifest PATH --checkpoint PATH [--workers N]
finalize --manifest PATH --checkpoint PATH --output PATH
figure --results PATH --output-dir PATH
```

Require `--help` to exit zero without importing pyplot. Reject workers below
one, invalid development counts, mutually incompatible flags, output paths
that alias inputs, and any release-population override. The three
`--development-*` options are manifest-only controls: using any of them marks
the manifest as development evidence, and `run` accepts no scientific
override.

### Step 2: Run RED tests

```bash
uv run pytest tests/test_cli.py -k population -q -W error
```

Expected: script is absent.

### Step 3: Implement argparse and narrow command dispatch

The script must contain no scientific formulas or validation duplication. Set
the Matplotlib backend before importing figure code. Convert package
`ValueError` failures to `parser.error` without swallowing unexpected errors.

Default release paths are:

```text
docs/reliability/echolocation_population_v1_manifest.json
output/echolocation_population_v1_checkpoint.json
docs/reliability/echolocation_population_v1_results.json
```

### Step 4: Add a tiny real smoke command

Generate a development manifest with one case per stratum and minimal
particle/observation controls encoded in that manifest, then run one selected
case without runtime scientific overrides. Prove it exercises real
simulation/inference and cannot finalize as release evidence.

### Step 5: Run CLI and full checks

```bash
uv run pytest tests/test_cli.py tests/test_reliability_runner.py -q -W error
uv run python scripts/run_echolocation_population.py --help
uv run python scripts/run_echolocation_population.py manifest --help
uv run ruff format --check .
uv run ruff check .
```

Expected: all pass.

### Step 6: Commit

```bash
git add scripts/run_echolocation_population.py tests/test_cli.py README.md
git commit -m "feat: add population study command"
git push
```

## Task 7: Add publication figures and report contracts without results

**Files:**

- Modify: `src/clocks/_reliability.py`
- Create: `tests/test_reliability_figures.py`
- Modify: `tests/test_site_prose_contract.py`
- Modify: `scripts/check_site_links.py`
- Modify: `docs/someday-maybe.md`

### Step 1: Invoke the scientific-visualization skill

Before figure implementation, read and follow @scientific-visualization. Use a
colorblind-safe palette, legible uncertainty intervals, shared typography, and
PNG output suitable for the current site. Do not fabricate release assets.

### Step 2: Write failing fake-artifact figure tests

From a complete synthetic artifact, require four nonempty PNGs:

```text
echolocation_population_reliability.png
echolocation_population_resolution.png
echolocation_population_information.png
echolocation_population_weakest_components.png
```

Inspect axes/artist semantics before closing figures: Wilson intervals,
stratum order, logarithmic axes only where positive, angular/range/mass labels,
and weakest-component stacks summing to one. Include all-zero/all-one and
rank-deficient synthetic cases.

### Step 3: Implement pure figure construction

Separate `build_*_figure(results) -> Figure` from
`write_population_figures(results, output_dir)`. Do not import pyplot at module
import time. Figures consume only validated result artifacts.

### Step 4: Extend prose/link contracts prospectively

Update `docs/someday-maybe.md` to mark population reliability as in progress,
not complete. Add contract patterns that forbid describing a local Fisher
diagnostic as global identifiability, a fixed-case artifact as population
reliability, or an unrun population study as evidence. Do not add the final
site/report page or release images yet.

### Step 5: Run tests and site contracts

```bash
uv run pytest tests/test_reliability_figures.py tests/test_site_prose_contract.py -q
uv run python scripts/check_site_links.py
uv run ruff format --check .
uv run ruff check .
```

Expected: all pass without release results.

### Step 6: Commit

```bash
git add src/clocks/_reliability.py tests/test_reliability_figures.py \
  tests/test_site_prose_contract.py scripts/check_site_links.py docs/someday-maybe.md
git commit -m "feat: prepare reliability study reporting"
git push
```

## Task 8: Verify implementation before freezing the manifest

**Files:**

- Modify only files required by verified failures.

### Step 1: Run the complete non-protected matrix

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
git diff --check main...HEAD
uv run pytest -q -W error
uv run pytest -m slow -q -W error
uv build
uv run python scripts/check_wheel_entrypoints.py dist/clocks-0.1.0-py3-none-any.whl python3
uv run python scripts/check_site_links.py
cd site && uv run --frozen quarto render
```

Expected: all existing and new tests pass; the old calibration artifacts and
generated assets remain byte-identical.

### Step 2: Run exact mathematical probes

```bash
uv run pytest tests/test_reliability_math.py -q -W error
uv run pytest tests/test_reliability_manifest.py \
  tests/test_reliability_statistics.py tests/test_reliability_runner.py -q -W error
```

Expected: all pass.

### Step 3: Request an independent whole-slice review

Review `main...HEAD` against the approved design. Resolve every verified
Critical/Important finding test-first; record Minor non-correctness ideas for
later.

### Step 4: Commit any review fixes separately

```bash
git add <verified-review-fix-files>
git commit -m "fix: address reliability study review"
git push
```

## Task 9: Generate and freeze the exact 384-case manifest

**Files:**

- Create: `docs/reliability/echolocation_population_v1_manifest.json`
- Modify: `tests/test_reliability_manifest.py`
- Modify: `docs/plans/2026-08-17-echolocation-population-reliability-design.md`
  only if the implementation revealed a pre-result specification defect.

### Step 1: Generate the release manifest exactly once for review

```bash
uv run python scripts/run_echolocation_population.py manifest \
  --output docs/reliability/echolocation_population_v1_manifest.json
```

This performs no inference. Capture the printed semantic hash.

### Step 2: Validate and independently inspect the manifest

```bash
uv run pytest tests/test_reliability_manifest.py -q -W error
uv run python - <<'PY'
from pathlib import Path
from clocks._reliability import load_manifest

manifest = load_manifest(
    Path("docs/reliability/echolocation_population_v1_manifest.json")
)
print(manifest["manifest_sha256"])
print(len(manifest["cases"]))
PY
```

Expected: 384 cases, exact six-by-64 allocation, and the same hash printed by
generation.

### Step 3: Pin the manifest hash in an ordinary test

Add a literal expected SHA-256 and prove the tracked bytes decode, validate,
and reproduce the semantic hash. Do not regenerate cases inside the assertion
and compare only derived summaries; inspect the tracked artifact itself.

### Step 4: Re-run the full non-protected matrix

Repeat Task 8 Steps 1-2. No protected population inference may have run yet.

### Step 5: Commit and push the freeze

```bash
git add docs/reliability/echolocation_population_v1_manifest.json \
  tests/test_reliability_manifest.py
git commit -m "test: freeze echolocation population manifest"
git push
```

Record the exact freeze SHA. Do not amend or force-push it.

### Step 6: Obtain exact-SHA independent and AGY approval

Create clean detached review worktrees at the freeze SHA. Require reviewers to
check population generation, random-stream independence, physical support,
Jacobian/Fisher math, endpoint/interval definitions, checkpoint integrity, and
absence of result-dependent tuning. End with a hard `LGTM`/`READY TO RUN` or
`NEEDS REVISION` verdict.

Verified fixes require a new freeze commit and a fresh review. Do not begin
Task 10 until the final exact freeze SHA is approved.

## Task 10: Execute or resume the protected 384-case population

**Files:**

- Create untracked: `output/echolocation_population_v1_checkpoint.json`
- No tracked source/specification changes are permitted in this task.

### Step 1: Reconfirm the reviewed freeze

```bash
git status --short
git rev-parse HEAD
git rev-parse @{u}
```

Expected: clean status and identical local/remote reviewed freeze SHA.

### Step 2: Run the exact manifest

```bash
uv run python scripts/run_echolocation_population.py run \
  --manifest docs/reliability/echolocation_population_v1_manifest.json \
  --checkpoint output/echolocation_population_v1_checkpoint.json \
  --workers 8
```

If interrupted, rerun the identical command. Never delete valid completed
records to obtain a different stochastic realization.

### Step 3: Validate checkpoint completeness without interpreting results

```bash
uv run python scripts/run_echolocation_population.py finalize \
  --manifest docs/reliability/echolocation_population_v1_manifest.json \
  --checkpoint output/echolocation_population_v1_checkpoint.json \
  --output output/echolocation_population_v1_results.json
```

Expected: exact 384-case completion and a canonical untracked final artifact.
If validation finds a defect, stop and preserve all raw/checkpoint evidence.
Do not tune or rerun selectively.

## Task 11: Publish the result artifact, figures, and honest report

**Files:**

- Create: `docs/reliability/echolocation_population_v1_results.json`
- Create: `docs/2026-08-17-echolocation-population-reliability.md`
- Create: `assets/echolocation_population_reliability.png`
- Create: `assets/echolocation_population_resolution.png`
- Create: `assets/echolocation_population_information.png`
- Create: `assets/echolocation_population_weakest_components.png`
- Create matching files under: `site/assets/`
- Create: `site/story/echolocation-population-reliability.qmd`
- Modify: `site/story/gravitational-echolocation.qmd`
- Modify: `site/_quarto.yml`
- Modify: `README.md`
- Modify: `docs/someday-maybe.md`
- Modify: `scripts/check_site_links.py`
- Modify: `tests/test_reliability_statistics.py`
- Modify: `tests/test_generated_assets.py`
- Modify: `tests/test_site_prose_contract.py`

### Step 1: Copy only the validated canonical artifact

Use the package finalizer/archive operation; do not hand-edit JSON:

```bash
uv run python scripts/run_echolocation_population.py finalize \
  --manifest docs/reliability/echolocation_population_v1_manifest.json \
  --checkpoint output/echolocation_population_v1_checkpoint.json \
  --output docs/reliability/echolocation_population_v1_results.json
```

Pin the source/checkpoint and tracked artifact hashes in the report and tests.

### Step 2: Generate release figures from the tracked artifact

```bash
uv run python scripts/run_echolocation_population.py figure \
  --results docs/reliability/echolocation_population_v1_results.json \
  --output-dir assets
```

Copy exact PNG bytes to `site/assets/`; assert paired files are byte-identical.
Do not rerun inference during figure generation.

### Step 3: Write the report from validated summaries

Record every preregistered table/figure and limitation, including outcomes that
are inconvenient. Explicitly state:

- the declared population and fixed head;
- per-stratum Wilson intervals;
- the stratified overall estimate/interval;
- continuous empirical summaries and coverage;
- local Fisher findings as local diagnostics only;
- no threshold, tuning, or geometry comparison; and
- exact freeze/result provenance.

Do not add post-hoc gates or imply that Fisher information proves global
identifiability.

### Step 4: Update site and README

Add the new story page after Gravitational Echolocation, link both directions,
and update reproducibility instructions. Mark population reliability shipped in
`docs/someday-maybe.md`; retain alternative-head comparison as future work.

### Step 5: Add tracked-artifact and prose contracts

Tests must load and semantically validate the actual tracked manifest/results,
recompute summaries, pin hashes, require all four asset pairs, verify PNG magic,
and prove current prose does not overstate the population or Fisher claims.

### Step 6: Render and run the full matrix

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest -q -W error
uv run pytest -m slow -q -W error
uv build
uv run python scripts/check_wheel_entrypoints.py dist/clocks-0.1.0-py3-none-any.whl python3
uv run python scripts/check_site_links.py
cd site && uv run --frozen quarto render
git diff --check
```

Expected: all pass; every release figure is generated solely from the tracked
validated result artifact.

### Step 7: Commit and push publication evidence

```bash
git add docs/reliability docs/2026-08-17-echolocation-population-reliability.md \
  assets site README.md docs/someday-maybe.md scripts/check_site_links.py \
  tests/test_reliability_statistics.py tests/test_generated_assets.py \
  tests/test_site_prose_contract.py
git commit -m "docs: publish echolocation population reliability"
git push
```

## Task 12: Final review, PR, merge, and deployment

**Files:**

- Modify only files required by reproduced review findings.

### Step 1: Run final clean-state verification

Repeat Task 11 Step 6, then:

```bash
git status --short
git log --oneline main..HEAD
git rev-parse HEAD
git rev-parse @{u}
```

Expected: clean status and identical local/remote SHA.

### Step 2: Request independent whole-branch review

Review physical support, random population semantics, exact Jacobian signs,
dimensionless Fisher transformation, SMC seed separation, posterior quantiles,
statistical intervals, checkpoint integrity, tracked-artifact provenance,
figures, and prose claims. Reproduce every finding before changing code.

### Step 3: Address verified findings test-first

For each real defect: add a failing focused test, implement the smallest fix,
rerun focused tests, then repeat the complete matrix. Commit review fixes
separately and push.

### Step 4: Obtain AGY review at the final pushed SHA

Use a clean detached worktree and the repository AGY protocol. Sanity-probe the
directory/SHA/content before the full review. Require a hard `LGTM` or
`NEEDS REVISION`; treat findings as leads and never permit AGY to modify the
live branch.

### Step 5: Create/ready the PR and verify CI

Open a draft PR with population, analysis, freeze, protected-run, and local
verification details. After AGY LGTM, mark ready. Confirm a nonempty successful
PR CI conclusion and clean mergeability.

### Step 6: Merge under AGY approval and verify deployment

Merge using the repository's normal merge-commit strategy. Fast-forward local
`main` with `git pull --no-rebase --ff-only`, assert the merge is present, and
wait for nonempty successful post-merge CI and Pages deployment conclusions.
Then remove only the clean merged feature/review worktrees and merged branches.
