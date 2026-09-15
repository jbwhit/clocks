# Certified Weight Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the reviewed R3 tiny-weight normalizer through the actual package, with fast speculative ESS searches and transactional failure on unresolved rounding.

**Architecture:** A private `clocks._weight_normalization` module owns the reviewed DD/Decimal arithmetic and two normalization functions. `clocks.inference` imports the strict function under its existing private name and calls a separately named fast function only in `_next_beta`. Numerical tests import production directly; the prototype remains historical evidence, not a runtime dependency.

**Tech Stack:** Python >=3.12, NumPy >=2.0, SciPy >=1.14, standard-library Decimal/Fraction; existing uv lock, Ruff and pytest. No new dependencies.

**Spec:** `docs/plans/2026-09-14-normalizer-r3-design.md`, with arithmetic derivation `docs/plans/normalizer-r3/dd-analysis.md` and reviewed prototype `docs/plans/normalizer-r3/prototype.py` at `3f7df94594a45e45bedfcf8877ca3bf06a7261fc`.

## Global Constraints

- Every exact subnormal weight must round correctly to nearest-even, or raise `WeightRoundingUndecided` if the explicit precision budget is exhausted.
- A narrow strip above the smallest normal value may also be corrected. No blanket bitwise-normal-weight or global trajectory invariance guarantee.
- Candidate shift `< -650`; zero cutoff `< -800`; denominator retains shifts `>= -160`; component clamp `2**-350`; quotient radius `2**-85`; correction limit `2**52 + 2**20`.
- Decimal precisions are 56, 112, 224; exact subtraction uses 1,500 digits with `Inexact` trapped. Preserve explicit isolated contexts, directed intervals and omitted finite-tail accounting.
- Preserve operation order, constant certificates, and the reviewed arithmetic assumptions: separate IEEE binary64 operations, round-to-nearest-even, gradual underflow. No algebraic simplification or JIT/compiler change.
- Evidence retains `float(logsumexp(values))`; existing centering and invalid-input behavior remain. Correctness is relative to supplied binary64 logs, not upstream exact likelihoods.
- `_next_beta` uses the inexpensive function; accepted `_update` stages use the strict function. Existing checkpoint rollback must restore state, RNG, evidence, histories and diagnostics on cap failure.
- Keep the existing linked worktree/PR. Never copy, activate, recreate or sync its `.venv`; use `uv run --no-sync` and assert package paths inside this worktree before trusting results.
- All shell calls have explicit workdir. Use apply_patch for edits. No unrelated changes, dependency updates, merge or deployment.
- Main session owns long jobs and commits/pushes after reading the complete gate output, in a separate tool call. Attribute each authored commit to the actual implementing model. Agents run focused tests and report; no child commits, pushes, or subagents.

## Review and execution gates

The user approved implementation of R3 on September 15. The design decision is settled; do not re-ask workflow or contract questions. Independent plan review precedes implementation. Each task receives independent spec/quality review, then the full local gate before its commit/push. Task 2 ends with a whole-branch Astra review and read-only Gemini AGY review in a pinned isolated worktree. Findings are reproduced and addressed before final handoff. No merge is part of this plan.

The standard gate is:

```sh
uv run --no-sync python -c 'from pathlib import Path; import clocks.inference as m; assert Path(m.__file__).resolve().is_relative_to(Path.cwd()); print(m.__file__)'
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync pytest --tb=short
uv run --no-sync pytest -m slow --tb=short
```

Task 1 also runs `uv run --no-sync pytest docs/plans/normalizer-r3 --tb=short`, since that remains a CI step until Task 2. Final validation additionally builds the wheel, runs the existing installed-entry-point smoke test, runs four full-suite production mutations, measures real normalizer/update costs, and checks nonempty successful CI for the pushed SHA.

### Task 1: Promote the certified arithmetic and direct numerical tests

**Files:**
- Create `src/clocks/_weight_normalization.py`.
- Create `tests/test_weight_normalization.py` and `tests/test_weight_intervals.py`.
- Read, do not modify, `docs/plans/normalizer-r3/prototype.py`, `test_prototype.py`, `test_intervals.py`, and `dd-analysis.md`.

**Interfaces:**
- Produces `_normalize_log_weights(log_weights: NDArray[np.floating]) -> tuple[NDArray[np.float64], float]` (strict).
- Produces `_normalize_log_weights_fast(log_weights: NDArray[np.floating]) -> tuple[NDArray[np.float64], float]` (shifted scale).
- Produces `WeightRoundingUndecided(RuntimeError)` and the reviewed private arithmetic helpers. No changes to inference bindings in this task.

- [ ] **Step 1: Pin red numerical behavior before adding production arithmetic.**

Start `tests/test_weight_normalization.py` with the regression fixtures/test body from archived `test_prototype.py`, temporarily importing `clocks.inference as p` and naming the call `p._normalize_log_weights`. The exact assertions include:

```python
values = np.array([98.28463995512519, -610.1117785771389])
weights, _ = p._normalize_log_weights(values)
assert weights[1] == np.ldexp(float(4503599627370492), -1074)
```

Run `uv run --no-sync pytest tests/test_weight_normalization.py -k rounding_regressions --tb=short`. Record the wrong numerical outputs, not a collection/import error, as RED evidence. Do not weaken fixtures or expected integers.

- [ ] **Step 2: Promote the reviewed implementation without changing arithmetic.**

Use the complete archived `prototype.py` as the source text. Preserve every arithmetic helper body, constant/certificate, DD reduction step, rounding-cell decision, and Decimal enclosure. Rename `normalize_branch` to `_normalize_log_weights_fast`, and `normalize_proto` to `_normalize_log_weights`, including their internal call. Remove the optional `stats` parameters and the three `if stats is not None` recording blocks; remove only the corresponding trailing argument in `_decide_exactly` calls. These counters belong in external instrumentation, not the package API.

Use the existing inference invalid-normalizer message verbatim:

```python
raise RuntimeError(
    "All particles have zero weight; the prior or forward model is "
    "inconsistent with the observation"
)
```

Add `numpy.typing.NDArray` and the two signatures above. Replace the experimental module docstring with the supported contract, link to the derivation, and a warning not to reassociate DD operations. Keep helper names/constant names unchanged to make proof-to-code comparison straightforward. No public stats object, lazy cache, configuration option, Cython or new algorithm.

- [ ] **Step 3: Port the existing numerical and interval tests onto production imports.**

Replace the temporary test import with `from clocks import _weight_normalization as p`. Port the 17 prototype and 12 interval cases into the two named files, retaining independent literal/Fraction/Decimal expectations. Do not import the archive in runtime tests. Rename calls to `_normalize_log_weights`; use the production `_decide_exactly` signature without stats.

For precision escalation tests, observe calls without modifying the implementation:

```python
precisions = []
original = p._isolated_context
def observing_context(prec, rounding=p.ROUND_HALF_EVEN):
    if rounding == p.ROUND_HALF_EVEN and prec in (56, 112, 224):
        precisions.append(prec)
    return original(prec, rounding)
monkeypatch.setattr(p, "_isolated_context", observing_context)
```

Assert the existing hard cases require `[56, 112]`, and the real cap fixture requires `[56, 112, 224]`. Retain duplicate-result equality; drop the experimental stats-only assertion about deduplication counts. All arithmetic remains real, including the exponentials and directed operations.

Add parameterized invalid-input checks for NaN, positive infinity, and all `-inf`, covering both normalization functions with the existing RuntimeError message prefix. Add bitwise equality of strict/fast healthy weights and exact equality of their evidence results on a finite non-candidate fixture.

- [ ] **Step 4: Focused green, self-review, independent review, then main-session gate/commit.**

Run `uv run --no-sync pytest tests/test_weight_normalization.py tests/test_weight_intervals.py --tb=short`. Report RED/GREEN commands and outputs plus exact promotion differences. Main controller compares operation ASTs against the reviewed prototype, obtains the task review, runs the complete gate and archives review evidence. The four production-inference xfails remain expected until Task 2. Commit/push only after gate output is read.

### Task 2: Integrate filter call sites and verify the actual package

**Files:**
- Modify `src/clocks/inference.py`, `src/clocks/__init__.py`, `tests/test_smc_rigorous.py`, `.github/workflows/ci.yml`, `README.md`.
- Create `tests/test_weight_filter.py`, `scripts/check_normalizer_mutations.py`, `scripts/benchmark_normalizer.py`.
- Update `docs/plans/2026-09-14-normalizer-r3-design.md` with an explicitly dated implementation-status note; create `docs/plans/2026-09-15-normalizer-implementation.md` for final evidence.
- Read the archived `test_filter.py`, `mutations.py`, and existing wheel smoke script; do not alter the historical prototype or proof.

**Interfaces:**
- Consumes `_normalize_log_weights`, `_normalize_log_weights_fast`, and `WeightRoundingUndecided` from `clocks._weight_normalization`.
- Existing `clocks.inference._normalize_log_weights` remains importable; `_next_beta` explicitly calls `_normalize_log_weights_fast`.
- `clocks.WeightRoundingUndecided` is the catchable public exception, exported through `__all__`, matching the existing PhysicsDomainError export convention.

- [ ] **Step 1: Expose the four existing regressions as ordinary failing tests.**

Remove only the strict xfail decorator on `test_normalization_keeps_a_weight_that_is_barely_representable` and `_ROUNDS_TWICE`/its three parameter marks. Keep all five parameter values, IDs, and existing assertions. Run:

```sh
uv run --no-sync pytest tests/test_smc_rigorous.py -k 'barely_representable or subnormal_recovery' --tb=short
```

Record the four expected numerical failures before changing inference. Correct stale docstrings that describe the old expressions as a universal fix; expected values remain independent literals.

- [ ] **Step 2: Wire strict accepted stages and fast speculative stages.**

Remove inference's old function body and import:

```python
from clocks._weight_normalization import (
    _normalize_log_weights,
    _normalize_log_weights_fast,
)
```

In `_next_beta`'s ESS trial only, replace the call with `_normalize_log_weights_fast(candidate_log_weights)`. Leave the accepted `_update` call on `_normalize_log_weights`. Keep `scipy.special.logsumexp` for ModelComparison. Do not change centering calculations, proposal/resampling logic, evidence arithmetic or checkpoint behavior. Update the comment after accepted normalization: supplied centered binary64 logs get the tiny-weight guarantee; centering can still change the upstream real target through rounding.

In `clocks.__init__`, import `WeightRoundingUndecided` directly from the new module and add it to `__all__`. Document the public failure in `ParticleFilter.update` and README: cap exhaustion rolls back the update, and callers should not blindly retry unchanged inputs. Correct rounding is not a promise that every finite population completes.

- [ ] **Step 3: Exercise real integration and rollback, not injected architecture.**

Adapt the literal PARTICLES and `make_filter` from archived `test_filter.py` into `tests/test_weight_filter.py`. The successful Gaussian update runs unpatched production and asserts seven stages. Compare its resulting particles, weights and evidence to a control using the fast accepted-stage function only; label this as measured equality for that case, not universal trajectory invariance.

Demonstrate the same full-beta candidate causes the actual strict function to raise `WeightRoundingUndecided`, while unpatched production update completes. Add a strict-helper trap to an `_next_beta`-only test and assert the returned beta remains valid; this catches accidental speculative certification.

For rollback, adapt the archived later-stage test using real production normalization and exception class. At its second accepted call, verify RNG/evidence progressed, then evaluate the genuine cap-triggering values. Assert exception plus every checkpoint field restored exactly, including the original state object. Do not replace `_next_beta`, use FunctionType, or inject prototype globals.

Add a fresh interpreter regression via `subprocess.run([sys.executable, '-c', code], check=True, ...)` that imports `clocks`, catches the public exception type, and checks the literal seam output. Assert its module path lies inside the active checkout. This directly closes the previous in-process-harness gap.

- [ ] **Step 4: Replace experimental CI coverage with production coverage and tools.**

Remove only the separate `Normalizer design regressions` CI step: all promoted regressions are now collected by default `pytest`. Keep the default suite, slow suite, wheel build and installed-entry-point smoke gates.

Implement `scripts/check_normalizer_mutations.py` as a PEP 723 script using existing numpy/scipy/pytest dependencies. It selects exactly one of the four archived mutation names via argparse, imports and asserts the production normalizer path, installs its in-memory fault before pytest collection, and exits with `pytest.main(["tests", "-m", "", "--tb=short"])`. Port mutation bodies from the archive with strict/fast name changes; when wrapping strict classification, update both the module and inference bindings. No xfail stripping, prototype injection, filesystem edits or environment setup. Print the selected mutation and verified import. Invalid names must fail argument parsing. Main controller runs four separate processes concurrently and checks intended failed tests in each full-suite result.

Implement `scripts/benchmark_normalizer.py` as a PEP 723 script for existing numpy/scipy. Use seed 731, populations 2,000 and 40,000, normal(0,1) heads, one tail from `[-700,-720,-740,-800,-inf]`, median of five warmed calls, and print strict/fast timings with Python/NumPy/platform provenance. Include the 32-particle real filter fixture from the integration test as an explicitly documented fixed workload and measure a fresh filter update; do not import tests from the benchmark. No timing assertions in CI and no worst-case/rarity claim. Main may record results in the evidence document after implementation review.

Mark archived design test/run commands as historical and reproducible at `3f7df94`, because their old binding/xfail assumptions no longer describe current source. Link to the new evidence and current production test/mutation commands. Keep the archived prototype/proof/test files unchanged as evidence; no package runtime code may import them.

- [ ] **Step 5: Focused green and reviewed whole-package finalization.**

Run the numerical, interval, filter and `test_smc_rigorous.py` suites. Self-review call-site routing and public export; report exact outputs. Main obtains independent task review, runs the full local gate and the four whole-suite mutations, then builds/smoke-tests:

```sh
uv run --no-sync python scripts/check_normalizer_mutations.py old_midpoint
uv run --no-sync python scripts/check_normalizer_mutations.py float_classification
uv run --no-sync python scripts/check_normalizer_mutations.py no_dd_interval
uv run --no-sync python scripts/check_normalizer_mutations.py collapsed_decimal_interval
uv run --no-sync python scripts/benchmark_normalizer.py
uv build
uv run --no-sync python scripts/check_wheel_entrypoints.py dist/*.whl python3
```

Main records measured results separately from proof claims, commits/pushes, and obtains independent whole-branch Astra and isolated AGY reviews of the committed SHA. Reproduce findings, dispatch scoped fixes and re-review. Complete only with no unresolved blocking findings, a clean worktree, and nonempty successful CI on the final SHA. Post attributed review trail on PR #18; leave merge/deployment untouched.
