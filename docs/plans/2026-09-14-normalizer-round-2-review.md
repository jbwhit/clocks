# Normalizer design: round-2 findings and continuation

Status: **revision 2 rejected; no production implementation of the prototype**.
This is the current continuation record for [PR #18](https://github.com/jbwhit/clocks/pull/18).
It supersedes the correctness, cost, environment, and readiness claims in the
archived revision-2 design. It is not an approved revision-3 specification.

The interrupted Claude session's Codex review completed on September 14, 2026.
Its reviewer was `gpt-6-astra`, effort `xhigh`, session
`01a0a066-7e33-7703-aab8-807cf2e39988`. Its final verdict was:

> NOT SOUND ENOUGH TO IMPLEMENT

The findings below were independently reproduced during the continuation, using
the existing worktree environment without synchronizing or replacing it.

## Verified environment and baseline

- Worktree: `/Users/jonathan/projects/clocks/.worktrees/normalizer`.
- Branch: `claude-weight-normalization`; source SHA:
  `7e9d4c9b0052deaa82348f8c1526648d9639592d`.
- Python 3.13.13, NumPy 2.4.2, SciPy 1.17.1, macOS x86_64.
- `clocks.inference.__file__` resolves to this worktree's `src/clocks/inference.py`.
  This was asserted inside the same Python process that invoked pytest.
- Formatter and linter pass. Default suite: **898 passed, 4 xfailed, 2 slow
  tests deselected**. Slow-suite results are recorded separately in the PR trail.

The earlier revision-2 document's Python 3.12 environment description, old SHA,
test counts, and alleged pre-existing physics failure are obsolete. The prior
round-1 review tested a contaminated environment importing another checkout. Its
prototype-only numerical measurements are separate evidence; they do not make
that suite result valid. Neither the old suite counts nor the physics-failure
claim should be reused.

## Confirmed findings

### 1. The decision tests only one rounding boundary

Input:

```python
[-14.171583200817665, -14.628378654822516, -722.077393183382,
 -44.60327978867792, -78.51114028012964]
```

For entry 2, in units of `2**-1074` (one quantum):

| Quantity | Measured result |
|---|---|
| 650-digit reference | `4503599627370499.5 - 3.09565900865e-27` |
| Correctly rounded reference | `4503599627370499` |
| Revision-2 output | `4503599627370500` |
| Decimal fallback count | `0` |

The double-double estimate is `qh = 4503599627370500`, `ql = -0.5`.
The code checks `floor(qh) + 0.5`; the uncertainty is at `qh - 0.5`.
This failure does not depend on the disputed exponential error bound. Although
the exact weight is slightly normal, the prototype classifies it as deep and
explicitly promises correct rounding for it.

### 2. Exact subnormal coverage contradicts float-normal bitwise preservation

Input `[98.28463995512519, -610.1117785771389]` has a second exact weight below
the smallest normal double. The reference rounds to `4503599627370492` quanta;
the prototype returns `4503599627370620` and reports `n_deep = 0`.

Thus preserving every float-normal branch result bitwise and correctly rounding
every exact subnormal weight are incompatible requirements. The original design
acknowledges a classification seam in section 2 but contradicts that limitation
in its summary and whole-range claims.

Recommendation for revision 3: prioritize exact-subnormal coverage and permit
corrections near the normal/subnormal boundary. The alternative is to narrow the
guarantee consistently to float-classified entries. The recommended policy is
proposed here, not recorded as an explicit user decision. Either choice still
requires fixing finding 1.

### 3. The precision cap can abort a real filter's speculative search

A constructed finite population of 32 particles, identity forward model,
observation zero, unit Gaussian noise, and default `ess_target = 0.8` produces
`Q - 0.5 = 7.7346280493e-250` for one candidate weight.

Measured through `ParticleFilter.update()`:

- Shifted-scale branch: completes seven tempering stages.
- Revision 2: raises `WeightRoundingUndecided` at 224 Decimal digits during
  `_next_beta`'s initial `ess_at(1.0)` check.
- The candidate being certified would be rejected by the ESS search.

This establishes reachability through a Gaussian model with deliberately chosen
particles. It does not establish a meaningful probability under random sampling.
The early exception left the checkpoint unchanged; that alone does not test
rollback after partial progress.

### 4. The exponential certificate has unfulfilled proof obligations

Executed precondition failures:

- At reduction `k = -19`, adding `k * LN2_LO2` makes the resulting pair
  unnormalized: `p1 != RN(p1 + p2)` before a normalized-input theorem is applied.
- `[0, -5e-324, -745]` reaches a Horner product with a nonzero exact result for
  which `_two_prod` returns `(-0.0, 0.0)`. The claimed absence of underflow is false.

Neither probe establishes a violation of `EXP_DD_EPS`. The recovered review
measured a worst exponential error of `2**-103.91` across 20,000 arguments, within
the proposed `2**-96` budget. This is empirical support, not a completed proof.
The independent continuation reviewer also confirmed that fixing normalization
alone does not settle the underflow issue: tiny shift corrections, their squares,
and low-word cross-products need explicit treatment.

The original review found the no-cancellation argument appropriate at the Horner
and positive-sum sites. The certificate must nevertheless be repaired before
those measurements can support a correctness guarantee.

### 5. Both stage costs and Decimal fallback costs were understated

Fresh continuation measurements; timings are machine- and load-dependent:

| Input | Result |
|---|---|
| Seed 731; 1,800 `N(0,1)` and 200 `N(20,1)` particles; `ll = -0.5*x*x*300`; target ESS 1,000 | 55 of 61 candidate calls take the DD path; stage median 85.1 ms vs 12.9 ms on main |
| 2,000-particle Gaussian construction with one residual-matched helper | `Q - 0.5 = 3.30377e-28`; one Decimal round with 1,997 terms; 59.3 ms |
| 1,999 healthy heads and a finite `-800` tail | Tail correctly becomes zero but costs 1.22 ms vs 0.14 ms on main |

The first population is sampled normally. The second is deliberately tuned, but
needs neither many helpers nor a `1e-70` residual. The third shows that the
`n_live` statistic means "eligible for deep computation", not "nonzero rounded
weight". The dead shortcut applies to `-inf` or shifts below `-1100`.

### 6. Evidence that survives review

Fresh continuation checks confirm:

- The round-1 counterexample `[-59.88145743137663, -782.1408195747232]` now returns
  the reference's `4294967297` quanta.
- Hostile Decimal contexts, including `FloatOperation` and eight new threads
  inheriting a mutated `DefaultContext`, preserve the expected values and caller
  flags in the reproduced cases.
- 1,500 random calls preserve branch float-normal outputs and the per-call
  log normalizer bitwise. This does not negate the classification counterexample.
- The table oracle must use `exp(j/256)`, not `exp(j/64)`; the corrected measured
  maximum relative error is `2**-107.0701`.
- `[-3e-14, -745.1332191019412]` refutes the proposed rounded-shift zero oracle:
  its exact subtraction can still produce a weight rounding to one quantum.

Per-call evidence identity does not imply exact telescoping across stages.
Quantization of stored weights remains outside this design's scope.

## Revision-3 direction and measured experiment

Separate candidate ESS calculations from certification of weights actually used
by the filter. Both paths retain centered log weights and shifted-scale
normalization. Candidate checks keep the branch arithmetic; stage weights get
the eventual reviewed certification algorithm.

An in-memory experiment made that split without editing production code:

- The 32-particle cap reproduction completed all seven stages.
- There were seven strict calls and 367 inexpensive candidate calls.
- Final particles, weights, and log evidence matched the branch exactly on that
  constructed filter, as asserted by the archived `ess-split.py.txt` experiment.
- For the sampled cost counterexample, candidate search returned exactly the
  same beta, `0x1.e4d5dce974f7cp-7`, with either normalizer: one measured call took
  14.3 ms on the branch versus 80.3 ms with revision 2.

Reasoning: the corrected tiny weights in this comparison square to zero in the
ESS calculation, while the normal entries are unchanged. This supports removing
certification from speculative checks. It does not establish general bitwise
equivalence for a future wider correction policy. That requires its own tests.
An accepted stage can still reach the precision cap; this split is not a general
availability guarantee.

Remaining requirements before declaring a revised design ready:

1. Resolve the guarantee/preservation conflict explicitly. A guard band justified
   only by measured `np.exp` error cannot certify full exact-subnormal coverage.
2. Test both boundaries of a candidate rounding cell. One conservative approach
   chooses `c = rint(qh)` and accepts only when the entire error interval around
   the exact expansion `(qh - c) + ql` lies strictly inside `(-0.5, 0.5)`.
   Use exact expansion comparisons or outward rounding; ordinary float
   subtraction must not erase the error radius. Escalate ambiguous cases.
3. Renormalize the modified reduction pair before applying normalized-input
   bounds, accounting for any added rounding (a valid `QuickTwoSum` is exact).
   Account for gradual underflow with
   absolute error bounds, or prove exponent ranges for every intermediate.
   Trimming tiny residuals needs a separate exponent-perturbation budget and
   does not by itself prove all low-word products remain normal.
4. Recompose the exponential, denominator, division, tail, and decision bounds,
   including higher-order terms and explicit supported population limits.
5. State cap failure as a possible accepted-stage error. Verify rollback after
   progress and RNG consumption, not only at the first candidate evaluation.
6. Benchmark complete searches and updates, and count path usage. Keep ordinary
   sampled and deliberately tuned costs separate. Do not advertise observed
   timings as worst-case bounds.
7. Promote each confirmed defect to a regression test when implementing, and
   mutation-test the decision and interval logic against the full suite.
8. Obtain independent review of the revised specification and final diff.
   Merge remains subject to the standing AGY verdict and green-CI requirements.

## Archived evidence and replay

[normalizer-r2](normalizer-r2/) contains the original design, prototype, final
review verdict, and executed reproduction scripts as `.txt` artifacts. They are
historical evidence with known defects, not supported package code. In particular,
their original comments and claims are not authoritative. The original design
and prototype are byte-preserved; their SHA-256 hashes are:

```text
design.md.txt    e7515672a842112cd3a0eeb351be5f1bdcff3bec3b8b9db8195c084e6d893322
prototype.py.txt 109503e83e1e98b9c51317d637980d74e7dacb8d8810a2030039edf09c59fc02
```

To replay on the original normalizer worktree without synchronizing its venv,
load the archived prototype as `proto` before executing a probe. This bypasses
the scripts' obsolete scratchpad import location. Their explicit original
worktree/import assertions remain active; a different checkout requires adapting
those assertions and verifying its imports first. The DD precondition probe also
pins Python to 3.13.13; other interpreters require explicitly revisiting that pin.

```sh
uv run --no-sync python - <<'PY'
import importlib.machinery
import importlib.util
import pathlib
import runpy
import sys
import clocks.inference as inference

root = pathlib.Path.cwd().resolve()
assert pathlib.Path(inference.__file__).resolve().is_relative_to(root)
archive = root / 'docs/plans/normalizer-r2'
loader = importlib.machinery.SourceFileLoader('proto', str(archive / 'prototype.py.txt'))
spec = importlib.util.spec_from_loader('proto', loader)
proto = importlib.util.module_from_spec(spec)
sys.modules['proto'] = proto
loader.exec_module(proto)
runpy.run_path(str(archive / 'reproduce.py.txt'))
runpy.run_path(str(archive / 'dd-preconditions.py.txt'))
runpy.run_path(str(archive / 'gaussian-cap.py.txt'))
runpy.run_path(str(archive / 'ess-split.py.txt'))
PY
```

The midpoint, seam, and DD precondition reproductions assert the rejected
prototype's defects. The historical Gaussian cap script only prints outcomes:
inspect its output for branch success with seven stages and revision-2
`WeightRoundingUndecided`. Its zero exit status alone is not evidence of either
outcome. The split experiment asserts its stated stage counts and final-state
agreement. None of these results establishes readiness to ship. Timing probes
are `cost.py.txt` and `single-helper-cost.py.txt` and use the same loader.
