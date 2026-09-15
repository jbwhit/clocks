# Production normalizer implementation evidence

This document records production integration evidence separately from the
revision-3 arithmetic analysis. The arithmetic contract is relative to supplied
binary64 log weights: exact subnormal weights round nearest-even or the strict
normalizer raises `WeightRoundingUndecided`; a narrow near-normal strip can
change. Ordinary weights remain approximate. It does not promise exact
trajectories or completion for every finite population.

## Current production checks

`ParticleFilter` uses the strict normalizer only for accepted stages. Its
speculative `_next_beta` ESS trials use the fast shifted-scale helper. A strict
cap failure rolls a public `update` back; callers should not blindly retry
unchanged inputs.

The production-focused commands are:

```sh
uv run --no-sync pytest tests/test_weight_normalization.py tests/test_weight_intervals.py tests/test_weight_filter.py tests/test_smc_rigorous.py tests/test_weight_tools.py --tb=short
uv run --no-sync python scripts/check_normalizer_mutations.py old_midpoint
uv run --no-sync python scripts/check_normalizer_mutations.py float_classification
uv run --no-sync python scripts/check_normalizer_mutations.py no_dd_interval
uv run --no-sync python scripts/check_normalizer_mutations.py collapsed_decimal_interval
uv run --no-sync python scripts/benchmark_normalizer.py
uv build
uv run --no-sync python scripts/check_wheel_entrypoints.py dist/*.whl python3
```

The mutation and benchmark commands are project-environment tools: they require
the active checkout's locked dependencies and neither bootstrap nor synchronize
an environment. Before mutating or timing, each verifies and prints the exact
normalizer and inference paths under this checkout's `src/clocks/` directory.

Each mutation command runs the whole package suite in one pytest interpreter
after installing one in-memory fault. Fresh child processes launched by that
suite are not mutated; they exercise the integrated source instead. Thus the
mutation runs provide in-process fault sensitivity, while the unmutated suite
and wheel smoke provide positive fresh-process/package coverage.

## Measured results

Before production wiring, the promoted numerical selection ran six tests: the
four intended subnormal regressions failed with the old shifted-scale helper,
while its two controls passed. After wiring and the focused tool provenance
checks, the numerical, interval, real-filter, rigorous-SMC, and tool files ran
140 tests successfully. These are focused integration measurements, not
whole-package or trajectory claims.

The final local default gate returned **946 passed, 2 slow deselected** in
70.17 seconds; the separate slow gate returned **2 passed** in 391.67 seconds.
Ruff formatter check and lint passed (73 Python files). `uv build` produced the
wheel and source distribution, and the installed-wheel smoke passed all seven
console commands using `/usr/local/bin/python3` for the temporary smoke environment.
The worktree environment itself was neither copied nor synchronized.

Each production mutation ran all **948 tests**, including both slow replays.
All four exited with pytest status 1, and the controller inspected the failures:

| Mutation | Result | Intended failure |
|---|---|---|
| `old_midpoint` | 1 failed, 947 passed (495.27 s) | Lower-midpoint literal is one quantum wrong |
| `float_classification` | 1 failed, 947 passed (493.54 s) | Exact-subnormal classification seam is wrong |
| `no_dd_interval` | 4 failed, 944 passed (493.47 s) | Two real cap controls stop raising, a zero/quantum decision is wrong, and a crossing interval is accepted |
| `collapsed_decimal_interval` | 4 failed, 944 passed (494.70 s) | Four independently computed exponential values are not enclosed |

No unrelated failure contributed to these counts. This demonstrates sensitivity
to the four selected faults, not proof of the certificate. Fresh subprocesses
remain unmutated, as described above.

## Implementation review trail

The implementation plan received an independent GPT-5.6 Sol review:
**READY TO IMPLEMENT**, with no blocking findings. GPT-5.6 Luna promoted
the arithmetic; GPT-5.6 Terra integrated the filter and tools; GPT-6 coordinated
verification. A full-module AST comparison against the reviewed prototype
passed after allowing only documented promotion changes (names, annotations,
documentation, removal of stats instrumentation, and the existing invalid-mass
error text). This checks preservation, not the mathematical proof itself.

Independent GPT-5.6 Sol task reviews found and verified fixes for:

- Overbroad contract prose and duplicated precision-observation test setup.
- A tautological public-exception check, replaced with a real cap failure.
- Missing fresh-process coverage of the exact classification seam.
- A stale positional description of the historical failing cases.
- Misleading standalone-tool packaging and incomplete source-path checks.
- A reference to disposable scratch evidence, replaced with durable results.

The final scoped verdict was **Task quality approved**, with no remaining
Critical/Important findings. Two explicit controller rulings were accepted:

1. Verification tools run only in the locked project environment through
   `uv run --no-sync python`; they do not bootstrap standalone environments.
   The tradeoff is that they require project setup, in return for clear
   dependency and checkout provenance.
2. Each tool keeps its small, self-contained source guard. The same parameterized
   tests check both implementations. This avoids extra script import-path
   plumbing but means a future guard change must update both copies. The
   reviewer reclassified hypothetical drift as non-blocking maintenance risk.

Whole-branch Astra and isolated Gemini/AGY reviews apply to the committed
implementation snapshot. See their exact verdicts, verified responses, reviewed
SHA, and subsequent CI conclusion in the
[PR #18 review trail](https://github.com/jbwhit/clocks/pull/18).

## Production cost measurements

Measured September 15, 2026, on macOS 26.6.2 x86_64, Python 3.13.13,
NumPy 2.4.2. Both imported production modules were verified against their
exact source paths before timing. Each entry is the median of five calls
after one warm call, using seed 731 and one tail appended to normal(0,1)
heads. These measurements ran before the concurrent full-suite gates.

| Population | Tail log weight | Strict ms | Fast ms |
|---|---|---|---|
| 2,000 | -700 | 2.810 | 0.154 |
| 2,000 | -720 | 2.726 | 0.152 |
| 2,000 | -740 | 2.605 | 0.160 |
| 2,000 | -800 | 0.171 | 0.179 |
| 2,000 | -inf | 0.169 | 0.149 |
| 40,000 | -700 | 24.682 | 0.876 |
| 40,000 | -720 | 23.314 | 0.880 |
| 40,000 | -740 | 22.959 | 0.861 |
| 40,000 | -800 | 0.913 | 0.861 |
| 40,000 | -inf | 0.916 | 0.897 |

The fixed 32-particle Gaussian workload took **60.196 ms**, including fresh
filter construction and its update. Its integration test observes seven
accepted stages and equality with the fast accepted-stage control for this
case only. These timings are neither a worst-case bound nor evidence that
costly or cap-exhausting inputs are rare. The DD error bound remains a
reasoned arithmetic argument supported by tests, not a machine-checked proof.
