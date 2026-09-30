# Revision 3: correctly rounded tiny SMC weights

Status: independently reviewed design and executable experiment; implemented in
the package on 2026-09-15. This document remains the historical design record.
This supersedes [revision 2](2026-09-14-normalizer-round-2-review.md).

**Implementation status (2026-09-15):** the historical prototype, proof, and
design tests below remain reproducible evidence at commit `3f7df94`. Their old
binding and xfail assumptions no longer describe the production source. Current
production evidence is recorded in
[the implementation evidence](2026-09-15-normalizer-implementation.md), using
the package tests and mutation commands named there.

## Contract and scope

Jonathan explicitly selected correct subnormal rounding over preservation of
float-normal outputs near the boundary. Given a nonempty vector of binary64 log
weights, define `w_i = exp(v_i) / sum(exp(v_j))` in real arithmetic. Finite entries
and `-inf` are supported; NaN, positive infinity, and all-zero mass retain the
existing loud failure. Array length is at most `2**63`, encompassing lengths
representable by NumPy on the supported 64-bit runtime.

Every exact subnormal weight must round correctly to nearest-even, or raise
`WeightRoundingUndecided` if the explicit precision budget is exhausted. Exact
zero weights from `-inf` entries remain zero. A narrow strip above the smallest
normal value may also be corrected. We no longer promise bitwise identity for
every entry the old floating calculation classified as normal. Away from that
strip, the branch's shifted-scale calculation remains the value path. The returned per-call evidence
normalizer is the same `float(logsumexp(values))` calculation.

Correctness is relative to the supplied binary64 log weights. This does not
repair rounding in upstream likelihood calculations, preserve particles whose
exact weights round to zero, or make evidence telescope exactly across stages.

## Two uses of normalization

`_next_beta` needs approximate weights only to compute ESS. Its trial
distributions use the inexpensive shifted-scale path. The chosen stage uses
the certified normalizer before weights affect resampling or stored state.
Both retain the existing centering and invalid-input guard.

The DD path changes weights only in a range whose squares are zero in binary64.
Consequently those corrections cannot change their own ESS contributions.
This is not a proof that the entire SMC trajectory is invariant: tiny weights
can matter to later log weights, evidence, and support. The split specifically
prevents a rejected candidate from consuming certification work or exhausting
the precision cap.

For the constructed 32-particle Gaussian failure, the experiment completes seven
stages using seven strict normalizations and 367 inexpensive candidate calls.
Final particles, weights, and evidence match the branch in that case. A second
test injects the real cap-triggering distribution at a later accepted stage,
after RNG consumption and an evidence increment, and verifies complete rollback.
It tests rollback deliberately; it does not claim that this later distribution
arises naturally in that run.

Production integration should give the two call sites distinct helper names.
The experiment uses isolated function globals to model the split in one Python
process; that test mechanism is not the proposed production implementation.

## Candidate coverage without trusting NumPy exp accuracy

Compute the branch weights, then examine the rounded shifts `s = RN(v_i - peak)`.
Only `s < -650` can require correction. This cutoff depends on mathematical
size bounds, not the possibly misclassified branch weight:

1. The exact shifted denominator is at most `N <= 2**63`.
2. Exact `w_i < 2**-1022` implies `exp(d_i) < 2**-959`, hence
   `d_i < -959*ln(2) < -661`, using `ln(2) > 0.69`.
3. Monotonic rounding cannot move such a difference above `-650`.

Terms with rounded shift below `-800` round to zero immediately. Their exact
shift is below `-800`, and `ln(2) < 0.7` implies
`exp(d_i)*2**1074 < 2**-68 < 0.5`; the denominator is at least one.
This includes finite differences that overflow to `-inf` and actual `-inf`
entries. The bounds `0.69 < ln(2) < 0.7` follow, for example, from the first two
positive terms of `ln(2)=2*atanh(1/3)` and the positive Taylor series for `exp(0.7)`.

All remaining candidate numerators use exact-difference pairs from `TwoSum`.
The DD denominator retains shifts at least `-160`. Every omitted exact
exponential is below `exp(-159)`, so all omitted mass is below `2**-96` using
`exp(1)>2` and `N<=2**63`. This tail allowance is included in the DD quotient
bound; no libm approximation is used to compute the allowance.

## DD arithmetic and its certificate

See the complete [arithmetic analysis](normalizer-r3/dd-analysis.md).
The changes from revision 2 are:

- Clamp components smaller than `2**-350` to zero at specified helper boundaries,
  and charge the discarded terms as absolute errors.
- Normalize helper inputs with `TwoSum`. Use `TwoSum` for final normalization
  too, eliminating unchecked `QuickTwoSum` magnitude assumptions.
- Derive absolute addition and relative product/quotient errors directly from
  exact primitives. Cancellation is permitted in the subtraction analysis.
- Clamp mantissa components before scaling, so low-word `ldexp` cannot underflow.
- Certify generated constants at import with exact `Fraction` comparisons to
  outward Decimal endpoints or exact rational coefficients. Incorrect float
  conversions fail a certificate check instead of silently entering a proof.

The crucial underflow argument is a lattice bound: retained components are
multiples of `2**-402`; split parts stay on that lattice and their products
are multiples of `2**-804`. Every nonzero intermediate is therefore above the
normal threshold. This covers low-word cross-products and cancellation, not
just high-word magnitudes. The analysis separately covers raw reduction,
division, and scaling operations.

The proposed composed quotient relative error is below `2**-92`, including
the denominator tail. The implementation uses an upward-rounded radius
`2**-85 * abs(qh)` around the represented pair. The proof relates the represented
pair and its high word explicitly. These are reasoned bounds, not conclusions
drawn from random sampling or the passing suite. Two independent reviewers checked
the operation order and composition; see the [review record](normalizer-r3/review.md).
This is not a machine-checked proof or a reviewed production implementation.

## Both rounding boundaries

Let `Q_i = w_i * 2**1074`. A DD candidate is corrected only when its high word
is at most `2**52 + 2**20`. Given the error bound, every exact subnormal entry
is included; skipped estimates are provably above `2**52`. All corrected
estimates remain far below `2**53`, throughout the unit-quantum grid.

Choose an integer near the DD estimate, form the exact expansion of its offset,
and adjust the integer if that expansion lies past either half-integer.
Compute lower bounds for the distances to both sides of the rounding cell.
`TwoSum` retains cancellation before two outward `nextafter` additions bound
the remaining sum. Certify only if both distances exceed the upward error radius.
Ambiguous candidates go to Decimal. Integer-to-binary64 conversion followed by
`ldexp(integer, -1074)` is exact for the certified integer range.

This fixes the round-2 lower-midpoint case without assuming the floor of the
high word is also the floor of the represented pair. It also retains low-word
information when the high word itself lands on a midpoint.

## Outward Decimal fallback

Use explicit isolated contexts for HALF_EVEN, FLOOR, and CEILING, setting every
context field and using only explicit context arithmetic. Float conversion to
Decimal uses `Decimal.from_float`. The exact subtraction uses 1,500 digits with
`Inexact` trapped; the maximum finite binary64 difference needs at most 1,383
decimal digits.

At precision `P`, a correctly rounded `Context.exp(d)` is enclosed by its two
adjacent Decimal numbers. Merely setting the exponential context to FLOOR or
CEILING is insufficient: Decimal exp is specified to round HALF_EVEN.
Accumulate lower endpoints downward and upper endpoints upward. The peak term
is exactly one, keeping the denominator strictly positive.

For cutoff `-3*(P+24)`, each omitted exponential is less than `10**-(P+24)`:
`exp(3)>13>10`. Count finite original values, including finite differences that
overflowed in binary64, and add the outward product of that count and the tail
bound only to the denominator's upper endpoint.

For each numerator interval `[a,b]` and denominator interval `[L,U]`, form
outward quantum endpoints `[a*2**1074/U, b*2**1074/L]`. Explicit HALF_EVEN
`to_integral_value` rounds both endpoints. If they are the same integer in
`[0,2**53)`, every enclosed real value rounds to that integer. Convert that
integer exactly and scale by `2**-1074`. This avoids relying on Decimal's
string-to-float conversion to round a subnormal correctly.

Precision runs through 56, 112, and 224 digits. Unresolved decisions raise;
there is no warn-and-return fallback. A finite Gaussian population can reach
this cap. Removing speculative certification improves availability but does
not prove every accepted stage succeeds. The explicit failure policy remains
part of this contract.

The Decimal premises are documented by Python's
[Decimal arithmetic specification](https://docs.python.org/3.13/library/decimal.html#decimal.Decimal.exp)
and [explicit context operations](https://docs.python.org/3.13/library/decimal.html#decimal.Context).
Arithmetic assumes separate IEEE binary64 operations, round-to-nearest-even,
and gradual underflow; flush-to-zero environments are outside the supported
contract.

## Measurements and cost limits

Initial macOS x86_64, Python 3.13.13 measurements (medians of five calls):

| Shape | N=2,000 | N=40,000 |
|---|---:|---:|
| Healthy heads plus one candidate tail | 2.6–2.8 ms | 23–25 ms |
| Heads plus a tail already below the zero cutoff | about 0.19 ms | about 0.9–1.0 ms |

Those are observations, not worst-case ceilings. The conservative candidate
gate can run DD even when no output needs correction: the measured N=2,000,
tail=-700 example did so. The safe arithmetic costs more per DD call than
revision 2. It is intended for accepted stages; candidate searches retain the
inexpensive path. A flagged weight still forces a shared Decimal denominator
and may cost tens or hundreds of milliseconds. A precision cap bounds rounds,
not latency independently of population size. No rarity claim makes that cost
disappear.

## Verification and promotion

The executable prototype and focused tests live in `normalizer-r3/`. The two
round-2 wrong-output tests were observed failing against the archived revision
and passing against revision 3. The final combined package and design suite,
including slow acceptance tests, with the prototype and ESS split injected
passed **935 tests**, after turning only the four known strict xfails into
ordinary assertions in the test process. Environment imports are asserted
against the active checkout before that run. All four selected mutations were
detected in complete 935-test runs; see the [evidence table](normalizer-r3/review.md).

The following archived design commands are historical and reproducible at
`3f7df94`; they no longer test the current production bindings:

```sh
uv run --no-sync pytest docs/plans/normalizer-r3 --tb=short
```

Run the entire package and design experiment, including slow acceptance tests:

```sh
uv run --no-sync python docs/plans/normalizer-r3/check_suite.py tests docs/plans/normalizer-r3 -m '' --tb=short
```

The suite harness uses an explicit allowlist to remove precisely four known
xfail marks and asserts the allowlist matched. It never changes source files
or virtual environments. Injection affects only the pytest interpreter: tests
that launch a fresh CLI process still exercise the branch implementation.
For current production verification, use the commands and measured evidence in
[the implementation evidence](2026-09-15-normalizer-implementation.md).

The independent design gate is recorded in [the review trail](normalizer-r3/review.md).
The next checkpoint is a production implementation plan and its independent review.
This design commit itself must not be taken as approval to merge PR #18 or
deploy anything.
