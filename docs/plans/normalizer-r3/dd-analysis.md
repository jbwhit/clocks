# Revision 3: a conservative DD certificate

Status: proposed arithmetic and a mathematical error analysis, not a certificate
for the unchanged revision-2 program. The implementation must match the changes
below. Numerical agreement is not used as a proof premise. This analysis covers
binary64 round-to-nearest-even arithmetic with separate operations (no compiler
reassociation), finite shifted arguments `s` in `[-800, 0]`, exact subtraction
corrections `|e| <= 2^-43`, and at most `2^63` entries.

The proposed implementation can retain `EXP_DD_EPS = 2^-96`,
`DD_OP_EPS = 2^-100`, `DD_DIV_EPS = 2^-98`, and `EPS_Q = 2^-90`.
The latter includes a denominator omission allowance of `2^-96`; do not charge
that allowance twice unless intentionally keeping additional slack.
The revision-3 prototype currently chooses the wider `EPS_Q = 2^-85`, which
is also conservative and increases the number of ambiguous DD decisions.

## Changes required

1. Define `trim(x) = 0` when `abs(x) < tau`, otherwise `x`, with
   `tau = 2^-350`. Trim `s` and `e` before exponential reduction and the raw
   `e*e` correction. Each such replacement changes its operand by less than
   `tau`, including when that operand was subnormal.
2. At entry to every DD add, multiply, and divide, trim all four components
   and normalize each input pair with `TwoSum`. Trimming and normalization
   are part of these helpers' contracts. Normalization is exact; it has no
   rounding-error charge beyond the preceding trim.
3. Use the following addition for both ordinary and former `nocancel` sites:

   ```python
   # a and b have been trimmed and normalized above.
   s, e = two_sum(ah, bh)
   e = e + (al + bl)
   return two_sum(s, e)
   ```

   This has an absolute bound, so cancellation in reduction and quotient
   remainders is permitted. Replace the final `QuickTwoSum` of DD multiplication
   with `TwoSum` as well. No unchecked magnitude hypothesis remains.
4. In division, use the ordinary DD multiplication to obtain `q1 * b`:

   ```python
   q1 = ah / bh
   p1, p2 = dd_mul(q1, 0.0, bh, bl)
   r1, r2 = dd_add(ah, al, -p1, -p2)
   q2 = r1 / bh
   return two_sum(q1, q2)
   ```

   Discarding `r2` is charged below. It is not an error-free quotient.
5. Trim both exponential mantissa components immediately before `ldexp`.
   This matters for the low component: a low word produced by a product may
   have bits far below its high word, and denominator scaling otherwise can
   underflow. Scaling after this trim is exact in the stated domain.
6. Normalize generated coefficient and table pairs with `TwoSum` before their
   first arithmetic use; `_prepare` at the helper entry suffices. Do the
   same to the reduction product after adding `k * LN2_LO2`, or rely on the
   explicitly specified input normalization in item 2. A pair modified by
   adding to its low word must not simply be declared normalized.

Keep the full exact-difference `TwoSum(values, -peak)` outside this trimming
wrapper. It supplies the exact argument; the subsequent deliberate argument
perturbation is charged in the exponential bound.

## Exact primitive source and algorithm match

The primitive results used here are `TwoSum`, `Split`, and `TwoProduct` from
Shewchuk's original [1997 paper, sections 2.3 and 2.5](https://people.eecs.berkeley.edu/~jrs/papers/robustr.pdf).
Theorem 7 establishes the unrestricted-size `TwoSum`. For binary64, `Split`
uses `2^27 + 1`. The paper's product correction is:

```python
p = a * b
ah, al = split(a)
bh, bl = split(b)
err1 = p - ah * bh
err2 = err1 - al * bh
err3 = err2 - ah * bl
err = al * bl - err3
```

The archived product is the sign-reversed recurrence with `a` and `b`
interchanged; symmetry gives the same theorem, including the operation order.
Using the displayed recurrence would make that correspondence explicit.
No double-word addition, multiplication, or division theorem from the 2017
Joldes–Muller–Popescu paper is invoked below. The following DD inequalities
are direct derivations using the exact primitives.

## Why no internal underflow is assumed away

A retained binary64 component of magnitude at least `2^-350` is an integer
multiple of `2^-402`. Zero also belongs to that lattice. Floating addition or
subtraction of lattice members, including cancellation, remains on the same
lattice. Multiplication by the integer splitter preserves it. Consequently
every split part belongs to that lattice, and every exact product of split
parts is a multiple of `2^-804`. Rounded products and their subsequent sums
remain on the `2^-804` lattice. A nonzero exact result is therefore at least
`2^-804`, far above binary64's normal threshold `2^-1022`.

This argument includes `al*bl`, both cross-products, residual cancellation,
and tiny polynomial arguments. It does not infer anything from the high-word
magnitude alone. Input normalization can create values smaller than `tau`,
but they remain on the `2^-402` lattice, which is sufficient for this proof.
No trimming inside `TwoSum`, `Split`, or `TwoProduct` is allowed: that would
destroy their exact identities.

All exponential-only intermediates have magnitude below `2^16`; the splitter
raises that bound by fewer than 28 exponent bits. Denominator sums are below
`2^64`. For candidate numerators with `s < -650`, quanta numerators are below
`2^138`. Division products and their splitter intermediates stay below
`2^170`. Thus overflow is also excluded.

For `s >= -800`, numerator quanta are greater than `2^-82`; division by a
denominator below `2^64` gives a first quotient above `2^-147` with margin.
After the remainder addition's input trims, its nonzero high word is a
multiple of `2^-402`. Dividing it by a denominator high word below `2^64`
therefore has magnitude at least `2^-466`. Both quotient divisions remain
normal even when the correction is extremely small.

Retained denominator arguments satisfy `s >= -160`, so their exponential
scale exponents satisfy `k >= -232`. Pre-scaling trimmed mantissas belong to
the `2^-402` lattice; scaling gives a lattice no finer than `2^-634`.
Numerator scale exponents `k+1074` are at least `-81`. Both component scalings
are therefore exact and normal or zero. The raw correction `0.5*e*e`, after
the initial trim, is at least `2^-701` when nonzero and cannot underflow.

The raw reduction products outside the DD wrappers also need an explicit
operand check. The archived constant construction produces these exact
binary64 representations (read from the existing runtime):

```text
LN2_HI  = 0x1.62e42fefa3800p-1
LN2_LO  = 0x1.ef35793c76730p-45
LN2_LO2 = 0x1.f97b57a079a19p-103
```

The revised implementation checks `tau <= abs(LN2_LO) <= 2^-42` and
`tau <= abs(LN2_LO2) <= 2^-94` at construction time. The nonzero integer `k`
and the two low constants therefore belong to the `2^-402` lattice, proving
safety of their raw products and splits. The hexadecimal values above are
observations only; the exponent acceptance checks supply the proof premise.

## Direct DD operation bounds

Write `u = 2^-53`, and first consider already trimmed, normalized operands
`a = ah + al`, `b = bh + bl`. Exact `TwoSum` normalization gives
`ah = RN(a)` and `|al| <= u|ah|`, and similarly for `b`. Let
`v = u/(1-u)`. Thus `|ah| <= |a|/(1-u)` and `|al| <= v|a|`.

### Addition

Let `S = |a| + |b|`. The first `TwoSum` is exact and its residual has magnitude
at most `v*S`. The rounding of `al+bl` costs at most `u*v*S`. The following
residual addition costs at most `u*(v*S + (1+u)*v*S)`. The final `TwoSum` is
exact. The total absolute error is bounded by

```text
(3+u)*u*v*S < 4*u^2*S.
```

This statement holds even if `a+b` vanishes or is arbitrarily small.
For same-sign positive summands it is a relative bound. For Horner addition,
the new coefficient dominates the product, so `S/|a+b| < 1.01`.

Trimming four input components adds less than `4*tau` absolute error, before
the displayed bound is applied to the trimmed operands. One may use
`4*u^2*(S+4*tau) + 4*tau` against the original operands.

### Multiplication

Let `C = |ah*bh|`. Exact `TwoProduct` supplies its rounded product and residual
of magnitude at most `u*C`. The two rounded cross-products have total rounding
error at most `2*u^2*C`. Their sum introduces at most
`2*u^2*(1+u)*C`; adding the product residual introduces at most
`u^2*(1 + 2*(1+u)^2)*C`. Omitting `al*bl` costs at most `u^2*C`.
Final `TwoSum` normalization is exact. Therefore

```text
absolute error <= (8*u^2 + 6*u^3 + 2*u^4)*C
               < 16*u^2*|a*b|.
```

The relaxed last constant follows from `|a*b| >= (1-u)^2*C` and binary64's
specific `u`. Zero operands produce exact zero before trim charges.

Trimming each pair changes its real value by less than `2*tau`. Hence the
additional product perturbation is bounded by
`2*tau*(|a|+|b|) + 4*tau^2`, multiplied by `1+16*u^2` to include subsequent
arithmetic. It is essential to use this absolute form when a polynomial
argument is tiny; a small absolute perturbation need not be small relative
to that tiny argument.

### One-correction division

For positive normalized `a,b`, put `q=a/b` and `q1=RN(ah/bh)`.
The inequalities above imply `|q1/q-1| < 4*u`, so
`|a-q1*b| < 4*u*a`. The DD product error is less than `17*u^2*a`.
The DD remainder addition costs less than `9*u^2*a`. Thus its represented
remainder differs from `a-q1*b` by less than `26*u^2*a`.

Discarding its normalized low word costs less than `5*u^2*a`.
Rounding `r1/bh` costs less than `5*u^2*q`. Using `bh` in place of `b` in
this small correction costs less than `5*u^2*q`. Converting the preceding
absolute remainder errors by `1/bh` adds a factor at most `1+2*u` relative
to division by `b`. The final `TwoSum` is exact. Substitution yields

```text
|computed_q/q - 1| < 64*u^2.
```

These inequalities include higher-order terms: the constants 17, 9, and 5
strictly exceed the expressions obtained by retaining their `1+u`,
`1/(1-u)`, and `1+4*u` factors. Their sum is below 42, leaving further room
inside 64. Input and nested-helper trims add less than `2^-250` relative
in the stated numerator and denominator domain (use `a>2^-82`, `b>1/2`,
`q1<2^139`, and the product perturbation bound). Consequently the budget
`DD_DIV_EPS=2^-98 = 256*u^2` is conservative.

## Exponential composition

Constant construction uses isolated 80-digit Decimal contexts and exact
rational acceptance checks. Decimal's `exp` and `ln` are correctly rounded;
their predecessor and successor in that context enclose the exact result.
These are specified properties of [Python's Decimal arithmetic](https://docs.python.org/3/library/decimal.html#decimal.Decimal.exp).
The `j/256` table arguments are exactly representable in this context.
Converting both endpoints to `Fraction` supplies exact rational bounds.

For a table pair, `_certified_pair` normalizes the two proposed float words,
then requires their exact rational sum to lie within `lower*2^-105` of both
endpoints. Since the exact positive table value is at least `lower`, this
certifies relative error at most `2^-105 = 2*u^2`. For a coefficient, both
endpoints equal the exact rational `1/i!`, giving the same guarantee.
Correct rounding of float conversions is no longer a proof premise: the
actual returned words are checked exactly, and a failure raises.

The logarithm constants are checked against analogous exact rational
endpoints. The checks establish scalar `LN2` absolute error at most `2^-53`,
the exact `2^-42` grid of `LN2_HI`, its absolute error at most `2^-42`, and
the triple's absolute error at most `2^-148`. This replaces the old measured
`2^-157` claim. Since `|k|<2^11`, the triple residual contributes at most
`2^-137` to reduction.

For the proposed domain, rounding `s/LN2` puts the first residual below 0.35.
The product `k*LN2_HI` is exact. Its subtraction from `s` is also exact:
for nonzero `k`, `|s| >= 1/4`, both operands lie on a grid no finer than
`2^-54`, and their difference has magnitude below 1/2; for `k=0` it is trivial.
Adding `k*LN2_LO2` to the product residual contributes less than `2^-132`
absolute error, including both roundings and the triple residual.

Two DD subtractions use the absolute addition bound. The first has operand
absolute sum below 0.36; the second below 0.71. Their combined arithmetic
error is below `5*u^2`, with more than sufficient room in `8*u^2` after
trims and the triple error. The second reduction satisfies
`|r''| <= 1/512 + 2^-50`. Its error changes the exponential by less than
`16*u^2` relative, using `|exp(delta)-1| <= |delta|*exp(|delta|)`.

Every rational coefficient and every table entry has DD relative
representation error at most `2*u^2`, including construction and conversion
errors, as checked above. A budget of `4*u^2` per entry is conservative.
The correction does not require the stronger measured table or coefficient
claims from revision 2.

In degree-9 Horner evaluation, each exact partial polynomial is positive and
has magnitude at most 1.003; each coefficient is at most 1, and a product
has magnitude below 0.002. The multiplication and addition bounds give less
than `8*u^2` absolute local arithmetic-plus-coefficient error per stage,
apart from trims. Propagation multiplies prior errors by at most 0.002;
even charging all nine local errors without this decay gives less than
`128*u^2` relative error after dividing by the final polynomial's lower
bound 0.998. The small feedback from previously accumulated arithmetic errors
is covered by this gap: `(1+16*u^2)^9/(1-0.002) < 1.003`.
Taylor truncation is below `2^-111`, hence below `u^2/32`.

The `e` correction has truncation bounded by
`exp(|e|)*|e|^3/6 < 2^-131`; rounding `0.5*e*e` contributes less than
`2^-139`. Its DD addition costs less than `5*u^2` relative. A `16*u^2`
allowance covers this whole correction. Each of the two final DD products
costs less than `16*u^2`; allow `32*u^2` per product.

Combining reduction (16), Horner (128), table (4), correction (16), and final
products (64) gives fewer than `229*u^2`. Multiplicative cross terms cannot
use the gap to `256*u^2`: all stage errors together are below `2^-97`, so
their product amplification is below `1+2^-96`. The fewer than 200 component
trims per scalar exponential contribute less than `2^12*tau` relative:
the relevant partials and table factors are bounded by 2, argument
perturbations have exponential condition number below 2, and Horner's
propagation contracts. This is negligible compared with `u^2`.

Thus the composed mantissa relative error is below `2^-98`, with
`EXP_DD_EPS=2^-96` providing a further factor-four allowance. Pre-scaling
mantissa trims cost less than `4*tau` relative and fit in that allowance.

## Denominator, tail, and final quotient

Retain all terms with rounded shift at least `-160`. Let `T` be the exact sum
of retained exponentials and `R` the omitted sum. The peak is retained and
contributes exactly 1, so `T>=1`. Monotonicity of rounding at the finite cutoff
implies every omitted exact difference is below `-159`, including differences
whose rounded result overflows to negative infinity. Since `exp(1)>2`,

```text
0 <= R/T < N*exp(-159) < 2^63*2^-159 = 2^-96.
```

This loose inequality avoids any libm assumption in the cutoff certificate.

There are at most 63 pairwise summation levels. The relative addition error
at each positive node is below `4*u^2`, and is conservatively budgeted as
`beta = 2^-100`. Component trimming adds at most `4*tau` per node; over the
whole tree this is less than `2^-285` absolute before amplification by
`(1+beta)^63`. The denominator relative error against the full exact sum is
therefore below

```text
((1+2^-96)*(1+2^-100)^63 - 1) + 2^-96 + 2^-284
< 2^-93.
```

Let `en=2^-96`, `ed=2^-93`, and `ev=2^-98`. Scaling is exact as proved above.
The final quanta estimate satisfies the fully composed bound

```text
|q_est/Q - 1| <= ((1+en)*(1+ev)/(1-ed)) - 1 < 2^-92 < 2^-90.
```

The corresponding lower-error expression is smaller. All denominators are
positive. This establishes a conservative `EPS_Q=2^-90` including the tail
under the stated contracts.

For the prototype's actual decision radius, write `q_est = qh + ql`, where
normalization gives `|ql| <= u*|qh|`. If `eta=2^-92`, the displayed relative
bound implies

```text
|Q-q_est| <= eta/(1-eta) * |q_est|
          <= eta*(1+u)/(1-eta) * |qh|
          < 2^-91 * |qh| < 2^-85 * |qh|.
```

Hence `nextafter(2^-85 * abs(qh), +inf)` covers the absolute radius even after
its multiplication rounds. The relevant positive `qh` is above `2^-147`,
so that product is normal. The cell-boundary arithmetic itself still requires
outward rounding or exact rational arithmetic; an ordinary rounded subtraction
of the radius is not justified by this proof.

## Review limits and remaining checks

This is a hand derivation with explicit algorithm contracts, not a formal
machine-checked proof. It does not certify the untouched revision-2 helper
implementations. In particular, a version that only trims `r` or `e`, keeps
unchecked `QuickTwoSum` calls, or omits the pre-scaling trim is outside the
proof. Correctly rounded Decimal `exp`/`ln` and exact rational arithmetic
remain stated platform premises; float conversion accuracy is checked.
The independent review must inspect the implemented operation order and all
trim sites against these contracts, including exact argument recovery.

The candidate-selection theorem, both rounding-cell boundaries, outward
interval construction, and Decimal fallback are separate obligations owned
by the enclosing revision-3 design. This document does not claim that those
parts are correct or that the complete normalizer is ready for production.

Environment check performed for this analysis: `uv run --no-sync python`
successfully imported `clocks.inference` from this worktree's
`src/clocks/inference.py`. No environment synchronization or production file
change was performed.

Five exact `Fraction` checks also passed for the displayed add and multiply
inequalities, Horner amplification, 63-level denominator composition, and final
quotient composition. These verify the algebraic constants at binary64's
specific `u`; they do not replace the preceding algorithm analysis.

The newly written `prototype.py` arithmetic helpers were read against these
contracts: `_prepare`, addition, multiplication, quotient, initial argument
trimming, and pre-scaling trimming match. The subsequently added
`_certified_pair` and log-two exact rational acceptance checks were also read
against their representation budgets and match the bounds used here.
