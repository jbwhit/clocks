"""Normalization with certified rounding for tiny binary64 weights.

For finite supplied binary64 log weights and ``-inf`` zero-mass entries,
``_normalize_log_weights`` correctly rounds every exact subnormal normalized
weight to nearest-even, or raises ``WeightRoundingUndecided`` if the
precision cap cannot decide.  It may also correct a narrow strip above the
smallest normal value.  Other weights retain the shifted-scale approximation
and have no correct-rounding guarantee.  NaN, positive infinity, and all-zero
mass raise ``RuntimeError``.  Correctness is relative to the supplied
binary64 logs, not upstream exact likelihoods.  The certificate assumes
separate IEEE-754 binary64 operations, round-to-nearest-even, and gradual
underflow; do not reassociate DD operations.

The arithmetic and error analysis are derived in
``docs/plans/normalizer-r3/dd-analysis.md``.
"""

import math
from decimal import (
    MAX_EMAX,
    MIN_EMIN,
    ROUND_CEILING,
    ROUND_FLOOR,
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DivisionByZero,
    Inexact,
    InvalidOperation,
    Overflow,
)
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

QUANTUM = 2.0**-1074
TINY = 2.0**-1022
CLAMP = 2.0**-350
EPS_Q = 2.0**-85
EXP_DD_EPS = 2.0**-96
CORRECTION_LIMIT = float(2**52 + 2**20)
P_START = 56
P_MAX = 224


class WeightRoundingUndecided(RuntimeError):
    """The precision budget did not settle a required rounding decision."""


def _isolated_context(prec, rounding=ROUND_HALF_EVEN):
    return Context(
        prec=prec,
        rounding=rounding,
        Emin=MIN_EMIN,
        Emax=MAX_EMAX,
        capitals=1,
        clamp=0,
        flags=[],
        traps=[InvalidOperation, DivisionByZero, Overflow],
    )


def _dec(value):
    return Decimal.from_float(float(value))


def _exact_difference(value, peak):
    context = _isolated_context(1500)
    context.traps[Inexact] = True
    return context.subtract(_dec(value), _dec(peak))


def _clean(value):
    return np.where(np.abs(value) < CLAMP, 0.0, value)


def _two_sum(a, b):
    total = a + b
    virtual_b = total - a
    error = (a - (total - virtual_b)) + (b - virtual_b)
    return total, error


def _prepare(high, low):
    return _two_sum(_clean(high), _clean(low))


def _split(value):
    product = 134217729.0 * value
    high = product - (product - value)
    return high, value - high


def _two_prod(a, b):
    product = a * b
    ah, al = _split(a)
    bh, bl = _split(b)
    error1 = product - ah * bh
    error2 = error1 - al * bh
    error3 = error2 - ah * bl
    return product, al * bl - error3


def _dd_add(ah, al, bh, bl):
    ah, al = _prepare(ah, al)
    bh, bl = _prepare(bh, bl)
    high, error = _two_sum(ah, bh)
    error = error + (al + bl)
    return _two_sum(high, error)


def _dd_mul(ah, al, bh, bl):
    ah, al = _prepare(ah, al)
    bh, bl = _prepare(bh, bl)
    high, error = _two_prod(ah, bh)
    error = error + (ah * bl + al * bh)
    return _two_sum(high, error)


def _dd_div(ah, al, bh, bl):
    ah, al = _prepare(ah, al)
    bh, bl = _prepare(bh, bl)
    first = ah / bh
    ph, pl = _dd_mul(first, 0.0, bh, bl)
    remainder, _ = _dd_add(ah, al, -ph, -pl)
    second = remainder / bh
    return _two_sum(first, second)


def _dd_sum(high, low):
    while high.size > 1:
        if high.size % 2:
            high = np.append(high, 0.0)
            low = np.append(low, 0.0)
        high, low = _dd_add(high[::2], low[::2], high[1::2], low[1::2])
    return float(high[0]), float(low[0])


def _dd_scale(high, low, exponent):
    # Clamp before scaling as well as at arithmetic boundaries.
    return np.ldexp(_clean(high), exponent), np.ldexp(_clean(low), exponent)


_context = _isolated_context(80)
_ln2 = _context.ln(Decimal(2))
LN2 = float(_ln2)
LN2_HI = float(np.ldexp(np.rint(np.ldexp(LN2, 42)), -42))
LN2_LO = float(_context.subtract(_ln2, _dec(LN2_HI)))
LN2_LO2 = float(_context.subtract(_context.subtract(_ln2, _dec(LN2_HI)), _dec(LN2_LO)))


def _decimal_pair(value):
    high = float(value)
    low = float(_context.subtract(value, _dec(high)))
    return _certified_pair(
        high,
        low,
        Fraction(_context.next_minus(value)),
        Fraction(_context.next_plus(value)),
    )


def _certified_pair(high, low, lower, upper):
    """Check representation error with exact rationals, including conversion."""
    high, low = _two_sum(high, low)
    represented = Fraction(float(high)) + Fraction(float(low))
    allowance = lower / 2**105
    if max(abs(represented - lower), abs(represented - upper)) > allowance:
        raise RuntimeError("double-double constant exceeds its certified error")
    return high, low


_ln2_lower = Fraction(_context.next_minus(_ln2))
_ln2_upper = Fraction(_context.next_plus(_ln2))
_ln2_triple = sum(map(Fraction, [LN2_HI, LN2_LO, LN2_LO2]))
if (
    max(abs(Fraction(LN2) - _ln2_lower), abs(Fraction(LN2) - _ln2_upper))
    > Fraction(1, 2**53)
    or (Fraction(LN2_HI) * 2**42).denominator != 1
    or max(abs(Fraction(LN2_HI) - _ln2_lower), abs(Fraction(LN2_HI) - _ln2_upper))
    > Fraction(1, 2**42)
    or max(abs(_ln2_triple - _ln2_lower), abs(_ln2_triple - _ln2_upper))
    > Fraction(1, 2**148)
    or not CLAMP <= abs(LN2_LO) <= 2.0**-42
    or not CLAMP <= abs(LN2_LO2) <= 2.0**-94
):
    raise RuntimeError("log-two reduction constants exceed their certified error")


TABLE = np.array(
    [
        _decimal_pair(_context.exp(_context.divide(Decimal(j), Decimal(256))))
        for j in range(-89, 90)
    ]
)
COEFFICIENTS = []
for _i in range(10):
    _coefficient = Fraction(1, math.factorial(_i))
    _high = float(_coefficient)
    COEFFICIENTS.append(
        _certified_pair(
            _high,
            float(_coefficient - Fraction(_high)),
            _coefficient,
            _coefficient,
        )
    )


def _exp_dd(shifted, correction):
    shifted = _clean(np.asarray(shifted, dtype=np.float64))
    correction = _clean(np.asarray(correction, dtype=np.float64))
    exponent = np.rint(shifted / LN2)
    residual = shifted - exponent * LN2_HI
    ph, pl = _two_prod(exponent, LN2_LO)
    pl = pl + exponent * LN2_LO2
    rh, rl = _dd_add(residual, 0.0, -ph, -pl)
    index = np.rint(rh * 256.0)
    rh, rl = _dd_add(rh, rl, -index / 256.0, 0.0)
    high = np.full_like(shifted, COEFFICIENTS[-1][0])
    low = np.full_like(shifted, COEFFICIENTS[-1][1])
    for ch, cl in reversed(COEFFICIENTS[:-1]):
        high, low = _dd_mul(high, low, rh, rl)
        high, low = _dd_add(high, low, ch, cl)
    table = TABLE[(index + 89).astype(np.intp)]
    high, low = _dd_mul(high, low, table[:, 0], table[:, 1])
    fh, fl = _two_sum(1.0, correction)
    fh, fl = _dd_add(fh, fl, 0.5 * correction * correction, 0.0)
    high, low = _dd_mul(high, low, fh, fl)
    return high, low, exponent.astype(np.int64)


def _exp_interval(context, argument):
    if not argument:
        return Decimal(1), Decimal(1)
    rounded = context.exp(argument)
    return max(Decimal(0), context.next_minus(rounded)), context.next_plus(rounded)


def _decide_exactly(flagged, values, shifted, peak):
    unique, inverse = np.unique(flagged, return_inverse=True)
    result = np.empty(unique.size)
    undecided = np.ones(unique.size, dtype=bool)
    finite = np.isfinite(values)
    scale = Decimal(2**1074)
    precision = P_START
    while True:
        context = _isolated_context(precision)
        down = _isolated_context(precision, ROUND_FLOOR)
        up = _isolated_context(precision, ROUND_CEILING)
        cutoff = -3 * (precision + 24)
        keep = finite & (shifted >= cutoff)
        total_low = Decimal(0)
        total_high = Decimal(0)
        for value in values[keep]:
            low, high = _exp_interval(context, _exact_difference(value, peak))
            total_low = down.add(total_low, low)
            total_high = up.add(total_high, high)
        dropped = int(finite.sum() - keep.sum())
        tail = up.multiply(Decimal(dropped), Decimal((0, (1,), -(precision + 24))))
        total_high = up.add(total_high, tail)
        for index in np.flatnonzero(undecided):
            low, high = _exp_interval(context, _exact_difference(unique[index], peak))
            low_q = down.divide(down.multiply(low, scale), total_high)
            high_q = up.divide(up.multiply(high, scale), total_low)
            low_integer = context.to_integral_value(low_q)
            high_integer = context.to_integral_value(high_q)
            if low_integer == high_integer:
                integer = int(low_integer)
                if not 0 <= integer < 2**53:
                    raise RuntimeError("certified weight outside the quantum grid")
                result[index] = math.ldexp(float(integer), -1074)
                undecided[index] = False
            elif precision >= P_MAX:
                raise WeightRoundingUndecided(
                    f"weight rounding undecided at {precision} Decimal digits"
                )
        if not undecided.any():
            return result[inverse]
        precision *= 2


def _lower_sum(high, low, other):
    """Lower bound on high + low + other, retaining cancellation."""
    first, error = _two_sum(high, other)
    tail = np.nextafter(error + low, -np.inf)
    return np.nextafter(first + tail, -np.inf)


def _round_quanta(high, low):
    integer = np.rint(high)
    dh, dl = _two_sum(high - integer, low)
    above = (dh > 0.5) | ((dh == 0.5) & (dl > 0))
    below = (dh < -0.5) | ((dh == -0.5) & (dl < 0))
    integer = integer + above.astype(float) - below.astype(float)
    dh, dl = _two_sum(high - integer, low)
    left = _lower_sum(dh, dl, 0.5)
    right = _lower_sum(-dh, -dl, 0.5)
    radius = np.nextafter(EPS_Q * np.abs(high), np.inf)
    ambiguous = (left <= radius) | (right <= radius)
    return integer * QUANTUM, ambiguous


def _normalize_log_weights_fast(
    log_weights: NDArray[np.floating],
) -> tuple[NDArray[np.float64], float]:
    values = np.asarray(log_weights, dtype=np.float64)
    normalizer = float(logsumexp(values))
    if not math.isfinite(normalizer):
        raise RuntimeError(
            "All particles have zero weight; the prior or forward model is "
            "inconsistent with the observation"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = np.exp(values - np.max(values))
    return scaled / scaled.sum(), normalizer


def _normalize_log_weights(
    log_weights: NDArray[np.floating],
) -> tuple[NDArray[np.float64], float]:
    """Normalize logs, certifying every exact subnormal result or raising.

    Ordinary weights retain the shifted-scale approximation.  A narrow strip
    above the smallest normal receives a double-double and directed-Decimal
    correction.  If the Decimal enclosure still crosses a rounding cell at
    the supported precision cap, raise ``WeightRoundingUndecided``.
    """
    values = np.asarray(log_weights, dtype=np.float64)
    weights, normalizer = _normalize_log_weights_fast(values)
    peak = float(np.max(values))
    with np.errstate(over="ignore", invalid="ignore"):
        shifted = values - peak
    candidates = shifted < -650.0
    zero = candidates & (shifted < -800.0)
    weights[zero] = 0.0
    candidates &= ~zero
    if not candidates.any():
        return weights, normalizer
    retained = shifted >= -160.0
    _, error = _two_sum(values[retained], -peak)
    high, low, exponent = _exp_dd(shifted[retained], error)
    total_high, total_low = _dd_sum(*_dd_scale(high, low, exponent))
    _, error = _two_sum(values[candidates], -peak)
    high, low, exponent = _exp_dd(shifted[candidates], error)
    qh, ql = _dd_div(*_dd_scale(high, low, exponent + 1074), total_high, total_low)
    correct = qh <= CORRECTION_LIMIT
    indices = np.flatnonzero(candidates)[correct]
    rounded, ambiguous = _round_quanta(qh[correct], ql[correct])
    if ambiguous.any():
        rounded[ambiguous] = _decide_exactly(
            values[indices[ambiguous]], values, shifted, peak
        )
    weights[indices] = rounded
    return weights, normalizer
