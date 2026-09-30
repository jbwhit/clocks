"""Independent tests of the directed Decimal fallback, bypassing DD decisions."""

import importlib.util
import math
from concurrent.futures import ThreadPoolExecutor
from decimal import (
    MAX_EMAX,
    MIN_EMIN,
    ROUND_DOWN,
    ROUND_HALF_EVEN,
    ROUND_UP,
    Context,
    Decimal,
    DefaultContext,
    DivisionByZero,
    FloatOperation,
    Inexact,
    InvalidOperation,
    Overflow,
    Rounded,
    Subnormal,
    Underflow,
    getcontext,
    setcontext,
)
from pathlib import Path

import numpy as np
import pytest

import clocks.inference as inference

ROOT = Path(__file__).resolve().parents[3]
assert Path(inference.__file__).resolve() == ROOT / "src/clocks/inference.py"
spec = importlib.util.spec_from_file_location(
    "normalizer_interval_candidate", Path(__file__).with_name("prototype.py")
)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)

# Archived round-1 hard case, retained in normalizer-r2/reproduce.py.txt.
HARD = [
    0.0,
    8.8448886818372e-14,
    -68.25321157879489,
    -100.6207187190934,
    -132.62286311459286,
    -166.42988573491488,
    -198.3315821246221,
    -744.4400719213812,
]


def reference_context(precision=650):
    return Context(
        prec=precision,
        rounding=ROUND_HALF_EVEN,
        Emin=MIN_EMIN,
        Emax=MAX_EMAX,
        capitals=1,
        clamp=0,
        flags=[],
        traps=[InvalidOperation, DivisionByZero, Overflow],
    )


def reference_quanta(values, index):
    context = reference_context()
    exact_context = reference_context(1500)
    exact_context.traps[Inexact] = True
    exact = [Decimal.from_float(float(value)) for value in values]
    peak = max(exact)
    terms = [context.exp(exact_context.subtract(value, peak)) for value in exact]
    total = Decimal(0)
    for term in terms:
        total = context.add(total, term)
    return context.divide(context.multiply(terms[index], Decimal(2**1074)), total)


def fallback(values, indexes, stats=None):
    values = np.asarray(values, dtype=np.float64)
    peak = float(values.max())
    with np.errstate(over="ignore"):
        shifted = values - peak
    return p._decide_exactly(values[indexes], values, shifted, peak, stats)


@pytest.mark.parametrize("argument", ["-0.125", "-1", "-744", "-5e-324"])
def test_exponential_endpoints_enclose_independent_high_precision_value(argument):
    exact = Decimal(argument)
    expected = reference_context().exp(exact)
    lower, upper = p._exp_interval(p._isolated_context(56), exact)
    assert Decimal(0) <= lower < expected < upper


@pytest.mark.parametrize("helper_step,expected_integer", [(0, 1), (1, 0)])
def test_hard_boundary_requires_more_than_56_digits(helper_step, expected_integer):
    values = HARD.copy()
    if helper_step:
        values[-2] = float(np.nextafter(values[-2], np.inf))
    q = reference_quanta(values, -1)
    residual = reference_context().subtract(q, Decimal("0.5"))
    assert Decimal(0) < residual.copy_abs() < Decimal("1e-90")
    assert (residual > 0) == bool(expected_integer)
    stats = {}
    actual = fallback(values, [-1, -1], stats)
    assert np.array_equal(actual, [math.ldexp(expected_integer, -1074)] * 2)
    assert [row[0] for row in stats["rounds"]] == [56, 112]
    assert all(row[2] == 1 for row in stats["rounds"])


def test_direct_fallback_rounds_just_above_smallest_normal_on_quantum_grid():
    values = [
        -14.171583200817665,
        -14.628378654822516,
        -722.077393183382,
        -44.60327978867792,
        -78.51114028012964,
    ]
    q = reference_quanta(values, 2)
    assert Decimal(4503599627370499) < q < Decimal("4503599627370499.5")
    assert fallback(values, [2])[0] == math.ldexp(4503599627370499.0, -1074)


def gaussian_cap_values():
    """Recreate the archived Gaussian cap population without archived imports."""
    context = reference_context()
    exact_context = reference_context(1500)
    base = float(np.log(1.0 / 32))
    normalizer = inference._gaussian_log_normalizer(1.0, 1.0)

    def log_weight(particle):
        likelihood = -normalizer - 0.5 * np.float64(particle) ** 2
        return float(base + (likelihood + normalizer))

    def term(particle):
        difference = exact_context.subtract(
            Decimal.from_float(log_weight(particle)), Decimal.from_float(base)
        )
        return context.exp(difference)

    tail = math.sqrt(2 * 744.7)
    tail_term = term(tail)
    target = context.multiply(tail_term, Decimal(2**1075))
    remainder = context.subtract(target, context.add(Decimal(1), tail_term))
    particles = [0.0, tail]
    while remainder > Decimal("1e-240"):
        particle = math.sqrt(-2 * float(context.ln(remainder)))
        while term(particle) >= remainder:
            particle = float(np.nextafter(particle, np.inf))
        particles.append(particle)
        remainder = context.subtract(remainder, term(particle))
        assert len(particles) <= 32
    particles.extend([100.0] * (32 - len(particles)))
    return [log_weight(particle) for particle in particles]


def test_real_gaussian_boundary_exhausts_precision_cap():
    values = gaussian_cap_values()
    q = reference_quanta(values, 1)
    residual = reference_context().subtract(q, Decimal("0.5"))
    assert Decimal("1e-251") < residual < Decimal("1e-248")
    assert len(values) == 32 and np.isfinite(values).all()
    stats = {}
    with pytest.raises(p.WeightRoundingUndecided, match="224"):
        fallback(values, [1], stats)
    assert [row[0] for row in stats["rounds"]] == [56, 112, 224]


def context_state(context):
    return (
        context.prec,
        context.rounding,
        context.Emin,
        context.Emax,
        context.capitals,
        context.clamp,
        dict(context.flags),
        dict(context.traps),
    )


def make_hostile(context):
    context.prec = 3
    context.rounding = ROUND_UP
    context.Emin = -5
    context.Emax = 5
    context.capitals = 0
    context.clamp = 1
    for signal in (Inexact, Rounded, Subnormal, Underflow, FloatOperation):
        context.traps[signal] = True
    context.clear_flags()
    context.flags[Rounded] = True


def assert_context_independent_fallback():
    context = getcontext()
    before = context_state(context)
    actual = fallback(HARD, [-1])
    assert actual[0] == math.ldexp(1.0, -1074)
    assert context_state(context) == before
    # This exact subtraction would round on a 56-digit or ambient context.
    actual = fallback([-3e-14, -745.1332191019412], [1])
    assert actual[0] == math.ldexp(1.0, -1074)
    assert context_state(context) == before
    # A tiny weight exercises Emin; an extreme exact difference also needs Emax.
    difference = p._exact_difference(-np.finfo(np.float64).max, math.ldexp(1.0, -1074))
    assert len(difference.as_tuple().digits) == 1383
    assert context_state(context) == before


def test_hostile_caller_context_and_existing_flags_are_unchanged():
    saved = getcontext().copy()
    try:
        make_hostile(getcontext())
        assert_context_independent_fallback()
    finally:
        setcontext(saved)


def test_fresh_thread_inherits_hostile_defaults_but_fallback_is_isolated():
    saved = DefaultContext.copy()
    try:
        make_hostile(DefaultContext)
        DefaultContext.rounding = ROUND_DOWN
        before = context_state(DefaultContext)

        def exercise():
            context = getcontext()
            assert context.prec == 3 and context.rounding == ROUND_DOWN
            assert context.Emax == 5 and context.traps[FloatOperation]
            assert_context_independent_fallback()

        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(exercise).result()
        assert context_state(DefaultContext) == before
    finally:
        for field in ("prec", "rounding", "Emin", "Emax", "capitals", "clamp"):
            setattr(DefaultContext, field, getattr(saved, field))
        DefaultContext.traps = saved.traps.copy()
        DefaultContext.flags = saved.flags.copy()


def test_overflowed_finite_shifts_remain_in_omitted_mass_count(monkeypatch):
    # Output zero alone cannot distinguish a valid interval from a lost term.
    # Observe the tail coefficient while executing the real Decimal operations.
    tail_counts = []
    original_context = p._isolated_context

    class RecordingContext:
        def __init__(self, context):
            self.context = context

        def __getattr__(self, name):
            return getattr(self.context, name)

        def multiply(self, left, right):
            if right == Decimal("1e-80"):
                tail_counts.append(int(left))
            return self.context.multiply(left, right)

    def recording_context(*args, **kwargs):
        return RecordingContext(original_context(*args, **kwargs))

    monkeypatch.setattr(p, "_isolated_context", recording_context)
    largest = np.finfo(np.float64).max
    values = [largest, -largest, -largest, -np.inf]
    assert np.array_equal(fallback(values, [1, 2]), [0.0, 0.0])
    assert tail_counts == [2]


def test_exact_difference_covers_full_binary64_span():
    largest = np.finfo(np.float64).max
    smallest = math.ldexp(1.0, -1074)
    difference = p._exact_difference(-largest, smallest)
    expected = reference_context(1500).subtract(
        Decimal.from_float(-largest), Decimal.from_float(smallest)
    )
    assert difference == expected
    assert len(difference.as_tuple().digits) == 1383
