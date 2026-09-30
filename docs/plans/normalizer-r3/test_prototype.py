"""Design experiments; run explicitly until the design is accepted.

Set NORMALIZER_REVIEW_REVISION=r2 to demonstrate the recovered regressions.
"""

import importlib.machinery
import importlib.util
import os
from decimal import ROUND_HALF_EVEN, Context, Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
REVISION = os.environ.get("NORMALIZER_REVIEW_REVISION", "r3")
SOURCE = (
    ROOT / "docs/plans/normalizer-r2/prototype.py.txt"
    if REVISION == "r2"
    else Path(__file__).with_name("prototype.py")
)
loader = importlib.machinery.SourceFileLoader("normalizer_candidate", str(SOURCE))
spec = importlib.util.spec_from_loader(loader.name, loader)
p = importlib.util.module_from_spec(spec)
loader.exec_module(p)


def reference(values):
    context = Context(prec=650, rounding=ROUND_HALF_EVEN)
    exact = [Decimal.from_float(float(v)) for v in values]
    peak = max(exact)
    terms = [context.exp(context.subtract(v, peak)) for v in exact]
    total = Decimal(0)
    for term in terms:
        total = context.add(total, term)
    return [float(context.divide(term, total)) for term in terms]


@pytest.mark.parametrize(
    "values,index,quanta",
    [
        (
            [
                -14.171583200817665,
                -14.628378654822516,
                -722.077393183382,
                -44.60327978867792,
                -78.51114028012964,
            ],
            2,
            4503599627370499,
        ),
        ([98.28463995512519, -610.1117785771389], 1, 4503599627370492),
        ([-59.88145743137663, -782.1408195747232], 1, 4294967297),
        ([-3e-14, -745.1332191019412], 1, 1),
        ([0.0, 0.0, -744.3], 2, 1),
        ([0.0, -np.log(2), -745.0], 2, 0),
        (
            [
                0.0,
                -1.778318882035904,
                -2.9406092642692605,
                -0.7020900597390054,
                -744.5924635348828,
                -29.488865949156533,
                -63.207674006605856,
                -98.21345526407063,
                -130.8394966545195,
            ],
            4,
            0,
        ),
        (
            [
                0.0,
                -1.1744667844096757,
                -0.6024338098404867,
                -0.5413190888213227,
                -744.2418600740691,
                -30.375152889020022,
                -67.2253162057163,
                -99.17755262428673,
                -132.63630998326656,
            ],
            4,
            1,
        ),
    ],
)
def test_rounding_regressions(values, index, quanta):
    expected = np.ldexp(float(quanta), -1074)
    assert reference(values)[index] == expected
    assert p.normalize_proto(values)[0][index] == expected


def test_large_offset_stays_normalized():
    weights, _ = p.normalize_proto([1e18, 1e18, 1e18 - 744.3])
    assert np.array_equal(weights, [0.5, 0.5, 0.0])


@pytest.mark.parametrize("peak", [0.0, 5e-324, -5e-324, 1e308, -1e308])
def test_equal_and_zero_weights(peak):
    weights, _ = p.normalize_proto([peak, peak, -np.inf])
    assert np.array_equal(weights, [0.5, 0.5, 0.0])


def test_exact_subnormal_band_against_reference():
    rng = np.random.default_rng(414)
    tiny = np.finfo(np.float64).tiny
    tested = 0
    for _ in range(100):
        heads = rng.normal(0, 3, 3)
        tail = rng.uniform(-748, -707)
        values = np.r_[heads, tail] + rng.uniform(-200, 200)
        expected = reference(values)
        actual, _ = p.normalize_proto(values)
        if expected[-1] < tiny:
            tested += 1
            assert actual[-1] == expected[-1]
    assert tested > 80


def test_exp_error_on_random_and_underflow_arguments(monkeypatch):
    checked_products = 0
    original_product = p._two_prod

    def checked_product(a, b):
        nonlocal checked_products
        high, low = original_product(a, b)
        for x, y, h, lo in zip(*np.broadcast_arrays(a, b, high, low), strict=True):
            assert Fraction(float(x)) * Fraction(float(y)) == (
                Fraction(float(h)) + Fraction(float(lo))
            )
            checked_products += 1
        return high, low

    monkeypatch.setattr(p, "_two_prod", checked_product)
    rng = np.random.default_rng(812)
    shifted = np.r_[rng.uniform(-800, 0, 200), 0.0, -5e-324, -(2.0**-351), -19 * p.LN2]
    error = rng.uniform(-(2.0**-43), 2.0**-43, shifted.size)
    error[-4:] = [0.0, -5e-324, -(2.0**-351), 0.0]
    high, low, powers = p._exp_dd(shifted, error)
    context = Context(prec=100, rounding=ROUND_HALF_EVEN)
    for s, e, h, lo, power in zip(shifted, error, high, low, powers, strict=True):
        exponent = context.add(
            Decimal.from_float(float(s)), Decimal.from_float(float(e))
        )
        expected = context.exp(exponent)
        mantissa = context.add(
            Decimal.from_float(float(h)), Decimal.from_float(float(lo))
        )
        actual = context.multiply(
            mantissa, context.power(Decimal(2), Decimal(int(power)))
        )
        relative = context.divide(context.subtract(actual, expected), expected)
        assert abs(float(relative)) < 2.0**-96
    assert checked_products > 2000


def test_rounding_refuses_a_cell_when_error_interval_crosses_its_boundary():
    # Either sign is possible for the exact Q within the advertised error:
    # an estimate just above 0.5 cannot certify rounding upward here.
    _, ambiguous = p._round_quanta(np.array([0.5]), np.array([2.0**-100]))
    assert ambiguous[0]
