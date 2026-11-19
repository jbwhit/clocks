"""Fuzz _normalize_log_weights against an exact Decimal reference."""

import math
from decimal import Context, Decimal

import numpy as np

from clocks._weight_normalization import (
    WeightRoundingUndecided,
    _normalize_log_weights,
)

CTX = Context(prec=400, Emin=-999999, Emax=999999)


def exact(values):
    finite = [v for v in values if math.isfinite(v)]
    peak = max(finite)
    terms = [
        CTX.exp(CTX.subtract(Decimal(v), Decimal(peak)))
        if math.isfinite(v)
        else Decimal(0)
        for v in values
    ]
    total = sum(terms, Decimal(0))
    total = CTX.plus(total)
    return np.array([float(CTX.divide(t, total)) for t in terms])


def check(values, label):
    values = np.asarray(values, dtype=np.float64)
    try:
        got, _ = _normalize_log_weights(values)
    except WeightRoundingUndecided as e:
        return ("undecided", label, str(e))
    ref = exact(values)
    tiny = ref < 2.0**-1022
    bad_sub = np.flatnonzero(tiny & (got != ref))
    # Ordinary weights: relative error
    big = ~tiny & (ref > 0)
    rel = np.max(np.abs(got[big] - ref[big]) / ref[big]) if big.any() else 0.0
    zero_but_ref_pos = np.flatnonzero((got == 0) & (ref > 0))
    pos_but_ref_zero = np.flatnonzero((got > 0) & (ref == 0))
    if bad_sub.size or zero_but_ref_pos.size or pos_but_ref_zero.size or rel > 1e-12:
        return (
            "BAD",
            label,
            values[bad_sub][:3],
            got[bad_sub][:3],
            ref[bad_sub][:3],
            rel,
            zero_but_ref_pos[:3],
            pos_but_ref_zero[:3],
        )
    return None


def main():
    rng = np.random.default_rng(0)
    problems = []
    n_cases = 0
    for trial in range(800):
        n = int(rng.choice([2, 3, 5, 17, 200]))
        offset = float(rng.choice([0.0, 1e3, -1e6, 1e12, 1e18, -3e17]))
        base = rng.uniform(-30, 0, n)
        k = rng.integers(1, n)
        base[:k] = -rng.uniform(700, 760, k)  # subnormal region
        if rng.random() < 0.3:
            base[0] = -np.inf
        values = offset + base
        r = check(values, f"trial {trial} offset {offset}")
        n_cases += 1
        if r:
            problems.append(r)
    # Hand-picked edges
    edges = {
        "single dominant": [0.0, -1000.0, -np.inf],
        "one finite": [-np.inf, 5.0, -np.inf],
        "subnormal boundary": [0.0, -math.log(2), -745.0],
        "exact half quantum": [0.0, -1075 * math.log(2.0)],
        "min normal": [0.0, math.log(2.0**-1022)],
        "max-magnitude finite": [1.7e308, -1.7e308],
        "dd peak": [1e18, 1e18, 1e18 - 744.3],
        "many near min": list(-745.0 + np.linspace(-0.5, 0.5, 50)) + [0.0],
    }
    for label, values in edges.items():
        n_cases += 1
        r = check(values, label)
        if r:
            problems.append(r)
    for label, values in {
        "all -inf": [-np.inf, -np.inf],
        "nan": [0.0, np.nan],
        "+inf": [0.0, np.inf],
        "empty": [],
    }.items():
        try:
            out = _normalize_log_weights(np.array(values, dtype=float))
            problems.append(("NO ERROR", label, out))
        except (RuntimeError, ValueError) as e:
            print(f"{label}: raises {type(e).__name__}: {e}")
    print(f"cases={n_cases} problems={len(problems)}")
    for p in problems[:20]:
        print(p)


if __name__ == "__main__":
    main()
