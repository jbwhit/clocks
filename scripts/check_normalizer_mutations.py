"""Run one in-memory production-normalizer fault in this project environment.

Invoke with ``uv run --no-sync python scripts/check_normalizer_mutations.py``.
This tool does not bootstrap or synchronize dependencies.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

MUTATIONS = (
    "old_midpoint",
    "float_classification",
    "no_dd_interval",
    "collapsed_decimal_interval",
)


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def verify_production_modules():
    import clocks._weight_normalization as normalizer
    import clocks.inference as inference

    expected_normalizer = _root() / "src/clocks/_weight_normalization.py"
    expected_inference = _root() / "src/clocks/inference.py"
    normalizer_path = Path(normalizer.__file__).resolve()
    inference_path = Path(inference.__file__).resolve()
    if normalizer_path != expected_normalizer:
        raise RuntimeError(
            "normalizer did not import from the expected production path: "
            f"{normalizer_path} != {expected_normalizer}"
        )
    if inference_path != expected_inference:
        raise RuntimeError(
            "inference did not import from the expected production path: "
            f"{inference_path} != {expected_inference}"
        )
    print(f"Verified production normalizer import: {normalizer_path}")
    print(f"Verified production inference import: {inference_path}")
    return normalizer, inference


def _apply_mutation(name: str) -> None:
    normalizer, inference = verify_production_modules()

    if name == "old_midpoint":

        def old_midpoint(high, low):
            integer = np.floor(high)
            first, second = normalizer._two_sum(high - integer - 0.5, low)
            near = np.abs(first) - np.abs(second) <= normalizer.EPS_Q * high
            return (integer + ((first > 0) & ~near)) * normalizer.QUANTUM, near

        normalizer._round_quanta = old_midpoint
    elif name == "float_classification":
        original = normalizer._normalize_log_weights

        def preserve_float_normal(values):
            weights, log_normalizer = original(values)
            branch, _ = normalizer._normalize_log_weights_fast(values)
            normal = branch >= normalizer.TINY
            weights[normal] = branch[normal]
            return weights, log_normalizer

        normalizer._normalize_log_weights = preserve_float_normal
        inference._normalize_log_weights = preserve_float_normal
    elif name == "no_dd_interval":
        normalizer.EPS_Q = 0.0
    elif name == "collapsed_decimal_interval":

        def collapsed(context, argument):
            rounded = context.exp(argument)
            return rounded, rounded

        normalizer._exp_interval = collapsed
    else:  # argparse choices makes this unreachable outside direct use.
        raise ValueError(f"unknown mutation: {name}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mutation", choices=MUTATIONS)
    args = parser.parse_args(argv)
    print(f"Selected mutation: {args.mutation}")
    _apply_mutation(args.mutation)
    return pytest.main(["tests", "-m", "", "--tb=short"])


if __name__ == "__main__":
    raise SystemExit(main())
