"""Measure production strict and fast normalizers on fixed representative inputs.

The final timing is a fixed 32-particle Gaussian filter update matching the
real-filter integration workload, written locally rather than imported from tests.
Invoke with ``uv run --no-sync python scripts/benchmark_normalizer.py``; this
project-environment tool does not bootstrap or synchronize dependencies.
"""

from __future__ import annotations

import platform
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from clocks._weight_normalization import (
    _normalize_log_weights,
    _normalize_log_weights_fast,
)
from clocks.inference import ParticleFilter
from clocks.types import Observation


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


PARTICLES = np.array(
    [
        0.0,
        38.59274543227004,
        1.1064306244017321,
        8.636106538188985,
        11.899052313958661,
        14.353592365887044,
        16.43727083070602,
        18.21641203455001,
        19.82741536843704,
        21.34993445096983,
        22.75099453369033,
        24.198471652646177,
        25.404748634761873,
        26.610777563855013,
        27.76342919878422,
        28.925871480130304,
        29.96695254782559,
        30.951311931387643,
        32.021162697594185,
        32.95202272094582,
    ]
    + [100.0] * 12
).reshape(-1, 1)


def make_filter() -> ParticleFilter:
    return ParticleFilter(
        32,
        lambda rng, n: PARTICLES.copy(),
        lambda particle: particle,
        1.0,
        log_prior_density=lambda particles: -0.5 * np.sum(particles**2, axis=1),
        forward_model_batch=lambda particles: particles,
        ess_target=0.8,
        rejuvenation_steps=1,
        rng=np.random.default_rng(54),
    )


def _median_seconds(call: Callable[[], object]) -> float:
    call()
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        call()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def main() -> int:
    verify_production_modules()
    print(f"Python: {sys.version.replace(chr(10), ' ')}")
    print(f"NumPy: {np.__version__}")
    print(f"Platform: {platform.platform()}")
    rng = np.random.default_rng(731)
    for population in (2_000, 40_000):
        heads = rng.normal(0.0, 1.0, population - 1)
        for tail in (-700.0, -720.0, -740.0, -800.0, -np.inf):
            values = np.concatenate((heads, np.array([tail])))
            strict = _median_seconds(lambda: _normalize_log_weights(values))
            fast = _median_seconds(lambda: _normalize_log_weights_fast(values))
            print(
                f"normalizer n={population:,} tail={tail:g} "
                f"strict_ms={strict * 1_000:.3f} fast_ms={fast * 1_000:.3f}"
            )
    observation = Observation(np.zeros(1), 0.0)
    update = _median_seconds(lambda: make_filter().update(observation))
    print(f"32-particle fixed Gaussian filter update_ms={update * 1_000:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
