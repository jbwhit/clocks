"""Shared multi-mass-2D scenario: demo, scan harness, and acceptance test.

This is the problem instance whose premature-collapse failure motivated
the annealed jitter mode (spec:
docs/superpowers/specs/2026-07-02-annealed-jitter-design.md). It lives in
the package (not scripts/) because the demo console-scripts launch via
runpy and pytest imports from the repo root; neither puts scripts/ on
sys.path.
"""

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from clocks.api import infer, simulate
from clocks.config import InferenceConfig, NoiseConfig, PriorConfig, SimulationConfig
from clocks.physics import clock_rates
from clocks.results import InferenceResult
from clocks.types import ClockArray, MassConfig

TRUE_POSITIONS = np.array([[-3.0, 2.0], [4.0, -1.0]])
TRUE_MASSES = np.array([0.6, 0.4])
TRUTH = np.array([-3.0, 2.0, 4.0, -1.0, 0.6, 0.4])
N_CLOCKS = 10
TRACK_OFFSET = 3.0
MIN_SEPARATION = 1.5
N_OBSERVATIONS = 80
NOISE_STD = 0.005
N_PARTICLES = 4000
POSITION_RANGE = (-8.0, 8.0)
MASS_RANGE = (0.1, 2.0)
# Pass rule (see spec §3): abs posterior-mean error per parameter. These
# tolerances reproduce the June 2026 ad-hoc scan's implicit criterion.
PASS_TOLERANCE = np.array([0.5, 0.5, 0.5, 0.5, 0.1, 0.1])


class RunResult(TypedDict):
    """One scan run: gate result plus non-gating diagnostics."""

    seed: int
    passed: bool
    mean: NDArray[np.floating]
    std: NDArray[np.floating]
    max_abs_error: float
    covered_3sigma: bool
    max_posterior_std: float
    residual_over_noise: float


def generate_random_clocks(
    n: int,
    rng: np.random.Generator,
    *,
    bounds: tuple[float, float] = (-6.0, 6.0),
    min_sep: float = MIN_SEPARATION,
    exclude: list[tuple[float, float]] | None = None,
) -> NDArray[np.floating]:
    """Place n clocks on a 2D plane via rejection sampling.

    Keeps clocks at least min_sep apart from each other and from any
    positions listed in exclude (e.g. true mass locations).
    """
    placed: list[NDArray[np.floating]] = []
    blocked = [np.array(p) for p in (exclude or [])]
    while len(placed) < n:
        candidate = rng.uniform(bounds[0], bounds[1], 2)
        too_close = any(
            np.linalg.norm(candidate - p) < min_sep for p in placed + blocked
        )
        if not too_close:
            placed.append(candidate)
    return np.array(placed)


def passes(posterior_mean: NDArray[np.floating]) -> bool:
    """Gate: every parameter within its absolute tolerance of truth."""
    return bool(np.all(np.abs(posterior_mean - TRUTH) <= PASS_TOLERANCE))


def run_multi_mass_2d(
    seed: int,
    *,
    jitter: str = "annealed",
    jitter_std: float = 0.02,
    jitter_tau: float = 15.0,
) -> RunResult:
    """One end-to-end run: seed drives clocks, sim noise, and filter rng."""
    rng = np.random.default_rng(seed)
    mass_config = MassConfig(positions=TRUE_POSITIONS, masses=TRUE_MASSES)
    clock_positions = generate_random_clocks(
        N_CLOCKS,
        rng,
        exclude=[tuple(p) for p in TRUE_POSITIONS],
    )
    clock_array = ClockArray(positions=clock_positions, track_offset=TRACK_OFFSET)
    sim = simulate(
        SimulationConfig(
            clock_array=clock_array,
            ground_truth=mass_config,
            noise=NoiseConfig(observation_std=NOISE_STD),
            n_observations=N_OBSERVATIONS,
            seed=seed,
        )
    )
    result = infer(
        sim.observations,
        InferenceConfig(
            clock_array=clock_array,
            noise=NoiseConfig(observation_std=NOISE_STD),
            prior=PriorConfig(position_range=POSITION_RANGE, mass_range=MASS_RANGE),
            n_particles=N_PARTICLES,
            n_masses=2,
            jitter=jitter,
            jitter_std=jitter_std,
            jitter_tau=jitter_tau,
            seed=seed,
        ),
    )
    assert isinstance(result, InferenceResult)  # fixed-K mode
    mean, std = result.posterior_mean, result.posterior_std
    error = np.abs(mean - TRUTH)
    predicted = clock_rates(
        MassConfig(positions=mean[:4].reshape(2, 2), masses=mean[4:]),
        clock_array,
    )
    return RunResult(
        seed=seed,
        passed=passes(mean),
        mean=mean,
        std=std,
        max_abs_error=float(error.max()),
        covered_3sigma=bool(np.all(error <= 3.0 * std)),
        max_posterior_std=float(std.max()),
        residual_over_noise=float(
            np.max(np.abs(predicted - sim.true_rates)) / NOISE_STD
        ),
    )
