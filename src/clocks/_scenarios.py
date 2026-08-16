"""Shared scenario builders: the multi-mass-2D problem and the 3D
echolocation study (demos, scan harnesses, acceptance tests).

These builders share deterministic, physically valid configurations across
demos, scan harnesses, and acceptance tests.
"""

import math
from collections.abc import Callable
from itertools import product
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import helmert

from clocks._support import make_point_mass_prior_sampler, point_mass_support_mask
from clocks.api import build_particle_filter, simulate
from clocks.config import InferenceConfig, NoiseConfig, PriorConfig, SimulationConfig
from clocks.inference import ParticleFilter
from clocks.physics import (
    WEAK_FIELD_LIMIT,
    _point_mass_potential_batch,
    clock_rates,
    clock_rates_batch,
)
from clocks.results import SimulationResult
from clocks.types import ClockArray, MassConfig, Observation

TRUE_POSITIONS = np.array([[-3.0, 2.0], [4.0, -1.0]])
TRUE_MASSES = np.array([0.050, 0.030])
TRUTH = np.array([-3.0, 2.0, 4.0, -1.0, 0.050, 0.030])
N_CLOCKS = 10
TRACK_OFFSET = 3.0
MIN_SEPARATION = 1.5
N_OBSERVATIONS = 80
NOISE_STD = 0.005
N_PARTICLES = 4000
POSITION_RANGE = (-8.0, 8.0)
MASS_RANGE = (0.005, 0.15)
MULTI_ESS_TARGET = 0.8
MULTI_REJUVENATION_STEPS = 2
MULTI_PROPOSAL_SCALE = 2.38
# Pass rule: absolute posterior-mean error per parameter. Mass tolerances are
# scaled to the retuned weak-field truth rather than inherited from old units.
PASS_TOLERANCE = np.array([0.5, 0.5, 0.5, 0.5, 0.01, 0.01])


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
    normalized_error: float
    forward_model_evaluations: int
    ess_target: float
    rejuvenation_steps: int
    proposal_scale: float


def _instrument_batch_forward(pf: ParticleFilter) -> Callable[[], int]:
    """Count candidate rows passed through a filter's batch forward model."""
    forward_batch = pf.forward_model_batch
    if forward_batch is None:
        raise ValueError("scenario filters require a batch forward model")
    evaluations = 0

    def counted(particles: NDArray[np.floating]) -> NDArray[np.floating]:
        nonlocal evaluations
        evaluations += len(particles)
        return forward_batch(particles)

    pf.forward_model_batch = counted
    return lambda: evaluations


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
    ess_target: float = MULTI_ESS_TARGET,
    rejuvenation_steps: int = MULTI_REJUVENATION_STEPS,
    proposal_scale: float = MULTI_PROPOSAL_SCALE,
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
    inference_config = InferenceConfig(
        clock_array=clock_array,
        noise=NoiseConfig(observation_std=NOISE_STD),
        prior=PriorConfig(position_range=POSITION_RANGE, mass_range=MASS_RANGE),
        n_particles=N_PARTICLES,
        n_masses=2,
        ess_target=ess_target,
        rejuvenation_steps=rejuvenation_steps,
        proposal_scale=proposal_scale,
        seed=seed,
    )
    pf = build_particle_filter(inference_config)
    evaluation_count = _instrument_batch_forward(pf)
    for observation in sim.observations:
        pf.update(observation)
    estimate = pf.estimate()
    mean, std = estimate["mean"], estimate["std"]
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
        normalized_error=float(np.mean(error / PASS_TOLERANCE)),
        forward_model_evaluations=evaluation_count(),
        ess_target=ess_target,
        rejuvenation_steps=rejuvenation_steps,
        proposal_scale=proposal_scale,
    )


# --- 3D echolocation scenario (spec 2026-07-19-3d-echolocation-design) ---

# The head: 3x3x3 cubic lattice, spacing 1.0, centered on the origin.
# Circumradius (center to corner clocks) — the unit for range_r.
ECHO_R_HEAD = float(np.sqrt(3.0))
# Fixed exterior direction: exact unit vector, off-axis and off-diagonal
# so no projection or lattice symmetry hides the mass.
ECHO_DIRECTION = np.array([2.0, 3.0, 6.0]) / 7.0
ECHO_M_TRUE = 0.080
ECHO_NOISE_STD = 0.001
ECHO_N_OBSERVATIONS = 80
ECHO_N_PARTICLES = 6000
ECHO_MASS_RANGE = (0.005, 0.15)
ECHO_MIN_RANGE_R = 2.0  # circumradii; exterior means exterior, with clearance
ECHO_POSITION_HALFWIDTH = 16.0  # prior box covers max swept range 8*R_head~13.9
ECHO_SWEEP_RANGES = (2.0, 2.6, 3.5, 4.6, 6.1, 8.0)  # log-ish, circumradii
ECHO_ESS_TARGET = 0.8
ECHO_REJUVENATION_STEPS = 2
ECHO_PROPOSAL_SCALE = 2.38


def build_head_lattice() -> ClockArray:
    """The 27-clock head: 3x3x3 grid over {-1, 0, 1}^3."""
    grid = (-1.0, 0.0, 1.0)
    positions = np.array(list(product(grid, grid, grid)))
    return ClockArray(positions=positions, track_offset=0.0)


def echo_mass_position(range_r: float) -> NDArray[np.floating]:
    """Exterior mass position at range_r circumradii along ECHO_DIRECTION."""
    return ECHO_DIRECTION * range_r * ECHO_R_HEAD


def echo_mass_config(range_r: float, m_true: float = ECHO_M_TRUE) -> MassConfig:
    return MassConfig(
        positions=echo_mass_position(range_r).reshape(1, 3),
        masses=np.array([m_true]),
    )


def validate_echo_geometry(
    range_r: float, m_true: float, clock_array: ClockArray
) -> None:
    """Fail fast on interior masses and weak-field violations (spec section 1)."""
    if not math.isfinite(range_r):
        raise ValueError(f"range_r must be finite, got {range_r}")
    if range_r < ECHO_MIN_RANGE_R:
        raise ValueError(
            f"range_r={range_r} is below the exterior minimum "
            f"{ECHO_MIN_RANGE_R} circumradii: exterior means exterior"
        )
    positions = echo_mass_position(range_r).reshape(1, 1, 3)
    masses = np.array([[m_true]])
    potential, valid = _point_mass_potential_batch(positions, masses, clock_array)
    strength = np.abs(2.0 * potential)
    if (
        not valid[0]
        or not np.all(np.isfinite(strength))
        or np.any(strength > WEAK_FIELD_LIMIT)
    ):
        raise ValueError(
            "weak-field constraint violated: echo mass must produce "
            f"finite nonsingular potentials with |2*Phi| <= {WEAK_FIELD_LIMIT} "
            f"(range_r={range_r}, M_true={m_true})"
        )


# Provisional pass tolerances at the closest swept range; frozen after the
# tuning sweep (spec section 3a) — the tuning task records final values.
ECHO_PASS_POS_TOL = 1.0
ECHO_PASS_MASS_TOL = 0.04


class EchoRunResult(TypedDict):
    """One echolocation run: gate result plus study metrics (spec section 1)."""

    seed: int
    range_r: float
    passed: bool
    mean: NDArray[np.floating]
    std: NDArray[np.floating]
    position_error: float
    mass_error: float
    pos_std: float
    mass_std: float
    covered_3sigma: bool
    residual_over_noise: float
    normalized_error: float
    forward_model_evaluations: int
    ess_target: float
    rejuvenation_steps: int
    proposal_scale: float


def _center(rates: NDArray[np.floating]) -> NDArray[np.floating]:
    """Remove the across-clock mean: the head has no external reference."""
    return rates - rates.mean()


def contrast_matrix(n_clocks: int) -> NDArray[np.float64]:
    """Return orthonormal contrasts perpendicular to the common mode."""
    return np.asarray(helmert(n_clocks, full=False), dtype=np.float64)


def _make_echo_forward_models(
    clock_array: ClockArray,
) -> tuple[
    Callable[[NDArray[np.floating]], NDArray[np.floating]],
    Callable[[NDArray[np.floating]], NDArray[np.floating]],
]:
    """Scalar and batch forward models emitting orthonormal contrasts."""
    q = contrast_matrix(len(clock_array.positions))

    def forward(params: NDArray[np.floating]) -> NDArray[np.floating]:
        rates = clock_rates(
            MassConfig(positions=params[:3].reshape(1, 3), masses=params[3:4]),
            clock_array,
        )
        return q @ rates

    def forward_batch(particles: NDArray[np.floating]) -> NDArray[np.floating]:
        rates = clock_rates_batch(particles[:, :3], particles[:, 3], clock_array)
        return rates @ q.T

    return forward, forward_batch


def make_echo_observations(
    seed: int,
    range_r: float,
    *,
    n_observations: int = ECHO_N_OBSERVATIONS,
    noise_std: float = ECHO_NOISE_STD,
) -> tuple[SimulationResult, list[Observation], list[Observation]]:
    """Return simulation, centered display data, and likelihood contrasts."""
    clock_array = build_head_lattice()
    sim = simulate(
        SimulationConfig(
            clock_array=clock_array,
            ground_truth=echo_mass_config(range_r),
            noise=NoiseConfig(observation_std=noise_std),
            n_observations=n_observations,
            seed=seed,
        )
    )
    centered = [
        Observation(rates=_center(obs.rates), time=obs.time) for obs in sim.observations
    ]
    q = contrast_matrix(len(clock_array.positions))
    contrasts = [
        Observation(rates=q @ obs.rates, time=obs.time) for obs in sim.observations
    ]
    return sim, centered, contrasts


def build_echolocation_filter(
    seed: int,
    *,
    n_particles: int = ECHO_N_PARTICLES,
    noise_std: float = ECHO_NOISE_STD,
    ess_target: float = ECHO_ESS_TARGET,
    rejuvenation_steps: int = ECHO_REJUVENATION_STEPS,
    proposal_scale: float = ECHO_PROPOSAL_SCALE,
) -> ParticleFilter:
    """Raw ParticleFilter for the (x, y, z, M) exterior-mass problem.

    Built directly because the public API cannot express a contrast-space
    measurement model. Its rectangular prior is conditioned on strict
    physical validity. Metropolis proposals outside it are rejected.
    """
    clock_array = build_head_lattice()
    hw = ECHO_POSITION_HALFWIDTH
    prior_sampler = make_point_mass_prior_sampler(
        n_masses=1,
        n_dims=3,
        clock_array=clock_array,
        position_range=(-hw, hw),
        mass_range=ECHO_MASS_RANGE,
    )

    def log_prior(particles: NDArray[np.floating]) -> NDArray[np.floating]:
        valid = point_mass_support_mask(
            particles,
            n_masses=1,
            n_dims=3,
            clock_array=clock_array,
            position_range=(-hw, hw),
            mass_range=ECHO_MASS_RANGE,
        )
        lp = np.full(particles.shape[0], -np.inf)
        lp[valid] = 0.0
        return lp

    forward, forward_batch = _make_echo_forward_models(clock_array)
    return ParticleFilter(
        n_particles=n_particles,
        prior_sampler=prior_sampler,
        forward_model=forward,
        noise_std=noise_std,
        rng=np.random.default_rng(seed),
        forward_model_batch=forward_batch,
        log_prior_density=log_prior,
        ess_target=ess_target,
        rejuvenation_steps=rejuvenation_steps,
        proposal_scale=proposal_scale,
    )


def run_echolocation_3d(
    seed: int,
    range_r: float,
    *,
    n_particles: int = ECHO_N_PARTICLES,
    n_observations: int = ECHO_N_OBSERVATIONS,
    ess_target: float = ECHO_ESS_TARGET,
    rejuvenation_steps: int = ECHO_REJUVENATION_STEPS,
    proposal_scale: float = ECHO_PROPOSAL_SCALE,
) -> EchoRunResult:
    """One end-to-end echolocation run at a given range (in circumradii)."""
    clock_array = build_head_lattice()
    validate_echo_geometry(range_r, ECHO_M_TRUE, clock_array)
    sim, _, filter_observations = make_echo_observations(
        seed, range_r, n_observations=n_observations
    )
    pf = build_echolocation_filter(
        seed,
        n_particles=n_particles,
        ess_target=ess_target,
        rejuvenation_steps=rejuvenation_steps,
        proposal_scale=proposal_scale,
    )
    evaluation_count = _instrument_batch_forward(pf)
    for obs in filter_observations:
        pf.update(obs)

    est = pf.estimate()
    mean, std = est["mean"], est["std"]
    truth = np.append(echo_mass_position(range_r), ECHO_M_TRUE)
    error = np.abs(mean - truth)
    position_error = float(np.linalg.norm(mean[:3] - truth[:3]))
    mass_error = float(error[3])
    predicted_rates = clock_rates(
        MassConfig(positions=mean[:3].reshape(1, 3), masses=mean[3:4]),
        clock_array,
    )
    predicted_centered = _center(predicted_rates)
    residual = float(
        np.max(np.abs(predicted_centered - _center(sim.true_rates))) / ECHO_NOISE_STD
    )
    return EchoRunResult(
        seed=seed,
        range_r=range_r,
        passed=bool(
            position_error <= ECHO_PASS_POS_TOL and mass_error <= ECHO_PASS_MASS_TOL
        ),
        mean=mean,
        std=std,
        position_error=position_error,
        mass_error=mass_error,
        pos_std=float(np.linalg.norm(std[:3])),
        mass_std=float(std[3]),
        covered_3sigma=bool(np.all(error <= 3.0 * std)),
        residual_over_noise=residual,
        normalized_error=float(
            np.mean(
                [
                    position_error / ECHO_PASS_POS_TOL,
                    mass_error / ECHO_PASS_MASS_TOL,
                ]
            )
        ),
        forward_model_evaluations=evaluation_count(),
        ess_target=ess_target,
        rejuvenation_steps=rejuvenation_steps,
        proposal_scale=proposal_scale,
    )
