"""End-to-end multi-mass 2D gravitational time dilation demo.

~10 clocks placed randomly on a 2D plane, two hidden masses.
The particle filter deduces both positions and magnitudes (6 parameters:
x1, y1, x2, y2, M1, M2) from noisy clock readings.
An ordering constraint (x1 < x2) breaks label-switching symmetry.
"""

from pathlib import Path

import numpy as np

from clocks.inference import ParticleFilter
from clocks.noise import add_clock_noise
from clocks.physics import clock_rates, clock_rates_batch_multi
from clocks.types import ClockArray, MassConfig, Observation
from clocks.viz import animate_inference_multi_2d

# --- Configuration ---
TRUE_X1, TRUE_Y1 = -3.0, 2.0
TRUE_X2, TRUE_Y2 = 4.0, -1.0
TRUE_M1 = 0.6
TRUE_M2 = 0.4
N_CLOCKS = 10
TRACK_OFFSET = 3.0
MIN_SEPARATION = 1.5
N_OBSERVATIONS = 80
NOISE_STD = 0.005
N_PARTICLES = 4000
JITTER_STD = 0.02
SEED = 42
OUTPUT_PATH = Path("output/demo_multi_mass_2d.gif")


def generate_random_clocks(
    n: int,
    rng: np.random.Generator,
    *,
    bounds: tuple[float, float] = (-6.0, 6.0),
    min_sep: float = MIN_SEPARATION,
    exclude: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Place n clocks on a 2D plane via rejection sampling.

    Keeps clocks at least min_sep apart from each other and from
    any positions listed in exclude (e.g. true mass locations).
    """
    placed: list[np.ndarray] = []
    blocked = [np.array(p) for p in (exclude or [])]

    while len(placed) < n:
        candidate = rng.uniform(bounds[0], bounds[1], 2)
        too_close = any(
            np.linalg.norm(candidate - p) < min_sep for p in placed + blocked
        )
        if not too_close:
            placed.append(candidate)

    return np.array(placed)


def enforce_ordering(particles: np.ndarray) -> np.ndarray:
    """Swap all params for mass 1/2 when x1 > x2 (breaks label-switching)."""
    swap = particles[:, 0] > particles[:, 2]
    particles[swap, 0], particles[swap, 2] = (
        particles[swap, 2].copy(),
        particles[swap, 0].copy(),
    )
    particles[swap, 1], particles[swap, 3] = (
        particles[swap, 3].copy(),
        particles[swap, 1].copy(),
    )
    particles[swap, 4], particles[swap, 5] = (
        particles[swap, 5].copy(),
        particles[swap, 4].copy(),
    )
    return particles


def main() -> None:
    rng = np.random.default_rng(SEED)

    # Ground truth
    mass_config = MassConfig(
        positions=np.array([[TRUE_X1, TRUE_Y1], [TRUE_X2, TRUE_Y2]]),
        masses=np.array([TRUE_M1, TRUE_M2]),
    )

    # Random clock placement
    clock_positions = generate_random_clocks(
        N_CLOCKS,
        rng,
        exclude=[(TRUE_X1, TRUE_Y1), (TRUE_X2, TRUE_Y2)],
    )
    clock_array = ClockArray(positions=clock_positions, track_offset=TRACK_OFFSET)

    true_rates = clock_rates(mass_config, clock_array)
    print(
        f"True masses: ({TRUE_X1},{TRUE_Y1}) M={TRUE_M1}, "
        f"({TRUE_X2},{TRUE_Y2}) M={TRUE_M2}"
    )
    print(f"Clocks: {N_CLOCKS} randomly placed")
    print(f"True rates: {true_rates}")

    # Generate observations
    observations: list[Observation] = []
    for t in range(N_OBSERVATIONS):
        noisy = add_clock_noise(true_rates, NOISE_STD, rng)
        observations.append(Observation(rates=noisy, time=float(t)))

    # Prior: uniform positions in [-8,8], uniform masses in [0.1,2.0], x1 < x2
    def prior_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        x1 = rng.uniform(-8, 8, n)
        y1 = rng.uniform(-8, 8, n)
        x2 = rng.uniform(-8, 8, n)
        y2 = rng.uniform(-8, 8, n)
        m1 = rng.uniform(0.1, 2.0, n)
        m2 = rng.uniform(0.1, 2.0, n)
        particles = np.column_stack([x1, y1, x2, y2, m1, m2])
        return enforce_ordering(particles)

    # Forward models
    def forward_model(params: np.ndarray) -> np.ndarray:
        mc = MassConfig(
            positions=np.array([[params[0], params[1]], [params[2], params[3]]]),
            masses=np.array([params[4], params[5]]),
        )
        return clock_rates(mc, clock_array)

    def forward_model_batch(particles: np.ndarray) -> np.ndarray:
        # particles: (n, 6) → [x1, y1, x2, y2, M1, M2]
        mass_pos = particles[:, :4].reshape(-1, 2, 2)  # (n, 2, 2)
        masses = particles[:, 4:]  # (n, 2)
        return clock_rates_batch_multi(mass_pos, masses, clock_array)

    def log_prior_fn(particles: np.ndarray) -> np.ndarray:
        lp = np.zeros(particles.shape[0])
        lp[np.any(particles[:, 4:] <= 0, axis=1)] = -np.inf  # masses > 0
        lp[np.any((particles[:, :4] < -8) | (particles[:, :4] > 8), axis=1)] = -np.inf
        return lp

    pf = ParticleFilter(
        n_particles=N_PARTICLES,
        prior_sampler=prior_sampler,
        forward_model=forward_model,
        noise_std=NOISE_STD,
        jitter_std=JITTER_STD,
        rng=rng,
        forward_model_batch=forward_model_batch,
        constraint_fn=enforce_ordering,
        jitter="covariance",
        log_prior=log_prior_fn,
    )

    # Animate and save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating animation → {OUTPUT_PATH}")
    animate_inference_multi_2d(
        clock_array=clock_array,
        mass_config=mass_config,
        observations=observations,
        pf=pf,
        output_path=OUTPUT_PATH,
        fps=4,
    )

    # Print final estimate
    est = pf.estimate()
    print(f"\nFinal estimate after {N_OBSERVATIONS} observations:")
    print(f"  x1 = {est['mean'][0]:.3f} ± {est['std'][0]:.3f}  (true: {TRUE_X1})")
    print(f"  y1 = {est['mean'][1]:.3f} ± {est['std'][1]:.3f}  (true: {TRUE_Y1})")
    print(f"  x2 = {est['mean'][2]:.3f} ± {est['std'][2]:.3f}  (true: {TRUE_X2})")
    print(f"  y2 = {est['mean'][3]:.3f} ± {est['std'][3]:.3f}  (true: {TRUE_Y2})")
    print(f"  M1 = {est['mean'][4]:.3f} ± {est['std'][4]:.3f}  (true: {TRUE_M1})")
    print(f"  M2 = {est['mean'][5]:.3f} ± {est['std'][5]:.3f}  (true: {TRUE_M2})")
    print(f"  ESS = {est['ess']:.0f} / {N_PARTICLES}")
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
