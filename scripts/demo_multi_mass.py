"""End-to-end multi-mass 1D gravitational time dilation demo.

5 clocks along a track, two hidden masses. The particle filter
deduces both positions and magnitudes from noisy clock readings.
An ordering constraint (x1 < x2) breaks label-switching symmetry.
"""

from pathlib import Path

import numpy as np

from clocks.inference import ParticleFilter
from clocks.noise import add_clock_noise
from clocks.physics import clock_rates, clock_rates_batch_multi
from clocks.types import ClockArray, MassConfig, Observation
from clocks.viz import animate_inference_multi_1d

# --- Configuration ---
TRUE_X1 = -2.0
TRUE_X2 = 4.5
TRUE_M1 = 0.6
TRUE_M2 = 0.4
CLOCK_POSITIONS = [-6.0, -3.0, 0.0, 3.0, 6.0]
TRACK_OFFSET = 1.0
N_OBSERVATIONS = 80
NOISE_STD = 0.005
N_PARTICLES = 3000
JITTER_STD = 0.02
SEED = 42
OUTPUT_PATH = Path("output/demo_multi_mass.gif")


def enforce_ordering(particles: np.ndarray) -> np.ndarray:
    """Swap x1/x2 and M1/M2 when x1 > x2 to break label-switching symmetry."""
    swap = particles[:, 0] > particles[:, 1]
    particles[swap, 0], particles[swap, 1] = (
        particles[swap, 1].copy(),
        particles[swap, 0].copy(),
    )
    particles[swap, 2], particles[swap, 3] = (
        particles[swap, 3].copy(),
        particles[swap, 2].copy(),
    )
    return particles


def main() -> None:
    rng = np.random.default_rng(SEED)

    # Ground truth
    mass_config = MassConfig(
        positions=np.array([[TRUE_X1], [TRUE_X2]]),
        masses=np.array([TRUE_M1, TRUE_M2]),
    )
    clock_array = ClockArray(
        positions=np.array([[x] for x in CLOCK_POSITIONS]),
        track_offset=TRACK_OFFSET,
    )
    true_rates = clock_rates(mass_config, clock_array)

    print(f"True masses: x1={TRUE_X1}, x2={TRUE_X2}, M1={TRUE_M1}, M2={TRUE_M2}")
    print(f"True rates: {true_rates}")

    # Generate observations
    observations: list[Observation] = []
    for t in range(N_OBSERVATIONS):
        noisy = add_clock_noise(true_rates, NOISE_STD, rng)
        observations.append(Observation(rates=noisy, time=float(t)))

    # Prior: sample x pair and sort so x1 < x2
    def prior_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        x = rng.uniform(-8, 8, (n, 2))
        x.sort(axis=1)  # enforce x1 < x2
        m1 = rng.uniform(0.1, 2.0, n)
        m2 = rng.uniform(0.1, 2.0, n)
        return np.column_stack([x[:, 0], x[:, 1], m1, m2])

    # Forward models
    def forward_model(params: np.ndarray) -> np.ndarray:
        mc = MassConfig(
            positions=np.array([[params[0]], [params[1]]]),
            masses=np.array([params[2], params[3]]),
        )
        return clock_rates(mc, clock_array)

    def forward_model_batch(particles: np.ndarray) -> np.ndarray:
        # particles: (n, 4) → [x1, x2, M1, M2]
        mass_pos = particles[:, :2, np.newaxis]  # (n, 2, 1)
        masses = particles[:, 2:]  # (n, 2)
        return clock_rates_batch_multi(mass_pos, masses, clock_array)

    pf = ParticleFilter(
        n_particles=N_PARTICLES,
        prior_sampler=prior_sampler,
        forward_model=forward_model,
        noise_std=NOISE_STD,
        jitter_std=JITTER_STD,
        rng=rng,
        forward_model_batch=forward_model_batch,
        constraint_fn=enforce_ordering,
    )

    # Animate and save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating animation → {OUTPUT_PATH}")
    animate_inference_multi_1d(
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
    print(f"  x2 = {est['mean'][1]:.3f} ± {est['std'][1]:.3f}  (true: {TRUE_X2})")
    print(f"  M1 = {est['mean'][2]:.3f} ± {est['std'][2]:.3f}  (true: {TRUE_M1})")
    print(f"  M2 = {est['mean'][3]:.3f} ± {est['std'][3]:.3f}  (true: {TRUE_M2})")
    print(f"  ESS = {est['ess']:.0f} / {N_PARTICLES}")
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
