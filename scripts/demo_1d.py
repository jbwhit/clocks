"""End-to-end 1D gravitational time dilation demo.

3 clocks along a track, one hidden mass. The particle filter
deduces position and magnitude from noisy clock readings.
"""

from pathlib import Path

import numpy as np

from clocks.inference import ParticleFilter
from clocks.noise import add_clock_noise
from clocks.physics import clock_rates, clock_rates_batch
from clocks.types import ClockArray, MassConfig, Observation
from clocks.viz import animate_inference

# --- Configuration ---
TRUE_X = 2.5
TRUE_M = 0.8
CLOCK_POSITIONS = [-5.0, 0.0, 5.0]
TRACK_OFFSET = 1.0
N_OBSERVATIONS = 50
NOISE_STD = 0.005
N_PARTICLES = 1000
JITTER_STD = 0.02
SEED = 42
OUTPUT_PATH = Path("output/demo_1d.gif")


def main() -> None:
    rng = np.random.default_rng(SEED)

    # Ground truth
    mass_config = MassConfig(
        positions=np.array([[TRUE_X]]),
        masses=np.array([TRUE_M]),
    )
    clock_array = ClockArray(
        positions=np.array([[x] for x in CLOCK_POSITIONS]),
        track_offset=TRACK_OFFSET,
    )
    true_rates = clock_rates(mass_config, clock_array)

    print(f"True mass: x={TRUE_X}, M={TRUE_M}")
    print(f"True rates: {true_rates}")

    # Generate observations
    observations: list[Observation] = []
    for t in range(N_OBSERVATIONS):
        noisy = add_clock_noise(true_rates, NOISE_STD, rng)
        observations.append(Observation(rates=noisy, time=float(t)))

    # Set up particle filter
    def prior_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        x = rng.uniform(-8, 8, n)
        m = rng.uniform(0.1, 2.0, n)
        return np.column_stack([x, m])

    def forward_model(params: np.ndarray) -> np.ndarray:
        mc = MassConfig(
            positions=np.array([[params[0]]]),
            masses=np.array([params[1]]),
        )
        return clock_rates(mc, clock_array)

    def forward_model_batch(particles: np.ndarray) -> np.ndarray:
        return clock_rates_batch(particles[:, :1], particles[:, 1], clock_array)

    pf = ParticleFilter(
        n_particles=N_PARTICLES,
        prior_sampler=prior_sampler,
        forward_model=forward_model,
        noise_std=NOISE_STD,
        jitter_std=JITTER_STD,
        rng=rng,
        forward_model_batch=forward_model_batch,
    )

    # Animate and save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating animation → {OUTPUT_PATH}")
    animate_inference(
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
    print(f"  x = {est['mean'][0]:.3f} ± {est['std'][0]:.3f}  (true: {TRUE_X})")
    print(f"  M = {est['mean'][1]:.3f} ± {est['std'][1]:.3f}  (true: {TRUE_M})")
    print(f"  ESS = {est['ess']:.0f} / {N_PARTICLES}")
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
