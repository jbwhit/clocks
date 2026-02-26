"""End-to-end 2D gravitational time dilation demo.

8 clocks in a ring, one hidden mass on a 2D plane. The particle filter
deduces (x, y, M) from noisy clock readings.
"""

from pathlib import Path

import numpy as np

from clocks.inference import ParticleFilter
from clocks.noise import add_clock_noise
from clocks.physics import clock_rates
from clocks.types import ClockArray, MassConfig, Observation
from clocks.viz import animate_inference_2d

# --- Configuration ---
TRUE_X = 1.5
TRUE_Y = -1.0
TRUE_M = 0.5
TRACK_OFFSET = 3.0  # clocks offset in z from mass plane (breaks M/r degeneracy)
N_OBSERVATIONS = 50
NOISE_STD = 0.005
N_PARTICLES = 2000
JITTER_STD = 0.02
SEED = 42
OUTPUT_PATH = Path("output/demo_2d.gif")


def main() -> None:
    rng = np.random.default_rng(SEED)

    # Ground truth
    mass_config = MassConfig(
        positions=np.array([[TRUE_X, TRUE_Y]]),
        masses=np.array([TRUE_M]),
    )

    # Clocks at varied distances (asymmetric to break radial degeneracy)
    clock_positions = np.array(
        [
            [-4.0, 0.0],
            [-2.0, 3.0],
            [1.0, 4.0],
            [4.0, 2.0],
            [5.0, -1.0],
            [2.0, -4.0],
            [-1.0, -3.0],
            [-3.0, -1.5],
        ]
    )
    clock_array = ClockArray(positions=clock_positions, track_offset=TRACK_OFFSET)

    true_rates = clock_rates(mass_config, clock_array)
    n_clocks = len(clock_positions)
    print(f"True mass: x={TRUE_X}, y={TRUE_Y}, M={TRUE_M}")
    print(f"Clocks: {n_clocks} at varied positions")
    print(f"True rates: {true_rates}")

    # Generate observations
    observations: list[Observation] = []
    for t in range(N_OBSERVATIONS):
        noisy = add_clock_noise(true_rates, NOISE_STD, rng)
        observations.append(Observation(rates=noisy, time=float(t)))

    # Set up particle filter — 3 params: x, y, M
    def prior_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        x = rng.uniform(-8, 8, n)
        y = rng.uniform(-8, 8, n)
        m = rng.uniform(0.1, 2.0, n)
        return np.column_stack([x, y, m])

    def forward_model(params: np.ndarray) -> np.ndarray:
        mc = MassConfig(
            positions=np.array([[params[0], params[1]]]),
            masses=np.array([params[2]]),
        )
        return clock_rates(mc, clock_array)

    pf = ParticleFilter(
        n_particles=N_PARTICLES,
        prior_sampler=prior_sampler,
        forward_model=forward_model,
        noise_std=NOISE_STD,
        jitter_std=JITTER_STD,
        rng=rng,
    )

    # Animate and save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating animation → {OUTPUT_PATH}")
    animate_inference_2d(
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
    print(f"  y = {est['mean'][1]:.3f} ± {est['std'][1]:.3f}  (true: {TRUE_Y})")
    print(f"  M = {est['mean'][2]:.3f} ± {est['std'][2]:.3f}  (true: {TRUE_M})")
    print(f"  ESS = {est['ess']:.0f} / {N_PARTICLES}")
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
