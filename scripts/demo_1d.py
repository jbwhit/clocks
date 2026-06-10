"""End-to-end 1D gravitational time dilation demo.

3 clocks along a track, one hidden mass. The particle filter
deduces position and magnitude from noisy clock readings.
"""

from pathlib import Path

import numpy as np

from clocks import (
    ClockArray,
    InferenceConfig,
    MassConfig,
    NoiseConfig,
    PriorConfig,
    SimulationConfig,
    build_particle_filter,
    simulate,
)
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
    mass_config = MassConfig(
        positions=np.array([[TRUE_X]]),
        masses=np.array([TRUE_M]),
    )
    clock_array = ClockArray(
        positions=np.array([[x] for x in CLOCK_POSITIONS]),
        track_offset=TRACK_OFFSET,
    )

    simulation = simulate(
        SimulationConfig(
            clock_array=clock_array,
            ground_truth=mass_config,
            noise=NoiseConfig(observation_std=NOISE_STD),
            n_observations=N_OBSERVATIONS,
            seed=SEED,
        )
    )
    print(f"True mass: x={TRUE_X}, M={TRUE_M}")
    print(f"True rates: {simulation.true_rates}")

    pf = build_particle_filter(
        InferenceConfig(
            clock_array=clock_array,
            noise=NoiseConfig(observation_std=NOISE_STD),
            prior=PriorConfig(position_range=(-8.0, 8.0), mass_range=(0.1, 2.0)),
            n_particles=N_PARTICLES,
            n_masses=1,
            jitter_std=JITTER_STD,
            seed=SEED,
        )
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating animation → {OUTPUT_PATH}")
    animate_inference(
        clock_array=clock_array,
        mass_config=mass_config,
        observations=simulation.observations,
        pf=pf,
        output_path=OUTPUT_PATH,
        fps=4,
    )

    est = pf.estimate()
    print(f"\nFinal estimate after {N_OBSERVATIONS} observations:")
    print(f"  x = {est['mean'][0]:.3f} ± {est['std'][0]:.3f}  (true: {TRUE_X})")
    print(f"  M = {est['mean'][1]:.3f} ± {est['std'][1]:.3f}  (true: {TRUE_M})")
    print(f"  ESS = {est['ess']:.0f} / {N_PARTICLES}")
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
