"""End-to-end multi-mass 1D gravitational time dilation demo.

5 clocks along a track, two hidden masses. The particle filter
deduces both positions and magnitudes from noisy clock readings.
An ordering constraint (x1 < x2) breaks label-switching symmetry.
"""

from pathlib import Path

import numpy as np

from clocks.api import build_particle_filter, simulate
from clocks.config import InferenceConfig, NoiseConfig, PriorConfig, SimulationConfig
from clocks.types import ClockArray, MassConfig
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


def main() -> None:
    mass_config = MassConfig(
        positions=np.array([[TRUE_X1], [TRUE_X2]]),
        masses=np.array([TRUE_M1, TRUE_M2]),
    )
    clock_array = ClockArray(
        positions=np.array([[x] for x in CLOCK_POSITIONS]),
        track_offset=TRACK_OFFSET,
    )
    sim_config = SimulationConfig(
        clock_array=clock_array,
        ground_truth=mass_config,
        noise=NoiseConfig(observation_std=NOISE_STD),
        n_observations=N_OBSERVATIONS,
        seed=SEED,
    )
    infer_config = InferenceConfig(
        clock_array=clock_array,
        noise=NoiseConfig(observation_std=NOISE_STD),
        prior=PriorConfig(position_range=(-8.0, 8.0), mass_range=(0.1, 2.0)),
        n_particles=N_PARTICLES,
        n_masses=2,
        jitter_std=JITTER_STD,
        seed=SEED,
    )
    simulation = simulate(sim_config)
    pf = build_particle_filter(infer_config)

    print(f"True masses: x1={TRUE_X1}, x2={TRUE_X2}, M1={TRUE_M1}, M2={TRUE_M2}")
    print(f"True rates: {simulation.true_rates}")

    # Animate and save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating animation → {OUTPUT_PATH}")
    animate_inference_multi_1d(
        clock_array=clock_array,
        mass_config=mass_config,
        observations=simulation.observations,
        pf=pf,
        output_path=OUTPUT_PATH,
        fps=4,
    )
    est = pf.estimate()

    print(f"\nFinal estimate after {N_OBSERVATIONS} observations:")
    labels = ["x1", "x2", "M1", "M2"]
    truths = [TRUE_X1, TRUE_X2, TRUE_M1, TRUE_M2]
    for i, (label, truth) in enumerate(zip(labels, truths)):
        print(
            f"  {label} = {est['mean'][i]:.3f} ± {est['std'][i]:.3f}  (true: {truth})"
        )
    print(f"  ESS = {est['ess']:.0f} / {N_PARTICLES}")
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
