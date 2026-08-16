"""End-to-end multi-mass 2D gravitational time dilation demo.

~10 clocks placed randomly on a 2D plane, two hidden masses.
The particle filter deduces both positions and magnitudes (6 parameters:
x1, y1, x2, y2, M1, M2) from noisy clock readings.
An ordering constraint (x1 < x2) breaks label-switching symmetry.
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
from clocks._scenarios import (
    N_CLOCKS,
    N_OBSERVATIONS,
    N_PARTICLES,
    NOISE_STD,
    TRACK_OFFSET,
    TRUE_MASSES,
    TRUE_POSITIONS,
    generate_random_clocks,
)
from clocks.viz import animate_inference_multi_2d

# --- Configuration ---
SEED = 11
OUTPUT_PATH = Path("output/demo_multi_mass_2d.gif")


def main() -> None:
    rng = np.random.default_rng(SEED)

    mass_config = MassConfig(positions=TRUE_POSITIONS, masses=TRUE_MASSES)

    # Random clock placement
    clock_positions = generate_random_clocks(
        N_CLOCKS,
        rng,
        exclude=[tuple(p) for p in TRUE_POSITIONS],
    )
    clock_array = ClockArray(positions=clock_positions, track_offset=TRACK_OFFSET)

    simulation = simulate(
        SimulationConfig(
            clock_array=clock_array,
            ground_truth=mass_config,
            noise=NoiseConfig(observation_std=NOISE_STD),
            n_observations=N_OBSERVATIONS,
            seed=SEED,
        )
    )
    print(
        f"True masses: ({TRUE_POSITIONS[0][0]},{TRUE_POSITIONS[0][1]}) "
        f"M={TRUE_MASSES[0]}, "
        f"({TRUE_POSITIONS[1][0]},{TRUE_POSITIONS[1][1]}) M={TRUE_MASSES[1]}"
    )
    print(f"Clocks: {N_CLOCKS} randomly placed")
    print(f"True rates: {simulation.true_rates}")

    pf = build_particle_filter(
        InferenceConfig(
            clock_array=clock_array,
            noise=NoiseConfig(observation_std=NOISE_STD),
            prior=PriorConfig(position_range=(-8.0, 8.0), mass_range=(0.005, 0.15)),
            n_particles=N_PARTICLES,
            n_masses=2,
            seed=SEED,
        )
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating animation → {OUTPUT_PATH}")
    animate_inference_multi_2d(
        clock_array=clock_array,
        mass_config=mass_config,
        observations=simulation.observations,
        pf=pf,
        output_path=OUTPUT_PATH,
        fps=4,
    )

    est = pf.estimate()
    print(f"\nFinal estimate after {N_OBSERVATIONS} observations:")
    labels = ["x1", "y1", "x2", "y2", "M1", "M2"]
    truths = [
        TRUE_POSITIONS[0][0],
        TRUE_POSITIONS[0][1],
        TRUE_POSITIONS[1][0],
        TRUE_POSITIONS[1][1],
        TRUE_MASSES[0],
        TRUE_MASSES[1],
    ]
    for i, (label, truth) in enumerate(zip(labels, truths)):
        print(
            f"  {label} = {est['mean'][i]:.3f} ± {est['std'][i]:.3f}  (true: {truth})"
        )
    print(f"  ESS = {est['ess']:.0f} / {N_PARTICLES}")
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
