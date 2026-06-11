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
JITTER_STD = 0.05
SEED = 11
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


def main() -> None:
    rng = np.random.default_rng(SEED)

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
        f"True masses: ({TRUE_X1},{TRUE_Y1}) M={TRUE_M1}, "
        f"({TRUE_X2},{TRUE_Y2}) M={TRUE_M2}"
    )
    print(f"Clocks: {N_CLOCKS} randomly placed")
    print(f"True rates: {simulation.true_rates}")

    pf = build_particle_filter(
        InferenceConfig(
            clock_array=clock_array,
            noise=NoiseConfig(observation_std=NOISE_STD),
            prior=PriorConfig(position_range=(-8.0, 8.0), mass_range=(0.1, 2.0)),
            n_particles=N_PARTICLES,
            n_masses=2,
            jitter_std=JITTER_STD,
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
    truths = [TRUE_X1, TRUE_Y1, TRUE_X2, TRUE_Y2, TRUE_M1, TRUE_M2]
    for i, (label, truth) in enumerate(zip(labels, truths)):
        print(
            f"  {label} = {est['mean'][i]:.3f} ± {est['std'][i]:.3f}  (true: {truth})"
        )
    print(f"  ESS = {est['ess']:.0f} / {N_PARTICLES}")
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
