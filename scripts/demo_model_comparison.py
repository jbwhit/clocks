"""Model comparison demo: infer the number of point masses.

Ground truth: K=2 masses. Runs parallel filters for K=1..3 and
prints per-K log-evidence, posterior probabilities, and MAP estimate.
Also generates an animated GIF showing posterior evolution.
"""

from pathlib import Path

import numpy as np

from clocks.api import build_model_comparison, infer, simulate
from clocks.config import InferenceConfig, NoiseConfig, PriorConfig, SimulationConfig
from clocks.types import ClockArray, MassConfig
from clocks.viz import animate_model_comparison

# --- Configuration ---
TRUE_X1 = -2.0
TRUE_X2 = 3.0
TRUE_M1 = 0.045
TRUE_M2 = 0.030
CLOCK_POSITIONS = [-6.0, -3.0, 0.0, 3.0, 6.0]
TRACK_OFFSET = 1.0
N_OBSERVATIONS = 80
NOISE_STD = 0.005
N_PARTICLES = 2000
SEED = 42


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
        prior=PriorConfig(position_range=(-8.0, 8.0), mass_range=(0.005, 0.15)),
        n_particles=N_PARTICLES,
        n_masses=(1, 2, 3),
        seed=SEED,
    )
    simulation = simulate(sim_config)
    result = infer(simulation.observations, infer_config)

    print(
        f"True model: K=2 masses at x=[{TRUE_X1}, {TRUE_X2}], M=[{TRUE_M1}, {TRUE_M2}]"
    )
    print(f"True rates: {simulation.true_rates}")
    print()

    for t, posterior in enumerate(result.history, start=1):
        if t % 20 == 0:
            print(f"After {t} observations:")
            for k in sorted(posterior):
                print(f"  K={k}: posterior={posterior[k]:.4f}")
            print()

    map_k = result.best_model
    estimate = result.result_by_model[map_k]

    print(f"MAP model: K={map_k}")
    print(f"Estimate: {estimate.posterior_mean}")
    print(f"Std:      {estimate.posterior_std}")
    print(f"ESS:      {estimate.ess:.0f} / {N_PARTICLES}")

    # --- Animated GIF ---
    print("\nGenerating model comparison GIF...")
    mc_gif = build_model_comparison(infer_config)
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    gif_path = output_dir / "demo_model_comparison.gif"
    animate_model_comparison(
        clock_array=clock_array,
        mass_config=mass_config,
        observations=simulation.observations,
        model_comparison=mc_gif,
        output_path=gif_path,
    )
    print(f"Saved: {gif_path}")


if __name__ == "__main__":
    main()
