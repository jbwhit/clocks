"""Seed spread of the posterior over K for the site's model-comparison scenario.

Usage: uv run python kspread.py <n_obs> <n_particles> <seed0> <n_seeds>
"""

import sys
import time

import numpy as np

from clocks import (
    ClockArray,
    InferenceConfig,
    MassConfig,
    NoiseConfig,
    PriorConfig,
    SimulationConfig,
    infer,
    simulate,
)


def main():
    n_obs, n_particles, seed0, n_seeds = map(int, sys.argv[1:5])
    clock_array = ClockArray(
        positions=np.array([[-6.0], [-3.0], [0.0], [3.0], [6.0]]), track_offset=1.0
    )
    truth = MassConfig(
        positions=np.array([[-2.0], [3.0]]), masses=np.array([0.045, 0.030])
    )
    sim = simulate(
        SimulationConfig(
            clock_array=clock_array,
            ground_truth=truth,
            noise=NoiseConfig(observation_std=0.005),
            n_observations=n_obs,
            seed=42,
        )
    )
    for seed in range(seed0, seed0 + n_seeds):
        t0 = time.time()
        result = infer(
            sim.observations,
            InferenceConfig(
                clock_array=clock_array,
                noise=NoiseConfig(observation_std=0.005),
                prior=PriorConfig(position_range=(-8.0, 8.0), mass_range=(0.005, 0.15)),
                n_particles=n_particles,
                n_masses=(1, 2, 3),
                seed=seed,
            ),
        )
        lz = [result.log_evidence_by_model[k] for k in (1, 2, 3)]
        post = [result.posterior_by_model[k] for k in (1, 2, 3)]
        print(
            f"seed {seed}: logZ={np.round(lz, 3)} P(K)={np.round(post, 4)} "
            f"[{time.time() - t0:.0f}s]",
            flush=True,
        )


if __name__ == "__main__":
    main()
