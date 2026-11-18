"""Reported ESS vs actual particle diversity for the model-comparison demo run."""

import numpy as np

from clocks import (
    ClockArray,
    InferenceConfig,
    MassConfig,
    NoiseConfig,
    PriorConfig,
    SimulationConfig,
    simulate,
)
from clocks.api import build_model_comparison

clock_array = ClockArray(
    positions=np.array([[-6.0], [-3.0], [0.0], [3.0], [6.0]]), track_offset=1.0
)
truth = MassConfig(positions=np.array([[-2.0], [3.0]]), masses=np.array([0.045, 0.030]))
sim = simulate(
    SimulationConfig(
        clock_array=clock_array,
        ground_truth=truth,
        noise=NoiseConfig(observation_std=0.005),
        n_observations=80,
        seed=42,
    )
)
cmp = build_model_comparison(
    InferenceConfig(
        clock_array=clock_array,
        noise=NoiseConfig(observation_std=0.005),
        prior=PriorConfig(position_range=(-8.0, 8.0), mass_range=(0.005, 0.15)),
        n_particles=2000,
        n_masses=(1, 2, 3),
        seed=42,
    )
)
for t, obs in enumerate(sim.observations, start=1):
    cmp.update(obs)
    if t in (1, 5, 20, 40, 80):
        for k, pf in cmp.filters.items():
            st = pf.state
            uniq = len(np.unique(st.particles, axis=0))
            d = pf.last_diagnostics
            ess = 1.0 / np.sum(st.weights**2)
            print(
                f"t={t:2d} K={k}: reported ESS={ess:7.1f} unique={uniq:5d} "
                f"stages={d.tempering_stages} acc={d.acceptance_rate if d.mh_proposals else float('nan'):.3f}"
            )
