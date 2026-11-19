"""Does the certified normalizer change any shipped result vs the fast path?"""

import numpy as np

import clocks.inference as inference
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
from clocks._weight_normalization import _normalize_log_weights_fast

calls = {"n": 0, "tiny": 0}
original = inference._normalize_log_weights


def counting(values):
    calls["n"] += 1
    w, z = original(values)
    if np.any((w > 0) & (w < 2.0**-1022)):
        calls["tiny"] += 1
    return w, z


def run():
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
            n_observations=80,
            seed=42,
        )
    )
    r = infer(
        sim.observations,
        InferenceConfig(
            clock_array=clock_array,
            noise=NoiseConfig(observation_std=0.005),
            prior=PriorConfig(position_range=(-8.0, 8.0), mass_range=(0.005, 0.15)),
            n_particles=2000,
            n_masses=(1, 2, 3),
            seed=42,
        ),
    )
    return r


inference._normalize_log_weights = counting
a = run()
print("certified calls:", calls)
inference._normalize_log_weights = _normalize_log_weights_fast
b = run()
same = all(
    np.array_equal(
        a.result_by_model[k].particle_state.particles,
        b.result_by_model[k].particle_state.particles,
    )
    and np.array_equal(
        a.result_by_model[k].particle_state.weights,
        b.result_by_model[k].particle_state.weights,
    )
    and a.log_evidence_by_model[k] == b.log_evidence_by_model[k]
    for k in (1, 2, 3)
)
print("bitwise identical particles, weights, evidence:", same)
print(a.posterior_by_model, b.posterior_by_model)
