"""Does x tighten faster than M on the 1D search page, and in what sense?"""

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

clock_array = ClockArray(positions=np.array([[-5.0], [0.0], [5.0]]), track_offset=1.0)
truth = MassConfig(positions=np.array([[2.5]]), masses=np.array([0.10]))
sim = simulate(
    SimulationConfig(
        clock_array=clock_array,
        ground_truth=truth,
        noise=NoiseConfig(observation_std=0.005),
        n_observations=25,
        seed=42,
    )
)
print("true rates", sim.true_rates, "dip", 1 - sim.true_rates)
result = infer(
    sim.observations,
    InferenceConfig(
        clock_array=clock_array,
        noise=NoiseConfig(observation_std=0.005),
        prior=PriorConfig(position_range=(-8.0, 8.0), mass_range=(0.005, 0.15)),
        n_particles=400,
        n_masses=1,
        ess_target=0.8,
        rejuvenation_steps=2,
        proposal_scale=2.38,
        seed=42,
    ),
)
for k, e in enumerate(result.history, 1):
    if k in (1, 2, 3, 5, 10, 15, 25):
        print(
            f"obs {k:2d}: x={e.mean[0]:.3f}±{e.std[0]:.3f} (rel {e.std[0] / 2.5:.3f}, "
            f"prior-range frac {e.std[0] / 16:.4f})  M={e.mean[1]:.4f}±{e.std[1]:.4f} "
            f"(rel {e.std[1] / 0.1:.3f}, prior-range frac {e.std[1] / 0.145:.4f})"
        )


# Fisher information per observation for (x, M)
def rates(x: float, m: float) -> np.ndarray:
    r = np.sqrt((clock_array.positions[:, 0] - x) ** 2 + 1.0)
    return np.sqrt(1 - 2 * m / r)


eps = 1e-6
J = np.column_stack(
    [
        (rates(2.5 + eps, 0.1) - rates(2.5 - eps, 0.1)) / (2 * eps),
        (rates(2.5, 0.1 + eps) - rates(2.5, 0.1 - eps)) / (2 * eps),
    ]
)
F = J.T @ J / 0.005**2
cov = np.linalg.inv(F)
print(
    "per-obs CRB std x, M:",
    np.sqrt(np.diag(cov)),
    "corr",
    cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]),
)
print("after 25 obs CRB std x, M:", np.sqrt(np.diag(cov) / 25))
