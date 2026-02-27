"""Model comparison demo: infer the number of point masses.

Ground truth: K=2 masses. Runs parallel filters for K=1..3 and
prints per-K log-evidence, posterior probabilities, and MAP estimate.
"""

import numpy as np

from clocks.inference import ModelComparison
from clocks.noise import add_clock_noise
from clocks.physics import clock_rates
from clocks.types import ClockArray, MassConfig, Observation

# --- Configuration ---
TRUE_X1 = -2.0
TRUE_X2 = 3.0
TRUE_M1 = 0.6
TRUE_M2 = 0.4
CLOCK_POSITIONS = [-6.0, -3.0, 0.0, 3.0, 6.0]
TRACK_OFFSET = 1.0
N_OBSERVATIONS = 80
NOISE_STD = 0.005
N_PARTICLES = 2000
JITTER_STD = 0.02
K_MAX = 3
SEED = 42


def main() -> None:
    rng = np.random.default_rng(SEED)

    # Ground truth
    mass_config = MassConfig(
        positions=np.array([[TRUE_X1], [TRUE_X2]]),
        masses=np.array([TRUE_M1, TRUE_M2]),
    )
    clock_array = ClockArray(
        positions=np.array([[x] for x in CLOCK_POSITIONS]),
        track_offset=TRACK_OFFSET,
    )
    true_rates = clock_rates(mass_config, clock_array)

    print(
        f"True model: K=2 masses at x=[{TRUE_X1}, {TRUE_X2}], M=[{TRUE_M1}, {TRUE_M2}]"
    )
    print(f"True rates: {true_rates}")
    print()

    mc = ModelComparison(
        clock_array=clock_array,
        noise_std=NOISE_STD,
        n_dims=1,
        k_max=K_MAX,
        n_particles=N_PARTICLES,
        jitter_std=JITTER_STD,
        rng=rng,
    )

    # Feed observations
    for t in range(N_OBSERVATIONS):
        noisy = add_clock_noise(true_rates, NOISE_STD, rng)
        obs = Observation(rates=noisy, time=float(t))
        mc.update(obs)

        # Print progress every 20 steps
        if (t + 1) % 20 == 0:
            result = mc.evidence()
            print(f"After {t + 1} observations:")
            for k in sorted(result["log_evidence"]):
                print(
                    f"  K={k}: log-evidence={result['log_evidence'][k]:.1f}"
                    f"  posterior={result['posterior'][k]:.4f}"
                )
            print()

    # Final results
    result = mc.evidence()
    map_k = max(result["posterior"], key=lambda x: result["posterior"][x])
    est = mc.estimate()

    print(f"MAP model: K={map_k}")
    print(f"Estimate: {est['mean']}")
    print(f"Std:      {est['std']}")
    print(f"ESS:      {est['ess']:.0f} / {N_PARTICLES}")


if __name__ == "__main__":
    main()
