"""Model-comparison evidence: SMC (public API) vs brute-force prior Monte Carlo.

The brute-force prior is the *labeled* (unordered) uniform box conditioned on
weak-field validity. The likelihood is permutation symmetric, so the evidence
must equal that of the ordered prior used by the library if the factorial
normalization is handled correctly.

Usage: uv run python mc_kcomp.py <sigma> <n_obs> <n_particles> <n_seeds> <mc_draws>
"""

import sys

import numpy as np
from scipy.special import logsumexp

from clocks import (
    ClockArray,
    InferenceConfig,
    MassConfig,
    NoiseConfig,
    PriorConfig,
    SimulationConfig,
    simulate,
)
from clocks.api import infer

CLOCKS = np.array([-6.0, -3.0, 0.0, 3.0, 6.0])
D = 1.0
XR = (-8.0, 8.0)
MR = (0.005, 0.15)


def brute_log_evidence(k, obs, sigma, draws, rng):
    n = obs.shape[0]
    ybar = obs.mean(axis=0)
    within = np.sum((obs - ybar) ** 2)
    const = -n * len(CLOCKS) * np.log(sigma * np.sqrt(2 * np.pi))
    chunk = 2_000_000
    lls = []
    n_valid = 0
    total = 0
    while total < draws:
        x = rng.uniform(*XR, size=(chunk, k))
        m = rng.uniform(*MR, size=(chunk, k))
        r = np.sqrt((CLOCKS[None, None, :] - x[:, :, None]) ** 2 + D * D)
        phi = -np.sum(m[:, :, None] / r, axis=1)
        valid = np.all(np.abs(phi) <= 0.05, axis=1)
        f = np.sqrt(1.0 + 2.0 * phi[valid])
        q = within + n * np.sum((f - ybar) ** 2, axis=1)
        lls.append(const - 0.5 * q / sigma**2)
        n_valid += int(valid.sum())
        total += chunk
    ll = np.concatenate(lls)
    log_z = logsumexp(ll) - np.log(len(ll))
    w = np.exp(ll - ll.max())
    rel_se = w.std() / (w.mean() * np.sqrt(len(w)))
    ess = w.sum() ** 2 / np.sum(w**2)
    return log_z, rel_se, n_valid / total, ess


def main():
    sigma = float(sys.argv[1])
    n_obs = int(sys.argv[2])
    n_particles = int(sys.argv[3])
    n_seeds = int(sys.argv[4])
    draws = int(float(sys.argv[5]))
    clock_array = ClockArray(CLOCKS[:, None], track_offset=D)
    truth = MassConfig(np.array([[-2.0], [3.0]]), np.array([0.045, 0.030]))
    sim = simulate(
        SimulationConfig(
            clock_array=clock_array,
            ground_truth=truth,
            noise=NoiseConfig(sigma),
            n_observations=n_obs,
            seed=42,
        )
    )
    obs = np.array([o.rates for o in sim.observations])
    rng = np.random.default_rng(12345)
    brute = {}
    for k in (1, 2, 3):
        log_z, rel_se, pvalid, ess = brute_log_evidence(k, obs, sigma, draws, rng)
        brute[k] = log_z
        print(
            f"brute K={k}: logZ={log_z:.4f} (rel se {rel_se:.4f}) "
            f"pvalid={pvalid:.4f} is-ess={ess:.0f}"
        )
    lz = np.array([brute[k] for k in (1, 2, 3)])
    post = np.exp(lz - logsumexp(lz))
    print(f"brute posterior over K: {np.round(post, 4)}")
    smc = []
    for seed in range(n_seeds):
        result = infer(
            sim.observations,
            InferenceConfig(
                clock_array=clock_array,
                noise=NoiseConfig(sigma),
                prior=PriorConfig(XR, MR),
                n_particles=n_particles,
                n_masses=(1, 2, 3),
                seed=500 + seed,
            ),
        )
        row = [result.log_evidence_by_model[k] for k in (1, 2, 3)]
        smc.append(row)
        print(
            f"seed {seed}: logZ={np.round(row, 4)} "
            f"post={np.round([result.posterior_by_model[k] for k in (1, 2, 3)], 4)}"
        )
    smc = np.array(smc)
    se = smc.std(0, ddof=1) / np.sqrt(len(smc))
    print(f"SMC mean logZ={np.round(smc.mean(0), 4)} se={np.round(se, 4)}")
    print(f"diff (SMC-brute)={np.round(smc.mean(0) - lz, 4)}")


if __name__ == "__main__":
    main()
