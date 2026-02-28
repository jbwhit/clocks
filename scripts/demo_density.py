"""Gaussian density forward model demo.

Ground truth: a Gaussian mass density (mu=1.5, sigma=2.0, amplitude=0.3).
Uses a standard ParticleFilter with the density forward model to infer
the parameters from noisy clock observations.
"""

import numpy as np

from clocks.inference import ParticleFilter
from clocks.noise import add_clock_noise
from clocks.physics import (
    clock_rates_density_gaussian,
    clock_rates_density_gaussian_batch,
)
from clocks.types import ClockArray, Observation

# --- Configuration ---
TRUE_MU = 1.5
TRUE_SIGMA = 2.0
TRUE_AMPLITUDE = 0.3
CLOCK_POSITIONS = [-6.0, -3.0, 0.0, 3.0, 6.0]
TRACK_OFFSET = 1.0
N_OBSERVATIONS = 60
NOISE_STD = 0.005
N_PARTICLES = 2000
JITTER_STD = 0.02
SEED = 42


def _fmt(label: str, est: dict, idx: int, true_val: float) -> str:
    m, s = est["mean"][idx], est["std"][idx]
    return f"  {label} = {m:.3f} ± {s:.3f}  (true: {true_val})"


def main() -> None:
    rng = np.random.default_rng(SEED)

    clock_array = ClockArray(
        positions=np.array([[x] for x in CLOCK_POSITIONS]),
        track_offset=TRACK_OFFSET,
    )

    # Ground truth rates
    true_params = np.array([TRUE_MU, TRUE_SIGMA, TRUE_AMPLITUDE])
    true_rates = clock_rates_density_gaussian(true_params, clock_array)

    print(f"True density: mu={TRUE_MU}, sigma={TRUE_SIGMA}, amplitude={TRUE_AMPLITUDE}")
    print(f"True rates: {true_rates}")
    print()

    # Prior: mu ~ U(-8, 8), sigma ~ U(0.1, 5.0), amplitude ~ U(0.01, 1.0)
    def prior_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        mu = rng.uniform(-8, 8, n)
        sigma = rng.uniform(0.1, 5.0, n)
        amplitude = rng.uniform(0.01, 1.0, n)
        return np.column_stack([mu, sigma, amplitude])

    def forward_model(params: np.ndarray) -> np.ndarray:
        return clock_rates_density_gaussian(params, clock_array)

    def forward_model_batch(particles: np.ndarray) -> np.ndarray:
        return clock_rates_density_gaussian_batch(particles, clock_array)

    def log_prior_fn(particles: np.ndarray) -> np.ndarray:
        lp = np.zeros(particles.shape[0])
        lp[particles[:, 1] < 0.1] = -np.inf  # sigma >= 0.1
        lp[particles[:, 2] < 0.01] = -np.inf  # amplitude >= 0.01
        return lp

    pf = ParticleFilter(
        n_particles=N_PARTICLES,
        prior_sampler=prior_sampler,
        forward_model=forward_model,
        noise_std=NOISE_STD,
        jitter_std=JITTER_STD,
        rng=rng,
        forward_model_batch=forward_model_batch,
        jitter="covariance",
        log_prior=log_prior_fn,
    )

    # Feed observations
    for t in range(N_OBSERVATIONS):
        noisy = add_clock_noise(true_rates, NOISE_STD, rng)
        obs = Observation(rates=noisy, time=float(t))
        pf.update(obs)

        if (t + 1) % 20 == 0:
            est = pf.estimate()
            print(f"After {t + 1} observations:")
            print(_fmt("mu   ", est, 0, TRUE_MU))
            print(_fmt("sigma", est, 1, TRUE_SIGMA))
            print(_fmt("A    ", est, 2, TRUE_AMPLITUDE))
            print(f"  ESS   = {est['ess']:.0f} / {N_PARTICLES}")
            print()

    # Final estimate
    est = pf.estimate()
    print("Final estimate:")
    print(_fmt("mu   ", est, 0, TRUE_MU))
    print(_fmt("sigma", est, 1, TRUE_SIGMA))
    print(_fmt("A    ", est, 2, TRUE_AMPLITUDE))
    print(f"  ESS   = {est['ess']:.0f} / {N_PARTICLES}")


if __name__ == "__main__":
    main()
