"""Gaussian density forward model demo.

Ground truth: a Gaussian mass density (mu=1.5, sigma=2.0, amplitude=0.01).
Uses a standard ParticleFilter with the density forward model to infer
the parameters from noisy clock observations.
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from clocks._support import density_support_mask, sample_conditioned_prior  # noqa: E402
from clocks.inference import ParticleFilter  # noqa: E402
from clocks.noise import add_clock_noise  # noqa: E402
from clocks.physics import (  # noqa: E402
    clock_rates_density_gaussian,
    clock_rates_density_gaussian_batch,
)
from clocks.types import ClockArray, Observation  # noqa: E402

# --- Configuration ---
TRUE_MU = 1.5
TRUE_SIGMA = 2.0
TRUE_AMPLITUDE = 0.010
CLOCK_POSITIONS = [-6.0, -3.0, 0.0, 3.0, 6.0]
TRACK_OFFSET = 1.0
N_OBSERVATIONS = 60
NOISE_STD = 0.005
N_PARTICLES = 2000
SEED = 42
OUTPUT_PATH = Path("output/demo_density.png")
MU_RANGE = (-8.0, 8.0)
SIGMA_RANGE = (0.1, 5.0)
AMPLITUDE_RANGE = (0.001, 0.030)


def _fmt(label: str, est: dict, idx: int, true_val: float) -> str:
    m, s = est["mean"][idx], est["std"][idx]
    return f"  {label} = {m:.3f} ± {s:.3f}  (true: {true_val})"


def build_density_filter(
    clock_array: ClockArray,
    rng: np.random.Generator,
    *,
    n_particles: int = N_PARTICLES,
) -> ParticleFilter:
    """Build the raw density filter on its true conditional prior."""

    def draw(draw_rng: np.random.Generator, n: int) -> np.ndarray:
        return np.column_stack(
            [
                draw_rng.uniform(*MU_RANGE, n),
                draw_rng.uniform(*SIGMA_RANGE, n),
                draw_rng.uniform(*AMPLITUDE_RANGE, n),
            ]
        )

    def valid(particles: np.ndarray) -> np.ndarray:
        return density_support_mask(
            particles,
            clock_array=clock_array,
            mu_range=MU_RANGE,
            sigma_range=SIGMA_RANGE,
            amplitude_range=AMPLITUDE_RANGE,
        )

    def prior_sampler(draw_rng: np.random.Generator, n: int) -> np.ndarray:
        return sample_conditioned_prior(
            draw_rng,
            n,
            draw,
            valid,
            description="Gaussian density",
        )

    def forward_model(params: np.ndarray) -> np.ndarray:
        return clock_rates_density_gaussian(params, clock_array)

    def forward_model_batch(particles: np.ndarray) -> np.ndarray:
        return clock_rates_density_gaussian_batch(particles, clock_array)

    def log_prior_fn(particles: np.ndarray) -> np.ndarray:
        lp = np.full(particles.shape[0], -np.inf)
        lp[valid(particles)] = 0.0
        return lp

    return ParticleFilter(
        n_particles=n_particles,
        prior_sampler=prior_sampler,
        forward_model=forward_model,
        noise_std=NOISE_STD,
        rng=rng,
        forward_model_batch=forward_model_batch,
        log_prior_density=log_prior_fn,
    )


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

    pf = build_density_filter(clock_array, rng)

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

    # --- Static summary figure ---
    fig, (ax_density, ax_rates, ax_conv) = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: true vs inferred density profile
    xs = np.linspace(-8, 8, 400)
    true_density = TRUE_AMPLITUDE * np.exp(-0.5 * ((xs - TRUE_MU) / TRUE_SIGMA) ** 2)
    mu_hat, sigma_hat, amp_hat = est["mean"]
    est_density = amp_hat * np.exp(-0.5 * ((xs - mu_hat) / sigma_hat) ** 2)
    ax_density.plot(xs, true_density, color="lightcoral", label="True")
    ax_density.plot(xs, est_density, color="steelblue", ls="--", label="Inferred")
    ax_density.set_xlabel("x")
    ax_density.set_ylabel("mass density")
    ax_density.set_title("Density profile")
    ax_density.legend()

    # Panel 2: true vs final predicted clock rates
    predicted = clock_rates_density_gaussian(est["mean"], clock_array)
    positions = clock_array.positions[:, 0]
    ax_rates.plot(positions, true_rates, "o-", color="lightcoral", label="True")
    ax_rates.plot(positions, predicted, "s--", color="steelblue", label="Inferred")
    ax_rates.set_xlabel("clock position")
    ax_rates.set_ylabel("tick rate")
    ax_rates.set_title("Clock rates")
    ax_rates.legend()

    # Panel 3: convergence of the three parameters
    history = pf.history[1:]
    means = np.array(
        [np.average(s.particles, weights=s.weights, axis=0) for s in history]
    )
    stds = np.array(
        [
            np.sqrt(
                np.average(
                    (s.particles - np.average(s.particles, weights=s.weights, axis=0))
                    ** 2,
                    weights=s.weights,
                    axis=0,
                )
            )
            for s in history
        ]
    )
    steps = np.arange(1, len(history) + 1)
    for j, (label, truth, color) in enumerate(
        [
            ("mu", TRUE_MU, "tab:blue"),
            ("sigma", TRUE_SIGMA, "tab:green"),
            ("A", TRUE_AMPLITUDE, "tab:orange"),
        ]
    ):
        ax_conv.plot(steps, means[:, j], color=color, label=f"{label} est")
        ax_conv.fill_between(
            steps,
            means[:, j] - stds[:, j],
            means[:, j] + stds[:, j],
            alpha=0.15,
            color=color,
        )
        ax_conv.axhline(truth, color=color, ls="--", alpha=0.5)
    ax_conv.set_xlabel("Observation #")
    ax_conv.set_title("Convergence")
    ax_conv.legend(fontsize=8)

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
