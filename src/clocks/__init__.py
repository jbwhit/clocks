"""Gravitational time dilation simulation and inference library."""

from clocks.inference import Estimate, ParticleFilter
from clocks.noise import (
    add_clock_noise,
    log_likelihood_gaussian,
    log_likelihood_gaussian_batch,
)
from clocks.physics import (
    clock_rates,
    clock_rates_batch,
    clock_rates_batch_multi,
    compute_distances,
    gravitational_potential,
    time_dilation_factor,
)
from clocks.types import ClockArray, MassConfig, Observation, ParticleState
from clocks.viz import (
    animate_inference,
    animate_inference_2d,
    animate_inference_multi_1d,
    create_inference_dashboard,
    plot_clock_rates,
    plot_clock_rates_2d,
    plot_clock_setup,
    plot_clock_setup_2d,
    plot_mass_histogram,
    plot_particle_cloud,
    plot_particle_cloud_2d,
    plot_particle_cloud_multi_1d,
)

__all__ = [
    "ClockArray",
    "Estimate",
    "MassConfig",
    "Observation",
    "ParticleFilter",
    "ParticleState",
    "add_clock_noise",
    "animate_inference",
    "animate_inference_2d",
    "animate_inference_multi_1d",
    "clock_rates",
    "clock_rates_batch",
    "clock_rates_batch_multi",
    "compute_distances",
    "create_inference_dashboard",
    "gravitational_potential",
    "log_likelihood_gaussian",
    "log_likelihood_gaussian_batch",
    "plot_clock_rates",
    "plot_clock_rates_2d",
    "plot_clock_setup",
    "plot_clock_setup_2d",
    "plot_mass_histogram",
    "plot_particle_cloud",
    "plot_particle_cloud_2d",
    "plot_particle_cloud_multi_1d",
    "time_dilation_factor",
]
