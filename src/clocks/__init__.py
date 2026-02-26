"""Gravitational time dilation simulation and inference library."""

from clocks.inference import ParticleFilter
from clocks.noise import (
    add_clock_noise,
    log_likelihood_gaussian,
    log_likelihood_gaussian_batch,
)
from clocks.physics import (
    clock_rates,
    clock_rates_batch,
    compute_distances,
    gravitational_potential,
    time_dilation_factor,
)
from clocks.types import ClockArray, MassConfig, Observation, ParticleState
from clocks.viz import (
    animate_inference,
    animate_inference_2d,
    create_inference_dashboard,
    create_inference_dashboard_2d,
    plot_clock_rates,
    plot_clock_rates_2d,
    plot_clock_setup,
    plot_clock_setup_2d,
    plot_mass_histogram,
    plot_particle_cloud,
    plot_particle_cloud_2d,
)

__all__ = [
    "ClockArray",
    "MassConfig",
    "Observation",
    "ParticleFilter",
    "ParticleState",
    "add_clock_noise",
    "animate_inference",
    "animate_inference_2d",
    "clock_rates",
    "clock_rates_batch",
    "compute_distances",
    "create_inference_dashboard",
    "create_inference_dashboard_2d",
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
    "time_dilation_factor",
]
