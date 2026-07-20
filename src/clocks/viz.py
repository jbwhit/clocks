"""Matplotlib plotting and animation (public facade).

Implementations live in ``clocks._panels`` (static primitives) and
``clocks._animate`` (animation drivers).
"""

from clocks._animate import (
    animate_echolocation,
    animate_inference,
    animate_inference_2d,
    animate_inference_multi_1d,
    animate_inference_multi_2d,
    animate_model_comparison,
)
from clocks._panels import (
    create_inference_dashboard,
    plot_clock_rates,
    plot_clock_rates_2d,
    plot_clock_setup,
    plot_clock_setup_2d,
    plot_mass_histogram,
    plot_particle_cloud,
    plot_particle_cloud_2d,
    plot_particle_cloud_multi_1d,
    plot_particle_cloud_multi_2d,
)
from clocks._panels3d import (
    create_echolocation_dashboard,
    plot_centered_rates,
    plot_scene_3d,
)

__all__ = [
    "animate_echolocation",
    "animate_inference",
    "animate_inference_2d",
    "animate_inference_multi_1d",
    "animate_inference_multi_2d",
    "animate_model_comparison",
    "create_echolocation_dashboard",
    "create_inference_dashboard",
    "plot_centered_rates",
    "plot_clock_rates",
    "plot_clock_rates_2d",
    "plot_clock_setup",
    "plot_clock_setup_2d",
    "plot_mass_histogram",
    "plot_particle_cloud",
    "plot_particle_cloud_2d",
    "plot_particle_cloud_multi_1d",
    "plot_particle_cloud_multi_2d",
    "plot_scene_3d",
]
