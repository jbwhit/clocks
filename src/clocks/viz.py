"""Matplotlib plotting and animation (public facade).

Implementations live in ``clocks._panels`` and ``clocks._panels3d`` (static
primitives), ``clocks._figures`` (figures drawn from replays),
``clocks._echo_study`` (the range-study figure), ``clocks._animate``
(animation drivers) and ``clocks._clips`` (video clips drawn from replays).
"""

from clocks._animate import (
    animate_echolocation,
    animate_inference,
    animate_inference_2d,
    animate_inference_multi_1d,
    animate_inference_multi_2d,
    animate_model_comparison,
)
from clocks._clips import render_clip
from clocks._echo_study import range_study_figure
from clocks._figures import density_figure
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
from clocks._replay import load_replay
from clocks._style import THEMES, Palette, palette, show, style

__all__ = [
    "THEMES",
    "Palette",
    "animate_echolocation",
    "animate_inference",
    "animate_inference_2d",
    "animate_inference_multi_1d",
    "animate_inference_multi_2d",
    "animate_model_comparison",
    "create_echolocation_dashboard",
    "create_inference_dashboard",
    "density_figure",
    "load_replay",
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
    "palette",
    "plot_scene_3d",
    "range_study_figure",
    "render_clip",
    "show",
    "style",
]
