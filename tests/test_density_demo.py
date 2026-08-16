"""Support contract for the raw Gaussian-density particle filter demo."""

import importlib.util
from pathlib import Path

import numpy as np

from clocks.types import ClockArray


def test_density_demo_builder_samples_only_its_physical_prior() -> None:
    spec = importlib.util.spec_from_file_location(
        "density_demo", Path(__file__).parents[1] / "scripts" / "demo_density.py"
    )
    assert spec is not None and spec.loader is not None
    density_demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(density_demo)
    builder = getattr(density_demo, "build_density_filter", None)
    assert callable(builder)
    clocks = ClockArray(
        np.array([[-6.0], [-3.0], [0.0], [3.0], [6.0]]), track_offset=1.0
    )
    particle_filter = builder(clocks, np.random.default_rng(9), n_particles=300)

    assert np.all(
        np.isfinite(particle_filter.log_prior_density(particle_filter.state.particles))
    )
    amplitudes = particle_filter.state.particles[:, 2]
    assert np.all(amplitudes >= 0.001)
    assert np.all(amplitudes <= 0.03)
