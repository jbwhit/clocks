"""Support contract for the raw Gaussian-density particle filter demo."""

import warnings

import numpy as np

from clocks._demos.demo_density import build_density_filter
from clocks._support import density_support_mask
from clocks.physics import clock_rates_density_gaussian
from clocks.types import ClockArray, Observation


def test_density_demo_builder_samples_only_its_physical_prior() -> None:
    clocks = ClockArray(
        np.array([[-6.0], [-3.0], [0.0], [3.0], [6.0]]), track_offset=1.0
    )
    particle_filter = build_density_filter(
        clocks, np.random.default_rng(9), n_particles=300
    )

    assert np.all(
        np.isfinite(particle_filter.log_prior_density(particle_filter.state.particles))
    )
    assert np.all(
        density_support_mask(
            particle_filter.state.particles,
            clock_array=clocks,
            mu_range=(-8.0, 8.0),
            sigma_range=(0.1, 5.0),
            amplitude_range=(0.001, 0.030),
        )
    )
    amplitudes = particle_filter.state.particles[:, 2]
    assert np.all(amplitudes >= 0.001)
    assert np.all(amplitudes <= 0.03)


def test_density_demo_never_forwards_an_invalid_candidate() -> None:
    clocks = ClockArray(
        np.array([[-6.0], [-3.0], [0.0], [3.0], [6.0]]), track_offset=1.0
    )
    particle_filter = build_density_filter(
        clocks, np.random.default_rng(9), n_particles=300
    )
    forward_model_batch = particle_filter.forward_model_batch
    assert forward_model_batch is not None

    def checked_forward(particles: np.ndarray) -> np.ndarray:
        valid = density_support_mask(
            particles,
            clock_array=clocks,
            mu_range=(-8.0, 8.0),
            sigma_range=(0.1, 5.0),
            amplitude_range=(0.001, 0.030),
        )
        assert np.all(valid)
        return forward_model_batch(particles)

    particle_filter.forward_model_batch = checked_forward
    rates = clock_rates_density_gaussian(np.array([1.5, 2.0, 0.010]), clocks)
    candidates = np.array([[1.5, 2.0, 0.010], [0.0, 1.0, 1.0]])
    log_likelihood = particle_filter._observation_log_likelihood(
        candidates, Observation(rates, time=0.0)
    )
    assert np.isfinite(log_likelihood[0])
    assert np.isneginf(log_likelihood[1])


def test_extreme_finite_density_candidate_rejects_without_warning() -> None:
    clocks = ClockArray(np.array([[0.0]]), track_offset=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        valid = density_support_mask(
            np.array([[0.0, 1.0, 1e308]]),
            clock_array=clocks,
            mu_range=(-1.0, 1.0),
            sigma_range=(0.1, 2.0),
            amplitude_range=(0.0, 1e308),
        )

    np.testing.assert_array_equal(valid, np.array([False]))
