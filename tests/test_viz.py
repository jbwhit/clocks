"""Tests for visualization and animation helpers."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from clocks.inference import ParticleFilter
from clocks.noise import add_clock_noise
from clocks.physics import clock_rates
from clocks.types import ClockArray, MassConfig, Observation, ParticleState
from clocks.viz import (
    animate_inference,
    animate_inference_2d,
    animate_inference_multi_1d,
    create_inference_dashboard,
    create_inference_dashboard_2d,
    plot_clock_rates,
    plot_clock_rates_2d,
    plot_clock_setup,
    plot_clock_setup_2d,
    plot_mass_histogram,
    plot_particle_cloud,
    plot_particle_cloud_2d,
    plot_particle_cloud_multi_1d,
)

matplotlib.use("Agg")


# -- Shared fixtures --


@pytest.fixture()
def clock_array_1d() -> ClockArray:
    return ClockArray(
        positions=np.array([[-5.0], [0.0], [5.0]]),
        track_offset=1.0,
    )


@pytest.fixture()
def mass_config_1d() -> MassConfig:
    return MassConfig(
        positions=np.array([[2.0]]),
        masses=np.array([0.5]),
    )


@pytest.fixture()
def clock_array_2d() -> ClockArray:
    return ClockArray(
        positions=np.array([[-3.0, 0.0], [0.0, 3.0], [3.0, 0.0], [0.0, -3.0]]),
        track_offset=2.0,
    )


@pytest.fixture()
def mass_config_2d() -> MassConfig:
    return MassConfig(
        positions=np.array([[1.0, -0.5]]),
        masses=np.array([0.4]),
    )


@pytest.fixture()
def particle_state_1d() -> ParticleState:
    rng = np.random.default_rng(0)
    particles = np.column_stack([rng.uniform(-5, 5, 100), rng.uniform(0.1, 2, 100)])
    weights = np.ones(100) / 100
    return ParticleState(particles=particles, weights=weights, observations_seen=3)


@pytest.fixture()
def particle_state_2d() -> ParticleState:
    rng = np.random.default_rng(0)
    particles = np.column_stack(
        [
            rng.uniform(-5, 5, 100),
            rng.uniform(-5, 5, 100),
            rng.uniform(0.1, 2, 100),
        ]
    )
    weights = np.ones(100) / 100
    return ParticleState(particles=particles, weights=weights, observations_seen=5)


# -- 1D plot functions --


class TestPlotClockSetup:
    def test_runs_without_mass(self, clock_array_1d: ClockArray) -> None:
        fig, ax = plt.subplots()
        plot_clock_setup(ax, clock_array_1d)
        assert ax.get_title() == "Physical Setup"
        plt.close(fig)

    def test_runs_with_mass(
        self, clock_array_1d: ClockArray, mass_config_1d: MassConfig
    ) -> None:
        fig, ax = plt.subplots()
        plot_clock_setup(ax, clock_array_1d, mass_config_1d)
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "Clocks" in legend_texts
        assert "Mass (true)" in legend_texts
        plt.close(fig)


class TestPlotParticleCloud:
    def test_runs_without_true(self, particle_state_1d: ParticleState) -> None:
        fig, ax = plt.subplots()
        plot_particle_cloud(ax, particle_state_1d)
        assert "n_obs=3" in ax.get_title()
        plt.close(fig)

    def test_runs_with_true(self, particle_state_1d: ParticleState) -> None:
        fig, ax = plt.subplots()
        plot_particle_cloud(ax, particle_state_1d, true_params=np.array([2.0, 0.5]))
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "True" in legend_texts
        plt.close(fig)


class TestPlotClockRates:
    def test_runs(self, clock_array_1d: ClockArray, mass_config_1d: MassConfig) -> None:
        fig, ax = plt.subplots()
        rates = clock_rates(mass_config_1d, clock_array_1d)
        plot_clock_rates(ax, rates, clock_array_1d, label="Test", color="red")
        assert ax.get_title() == "Clock Rates"
        plt.close(fig)


# -- 2D plot functions --


class TestPlotClockSetup2d:
    def test_runs_without_mass(self, clock_array_2d: ClockArray) -> None:
        fig, ax = plt.subplots()
        plot_clock_setup_2d(ax, clock_array_2d)
        assert ax.get_title() == "Physical Setup"
        assert ax.get_aspect() == 1.0  # matplotlib normalizes "equal" to 1.0
        plt.close(fig)

    def test_runs_with_mass(
        self, clock_array_2d: ClockArray, mass_config_2d: MassConfig
    ) -> None:
        fig, ax = plt.subplots()
        plot_clock_setup_2d(ax, clock_array_2d, mass_config_2d)
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "Mass (true)" in legend_texts
        plt.close(fig)


class TestPlotParticleCloud2d:
    def test_runs_without_true(self, particle_state_2d: ParticleState) -> None:
        fig, ax = plt.subplots()
        plot_particle_cloud_2d(ax, particle_state_2d)
        assert "n_obs=5" in ax.get_title()
        plt.close(fig)

    def test_runs_with_true(self, particle_state_2d: ParticleState) -> None:
        fig, ax = plt.subplots()
        plot_particle_cloud_2d(ax, particle_state_2d, true_params=np.array([1.0, -0.5]))
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "True" in legend_texts
        plt.close(fig)


class TestPlotMassHistogram:
    def test_runs_without_true(self, particle_state_2d: ParticleState) -> None:
        fig, ax = plt.subplots()
        plot_mass_histogram(ax, particle_state_2d)
        assert ax.get_title() == "Mass Marginal"
        plt.close(fig)

    def test_runs_with_true(self, particle_state_2d: ParticleState) -> None:
        fig, ax = plt.subplots()
        plot_mass_histogram(ax, particle_state_2d, true_mass=0.4)
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "True M" in legend_texts
        plt.close(fig)


class TestPlotClockRates2d:
    def test_runs(self, clock_array_2d: ClockArray, mass_config_2d: MassConfig) -> None:
        fig, ax = plt.subplots()
        rates = clock_rates(mass_config_2d, clock_array_2d)
        plot_clock_rates_2d(ax, rates, clock_array_2d, label="Test Rates")
        assert ax.get_title() == "Test Rates"
        assert ax.get_aspect() == 1.0
        plt.close(fig)


# -- Dashboard creation --


class TestCreateDashboards:
    def test_1d_dashboard_keys(self) -> None:
        fig, axes = create_inference_dashboard()
        assert set(axes.keys()) == {"setup", "particles", "rates", "history"}
        plt.close(fig)

    def test_2d_dashboard_keys(self) -> None:
        fig, axes = create_inference_dashboard_2d()
        assert set(axes.keys()) == {"setup", "particles", "rates", "history"}
        plt.close(fig)


# -- Animation (end-to-end with small data) --


def _make_pf_and_obs_1d(
    clock_array: ClockArray, mass_config: MassConfig, n_obs: int = 3
) -> tuple[ParticleFilter, list[Observation]]:
    rng = np.random.default_rng(99)
    true_rates = clock_rates(mass_config, clock_array)
    observations = [
        Observation(rates=add_clock_noise(true_rates, 0.005, rng), time=float(t))
        for t in range(n_obs)
    ]

    def prior(rng: np.random.Generator, n: int) -> np.ndarray:
        return np.column_stack([rng.uniform(-5, 5, n), rng.uniform(0.1, 2, n)])

    def fwd(params: np.ndarray) -> np.ndarray:
        mc = MassConfig(positions=np.array([[params[0]]]), masses=np.array([params[1]]))
        return clock_rates(mc, clock_array)

    pf = ParticleFilter(
        n_particles=50,
        prior_sampler=prior,
        forward_model=fwd,
        noise_std=0.005,
        jitter_std=0.02,
        rng=rng,
    )
    return pf, observations


def _make_pf_and_obs_2d(
    clock_array: ClockArray, mass_config: MassConfig, n_obs: int = 3
) -> tuple[ParticleFilter, list[Observation]]:
    rng = np.random.default_rng(99)
    true_rates = clock_rates(mass_config, clock_array)
    observations = [
        Observation(rates=add_clock_noise(true_rates, 0.005, rng), time=float(t))
        for t in range(n_obs)
    ]

    def prior(rng: np.random.Generator, n: int) -> np.ndarray:
        return np.column_stack(
            [
                rng.uniform(-5, 5, n),
                rng.uniform(-5, 5, n),
                rng.uniform(0.1, 2, n),
            ]
        )

    def fwd(params: np.ndarray) -> np.ndarray:
        mc = MassConfig(
            positions=np.array([[params[0], params[1]]]),
            masses=np.array([params[2]]),
        )
        return clock_rates(mc, clock_array)

    pf = ParticleFilter(
        n_particles=50,
        prior_sampler=prior,
        forward_model=fwd,
        noise_std=0.005,
        jitter_std=0.02,
        rng=rng,
    )
    return pf, observations


class TestAnimateInference:
    def test_produces_gif(
        self,
        clock_array_1d: ClockArray,
        mass_config_1d: MassConfig,
        tmp_path: Path,
    ) -> None:
        pf, obs = _make_pf_and_obs_1d(clock_array_1d, mass_config_1d, n_obs=3)
        out = tmp_path / "test.gif"
        animate_inference(
            clock_array=clock_array_1d,
            mass_config=mass_config_1d,
            observations=obs,
            pf=pf,
            output_path=out,
        )
        assert out.exists()
        assert out.stat().st_size > 0


class TestAnimateInference2d:
    def test_produces_gif(
        self,
        clock_array_2d: ClockArray,
        mass_config_2d: MassConfig,
        tmp_path: Path,
    ) -> None:
        pf, obs = _make_pf_and_obs_2d(clock_array_2d, mass_config_2d, n_obs=3)
        out = tmp_path / "test_2d.gif"
        animate_inference_2d(
            clock_array=clock_array_2d,
            mass_config=mass_config_2d,
            observations=obs,
            pf=pf,
            output_path=out,
        )
        assert out.exists()
        assert out.stat().st_size > 0


# -- Multi-mass (2 masses in 1D) --


@pytest.fixture()
def clock_array_multi_1d() -> ClockArray:
    return ClockArray(
        positions=np.array([[-6.0], [-3.0], [0.0], [3.0], [6.0]]),
        track_offset=1.0,
    )


@pytest.fixture()
def mass_config_multi_1d() -> MassConfig:
    return MassConfig(
        positions=np.array([[-2.0], [3.0]]),
        masses=np.array([0.6, 0.4]),
    )


@pytest.fixture()
def particle_state_multi_1d() -> ParticleState:
    rng = np.random.default_rng(0)
    x = rng.uniform(-5, 5, (100, 2))
    x.sort(axis=1)
    particles = np.column_stack(
        [x[:, 0], x[:, 1], rng.uniform(0.1, 2, 100), rng.uniform(0.1, 2, 100)]
    )
    weights = np.ones(100) / 100
    return ParticleState(particles=particles, weights=weights, observations_seen=5)


class TestPlotParticleCloudMulti1d:
    def test_runs_without_true(self, particle_state_multi_1d: ParticleState) -> None:
        fig, ax = plt.subplots()
        plot_particle_cloud_multi_1d(ax, particle_state_multi_1d)
        assert "n_obs=5" in ax.get_title()
        plt.close(fig)

    def test_runs_with_true(self, particle_state_multi_1d: ParticleState) -> None:
        fig, ax = plt.subplots()
        plot_particle_cloud_multi_1d(
            ax, particle_state_multi_1d, true_params=np.array([-2.0, 3.0])
        )
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "True" in legend_texts
        plt.close(fig)


def _make_pf_and_obs_multi_1d(
    clock_array: ClockArray, mass_config: MassConfig, n_obs: int = 3
) -> tuple[ParticleFilter, list[Observation]]:
    rng = np.random.default_rng(99)
    true_rates = clock_rates(mass_config, clock_array)
    observations = [
        Observation(rates=add_clock_noise(true_rates, 0.005, rng), time=float(t))
        for t in range(n_obs)
    ]

    def prior(rng: np.random.Generator, n: int) -> np.ndarray:
        x = rng.uniform(-5, 5, (n, 2))
        x.sort(axis=1)
        return np.column_stack(
            [x[:, 0], x[:, 1], rng.uniform(0.1, 2, n), rng.uniform(0.1, 2, n)]
        )

    def fwd(params: np.ndarray) -> np.ndarray:
        mc = MassConfig(
            positions=np.array([[params[0]], [params[1]]]),
            masses=np.array([params[2], params[3]]),
        )
        return clock_rates(mc, clock_array)

    pf = ParticleFilter(
        n_particles=50,
        prior_sampler=prior,
        forward_model=fwd,
        noise_std=0.005,
        jitter_std=0.02,
        rng=rng,
    )
    return pf, observations


class TestAnimateInferenceMulti1d:
    def test_produces_gif(
        self,
        clock_array_multi_1d: ClockArray,
        mass_config_multi_1d: MassConfig,
        tmp_path: Path,
    ) -> None:
        pf, obs = _make_pf_and_obs_multi_1d(
            clock_array_multi_1d, mass_config_multi_1d, n_obs=3
        )
        out = tmp_path / "test_multi.gif"
        animate_inference_multi_1d(
            clock_array=clock_array_multi_1d,
            mass_config=mass_config_multi_1d,
            observations=obs,
            pf=pf,
            output_path=out,
        )
        assert out.exists()
        assert out.stat().st_size > 0
