"""Tests for visualization and animation helpers."""

import inspect
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from clocks.api import build_model_comparison
from clocks.config import InferenceConfig, NoiseConfig, PriorConfig
from clocks.inference import ParticleFilter
from clocks.noise import add_clock_noise
from clocks.physics import clock_rates
from clocks.types import ClockArray, MassConfig, Observation, ParticleState
from clocks.viz import (
    animate_echolocation,
    animate_inference,
    animate_inference_2d,
    animate_inference_multi_1d,
    animate_inference_multi_2d,
    animate_model_comparison,
    create_echolocation_dashboard,
    create_inference_dashboard,
    plot_centered_rates,
    plot_clock_rates,
    plot_clock_rates_2d,
    plot_clock_setup,
    plot_clock_setup_2d,
    plot_mass_histogram,
    plot_particle_cloud,
    plot_particle_cloud_2d,
    plot_particle_cloud_multi_1d,
    plot_particle_cloud_multi_2d,
    plot_scene_3d,
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
        masses=np.array([0.10]),
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
        masses=np.array([0.10]),
    )


@pytest.fixture()
def particle_state_1d() -> ParticleState:
    rng = np.random.default_rng(0)
    particles = np.column_stack(
        [rng.uniform(-5, 5, 100), rng.uniform(0.005, 0.15, 100)]
    )
    weights = np.ones(100) / 100
    return ParticleState(particles=particles, weights=weights, observations_seen=3)


@pytest.fixture()
def particle_state_2d() -> ParticleState:
    rng = np.random.default_rng(0)
    particles = np.column_stack(
        [
            rng.uniform(-5, 5, 100),
            rng.uniform(-5, 5, 100),
            rng.uniform(0.005, 0.15, 100),
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

    def test_custom_figsize(self) -> None:
        fig, axes = create_inference_dashboard(figsize=(13, 10))
        assert set(axes.keys()) == {"setup", "particles", "rates", "history"}
        w, h = fig.get_size_inches()
        assert (w, h) == (13, 10)
        plt.close(fig)


# -- Animation (end-to-end with small data) --


def _make_pf_and_obs(
    clock_array: ClockArray,
    mass_config: MassConfig,
    prior_sampler: callable,
    forward_model: callable,
    n_obs: int = 3,
) -> tuple[ParticleFilter, list[Observation]]:
    """Build a ParticleFilter and observations for any scenario."""
    rng = np.random.default_rng(99)
    true_rates = clock_rates(mass_config, clock_array)
    observations = [
        Observation(rates=add_clock_noise(true_rates, 0.005, rng), time=float(t))
        for t in range(n_obs)
    ]
    pf = ParticleFilter(
        n_particles=50,
        prior_sampler=prior_sampler,
        forward_model=forward_model,
        noise_std=0.005,
        log_prior_density=lambda values: np.where(
            np.all(np.isfinite(values), axis=1)
            & np.all(values[:, -mass_config.masses.size :] > 0.0, axis=1),
            0.0,
            -np.inf,
        ),
        proposal_scale=1e-6,
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
        def prior(rng: np.random.Generator, n: int) -> np.ndarray:
            return np.column_stack([rng.uniform(-5, 5, n), rng.uniform(0.005, 0.04, n)])

        def fwd(params: np.ndarray) -> np.ndarray:
            mc = MassConfig(
                positions=np.array([[params[0]]]), masses=np.array([params[1]])
            )
            return clock_rates(mc, clock_array_1d)

        pf, obs = _make_pf_and_obs(clock_array_1d, mass_config_1d, prior, fwd, n_obs=3)
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
        def prior(rng: np.random.Generator, n: int) -> np.ndarray:
            return np.column_stack(
                [
                    rng.uniform(-5, 5, n),
                    rng.uniform(-5, 5, n),
                    rng.uniform(0.005, 0.04, n),
                ]
            )

        def fwd(params: np.ndarray) -> np.ndarray:
            mc = MassConfig(
                positions=np.array([[params[0], params[1]]]),
                masses=np.array([params[2]]),
            )
            return clock_rates(mc, clock_array_2d)

        pf, obs = _make_pf_and_obs(clock_array_2d, mass_config_2d, prior, fwd, n_obs=3)
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
        positions=np.array([[-2.0], [4.5]]),
        masses=np.array([0.045, 0.030]),
    )


@pytest.fixture()
def particle_state_multi_1d() -> ParticleState:
    rng = np.random.default_rng(0)
    x = rng.uniform(-5, 5, (100, 2))
    x.sort(axis=1)
    particles = np.column_stack(
        [
            x[:, 0],
            x[:, 1],
            rng.uniform(0.005, 0.02, 100),
            rng.uniform(0.005, 0.02, 100),
        ]
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
            ax, particle_state_multi_1d, true_params=np.array([-2.0, 4.5])
        )
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "True" in legend_texts
        plt.close(fig)


class TestAnimateInferenceMulti1d:
    def test_produces_gif(
        self,
        clock_array_multi_1d: ClockArray,
        mass_config_multi_1d: MassConfig,
        tmp_path: Path,
    ) -> None:
        def prior(rng: np.random.Generator, n: int) -> np.ndarray:
            x = rng.uniform(-5, 5, (n, 2))
            x.sort(axis=1)
            return np.column_stack(
                [
                    x[:, 0],
                    x[:, 1],
                    rng.uniform(0.005, 0.02, n),
                    rng.uniform(0.005, 0.02, n),
                ]
            )

        def fwd(params: np.ndarray) -> np.ndarray:
            mc = MassConfig(
                positions=np.array([[params[0]], [params[1]]]),
                masses=np.array([params[2], params[3]]),
            )
            return clock_rates(mc, clock_array_multi_1d)

        pf, obs = _make_pf_and_obs(
            clock_array_multi_1d, mass_config_multi_1d, prior, fwd, n_obs=3
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


# -- Multi-mass (2 masses in 2D) --


@pytest.fixture()
def clock_array_multi_2d() -> ClockArray:
    return ClockArray(
        positions=np.array(
            [
                [-4.0, 0.0],
                [-2.0, 3.0],
                [1.0, 4.0],
                [4.0, 2.0],
                [5.0, -1.0],
                [2.0, -4.0],
            ]
        ),
        track_offset=3.0,
    )


@pytest.fixture()
def mass_config_multi_2d() -> MassConfig:
    return MassConfig(
        positions=np.array([[-3.0, 2.0], [4.0, -1.0]]),
        masses=np.array([0.050, 0.030]),
    )


@pytest.fixture()
def particle_state_multi_2d() -> ParticleState:
    rng = np.random.default_rng(0)
    x1 = rng.uniform(-5, 5, 100)
    y1 = rng.uniform(-5, 5, 100)
    x2 = rng.uniform(-5, 5, 100)
    y2 = rng.uniform(-5, 5, 100)
    m1 = rng.uniform(0.005, 0.04, 100)
    m2 = rng.uniform(0.005, 0.04, 100)
    particles = np.column_stack([x1, y1, x2, y2, m1, m2])
    # enforce x1 < x2
    swap = particles[:, 0] > particles[:, 2]
    particles[swap, 0], particles[swap, 2] = (
        particles[swap, 2].copy(),
        particles[swap, 0].copy(),
    )
    particles[swap, 1], particles[swap, 3] = (
        particles[swap, 3].copy(),
        particles[swap, 1].copy(),
    )
    particles[swap, 4], particles[swap, 5] = (
        particles[swap, 5].copy(),
        particles[swap, 4].copy(),
    )
    weights = np.ones(100) / 100
    return ParticleState(particles=particles, weights=weights, observations_seen=5)


class TestPlotParticleCloudMulti2d:
    def test_runs_without_true(self, particle_state_multi_2d: ParticleState) -> None:
        fig, ax = plt.subplots()
        plot_particle_cloud_multi_2d(ax, particle_state_multi_2d)
        assert "n_obs=5" in ax.get_title()
        plt.close(fig)

    def test_runs_with_true(self, particle_state_multi_2d: ParticleState) -> None:
        fig, ax = plt.subplots()
        plot_particle_cloud_multi_2d(
            ax, particle_state_multi_2d, true_params=np.array([-3.0, 2.0])
        )
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "True" in legend_texts
        plt.close(fig)


class TestAnimateInferenceMulti2d:
    def test_produces_gif(
        self,
        clock_array_multi_2d: ClockArray,
        mass_config_multi_2d: MassConfig,
        tmp_path: Path,
    ) -> None:
        def prior(rng: np.random.Generator, n: int) -> np.ndarray:
            x1 = rng.uniform(-5, 5, n)
            y1 = rng.uniform(-5, 5, n)
            x2 = rng.uniform(-5, 5, n)
            y2 = rng.uniform(-5, 5, n)
            m1 = rng.uniform(0.005, 0.04, n)
            m2 = rng.uniform(0.005, 0.04, n)
            return np.column_stack([x1, y1, x2, y2, m1, m2])

        def fwd(params: np.ndarray) -> np.ndarray:
            mc = MassConfig(
                positions=np.array([[params[0], params[1]], [params[2], params[3]]]),
                masses=np.array([params[4], params[5]]),
            )
            return clock_rates(mc, clock_array_multi_2d)

        pf, obs = _make_pf_and_obs(
            clock_array_multi_2d, mass_config_multi_2d, prior, fwd, n_obs=3
        )
        out = tmp_path / "test_multi_2d.gif"
        animate_inference_multi_2d(
            clock_array=clock_array_multi_2d,
            mass_config=mass_config_multi_2d,
            observations=obs,
            pf=pf,
            output_path=out,
        )
        assert out.exists()
        assert out.stat().st_size > 0


# -- Model comparison animation --


class TestAnimateModelComparison:
    def test_produces_gif(
        self,
        clock_array_multi_1d: ClockArray,
        mass_config_multi_1d: MassConfig,
        tmp_path: Path,
    ) -> None:
        rng = np.random.default_rng(42)
        true_rates = clock_rates(mass_config_multi_1d, clock_array_multi_1d)
        observations = [
            Observation(rates=add_clock_noise(true_rates, 0.005, rng), time=float(t))
            for t in range(3)
        ]
        mc = build_model_comparison(
            InferenceConfig(
                clock_array=clock_array_multi_1d,
                noise=NoiseConfig(0.005),
                prior=PriorConfig((-8.0, 8.0), (0.005, 0.15)),
                n_particles=50,
                n_masses=(1, 2, 3),
                seed=42,
            )
        )
        out = tmp_path / "test_model_comparison.gif"
        animate_model_comparison(
            clock_array=clock_array_multi_1d,
            mass_config=mass_config_multi_1d,
            observations=observations,
            model_comparison=mc,
            output_path=out,
        )
        assert out.exists()
        assert out.stat().st_size > 0

    def test_produces_gif_with_sparse_k_values(
        self,
        clock_array_multi_1d: ClockArray,
        mass_config_multi_1d: MassConfig,
        tmp_path: Path,
    ) -> None:
        rng = np.random.default_rng(0)
        true_rates = clock_rates(mass_config_multi_1d, clock_array_multi_1d)
        observations = [
            Observation(rates=add_clock_noise(true_rates, 0.005, rng), time=float(t))
            for t in range(2)
        ]
        mc = build_model_comparison(
            InferenceConfig(
                clock_array=clock_array_multi_1d,
                noise=NoiseConfig(0.005),
                prior=PriorConfig((-8.0, 8.0), (0.005, 0.15)),
                n_particles=20,
                n_masses=(2, 3),
                seed=1,
            )
        )
        out = tmp_path / "test_model_comparison_sparse.gif"

        animate_model_comparison(
            clock_array=clock_array_multi_1d,
            mass_config=mass_config_multi_1d,
            observations=observations,
            model_comparison=mc,
            output_path=out,
        )

        assert out.exists()


class TestAnimationProcessesObservationsOnce:
    def test_dashboard_animation_observation_count(
        self, tmp_path: Path, clock_array_1d: ClockArray, mass_config_1d: MassConfig
    ) -> None:
        rng = np.random.default_rng(0)
        true_rates = clock_rates(mass_config_1d, clock_array_1d)
        observations = [
            Observation(
                rates=true_rates + rng.normal(0, 0.01, true_rates.shape),
                time=float(t),
            )
            for t in range(4)
        ]
        pf = ParticleFilter(
            n_particles=50,
            prior_sampler=lambda r, n: np.column_stack(
                [r.uniform(-8, 8, n), r.uniform(0.005, 0.04, n)]
            ),
            forward_model=lambda p: clock_rates(
                MassConfig(positions=p[:1].reshape(1, 1), masses=p[1:]),
                clock_array_1d,
            ),
            noise_std=0.01,
            log_prior_density=lambda values: np.where(
                (values[:, 0] >= -8.0)
                & (values[:, 0] <= 8.0)
                & (values[:, 1] >= 0.005)
                & (values[:, 1] <= 0.04),
                0.0,
                -np.inf,
            ),
            proposal_scale=1e-6,
        )
        animate_inference(
            clock_array_1d,
            mass_config_1d,
            observations,
            pf,
            tmp_path / "anim.gif",
        )
        assert pf.state.observations_seen == len(observations)

    def test_model_comparison_animation_observation_count(
        self, tmp_path: Path, clock_array_1d: ClockArray, mass_config_1d: MassConfig
    ) -> None:
        rng = np.random.default_rng(0)
        true_rates = clock_rates(mass_config_1d, clock_array_1d)
        observations = [
            Observation(
                rates=true_rates + rng.normal(0, 0.01, true_rates.shape),
                time=float(t),
            )
            for t in range(3)
        ]
        mc = build_model_comparison(
            InferenceConfig(
                clock_array=clock_array_1d,
                noise=NoiseConfig(0.01),
                prior=PriorConfig((-8.0, 8.0), (0.005, 0.15)),
                n_particles=50,
                n_masses=(1, 2),
            )
        )
        animate_model_comparison(
            clock_array_1d, mass_config_1d, observations, mc, tmp_path / "mc.gif"
        )
        for pf in mc.filters.values():
            assert pf.state.observations_seen == len(observations)


# -- Echolocation 3D panels --


@pytest.fixture()
def head_state() -> ParticleState:
    rng = np.random.default_rng(0)
    particles = np.column_stack(
        [
            rng.uniform(-5, 5, size=(200, 3)),
            rng.uniform(0.005, 0.15, size=(200, 1)),
        ]
    )
    return ParticleState(
        particles=particles,
        weights=np.ones(200) / 200,
        observations_seen=5,
    )


class TestEcholocationDashboard:
    def test_dashboard_has_expected_axes(self) -> None:
        fig, axes = create_echolocation_dashboard()
        assert set(axes) == {"scene", "history", "mass", "rates"}
        assert axes["scene"].name == "3d"
        plt.close(fig)

    def test_scene_renders_without_error(self, head_state: ParticleState) -> None:
        from clocks._scenarios import build_head_lattice, echo_mass_config

        fig, axes = create_echolocation_dashboard()
        plot_scene_3d(
            axes["scene"],
            build_head_lattice(),
            echo_mass_config(4.0),
            head_state,
            azim=30.0,
        )
        assert len(axes["scene"].collections) > 0
        plt.close(fig)

    def test_centered_rates_panel(self) -> None:
        fig, axes = create_echolocation_dashboard()
        rng = np.random.default_rng(1)
        predicted_centered = rng.normal(0, 0.005, 27)
        observed = predicted_centered + rng.normal(0, 0.005, 27)
        plot_centered_rates(axes["rates"], observed, predicted_centered)
        assert len(axes["rates"].patches) == 54  # two bar sets, 27 clocks each
        plt.close(fig)


class TestAnimateEcholocation:
    def test_creates_gif_and_processes_all_observations(self, tmp_path: Path) -> None:
        from clocks._scenarios import (
            build_echolocation_filter,
            build_head_lattice,
            echo_mass_config,
            make_echo_observations,
        )

        _, centered, contrasts = make_echo_observations(
            seed=0, range_r=2.0, n_observations=4
        )
        pf = build_echolocation_filter(seed=0, n_particles=300)
        out = tmp_path / "echo.gif"
        animate_echolocation(
            clock_array=build_head_lattice(),
            mass_config=echo_mass_config(2.0),
            observations=centered,
            filter_observations=contrasts,
            pf=pf,
            output_path=out,
            fps=2,
        )
        assert out.exists()
        assert pf.state.observations_seen == 4  # frame-0 fix invariant

    def test_requires_keyword_only_filter_observations(self) -> None:
        params = inspect.signature(animate_echolocation).parameters
        assert params["filter_observations"].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["filter_observations"].default is inspect.Parameter.empty

    @pytest.mark.parametrize(
        ("n_display", "n_filter"),
        [(0, 0), (2, 1)],
    )
    def test_rejects_empty_or_mismatched_observation_streams(
        self, n_display: int, n_filter: int, tmp_path: Path
    ) -> None:
        from clocks._scenarios import build_echolocation_filter, build_head_lattice

        observation = Observation(rates=np.zeros(27), time=0.0)
        filter_observation = Observation(rates=np.zeros(26), time=0.0)
        with pytest.raises(ValueError, match="same nonzero length"):
            animate_echolocation(
                clock_array=build_head_lattice(),
                mass_config=MassConfig(np.array([[2.0, 3.0, 6.0]]), np.array([0.08])),
                observations=[observation] * n_display,
                filter_observations=[filter_observation] * n_filter,
                pf=build_echolocation_filter(seed=0, n_particles=20),
                output_path=tmp_path / "unused.gif",
            )

    @pytest.mark.parametrize(
        ("display_channels", "filter_channels"),
        [(26, 26), (27, 27)],
    )
    def test_rejects_invalid_display_or_filter_channel_counts(
        self, display_channels: int, filter_channels: int, tmp_path: Path
    ) -> None:
        from clocks._scenarios import build_echolocation_filter, build_head_lattice

        with pytest.raises(ValueError, match="channel"):
            animate_echolocation(
                clock_array=build_head_lattice(),
                mass_config=MassConfig(np.array([[2.0, 3.0, 6.0]]), np.array([0.08])),
                observations=[Observation(np.zeros(display_channels), 0.0)],
                filter_observations=[Observation(np.zeros(filter_channels), 0.0)],
                pf=build_echolocation_filter(seed=0, n_particles=20),
                output_path=tmp_path / "unused.gif",
            )
