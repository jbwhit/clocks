"""Fast tests for the shared multi-mass-2D scenario module."""

import importlib.util
import inspect
import math
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

import clocks._scenarios as scenarios
from clocks._calibration import load_study
from clocks._scenarios import (
    ECHO_DIRECTION,
    ECHO_M_TRUE,
    ECHO_MIN_RANGE_R,
    ECHO_R_HEAD,
    MIN_SEPARATION,
    PASS_TOLERANCE,
    TRUTH,
    EchoRunResult,
    build_echolocation_filter,
    build_head_lattice,
    contrast_matrix,
    echo_mass_config,
    echo_mass_position,
    generate_random_clocks,
    make_echo_observations,
    passes,
    run_echolocation_3d,
    run_multi_mass_2d,
    validate_echo_geometry,
)
from clocks.inference import ParticleFilter
from clocks.physics import _point_mass_potential_batch, clock_rates
from clocks.types import MassConfig, Observation, ParticleState

# Fields that survive a change of platform exactly: integers, booleans, and the
# controls echoed back from the call. Everything else is a float64 that reaches
# the caller through BLAS and libm, which differ by a few ulps between macOS
# arm64 and Linux x86-64.
_EXACT_ECHO_RUN_FIELDS = (
    "seed",
    "range_r",
    "passed",
    "covered_3sigma",
    "forward_model_evaluations",
    "ess_target",
    "rejuvenation_steps",
    "proposal_scale",
)
_FLOAT_ECHO_RUN_FIELDS = (
    "position_error",
    "mass_error",
    "pos_std",
    "mass_std",
    "residual_over_noise",
    "normalized_error",
)


def _assert_frozen_echo_run(actual: EchoRunResult, expected: dict[str, object]) -> None:
    """Compare a frozen replay: exact where exactness is meaningful.

    Bitwise equality would pin the platform, not the behaviour -- the posterior
    mean moves by ~4e-15 between macOS and Linux. A 1e-12 relative tolerance is
    still hundreds of times tighter than any real change to the filter, while
    the counts and flags stay exact because those are what a dropped
    observation or a changed control would move.
    """
    assert set(actual) == set(expected)
    for field in _EXACT_ECHO_RUN_FIELDS:
        assert actual[field] == expected[field], field
    for field in ("mean", "std"):
        np.testing.assert_allclose(
            actual[field], expected[field], rtol=1e-12, atol=0.0, err_msg=field
        )
    for field in _FLOAT_ECHO_RUN_FIELDS:
        assert actual[field] == pytest.approx(expected[field], rel=1e-12), field
    assert set(_EXACT_ECHO_RUN_FIELDS) | set(_FLOAT_ECHO_RUN_FIELDS) | {
        "mean",
        "std",
    } == set(expected)


def _load_scan(script_name: str) -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / f"{script_name}.py"
    spec = importlib.util.spec_from_file_location(script_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _TerminalStateEchoFilter:
    """Minimal filter double retaining a caller-supplied terminal state."""

    ess_target = 0.9
    rejuvenation_steps = 1
    proposal_scale = 1.5

    def __init__(
        self,
        state: ParticleState,
        forward_model_batch: object,
    ) -> None:
        self.state = state
        self.forward_model_batch = forward_model_batch

    def update(self, observation: object) -> ParticleState:
        assert callable(self.forward_model_batch)
        self.forward_model_batch(self.state.particles)
        return self.state

    def estimate(self) -> dict[str, object]:
        mean = np.average(self.state.particles, weights=self.state.weights, axis=0)
        variance = np.average(
            (self.state.particles - mean) ** 2,
            weights=self.state.weights,
            axis=0,
        )
        return {"mean": mean, "std": np.sqrt(variance), "ess": 1.0}


def test_scan_raw_study_paths_are_seed_block_specific() -> None:
    multi = _load_scan("scan_multi_mass_2d")
    echo = _load_scan("scan_echolocation_range")
    assert multi._study_json_path(0) == Path(
        "output/multi_mass_2d_study_seed_block_0.json"
    )
    assert multi._study_json_path(500) == Path(
        "output/multi_mass_2d_study_seed_block_500.json"
    )
    assert echo._study_json_path(0) == Path(
        "output/echolocation_range_study_seed_block_0.json"
    )
    assert echo._study_json_path(500) == Path(
        "output/echolocation_range_study_seed_block_500.json"
    )


def test_multi_scan_serializes_raw_results_with_gate_metadata(tmp_path: Path) -> None:
    scan = _load_scan("scan_multi_mass_2d")
    results = [
        {
            "seed": seed,
            "passed": True,
            "mean": TRUTH.copy(),
            "std": np.ones(6),
            "max_abs_error": 0.0,
            "covered_3sigma": True,
            "max_posterior_std": 1.0,
            "residual_over_noise": 1.0,
            "normalized_error": 0.0,
            "forward_model_evaluations": 42,
            "ess_target": 0.7,
            "rejuvenation_steps": 2,
            "proposal_scale": 3.0,
        }
        for seed in (0, 1)
    ]
    path = tmp_path / "multi.json"
    scan._write_study(
        path,
        seed_block=0,
        seeds=(0, 1),
        cells=[(0.7, 2, 3.0)],
        results=results,
    )
    raw = load_study(path)
    assert raw["schema_version"] == 1
    assert raw["study"] == "multi_mass_2d"
    assert raw["seed_block"] == 0
    assert raw["seeds"] == [0, 1]
    assert raw["control_grid"] == {
        "ess_target": [0.7],
        "proposal_scale": [3.0],
        "rejuvenation_steps": [2],
    }
    assert raw["tolerances"] == {
        "absolute_parameter_error": [2.5, 2.5, 2.5, 2.5, 0.012, 0.012]
    }
    assert len(raw["results"]) == 2
    assert raw["results"][0]["mean"] == TRUTH.tolist()


def test_multi_scan_validation_preserves_existing_evidence(tmp_path: Path) -> None:
    scan = _load_scan("scan_multi_mass_2d")
    result = {
        "seed": 0,
        "passed": False,
        "mean": TRUTH.copy(),
        "std": np.ones(6),
        "max_abs_error": 0.0,
        "covered_3sigma": True,
        "max_posterior_std": 1.0,
        "residual_over_noise": 1.0,
        "normalized_error": 0.0,
        "forward_model_evaluations": 42,
        "ess_target": 0.7,
        "rejuvenation_steps": 2,
        "proposal_scale": 3.0,
    }
    path = tmp_path / "multi.json"
    path.write_bytes(b"existing evidence\n")

    with pytest.raises(ValueError, match="passed.*derived"):
        scan._write_study(
            path,
            seed_block=0,
            seeds=(0,),
            cells=[(0.7, 2, 3.0)],
            results=[result],
        )

    assert path.read_bytes() == b"existing evidence\n"


@pytest.mark.parametrize(
    "script_name",
    ["scan_multi_mass_2d", "scan_echolocation_range"],
)
def test_protected_seed_blocks_forbid_explicit_control_overrides(
    script_name: str,
) -> None:
    scan = _load_scan(script_name)
    assert scan._seeds_for_block(500) == tuple(range(500, 512))
    expected = (
        [(0.7, 2, 3.0)] if script_name == "scan_multi_mass_2d" else [(0.9, 1, 1.5)]
    )
    assert scan._control_cells(500, None, None, None) == expected
    for explicit_controls in (
        ([0.8], None, None),
        (None, [2], None),
        (None, None, [2.38]),
    ):
        with pytest.raises(ValueError, match="explicit control overrides"):
            scan._control_cells(500, *explicit_controls)


@pytest.mark.parametrize(
    "script_name",
    ["scan_multi_mass_2d", "scan_echolocation_range"],
)
def test_development_seed_block_uses_declared_grid_and_allows_overrides(
    script_name: str,
) -> None:
    scan = _load_scan(script_name)
    assert scan._seeds_for_block(0) == tuple(range(12))
    development = scan._control_cells(0, None, None, None)
    assert len(development) == 27
    assert {cell[0] for cell in development} == {0.7, 0.8, 0.9}
    assert {cell[1] for cell in development} == {1, 2, 4}
    assert {cell[2] for cell in development} == {1.5, 2.38, 3.0}
    assert scan._control_cells(0, [0.7], [1], [1.5]) == [(0.7, 1, 1.5)]
    for invalid_block in (-100, 200, 300, 501):
        with pytest.raises(ValueError, match="seed block"):
            scan._seeds_for_block(invalid_block)
    for invalid_controls in (
        ([0.0], None, None),
        (None, [0], None),
        (None, None, [np.inf]),
    ):
        with pytest.raises(ValueError, match="control"):
            scan._control_cells(0, *invalid_controls)
    for duplicate_controls in (
        ([0.8, 0.80], None, None),
        (None, [2, 2], None),
        (None, None, [2.38, 2.380]),
    ):
        with pytest.raises(ValueError, match="duplicate"):
            scan._control_cells(0, *duplicate_controls)


def test_echolocation_scan_protects_ranges_and_rejects_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scan = _load_scan("scan_echolocation_range")
    assert hasattr(scan, "_ranges_for_block")
    canonical = scan._ranges_for_block(500, None)
    assert canonical == list(scan.ECHO_SWEEP_RANGES)
    with pytest.raises(ValueError, match="explicit range overrides"):
        scan._ranges_for_block(500, list(scan.ECHO_SWEEP_RANGES))
    with pytest.raises(ValueError, match="duplicate"):
        scan._ranges_for_block(0, [2.0, 2])
    assert scan._ranges_for_block(0, [2.0, 3.0]) == [2.0, 3.0]

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scan_echolocation_range.py",
            "--seed-block",
            "500",
            "--ranges",
            "2.0",
            "--figure-only",
        ],
    )
    monkeypatch.setattr(scan, "load_study", lambda _path: {"results": []})
    monkeypatch.setattr(scan, "write_summary_figure", lambda *_args: None)
    with pytest.raises(SystemExit):
        scan.main()


class TestPassRule:
    def test_truth_passes(self) -> None:
        assert passes(TRUTH)

    def test_position_error_at_tolerance_passes(self) -> None:
        assert passes(TRUTH + np.array([2.5, 0, 0, 0, 0, 0]))

    def test_position_error_beyond_tolerance_fails(self) -> None:
        assert not passes(TRUTH + np.array([2.5001, 0, 0, 0, 0, 0]))

    def test_mass_error_beyond_tolerance_fails(self) -> None:
        assert not passes(TRUTH + np.array([0, 0, 0, 0, 0.0121, 0]))

    def test_tolerance_values(self) -> None:
        assert np.array_equal(
            PASS_TOLERANCE,
            np.array([2.5, 2.5, 2.5, 2.5, 0.012, 0.012]),
        )

    def test_tolerance_uses_one_rule_per_parameter_kind(self) -> None:
        assert len(set(PASS_TOLERANCE[:4])) == 1
        assert len(set(PASS_TOLERANCE[4:])) == 1


class TestClockPlacement:
    def test_respects_min_separation_and_exclusions(self) -> None:
        rng = np.random.default_rng(11)
        exclude = [(-3.0, 2.0), (4.0, -1.0)]
        clocks = generate_random_clocks(10, rng, exclude=exclude)
        assert clocks.shape == (10, 2)
        for i in range(10):
            for j in range(i + 1, 10):
                assert np.linalg.norm(clocks[i] - clocks[j]) >= MIN_SEPARATION
            for p in exclude:
                assert np.linalg.norm(clocks[i] - np.array(p)) >= MIN_SEPARATION


class TestPosteriorUncertaintyRegression:
    def test_seed_101_retains_uncertainty(self) -> None:
        """A formerly degenerate seed must retain posterior uncertainty."""
        result = run_multi_mass_2d(101)
        assert result["max_posterior_std"] > 1e-6
        assert result["normalized_error"] >= 0.0
        assert result["forward_model_evaluations"] > 0


class TestEchoGeometry:
    def test_lattice_is_3x3x3_cube(self) -> None:
        head = build_head_lattice()
        assert head.positions.shape == (27, 3)
        assert head.track_offset == 0.0
        # Every coordinate is exactly -1, 0, or 1; all 27 cells distinct.
        assert set(np.unique(head.positions)) == {-1.0, 0.0, 1.0}
        assert len({tuple(p) for p in head.positions}) == 27

    def test_circumradius_is_sqrt_3(self) -> None:
        head = build_head_lattice()
        radii = np.linalg.norm(head.positions, axis=1)
        assert np.isclose(radii.max(), ECHO_R_HEAD)
        assert np.isclose(ECHO_R_HEAD, np.sqrt(3.0))

    def test_direction_is_unit_and_off_axis(self) -> None:
        assert np.isclose(np.linalg.norm(ECHO_DIRECTION), 1.0)
        # No zero component (off-axis) and components distinct (off-diagonal).
        assert np.all(np.abs(ECHO_DIRECTION) > 0.1)
        assert len(set(np.round(np.abs(ECHO_DIRECTION), 6))) == 3

    def test_mass_position_at_requested_range(self) -> None:
        pos = echo_mass_position(3.0)
        assert pos.shape == (3,)
        assert np.isclose(np.linalg.norm(pos), 3.0 * ECHO_R_HEAD)
        config = echo_mass_config(3.0)
        assert config.positions.shape == (1, 3)
        assert np.isclose(config.masses[0], ECHO_M_TRUE)

    def test_validate_rejects_interior_range(self) -> None:
        head = build_head_lattice()
        with pytest.raises(ValueError, match="exterior"):
            validate_echo_geometry(ECHO_MIN_RANGE_R - 0.5, ECHO_M_TRUE, head)

    def test_validate_rejects_weak_field_violation(self) -> None:
        head = build_head_lattice()
        # Heavy mass at minimum range: d_min ~ 2.0 < 10 * 0.5.
        with pytest.raises(ValueError, match="weak-field"):
            validate_echo_geometry(ECHO_MIN_RANGE_R, 0.5, head)

    def test_validate_accepts_shipped_defaults(self) -> None:
        head = build_head_lattice()
        validate_echo_geometry(ECHO_MIN_RANGE_R, ECHO_M_TRUE, head)
        config = echo_mass_config(ECHO_MIN_RANGE_R)
        potential, valid = _point_mass_potential_batch(
            config.positions.reshape(1, 1, 3), config.masses.reshape(1, 1), head
        )
        assert valid[0]
        assert np.max(np.abs(2.0 * potential)) <= 0.08

    def test_validate_rejects_non_finite_range(self) -> None:
        head = build_head_lattice()
        for bad in (float("inf"), float("nan")):
            with pytest.raises(ValueError, match="finite"):
                validate_echo_geometry(bad, ECHO_M_TRUE, head)

    def test_validate_uses_shared_weak_field_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        head = build_head_lattice()
        monkeypatch.setattr(scenarios, "WEAK_FIELD_LIMIT", 0.01)
        with pytest.raises(ValueError, match=r"\|2\*Phi\| <= 0.01"):
            validate_echo_geometry(ECHO_MIN_RANGE_R, ECHO_M_TRUE, head)

    @pytest.mark.parametrize(
        ("range_r", "message"),
        [(ECHO_MIN_RANGE_R - 0.1, "exterior"), (float("nan"), "finite")],
    )
    def test_make_observations_rejects_invalid_geometry(
        self, range_r: float, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            make_echo_observations(seed=0, range_r=range_r, n_observations=1)


class TestEchoMeasurementModel:
    @pytest.mark.parametrize("n_clocks", [0, 1, True, 2.5])
    def test_contrast_matrix_rejects_invalid_clock_count(
        self, n_clocks: object
    ) -> None:
        with pytest.raises(ValueError, match="integer >= 2"):
            contrast_matrix(n_clocks)

    def test_contrast_matrix_accepts_numpy_integer(self) -> None:
        assert contrast_matrix(np.int64(2)).shape == (1, 2)

    def test_contrast_matrix_is_orthonormal_and_removes_common_mode(self) -> None:
        q = contrast_matrix(27)
        assert q.shape == (26, 27)
        np.testing.assert_allclose(q @ q.T, np.eye(26), atol=1e-14)
        np.testing.assert_allclose(q @ np.ones(27), 0.0, atol=1e-14)

    def test_contrast_noise_retains_iid_variance(self) -> None:
        rng = np.random.default_rng(0)
        draws = rng.normal(0.0, 0.001, size=(100_000, 27))
        contrasts = draws @ contrast_matrix(27).T
        np.testing.assert_allclose(
            np.cov(contrasts, rowvar=False),
            1e-6 * np.eye(26),
            rtol=0.04,
            atol=2e-8,
        )

    def test_centered_observations_have_zero_mean(self) -> None:
        _, centered, contrasts = make_echo_observations(seed=0, range_r=2.0)
        assert len(centered) == len(contrasts)
        for display_obs, filter_obs in zip(centered, contrasts, strict=True):
            assert display_obs.rates.shape == (27,)
            assert filter_obs.rates.shape == (26,)
            np.testing.assert_allclose(
                filter_obs.rates,
                contrast_matrix(27) @ display_obs.rates,
                atol=1e-14,
            )
        for obs in centered:
            assert np.isclose(obs.rates.mean(), 0.0, atol=1e-12)

    def test_centering_removes_constant_offset(self) -> None:
        # A uniform offset (the M/R common mode) must vanish under centering.
        rates = np.full(27, 0.997)
        assert np.allclose(rates - rates.mean(), 0.0)

    def test_forward_model_batch_rows_are_centered(self) -> None:
        pf = build_echolocation_filter(seed=0, n_particles=50)
        assert pf.forward_model_batch is not None
        predicted = pf.forward_model_batch(pf.state.particles)
        assert predicted.shape == (50, 26)

    def test_scalar_forward_model_matches_batch(self) -> None:
        pf = build_echolocation_filter(seed=1, n_particles=8)
        assert pf.forward_model_batch is not None
        batch = pf.forward_model_batch(pf.state.particles)
        for i in range(8):
            single = pf.forward_model(pf.state.particles[i])
            assert np.allclose(single, batch[i])
            params = pf.state.particles[i]
            rates = clock_rates(
                MassConfig(params[:3].reshape(1, 3), params[3:4]),
                build_head_lattice(),
            )
            np.testing.assert_allclose(single, contrast_matrix(27) @ rates)

    def test_filter_builder_exposes_rigorous_smc_controls(self) -> None:
        params = inspect.signature(build_echolocation_filter).parameters
        assert params["ess_target"].default == 0.9
        assert params["rejuvenation_steps"].default == 1
        assert params["proposal_scale"].default == 1.5
        assert scenarios.ECHO_FAR_STD_FACTOR == 20.0
        pf = build_echolocation_filter(
            seed=0,
            n_particles=20,
            ess_target=0.7,
            rejuvenation_steps=4,
            proposal_scale=1.5,
        )
        assert pf.ess_target == 0.7
        assert pf.rejuvenation_steps == 4
        assert pf.proposal_scale == 1.5

    def test_prior_sampler_and_log_prior_match_physical_support(self) -> None:
        pf = build_echolocation_filter(seed=0, n_particles=100)
        inside = pf.state.particles
        assert np.all(pf.log_prior_density(inside) == 0.0)
        outside = inside.copy()
        outside[:, 3] = 0.2
        assert np.all(np.isneginf(pf.log_prior_density(outside)))
        singular = inside[:1].copy()
        singular[0, :3] = build_head_lattice().positions[0]
        singular[0, 3] = 0.01
        assert np.isneginf(pf.log_prior_density(singular)[0])


class TestEchoRunResult:
    def test_small_run_populates_fields(self) -> None:
        result: EchoRunResult = run_echolocation_3d(
            seed=0, range_r=2.0, n_particles=400, n_observations=15
        )
        assert result["seed"] == 0
        assert result["range_r"] == 2.0
        assert result["mean"].shape == (4,)
        assert result["std"].shape == (4,)
        assert result["position_error"] >= 0.0
        assert result["mass_error"] >= 0.0
        assert result["pos_std"] > 0.0
        assert result["mass_std"] > 0.0
        assert isinstance(result["passed"], bool)
        assert isinstance(result["covered_3sigma"], bool)
        assert result["residual_over_noise"] >= 0.0
        assert result["normalized_error"] >= 0.0
        assert result["forward_model_evaluations"] > 0
        assert result["ess_target"] == 0.9
        assert result["rejuvenation_steps"] == 1
        assert result["proposal_scale"] == 1.5

    def test_run_rejects_invalid_geometry(self) -> None:
        with pytest.raises(ValueError, match="exterior"):
            run_echolocation_3d(seed=0, range_r=1.0)


class TestWeightedQuantile:
    @pytest.mark.parametrize(
        ("quantile", "expected"),
        [(0.0, 10.0), (0.5, 10.0), (0.500001, 20.0), (0.8, 20.0), (1.0, 30.0)],
    )
    def test_quantile_exact_hand_case_with_unsorted_values(
        self, quantile: float, expected: float
    ) -> None:
        actual = scenarios._weighted_quantile(
            np.array([30.0, 10.0, 20.0]),
            np.array([0.2, 0.5, 0.3]),
            quantile,
        )
        assert actual == expected

    @pytest.mark.parametrize(
        ("quantile", "expected"),
        [(0.0, 1.0), (0.1, 1.0), (0.100001, 2.0), (0.6, 2.0), (1.0, 3.0)],
    )
    def test_quantile_combines_repeated_values(
        self, quantile: float, expected: float
    ) -> None:
        actual = scenarios._weighted_quantile(
            np.array([2.0, 1.0, 3.0, 2.0]),
            np.array([0.2, 0.1, 0.4, 0.3]),
            quantile,
        )
        assert actual == expected

    def test_quantile_endpoints_ignore_zero_weight_values(self) -> None:
        values = np.array([-100.0, 4.0, 100.0])
        weights = np.array([0.0, 2.0, 0.0])
        assert scenarios._weighted_quantile(values, weights, 0.0) == 4.0
        assert scenarios._weighted_quantile(values, weights, 1.0) == 4.0

    @pytest.mark.parametrize(
        "weights",
        [
            np.array([0.0, 0.0]),
            np.array([1.0, -0.1]),
            np.array([1.0, np.nan]),
            np.array([1.0, np.inf]),
        ],
    )
    def test_quantile_rejects_invalid_weights_without_warnings(
        self, weights: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="weights"):
            scenarios._weighted_quantile(np.array([1.0, 2.0]), weights, 0.5)

    @pytest.mark.parametrize(
        ("values", "weights", "message"),
        [
            (np.ones((2, 1)), np.ones(2), "1-D"),
            (np.ones(2), np.ones((2, 1)), "1-D"),
            (np.ones(2), np.ones(3), "matching"),
            (np.array([1.0 + 0.0j, 2.0]), np.ones(2), "real-valued"),
            (np.ones(2), np.array([1.0 + 0.0j, 2.0]), "real-valued"),
        ],
    )
    def test_quantile_rejects_complex_or_mismatched_inputs(
        self, values: np.ndarray, weights: np.ndarray, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            scenarios._weighted_quantile(values, weights, 0.5)


class TestArbitraryEcholocationCase:
    @staticmethod
    def _truth_position() -> np.ndarray:
        return np.array([4.5, -3.75, 3.5])

    def test_arbitrary_truth_runs_with_distinct_seed_provenance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        simulation_seeds: list[int | None] = []
        simulation_noise: list[float] = []
        inference_seeds: list[int] = []
        inference_noise: list[object] = []
        built_filters: list[object] = []
        raw_forward_batches: list[object] = []
        real_simulate = scenarios.simulate
        real_build_filter = scenarios.build_echolocation_filter

        def spy_simulate(config: object):
            simulation_seeds.append(config.seed)  # type: ignore[attr-defined]
            simulation_noise.append(  # type: ignore[attr-defined]
                config.noise.observation_std
            )
            return real_simulate(config)

        def spy_build_filter(seed: int, **kwargs: object):
            inference_seeds.append(seed)
            inference_noise.append(kwargs["noise_std"])
            particle_filter = real_build_filter(seed, **kwargs)
            built_filters.append(particle_filter)
            raw_forward_batches.append(particle_filter.forward_model_batch)
            return particle_filter

        monkeypatch.setattr(scenarios, "simulate", spy_simulate)
        monkeypatch.setattr(scenarios, "build_echolocation_filter", spy_build_filter)

        result = scenarios.run_echolocation_case(
            truth_position=self._truth_position(),
            truth_mass=0.065,
            observation_seed=np.int64(101),
            inference_seed=np.int32(202),
            n_particles=40,
            n_observations=1,
            noise_std=0.002,
        )

        assert simulation_seeds == [101]
        assert simulation_noise == [0.002]
        assert inference_seeds == [202]
        assert inference_noise == [0.002]
        assert type(simulation_seeds[0]) is int
        assert type(inference_seeds[0]) is int
        assert result.observation_seed == 101
        assert result.inference_seed == 202
        assert type(result.observation_seed) is int
        assert type(result.inference_seed) is int
        assert result.n_particles == 40
        assert result.n_observations == 1
        assert result.noise_std == 0.002
        assert result.range_r == pytest.approx(
            np.linalg.norm(self._truth_position()) / ECHO_R_HEAD
        )
        assert result.mean.shape == (4,)
        assert result.std.shape == (4,)
        assert result.marginal_95_lower.shape == (4,)
        assert result.marginal_95_upper.shape == (4,)
        assert result.marginal_95_covered.shape == (4,)
        assert result.position_error >= 0.0
        assert result.mass_error >= 0.0
        assert 0.0 <= result.angular_error_rad <= np.pi
        assert result.log_range_error >= 0.0
        assert result.log_mass_error >= 0.0
        assert result.residual_over_noise >= 0.0
        assert result.forward_model_evaluations > 0
        assert result.ess_target == 0.9
        assert result.rejuvenation_steps == 1
        assert result.proposal_scale == 1.5

        truth_position = self._truth_position()
        truth = np.append(truth_position, 0.065)
        expected_position_error = math.hypot(
            *(float(value) for value in result.mean[:3] - truth_position)
        )
        expected_mass_error = abs(float(result.mean[3]) - 0.065)
        truth_radius = math.hypot(*(float(value) for value in truth_position))
        estimated_radius = math.hypot(*(float(value) for value in result.mean[:3]))
        clipped_dot = float(
            np.clip(
                np.dot(
                    truth_position / truth_radius,
                    result.mean[:3] / estimated_radius,
                ),
                -1.0,
                1.0,
            )
        )
        truth_direction = truth_position / truth_radius
        estimated_direction = result.mean[:3] / estimated_radius
        expected_angular_error = math.atan2(
            math.hypot(
                *(
                    float(value)
                    for value in np.cross(estimated_direction, truth_direction)
                )
            ),
            clipped_dot,
        )
        expected_log_range_error = abs(
            math.log(estimated_radius) - math.log(truth_radius)
        )
        expected_log_mass_error = abs(math.log(float(result.mean[3])) - math.log(0.065))
        expected_coverage = (truth >= result.marginal_95_lower) & (
            truth <= result.marginal_95_upper
        )
        expected_pos_std = math.hypot(*(float(value) for value in result.std[:3]))
        expected_mass_std = float(result.std[3])
        expected_normalized_error = 0.5 * (
            expected_position_error / scenarios.ECHO_PASS_POS_TOL
            + expected_mass_error / scenarios.ECHO_PASS_MASS_TOL
        )
        clocks = build_head_lattice()
        true_rates = clock_rates(
            MassConfig(truth_position.reshape(1, 3), np.array([0.065])), clocks
        )
        true_contrasts = contrast_matrix(len(clocks.positions)) @ true_rates
        terminal_state = built_filters[0].state  # type: ignore[attr-defined]
        raw_forward_batch = raw_forward_batches[0]
        assert callable(raw_forward_batch)
        terminal_predictions = raw_forward_batch(terminal_state.particles)
        posterior_prediction = np.average(
            terminal_predictions, weights=terminal_state.weights, axis=0
        )
        expected_residual = float(
            np.max(np.abs(posterior_prediction - true_contrasts)) / 0.002
        )

        assert result.position_error == pytest.approx(expected_position_error)
        assert result.mass_error == pytest.approx(expected_mass_error)
        assert result.angular_error_rad == pytest.approx(expected_angular_error)
        assert result.log_range_error == pytest.approx(expected_log_range_error)
        assert result.log_mass_error == pytest.approx(expected_log_mass_error)
        np.testing.assert_array_equal(result.marginal_95_covered, expected_coverage)
        assert result.pos_std == pytest.approx(expected_pos_std)
        assert result.mass_std == pytest.approx(expected_mass_std)
        assert result.normalized_error == pytest.approx(expected_normalized_error)
        assert result.residual_over_noise == pytest.approx(expected_residual)

    @pytest.mark.parametrize(
        "invalid_seed",
        [None, True, 1.5, -1, 3 + 0.0j],
        ids=("none", "bool", "float", "negative", "complex"),
    )
    @pytest.mark.parametrize("seed_name", ["observation_seed", "inference_seed"])
    def test_arbitrary_case_rejects_invalid_seeds_without_warnings(
        self, seed_name: str, invalid_seed: object
    ) -> None:
        arguments: dict[str, object] = {
            "truth_position": self._truth_position(),
            "truth_mass": 0.065,
            "observation_seed": 101,
            "inference_seed": 202,
            "n_particles": 4,
            "n_observations": 1,
        }
        arguments[seed_name] = invalid_seed

        with pytest.raises(ValueError, match=rf"{seed_name}.*nonnegative integer"):
            scenarios.run_echolocation_case(**arguments)

    def test_arbitrary_case_survives_nonconvex_invalid_cartesian_mean(self) -> None:
        truth_direction = np.array([1.0, 2.0, 3.0])
        truth_direction /= np.linalg.norm(truth_direction)

        result = scenarios.run_echolocation_case(
            truth_position=truth_direction * (8.0 * ECHO_R_HEAD),
            truth_mass=0.065,
            observation_seed=2,
            inference_seed=102,
            n_particles=10,
            n_observations=1,
        )

        assert np.isfinite(result.residual_over_noise)
        assert result.forward_model_evaluations == 59

    def test_predictive_residual_uses_normalized_terminal_weights(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        truth_position = self._truth_position()
        clocks = build_head_lattice()
        truth_rates = clock_rates(
            MassConfig(truth_position.reshape(1, 3), np.array([0.065])), clocks
        )
        truth_contrasts = contrast_matrix(len(clocks.positions)) @ truth_rates
        prediction_rows = np.multiply.outer(
            np.array([0.0, 1.0, 2.0, 3.0]), truth_contrasts
        )
        terminal_state = ParticleState(
            particles=np.array(
                [
                    [3.5, -2.5, 4.0, 0.02],
                    [4.0, -3.0, 4.5, 0.04],
                    [4.5, -3.5, 5.0, 0.06],
                    [5.0, -4.0, 5.5, 0.08],
                ]
            ),
            weights=np.array([0.01, 0.01, 0.96, 0.02]),
            observations_seen=1,
        )

        def raw_forward(particles: np.ndarray) -> np.ndarray:
            assert len(particles) == len(prediction_rows)
            return prediction_rows.copy()

        fake_filter = _TerminalStateEchoFilter(terminal_state, raw_forward)
        monkeypatch.setattr(
            scenarios,
            "build_echolocation_filter",
            lambda *args, **kwargs: fake_filter,
        )

        result = scenarios.run_echolocation_case(
            truth_position=truth_position,
            truth_mass=0.065,
            observation_seed=101,
            inference_seed=202,
            n_particles=4,
            n_observations=1,
        )

        weighted_prediction = np.average(
            prediction_rows, weights=terminal_state.weights, axis=0
        )
        expected = float(np.max(np.abs(weighted_prediction - truth_contrasts)) / 0.001)
        uniform_prediction = prediction_rows.mean(axis=0)
        uniform_residual = float(
            np.max(np.abs(uniform_prediction - truth_contrasts)) / 0.001
        )
        assert result.residual_over_noise == pytest.approx(expected)
        assert not np.isclose(expected, uniform_residual)
        assert result.forward_model_evaluations == 4

    def test_angular_error_is_exactly_zero_for_identical_directions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        truth_position = np.array([-3.0, -3.0, -1.0])
        particle = np.append(truth_position, 0.065)
        terminal_state = ParticleState(
            particles=np.repeat(particle[np.newaxis, :], 4, axis=0),
            weights=np.array([0.1, 0.2, 0.3, 0.4]),
            observations_seen=1,
        )
        _, raw_forward = scenarios._make_echo_forward_models(build_head_lattice())
        fake_filter = _TerminalStateEchoFilter(terminal_state, raw_forward)
        monkeypatch.setattr(
            scenarios,
            "build_echolocation_filter",
            lambda *args, **kwargs: fake_filter,
        )

        result = scenarios.run_echolocation_case(
            truth_position=truth_position,
            truth_mass=0.065,
            observation_seed=101,
            inference_seed=202,
            n_particles=4,
            n_observations=1,
        )

        assert result.angular_error_rad == 0.0

    def test_zero_posterior_mean_has_explicit_undefined_direction_convention(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        terminal_state = ParticleState(
            particles=np.array([[-4.0, 0.0, 0.0, 0.05], [4.0, 0.0, 0.0, 0.05]]),
            weights=np.array([0.5, 0.5]),
            observations_seen=1,
        )
        _, raw_forward = scenarios._make_echo_forward_models(build_head_lattice())
        fake_filter = _TerminalStateEchoFilter(terminal_state, raw_forward)
        monkeypatch.setattr(
            scenarios,
            "build_echolocation_filter",
            lambda *args, **kwargs: fake_filter,
        )

        result = scenarios.run_echolocation_case(
            truth_position=np.array([4.0, 4.0, 4.0]),
            truth_mass=0.065,
            observation_seed=101,
            inference_seed=202,
            n_particles=2,
            n_observations=1,
        )

        assert math.isnan(result.angular_error_rad)
        assert result.log_range_error == math.inf
        assert np.isfinite(result.residual_over_noise)
        assert result.forward_model_evaluations == 2

    def test_arbitrary_result_is_a_deeply_immutable_snapshot(self) -> None:
        result = scenarios.run_echolocation_case(
            truth_position=self._truth_position(),
            truth_mass=0.065,
            observation_seed=101,
            inference_seed=202,
            n_particles=20,
            n_observations=1,
        )

        with pytest.raises(FrozenInstanceError):
            result.truth_mass = 0.08  # type: ignore[misc]
        for array in (
            result.truth_position,
            result.mean,
            result.std,
            result.marginal_95_lower,
            result.marginal_95_upper,
            result.marginal_95_covered,
        ):
            assert not array.flags.writeable
            with pytest.raises(ValueError, match="WRITEABLE"):
                array.setflags(write=True)

    def test_arbitrary_case_uses_terminal_particle_weights_for_marginal_bounds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        terminal_state = ParticleState(
            particles=np.array(
                [
                    [3.5, -2.5, 4.0, 0.02],
                    [4.0, -3.0, 4.5, 0.04],
                    [4.5, -3.5, 5.0, 0.06],
                    [5.0, -4.0, 5.5, 0.08],
                ]
            ),
            weights=np.array([0.01, 0.01, 0.96, 0.02]),
            observations_seen=1,
        )

        class FakeFilter:
            ess_target = 0.9
            rejuvenation_steps = 1
            proposal_scale = 1.5
            state = terminal_state

            def __init__(self) -> None:
                self.forward_model_batch = lambda particles: np.zeros(
                    (len(particles), 26)
                )

            def update(self, observation: object) -> ParticleState:
                assert self.forward_model_batch is not None
                self.forward_model_batch(self.state.particles)
                return self.state

            def estimate(self) -> dict[str, object]:
                mean = np.average(
                    self.state.particles, weights=self.state.weights, axis=0
                )
                variance = np.average(
                    (self.state.particles - mean) ** 2,
                    weights=self.state.weights,
                    axis=0,
                )
                return {"mean": mean, "std": np.sqrt(variance), "ess": 1.0}

        monkeypatch.setattr(
            scenarios, "build_echolocation_filter", lambda *args, **kwargs: FakeFilter()
        )

        result = scenarios.run_echolocation_case(
            truth_position=self._truth_position(),
            truth_mass=0.065,
            observation_seed=101,
            inference_seed=202,
            n_particles=4,
            n_observations=1,
        )

        np.testing.assert_array_equal(
            result.marginal_95_lower, terminal_state.particles[2]
        )
        np.testing.assert_array_equal(
            result.marginal_95_upper, terminal_state.particles[2]
        )

    @pytest.mark.parametrize(
        ("truth_position", "message"),
        [
            (np.array([1.0, 2.0]), "shape"),
            (np.ones((1, 3)), "shape"),
            (np.array([4.0, np.nan, 4.0]), "finite"),
            (np.array([4.0, np.inf, 4.0]), "finite"),
            (np.array([4.0 + 0.0j, 4.0, 4.0]), "real-valued"),
            (np.array([1.9, 1.9, 1.9]), "exterior"),
        ],
    )
    def test_arbitrary_truth_rejects_invalid_positions_without_warnings(
        self, truth_position: np.ndarray, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            scenarios.run_echolocation_case(
                truth_position=truth_position,
                truth_mass=0.065,
                observation_seed=101,
                inference_seed=202,
                n_particles=4,
                n_observations=1,
            )

    def test_arbitrary_truth_rejects_position_one_ulp_inside_exterior(self) -> None:
        radius_inside = np.nextafter(2.0 * ECHO_R_HEAD, -np.inf)

        with pytest.raises(ValueError, match="exterior"):
            scenarios.run_echolocation_case(
                truth_position=np.array([radius_inside, 0.0, 0.0]),
                truth_mass=0.065,
                observation_seed=101,
                inference_seed=202,
                n_particles=4,
                n_observations=1,
            )

    @pytest.mark.parametrize(
        ("truth_mass", "message"),
        [
            (0.0, "positive"),
            (-0.1, "positive"),
            (True, "positive"),
            (np.nan, "finite"),
            (np.inf, "finite"),
            (0.08 + 0.0j, "real-valued"),
            (0.5, "weak-field"),
        ],
    )
    def test_arbitrary_truth_rejects_invalid_mass_without_warnings(
        self, truth_mass: object, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            scenarios.run_echolocation_case(
                truth_position=np.array([2.0 * ECHO_R_HEAD, 0.0, 0.0]),
                truth_mass=truth_mass,
                observation_seed=101,
                inference_seed=202,
                n_particles=4,
                n_observations=1,
            )

    def test_fixed_echo_wrapper_matches_frozen_c6e0744_small_run(self) -> None:
        fixed = run_echolocation_3d(
            seed=7,
            range_r=2.0,
            n_particles=30,
            n_observations=1,
            ess_target=0.9,
            rejuvenation_steps=1,
            proposal_scale=1.5,
        )
        expected = {
            "seed": 7,
            "range_r": 2.0,
            "passed": True,
            "mean": np.array(
                [
                    1.1365954774440301,
                    1.5184478329925997,
                    3.020741480218203,
                    0.08909366235239165,
                ]
            ),
            "std": np.array(
                [
                    0.10900237729155642,
                    0.1002304279955645,
                    0.15431612366017894,
                    0.009787139507277973,
                ]
            ),
            "position_error": 0.1592597117326266,
            "mass_error": 0.009093662352391646,
            "pos_std": 0.21387174421341518,
            "mass_std": 0.009787139507277973,
            "covered_3sigma": True,
            "residual_over_noise": 1.3604684572596781,
            "normalized_error": 0.1933006352712089,
            "forward_model_evaluations": 2094,
            "ess_target": 0.9,
            "rejuvenation_steps": 1,
            "proposal_scale": 1.5,
        }
        _assert_frozen_echo_run(fixed, expected)

    def test_fixed_echo_wrapper_matches_frozen_multi_observation_run(self) -> None:
        """Frozen replay over several observations, verified against pre-PR main.

        The single-observation replay above cannot distinguish consuming every
        observation from consuming only the first, so on its own it would let
        the certified 80-observation default silently collapse to one update.
        """
        fixed = run_echolocation_3d(
            seed=3,
            range_r=3.0,
            n_particles=60,
            n_observations=8,
            ess_target=0.9,
            rejuvenation_steps=1,
            proposal_scale=1.5,
        )
        expected = {
            "seed": 3,
            "range_r": 3.0,
            "passed": False,
            "mean": np.array(
                [
                    1.5220716586782868,
                    2.8047042691472983,
                    5.300753194916001,
                    0.11351651043544787,
                ]
            ),
            "std": np.array(
                [
                    0.2330620701460182,
                    0.338417846207634,
                    0.49297174041279135,
                    0.02073619325108756,
                ]
            ),
            "position_error": 1.025908581114942,
            "mass_error": 0.03351651043544787,
            "pos_std": 0.6417676402080257,
            "mass_std": 0.02073619325108756,
            "covered_3sigma": True,
            "residual_over_noise": 0.4089018343443884,
            "normalized_error": 0.9319106710005693,
            "forward_model_evaluations": 6616,
            "ess_target": 0.9,
            "rejuvenation_steps": 1,
            "proposal_scale": 1.5,
        }
        _assert_frozen_echo_run(fixed, expected)

    def test_nondefault_smc_controls_reach_the_filter_builder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Assert the forwarded arguments, not the echoed ones.

        ``EcholocationCaseResult`` reports the controls it was *asked* for, so a
        runner that quietly dropped them and built a default filter would still
        produce a result that looks correct. Only the builder's own arguments
        distinguish the two, and only at non-default values.
        """
        forwarded: list[dict[str, object]] = []
        real_build_filter = scenarios.build_echolocation_filter

        def spy_build_filter(seed: int, **kwargs: object) -> ParticleFilter:
            forwarded.append(dict(kwargs))
            return real_build_filter(seed, **kwargs)

        monkeypatch.setattr(scenarios, "build_echolocation_filter", spy_build_filter)
        result = scenarios.run_echolocation_case(
            truth_position=self._truth_position(),
            truth_mass=0.065,
            observation_seed=101,
            inference_seed=202,
            n_particles=7,
            n_observations=1,
            noise_std=0.003,
            ess_target=0.55,
            rejuvenation_steps=3,
            proposal_scale=2.38,
        )

        assert forwarded == [
            {
                "n_particles": 7,
                "noise_std": 0.003,
                "ess_target": 0.55,
                "rejuvenation_steps": 3,
                "proposal_scale": 2.38,
            }
        ]
        assert (
            result.ess_target,
            result.rejuvenation_steps,
            result.proposal_scale,
        ) != (
            scenarios.ECHO_ESS_TARGET,
            scenarios.ECHO_REJUVENATION_STEPS,
            scenarios.ECHO_PROPOSAL_SCALE,
        )

    def test_every_contrast_observation_reaches_the_filter_in_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pin the update identities, not just the numbers they happen to produce.

        A frozen-value replay proves the outcome; this proves the filter was fed
        exactly the simulated contrasts, all of them, once each, in time order.
        """
        clocks = build_head_lattice()
        truth_position = echo_mass_position(2.5)
        seen: list[Observation] = []
        real_build_filter = scenarios.build_echolocation_filter

        def spy_build_filter(seed: int, **kwargs: object) -> ParticleFilter:
            particle_filter = real_build_filter(seed, **kwargs)
            real_update = particle_filter.update

            def recording_update(observation: Observation) -> object:
                seen.append(observation)
                return real_update(observation)

            monkeypatch.setattr(particle_filter, "update", recording_update)
            return particle_filter

        monkeypatch.setattr(scenarios, "build_echolocation_filter", spy_build_filter)
        run_echolocation_3d(seed=5, range_r=2.5, n_particles=8, n_observations=6)

        sim, _, expected_contrasts = scenarios._simulate_echolocation_observations(
            truth_position=truth_position,
            truth_mass=scenarios.ECHO_M_TRUE,
            observation_seed=5,
            n_observations=6,
            noise_std=scenarios.ECHO_NOISE_STD,
        )
        assert len(sim.observations) == 6
        assert len(seen) == 6
        assert [observation.time for observation in seen] == [
            float(index) for index in range(6)
        ]
        for actual, expected in zip(seen, expected_contrasts, strict=True):
            np.testing.assert_array_equal(actual.rates, expected.rates)
            assert actual.rates.shape == (len(clocks.positions) - 1,)
            assert actual.time == expected.time
