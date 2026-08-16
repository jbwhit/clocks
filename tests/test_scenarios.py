"""Fast tests for the shared multi-mass-2D scenario module."""

import importlib.util
import inspect
import sys
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
from clocks.physics import _point_mass_potential_batch, clock_rates
from clocks.types import MassConfig


def _load_scan(script_name: str) -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / f"{script_name}.py"
    spec = importlib.util.spec_from_file_location(script_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
