"""Fast tests for deterministic calibration evidence and archival."""

import importlib
import importlib.util
import itertools
import json
import math
import re
import statistics
from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest


def _calibration_module():
    spec = importlib.util.find_spec("clocks._calibration")
    assert spec is not None, "clocks._calibration must define the raw-study schema"
    return importlib.import_module("clocks._calibration")


def _archive_module():
    path = Path(__file__).parents[1] / "scripts/archive_development_calibration.py"
    assert path.exists(), "development calibration archiver must exist"
    spec = importlib.util.spec_from_file_location(
        "archive_development_calibration", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ESS_TARGETS = (0.7, 0.8, 0.9)
REJUVENATION_STEPS = (1, 2, 4)
PROPOSAL_SCALES = (1.5, 2.38, 3.0)
SEEDS = tuple(range(12))
RANGES = (2.0, 2.6, 3.5, 4.6, 6.1, 8.0)
CONTROL_GRID = {
    "ess_target": ESS_TARGETS,
    "rejuvenation_steps": REJUVENATION_STEPS,
    "proposal_scale": PROPOSAL_SCALES,
}
REPO_ROOT = Path(__file__).parents[1]
TRACKED_MULTI = REPO_ROOT / "docs/calibration/multi_mass_2d_development.json"
TRACKED_ECHO = REPO_ROOT / "docs/calibration/echolocation_range_development.json"
MULTI_SOURCE_SHA256 = "f51c2f0733d3f558daf6a4d6c50efa16fdcea392156bcca6f51aa26002be81d8"
ECHO_SOURCE_SHA256 = "eafc3ae9b74e33b278543bd76203979a7ab3e8d9b9f36cb02a150e63c158e7d0"


def _multi_results() -> list[dict]:
    truth = np.array([-3.0, 2.0, 4.0, -1.0, 0.050, 0.030])
    return [
        {
            "seed": seed,
            "passed": True,
            "mean": truth.copy(),
            "std": np.ones(6),
            "max_abs_error": 0.0,
            "covered_3sigma": True,
            "max_posterior_std": 1.0,
            "residual_over_noise": 0.0,
            "normalized_error": 0.0,
            "forward_model_evaluations": 100,
            "ess_target": ess,
            "rejuvenation_steps": steps,
            "proposal_scale": scale,
        }
        for ess, steps, scale in itertools.product(
            ESS_TARGETS, REJUVENATION_STEPS, PROPOSAL_SCALES
        )
        for seed in SEEDS
    ]


def _echo_results() -> list[dict]:
    direction = np.array([2.0, 3.0, 6.0]) / 7.0
    return [
        {
            "seed": seed,
            "range_r": range_r,
            "passed": True,
            "mean": np.append(range_r * math.sqrt(3.0) * direction, 0.08),
            "std": np.ones(4),
            "position_error": 0.0,
            "mass_error": 0.0,
            "pos_std": math.sqrt(3.0),
            "mass_std": 1.0,
            "covered_3sigma": True,
            "residual_over_noise": 0.0,
            "normalized_error": 0.0,
            "forward_model_evaluations": 100,
            "ess_target": ess,
            "rejuvenation_steps": steps,
            "proposal_scale": scale,
        }
        for ess, steps, scale in itertools.product(
            ESS_TARGETS, REJUVENATION_STEPS, PROPOSAL_SCALES
        )
        for range_r in RANGES
        for seed in SEEDS
    ]


def _write_valid_inputs(tmp_path: Path) -> tuple[Path, Path]:
    calibration = _calibration_module()
    multi_path = tmp_path / "multi.json"
    echo_path = tmp_path / "legacy-echo.json"
    multi = calibration.build_study_document(
        study="multi_mass_2d",
        seed_block=0,
        seeds=SEEDS,
        control_grid=CONTROL_GRID,
        tolerances={"absolute_parameter_error": [2.5, 2.5, 2.5, 2.5, 0.012, 0.012]},
        results=_multi_results(),
    )
    calibration.write_study(multi_path, multi)
    legacy_echo = {
        "seed_block": 0,
        "results": calibration.build_study_document(
            study="echolocation_range",
            seed_block=0,
            seeds=SEEDS,
            control_grid=CONTROL_GRID,
            tolerances={
                "position_error_max": 1.0,
                "mass_error_max": 0.04,
                "far_std_factor_min": 20.0,
            },
            ranges=RANGES,
            results=_echo_results(),
        )["results"],
    }
    calibration.write_study(echo_path, legacy_echo)
    return multi_path, echo_path


def _ranked_cells(study: dict) -> tuple[list[tuple], dict[tuple, list[dict]]]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for result in study["results"]:
        key = (
            result["ess_target"],
            result["rejuvenation_steps"],
            result["proposal_scale"],
        )
        grouped[key].append(result)
    ranked = []
    for key, cell in grouped.items():
        ranked.append(
            (
                (
                    -sum(result["passed"] for result in cell),
                    statistics.median(result["normalized_error"] for result in cell),
                    statistics.median(
                        result["forward_model_evaluations"] for result in cell
                    ),
                    key[1],
                ),
                key,
            )
        )
    return sorted(ranked), grouped


def _assert_all_finite(value) -> None:
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, list):
        for item in value:
            _assert_all_finite(item)
    elif isinstance(value, dict):
        for item in value.values():
            _assert_all_finite(item)


def test_schema_round_trip_is_deterministic_and_json_safe(tmp_path: Path) -> None:
    calibration = _calibration_module()
    results = [
        {
            "seed": 1,
            "ess_target": 0.8,
            "rejuvenation_steps": 2,
            "proposal_scale": 3.0,
            "mean": np.array([2.0, 3.0]),
        },
        {
            "seed": 0,
            "ess_target": 0.8,
            "rejuvenation_steps": 2,
            "proposal_scale": 3.0,
            "mean": np.array([1.0, 2.0]),
        },
    ]
    study = calibration.build_study_document(
        study="multi_mass_2d",
        seed_block=0,
        seeds=(0, 1),
        control_grid={
            "ess_target": (0.8,),
            "rejuvenation_steps": (2,),
            "proposal_scale": (3.0,),
        },
        tolerances={"absolute_parameter_error": np.array([2.5, 0.012])},
        results=results,
    )
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    calibration.write_study(first, study)
    calibration.write_study(second, study)

    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes().endswith(b"\n")
    loaded = calibration.load_study(first)
    assert loaded["schema_version"] == 1
    assert loaded["study"] == "multi_mass_2d"
    assert loaded["seed_block"] == 0
    assert loaded["seed_role"] == "development"
    assert loaded["seeds"] == [0, 1]
    assert loaded["tolerances"]["absolute_parameter_error"] == [2.5, 0.012]
    assert [result["seed"] for result in loaded["results"]] == [0, 1]
    assert loaded["results"][0]["mean"] == [1.0, 2.0]


def test_schema_writer_rejects_nonfinite_json_values(tmp_path: Path) -> None:
    calibration = _calibration_module()
    path = tmp_path / "bad.json"
    study = calibration.build_study_document(
        study="multi_mass_2d",
        seed_block=0,
        seeds=(0,),
        control_grid={
            "ess_target": (0.8,),
            "rejuvenation_steps": (2,),
            "proposal_scale": (3.0,),
        },
        tolerances={"absolute_parameter_error": (2.5,)},
        results=[
            {
                "seed": 0,
                "ess_target": 0.8,
                "rejuvenation_steps": 2,
                "proposal_scale": 3.0,
                "normalized_error": np.nan,
            }
        ],
    )

    path.write_bytes(b"existing evidence\n")
    with np.testing.assert_raises(ValueError):
        calibration.write_study(path, study)
    assert path.read_bytes() == b"existing evidence\n"


def test_schema_file_is_plain_json_without_numpy_type_tags(tmp_path: Path) -> None:
    calibration = _calibration_module()
    study = calibration.build_study_document(
        study="echolocation_range",
        seed_block=0,
        seeds=(0,),
        control_grid={
            "ess_target": (0.9,),
            "rejuvenation_steps": (1,),
            "proposal_scale": (1.5,),
        },
        tolerances={"position_error": 1.0, "mass_error": 0.04},
        ranges=(2.0,),
        results=[
            {
                "seed": 0,
                "range_r": 2.0,
                "ess_target": 0.9,
                "rejuvenation_steps": 1,
                "proposal_scale": 1.5,
                "mean": np.array([1.0, 2.0]),
            }
        ],
    )
    path = tmp_path / "echo.json"
    calibration.write_study(path, study)

    raw = json.loads(path.read_text())
    assert raw["ranges"] == [2.0]
    assert raw["results"][0]["mean"] == [1.0, 2.0]
    assert "numpy" not in path.read_text().lower()


def test_tracked_development_artifacts_are_canonical_complete_and_provenanced() -> None:
    calibration = _calibration_module()
    archive = _archive_module()
    multi = calibration.load_study(TRACKED_MULTI)
    echo = calibration.load_study(TRACKED_ECHO)

    archive._require_schema_metadata(
        multi, study="multi_mass_2d", tolerances=archive.MULTI_TOLERANCES
    )
    multi_results = archive._results(multi, 324)
    archive._validate_multi_results(multi_results)
    archive._require_schema_metadata(
        echo,
        study="echolocation_range",
        tolerances=archive.ECHO_TOLERANCES,
        ranges=RANGES,
    )
    echo_results = archive._results(echo, 1944)
    archive._validate_echo_results(echo_results)

    assert TRACKED_MULTI.read_text() == calibration.encode_study(multi)
    assert TRACKED_ECHO.read_text() == calibration.encode_study(echo)
    assert multi["source"] == {
        "format": "schema_v1",
        "sha256": MULTI_SOURCE_SHA256,
    }
    assert echo["source"] == {
        "format": "legacy_seed_block_results",
        "sha256": ECHO_SOURCE_SHA256,
    }
    _assert_all_finite(multi)
    _assert_all_finite(echo)


def test_tracked_development_artifacts_reproduce_frozen_winners_and_gates() -> None:
    calibration = _calibration_module()
    multi = calibration.load_study(TRACKED_MULTI)
    echo = calibration.load_study(TRACKED_ECHO)
    multi_ranked, multi_grouped = _ranked_cells(multi)
    echo_ranked, echo_grouped = _ranked_cells(echo)

    assert multi_ranked[0][1] == (0.7, 2, 3.0)
    multi_winner = multi_grouped[multi_ranked[0][1]]
    assert sum(result["passed"] for result in multi_winner) == 11
    assert [result["seed"] for result in multi_winner if not result["passed"]] == [6]

    assert echo_ranked[0][1] == (0.9, 1, 1.5)
    echo_winner = echo_grouped[echo_ranked[0][1]]
    by_range: dict[float, list[dict]] = defaultdict(list)
    for result in echo_winner:
        by_range[result["range_r"]].append(result)
    ranges = sorted(by_range)
    assert [sum(result["passed"] for result in by_range[r]) for r in ranges] == [
        12,
        12,
        12,
        8,
        4,
        1,
    ]
    assert max(result["position_error"] for result in by_range[2.0]) == pytest.approx(
        0.06330269304026884
    )
    assert max(result["mass_error"] for result in by_range[2.0]) == pytest.approx(
        0.0030868621553362674
    )
    assert sum(result["covered_3sigma"] for result in by_range[8.0]) == 12
    far_close_ratio = statistics.median(
        result["pos_std"] for result in by_range[8.0]
    ) / statistics.median(result["pos_std"] for result in by_range[2.0])
    assert far_close_ratio == pytest.approx(66.22618610746335)


def test_development_report_cites_raw_artifacts_and_matches_cell_summaries() -> None:
    calibration = _calibration_module()
    report = (REPO_ROOT / "docs/2026-08-16-development-calibration.md").read_text()
    multi_link = (
        "[multi-mass raw development artifact]"
        "(calibration/multi_mass_2d_development.json)"
    )
    echo_link = (
        "[echolocation raw development artifact]"
        "(calibration/echolocation_range_development.json)"
    )
    assert multi_link in report
    assert echo_link in report
    assert "does not exist yet" not in report
    assert "not yet a tracked artifact" not in report
    assert MULTI_SOURCE_SHA256 in report
    assert ECHO_SOURCE_SHA256 in report

    blocks = re.findall(r"```text\n(.*?)```", report, re.S)
    assert len(blocks) == 2
    pattern = re.compile(
        r"ess=(\d\.\d\d) steps=(\d) scale=(\d\.\d\d): "
        r"(\d+)/(\d+), median normalized error=(\d\.\d{3}), "
        r"median forward evaluations=(\d+)"
    )
    for block, path in zip(blocks, (TRACKED_MULTI, TRACKED_ECHO), strict=True):
        actual = [
            pattern.fullmatch(line).groups() for line in block.strip().splitlines()
        ]
        ranked, grouped = _ranked_cells(calibration.load_study(path))
        del ranked
        expected = []
        for (ess, steps, scale), cell in sorted(grouped.items()):
            median_error = statistics.median(
                result["normalized_error"] for result in cell
            )
            median_evaluations = statistics.median(
                result["forward_model_evaluations"] for result in cell
            )
            expected.append(
                (
                    f"{ess:.2f}",
                    str(steps),
                    f"{scale:.2f}",
                    str(sum(result["passed"] for result in cell)),
                    str(len(cell)),
                    f"{median_error:.3f}",
                    f"{median_evaluations:.0f}",
                )
            )
        assert actual == expected


def test_archive_validates_and_canonicalizes_complete_development_grids(
    tmp_path: Path,
) -> None:
    archive = _archive_module()
    multi_path, echo_path = _write_valid_inputs(tmp_path)
    output_dir = tmp_path / "tracked"

    written = archive.archive_development_studies(
        multi_path=multi_path,
        echo_path=echo_path,
        output_dir=output_dir,
    )

    assert written == (
        output_dir / "multi_mass_2d_development.json",
        output_dir / "echolocation_range_development.json",
    )
    multi = json.loads(written[0].read_text())
    echo = json.loads(written[1].read_text())
    assert len(multi["results"]) == 324
    assert len(echo["results"]) == 1944
    assert multi["source"] == {
        "format": "schema_v1",
        "sha256": sha256(multi_path.read_bytes()).hexdigest(),
    }
    assert echo["schema_version"] == 1
    assert echo["study"] == "echolocation_range"
    assert echo["source"] == {
        "format": "legacy_seed_block_results",
        "sha256": sha256(echo_path.read_bytes()).hexdigest(),
    }
    assert echo["control_grid"] == {
        key: list(values) for key, values in CONTROL_GRID.items()
    }
    assert echo["ranges"] == list(RANGES)


def test_archive_validates_strict_json_before_writing_either_artifact(
    tmp_path: Path,
) -> None:
    archive = _archive_module()
    calibration = _calibration_module()
    multi_path, echo_path = _write_valid_inputs(tmp_path)
    echo = calibration.load_study(echo_path)
    echo["results"][0]["normalized_error"] = np.nan
    echo_path.write_text(json.dumps(echo), encoding="utf-8")
    output_dir = tmp_path / "tracked"

    with pytest.raises(ValueError, match="normalized_error.*finite"):
        archive.archive_development_studies(
            multi_path=multi_path,
            echo_path=echo_path,
            output_dir=output_dir,
        )
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("target", "mutation", "match"),
    [
        ("multi", lambda r: r.update(mean="bad"), "mean.*array"),
        ("multi", lambda r: r.update(mean=[0.0] * 5), "mean.*length 6"),
        ("multi", lambda r: r["mean"].__setitem__(0, math.nan), "mean.*finite"),
        ("multi", lambda r: r.update(std=[1.0] * 5), "std.*length 6"),
        ("multi", lambda r: r["std"].__setitem__(0, -1.0), "std.*nonnegative"),
        ("multi", lambda r: r.update(passed=False), "passed.*derived"),
        ("multi", lambda r: r.update(max_abs_error=0.1), "max_abs_error"),
        ("multi", lambda r: r.update(covered_3sigma=False), "covered_3sigma"),
        ("multi", lambda r: r.update(max_posterior_std=2.0), "max_posterior_std"),
        ("multi", lambda r: r.update(normalized_error=0.1), "normalized_error"),
        ("multi", lambda r: r.update(residual_over_noise=-0.1), "residual_over_noise"),
        (
            "multi",
            lambda r: r.update(forward_model_evaluations=True),
            "forward_model_evaluations",
        ),
        (
            "multi",
            lambda r: r.update(forward_model_evaluations=0),
            "forward_model_evaluations",
        ),
        ("multi", lambda r: r.update(ess_target=True), "ess_target"),
        ("multi", lambda r: r.update(rejuvenation_steps=1.0), "rejuvenation_steps"),
        (
            "multi",
            lambda r: r.update(proposal_scale=math.inf),
            "proposal_scale.*finite",
        ),
        ("multi", lambda r: r.update(seed=False), "seed"),
        ("echo", lambda r: r.update(mean=[0.0] * 3), "mean.*length 4"),
        ("echo", lambda r: r.update(std=[1.0, 1.0, -1.0, 1.0]), "std.*nonnegative"),
        ("echo", lambda r: r.update(position_error=0.1), "position_error"),
        ("echo", lambda r: r.update(mass_error=0.1), "mass_error"),
        ("echo", lambda r: r.update(pos_std=1.0), "pos_std"),
        ("echo", lambda r: r.update(mass_std=2.0), "mass_std"),
        ("echo", lambda r: r.update(passed=False), "passed.*derived"),
        ("echo", lambda r: r.update(covered_3sigma=False), "covered_3sigma"),
        ("echo", lambda r: r.update(normalized_error=0.1), "normalized_error"),
        ("echo", lambda r: r.update(residual_over_noise=True), "residual_over_noise"),
        (
            "echo",
            lambda r: r.update(forward_model_evaluations=1.5),
            "forward_model_evaluations",
        ),
        ("echo", lambda r: r.update(range_r=False), "range_r"),
        ("echo", lambda r: r.update(seed="0"), "seed"),
    ],
)
def test_archive_rejects_semantically_corrupt_complete_results(
    target: str,
    mutation,
    match: str,
) -> None:
    archive = _archive_module()
    results = deepcopy(_multi_results() if target == "multi" else _echo_results())
    mutation(results[0])

    validator = (
        archive._validate_multi_results
        if target == "multi"
        else archive._validate_echo_results
    )
    with pytest.raises(ValueError, match=match):
        validator(results)


@pytest.mark.parametrize(
    ("target", "mutation", "match"),
    [
        ("multi", lambda data: data.update(seed_block=500), "seed_block=0"),
        ("multi", lambda data: data.update(seed_block=False), "seed_block=0"),
        ("multi", lambda data: data.update(schema_version=True), "schema_version"),
        ("multi", lambda data: data["seeds"].__setitem__(0, False), "seeds"),
        (
            "multi",
            lambda data: data["results"][0].update(seed=12),
            "result tuples",
        ),
        ("multi", lambda data: data["results"].pop(), "324"),
        (
            "multi",
            lambda data: data["results"].__setitem__(-1, data["results"][0].copy()),
            "result tuples",
        ),
        (
            "multi",
            lambda data: data["control_grid"]["ess_target"].append(0.95),
            "control_grid",
        ),
        (
            "multi",
            lambda data: data["control_grid"]["rejuvenation_steps"].__setitem__(
                0, True
            ),
            "control_grid",
        ),
        (
            "multi",
            lambda data: data["tolerances"].update(
                absolute_parameter_error=[3.0, 3.0, 3.0, 3.0, 0.012, 0.012]
            ),
            "tolerances",
        ),
        ("echo", lambda data: data.update(seed_block=False), "seed_block=0"),
        ("echo", lambda data: data["results"].pop(), "1944"),
        (
            "echo",
            lambda data: data["results"][0].update(range_r=9.0),
            "result tuples",
        ),
    ],
)
def test_archive_refuses_non_development_or_incomplete_inputs(
    tmp_path: Path,
    target: str,
    mutation,
    match: str,
) -> None:
    archive = _archive_module()
    calibration = _calibration_module()
    multi_path, echo_path = _write_valid_inputs(tmp_path)
    path = multi_path if target == "multi" else echo_path
    data = calibration.load_study(path)
    mutation(data)
    calibration.write_study(path, data)

    with pytest.raises(ValueError, match=match):
        archive.archive_development_studies(
            multi_path=multi_path,
            echo_path=echo_path,
            output_dir=tmp_path / "tracked",
        )
    assert not (tmp_path / "tracked").exists()
