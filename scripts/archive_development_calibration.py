"""Validate and archive complete development or certification evidence."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from hashlib import sha256
from itertools import product
from pathlib import Path
from typing import Any

from clocks._calibration import (
    DEVELOPMENT_ESS_TARGETS,
    DEVELOPMENT_PROPOSAL_SCALES,
    DEVELOPMENT_REJUVENATION_STEPS,
    build_study_document,
    encode_study,
    load_study,
    validate_echo_results,
    validate_multi_results,
    write_study,
)
from clocks._scenarios import (
    ECHO_FAR_STD_FACTOR,
    ECHO_PASS_MASS_TOL,
    ECHO_PASS_POS_TOL,
    ECHO_SWEEP_RANGES,
    PASS_TOLERANCE,
)

DEVELOPMENT_SEEDS = tuple(range(12))
DEVELOPMENT_CONTROL_GRID = {
    "ess_target": list(DEVELOPMENT_ESS_TARGETS),
    "rejuvenation_steps": list(DEVELOPMENT_REJUVENATION_STEPS),
    "proposal_scale": list(DEVELOPMENT_PROPOSAL_SCALES),
}
MULTI_RESULT_COUNT = 27 * len(DEVELOPMENT_SEEDS)
ECHO_RESULT_COUNT = 27 * len(ECHO_SWEEP_RANGES) * len(DEVELOPMENT_SEEDS)
CERTIFICATION_SEEDS = (400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411)
MULTI_CERTIFICATION_CONTROL_GRID = {
    "ess_target": [0.7],
    "rejuvenation_steps": [2],
    "proposal_scale": [3.0],
}
ECHO_CERTIFICATION_CONTROL_GRID = {
    "ess_target": [0.9],
    "rejuvenation_steps": [1],
    "proposal_scale": [1.5],
}
MULTI_CERTIFICATION_RESULT_COUNT = len(CERTIFICATION_SEEDS)
ECHO_CERTIFICATION_RESULT_COUNT = len(ECHO_SWEEP_RANGES) * len(CERTIFICATION_SEEDS)

MULTI_TOLERANCES = {"absolute_parameter_error": PASS_TOLERANCE.tolist()}
ECHO_TOLERANCES = {
    "position_error_max": ECHO_PASS_POS_TOL,
    "mass_error_max": ECHO_PASS_MASS_TOL,
    "far_std_factor_min": ECHO_FAR_STD_FACTOR,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _exact_json_value(actual: object, expected: object) -> bool:
    """Compare JSON values without Python's bool/int numeric equivalence."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        return actual.keys() == expected.keys() and all(
            _exact_json_value(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        assert isinstance(actual, list)
        return len(actual) == len(expected) and all(
            _exact_json_value(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def _require_schema_metadata(
    document: Mapping[str, object],
    *,
    study: str,
    tolerances: Mapping[str, object],
    ranges: Sequence[float] | None = None,
) -> None:
    schema_version = document.get("schema_version")
    _require(
        type(schema_version) is int and schema_version == 1,
        "schema_version must be integer 1",
    )
    _require(document.get("study") == study, f"study must be {study!r}")
    seed_block = document.get("seed_block")
    _require(
        type(seed_block) is int and seed_block == 0,
        "archive requires integer seed_block=0",
    )
    _require(
        document.get("seed_role") == "development", "seed role must be development"
    )
    seeds = document.get("seeds")
    _require(
        seeds == list(DEVELOPMENT_SEEDS)
        and isinstance(seeds, list)
        and all(type(seed) is int for seed in seeds),
        "seeds must be exactly the integers 0-11",
    )
    _require(
        _exact_json_value(document.get("control_grid"), DEVELOPMENT_CONTROL_GRID),
        "control_grid must be the complete declared 27-cell development grid",
    )
    _require(
        _exact_json_value(document.get("tolerances"), dict(tolerances)),
        "tolerances do not match frozen gates",
    )
    if ranges is None:
        _require("ranges" not in document, "multi-mass study must not declare ranges")
    else:
        _require(
            _exact_json_value(document.get("ranges"), list(ranges)),
            "ranges do not match the declared sweep",
        )


def _results(
    document: Mapping[str, object], expected_count: int
) -> list[dict[str, Any]]:
    results = document.get("results")
    _require(isinstance(results, list), "results must be a JSON array")
    assert isinstance(results, list)
    _require(
        len(results) == expected_count, f"expected exactly {expected_count} results"
    )
    _require(
        all(isinstance(result, dict) for result in results),
        "every result must be an object",
    )
    return results


def _validate_multi_results(results: list[dict[str, Any]]) -> None:
    expected = set(
        product(
            DEVELOPMENT_ESS_TARGETS,
            DEVELOPMENT_REJUVENATION_STEPS,
            DEVELOPMENT_PROPOSAL_SCALES,
            DEVELOPMENT_SEEDS,
        )
    )
    validate_multi_results(results, expected_tuples=expected)


def _validate_echo_results(results: list[dict[str, Any]]) -> None:
    expected = set(
        product(
            DEVELOPMENT_ESS_TARGETS,
            DEVELOPMENT_REJUVENATION_STEPS,
            DEVELOPMENT_PROPOSAL_SCALES,
            ECHO_SWEEP_RANGES,
            DEVELOPMENT_SEEDS,
        )
    )
    validate_echo_results(results, expected_tuples=expected)


def _source_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _validate_certification_document(
    document: Mapping[str, object], *, study: str
) -> list[dict[str, Any]]:
    """Require the exact one-shot block and independently validate every row."""
    _require(
        type(document.get("schema_version")) is int
        and document.get("schema_version") == 1,
        "schema_version must be integer 1",
    )
    _require(document.get("study") == study, f"study must be {study!r}")
    _require(
        type(document.get("seed_block")) is int and document.get("seed_block") == 400,
        "certification archive requires integer seed_block=400",
    )
    _require(document.get("seed_role") == "protected", "seed role must be protected")
    seeds = document.get("seeds")
    _require(
        seeds == list(CERTIFICATION_SEEDS)
        and isinstance(seeds, list)
        and all(type(seed) is int for seed in seeds),
        "certification seeds must be exactly the integers 400-411",
    )
    if study == "multi_mass_2d":
        _require(
            _exact_json_value(
                document.get("control_grid"), MULTI_CERTIFICATION_CONTROL_GRID
            ),
            "control_grid must be the frozen multi certification cell",
        )
        _require(
            _exact_json_value(document.get("tolerances"), MULTI_TOLERANCES),
            "tolerances do not match frozen gates",
        )
        _require("ranges" not in document, "multi-mass study must not declare ranges")
        results = _results(document, MULTI_CERTIFICATION_RESULT_COUNT)
        expected = set(
            product(
                (0.7,),
                (2,),
                (3.0,),
                CERTIFICATION_SEEDS,
            )
        )
        validate_multi_results(results, expected_tuples=expected)
        return results
    if study == "echolocation_range":
        _require(
            _exact_json_value(
                document.get("control_grid"), ECHO_CERTIFICATION_CONTROL_GRID
            ),
            "control_grid must be the frozen echo certification cell",
        )
        _require(
            _exact_json_value(document.get("tolerances"), ECHO_TOLERANCES),
            "tolerances do not match frozen gates",
        )
        _require(
            _exact_json_value(document.get("ranges"), list(ECHO_SWEEP_RANGES)),
            "ranges do not match the declared sweep",
        )
        results = _results(document, ECHO_CERTIFICATION_RESULT_COUNT)
        expected = set(
            product(
                (0.9,),
                (1,),
                (1.5,),
                ECHO_SWEEP_RANGES,
                CERTIFICATION_SEEDS,
            )
        )
        validate_echo_results(results, expected_tuples=expected)
        return results
    raise ValueError(f"unknown certification study: {study!r}")


def _canonical_certification(path: Path, *, study: str) -> dict[str, Any]:
    document = load_study(path)
    results = _validate_certification_document(document, study=study)
    if study == "multi_mass_2d":
        canonical = build_study_document(
            study=study,
            seed_block=400,
            seeds=CERTIFICATION_SEEDS,
            control_grid=MULTI_CERTIFICATION_CONTROL_GRID,
            tolerances=MULTI_TOLERANCES,
            results=results,
        )
    else:
        canonical = build_study_document(
            study=study,
            seed_block=400,
            seeds=CERTIFICATION_SEEDS,
            control_grid=ECHO_CERTIFICATION_CONTROL_GRID,
            tolerances=ECHO_TOLERANCES,
            ranges=ECHO_SWEEP_RANGES,
            results=results,
        )
    canonical["source"] = {"format": "schema_v1", "sha256": _source_hash(path)}
    return canonical


def _canonical_multi(path: Path) -> dict[str, Any]:
    document = load_study(path)
    _require_schema_metadata(
        document, study="multi_mass_2d", tolerances=MULTI_TOLERANCES
    )
    results = _results(document, MULTI_RESULT_COUNT)
    _validate_multi_results(results)
    canonical = build_study_document(
        study="multi_mass_2d",
        seed_block=0,
        seeds=DEVELOPMENT_SEEDS,
        control_grid=DEVELOPMENT_CONTROL_GRID,
        tolerances=MULTI_TOLERANCES,
        results=results,
    )
    canonical["source"] = {"format": "schema_v1", "sha256": _source_hash(path)}
    return canonical


def _canonical_echo(path: Path) -> dict[str, Any]:
    source = load_study(path)
    if "schema_version" in source:
        document = source
        source_format = "schema_v1"
    else:
        seed_block = source.get("seed_block")
        _require(
            type(seed_block) is int and seed_block == 0,
            "archive requires integer seed_block=0",
        )
        raw_results = source.get("results")
        _require(isinstance(raw_results, list), "legacy echo results must be an array")
        document = build_study_document(
            study="echolocation_range",
            seed_block=0,
            seeds=DEVELOPMENT_SEEDS,
            control_grid=DEVELOPMENT_CONTROL_GRID,
            tolerances=ECHO_TOLERANCES,
            ranges=ECHO_SWEEP_RANGES,
            results=raw_results,
        )
        source_format = "legacy_seed_block_results"
    _require_schema_metadata(
        document,
        study="echolocation_range",
        tolerances=ECHO_TOLERANCES,
        ranges=ECHO_SWEEP_RANGES,
    )
    results = _results(document, ECHO_RESULT_COUNT)
    _validate_echo_results(results)
    canonical = build_study_document(
        study="echolocation_range",
        seed_block=0,
        seeds=DEVELOPMENT_SEEDS,
        control_grid=DEVELOPMENT_CONTROL_GRID,
        tolerances=ECHO_TOLERANCES,
        ranges=ECHO_SWEEP_RANGES,
        results=results,
    )
    canonical["source"] = {"format": source_format, "sha256": _source_hash(path)}
    return canonical


def archive_development_studies(
    *, multi_path: Path, echo_path: Path, output_dir: Path
) -> tuple[Path, Path]:
    """Validate both complete dev grids before writing either tracked artifact."""
    multi = _canonical_multi(multi_path)
    echo = _canonical_echo(echo_path)
    # Validate strict JSON for both before either tracked path is created.
    encode_study(multi)
    encode_study(echo)
    multi_output = output_dir / "multi_mass_2d_development.json"
    echo_output = output_dir / "echolocation_range_development.json"
    write_study(multi_output, multi)
    write_study(echo_output, echo)
    return multi_output, echo_output


def archive_certification_studies(
    *, multi_path: Path, echo_path: Path, output_dir: Path
) -> tuple[Path, Path]:
    """Archive only the exact frozen one-shot certification block."""
    multi = _canonical_certification(multi_path, study="multi_mass_2d")
    echo = _canonical_certification(echo_path, study="echolocation_range")
    encode_study(multi)
    encode_study(echo)
    multi_output = output_dir / "multi_mass_2d_certification.json"
    echo_output = output_dir / "echolocation_range_certification.json"
    write_study(multi_output, multi)
    write_study(echo_output, echo)
    return multi_output, echo_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--certification",
        action="store_true",
        help="archive exact schema-v1 seed-block-400 evidence",
    )
    parser.add_argument(
        "--multi-input",
        type=Path,
    )
    parser.add_argument(
        "--echo-input",
        type=Path,
        help="schema-v1 block-specific output or the fully validated legacy dev JSON",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/calibration"))
    args = parser.parse_args()
    if args.certification:
        multi_path = args.multi_input or Path(
            "output/multi_mass_2d_study_seed_block_400.json"
        )
        echo_path = args.echo_input or Path(
            "output/echolocation_range_study_seed_block_400.json"
        )
        written = archive_certification_studies(
            multi_path=multi_path,
            echo_path=echo_path,
            output_dir=args.output_dir,
        )
    else:
        multi_path = args.multi_input or Path(
            "output/multi_mass_2d_study_seed_block_0.json"
        )
        echo_path = args.echo_input or Path("output/echolocation_range_study.json")
        written = archive_development_studies(
            multi_path=multi_path,
            echo_path=echo_path,
            output_dir=args.output_dir,
        )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
