"""Preregistered populations and identifiability for echolocation reliability."""

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from clocks._scenarios import (
    ECHO_ESS_TARGET,
    ECHO_N_OBSERVATIONS,
    ECHO_N_PARTICLES,
    ECHO_NOISE_STD,
    ECHO_PASS_MASS_TOL,
    ECHO_PASS_POS_TOL,
    ECHO_PROPOSAL_SCALE,
    ECHO_R_HEAD,
    ECHO_REJUVENATION_STEPS,
    _validate_echolocation_truth,
    build_head_lattice,
    contrast_matrix,
)
from clocks._validation import finite_float, finite_float_array, real_float_array
from clocks.physics import (
    PhysicsDomainError,
    compute_distances,
    gravitational_potential,
    time_dilation_factor,
)
from clocks.types import ClockArray

PARAMETER_NAMES = ("angular_1", "angular_2", "log_range", "log_mass")

MANIFEST_SCHEMA_VERSION = 1
RELIABILITY_STUDY_VERSION = 1
RELIABILITY_MASTER_SEED = 20260817
RELIABILITY_N_STRATA = 6
RELIABILITY_CASES_PER_STRATUM = 64
RELIABILITY_RANGE_R_BOUNDS = (2.0, 8.0)
RELIABILITY_MASS_BOUNDS = (0.02, 0.08)
# Exact schema-v1 geomspace results, frozen so archived evidence never depends on
# a future NumPy implementation or runtime rounding behavior.
RELIABILITY_RANGE_STRATUM_EDGES = (
    float.fromhex("0x1.0000000000000p+1"),
    float.fromhex("0x1.428a2f98d728bp+1"),
    float.fromhex("0x1.965fea53d6e3dp+1"),
    float.fromhex("0x1.ffffffffffffep+1"),
    float.fromhex("0x1.428a2f98d728bp+2"),
    float.fromhex("0x1.965fea53d6e3ap+2"),
    float.fromhex("0x1.0000000000000p+3"),
)

_RELIABILITY_CASE_COUNT = RELIABILITY_N_STRATA * RELIABILITY_CASES_PER_STRATUM
_RELIABILITY_STUDY = "echolocation_population"
_RELEASE_IDENTITY = "echolocation_population_v1_release"
_RELEASE_STATUS = "release"
_GENERATOR_METADATA = {
    "bit_generator": "PCG64",
    "generator": "numpy.random.Generator",
    "parameter_draw_recipe": (
        "normal(3), retry exact zero norm; "
        "exp(uniform(log(stratum_lower), log(stratum_upper))); "
        "exp(uniform(log(0.02), log(0.08)))"
    ),
    "seed_sequence": "numpy.random.SeedSequence",
    "spawn_policy": (
        "root.spawn(385): analysis then stratum-major cases; "
        "case.spawn(3): parameter, observation, inference"
    ),
}
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "study_version",
    "study",
    "identity",
    "status",
    "master_seed",
    "analysis_seed",
    "generator",
    "head",
    "population",
    "controls",
    "acceptance",
    "intervals",
    "cases",
    "semantic_sha256",
}
_GENERATOR_FIELDS = set(_GENERATOR_METADATA)
_HEAD_FIELDS = {"clock_count", "geometry", "positions", "r_head", "track_offset"}
_POPULATION_FIELDS = {"n_cases", "case_order", "range_r", "mass", "direction"}
_RANGE_FIELDS = {
    "distribution",
    "bounds",
    "n_strata",
    "cases_per_stratum",
    "stratum_edges",
    "stratum_allocation",
    "stratum_boundary_policy",
}
_MASS_FIELDS = {"distribution", "bounds"}
_DIRECTION_FIELDS = {"distribution", "dimension"}
_CONTROL_FIELDS = {
    "n_observations",
    "noise_std",
    "n_particles",
    "ess_target",
    "rejuvenation_steps",
    "proposal_scale",
}
_ACCEPTANCE_FIELDS = {
    "position_error_max",
    "mass_error_max",
    "threshold_policy",
}
_INTERVAL_FIELDS = {"strata", "overall"}
_STRATUM_INTERVAL_FIELDS = {"method", "count", "confidence_level"}
_OVERALL_INTERVAL_FIELDS = {
    "method",
    "resamples",
    "quantile_method",
    "confidence_level",
}
_CASE_FIELDS = {
    "case_id",
    "case_index",
    "stratum_index",
    "stratum_case_index",
    "direction",
    "range_r",
    "mass",
    "position",
    "parameter_seed",
    "observation_seed",
    "inference_seed",
}
_SEMANTIC_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")


class _JSONObjectPairs(list[tuple[str, object]]):
    """Distinguish decoded JSON objects from arrays until duplicates are checked."""


def _range_stratum_edges() -> tuple[float, ...]:
    return RELIABILITY_RANGE_STRATUM_EDGES


def _seed_from_sequence(sequence: np.random.SeedSequence) -> int:
    """Materialize a portable integer token for one spawned seed stream."""
    words = sequence.generate_state(4, dtype=np.uint32)
    return sum(int(word) << (32 * index) for index, word in enumerate(words))


def _plain(value: object) -> object:
    """Copy immutable manifest containers into strict JSON-native values."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        _plain(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _semantic_sha256(manifest: Mapping[str, object]) -> str:
    payload = {
        key: value for key, value in manifest.items() if key != "semantic_sha256"
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _deep_freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _materialize_json(value: object, path: str) -> object:
    if isinstance(value, _JSONObjectPairs):
        result: dict[str, object] = {}
        for key, item in value:
            field_path = f"{path}.{key}"
            if key in result:
                raise ValueError(f"{field_path} is a duplicate JSON field")
            result[key] = _materialize_json(item, field_path)
        return result
    if isinstance(value, list):
        return [
            _materialize_json(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    return value


def _decode_manifest_json(encoded: str) -> object:
    try:
        paired = json.loads(encoded, object_pairs_hook=_JSONObjectPairs)
    except json.JSONDecodeError as error:
        raise ValueError(f"manifest contains invalid JSON: {error.msg}") from error
    return _materialize_json(paired, "manifest")


def _object(
    value: object, path: str, expected_fields: set[str]
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    non_string = [key for key in value if not isinstance(key, str)]
    if non_string:
        raise ValueError(f"{path} has non-string fields: {non_string!r}")
    actual_fields = set(value)
    missing = expected_fields - actual_fields
    if missing:
        raise ValueError(f"{path} is missing fields: {sorted(missing)}")
    unknown = actual_fields - expected_fields
    if unknown:
        raise ValueError(f"{path} has unknown fields: {sorted(unknown)}")
    return value


def _sequence(
    value: object, path: str, length: int
) -> list[object] | tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{path} must be an array of length {length}")
    if len(value) != length:
        raise ValueError(f"{path} must contain exactly {length} entries")
    return value


def _strict_integer(value: object, path: str, *, nonnegative: bool = False) -> int:
    if type(value) is not int:
        qualifier = "nonnegative " if nonnegative else ""
        raise ValueError(f"{path} must be a canonical {qualifier}integer")
    result = value
    if nonnegative and result < 0:
        raise ValueError(f"{path} must be a canonical nonnegative integer")
    return result


def _strict_number(value: object, path: str) -> float:
    if type(value) is not float:
        raise ValueError(f"{path} must be a canonical finite float")
    result = value
    if not math.isfinite(result):
        raise ValueError(f"{path} must be a canonical finite float")
    return result


def _same_float(left: float, right: float) -> bool:
    return left.hex() == right.hex()


def _strict_string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a string")
    return value


def _expect_integer(value: object, expected: int, path: str) -> int:
    result = _strict_integer(value, path)
    if result != expected:
        raise ValueError(f"{path} must equal {expected}")
    return result


def _expect_number(value: object, expected: float, path: str) -> float:
    result = _strict_number(value, path)
    if not _same_float(result, expected):
        raise ValueError(f"{path} must equal {expected}")
    return result


def _expect_string(value: object, expected: str, path: str) -> str:
    result = _strict_string(value, path)
    if result != expected:
        raise ValueError(f"{path} must equal {expected!r}")
    return result


def _strict_vector(value: object, path: str, length: int) -> tuple[float, ...]:
    entries = _sequence(value, path, length)
    return tuple(
        _strict_number(entry, f"{path}[{index}]") for index, entry in enumerate(entries)
    )


def _expect_vector(
    value: object, expected: tuple[float, ...], path: str
) -> tuple[float, ...]:
    result = _strict_vector(value, path, len(expected))
    if not all(
        _same_float(actual, expected_value)
        for actual, expected_value in zip(result, expected)
    ):
        raise ValueError(f"{path} does not match the preregistered values")
    return result


def _case_id(stratum_index: int, stratum_case_index: int) -> str:
    return (
        "echolocation-population-v1-release-"
        f"s{stratum_index:02d}-c{stratum_case_index:03d}"
    )


def generate_release_manifest() -> Mapping[str, object]:
    """Generate the deterministic, realized schema-v1 release population."""
    root = np.random.SeedSequence(RELIABILITY_MASTER_SEED)
    root_children = root.spawn(_RELIABILITY_CASE_COUNT + 1)
    analysis_seed = _seed_from_sequence(root_children[0])
    edges = _range_stratum_edges()
    cases: list[dict[str, object]] = []
    stream_seeds = {analysis_seed}

    for case_index, case_child in enumerate(root_children[1:]):
        stratum_index, stratum_case_index = divmod(
            case_index, RELIABILITY_CASES_PER_STRATUM
        )
        parameter_child, observation_child, inference_child = case_child.spawn(3)
        parameter_seed = _seed_from_sequence(parameter_child)
        observation_seed = _seed_from_sequence(observation_child)
        inference_seed = _seed_from_sequence(inference_child)
        new_seeds = {parameter_seed, observation_seed, inference_seed}
        if len(new_seeds) != 3 or stream_seeds.intersection(new_seeds):
            raise RuntimeError("spawned reliability stream seeds must be unique")
        stream_seeds.update(new_seeds)

        rng = np.random.Generator(np.random.PCG64(parameter_seed))
        direction_draw = rng.normal(size=3)
        direction_norm = math.hypot(*(float(component) for component in direction_draw))
        while direction_norm == 0.0:
            direction_draw = rng.normal(size=3)
            direction_norm = math.hypot(
                *(float(component) for component in direction_draw)
            )
        direction = np.asarray(direction_draw / direction_norm, dtype=np.float64)
        range_r = math.exp(
            float(
                rng.uniform(
                    math.log(edges[stratum_index]),
                    math.log(edges[stratum_index + 1]),
                )
            )
        )
        mass = math.exp(
            float(
                rng.uniform(
                    math.log(RELIABILITY_MASS_BOUNDS[0]),
                    math.log(RELIABILITY_MASS_BOUNDS[1]),
                )
            )
        )
        position = direction * range_r * ECHO_R_HEAD
        cases.append(
            {
                "case_id": _case_id(stratum_index, stratum_case_index),
                "case_index": case_index,
                "stratum_index": stratum_index,
                "stratum_case_index": stratum_case_index,
                "direction": direction.tolist(),
                "range_r": range_r,
                "mass": mass,
                "position": position.tolist(),
                "parameter_seed": parameter_seed,
                "observation_seed": observation_seed,
                "inference_seed": inference_seed,
            }
        )

    head = build_head_lattice()
    manifest: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "study_version": RELIABILITY_STUDY_VERSION,
        "study": _RELIABILITY_STUDY,
        "identity": _RELEASE_IDENTITY,
        "status": _RELEASE_STATUS,
        "master_seed": RELIABILITY_MASTER_SEED,
        "analysis_seed": analysis_seed,
        "generator": dict(_GENERATOR_METADATA),
        "head": {
            "clock_count": 27,
            "geometry": "3x3x3_cubic_lattice",
            "positions": head.positions.tolist(),
            "r_head": ECHO_R_HEAD,
            "track_offset": head.track_offset,
        },
        "population": {
            "n_cases": _RELIABILITY_CASE_COUNT,
            "case_order": "stratum_major_then_case",
            "range_r": {
                "distribution": "log_uniform_stratified",
                "bounds": list(RELIABILITY_RANGE_R_BOUNDS),
                "n_strata": RELIABILITY_N_STRATA,
                "cases_per_stratum": RELIABILITY_CASES_PER_STRATUM,
                "stratum_edges": list(edges),
                "stratum_allocation": "equal",
                "stratum_boundary_policy": ("left_closed_right_open_final_closed"),
            },
            "mass": {
                "distribution": "log_uniform",
                "bounds": list(RELIABILITY_MASS_BOUNDS),
            },
            "direction": {
                "distribution": "uniform_sphere",
                "dimension": 3,
            },
        },
        "controls": {
            "n_observations": ECHO_N_OBSERVATIONS,
            "noise_std": ECHO_NOISE_STD,
            "n_particles": ECHO_N_PARTICLES,
            "ess_target": ECHO_ESS_TARGET,
            "rejuvenation_steps": ECHO_REJUVENATION_STEPS,
            "proposal_scale": ECHO_PROPOSAL_SCALE,
        },
        "acceptance": {
            "position_error_max": ECHO_PASS_POS_TOL,
            "mass_error_max": ECHO_PASS_MASS_TOL,
            "threshold_policy": "inclusive",
        },
        "intervals": {
            "strata": {
                "method": "wilson",
                "count": RELIABILITY_N_STRATA,
                "confidence_level": 0.95,
            },
            "overall": {
                "method": "stratified_bootstrap",
                "resamples": 10_000,
                "quantile_method": "linear",
                "confidence_level": 0.95,
            },
        },
        "cases": cases,
    }
    manifest["semantic_sha256"] = _semantic_sha256(manifest)
    validate_manifest(manifest)
    return _deep_freeze(manifest)  # type: ignore[return-value]


def validate_manifest(manifest: object) -> None:
    """Strictly validate all release-manifest semantics and its payload hash."""
    document = _object(manifest, "manifest", _TOP_LEVEL_FIELDS)
    _expect_integer(
        document["schema_version"], MANIFEST_SCHEMA_VERSION, "schema_version"
    )
    _expect_integer(
        document["study_version"], RELIABILITY_STUDY_VERSION, "study_version"
    )
    _expect_string(document["study"], _RELIABILITY_STUDY, "study")
    _expect_string(document["identity"], _RELEASE_IDENTITY, "identity")
    _expect_string(document["status"], _RELEASE_STATUS, "status")
    master_seed = _strict_integer(
        document["master_seed"], "master_seed", nonnegative=True
    )
    if master_seed != RELIABILITY_MASTER_SEED:
        raise ValueError(f"master_seed must equal {RELIABILITY_MASTER_SEED}")
    analysis_seed = _strict_integer(
        document["analysis_seed"], "analysis_seed", nonnegative=True
    )
    if analysis_seed == master_seed:
        raise ValueError("analysis_seed must be distinct from master_seed")

    generator = _object(document["generator"], "generator", _GENERATOR_FIELDS)
    for field, expected in _GENERATOR_METADATA.items():
        _expect_string(generator[field], expected, f"generator.{field}")

    expected_head = build_head_lattice()
    head = _object(document["head"], "head", _HEAD_FIELDS)
    _expect_integer(head["clock_count"], 27, "head.clock_count")
    _expect_string(head["geometry"], "3x3x3_cubic_lattice", "head.geometry")
    positions = _sequence(head["positions"], "head.positions", 27)
    expected_positions = expected_head.positions.tolist()
    for index, (position, expected) in enumerate(zip(positions, expected_positions)):
        _expect_vector(
            position,
            tuple(float(component) for component in expected),
            f"head.positions[{index}]",
        )
    _expect_number(head["r_head"], ECHO_R_HEAD, "head.r_head")
    _expect_number(head["track_offset"], 0.0, "head.track_offset")

    population = _object(document["population"], "population", _POPULATION_FIELDS)
    _expect_integer(
        population["n_cases"], _RELIABILITY_CASE_COUNT, "population.n_cases"
    )
    _expect_string(
        population["case_order"],
        "stratum_major_then_case",
        "population.case_order",
    )
    range_spec = _object(population["range_r"], "population.range_r", _RANGE_FIELDS)
    _expect_string(
        range_spec["distribution"],
        "log_uniform_stratified",
        "population.range_r.distribution",
    )
    _expect_vector(
        range_spec["bounds"],
        RELIABILITY_RANGE_R_BOUNDS,
        "population.range_r.bounds",
    )
    _expect_integer(
        range_spec["n_strata"],
        RELIABILITY_N_STRATA,
        "population.range_r.n_strata",
    )
    _expect_integer(
        range_spec["cases_per_stratum"],
        RELIABILITY_CASES_PER_STRATUM,
        "population.range_r.cases_per_stratum",
    )
    edges = _expect_vector(
        range_spec["stratum_edges"],
        _range_stratum_edges(),
        "population.range_r.stratum_edges",
    )
    _expect_string(
        range_spec["stratum_allocation"],
        "equal",
        "population.range_r.stratum_allocation",
    )
    _expect_string(
        range_spec["stratum_boundary_policy"],
        "left_closed_right_open_final_closed",
        "population.range_r.stratum_boundary_policy",
    )
    mass_spec = _object(population["mass"], "population.mass", _MASS_FIELDS)
    _expect_string(
        mass_spec["distribution"], "log_uniform", "population.mass.distribution"
    )
    _expect_vector(
        mass_spec["bounds"], RELIABILITY_MASS_BOUNDS, "population.mass.bounds"
    )
    direction_spec = _object(
        population["direction"], "population.direction", _DIRECTION_FIELDS
    )
    _expect_string(
        direction_spec["distribution"],
        "uniform_sphere",
        "population.direction.distribution",
    )
    _expect_integer(direction_spec["dimension"], 3, "population.direction.dimension")

    controls = _object(document["controls"], "controls", _CONTROL_FIELDS)
    _expect_integer(
        controls["n_observations"], ECHO_N_OBSERVATIONS, "controls.n_observations"
    )
    _expect_number(controls["noise_std"], ECHO_NOISE_STD, "controls.noise_std")
    _expect_integer(controls["n_particles"], ECHO_N_PARTICLES, "controls.n_particles")
    _expect_number(controls["ess_target"], ECHO_ESS_TARGET, "controls.ess_target")
    _expect_integer(
        controls["rejuvenation_steps"],
        ECHO_REJUVENATION_STEPS,
        "controls.rejuvenation_steps",
    )
    _expect_number(
        controls["proposal_scale"],
        ECHO_PROPOSAL_SCALE,
        "controls.proposal_scale",
    )

    acceptance = _object(document["acceptance"], "acceptance", _ACCEPTANCE_FIELDS)
    _expect_number(
        acceptance["position_error_max"],
        ECHO_PASS_POS_TOL,
        "acceptance.position_error_max",
    )
    _expect_number(
        acceptance["mass_error_max"],
        ECHO_PASS_MASS_TOL,
        "acceptance.mass_error_max",
    )
    _expect_string(
        acceptance["threshold_policy"],
        "inclusive",
        "acceptance.threshold_policy",
    )

    intervals = _object(document["intervals"], "intervals", _INTERVAL_FIELDS)
    stratum_intervals = _object(
        intervals["strata"], "intervals.strata", _STRATUM_INTERVAL_FIELDS
    )
    _expect_string(stratum_intervals["method"], "wilson", "intervals.strata.method")
    _expect_integer(
        stratum_intervals["count"], RELIABILITY_N_STRATA, "intervals.strata.count"
    )
    _expect_number(
        stratum_intervals["confidence_level"],
        0.95,
        "intervals.strata.confidence_level",
    )
    overall_interval = _object(
        intervals["overall"], "intervals.overall", _OVERALL_INTERVAL_FIELDS
    )
    _expect_string(
        overall_interval["method"],
        "stratified_bootstrap",
        "intervals.overall.method",
    )
    _expect_integer(
        overall_interval["resamples"], 10_000, "intervals.overall.resamples"
    )
    _expect_string(
        overall_interval["quantile_method"],
        "linear",
        "intervals.overall.quantile_method",
    )
    _expect_number(
        overall_interval["confidence_level"],
        0.95,
        "intervals.overall.confidence_level",
    )

    cases = _sequence(document["cases"], "cases", _RELIABILITY_CASE_COUNT)
    all_stream_seeds = {analysis_seed}
    for case_index, case_value in enumerate(cases):
        path = f"cases[{case_index}]"
        case = _object(case_value, path, _CASE_FIELDS)
        stratum_index, stratum_case_index = divmod(
            case_index, RELIABILITY_CASES_PER_STRATUM
        )
        _expect_string(
            case["case_id"],
            _case_id(stratum_index, stratum_case_index),
            f"{path}.case_id",
        )
        _expect_integer(case["case_index"], case_index, f"{path}.case_index")
        _expect_integer(case["stratum_index"], stratum_index, f"{path}.stratum_index")
        _expect_integer(
            case["stratum_case_index"],
            stratum_case_index,
            f"{path}.stratum_case_index",
        )
        direction = _strict_vector(case["direction"], f"{path}.direction", 3)
        direction_norm = math.hypot(*direction)
        if not math.isclose(direction_norm, 1.0, rel_tol=0.0, abs_tol=2e-15):
            raise ValueError(f"{path}.direction must be a unit vector")
        range_r = _strict_number(case["range_r"], f"{path}.range_r")
        lower = edges[stratum_index]
        upper = edges[stratum_index + 1]
        in_stratum = lower <= range_r < upper
        if stratum_index == RELIABILITY_N_STRATA - 1:
            in_stratum = lower <= range_r <= upper
        if not in_stratum:
            raise ValueError(f"{path}.range_r is outside its declared stratum")
        mass = _strict_number(case["mass"], f"{path}.mass")
        if not RELIABILITY_MASS_BOUNDS[0] <= mass <= RELIABILITY_MASS_BOUNDS[1]:
            raise ValueError(f"{path}.mass is outside the preregistered bounds")
        position = _strict_vector(case["position"], f"{path}.position", 3)
        expected_position = tuple(
            component * range_r * ECHO_R_HEAD for component in direction
        )
        if not all(
            _same_float(actual, expected)
            for actual, expected in zip(position, expected_position)
        ):
            raise ValueError(
                f"{path}.position must equal direction * range_r * head.r_head"
            )
        try:
            _validate_echolocation_truth(
                np.asarray(position, dtype=np.float64), mass, expected_head
            )
        except ValueError as error:
            raise ValueError(f"{path}.physical_support is invalid: {error}") from error

        for seed_name in ("parameter_seed", "observation_seed", "inference_seed"):
            seed = _strict_integer(
                case[seed_name], f"{path}.{seed_name}", nonnegative=True
            )
            if seed in all_stream_seeds:
                raise ValueError(f"duplicate stream seed at {path}.{seed_name}")
            all_stream_seeds.add(seed)

    semantic_hash = _strict_string(document["semantic_sha256"], "semantic_sha256")
    if _SEMANTIC_HASH_PATTERN.fullmatch(semantic_hash) is None:
        raise ValueError("semantic_sha256 must be 64 lowercase hexadecimal digits")
    expected_hash = _semantic_sha256(document)
    if semantic_hash != expected_hash:
        raise ValueError("semantic_sha256 does not match the canonical payload")


def encode_manifest(manifest: object) -> str:
    """Return deterministic UTF-8-compatible canonical JSON with one newline."""
    validate_manifest(manifest)
    return _canonical_json(manifest) + "\n"


def load_manifest(path: Path) -> Mapping[str, object]:
    """Load exact canonical UTF-8, rejecting duplicate keys before validation."""
    raw = Path(path).read_bytes()
    try:
        encoded = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("manifest must be encoded as canonical UTF-8") from error
    value = _decode_manifest_json(encoded)
    validate_manifest(value)
    canonical = encode_manifest(value).encode("utf-8")
    if raw != canonical:
        raise ValueError(
            "manifest JSON must use canonical UTF-8 bytes: sorted compact keys, "
            "canonical numeric forms, and exactly one trailing newline"
        )
    return _deep_freeze(value)  # type: ignore[return-value]


def write_manifest(path: Path, manifest: object) -> None:
    """Atomically write a validated canonical manifest in its destination directory."""
    encoded = encode_manifest(manifest).encode("utf-8")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class IdentifiabilityResult:
    """Immutable local sensitivity diagnostics in dimensionless coordinates."""

    jacobian: NDArray[np.float64]
    scaled_jacobian: NDArray[np.float64]
    fisher_information: NDArray[np.float64]
    singular_values: NDArray[np.float64]
    rank: int
    rank_tolerance: float
    condition_number: float | None
    crlb_std: NDArray[np.float64] | None
    weakest_direction: NDArray[np.float64]
    weakest_mode_loadings: Mapping[str, float]

    parameter_names: ClassVar[tuple[str, str, str, str]] = PARAMETER_NAMES

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "jacobian", finite_float_array("jacobian", self.jacobian, ndim=2)
        )
        object.__setattr__(
            self,
            "scaled_jacobian",
            finite_float_array("scaled_jacobian", self.scaled_jacobian, ndim=2),
        )
        object.__setattr__(
            self,
            "fisher_information",
            finite_float_array("fisher_information", self.fisher_information, ndim=2),
        )
        object.__setattr__(
            self,
            "singular_values",
            finite_float_array("singular_values", self.singular_values, ndim=1),
        )
        if self.crlb_std is not None:
            object.__setattr__(
                self,
                "crlb_std",
                finite_float_array("crlb_std", self.crlb_std, ndim=1),
            )
        object.__setattr__(
            self,
            "weakest_direction",
            finite_float_array("weakest_direction", self.weakest_direction, ndim=1),
        )
        object.__setattr__(
            self,
            "weakest_mode_loadings",
            MappingProxyType(
                {
                    str(name): finite_float(f"loading {name}", value)
                    for name, value in self.weakest_mode_loadings.items()
                }
            ),
        )


def _position_vector(position: object) -> NDArray[np.float64]:
    """Return a finite, nonzero three-dimensional position vector."""
    vector = real_float_array("position", position)
    if vector.shape != (3,):
        raise ValueError(f"position must have shape (3,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("position must contain only finite values")
    if not np.any(vector):
        raise ValueError("position must be nonzero")
    return vector


def _positive_mass(mass: object) -> float:
    if isinstance(mass, (bool, np.bool_)):
        raise ValueError("mass must be a real-valued positive number")
    value = finite_float("mass", mass)
    if value <= 0.0:
        raise ValueError("mass must be positive")
    return value


def tangent_basis(position: NDArray[np.floating]) -> NDArray[np.float64]:
    """Return a deterministic orthonormal basis tangent to a radial direction."""
    vector = _position_vector(position)
    norm = math.hypot(*(float(component) for component in vector))
    if not math.isfinite(norm):
        raise ValueError("position norm must be finite")
    direction = vector / norm

    reference = np.zeros(3, dtype=np.float64)
    reference[int(np.argmin(np.abs(direction)))] = 1.0
    first = reference - np.dot(reference, direction) * direction
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    return np.column_stack((first, second))


def contrast_jacobian(
    position: NDArray[np.floating],
    mass: float,
    clock_array: ClockArray,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return exact position and mass derivatives of orthonormal contrasts."""
    vector = _position_vector(position)
    mass_value = _positive_mass(mass)
    if clock_array.positions.shape[1] != 3:
        raise ValueError(
            "position and clock positions must have matching spatial dimensions"
        )

    distances = compute_distances(
        clock_array.positions, vector.reshape(1, 3), clock_array.track_offset
    )[:, 0]
    masses = np.array([mass_value], dtype=np.float64)
    potential = gravitational_potential(distances[:, np.newaxis], masses)
    rates = time_dilation_factor(potential)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        differences = clock_array.positions - vector
        directions = differences / distances[:, np.newaxis]
        mass_over_rate_distance = (mass_value / distances) / rates
        position_derivatives = (
            -mass_over_rate_distance[:, np.newaxis]
            * directions
            / distances[:, np.newaxis]
        )
        mass_derivatives = -(1.0 / distances) / rates
    if not np.all(np.isfinite(position_derivatives)) or not np.all(
        np.isfinite(mass_derivatives)
    ):
        raise PhysicsDomainError("clock-rate derivatives must be finite")

    contrasts = contrast_matrix(len(clock_array.positions))
    with np.errstate(invalid="ignore", over="ignore"):
        contrast_position = contrasts @ position_derivatives
        contrast_mass = contrasts @ mass_derivatives
    if not np.all(np.isfinite(contrast_position)) or not np.all(
        np.isfinite(contrast_mass)
    ):
        raise PhysicsDomainError("contrast derivatives must be finite")
    return contrast_position, contrast_mass


def _validated_tangent_basis(
    position: NDArray[np.float64], basis: object | None
) -> NDArray[np.float64]:
    if basis is None:
        return tangent_basis(position)
    vectors = real_float_array("basis", basis)
    if vectors.shape != (3, 2):
        raise ValueError(f"basis must have shape (3, 2), got {vectors.shape}")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("basis must contain only finite values")
    direction = position / math.hypot(*(float(value) for value in position))
    if not np.allclose(vectors.T @ vectors, np.eye(2), rtol=1e-12, atol=1e-12):
        raise ValueError("basis must be orthonormal")
    if not np.allclose(direction @ vectors, np.zeros(2), rtol=0.0, atol=1e-12):
        raise ValueError("basis must be perpendicular to position")
    return vectors


def _dimensionless_jacobian(
    position: object,
    mass: object,
    clock_array: ClockArray,
    *,
    basis: object | None = None,
) -> NDArray[np.float64]:
    """Return contrast derivatives in angular/log-range/log-mass coordinates."""
    vector = _position_vector(position)
    mass_value = _positive_mass(mass)
    vectors = _validated_tangent_basis(vector, basis)
    position_jacobian, mass_jacobian = contrast_jacobian(
        vector, mass_value, clock_array
    )
    radius = math.hypot(*(float(component) for component in vector))
    with np.errstate(invalid="ignore", over="ignore"):
        jacobian = np.column_stack(
            (
                position_jacobian @ (radius * vectors[:, 0]),
                position_jacobian @ (radius * vectors[:, 1]),
                position_jacobian @ vector,
                mass_jacobian * mass_value,
            )
        )
    if not np.all(np.isfinite(jacobian)):
        raise PhysicsDomainError("dimensionless contrast derivatives must be finite")
    return jacobian


def _positive_noise(noise_std: object) -> float:
    if isinstance(noise_std, (bool, np.bool_)):
        raise ValueError("noise_std must be a real-valued positive number")
    value = finite_float("noise_std", noise_std)
    if value <= 0.0:
        raise ValueError("noise_std must be positive")
    return value


def _positive_count(n_observations: object) -> int:
    if isinstance(n_observations, (bool, np.bool_)) or not isinstance(
        n_observations, Integral
    ):
        raise ValueError("n_observations must be a positive non-bool integer")
    value = int(n_observations)
    if value <= 0:
        raise ValueError("n_observations must be a positive non-bool integer")
    return value


def _stable_fisher_information(
    scaled_jacobian: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return a representable Gram matrix without premature underflow."""
    scale = float(np.max(np.abs(scaled_jacobian)))
    if scale == 0.0:
        return np.zeros((scaled_jacobian.shape[1], scaled_jacobian.shape[1]))
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        normalized_jacobian = scaled_jacobian / scale
        normalized_fisher = normalized_jacobian.T @ normalized_jacobian
        fisher_information = (normalized_fisher * scale) * scale
    if not np.any(fisher_information):
        raise PhysicsDomainError(
            "nonzero Fisher information is below the float64 representable range"
        )
    if not np.all(np.isfinite(fisher_information)):
        raise PhysicsDomainError("Fisher information must be finite")
    if not np.array_equal(fisher_information, fisher_information.T):
        raise PhysicsDomainError("Fisher information must be symmetric")
    information_scale = float(np.max(np.abs(fisher_information)))
    normalized_information = fisher_information / information_scale
    minimum_eigenvalue = float(np.linalg.eigvalsh(normalized_information)[0])
    operation_scale = max(
        1.0, float(np.max(np.sum(np.abs(normalized_information), axis=1)))
    )
    psd_tolerance = (
        np.finfo(np.float64).eps * max(normalized_information.shape) * operation_scale
    )
    if minimum_eigenvalue < -psd_tolerance:
        raise PhysicsDomainError("Fisher information is not representably PSD")
    return fisher_information


def local_identifiability(
    position: NDArray[np.floating],
    mass: float,
    clock_array: ClockArray,
    *,
    n_observations: int,
    noise_std: float,
) -> IdentifiabilityResult:
    """Compute exact local identifiability for one exterior point mass."""
    count = _positive_count(n_observations)
    noise = _positive_noise(noise_std)
    jacobian = _dimensionless_jacobian(position, mass, clock_array)
    try:
        scale = math.sqrt(count) / noise
    except OverflowError as error:
        raise ValueError("n_observations is too large") from error
    if not math.isfinite(scale):
        raise ValueError("sqrt(n_observations) / noise_std must be finite")
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        scaled_jacobian = scale * jacobian
    if not np.all(np.isfinite(scaled_jacobian)):
        raise PhysicsDomainError("scaled contrast derivatives must be finite")
    if not np.any(scaled_jacobian) and np.any(jacobian):
        raise PhysicsDomainError("scaled contrast derivatives underflowed completely")

    full_matrices = scaled_jacobian.shape[0] < scaled_jacobian.shape[1]
    _, singular_values, right_vectors = np.linalg.svd(
        scaled_jacobian, full_matrices=full_matrices
    )
    if singular_values.size < len(PARAMETER_NAMES):
        singular_values = np.append(
            singular_values,
            np.zeros(len(PARAMETER_NAMES) - singular_values.size),
        )
    if not np.all(np.isfinite(singular_values)) or np.any(singular_values < 0.0):
        raise PhysicsDomainError("singular values must be finite and nonnegative")
    largest = float(singular_values[0])
    rank_tolerance = np.finfo(np.float64).eps * max(scaled_jacobian.shape) * largest
    rank = int(np.count_nonzero(singular_values > rank_tolerance))

    condition_number: float | None = None
    crlb_std: NDArray[np.float64] | None = None
    if rank == len(PARAMETER_NAMES):
        condition_number = largest / float(singular_values[-1])
        crlb_std = np.array(
            [
                math.hypot(
                    *(
                        float(right_vectors[row, column]) / float(singular_values[row])
                        for row in range(len(PARAMETER_NAMES))
                    )
                )
                for column in range(len(PARAMETER_NAMES))
            ]
        )
        if not np.all(np.isfinite(crlb_std)):
            raise PhysicsDomainError("CRLB standard deviations must be finite")

    fisher_information = _stable_fisher_information(scaled_jacobian)

    weakest_direction = right_vectors[-1]
    squared = np.square(weakest_direction)
    squared /= np.sum(squared)
    weakest_mode_loadings = {
        "angular": float(np.sum(squared[:2])),
        "log_range": float(squared[2]),
        "log_mass": float(squared[3]),
    }
    return IdentifiabilityResult(
        jacobian=jacobian,
        scaled_jacobian=scaled_jacobian,
        fisher_information=fisher_information,
        singular_values=singular_values,
        rank=rank,
        rank_tolerance=float(rank_tolerance),
        condition_number=condition_number,
        crlb_std=crlb_std,
        weakest_direction=weakest_direction,
        weakest_mode_loadings=weakest_mode_loadings,
    )
