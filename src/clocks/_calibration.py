"""Private deterministic JSON format for calibration scan evidence."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
DEVELOPMENT_ESS_TARGETS = (0.7, 0.8, 0.9)
DEVELOPMENT_REJUVENATION_STEPS = (1, 2, 4)
DEVELOPMENT_PROPOSAL_SCALES = (1.5, 2.38, 3.0)

_RESULT_ORDER = {
    "multi_mass_2d": (
        "ess_target",
        "rejuvenation_steps",
        "proposal_scale",
        "seed",
    ),
    "echolocation_range": (
        "ess_target",
        "rejuvenation_steps",
        "proposal_scale",
        "range_r",
        "seed",
    ),
}

_MULTI_RESULT_FIELDS = {
    "seed",
    "passed",
    "mean",
    "std",
    "max_abs_error",
    "covered_3sigma",
    "max_posterior_std",
    "residual_over_noise",
    "normalized_error",
    "forward_model_evaluations",
    "ess_target",
    "rejuvenation_steps",
    "proposal_scale",
}
_ECHO_RESULT_FIELDS = {
    "seed",
    "range_r",
    "passed",
    "mean",
    "std",
    "position_error",
    "mass_error",
    "pos_std",
    "mass_std",
    "covered_3sigma",
    "residual_over_noise",
    "normalized_error",
    "forward_model_evaluations",
    "ess_target",
    "rejuvenation_steps",
    "proposal_scale",
}
_COMPARISON_REL_TOL = 1e-12
_COMPARISON_ABS_TOL = 1e-12


def control_grid_from_cells(
    cells: Sequence[tuple[float, int, float]],
) -> dict[str, list[float | int]]:
    """Return sorted control axes for a Cartesian scan cell collection."""
    return {
        "ess_target": sorted({cell[0] for cell in cells}),
        "rejuvenation_steps": sorted({cell[1] for cell in cells}),
        "proposal_scale": sorted({cell[2] for cell in cells}),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    return value


def build_study_document(
    *,
    study: str,
    seed_block: int,
    seeds: Sequence[int],
    control_grid: Mapping[str, Sequence[float | int]],
    tolerances: Mapping[str, object],
    results: Sequence[Mapping[str, object]],
    ranges: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Build schema-v1 data with results in a reproducible semantic order."""
    if study not in _RESULT_ORDER:
        raise ValueError(f"unknown calibration study: {study!r}")
    if isinstance(seed_block, bool) or not isinstance(seed_block, int):
        raise ValueError("seed_block must be a non-bool integer")
    order = _RESULT_ORDER[study]
    serializable_results = [_jsonable(result) for result in results]
    serializable_results.sort(key=lambda result: tuple(result[key] for key in order))
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study": study,
        "seed_block": seed_block,
        "seed_role": "development" if seed_block == 0 else "protected",
        "seeds": _jsonable(seeds),
        "control_grid": _jsonable(control_grid),
        "tolerances": _jsonable(tolerances),
        "results": serializable_results,
    }
    if ranges is not None:
        document["ranges"] = _jsonable(ranges)
    return document


def encode_study(study: Mapping[str, object]) -> str:
    """Encode byte-stable strict JSON without touching the filesystem."""
    return (
        json.dumps(_jsonable(study), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def write_study(path: Path, study: Mapping[str, object]) -> None:
    """Atomically publish byte-stable JSON after strict encoding succeeds."""
    encoded = encode_study(study)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def load_study(path: Path) -> dict[str, Any]:
    """Load a raw or archived calibration JSON document."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("calibration study must be a JSON object")
    return value


def _strict_number(record: Mapping[str, object], key: str) -> float:
    value = record.get(key)
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"result {key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"result {key} must be finite")
    return result


def _strict_integer(record: Mapping[str, object], key: str) -> int:
    value = record.get(key)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"result {key} must be an integer")
    return int(value)


def _strict_bool(record: Mapping[str, object], key: str) -> bool:
    value = record.get(key)
    if type(value) is not bool:
        raise ValueError(f"result {key} must be a boolean")
    return value


def _strict_vector(record: Mapping[str, object], key: str, length: int) -> list[float]:
    value = record.get(key)
    if isinstance(value, np.ndarray):
        if value.shape != (length,):
            raise ValueError(f"result {key} must have length {length}")
        items = value.tolist()
    elif isinstance(value, (list, tuple)):
        if len(value) != length:
            raise ValueError(f"result {key} must have length {length}")
        items = list(value)
    else:
        raise ValueError(f"result {key} must be an array")
    numbers = []
    for item in items:
        if isinstance(item, (bool, np.bool_)) or not isinstance(
            item, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"result {key} entries must be numeric")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"result {key} entries must be finite")
        numbers.append(number)
    return numbers


def _require_close(actual: float, expected: float, key: str) -> None:
    if not math.isclose(
        actual,
        expected,
        rel_tol=_COMPARISON_REL_TOL,
        abs_tol=_COMPARISON_ABS_TOL,
    ):
        raise ValueError(f"result {key} does not match its derived value")


def _require_result_fields(
    result: Mapping[str, object], required: set[str], study: str
) -> None:
    missing = required - result.keys()
    if missing:
        raise ValueError(f"{study} result is missing fields: {sorted(missing)}")


def validate_multi_results(
    results: Sequence[Mapping[str, object]],
    *,
    expected_tuples: set[tuple[float, int, float, int]],
) -> None:
    """Validate exact multi scan membership and independently derived metrics."""
    from clocks._scenarios import PASS_TOLERANCE, TRUTH

    truth = [float(value) for value in TRUTH]
    tolerances = [float(value) for value in PASS_TOLERANCE]
    actual_tuples = {
        (
            _strict_number(result, "ess_target"),
            _strict_integer(result, "rejuvenation_steps"),
            _strict_number(result, "proposal_scale"),
            _strict_integer(result, "seed"),
        )
        for result in results
    }
    if actual_tuples != expected_tuples or len(actual_tuples) != len(results):
        raise ValueError("multi result tuples are mixed, missing, or duplicated")
    for result in results:
        _require_result_fields(result, _MULTI_RESULT_FIELDS, "multi")

        mean = _strict_vector(result, "mean", 6)
        std = _strict_vector(result, "std", 6)
        if any(value < 0.0 for value in std):
            raise ValueError("result std entries must be nonnegative")
        errors = [abs(estimate - expected) for estimate, expected in zip(mean, truth)]
        expected_pass = all(
            error <= tolerance for error, tolerance in zip(errors, tolerances)
        )
        if _strict_bool(result, "passed") is not expected_pass:
            raise ValueError("result passed does not match its derived value")
        expected_coverage = all(
            error <= 3.0 * uncertainty for error, uncertainty in zip(errors, std)
        )
        if _strict_bool(result, "covered_3sigma") is not expected_coverage:
            raise ValueError("result covered_3sigma does not match its derived value")
        _require_close(
            _strict_number(result, "max_abs_error"), max(errors), "max_abs_error"
        )
        _require_close(
            _strict_number(result, "max_posterior_std"),
            max(std),
            "max_posterior_std",
        )
        _require_close(
            _strict_number(result, "normalized_error"),
            sum(error / tolerance for error, tolerance in zip(errors, tolerances))
            / len(errors),
            "normalized_error",
        )
        residual = _strict_number(result, "residual_over_noise")
        if residual < 0.0:
            raise ValueError("result residual_over_noise must be nonnegative")
        evaluations = _strict_integer(result, "forward_model_evaluations")
        if evaluations <= 0:
            raise ValueError("result forward_model_evaluations must be positive")


def validate_echo_results(
    results: Sequence[Mapping[str, object]],
    *,
    expected_tuples: set[tuple[float, int, float, float, int]],
) -> None:
    """Validate exact echo scan membership and independently derived metrics."""
    from clocks._scenarios import (
        ECHO_DIRECTION,
        ECHO_M_TRUE,
        ECHO_PASS_MASS_TOL,
        ECHO_PASS_POS_TOL,
        ECHO_R_HEAD,
    )

    direction = [float(value) for value in ECHO_DIRECTION]
    actual_tuples = {
        (
            _strict_number(result, "ess_target"),
            _strict_integer(result, "rejuvenation_steps"),
            _strict_number(result, "proposal_scale"),
            _strict_number(result, "range_r"),
            _strict_integer(result, "seed"),
        )
        for result in results
    }
    if actual_tuples != expected_tuples or len(actual_tuples) != len(results):
        raise ValueError("echo result tuples are mixed, missing, or duplicated")
    for result in results:
        _require_result_fields(result, _ECHO_RESULT_FIELDS, "echo")
        range_r = _strict_number(result, "range_r")

        mean = _strict_vector(result, "mean", 4)
        std = _strict_vector(result, "std", 4)
        if any(value < 0.0 for value in std):
            raise ValueError("result std entries must be nonnegative")
        truth = [
            component * range_r * float(ECHO_R_HEAD) for component in direction
        ] + [float(ECHO_M_TRUE)]
        errors = [abs(estimate - expected) for estimate, expected in zip(mean, truth)]
        position_error = math.sqrt(
            sum((mean[index] - truth[index]) ** 2 for index in range(3))
        )
        mass_error = errors[3]
        pos_std = math.sqrt(sum(value**2 for value in std[:3]))
        mass_std = std[3]
        _require_close(
            _strict_number(result, "position_error"),
            position_error,
            "position_error",
        )
        _require_close(_strict_number(result, "mass_error"), mass_error, "mass_error")
        _require_close(_strict_number(result, "pos_std"), pos_std, "pos_std")
        _require_close(_strict_number(result, "mass_std"), mass_std, "mass_std")
        expected_pass = position_error <= float(
            ECHO_PASS_POS_TOL
        ) and mass_error <= float(ECHO_PASS_MASS_TOL)
        if _strict_bool(result, "passed") is not expected_pass:
            raise ValueError("result passed does not match its derived value")
        expected_coverage = all(
            error <= 3.0 * uncertainty for error, uncertainty in zip(errors, std)
        )
        if _strict_bool(result, "covered_3sigma") is not expected_coverage:
            raise ValueError("result covered_3sigma does not match its derived value")
        _require_close(
            _strict_number(result, "normalized_error"),
            (
                position_error / float(ECHO_PASS_POS_TOL)
                + mass_error / float(ECHO_PASS_MASS_TOL)
            )
            / 2.0,
            "normalized_error",
        )
        residual = _strict_number(result, "residual_over_noise")
        if residual < 0.0:
            raise ValueError("result residual_over_noise must be nonnegative")
        evaluations = _strict_integer(result, "forward_model_evaluations")
        if evaluations <= 0:
            raise ValueError("result forward_model_evaluations must be positive")
