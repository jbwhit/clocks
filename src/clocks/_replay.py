"""Replay archives: what a packaged demo run produced, kept in a ``.npz`` file.

An archive holds a JSON ``meta`` string and a fixed set of named arrays, so a
figure or animation can be drawn from the file instead of pinning image bytes.
``load_replay`` and ``save_replay`` validate everything they touch; a file that
loads is shaped exactly as the format says.
"""

import io
import json
import math
import os
import platform
import tempfile
import zipfile
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import scipy
from numpy.typing import NDArray

from clocks._notes import attempt, left_behind
from clocks.api import build_particle_filter, simulate
from clocks.config import InferenceConfig, NoiseConfig, PriorConfig, SimulationConfig
from clocks.types import ClockArray, MassConfig, Observation

FORMAT = "clocks-replay"
VERSION = 1
CLOUD_SIZE = 500  # most particles ever displayed
THINNING_OFFSET = (math.sqrt(5.0) - 1.0) / 2.0  # φ: where the first pick falls

DEMOS: tuple[str, ...] = (
    "demo_1d",
    "demo_2d",
    "demo_multi_mass",
    "demo_multi_mass_2d",
    "demo_model_comparison",
    "demo_density",
    "demo_echolocation_3d",
)

SPATIAL_DIMENSIONS: dict[str, int] = {
    "demo_1d": 1,
    "demo_2d": 2,
    "demo_multi_mass": 1,
    "demo_multi_mass_2d": 2,
    "demo_model_comparison": 1,
    "demo_density": 1,
    "demo_echolocation_3d": 3,
}

# Each demo's parameters, in the column order of mean, cov and cloud.
PARAM_NAMES: dict[str, tuple[str, ...]] = {
    "demo_1d": ("x", "M"),
    "demo_2d": ("x", "y", "M"),
    "demo_multi_mass": ("x₁", "x₂", "M₁", "M₂"),
    "demo_multi_mass_2d": ("x₁", "y₁", "x₂", "y₂", "M₁", "M₂"),
    "demo_model_comparison": ("x₁", "x₂", "M₁", "M₂"),
    "demo_density": ("μ", "σ", "A"),
    "demo_echolocation_3d": ("x", "y", "z", "M"),
}

# Echolocation's readings are contrasts between clocks, which need two.
_MINIMUM_CLOCKS = {"demo_echolocation_3d": 2}


@dataclass(frozen=True, eq=False)  # eq=False: comparing dicts of arrays raises
class Replay:
    """The decoded ``meta`` object and the named arrays of one archive."""

    meta: Mapping[str, object]
    arrays: Mapping[str, NDArray]


def validate_replay(replay: Replay, source: str = "<memory>") -> None:
    """Raise ``ValueError`` unless ``replay`` is exactly what the format says."""
    meta = replay.meta
    _check_meta(meta, source)
    arrays = replay.arrays
    ks = _check_clocks_and_candidates(meta, arrays, source)
    specs = _array_specs(meta, arrays, ks)
    _check_key_set(specs, arrays, source)
    for key, (shape, dtype) in specs.items():
        _check_array(arrays[key], key, shape, dtype, source)
    if meta["kind"] == "model_comparison":
        _check_true_model(meta, arrays, ks, source)
    _check_finite(arrays, source)


def save_replay(path: Path, replay: Replay) -> None:
    """Validate ``replay`` and write it to ``path``, which must end in ``.npz``.

    ``path`` is replaced whole or left untouched: the archive is written to a
    temporary file beside it and moved into place once complete, so a failed
    write (a full disk, say) cannot truncate an existing archive.
    """
    _write_archive(path, replay, os.replace)


def create_replay(path: Path, replay: Replay) -> None:
    """Like ``save_replay``, but never replaces: ``FileExistsError`` if anything
    (a file, a link, even a dangling one) is already at ``path``.

    The finished temporary file is hard-linked into place, which fails
    atomically when any directory entry exists.
    """
    _write_archive(path, replay, _link_and_forget)


def _link_and_forget(temporary: Path, path: Path) -> None:
    """Link ``temporary`` to ``path``, then remove the temporary name.

    After a link the temporary is a second name for the finished file, so this
    is the one success path on which it still exists and must be removed. Its
    already being gone (another process removed it) is not a failure: the
    archive is published.
    """
    os.link(temporary, path)
    temporary.unlink(missing_ok=True)


def _write_archive(
    path: Path, replay: Replay, publish: Callable[[Path, Path], None]
) -> None:
    """Write ``replay`` to a temporary file beside ``path``, then ``publish`` it.

    ``publish(temporary, path)`` puts the finished file in place and leaves no
    temporary name behind (``os.replace`` moves it; ``_link_and_forget`` removes
    it). The temporary is removed here only on a failure, before or during
    publication, when it can still exist; after success it is gone, and
    unlinking the name again could remove another process's entry.
    """
    validate_replay(replay, str(path))
    if path.suffix != ".npz":
        # np.savez_compressed would silently append the suffix and write elsewhere.
        raise ValueError(f"{path}: path: expected a .npz suffix, got {path.suffix!r}")
    text = json.dumps(dict(replay.meta), ensure_ascii=False, sort_keys=True)
    handle = tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:  # a file object, so NumPy appends no suffix of its own
            np.savez_compressed(handle, meta=np.array(text), **replay.arrays)
        publish(temporary, path)
    except BaseException as error:
        # The write or the publishing failed: remove the temporary file, and if
        # that fails too, keep the original error and say what may be left.
        attempt(
            error,
            lambda: temporary.unlink(missing_ok=True),
            lambda failure: left_behind(temporary, failure),
        )
        raise


def load_replay(path: Path, data: bytes | None = None) -> Replay:
    """Read and validate the archive at ``path``.

    With ``data``, the archive is read from those bytes instead, and ``path``
    only names it in errors: a caller that hashed ``data`` then knows which
    replay it holds, whatever happens to the file meanwhile.
    """
    meta_text, arrays = _read_members(path, data)
    meta = _decode_meta(meta_text, str(path))
    replay = Replay(meta, arrays)
    validate_replay(replay, str(path))
    return replay


class Difference(NamedTuple):
    """Where two replays first differ.

    ``row`` is the first index along axis 0 outside tolerance, or ``None`` when
    the difference is not in a row (meta, key sets, shape, dtype). ``largest`` is
    the largest absolute difference in the array, or NaN when there is no
    numeric difference to measure.
    """

    key: str
    row: int | None
    largest: float


# The report's tolerances (the gate's tighter ones for inputs live in the tests).
_FLOAT_TOLERANCE = (1e-9, 1e-12)  # rtol, atol
_CLOUD_TOLERANCE = (1e-5, 1e-6)
_INPUT_ORDER = (
    "clock_positions",
    "true_params",
    "true_rates",
    "observations",
    "filter_observations",
    "model_ks",
    "model_prior",
)


def first_difference(reference: Replay, other: Replay) -> Difference | None:
    """Compare two replays in full; name the first difference, or return None.

    Order: ``meta`` (without ``created_with``), the key sets, then the arrays,
    inputs first, then the rest alphabetically. Does not validate: a shape or
    dtype mismatch is a difference to report.
    """
    field = _first_meta_difference(reference.meta, other.meta)
    if field is not None:
        return Difference(f"meta.{field}", None, math.nan)
    if set(reference.arrays) != set(other.arrays):
        return Difference("keys", None, math.nan)
    for key in _comparison_order(reference.arrays):
        found = _array_difference(key, reference.arrays[key], other.arrays[key])
        if found is not None:
            return found
    return None


def _first_meta_difference(
    reference: Mapping[str, object], other: Mapping[str, object]
) -> str | None:
    """The first field, in sorted order, whose value differs or one side lacks."""
    for field in sorted((set(reference) | set(other)) - {"created_with"}):
        if field not in reference or field not in other:
            return field
        if reference[field] != other[field]:
            return field
    return None


def _comparison_order(arrays: Mapping[str, NDArray]) -> list[str]:
    inputs = [key for key in _INPUT_ORDER if key in arrays]
    return inputs + sorted(set(arrays) - set(inputs))


def _array_difference(key: str, ref: NDArray, other: NDArray) -> Difference | None:
    if ref.shape != other.shape or ref.dtype != other.dtype:
        return Difference(key, None, math.nan)
    if ref.size == 0:
        return None
    if ref.dtype.kind == "f":
        gap = np.abs(other.astype(np.float64) - ref.astype(np.float64))
        rtol, atol = _CLOUD_TOLERANCE if key == "cloud" else _FLOAT_TOLERANCE
        # `not <=` so a NaN gap counts as a difference
        outside = ~(gap <= atol + rtol * np.abs(ref.astype(np.float64)))
    else:
        outside = other != ref
    if not outside.any():
        return None
    if ref.dtype.kind == "f":
        largest = float(np.max(gap))
    else:
        # In Python integers: as float64, 2**53 and 2**53 + 1 are one value.
        pairs = zip(ref[outside].tolist(), other[outside].tolist())
        largest = float(max(abs(a - b) for a, b in pairs))
    rows = outside.reshape(len(outside), -1).any(axis=1)
    return Difference(key, int(np.argmax(rows)), largest)


def thin(particles: NDArray, weights: NDArray, size: int) -> NDArray[np.float32]:
    """Pick ``min(size, N)`` particles to display, deterministically.

    Targets are evenly spaced quantiles of the normalised cumulative weight,
    offset by φ so no target lands on a step exactly; no random numbers. A
    target that does equal a cumulative step picks the left side: the
    particle whose cumulative weight ends there.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if np.any(weights < 0) or not weights.sum() > 0:
        raise ValueError(
            f"weights must be nonnegative with a positive sum, got {weights.tolist()}"
        )
    count = min(size, len(weights))
    cumulative = np.cumsum(weights)
    cumulative = cumulative / cumulative[-1]
    targets = (np.arange(count) + THINNING_OFFSET) / count
    indices = np.searchsorted(cumulative, targets, side="left")
    return np.asarray(particles)[indices].astype(np.float32)


def record_filter(
    pf,
    observations: Sequence[Observation],
    *,
    meta: Mapping[str, object],
    arrays: Mapping[str, NDArray],
    predict: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
) -> Replay:
    """Run ``pf`` over ``observations`` and record its history as a replay.

    Reads only ``state.observations_seen``, ``n_particles``, ``update``,
    ``history``, ``diagnostics_history`` and ``log_evidence_history``, and draws
    nothing itself. ``predict`` is applied to each row's particles, never to a
    mean. The result is not validated; the caller adds its arrays and does.
    """
    seen = pf.state.observations_seen
    if seen != 0:
        raise RuntimeError(
            f"record_filter needs a fresh filter; this one has seen {seen} observations"
        )
    for observation in observations:
        pf.update(observation)
    history = pf.history
    diagnostics = pf.diagnostics_history
    shown = min(CLOUD_SIZE, pf.n_particles)

    out: dict[str, NDArray] = dict(arrays)
    moments = [_moments(state.particles, state.weights) for state in history]
    out["mean"] = np.array([m for m, _ in moments], dtype=np.float64)
    out["cov"] = np.array([c for _, c in moments], dtype=np.float64)
    out["ess"] = np.array(
        [float(pf.n_particles)]
        + [1.0 / float(np.sum(s.weights**2)) for s in history[1:]],
        dtype=np.float64,
    )
    out["log_evidence"] = np.array([0.0, *pf.log_evidence_history], dtype=np.float64)
    for key in ("tempering_stages", "mh_proposals", "mh_acceptances"):
        out[key] = np.array([getattr(d, key) for d in diagnostics], dtype=np.int64)
    out["cloud"] = np.array(
        [thin(s.particles, s.weights, shown) for s in history], dtype=np.float32
    )
    if "observations" not in out:
        out["observations"] = np.array(
            [o.rates for o in observations], dtype=np.float64
        )
    if predict is not None:
        predicted = [
            _moments(np.asarray(predict(s.particles)), s.weights) for s in history
        ]
        out["predicted_mean"] = np.array([m for m, _ in predicted], dtype=np.float64)
        out["predicted_std"] = np.sqrt(
            np.array([np.diag(c) for _, c in predicted], dtype=np.float64)
        )
    return Replay(meta, out)


def _moments(
    values: NDArray, weights: NDArray
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Weighted mean and covariance (divisor: the weights' sum)."""
    w = np.asarray(weights, dtype=np.float64)
    x = np.asarray(values, dtype=np.float64)
    total = w.sum()
    mean = (w[:, None] * x).sum(axis=0) / total
    centred = x - mean
    cov = (w[:, None, None] * centred[:, :, None] * centred[:, None, :]).sum(
        axis=0
    ) / total
    return mean, cov


def provenance() -> dict[str, str]:
    """Versions of what produced an archive; recorded, never compared."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    }


@dataclass(frozen=True, eq=False)  # eq=False: it holds arrays
class Scenario:
    """A demo's packaged inputs, as data: what ``record`` simulates and infers."""

    demo: str
    clock_array: ClockArray
    true_params: NDArray[np.float64]
    true_rates: NDArray[np.float64]
    param_names: tuple[str, ...]
    prior_ranges: tuple[tuple[float, float], ...]
    seed: int
    noise_std: float
    n_observations: int
    n_particles: int


def scenario_meta(
    scenario: Scenario,
    *,
    kind: str,
    n_observations: int,
    n_particles: int,
    controls: Mapping[str, object],
    **extra: object,
) -> dict[str, object]:
    """The complete ``meta`` of a replay, in plain JSON-round-trippable values."""
    meta: dict[str, object] = {
        "format": FORMAT,
        "version": VERSION,
        "demo": scenario.demo,
        "kind": kind,
        "seed": int(scenario.seed),
        "n_observations": int(n_observations),
        "n_particles": int(n_particles),
        "noise_std": float(scenario.noise_std),
        "track_offset": float(scenario.clock_array.track_offset),
        "param_names": list(scenario.param_names),
        "prior_ranges": [[float(lo), float(hi)] for lo, hi in scenario.prior_ranges],
        "controls": dict(controls),
        "created_with": provenance(),
    }
    if kind == "filter":
        meta["cloud_size"] = min(CLOUD_SIZE, int(n_particles))
    meta.update(extra)
    return meta


def record_point_masses(
    scenario: Scenario,
    *,
    n_observations: int,
    n_particles: int,
    predictions: bool,
) -> Replay:
    """Simulate and infer a point-mass scenario once; return a validated replay.

    The mass count and truth come from ``scenario.true_params``, laid out as the
    library lays out parameters: positions row-major, then masses. The library's
    prior has one position range and one mass range, so every position range
    must equal the first and every mass range the first mass range.
    """
    dimensions = scenario.clock_array.positions.shape[1]
    params = scenario.true_params
    n_masses = len(params) // (dimensions + 1)
    split = n_masses * dimensions
    _check_shared_ranges(scenario, split)
    truth = MassConfig(params[:split].reshape(n_masses, dimensions), params[split:])
    simulation = simulate(
        SimulationConfig(
            clock_array=scenario.clock_array,
            ground_truth=truth,
            noise=NoiseConfig(scenario.noise_std),
            n_observations=n_observations,
            seed=scenario.seed,
        )
    )
    particle_filter = build_particle_filter(
        InferenceConfig(
            clock_array=scenario.clock_array,
            noise=NoiseConfig(scenario.noise_std),
            prior=PriorConfig(scenario.prior_ranges[0], scenario.prior_ranges[split]),
            n_particles=n_particles,
            n_masses=n_masses,
            seed=scenario.seed,
        )
    )
    replay = record_filter(
        particle_filter,
        simulation.observations,
        meta=scenario_meta(
            scenario,
            kind="filter",
            n_observations=n_observations,
            n_particles=n_particles,
            controls=filter_controls(particle_filter),
        ),
        arrays={
            "clock_positions": scenario.clock_array.positions,
            "true_params": scenario.true_params,
            "true_rates": scenario.true_rates,
        },
        predict=particle_filter.forward_model_batch if predictions else None,
    )
    validate_replay(replay)
    return replay


def _check_shared_ranges(scenario: Scenario, split: int) -> None:
    """Raise unless the prior the filter is built with is the one ``meta`` records."""
    ranges = [tuple(pair) for pair in scenario.prior_ranges]
    positions, masses = ranges[:split], ranges[split:]
    if any(pair != positions[0] for pair in positions) or any(
        pair != masses[0] for pair in masses
    ):
        raise ValueError(
            f"{scenario.demo}: prior_ranges: expected one range shared by every "
            f"position and one by every mass, got {ranges}"
        )


def filter_controls(pf) -> dict[str, object]:
    """The tuning controls a particle filter was built with, as plain values."""
    return {
        "ess_target": float(pf.ess_target),
        "rejuvenation_steps": int(pf.rejuvenation_steps),
        "proposal_scale": float(pf.proposal_scale),
        "resampling": str(pf.resampling),
    }


# --- reading -----------------------------------------------------------------

# zlib.error: a damaged compressed member fails while it is decompressed.
_OPEN_ERRORS = (ValueError, OSError, EOFError, zipfile.BadZipFile, zlib.error)


def _read_members(
    path: Path, data: bytes | None = None
) -> tuple[NDArray, dict[str, NDArray]]:
    """Return the ``meta`` member and every array, with the file closed.

    The archive is read from ``data`` when given, else from ``path``.
    """
    source = path if data is None else io.BytesIO(data)
    try:
        archive = np.load(source, allow_pickle=False)
    except _OPEN_ERRORS as error:
        raise ValueError(f"{path}: not a replay archive: {error}") from error
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError(f"{path}: not a replay archive: expected an .npz archive")
    with archive:
        _check_unique_members(archive.files, path)
        if "meta" not in archive.files:
            raise ValueError(f"{path}: meta: expected a member, got missing")
        members: dict[str, NDArray] = {}
        for key in archive.files:
            members[key] = _read_member(archive, key, path)
    meta_text = members.pop("meta")
    return meta_text, members


def _check_unique_members(keys: Sequence[str], path: Path) -> None:
    """Raise if a key appears twice: a reader would see only one of the two."""
    seen: set[str] = set()
    for key in keys:
        if key in seen:
            raise ValueError(f"{path}: {key}: expected one member, got a duplicate")
        seen.add(key)


def _read_member(archive: np.lib.npyio.NpzFile, key: str, path: Path) -> NDArray:
    try:
        member = archive[key]
    except _OPEN_ERRORS as error:
        raise ValueError(f"{path}: {key}: unreadable: {error}") from error
    if not isinstance(member, np.ndarray):
        # NpzFile returns the raw bytes of a member that is not an .npy array.
        raise ValueError(
            f"{path}: {key}: expected an .npy array, got {type(member).__name__}"
        )
    return member


def _decode_meta(stored: NDArray, source: str) -> dict[str, object]:
    if stored.ndim != 0 or stored.dtype.kind != "U":
        raise ValueError(
            f"{source}: meta: expected one string, got "
            f"{stored.dtype} array of shape {stored.shape}"
        )
    try:
        meta = json.loads(stored.item())
    except json.JSONDecodeError as error:
        raise ValueError(f"{source}: meta: expected JSON, got {error}") from error
    if not isinstance(meta, dict):
        raise ValueError(
            f"{source}: meta: expected a JSON object, got {type(meta).__name__}"
        )
    return meta


# --- validation: meta --------------------------------------------------------

_COMMON_META_KEYS = frozenset(
    {
        "format",
        "version",
        "demo",
        "kind",
        "seed",
        "n_observations",
        "n_particles",
        "noise_std",
        "track_offset",
        "param_names",
        "prior_ranges",
        "controls",
        "created_with",
    }
)
_KIND_META_KEYS = {
    "filter": frozenset({"cloud_size"}),
    "model_comparison": frozenset({"true_k"}),
}


def _fail(source: str, key: str, expected: str, got: object) -> None:
    raise ValueError(f"{source}: {key}: expected {expected}, got {got}")


def _check_meta(meta: Mapping[str, object], source: str) -> None:
    kind = meta.get("kind")
    if not isinstance(kind, str) or kind not in _KIND_META_KEYS:
        _fail(source, "meta.kind", "'filter' or 'model_comparison'", repr(kind))
    _check_meta_key_set(meta, _COMMON_META_KEYS | _KIND_META_KEYS[kind], source)
    if meta["format"] != FORMAT:
        _fail(source, "meta.format", repr(FORMAT), repr(meta["format"]))
    if not _is_int(meta["version"]) or meta["version"] != VERSION:
        _fail(source, "meta.version", str(VERSION), repr(meta["version"]))
    _check_demo(meta, source)
    for key in ("seed", "n_observations", "n_particles"):
        _check_int(meta[key], f"meta.{key}", source, minimum=0 if key == "seed" else 1)
    for key in ("noise_std", "track_offset"):
        _check_finite_number(meta[key], f"meta.{key}", source)
    # track_offset may be any finite number (echolocation's is 0); noise may not.
    if meta["noise_std"] <= 0:
        _fail(source, "meta.noise_std", "a positive number", repr(meta["noise_std"]))
    _check_names_and_ranges(meta, source)
    for key in ("controls", "created_with"):
        if not isinstance(meta[key], Mapping):
            _fail(source, f"meta.{key}", "an object", type(meta[key]).__name__)
    _check_kind_fields(meta, source)


def _check_meta_key_set(
    meta: Mapping[str, object], required: frozenset[str], source: str
) -> None:
    for key in sorted(required - set(meta)):
        _fail(source, f"meta.{key}", "a field", "missing")
    for key in sorted(set(meta) - required):
        _fail(source, f"meta.{key}", "no such field", "an unexpected field")


def _check_demo(meta: Mapping[str, object], source: str) -> None:
    demo = meta["demo"]
    if demo not in DEMOS:
        _fail(source, "meta.demo", f"one of {', '.join(DEMOS)}", repr(demo))
    expected_kind = "model_comparison" if demo == "demo_model_comparison" else "filter"
    if meta["kind"] != expected_kind:
        _fail(source, "meta.kind", repr(expected_kind), repr(meta["kind"]))


def _check_names_and_ranges(meta: Mapping[str, object], source: str) -> None:
    names = meta["param_names"]
    if (
        not isinstance(names, (list, tuple))
        or not names
        or not all(isinstance(name, str) for name in names)
    ):
        _fail(source, "meta.param_names", "a non-empty list of strings", repr(names))
    documented = PARAM_NAMES[meta["demo"]]
    if tuple(names) != documented:
        _fail(source, "meta.param_names", repr(list(documented)), repr(names))
    ranges = meta["prior_ranges"]
    want = len(names) if meta["kind"] == "filter" else 2
    if not isinstance(ranges, (list, tuple)) or len(ranges) != want:
        _fail(source, "meta.prior_ranges", f"{want} [low, high] pairs", repr(ranges))
    for pair in ranges:
        if not _is_ordered_pair(pair):
            _fail(
                source, "meta.prior_ranges", "finite pairs with low < high", repr(pair)
            )


def _check_kind_fields(meta: Mapping[str, object], source: str) -> None:
    if meta["kind"] == "model_comparison":
        _check_int(meta["true_k"], "meta.true_k", source, minimum=1)
        return
    limit = min(CLOUD_SIZE, meta["n_particles"])
    size = meta["cloud_size"]
    if not _is_int(size) or not 1 <= size <= limit:
        _fail(source, "meta.cloud_size", f"an integer from 1 to {limit}", repr(size))


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_ordered_pair(pair: object) -> bool:
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        return False
    low, high = pair
    return _is_finite_number(low) and _is_finite_number(high) and low < high


def _check_int(value: object, key: str, source: str, minimum: int) -> None:
    if not _is_int(value) or value < minimum:
        _fail(source, key, f"an integer >= {minimum}", repr(value))


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # an integer too large for a float
        return False


def _check_finite_number(value: object, key: str, source: str) -> None:
    if not _is_finite_number(value):
        _fail(source, key, "a finite number", repr(value))


# --- validation: arrays ------------------------------------------------------

_PREDICTING_DEMOS = {"demo_1d", "demo_multi_mass", "demo_echolocation_3d"}

Spec = tuple[tuple[int | None, ...], np.dtype]
_F8, _F4, _I8 = np.dtype(np.float64), np.dtype(np.float32), np.dtype(np.int64)


def _check_clocks_and_candidates(
    meta: Mapping[str, object], arrays: Mapping[str, NDArray], source: str
) -> list[int]:
    """Check the arrays whose shapes everything else is derived from.

    Returns the candidate counts ``model_ks`` (empty unless model comparison).
    """
    dimensions = SPATIAL_DIMENSIONS[meta["demo"]]
    _require(arrays, "clock_positions", source)
    _check_array(
        arrays["clock_positions"], "clock_positions", (None, dimensions), _F8, source
    )
    clocks = len(arrays["clock_positions"])
    minimum = _MINIMUM_CLOCKS.get(meta["demo"], 1)
    if clocks < minimum:
        _fail(source, "clock_positions", f"at least {minimum} clocks", clocks)
    if meta["kind"] != "model_comparison":
        return []
    _require(arrays, "model_ks", source)
    ks = arrays["model_ks"]
    _check_array(ks, "model_ks", (None,), _I8, source)
    if ks.size == 0 or ks[0] < 1 or np.any(np.diff(ks) <= 0):
        _fail(source, "model_ks", "strictly ascending positive integers", ks.tolist())
    return [int(k) for k in ks]


def _array_specs(
    meta: Mapping[str, object], arrays: Mapping[str, NDArray], ks: list[int]
) -> dict[str, Spec]:
    demo = meta["demo"]
    clocks, dimensions = arrays["clock_positions"].shape
    steps = meta["n_observations"]
    params = len(meta["param_names"])
    specs: dict[str, Spec] = {
        "clock_positions": ((clocks, dimensions), _F8),
        "true_params": ((params,), _F8),
        "true_rates": ((clocks,), _F8),
        "observations": ((steps, clocks), _F8),
    }
    if meta["kind"] == "model_comparison":
        specs.update(_model_specs(steps, dimensions, ks))
        return specs
    rows = steps + 1
    shown = meta["cloud_size"]
    specs.update(
        {
            "mean": ((rows, params), _F8),
            "cov": ((rows, params, params), _F8),
            "ess": ((rows,), _F8),
            "log_evidence": ((rows,), _F8),
            "tempering_stages": ((steps,), _I8),
            "mh_proposals": ((steps,), _I8),
            "mh_acceptances": ((steps,), _I8),
            "cloud": ((rows, shown, params), _F4),
        }
    )
    if demo in _PREDICTING_DEMOS:
        specs["predicted_mean"] = ((rows, clocks), _F8)
        specs["predicted_std"] = ((rows, clocks), _F8)
    if demo == "demo_echolocation_3d":
        specs["filter_observations"] = ((steps, clocks - 1), _F8)
        specs["view_bounds"] = ((2,), _F8)
        specs["weight_in_view"] = ((rows,), _F8)
    return specs


def _check_true_model(
    meta: Mapping[str, object],
    arrays: Mapping[str, NDArray],
    ks: list[int],
    source: str,
) -> None:
    """The true count is a candidate, and the truth has that many masses."""
    true_k = meta["true_k"]
    if true_k not in ks:
        _fail(source, "meta.true_k", f"one of model_ks {ks}", repr(true_k))
    dimensions = arrays["clock_positions"].shape[1]
    want = true_k * (dimensions + 1)
    have = len(arrays["true_params"])
    if have != want:
        _fail(source, "true_params", f"{want} values for true_k = {true_k}", have)


def _model_specs(steps: int, dimensions: int, ks: list[int]) -> dict[str, Spec]:
    count, rows = len(ks), steps + 1
    specs: dict[str, Spec] = {
        "model_ks": ((count,), _I8),
        "model_prior": ((count,), _F8),
        "model_posterior": ((rows, count), _F8),
        "model_log_evidence": ((rows, count), _F8),
        "model_ess": ((rows, count), _F8),
    }
    for k in ks:
        specs[f"final_mean_k{k}"] = ((k * (dimensions + 1),), _F8)
        specs[f"final_std_k{k}"] = ((k * (dimensions + 1),), _F8)
    return specs


def _require(arrays: Mapping[str, NDArray], key: str, source: str) -> None:
    if key not in arrays:
        _fail(source, key, "an array", "missing")


def _check_key_set(
    specs: Mapping[str, Spec], arrays: Mapping[str, NDArray], source: str
) -> None:
    for key in specs:
        _require(arrays, key, source)
    for key in sorted(set(arrays) - set(specs)):
        _fail(source, key, "no such array", "an unexpected array")


def _check_array(
    array: NDArray,
    key: str,
    shape: tuple[int | None, ...],
    dtype: np.dtype,
    source: str,
) -> None:
    matches = len(array.shape) == len(shape) and all(
        want is None or want == have for want, have in zip(shape, array.shape)
    )
    if not matches:
        wanted = "(" + ", ".join("*" if n is None else str(n) for n in shape) + ")"
        _fail(source, key, f"shape {wanted}", f"shape {array.shape}")
    if array.dtype != dtype:
        _fail(source, key, f"dtype {dtype.name}", f"dtype {array.dtype.name}")


def _check_finite(arrays: Mapping[str, NDArray], source: str) -> None:
    for key, array in arrays.items():
        if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
            _fail(source, key, "finite values", "NaN or infinity")
