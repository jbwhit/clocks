import errno
import json
import math
import os
import re
import struct
import zipfile
import zlib
from pathlib import Path
from typing import IO

import numpy as np
import pytest

from clocks import _replay
from clocks._demos import demo_1d
from clocks._replay import (
    CLOUD_SIZE,
    DEMOS,
    FORMAT,
    PARAM_NAMES,
    SPATIAL_DIMENSIONS,
    VERSION,
    Replay,
    create_replay,
    first_difference,
    load_replay,
    provenance,
    record_filter,
    save_replay,
    thin,
    validate_replay,
)
from clocks.api import build_particle_filter, simulate
from clocks.config import InferenceConfig, NoiseConfig, PriorConfig, SimulationConfig
from clocks.types import (
    ClockArray,
    MassConfig,
    Observation,
    ParticleState,
    UpdateDiagnostics,
)

F8, I8 = np.float64, np.int64
CONTROLS = {
    "ess_target": 0.8,
    "rejuvenation_steps": 2,
    "proposal_scale": 2.38,
    "resampling": "systematic",
}
CREATED = {"python": "3.14.0", "numpy": "2.4.2", "scipy": "1.17.1", "platform": "test"}
BASE_META = {
    "format": "clocks-replay",
    "version": 1,
    "seed": 7,
    "n_particles": 4,
    "noise_std": 0.005,
    "controls": CONTROLS,
    "created_with": CREATED,
}


def filter_fixture():  # T = 2, C = 2 clocks on a line (D = 1), P = 2, N = 4, m = 3
    meta = {
        **BASE_META,
        "demo": "demo_1d",
        "kind": "filter",
        "n_observations": 2,
        "track_offset": 1.0,
        "param_names": ["x", "M"],
        "prior_ranges": [[-8.0, 8.0], [0.005, 0.15]],
        "cloud_size": 3,
    }
    arrays = {
        "clock_positions": np.array([[-1.0], [1.0]], F8),
        "true_params": np.array([0.5, 0.05], F8),
        "true_rates": np.array([0.97, 0.98], F8),
        "observations": np.array([[0.971, 0.979], [0.969, 0.981]], F8),
        "mean": np.array([[0.0, 0.08], [0.4, 0.06], [0.5, 0.05]], F8),
        "cov": np.array(
            [
                [[21.0, 0.0], [0.0, 0.002]],
                [[0.3, 0.001], [0.001, 0.0002]],
                [[0.1, 0.0005], [0.0005, 0.0001]],
            ],
            F8,
        ),
        "ess": np.array([4.0, 3.5, 3.8], F8),
        "log_evidence": np.array([0.0, 5.0, 11.0], F8),
        "tempering_stages": np.array([3, 1], I8),
        "mh_proposals": np.array([24, 8], I8),
        "mh_acceptances": np.array([6, 3], I8),
        "cloud": np.array(
            [
                [[-4.0, 0.02], [0.0, 0.08], [4.0, 0.14]],
                [[0.2, 0.05], [0.4, 0.06], [0.6, 0.07]],
                [[0.45, 0.049], [0.5, 0.05], [0.55, 0.051]],
            ],
            np.float32,
        ),
        "predicted_mean": np.array([[0.96, 0.97], [0.972, 0.979], [0.97, 0.98]], F8),
        "predicted_std": np.array([[0.02, 0.02], [0.004, 0.003], [0.002, 0.002]], F8),
    }
    return meta, arrays


def echo_fixture():  # T = 1, C = 2 clocks in space (D = 3), P = 4, N = 4, m = 3
    meta = {
        **BASE_META,
        "demo": "demo_echolocation_3d",
        "kind": "filter",
        "n_observations": 1,
        "track_offset": 0.0,
        "param_names": ["x", "y", "z", "M"],
        "prior_ranges": [[-16.0, 16.0]] * 3 + [[0.005, 0.15]],
        "cloud_size": 3,
    }
    arrays = {
        "clock_positions": np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], F8),
        "true_params": np.array([1.0, 1.5, 3.0, 0.08], F8),
        "true_rates": np.array([0.98, 0.97], F8),
        "observations": np.array([[0.004, -0.004]], F8),
        "filter_observations": np.array([[0.005656854249492381]], F8),
        "mean": np.array([[0.0, 0.0, 0.0, 0.08], [1.1, 1.6, 3.2, 0.09]], F8),
        "cov": np.stack(
            [
                np.diag([85.0, 85.0, 85.0, 0.0018]),
                np.diag([0.03, 0.04, 0.07, 0.0004]),
            ]
        ).astype(F8),
        "ess": np.array([4.0, 3.2], F8),
        "log_evidence": np.array([0.0, 7.5], F8),
        "tempering_stages": np.array([2], I8),
        "mh_proposals": np.array([8], I8),
        "mh_acceptances": np.array([3], I8),
        "cloud": np.array(
            [
                [
                    [-9.0, 2.0, 5.0, 0.03],
                    [0.5, -7.0, 1.0, 0.09],
                    [8.0, 4.0, -3.0, 0.14],
                ],
                [[1.0, 1.5, 3.1, 0.085], [1.1, 1.6, 3.2, 0.09], [1.2, 1.7, 3.3, 0.095]],
            ],
            np.float32,
        ),
        "predicted_mean": np.array([[0.0, 0.0], [0.0041, -0.0041]], F8),
        "predicted_std": np.array([[0.01, 0.01], [0.0005, 0.0005]], F8),
        "view_bounds": np.array([-1.3, 4.8], F8),
        "weight_in_view": np.array([0.25, 1.0], F8),
    }
    return meta, arrays


def model_fixture():  # T = 1, C = 2, candidates K = 1 and 2, N = 16
    meta = {
        **BASE_META,
        "demo": "demo_model_comparison",
        "kind": "model_comparison",
        "n_observations": 1,
        "n_particles": 16,
        "track_offset": 1.0,
        "param_names": ["x₁", "x₂", "M₁", "M₂"],
        "prior_ranges": [[-8.0, 8.0], [0.005, 0.15]],
        "true_k": 2,
    }
    arrays = {
        "clock_positions": np.array([[-1.0], [1.0]], F8),
        "true_params": np.array([-2.0, 3.0, 0.045, 0.030], F8),
        "true_rates": np.array([0.97, 0.98], F8),
        "observations": np.array([[0.971, 0.979]], F8),
        "model_ks": np.array([1, 2], I8),
        "model_prior": np.array([0.5, 0.5], F8),
        "model_posterior": np.array([[0.5, 0.5], [0.2, 0.8]], F8),
        "model_log_evidence": np.array([[0.0, 0.0], [-3.0, -1.6137056388801094]], F8),
        "model_ess": np.array([[16.0, 16.0], [12.5, 14.0]], F8),
        "final_mean_k1": np.array([0.4, 0.07], F8),
        "final_std_k1": np.array([0.9, 0.02], F8),
        "final_mean_k2": np.array([-2.1, 3.2, 0.044, 0.031], F8),
        "final_std_k2": np.array([0.3, 0.4, 0.004, 0.005], F8),
    }
    return meta, arrays


FIXTURES = {
    "demo_1d": filter_fixture,
    "demo_echolocation_3d": echo_fixture,
    "demo_model_comparison": model_fixture,
}

# Parametrizations are generated from the fixtures, so a key added to one is
# covered by every "each key" test automatically.
EACH_FIXTURE = [pytest.param(name, id=name) for name in FIXTURES]
EACH_KEY = [
    pytest.param(name, key, id=f"{name}-{key}")
    for name, make in FIXTURES.items()
    for key in make()[1]
]
EACH_FLOATING_KEY = [
    pytest.param(name, key, id=f"{name}-{key}")
    for name, make in FIXTURES.items()
    for key, array in make()[1].items()
    if array.dtype.kind == "f"
]
EACH_AXIS = [
    pytest.param(name, key, axis, id=f"{name}-{key}-axis{axis}")
    for name, make in FIXTURES.items()
    for key, array in make()[1].items()
    if key not in {"clock_positions", "model_ks"}
    for axis in range(array.ndim)
]


def _write_plain(path: Path, meta, arrays, raw_meta=None) -> None:
    stored = np.array(json.dumps(meta)) if raw_meta is None else raw_meta
    np.savez_compressed(path, meta=stored, **arrays)


def _invalid(meta, arrays, match: str | None) -> None:
    with pytest.raises(ValueError, match=match):
        validate_replay(Replay(meta, arrays))


def _loose(name: str) -> str:
    return re.escape(name)


def test_module_constants() -> None:
    assert (FORMAT, VERSION, CLOUD_SIZE) == ("clocks-replay", 1, 500)
    assert DEMOS == (
        "demo_1d",
        "demo_2d",
        "demo_multi_mass",
        "demo_multi_mass_2d",
        "demo_model_comparison",
        "demo_density",
        "demo_echolocation_3d",
    )
    assert SPATIAL_DIMENSIONS == {
        "demo_1d": 1,
        "demo_multi_mass": 1,
        "demo_model_comparison": 1,
        "demo_density": 1,
        "demo_2d": 2,
        "demo_multi_mass_2d": 2,
        "demo_echolocation_3d": 3,
    }
    assert PARAM_NAMES == {
        "demo_1d": ("x", "M"),
        "demo_2d": ("x", "y", "M"),
        "demo_multi_mass": ("x₁", "x₂", "M₁", "M₂"),
        "demo_multi_mass_2d": ("x₁", "y₁", "x₂", "y₂", "M₁", "M₂"),
        "demo_model_comparison": ("x₁", "x₂", "M₁", "M₂"),
        "demo_density": ("μ", "σ", "A"),
        "demo_echolocation_3d": ("x", "y", "z", "M"),
    }


def test_provenance_names_its_four_fields() -> None:
    info = provenance()
    assert set(info) == {"python", "numpy", "scipy", "platform"}
    assert info["numpy"] == np.__version__


@pytest.mark.parametrize("name", EACH_FIXTURE)
def test_literal_archive_loads(name: str, tmp_path: Path) -> None:
    meta, arrays = FIXTURES[name]()
    path = tmp_path / "plain.npz"
    _write_plain(path, meta, arrays)

    replay = load_replay(path)

    assert dict(replay.meta) == meta
    assert set(replay.arrays) == set(arrays)
    for key, expected in arrays.items():
        assert np.array_equal(replay.arrays[key], expected), key
        assert replay.arrays[key].dtype == expected.dtype, key


@pytest.mark.parametrize("name", EACH_FIXTURE)
def test_saved_archive_reads_back_as_literals(name: str, tmp_path: Path) -> None:
    meta, arrays = FIXTURES[name]()
    path = tmp_path / "saved.npz"

    save_replay(path, Replay(meta, arrays))

    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {"meta", *arrays}
        assert json.loads(z["meta"].item()) == meta
        for key, expected in arrays.items():
            assert np.array_equal(z[key], expected), key
            assert z[key].dtype == expected.dtype, key


def test_load_replay_reads_the_bytes_it_is_given(tmp_path: Path) -> None:
    """With ``data``, the file at ``path`` is not read; ``path`` names errors."""
    meta, arrays = filter_fixture()
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    save_replay(first, Replay(meta, arrays))
    changed = {**arrays, "cloud": arrays["cloud"] + 0.5}
    save_replay(second, Replay(meta, changed))

    replay = load_replay(first, second.read_bytes())

    assert np.array_equal(replay.arrays["cloud"], changed["cloud"])
    with pytest.raises(ValueError, match=re.escape(f"{first}: not a replay archive")):
        load_replay(first, b"not an archive")


@pytest.mark.parametrize(("name", "key"), EACH_KEY)
def test_missing_key_is_rejected(name: str, key: str) -> None:
    meta, arrays = FIXTURES[name]()
    del arrays[key]
    _invalid(meta, arrays, rf"(?s)(?=.*{_loose(key)})(?=.*missing)")


@pytest.mark.parametrize("name", EACH_FIXTURE)
def test_extra_key_is_rejected(name: str) -> None:
    meta, arrays = FIXTURES[name]()
    arrays["surplus"] = np.zeros(1)
    _invalid(meta, arrays, "surplus")


@pytest.mark.parametrize(("name", "key"), EACH_KEY)
def test_wrong_dtype_is_rejected(name: str, key: str) -> None:
    meta, arrays = FIXTURES[name]()
    original = arrays[key].dtype
    wrong = {"float64": np.float32, "float32": np.float64, "int64": np.int32}
    arrays[key] = arrays[key].astype(wrong[original.name])
    _invalid(meta, arrays, rf"(?s)(?=.*{_loose(key)})(?=.*{original.name})")


@pytest.mark.parametrize(("name", "key"), EACH_KEY)
def test_wrong_rank_is_rejected(name: str, key: str) -> None:
    meta, arrays = FIXTURES[name]()
    arrays[key] = arrays[key][None]
    _invalid(meta, arrays, _loose(key))


@pytest.mark.parametrize(("name", "key", "axis"), EACH_AXIS)
def test_wrong_length_is_rejected(name: str, key: str, axis: int) -> None:
    meta, arrays = FIXTURES[name]()
    arrays[key] = np.delete(arrays[key], -1, axis=axis)
    _invalid(meta, arrays, _loose(key))


@pytest.mark.parametrize("name", EACH_FIXTURE)
def test_wrong_coordinate_width_is_rejected(name: str) -> None:
    meta, arrays = FIXTURES[name]()
    positions = arrays["clock_positions"]
    arrays["clock_positions"] = np.hstack([positions, np.zeros((len(positions), 1))])
    _invalid(meta, arrays, "clock_positions")


@pytest.mark.parametrize("name", EACH_FIXTURE)
def test_clock_count_must_agree(name: str) -> None:
    meta, arrays = FIXTURES[name]()
    positions = arrays["clock_positions"]
    arrays["clock_positions"] = np.vstack([positions, positions[:1]])
    # The first array sized by the clock count, checked after clock_positions.
    _invalid(meta, arrays, re.escape("true_rates: expected shape (3)"))


def _with_clocks(name: str, count: int):
    """The fixture with ``count`` clocks and every clock-sized axis to match."""
    meta, arrays = FIXTURES[name]()
    rows = meta["n_observations"]
    dimensions = arrays["clock_positions"].shape[1]
    arrays["clock_positions"] = np.linspace(-1.0, 1.0, count * dimensions).reshape(
        count, dimensions
    )
    arrays["true_rates"] = np.full(count, 0.97)
    arrays["observations"] = np.zeros((rows, count))
    if "filter_observations" in arrays:
        arrays["filter_observations"] = np.zeros((rows, count - 1))
    for key in ("predicted_mean", "predicted_std"):
        if key in arrays:
            arrays[key] = np.full((rows + 1, count), 0.01)
    return meta, arrays


@pytest.mark.parametrize(
    ("name", "too_few", "enough"),
    [
        pytest.param("demo_1d", 0, 1, id="demo_1d"),
        pytest.param("demo_model_comparison", 0, 1, id="demo_model_comparison"),
        pytest.param("demo_echolocation_3d", 1, 2, id="demo_echolocation_3d"),
    ],
)
def test_too_few_clocks_are_rejected(name: str, too_few: int, enough: int) -> None:
    validate_replay(Replay(*_with_clocks(name, enough)))
    _invalid(*_with_clocks(name, too_few), r"clock_positions: expected at least")


def test_candidate_count_must_agree() -> None:
    meta, arrays = model_fixture()
    arrays["model_ks"] = np.array([1, 2, 3], I8)
    _invalid(meta, arrays, re.escape("final_mean_k3: expected an array, got missing"))


@pytest.mark.parametrize(("name", "key"), EACH_FLOATING_KEY)
def test_non_finite_value_is_rejected(name: str, key: str) -> None:
    meta, arrays = FIXTURES[name]()
    arrays[key] = arrays[key].copy()
    arrays[key].flat[0] = np.nan
    _invalid(meta, arrays, _loose(key))


def test_true_k_must_be_a_candidate() -> None:
    meta, arrays = model_fixture()
    meta["true_k"] = 3
    _invalid(meta, arrays, re.escape("meta.true_k: expected one of model_ks [1, 2]"))


def test_true_params_must_hold_true_k_masses() -> None:
    # The names (and so true_params' shape) are fixed by PARAM_NAMES; a true
    # count of 1 against that two-mass truth is what this check alone catches.
    meta, arrays = model_fixture()
    meta["true_k"] = 1
    _invalid(
        meta, arrays, re.escape("true_params: expected 2 values for true_k = 1, got 4")
    )


def test_descending_model_ks_is_rejected() -> None:
    meta, arrays = model_fixture()
    arrays["model_ks"] = np.array([2, 1], I8)
    _invalid(meta, arrays, "model_ks")


HUGE = 10**400  # an integer too large for a float: math.isfinite overflows
NAN = float("nan")
BAD_META = [
    ("demo_1d", "seed", -1, "meta.seed"),
    ("demo_1d", "n_observations", 0, "meta.n_observations"),
    ("demo_1d", "n_particles", 0, "meta.n_particles"),
    ("demo_1d", "noise_std", NAN, "meta.noise_std"),
    ("demo_1d", "noise_std", HUGE, "meta.noise_std"),
    ("demo_1d", "noise_std", 0.0, "meta.noise_std"),
    ("demo_1d", "noise_std", -0.005, "meta.noise_std"),
    ("demo_1d", "track_offset", float("inf"), "meta.track_offset"),
    ("demo_1d", "track_offset", HUGE, "meta.track_offset"),
    ("demo_1d", "param_names", [], "meta.param_names"),
    ("demo_1d", "param_names", ["x", 1], "meta.param_names"),
    ("demo_1d", "param_names", "xM", "meta.param_names"),
    ("demo_1d", "param_names", ["M", "x"], "meta.param_names"),
    ("demo_1d", "param_names", ["x", "y"], "meta.param_names"),
    ("demo_echolocation_3d", "param_names", ["y", "x", "z", "M"], "meta.param_names"),
    (
        "demo_model_comparison",
        "param_names",
        ["x₁", "M₁", "x₂", "M₂"],
        "meta.param_names",
    ),
    ("demo_1d", "prior_ranges", [[-8.0, 8.0]], "meta.prior_ranges"),
    ("demo_1d", "prior_ranges", [[-8.0, 8.0], [0.15, 0.005]], "meta.prior_ranges"),
    ("demo_1d", "prior_ranges", [[-8.0, 8.0], [0.005, 0.005]], "meta.prior_ranges"),
    ("demo_1d", "prior_ranges", [[-8.0, 8.0], [0.005, NAN]], "meta.prior_ranges"),
    ("demo_1d", "prior_ranges", [[-8.0, 8.0], [0.005, HUGE]], "meta.prior_ranges"),
    ("demo_1d", "controls", "systematic", "meta.controls"),
    ("demo_1d", "created_with", [], "meta.created_with"),
    ("demo_1d", "kind", ["filter"], "meta.kind"),
    ("demo_1d", "kind", {"a": 1}, "meta.kind"),
    ("demo_model_comparison", "true_k", 0, "meta.true_k"),
    ("demo_model_comparison", "true_k", 2.5, "meta.true_k"),
    ("demo_model_comparison", "prior_ranges", [[-8.0, 8.0]], "meta.prior_ranges"),
]


@pytest.mark.parametrize(
    ("name", "field", "value", "fragment"),
    [
        pytest.param(*case, id=f"{case[0]}-{case[1]}-{i}")
        for i, case in enumerate(BAD_META)
    ],
)
def test_bad_meta_field_is_rejected(
    name: str, field: str, value: object, fragment: str
) -> None:
    meta, arrays = FIXTURES[name]()
    meta[field] = value
    _invalid(meta, arrays, re.escape(fragment))


def test_non_positive_model_ks_is_rejected() -> None:
    meta, arrays = model_fixture()
    arrays["model_ks"] = np.array([0, 1], I8)
    _invalid(meta, arrays, "model_ks")


def test_cloud_size_must_match_meta() -> None:
    meta, arrays = filter_fixture()
    meta["cloud_size"] = 4
    _invalid(meta, arrays, "cloud")


def test_cloud_size_is_capped_at_500() -> None:
    meta, arrays = filter_fixture()
    meta["n_particles"] = 600
    meta["cloud_size"] = 501
    arrays["cloud"] = np.zeros((3, 501, 2), np.float32)
    _invalid(meta, arrays, re.escape("meta.cloud_size"))


@pytest.mark.parametrize("name", EACH_FIXTURE)
@pytest.mark.parametrize("fault", ["missing", "extra"])
def test_meta_keys_are_exact(name: str, fault: str) -> None:
    meta, arrays = FIXTURES[name]()
    if fault == "missing":
        del meta["seed"]
        match = r"(?s)(?=.*meta)(?=.*seed)"
    else:
        meta["surplus"] = 1
        match = r"(?s)(?=.*meta)(?=.*surplus)"
    _invalid(meta, arrays, match)


@pytest.mark.parametrize(("field", "value"), [("format", "other"), ("version", 2)])
def test_wrong_format_or_version_is_rejected(field: str, value: object) -> None:
    meta, arrays = filter_fixture()
    meta[field] = value
    _invalid(meta, arrays, field)


def test_unknown_demo_is_rejected() -> None:
    meta, arrays = filter_fixture()
    meta["demo"] = "demo_9d"
    _invalid(meta, arrays, "demo_9d")


@pytest.mark.parametrize(
    "raw",
    [np.array("null"), np.array("[1]"), np.array("3")],
    ids=["null", "list", "number"],
)
def test_meta_must_decode_to_an_object(raw: np.ndarray, tmp_path: Path) -> None:
    meta, arrays = filter_fixture()
    path = tmp_path / "bad.npz"
    _write_plain(path, meta, arrays, raw_meta=raw)
    with pytest.raises(ValueError, match="meta"):
        load_replay(path)


@pytest.mark.parametrize(
    "raw",
    [np.array([1, 2]), np.array(["a", "b"])],
    ids=["integers", "two_strings"],
)
def test_meta_must_be_one_string(raw: np.ndarray, tmp_path: Path) -> None:
    meta, arrays = filter_fixture()
    path = tmp_path / "bad.npz"
    _write_plain(path, meta, arrays, raw_meta=raw)
    with pytest.raises(ValueError, match="meta"):
        load_replay(path)


def test_an_npy_payload_is_a_value_error(tmp_path: Path) -> None:
    path = tmp_path / "plain.npz"
    # np.save(path, ...) would append .npy and write elsewhere; a handle does not.
    with path.open("wb") as handle:
        np.save(handle, np.zeros(3))
    with pytest.raises(ValueError, match=re.escape(str(path))):
        load_replay(path)


@pytest.mark.parametrize("content", [b"not a zip", b""], ids=["not_a_zip", "empty"])
def test_an_unreadable_file_is_a_value_error(content: bytes, tmp_path: Path) -> None:
    path = tmp_path / "broken.npz"
    path.write_bytes(content)
    with pytest.raises(ValueError, match=re.escape(str(path))):
        load_replay(path)


def test_an_unreadable_member_is_a_value_error(tmp_path: Path) -> None:
    meta, arrays = filter_fixture()
    arrays["mean"] = np.array([{"a": 1}], dtype=object)
    path = tmp_path / "object.npz"
    _write_plain(path, meta, arrays)
    with pytest.raises(
        ValueError, match=rf"(?s)(?=.*{re.escape(str(path))})(?=.*mean)"
    ):
        load_replay(path)


def _members(path: Path) -> list[tuple[str, bytes]]:
    with zipfile.ZipFile(path) as archive:
        return [(name, archive.read(name)) for name in archive.namelist()]


def _write_members(path: Path, members: list[tuple[str, bytes]]) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members:
            archive.writestr(name, data)


def _saved_fixture(tmp_path: Path) -> Path:
    meta, arrays = filter_fixture()
    path = tmp_path / "real.npz"
    save_replay(path, Replay(meta, arrays))
    return path


@pytest.mark.parametrize("key", ["meta", "cloud"])
def test_a_member_that_is_not_an_array_is_a_value_error(
    key: str, tmp_path: Path
) -> None:
    path = _saved_fixture(tmp_path)
    members = [
        (name, b"raw bytes" if name == f"{key}.npy" else data)
        for name, data in _members(path)
    ]
    _write_members(path, members)
    with pytest.raises(
        ValueError, match=rf"(?s)(?=.*{re.escape(str(path))})(?=.*{key}: expected)"
    ):
        load_replay(path)


def test_a_corrupted_compressed_member_is_a_value_error(tmp_path: Path) -> None:
    path = _saved_fixture(tmp_path)
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo("cloud.npy")
    raw = bytearray(path.read_bytes())
    name_length, extra_length = struct.unpack_from("<HH", raw, info.header_offset + 26)
    data_start = info.header_offset + 30 + name_length + extra_length
    # Bits 1-2 of the first deflate byte are the block type; 0b11 is reserved,
    # so decompression fails outright. (Flipped bytes elsewhere in so small a
    # member tend to decompress and fail the CRC check instead.)
    raw[data_start] |= 0b110
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match=re.escape(f"{path}: cloud")) as raised:
        load_replay(path)
    assert isinstance(raised.value.__cause__, zlib.error)


def test_a_duplicated_member_is_a_value_error(tmp_path: Path) -> None:
    path = _saved_fixture(tmp_path)
    members = _members(path)
    with pytest.warns(UserWarning, match="Duplicate name"):
        _write_members(path, [*members, ("mean.npy", dict(members)["mean.npy"])])
    with pytest.raises(
        ValueError, match=rf"(?s)(?=.*{re.escape(str(path))})(?=.*mean)(?=.*duplicate)"
    ):
        load_replay(path)


def test_save_requires_an_npz_path(tmp_path: Path) -> None:
    meta, arrays = filter_fixture()
    with pytest.raises(ValueError, match="npz"):
        save_replay(tmp_path / "x", Replay(meta, arrays))
    assert list(tmp_path.iterdir()) == []


def test_save_refuses_an_invalid_replay(tmp_path: Path) -> None:
    meta, arrays = filter_fixture()
    del arrays["mean"]
    path = tmp_path / "x.npz"
    with pytest.raises(ValueError, match="mean"):
        save_replay(path, Replay(meta, arrays))
    assert not path.exists()


def _another_replay() -> Replay:
    meta, arrays = filter_fixture()
    return Replay({**meta, "seed": meta["seed"] + 1}, arrays)


def test_a_failed_save_leaves_the_previous_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "x.npz"
    first = Replay(*filter_fixture())
    save_replay(path, first)
    before = path.read_bytes()

    def disk_full(target: IO[bytes] | Path, **arrays: object) -> None:
        # Like NumPy: a path is opened for writing (truncating), a file is used.
        if isinstance(target, Path):
            target = target.open("wb")
        target.write(b"PK a few bytes")
        target.flush()
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(np, "savez_compressed", disk_full)
    with pytest.raises(OSError, match="No space left"):
        save_replay(path, _another_replay())

    assert path.read_bytes() == before
    assert load_replay(path).meta == first.meta
    assert [p.name for p in tmp_path.iterdir()] == ["x.npz"]


def test_a_failed_cleanup_does_not_mask_the_write_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "x.npz"
    save_replay(path, Replay(*filter_fixture()))
    before = path.read_bytes()

    def disk_full(target: IO[bytes], **arrays: object) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    real_unlink = Path.unlink

    def cannot_unlink(self: Path, missing_ok: bool = False) -> None:
        if self.name.endswith(".tmp"):
            raise PermissionError("cannot remove")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(np, "savez_compressed", disk_full)
    monkeypatch.setattr(Path, "unlink", cannot_unlink)
    with pytest.raises(OSError, match="No space left") as raised:
        save_replay(path, _another_replay())
    monkeypatch.undo()

    leftover = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert raised.value.errno == errno.ENOSPC
    assert len(leftover) == 1
    assert any(str(leftover[0]) in note for note in raised.value.__notes__)
    assert path.read_bytes() == before


def test_a_cleanup_note_that_cannot_be_added_keeps_the_write_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only this catches the archive's cleanup note replacing the write error
    when that error's ``__notes__`` is not a list (``add_note`` raises)."""
    path = tmp_path / "x.npz"
    error = OSError(errno.ENOSPC, "No space left on device")
    error.__notes__ = ()

    def disk_full(target: IO[bytes], **arrays: object) -> None:
        raise error

    real_unlink = Path.unlink

    def cannot_unlink(self: Path, missing_ok: bool = False) -> None:
        if self.name.endswith(".tmp"):
            raise PermissionError("cannot remove")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(np, "savez_compressed", disk_full)
    monkeypatch.setattr(Path, "unlink", cannot_unlink)
    with pytest.raises(OSError) as raised:
        save_replay(path, _another_replay())
    monkeypatch.undo()

    assert raised.value is error
    assert error.__notes__ == ()


def test_an_interrupted_cleanup_keeps_the_write_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only this catches an interrupt while the archive's temporary file is
    removed replacing the write error; the interrupt becomes the note."""
    path = tmp_path / "x.npz"
    error = OSError(errno.ENOSPC, "No space left on device")

    def disk_full(target: IO[bytes], **arrays: object) -> None:
        raise error

    real_unlink = Path.unlink

    def interrupted(self: Path, missing_ok: bool = False) -> None:
        if self.name.endswith(".tmp"):
            raise KeyboardInterrupt("during the unlink")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(np, "savez_compressed", disk_full)
    monkeypatch.setattr(Path, "unlink", interrupted)
    # BaseException: an escaped interrupt would end the session.
    with pytest.raises(BaseException) as raised:
        save_replay(path, _another_replay())
    monkeypatch.undo()

    assert raised.value is error
    [note] = error.__notes__
    assert note.endswith(".tmp was left behind: during the unlink")


@pytest.mark.parametrize("name", EACH_FIXTURE)
def test_create_writes_a_loadable_archive_at_a_free_path(
    name: str, tmp_path: Path
) -> None:
    meta, arrays = FIXTURES[name]()
    path = tmp_path / "x.npz"

    create_replay(path, Replay(meta, arrays))

    replay = load_replay(path)
    assert dict(replay.meta) == meta
    for key, expected in arrays.items():
        assert np.array_equal(replay.arrays[key], expected), key
    assert [p.name for p in tmp_path.iterdir()] == ["x.npz"]


@pytest.mark.parametrize("kind", ["file", "symlink", "dangling_symlink"])
def test_create_refuses_to_replace_anything(tmp_path: Path, kind: str) -> None:
    path, target = tmp_path / "x.npz", tmp_path / "target.bin"
    if kind == "file":
        path.write_bytes(b"earlier")
    elif kind == "symlink":
        target.write_bytes(b"earlier")
        path.symlink_to(target)
    else:
        path.symlink_to(target)
    before = path.lstat().st_ino
    contents = target.read_bytes() if target.exists() else None

    with pytest.raises(FileExistsError):
        create_replay(path, Replay(*filter_fixture()))

    assert path.lstat().st_ino == before
    assert path.is_symlink() == (kind != "file")
    assert (target.read_bytes() if target.exists() else None) == contents
    if kind == "file":
        assert path.read_bytes() == b"earlier"
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []


@pytest.mark.parametrize("writer", [save_replay, create_replay])
def test_a_successful_write_unlinks_no_name_but_the_one_a_link_leaves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writer: object
) -> None:
    unlinked: list[Path] = []
    real_unlink = Path.unlink

    def recording_unlink(self: Path, missing_ok: bool = False) -> None:
        unlinked.append(self)
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", recording_unlink)

    writer(tmp_path / "x.npz", Replay(*filter_fixture()))  # type: ignore[operator]

    # A replace leaves no temporary name to remove; a link leaves exactly one.
    assert len(unlinked) == (1 if writer is create_replay else 0)
    assert [p.name for p in tmp_path.iterdir()] == ["x.npz"]


def test_create_succeeds_when_the_temporary_name_is_already_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_link = os.link

    def link_then_lose_the_temporary_name(source: Path, target: Path) -> None:
        real_link(source, target)
        Path(source).unlink()  # another process removes it first

    monkeypatch.setattr(os, "link", link_then_lose_the_temporary_name)
    path = tmp_path / "x.npz"

    create_replay(path, Replay(*filter_fixture()))

    monkeypatch.undo()
    assert load_replay(path).meta == Replay(*filter_fixture()).meta
    assert [p.name for p in tmp_path.iterdir()] == ["x.npz"]


def test_save_replaces_an_existing_archive_whole(tmp_path: Path) -> None:
    path = tmp_path / "x.npz"
    save_replay(path, Replay(*filter_fixture()))
    second = _another_replay()

    save_replay(path, second)

    assert load_replay(path).meta == second.meta
    assert [p.name for p in tmp_path.iterdir()] == ["x.npz"]


# --- thin and record_filter --------------------------------------------------


PHI = (math.sqrt(5.0) - 1.0) / 2.0


@pytest.mark.parametrize(
    ("weights", "size", "expected"),
    [
        ([0.1, 0.2, 0.0, 0.3, 0.4], 5, [1, 3, 3, 4, 4]),
        ([1 / 7] * 7, 3, [1, 3, 6]),
        ([0.25] * 4, 4, [0, 1, 2, 3]),
        ([PHI / 2, 0.0, 1 - PHI / 2], 2, [0, 2]),
    ],
    ids=["unequal", "uniform_7_3", "uniform_4_4", "tie"],
)
def test_thin_picks(weights, size, expected) -> None:
    particles = np.arange(len(weights), dtype=float)[:, None]
    picked = thin(particles, np.array(weights), size)
    assert picked.dtype == np.float32
    assert picked.shape == (len(expected), 1)
    assert picked[:, 0].tolist() == expected


def test_the_thinning_offset_is_the_golden_ratio_conjugate() -> None:
    assert _replay.THINNING_OFFSET == (math.sqrt(5.0) - 1.0) / 2.0


def test_thin_ignores_the_scale_of_the_weights() -> None:
    w = np.array([0.1, 0.2, 0.0, 0.3, 0.4])
    particles = np.arange(5, dtype=float)[:, None]
    assert np.array_equal(thin(particles, 0.5 * w, 5), thin(particles, w, 5))


def test_thin_is_capped_by_the_particle_count() -> None:
    particles = np.arange(4, dtype=float)[:, None]
    assert thin(particles, np.full(4, 0.25), 500).shape == (4, 1)


@pytest.mark.parametrize(
    "weights", [[0.5, -0.1, 0.6], [0.0, 0.0, 0.0]], ids=["negative", "all_zero"]
)
def test_thin_rejects_unusable_weights(weights) -> None:
    with pytest.raises(ValueError, match="weights"):
        thin(np.arange(3, dtype=float)[:, None], np.array(weights), 3)


STATE_PARTICLES = [
    [[-4.0, 0.02], [-2.0, 0.05], [0.0, 0.08], [2.0, 0.11], [4.0, 0.14]],
    [[0.1, 0.050], [0.2, 0.055], [0.4, 0.060], [0.5, 0.066], [0.7, 0.070]],
    [[0.42, 0.0480], [0.47, 0.0495], [0.50, 0.0500], [0.53, 0.0512], [0.58, 0.0520]],
]
STATE_WEIGHTS = [
    [0.2] * 5,
    [0.1, 0.2, 0.3, 0.25, 0.15],
    [0.05, 0.4, 0.05, 0.3, 0.2],
]


class ScriptedFilter:
    """Exactly the surface record_filter may read; no rng."""

    def __init__(self, states, diagnostics, log_evidence) -> None:
        self._states = states
        self._diagnostics = diagnostics
        self._log_evidence = log_evidence
        self.n_particles = len(states[0].weights)
        self.history = [states[0]]
        self.diagnostics_history: list = []
        self.log_evidence_history: list = []

    @property
    def state(self) -> ParticleState:
        return self.history[-1]

    def update(self, observation) -> ParticleState:
        i = len(self.diagnostics_history)
        self.history.append(self._states[i + 1])
        self.diagnostics_history.append(self._diagnostics[i])
        self.log_evidence_history.append(self._log_evidence[i])
        return self.state


def scripted():
    states = [
        ParticleState(np.array(p), np.array(w), i)
        for i, (p, w) in enumerate(zip(STATE_PARTICLES, STATE_WEIGHTS))
    ]
    pf = ScriptedFilter(
        states, [UpdateDiagnostics(3, 24, 6), UpdateDiagnostics(1, 8, 3)], [5.0, 11.0]
    )
    obs = [
        Observation(np.array([0.97, 0.98]), 0.0),
        Observation(np.array([0.9, 1.0]), 1.0),
    ]
    return pf, obs


def record_scripted(predict=None, arrays=None):
    pf, obs = scripted()
    return record_filter(pf, obs, meta={"m": 1}, arrays=arrays or {}, predict=predict)


def explicit_moments(x, w):
    total = sum(w)
    m = [sum(wi * xi[j] for wi, xi in zip(w, x)) / total for j in range(len(x[0]))]
    p = len(m)
    c = [
        [
            sum(wi * (xi[a] - m[a]) * (xi[b] - m[b]) for wi, xi in zip(w, x)) / total
            for b in range(p)
        ]
        for a in range(p)
    ]
    return m, c


def test_moments_match_an_independent_calculation() -> None:
    replay = record_scripted()
    for row in range(3):
        m, c = explicit_moments(STATE_PARTICLES[row], STATE_WEIGHTS[row])
        np.testing.assert_allclose(
            replay.arrays["mean"][row], m, rtol=1e-12, atol=1e-15
        )
        np.testing.assert_allclose(replay.arrays["cov"][row], c, rtol=1e-12)
    assert abs(replay.arrays["cov"][1][0, 1]) > 0


def test_row_zero_holds_sentinels() -> None:
    arrays = record_scripted().arrays
    assert arrays["ess"][0] == 5.0
    assert arrays["log_evidence"][0] == 0.0
    m, c = explicit_moments(STATE_PARTICLES[0], STATE_WEIGHTS[0])
    np.testing.assert_allclose(arrays["mean"][0], m, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(arrays["cov"][0], c, rtol=1e-12)
    expected = thin(
        np.array(STATE_PARTICLES[0]), np.array(STATE_WEIGHTS[0]), CLOUD_SIZE
    )
    assert np.array_equal(arrays["cloud"][0], expected)


def test_ess_rows_follow_the_weights() -> None:
    ess = record_scripted().arrays["ess"]
    for row in (1, 2):
        expected = 1 / sum(w * w for w in STATE_WEIGHTS[row])
        np.testing.assert_allclose(ess[row], expected, rtol=1e-12)


def test_diagnostics_and_evidence_are_copied() -> None:
    arrays = record_scripted().arrays
    for key, expected in (
        ("tempering_stages", [3, 1]),
        ("mh_proposals", [24, 8]),
        ("mh_acceptances", [6, 3]),
    ):
        assert arrays[key].dtype == np.int64
        assert arrays[key].tolist() == expected
    assert arrays["log_evidence"].tolist() == [0.0, 5.0, 11.0]


def test_predictions_are_moments_of_predictions() -> None:
    def predict(p):
        return np.column_stack([p[:, 0] ** 2, np.exp(p[:, 1])])

    arrays = record_scripted(predict=predict).arrays
    for row in range(3):
        values = predict(np.array(STATE_PARTICLES[row]))
        w = STATE_WEIGHTS[row]
        for c in range(2):
            mean = sum(wi * v for wi, v in zip(w, values[:, c]))
            var = sum(wi * (v - mean) ** 2 for wi, v in zip(w, values[:, c]))
            np.testing.assert_allclose(
                arrays["predicted_mean"][row, c], mean, rtol=1e-12
            )
            np.testing.assert_allclose(
                arrays["predicted_std"][row, c], var**0.5, rtol=1e-12
            )
    at_mean = predict(arrays["mean"][1][None])[0]
    assert np.max(np.abs(arrays["predicted_mean"][1] - at_mean)) > 1e-6


def test_observations_are_what_the_filter_consumed() -> None:
    assert np.array_equal(
        record_scripted().arrays["observations"], [[0.97, 0.98], [0.9, 1.0]]
    )
    kept = np.array([[1.0, 2.0]])
    assert record_scripted(arrays={"observations": kept}).arrays["observations"] is kept


N_REAL = 64
CLOCKS = ClockArray(np.array([[p] for p in demo_1d.CLOCK_POSITIONS]), track_offset=1.0)


def real_filter():
    return build_particle_filter(
        InferenceConfig(
            clock_array=CLOCKS,
            noise=NoiseConfig(demo_1d.NOISE_STD),
            prior=PriorConfig((-8.0, 8.0), (0.005, 0.15)),
            n_particles=N_REAL,
            n_masses=1,
            seed=42,
        )
    )


def real_observations():
    truth = MassConfig(np.array([[demo_1d.TRUE_X]]), np.array([demo_1d.TRUE_M]))
    return simulate(
        SimulationConfig(
            clock_array=CLOCKS,
            ground_truth=truth,
            noise=NoiseConfig(demo_1d.NOISE_STD),
            n_observations=3,
            seed=42,
        )
    ).observations


def real_runs():
    obs = real_observations()
    recorded, plain = real_filter(), real_filter()
    replay = record_filter(
        recorded, obs, meta={}, arrays={}, predict=recorded.forward_model_batch
    )
    for o in obs:
        plain.update(o)
    return recorded, plain, replay


def test_recording_does_not_change_the_run() -> None:
    recorded, plain, replay = real_runs()
    fresh = real_filter()
    assert sum(d.mh_acceptances for d in plain.diagnostics_history) > 0
    assert plain.rng.bit_generator.state != fresh.rng.bit_generator.state
    assert len(recorded.history) == len(plain.history) == 4
    for a, b in zip(recorded.history, plain.history):
        assert np.array_equal(a.particles, b.particles)
        assert np.array_equal(a.weights, b.weights)
    assert recorded.rng.bit_generator.state == plain.rng.bit_generator.state
    assert replay.arrays["cloud"].shape[0] == 4


def test_final_row_matches_the_filter_estimate() -> None:
    recorded, _, replay = real_runs()
    estimate = recorded.estimate()
    arrays = replay.arrays
    np.testing.assert_allclose(arrays["mean"][-1], estimate["mean"], rtol=1e-12)
    np.testing.assert_allclose(
        np.sqrt(np.diag(arrays["cov"][-1])), estimate["std"], rtol=1e-12
    )
    np.testing.assert_allclose(arrays["ess"][-1], estimate["ess"], rtol=1e-12)
    assert arrays["cloud"].shape == (4, 64, 2)
    assert arrays["cloud"].dtype == np.float32


def test_a_used_filter_is_refused() -> None:
    pf = real_filter()
    pf.update(real_observations()[0])
    with pytest.raises(RuntimeError, match="fresh"):
        record_filter(pf, real_observations(), meta={}, arrays={})
    assert pf.state.observations_seen == 1
    assert len(pf.history) == 2


def test_cloud_rows_are_thinned_from_their_own_state() -> None:
    cloud = record_scripted().arrays["cloud"]
    # Picks worked out by hand from each row's cumulative weights and phi offsets.
    literals = {
        1: [[0.2, 0.055], [0.4, 0.060], [0.4, 0.060], [0.5, 0.066], [0.7, 0.070]],
        2: [
            [0.47, 0.0495],
            [0.47, 0.0495],
            [0.53, 0.0512],
            [0.53, 0.0512],
            [0.58, 0.052],
        ],
    }
    for row, expected in literals.items():
        assert np.array_equal(cloud[row], np.array(expected, dtype=np.float32))
        direct = thin(
            np.array(STATE_PARTICLES[row]), np.array(STATE_WEIGHTS[row]), CLOUD_SIZE
        )
        assert np.array_equal(cloud[row], direct)


def test_cloud_is_capped_at_500_particles() -> None:
    n = CLOUD_SIZE + 1
    prior = ParticleState(np.linspace(0.0, 1.0, n)[:, None], np.full(n, 1.0 / n), 0)
    pf = ScriptedFilter([prior], [], [])
    replay = record_filter(pf, [], meta={}, arrays={})
    assert replay.arrays["cloud"].shape == (1, CLOUD_SIZE, 1)


# --- first_difference --------------------------------------------------------


def _pair(make=filter_fixture):
    meta, arrays = make()
    return Replay(meta, arrays), Replay(
        dict(meta), {k: v.copy() for k, v in arrays.items()}
    )


def _scaled(replay: Replay, key: str, index: tuple, factor: float) -> None:
    replay.arrays[key][index] = replay.arrays[key][index] * factor


def test_equal_replays_have_no_difference() -> None:
    reference, other = _pair()
    assert first_difference(reference, other) is None
    other.meta["created_with"] = {k: "elsewhere" for k in CREATED}
    assert first_difference(reference, other) is None
    reference, other = _pair(model_fixture)
    assert first_difference(reference, other) is None


def test_meta_difference_is_reported_first() -> None:
    reference, other = _pair()
    other.meta["seed"] = 8
    other.arrays["mean"][0, 0] += 1.0
    found = first_difference(reference, other)
    assert found is not None
    assert (found.key, found.row) == ("meta.seed", None)


def test_meta_difference_names_the_first_field_in_sorted_order() -> None:
    reference, other = _pair()
    order = list(reference.meta)
    assert order.index("seed") < order.index("noise_std")
    other.meta["seed"] = 8
    other.meta["noise_std"] = 0.5
    found = first_difference(reference, other)
    assert found is not None
    assert found.key == "meta.noise_std"


@pytest.mark.parametrize(
    "lacking",
    [
        pytest.param("other", id="other_lacks"),
        pytest.param("reference", id="reference_lacks"),
    ],
)
def test_a_meta_key_missing_on_one_side_is_named(lacking: str) -> None:
    reference, other = _pair()
    del (other if lacking == "other" else reference).meta["cloud_size"]
    found = first_difference(reference, other)
    assert found is not None
    assert found.key == "meta.cloud_size"


def test_key_set_difference() -> None:
    reference, other = _pair()
    del other.arrays["ess"]
    found = first_difference(reference, other)
    assert found is not None
    assert (found.key, found.row) == ("keys", None)
    found = first_difference(other, reference)
    assert found is not None
    assert found.key == "keys"


def test_shape_difference() -> None:
    reference, other = _pair()
    other.arrays["mean"] = other.arrays["mean"][:-1]
    found = first_difference(reference, other)
    assert found is not None
    assert (found.key, found.row) == ("mean", None)


def test_dtype_difference() -> None:
    reference, other = _pair()
    other.arrays["mean"] = other.arrays["mean"].astype(np.float32)
    found = first_difference(reference, other)
    assert found is not None
    assert (found.key, found.row) == ("mean", None)


def test_integer_arrays_compare_exactly() -> None:
    reference, other = _pair()
    other.arrays["mh_proposals"][1] += 1
    found = first_difference(reference, other)
    assert found == ("mh_proposals", 1, 1.0)


def test_an_integer_gap_is_measured_in_integers() -> None:
    reference, other = _pair()
    reference.arrays["mh_proposals"][0] = 2**53
    other.arrays["mh_proposals"][0] = 2**53 + 1
    found = first_difference(reference, other)
    assert found == ("mh_proposals", 0, 1.0)


@pytest.mark.parametrize(
    ("factor", "different"),
    [
        pytest.param(1 + 1e-10, False, id="inside"),
        pytest.param(1 + 1e-8, True, id="outside"),
    ],
)
def test_float_tolerance(factor: float, different: bool) -> None:
    reference, other = _pair()
    _scaled(other, "mean", (2, 0), factor)
    found = first_difference(reference, other)
    if not different:
        assert found is None
        return
    assert found is not None
    assert (found.key, found.row) == ("mean", 2)


@pytest.mark.parametrize(
    ("factor", "different"),
    [
        pytest.param(1 + 1e-10, False, id="inside"),
        pytest.param(1 + 1e-8, True, id="outside"),
    ],
)
def test_inputs_use_the_same_tolerance(factor: float, different: bool) -> None:
    reference, other = _pair()
    _scaled(other, "observations", (1, 0), factor)
    found = first_difference(reference, other)
    if not different:
        assert found is None
        return
    assert found is not None
    assert (found.key, found.row) == ("observations", 1)


@pytest.mark.parametrize(
    ("factor", "different"),
    [
        pytest.param(1 + 1e-6, False, id="inside"),
        pytest.param(1 + 1e-4, True, id="outside"),
    ],
)
def test_cloud_tolerance(factor: float, different: bool) -> None:
    reference, other = _pair()
    _scaled(other, "cloud", (1, 0, 0), factor)
    found = first_difference(reference, other)
    if not different:
        assert found is None
        return
    assert found is not None
    assert (found.key, found.row) == ("cloud", 1)


def test_the_earliest_row_is_reported() -> None:
    reference, other = _pair()
    other.arrays["mean"][1, 0] += 0.25
    other.arrays["mean"][2, 1] += 0.5
    found = first_difference(reference, other)
    assert found is not None
    assert (found.key, found.row) == ("mean", 1)
    assert found.largest == pytest.approx(0.5)


def test_keys_are_compared_in_a_fixed_order() -> None:
    reference, other = _pair()
    other.arrays["cov"][1, 0, 0] += 1.0
    other.arrays["mean"][1, 0] += 1.0
    found = first_difference(reference, other)
    assert found is not None
    assert found.key == "cov"
    other.arrays["observations"][0, 0] += 1.0
    found = first_difference(reference, other)
    assert found is not None
    assert found.key == "observations"
