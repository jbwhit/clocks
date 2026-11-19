import json
import re
from pathlib import Path

import numpy as np
import pytest

from clocks._replay import (
    CLOUD_SIZE,
    DEMOS,
    FORMAT,
    SPATIAL_DIMENSIONS,
    VERSION,
    Replay,
    load_replay,
    provenance,
    save_replay,
    validate_replay,
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
    _invalid(meta, arrays, None)


def test_candidate_count_must_agree() -> None:
    meta, arrays = model_fixture()
    arrays["model_ks"] = np.array([1, 2, 3], I8)
    _invalid(meta, arrays, None)


@pytest.mark.parametrize(("name", "key"), EACH_FLOATING_KEY)
def test_non_finite_value_is_rejected(name: str, key: str) -> None:
    meta, arrays = FIXTURES[name]()
    arrays[key] = arrays[key].copy()
    arrays[key].flat[0] = np.nan
    _invalid(meta, arrays, _loose(key))


def test_descending_model_ks_is_rejected() -> None:
    meta, arrays = model_fixture()
    arrays["model_ks"] = np.array([2, 1], I8)
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
    np.save(path, np.zeros(3))
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
