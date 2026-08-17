"""Release-manifest contracts for the echolocation population study."""

import copy
import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
from numpy.testing import assert_allclose

import clocks._reliability as reliability
from clocks._reliability import (
    encode_manifest,
    generate_release_manifest,
    load_manifest,
    validate_manifest,
    write_manifest,
)
from clocks._scenarios import build_head_lattice
from clocks._support import point_mass_support_mask

# Independent preregistration literals: do not import these from production.
TEST_MASTER_SEED = 20260817
TEST_N_STRATA = 6
TEST_CASES_PER_STRATUM = 64
TEST_RANGE_R_BOUNDS = (2.0, 8.0)
TEST_MASS_BOUNDS = (0.02, 0.08)
TEST_R_HEAD = math.sqrt(3.0)
TEST_CASE_COUNT = TEST_N_STRATA * TEST_CASES_PER_STRATUM


@pytest.fixture(scope="module")
def release_document() -> dict[str, object]:
    return json.loads(encode_manifest(generate_release_manifest()))


def _semantic_sha256(document: dict[str, object]) -> str:
    payload = {
        key: value for key, value in document.items() if key != "semantic_sha256"
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _changed(
    document: dict[str, object], mutation: Callable[[dict[str, object]], None]
) -> dict[str, object]:
    changed = copy.deepcopy(document)
    mutation(changed)
    return changed


def _assert_invalid(
    document: dict[str, object],
    mutation: Callable[[dict[str, object]], None],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        validate_manifest(_changed(document, mutation))


def test_generate_release_manifest_has_exact_identity_and_population() -> None:
    manifest = generate_release_manifest()

    assert manifest["schema_version"] == 1
    assert manifest["study_version"] == 1
    assert manifest["study"] == "echolocation_population"
    assert manifest["identity"] == "echolocation_population_v1_release"
    assert manifest["status"] == "release"
    assert manifest["master_seed"] == TEST_MASTER_SEED

    population = manifest["population"]
    assert population["n_cases"] == TEST_CASE_COUNT
    assert population["case_order"] == "stratum_major_then_case"
    assert population["range_r"]["n_strata"] == TEST_N_STRATA
    assert population["range_r"]["cases_per_stratum"] == TEST_CASES_PER_STRATUM
    assert population["range_r"]["bounds"] == TEST_RANGE_R_BOUNDS
    assert population["mass"]["bounds"] == TEST_MASS_BOUNDS


def test_release_cases_have_stable_unique_ids_and_stratum_major_order(
    release_document: dict[str, object],
) -> None:
    cases = release_document["cases"]
    assert len(cases) == TEST_CASE_COUNT
    expected_ids = [
        f"echolocation-population-v1-release-s{stratum:02d}-c{within:03d}"
        for stratum in range(TEST_N_STRATA)
        for within in range(TEST_CASES_PER_STRATUM)
    ]

    assert [case["case_id"] for case in cases] == expected_ids
    assert len(set(expected_ids)) == TEST_CASE_COUNT
    assert [case["case_index"] for case in cases] == list(range(TEST_CASE_COUNT))
    assert [case["stratum_index"] for case in cases] == [
        stratum
        for stratum in range(TEST_N_STRATA)
        for _ in range(TEST_CASES_PER_STRATUM)
    ]
    assert [case["stratum_case_index"] for case in cases] == [
        within for _ in range(TEST_N_STRATA) for within in range(TEST_CASES_PER_STRATUM)
    ]


def test_release_population_is_realized_from_declared_log_distributions(
    release_document: dict[str, object],
) -> None:
    population = release_document["population"]
    range_spec = population["range_r"]
    mass_spec = population["mass"]
    direction_spec = population["direction"]
    edges = range_spec["stratum_edges"]
    expected_edges = np.geomspace(
        TEST_RANGE_R_BOUNDS[0],
        TEST_RANGE_R_BOUNDS[1],
        TEST_N_STRATA + 1,
    )
    expected_edges[0] = TEST_RANGE_R_BOUNDS[0]
    expected_edges[-1] = TEST_RANGE_R_BOUNDS[1]

    assert range_spec["distribution"] == "log_uniform_stratified"
    assert range_spec["stratum_allocation"] == "equal"
    assert (
        range_spec["stratum_boundary_policy"] == "left_closed_right_open_final_closed"
    )
    assert edges[0] == 2.0
    assert edges[-1] == 8.0
    assert_allclose(edges, expected_edges, rtol=0.0, atol=0.0)
    assert mass_spec["distribution"] == "log_uniform"
    assert direction_spec == {
        "dimension": 3,
        "distribution": "uniform_sphere",
    }

    for case in release_document["cases"]:
        stratum = case["stratum_index"]
        direction = np.asarray(case["direction"])
        position = np.asarray(case["position"])
        range_r = case["range_r"]
        mass = case["mass"]
        assert direction.shape == (3,)
        assert_allclose(np.linalg.norm(direction), 1.0, rtol=0.0, atol=2e-15)
        assert_allclose(
            position,
            direction * range_r * TEST_R_HEAD,
            rtol=0.0,
            atol=0.0,
        )
        assert edges[stratum] <= range_r < edges[stratum + 1]
        assert TEST_MASS_BOUNDS[0] <= mass < TEST_MASS_BOUNDS[1]


def test_release_truths_are_in_actual_head_physical_support(
    release_document: dict[str, object],
) -> None:
    particles = np.array(
        [case["position"] + [case["mass"]] for case in release_document["cases"]]
    )
    valid = point_mass_support_mask(
        particles,
        n_masses=1,
        n_dims=3,
        clock_array=build_head_lattice(),
        position_range=(-16.0, 16.0),
        mass_range=(0.005, 0.15),
    )

    assert valid.shape == (TEST_CASE_COUNT,)
    assert np.all(valid)


def test_release_stream_seeds_are_nonnegative_non_bool_and_globally_disjoint(
    release_document: dict[str, object],
) -> None:
    stream_seeds = [release_document["analysis_seed"]]
    for case in release_document["cases"]:
        stream_seeds.extend(
            case[name]
            for name in ("parameter_seed", "observation_seed", "inference_seed")
        )

    assert all(type(seed) is int and seed >= 0 for seed in stream_seeds)
    assert len(stream_seeds) == 1 + 3 * TEST_CASE_COUNT
    assert len(set(stream_seeds)) == len(stream_seeds)
    assert release_document["analysis_seed"] != TEST_MASTER_SEED


def test_release_records_exact_generator_head_controls_and_analysis(
    release_document: dict[str, object],
) -> None:
    assert release_document["generator"] == {
        "bit_generator": "PCG64",
        "generator": "numpy.random.Generator",
        "numpy_version": np.__version__,
        "seed_sequence": "numpy.random.SeedSequence",
        "spawn_policy": (
            "root.spawn(385): analysis then stratum-major cases; "
            "case.spawn(3): parameter, observation, inference"
        ),
    }
    assert release_document["head"] == {
        "clock_count": 27,
        "geometry": "3x3x3_cubic_lattice",
        "positions": build_head_lattice().positions.tolist(),
        "r_head": TEST_R_HEAD,
        "track_offset": 0.0,
    }
    assert release_document["controls"] == {
        "ess_target": 0.9,
        "n_observations": 80,
        "n_particles": 6000,
        "noise_std": 0.001,
        "proposal_scale": 1.5,
        "rejuvenation_steps": 1,
    }
    assert release_document["acceptance"] == {
        "mass_error_max": 0.04,
        "position_error_max": 1.0,
        "threshold_policy": "inclusive",
    }
    assert release_document["intervals"] == {
        "overall": {
            "confidence_level": 0.95,
            "method": "stratified_bootstrap",
            "quantile_method": "linear",
            "resamples": 10_000,
        },
        "strata": {
            "confidence_level": 0.95,
            "count": 6,
            "method": "wilson",
        },
    }


def test_generate_release_manifest_is_deeply_immutable() -> None:
    manifest = generate_release_manifest()

    assert isinstance(manifest, MappingProxyType)
    with pytest.raises(TypeError):
        manifest["status"] = "development"
    with pytest.raises(TypeError):
        manifest["population"]["n_cases"] = 0
    with pytest.raises(TypeError):
        manifest["cases"][0]["mass"] = 1.0
    with pytest.raises(TypeError):
        manifest["cases"][0]["direction"][0] = 0.0


def test_generation_is_canonical_and_byte_stable() -> None:
    first = generate_release_manifest()
    second = generate_release_manifest()
    first_bytes = encode_manifest(first).encode("utf-8")
    second_bytes = encode_manifest(second).encode("utf-8")

    assert first is not second
    assert first_bytes == second_bytes
    assert first_bytes.endswith(b"\n")
    assert first_bytes.count(b"\n") == 1
    assert json.loads(first_bytes)["semantic_sha256"] == first["semantic_sha256"]


def test_semantic_hash_excludes_only_itself_and_covers_payload(
    release_document: dict[str, object],
) -> None:
    assert len(release_document["semantic_sha256"]) == 64
    assert release_document["semantic_sha256"] == _semantic_sha256(release_document)

    changed = copy.deepcopy(release_document)
    changed["semantic_sha256"] = "0" * 64
    assert _semantic_sha256(changed) == _semantic_sha256(release_document)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("study_version", 2),
        ("study", "other"),
        ("identity", "echolocation_population_v1_development"),
        ("status", "development"),
        ("master_seed", TEST_MASTER_SEED + 1),
    ],
)
def test_validate_manifest_rejects_top_level_semantic_mutations(
    release_document: dict[str, object], field: str, value: object
) -> None:
    _assert_invalid(
        release_document,
        lambda document: document.__setitem__(field, value),
        field,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bit_generator", "MT19937"),
        ("generator", "numpy.random.RandomState"),
        ("numpy_version", "0.0.0"),
        ("seed_sequence", "seed"),
        ("spawn_policy", "independent roots"),
    ],
)
def test_validate_manifest_rejects_generator_metadata_mutations(
    release_document: dict[str, object], field: str, value: object
) -> None:
    _assert_invalid(
        release_document,
        lambda document: document["generator"].__setitem__(field, value),
        rf"generator\.{field}",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("clock_count", 26),
        ("geometry", "sphere"),
        ("positions", [[0.0, 0.0, 0.0]] * 27),
        ("r_head", 2.0),
        ("track_offset", 1.0),
    ],
)
def test_validate_manifest_rejects_head_geometry_mutations(
    release_document: dict[str, object], field: str, value: object
) -> None:
    _assert_invalid(
        release_document,
        lambda document: document["head"].__setitem__(field, value),
        rf"head\.{field}",
    )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("n_cases",), 383, "population.n_cases"),
        (("case_order",), "random", "population.case_order"),
        (("range_r", "distribution"), "uniform", "population.range_r.distribution"),
        (("range_r", "bounds"), [2.0, 9.0], "population.range_r.bounds"),
        (("range_r", "n_strata"), 5, "population.range_r.n_strata"),
        (
            ("range_r", "cases_per_stratum"),
            63,
            "population.range_r.cases_per_stratum",
        ),
        (
            ("range_r", "stratum_allocation"),
            "unequal",
            "population.range_r.stratum_allocation",
        ),
        (
            ("range_r", "stratum_boundary_policy"),
            "closed",
            "population.range_r.stratum_boundary_policy",
        ),
        (("mass", "distribution"), "uniform", "population.mass.distribution"),
        (("mass", "bounds"), [0.01, 0.08], "population.mass.bounds"),
        (("direction", "distribution"), "cube", "population.direction.distribution"),
        (("direction", "dimension"), 2, "population.direction.dimension"),
    ],
)
def test_validate_manifest_rejects_population_spec_mutations(
    release_document: dict[str, object],
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    def mutate(document: dict[str, object]) -> None:
        target = document["population"]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    _assert_invalid(release_document, mutate, message)


def test_validate_manifest_rejects_non_log_spaced_or_wrong_endpoint_edges(
    release_document: dict[str, object],
) -> None:
    for index, value in ((0, 2.1), (3, 4.1), (6, 7.9)):
        _assert_invalid(
            release_document,
            lambda document, index=index, value=value: document["population"][
                "range_r"
            ]["stratum_edges"].__setitem__(index, value),
            "population.range_r.stratum_edges",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_observations", 79),
        ("noise_std", 0.002),
        ("n_particles", 5999),
        ("ess_target", 0.8),
        ("rejuvenation_steps", 2),
        ("proposal_scale", 2.0),
    ],
)
def test_validate_manifest_rejects_smc_and_observation_control_mutations(
    release_document: dict[str, object], field: str, value: object
) -> None:
    _assert_invalid(
        release_document,
        lambda document: document["controls"].__setitem__(field, value),
        rf"controls\.{field}",
    )


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("acceptance", "position_error_max", 1.1),
        ("acceptance", "mass_error_max", 0.05),
        ("acceptance", "threshold_policy", "strict"),
        ("strata", "method", "wald"),
        ("strata", "count", 5),
        ("strata", "confidence_level", 0.9),
        ("overall", "method", "bootstrap"),
        ("overall", "resamples", 9999),
        ("overall", "quantile_method", "nearest"),
        ("overall", "confidence_level", 0.9),
    ],
)
def test_validate_manifest_rejects_tolerance_and_interval_mutations(
    release_document: dict[str, object], section: str, field: str, value: object
) -> None:
    def mutate(document: dict[str, object]) -> None:
        target = (
            document["acceptance"]
            if section == "acceptance"
            else document["intervals"][section]
        )
        target[field] = value

    prefix = "acceptance" if section == "acceptance" else f"intervals.{section}"
    _assert_invalid(release_document, mutate, rf"{prefix}\.{field}")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("case_id", "wrong"),
        ("case_index", 1),
        ("stratum_index", 1),
        ("stratum_case_index", 1),
    ],
)
def test_validate_manifest_rejects_case_identity_or_order_mutations(
    release_document: dict[str, object], field: str, value: object
) -> None:
    _assert_invalid(
        release_document,
        lambda document: document["cases"][0].__setitem__(field, value),
        rf"cases\[0\]\.{field}",
    )


def test_validate_manifest_rejects_direction_shape_norm_and_position_mutations(
    release_document: dict[str, object],
) -> None:
    _assert_invalid(
        release_document,
        lambda document: document["cases"][0].__setitem__("direction", [1.0, 0.0]),
        r"cases\[0\]\.direction",
    )
    _assert_invalid(
        release_document,
        lambda document: document["cases"][0].__setitem__("direction", [1.0, 1.0, 1.0]),
        r"cases\[0\]\.direction",
    )
    _assert_invalid(
        release_document,
        lambda document: document["cases"][0]["position"].__setitem__(0, 0.0),
        r"cases\[0\]\.position",
    )


def test_validate_manifest_rejects_case_range_mass_stratum_and_bounds(
    release_document: dict[str, object],
) -> None:
    def set_range(document: dict[str, object], value: float) -> None:
        case = document["cases"][0]
        case["range_r"] = value
        case["position"] = [
            component * value * TEST_R_HEAD for component in case["direction"]
        ]

    _assert_invalid(
        release_document,
        lambda document: set_range(document, 1.99),
        r"cases\[0\]\.range_r",
    )
    _assert_invalid(
        release_document,
        lambda document: set_range(document, 4.0),
        r"cases\[0\]\.range_r.*stratum",
    )
    _assert_invalid(
        release_document,
        lambda document: document["cases"][0].__setitem__("mass", 0.019),
        r"cases\[0\]\.mass",
    )
    _assert_invalid(
        release_document,
        lambda document: document["cases"][0].__setitem__("mass", 0.081),
        r"cases\[0\]\.mass",
    )


def test_validate_manifest_accepts_explicit_final_range_and_mass_endpoints(
    release_document: dict[str, object],
) -> None:
    changed = copy.deepcopy(release_document)
    case = changed["cases"][-1]
    case["range_r"] = TEST_RANGE_R_BOUNDS[1]
    case["mass"] = TEST_MASS_BOUNDS[1]
    case["position"] = [
        component * case["range_r"] * TEST_R_HEAD for component in case["direction"]
    ]
    changed["semantic_sha256"] = _semantic_sha256(changed)

    validate_manifest(changed)


@pytest.mark.parametrize(
    "field", ["parameter_seed", "observation_seed", "inference_seed"]
)
@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_validate_manifest_rejects_invalid_case_seeds(
    release_document: dict[str, object], field: str, value: object
) -> None:
    _assert_invalid(
        release_document,
        lambda document: document["cases"][0].__setitem__(field, value),
        rf"cases\[0\]\.{field}",
    )


@pytest.mark.parametrize(
    "field", ["parameter_seed", "observation_seed", "inference_seed"]
)
def test_validate_manifest_rejects_changed_release_case_seeds(
    release_document: dict[str, object], field: str
) -> None:
    _assert_invalid(
        release_document,
        lambda document: document["cases"][0].__setitem__(
            field, document["cases"][0][field] + 1
        ),
        rf"cases\[0\]\.{field}",
    )


def test_validate_manifest_rejects_duplicate_stream_seeds(
    release_document: dict[str, object],
) -> None:
    _assert_invalid(
        release_document,
        lambda document: document["cases"][0].__setitem__(
            "parameter_seed", document["analysis_seed"]
        ),
        "duplicate.*seed",
    )
    _assert_invalid(
        release_document,
        lambda document: document["cases"][1].__setitem__(
            "inference_seed", document["cases"][0]["observation_seed"]
        ),
        "duplicate.*seed",
    )


@pytest.mark.parametrize("value", [-1, 1.5, True, TEST_MASTER_SEED])
def test_validate_manifest_rejects_invalid_or_nondistinct_analysis_seed(
    release_document: dict[str, object], value: object
) -> None:
    _assert_invalid(
        release_document,
        lambda document: document.__setitem__("analysis_seed", value),
        "analysis_seed",
    )


def test_validate_manifest_rejects_changed_release_analysis_seed(
    release_document: dict[str, object],
) -> None:
    _assert_invalid(
        release_document,
        lambda document: document.__setitem__(
            "analysis_seed", document["analysis_seed"] + 1
        ),
        "analysis_seed",
    )


def test_validate_manifest_rejects_missing_extra_and_duplicate_cases(
    release_document: dict[str, object],
) -> None:
    _assert_invalid(
        release_document,
        lambda document: document["cases"].pop(),
        "cases",
    )
    _assert_invalid(
        release_document,
        lambda document: document["cases"].append(copy.deepcopy(document["cases"][0])),
        "cases",
    )
    _assert_invalid(
        release_document,
        lambda document: document["cases"].__setitem__(
            -1, copy.deepcopy(document["cases"][-2])
        ),
        r"cases\[383\]\.case_id|duplicate.*case",
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda document: document.__setitem__("master_seed", True), "master_seed"),
        (
            lambda document: document["controls"].__setitem__("n_observations", True),
            "controls.n_observations",
        ),
        (
            lambda document: document["controls"].__setitem__("noise_std", True),
            "controls.noise_std",
        ),
        (
            lambda document: document["cases"][0].__setitem__("range_r", True),
            r"cases\[0\].range_r",
        ),
        (
            lambda document: document["cases"][0]["direction"].__setitem__(0, True),
            r"cases\[0\].direction",
        ),
    ],
)
def test_validate_manifest_never_accepts_bool_as_a_number(
    release_document: dict[str, object],
    mutation: Callable[[dict[str, object]], None],
    message: str,
) -> None:
    _assert_invalid(release_document, mutation, message)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, 1.0 + 0.0j])
def test_validate_and_encode_reject_nonfinite_or_complex_before_json_encoding(
    release_document: dict[str, object], value: object
) -> None:
    changed = _changed(
        release_document,
        lambda document: document["cases"][0].__setitem__("mass", value),
    )
    with pytest.raises(ValueError, match=r"cases\[0\]\.mass"):
        validate_manifest(changed)
    with pytest.raises(ValueError, match=r"cases\[0\]\.mass"):
        encode_manifest(changed)


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ((), "manifest"),
        (("generator",), "generator"),
        (("head",), "head"),
        (("population",), "population"),
        (("population", "range_r"), "population.range_r"),
        (("population", "mass"), "population.mass"),
        (("population", "direction"), "population.direction"),
        (("controls",), "controls"),
        (("acceptance",), "acceptance"),
        (("intervals",), "intervals"),
        (("intervals", "strata"), "intervals.strata"),
        (("intervals", "overall"), "intervals.overall"),
        (("cases", 0), r"cases\[0\]"),
    ],
)
def test_validate_manifest_rejects_unknown_fields_at_every_object_level(
    release_document: dict[str, object], path: tuple[object, ...], message: str
) -> None:
    def mutate(document: dict[str, object]) -> None:
        target = document
        for key in path:
            target = target[key]
        target["unexpected"] = 1

    _assert_invalid(release_document, mutate, message)


@pytest.mark.parametrize(
    ("path", "field", "message"),
    [
        ((), "study_version", "manifest.*missing.*study_version"),
        (("generator",), "bit_generator", "generator.*missing.*bit_generator"),
        (("head",), "positions", "head.*missing.*positions"),
        (("population",), "range_r", "population.*missing.*range_r"),
        (
            ("population", "range_r"),
            "stratum_edges",
            "population.range_r.*missing.*stratum_edges",
        ),
        (("population", "mass"), "bounds", "population.mass.*missing.*bounds"),
        (
            ("population", "direction"),
            "distribution",
            "population.direction.*missing.*distribution",
        ),
        (("controls",), "n_particles", "controls.*missing.*n_particles"),
        (
            ("acceptance",),
            "mass_error_max",
            "acceptance.*missing.*mass_error_max",
        ),
        (("intervals",), "overall", "intervals.*missing.*overall"),
        (
            ("intervals", "strata"),
            "method",
            "intervals.strata.*missing.*method",
        ),
        (
            ("intervals", "overall"),
            "resamples",
            "intervals.overall.*missing.*resamples",
        ),
        (("cases", 0), "case_id", r"cases\[0\].*missing.*case_id"),
    ],
)
def test_validate_manifest_rejects_missing_fields_at_every_object_level(
    release_document: dict[str, object],
    path: tuple[object, ...],
    field: str,
    message: str,
) -> None:
    def mutate(document: dict[str, object]) -> None:
        target = document
        for key in path:
            target = target[key]
        del target[field]

    _assert_invalid(release_document, mutate, message)


def test_validate_manifest_rejects_semantic_hash_mutations(
    release_document: dict[str, object],
) -> None:
    _assert_invalid(
        release_document,
        lambda document: document.__setitem__("semantic_sha256", "0" * 64),
        "semantic_sha256",
    )
    _assert_invalid(
        release_document,
        lambda document: document.__setitem__("semantic_sha256", True),
        "semantic_sha256",
    )


def test_write_and_load_manifest_are_canonical_hashed_and_immutable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nested" / "manifest.json"
    manifest = generate_release_manifest()

    write_manifest(path, manifest)
    loaded = load_manifest(path)

    assert path.read_bytes() == encode_manifest(manifest).encode("utf-8")
    assert encode_manifest(loaded) == encode_manifest(manifest)
    assert isinstance(loaded, MappingProxyType)
    with pytest.raises(TypeError):
        loaded["cases"][0]["mass"] = 1.0


def test_load_manifest_verifies_hash_before_exposing_data(
    tmp_path: Path, release_document: dict[str, object]
) -> None:
    path = tmp_path / "manifest.json"
    changed = copy.deepcopy(release_document)
    changed["semantic_sha256"] = "0" * 64
    path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ValueError, match="semantic_sha256"):
        load_manifest(path)


def test_write_manifest_preserves_destination_on_validation_failure(
    tmp_path: Path, release_document: dict[str, object]
) -> None:
    path = tmp_path / "manifest.json"
    sentinel = b"sentinel\n"
    path.write_bytes(sentinel)
    invalid = _changed(
        release_document,
        lambda document: document.__setitem__("status", "development"),
    )

    with pytest.raises(ValueError, match="status"):
        write_manifest(path, invalid)

    assert path.read_bytes() == sentinel
    assert list(tmp_path.iterdir()) == [path]


def test_write_manifest_preserves_destination_on_serialization_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "manifest.json"
    sentinel = b"sentinel\n"
    path.write_bytes(sentinel)
    manifest = generate_release_manifest()

    def fail_serialization(*args: object, **kwargs: object) -> str:
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(reliability.json, "dumps", fail_serialization)
    with pytest.raises(RuntimeError, match="serialization failed"):
        write_manifest(path, manifest)

    assert path.read_bytes() == sentinel
    assert list(tmp_path.iterdir()) == [path]


def test_write_manifest_cleans_temporary_file_if_atomic_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "manifest.json"
    sentinel = b"sentinel\n"
    path.write_bytes(sentinel)

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(reliability.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        write_manifest(path, generate_release_manifest())

    assert path.read_bytes() == sentinel
    assert list(tmp_path.iterdir()) == [path]
