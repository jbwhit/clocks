"""Release-manifest contracts for the echolocation population study."""

import copy
import hashlib
import json
import math
import os
from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
from numpy.testing import assert_allclose

import clocks._reliability as reliability
import clocks._rng as rng_module
import clocks._scenarios as scenarios
from clocks._reliability import (
    encode_manifest,
    generate_release_manifest,
    load_manifest,
    require_frozen_release,
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
TEST_RANGE_EDGE_HEX = (
    "0x1.0000000000000p+1",
    "0x1.428a2f98d728bp+1",
    "0x1.965fea53d6e3dp+1",
    "0x1.ffffffffffffep+1",
    "0x1.428a2f98d728bp+2",
    "0x1.965fea53d6e3ap+2",
    "0x1.0000000000000p+3",
)
TEST_APPROVED_MANIFEST_BYTES = 189_258
TEST_APPROVED_MANIFEST_SHA256 = (
    "a4868fe22396987b2675c1a25f9d9b25c8a92fdec44a4402940e497dd1ba827e"
)
TEST_APPROVED_SEMANTIC_SHA256 = (
    "98a08d394ca249edb307a7ac78f30d6708180f716105f09b4768f9ff2630fcd4"
)


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


def _canonical_test_json(document: dict[str, object]) -> str:
    return (
        json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def _independent_seed(sequence: np.random.SeedSequence) -> int:
    words = sequence.generate_state(4, dtype=np.uint32)
    return sum(int(word) << (32 * index) for index, word in enumerate(words))


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
    expected_edges = np.array([float.fromhex(value) for value in TEST_RANGE_EDGE_HEX])

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


def test_release_freezes_exact_edge_bits_and_approved_canonical_bytes() -> None:
    manifest = generate_release_manifest()
    encoded = encode_manifest(manifest).encode("utf-8")
    edges = manifest["population"]["range_r"]["stratum_edges"]

    assert tuple(value.hex() for value in edges) == TEST_RANGE_EDGE_HEX
    assert len(encoded) == TEST_APPROVED_MANIFEST_BYTES
    assert hashlib.sha256(encoded).hexdigest() == TEST_APPROVED_MANIFEST_SHA256
    assert manifest["semantic_sha256"] == TEST_APPROVED_SEMANTIC_SHA256


@pytest.mark.parametrize("consumer", ["validate", "load"])
@pytest.mark.parametrize("sabotage", ["raise", "one_ulp"])
def test_frozen_manifest_consumers_are_runtime_geomspace_independent(
    tmp_path: Path,
    release_document: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    consumer: str,
    sabotage: str,
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(_canonical_test_json(release_document), encoding="utf-8")

    def fail_geomspace(*args: object, **kwargs: object) -> None:
        raise AssertionError("frozen validation must not call numpy.geomspace")

    def shifted_geomspace(*args: object, **kwargs: object) -> np.ndarray:
        edges = np.array([float.fromhex(value) for value in TEST_RANGE_EDGE_HEX])
        edges[3] = np.nextafter(edges[3], math.inf)
        return edges

    replacement = fail_geomspace if sabotage == "raise" else shifted_geomspace
    monkeypatch.setattr(reliability.np, "geomspace", replacement)

    if consumer == "validate":
        validate_manifest(release_document)
    else:
        loaded = load_manifest(path)
        assert loaded["semantic_sha256"] == TEST_APPROVED_SEMANTIC_SHA256


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


def test_release_stream_seeds_replay_the_declared_hierarchical_spawn_recipe(
    release_document: dict[str, object],
) -> None:
    root = np.random.SeedSequence(TEST_MASTER_SEED)
    root_children = root.spawn(TEST_CASE_COUNT + 1)
    analysis_child = root_children[0]

    assert analysis_child.spawn_key == (0,)
    assert release_document["analysis_seed"] == _independent_seed(analysis_child)
    expected_seeds = [release_document["analysis_seed"]]
    for case_index, (case, case_child) in enumerate(
        zip(release_document["cases"], root_children[1:])
    ):
        assert case_child.spawn_key == (case_index + 1,)
        stream_children = case_child.spawn(3)
        assert [child.spawn_key for child in stream_children] == [
            (case_index + 1, stream_index) for stream_index in range(3)
        ]
        actual = [
            case["parameter_seed"],
            case["observation_seed"],
            case["inference_seed"],
        ]
        expected = [_independent_seed(child) for child in stream_children]
        assert actual == expected
        expected_seeds.extend(expected)

    assert len(expected_seeds) == 1 + 3 * TEST_CASE_COUNT
    assert len(set(expected_seeds)) == len(expected_seeds)


def test_release_parameters_replay_the_declared_log_uniform_draw_recipe(
    release_document: dict[str, object],
) -> None:
    edges = np.array([float.fromhex(value) for value in TEST_RANGE_EDGE_HEX])

    for case in release_document["cases"]:
        rng = np.random.Generator(np.random.PCG64(case["parameter_seed"]))
        direction_draw = rng.normal(size=3)
        direction_norm = math.hypot(*(float(component) for component in direction_draw))
        while direction_norm == 0.0:
            direction_draw = rng.normal(size=3)
            direction_norm = math.hypot(
                *(float(component) for component in direction_draw)
            )
        direction = direction_draw / direction_norm
        stratum = case["stratum_index"]
        range_r = math.exp(
            float(rng.uniform(math.log(edges[stratum]), math.log(edges[stratum + 1])))
        )
        mass = math.exp(
            float(
                rng.uniform(
                    math.log(TEST_MASS_BOUNDS[0]),
                    math.log(TEST_MASS_BOUNDS[1]),
                )
            )
        )
        position = direction * range_r * TEST_R_HEAD

        assert_allclose(case["direction"], direction, rtol=0.0, atol=0.0)
        assert case["range_r"] == range_r
        assert case["mass"] == mass
        assert_allclose(case["position"], position, rtol=0.0, atol=0.0)


def test_generation_uses_explicit_pcg64_without_default_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual_pcg64 = np.random.PCG64
    pcg64_seeds: list[int] = []

    def recording_pcg64(seed: int) -> np.random.BitGenerator:
        pcg64_seeds.append(seed)
        return actual_pcg64(seed)

    def reject_default_rng(*args: object, **kwargs: object) -> None:
        raise AssertionError("generation must construct the declared PCG64 explicitly")

    monkeypatch.setattr(reliability.np.random, "PCG64", recording_pcg64)
    monkeypatch.setattr(reliability.np.random, "default_rng", reject_default_rng)
    manifest = generate_release_manifest()

    assert pcg64_seeds == [case["parameter_seed"] for case in manifest["cases"]]


def test_release_records_exact_generator_head_controls_and_analysis(
    release_document: dict[str, object],
) -> None:
    assert release_document["generator"] == {
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


def test_canonical_manifest_never_embeds_the_runtime_numpy_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only that the version *string* is absent from the hashed payload.

    This deliberately does not claim independence from NumPy's RNG behaviour:
    patching ``np.__version__`` cannot test that. The streams are pinned instead
    by constructing the declared bit generator explicitly -- see
    ``test_observation_and_inference_streams_use_the_declared_bit_generator``.
    """
    original = encode_manifest(generate_release_manifest())
    assert "numpy_version" not in json.loads(original)["generator"]
    monkeypatch.setattr(np, "__version__", "runtime-version-must-not-be-hashed")
    regenerated = encode_manifest(generate_release_manifest())

    assert regenerated == original
    assert "runtime-version-must-not-be-hashed" not in regenerated
    validate_manifest(json.loads(original))


def test_load_and_validation_treat_recorded_truths_and_seeds_as_authoritative(
    tmp_path: Path,
    release_document: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_rng_regeneration(*args: object, **kwargs: object) -> None:
        raise AssertionError("validation must not regenerate frozen streams")

    path = tmp_path / "manifest.json"
    path.write_text(_canonical_test_json(release_document), encoding="utf-8")
    monkeypatch.setattr(reliability.np.random, "SeedSequence", reject_rng_regeneration)
    monkeypatch.setattr(reliability.np.random, "Generator", reject_rng_regeneration)
    monkeypatch.setattr(reliability.np.random, "PCG64", reject_rng_regeneration)
    monkeypatch.setattr(reliability.np.random, "default_rng", reject_rng_regeneration)

    validate_manifest(release_document)
    loaded = load_manifest(path)

    assert loaded["semantic_sha256"] == release_document["semantic_sha256"]


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
        ("parameter_draw_recipe", "linear interpolation"),
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


def test_non_final_stratum_upper_edge_is_exclusive_at_the_exact_edge(
    release_document: dict[str, object],
) -> None:
    """Pin the declared half-open boundary at the edge itself.

    Rejecting 4.0 from stratum 0 only shows the range is checked at all; the
    boundary policy is only pinned by the edge value, which must land in the
    next stratum and nowhere else.
    """
    edge = float.fromhex(TEST_RANGE_EDGE_HEX[1])

    def set_first_case_range(document: dict[str, object], value: float) -> None:
        case = document["cases"][0]
        case["range_r"] = value
        case["position"] = [
            component * value * TEST_R_HEAD for component in case["direction"]
        ]

    _assert_invalid(
        release_document,
        lambda document: set_first_case_range(document, edge),
        r"cases\[0\]\.range_r.*stratum",
    )
    # One ulp below the edge is the last value stratum 0 admits.
    inside = copy.deepcopy(release_document)
    set_first_case_range(inside, math.nextafter(edge, -math.inf))
    inside["semantic_sha256"] = _semantic_sha256(inside)
    validate_manifest(inside)


def test_validate_manifest_rejects_negative_zero_float_aliases(
    release_document: dict[str, object],
) -> None:
    """Negative zero would give one population two canonical forms and hashes."""
    _assert_invalid(
        release_document,
        lambda document: document["cases"][0]["direction"].__setitem__(0, -0.0),
        r"cases\[0\]\.direction\[0\].*negative zero",
    )
    _assert_invalid(
        release_document,
        lambda document: document["head"].__setitem__("track_offset", -0.0),
        r"head\.track_offset.*negative zero",
    )


def test_frozen_release_identity_rejects_a_different_valid_population(
    release_document: dict[str, object], tmp_path: Path
) -> None:
    """Semantic validity is not identity.

    A hand-edited population can stay inside every declared bound and rehash
    consistently. Only the pinned digest distinguishes the preregistered
    population from a plausible substitute wearing its name.
    """
    substitute = copy.deepcopy(release_document)
    case = substitute["cases"][0]
    case["mass"] = math.nextafter(case["mass"], math.inf)
    substitute["semantic_sha256"] = _semantic_sha256(substitute)

    validate_manifest(substitute)  # still semantically valid ...
    with pytest.raises(ValueError, match="reserved for the frozen population"):
        require_frozen_release(substitute)  # ... but not the frozen population

    path = tmp_path / "manifest.json"
    path.write_text(_canonical_test_json(substitute), encoding="utf-8")
    with pytest.raises(ValueError, match="reserved for the frozen population"):
        load_manifest(path)


def test_generated_release_matches_the_pinned_frozen_digest() -> None:
    manifest = generate_release_manifest()

    assert reliability.RELEASE_SEMANTIC_SHA256 == TEST_APPROVED_SEMANTIC_SHA256
    assert manifest["semantic_sha256"] == TEST_APPROVED_SEMANTIC_SHA256
    require_frozen_release(manifest)


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


def test_validate_manifest_reports_case_physical_support_failures(
    release_document: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_physical_support(*args: object, **kwargs: object) -> None:
        raise ValueError("forced physical rejection")

    monkeypatch.setattr(
        reliability, "_validate_echolocation_truth", reject_physical_support
    )
    with pytest.raises(
        ValueError,
        match=r"cases\[0\]\.physical_support.*forced physical rejection",
    ):
        validate_manifest(release_document)


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
def test_semantic_hash_catches_changed_authoritative_case_seeds(
    release_document: dict[str, object], field: str
) -> None:
    _assert_invalid(
        release_document,
        lambda document: document["cases"][0].__setitem__(
            field, document["cases"][0][field] + 1
        ),
        "semantic_sha256",
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


def test_semantic_hash_catches_changed_authoritative_analysis_seed(
    release_document: dict[str, object],
) -> None:
    _assert_invalid(
        release_document,
        lambda document: document.__setitem__(
            "analysis_seed", document["analysis_seed"] + 1
        ),
        "semantic_sha256",
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
    ("mutation", "message"),
    [
        (
            lambda document: document["head"].__setitem__("track_offset", -0.0),
            r"head\.track_offset",
        ),
        (
            lambda document: document["acceptance"].__setitem__(
                "position_error_max", 1
            ),
            r"acceptance\.position_error_max",
        ),
        (
            lambda document: document["head"]["positions"][0].__setitem__(0, -1),
            r"head\.positions\[0\]",
        ),
        (
            lambda document: document["controls"].__setitem__("n_observations", 80.0),
            r"controls\.n_observations",
        ),
    ],
)
def test_load_rejects_noncanonical_numeric_aliases_with_recomputed_hash(
    tmp_path: Path,
    release_document: dict[str, object],
    mutation: Callable[[dict[str, object]], None],
    message: str,
) -> None:
    changed = _changed(release_document, mutation)
    changed["semantic_sha256"] = _semantic_sha256(changed)
    path = tmp_path / "manifest.json"
    path.write_text(_canonical_test_json(changed), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_manifest(path)


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


@pytest.mark.parametrize(
    ("raw_mutation", "message"),
    [
        (
            lambda encoded: encoded.replace("{", '{"schema_version":2,', 1),
            r"manifest\.schema_version.*duplicate",
        ),
        (
            lambda encoded: encoded.replace(
                '"generator":{',
                '"generator":{"bit_generator":"MT19937",',
                1,
            ),
            r"manifest\.generator\.bit_generator.*duplicate",
        ),
        (
            lambda encoded: encoded.replace('"cases":[{', '"cases":[{"mass":0.01,', 1),
            r"manifest\.cases\[0\]\.mass.*duplicate",
        ),
        (
            lambda encoded: encoded.replace(
                "{", f'{{"semantic_sha256":"{"0" * 64}",', 1
            ),
            r"manifest\.semantic_sha256.*duplicate",
        ),
    ],
)
def test_load_manifest_rejects_duplicate_json_fields_before_validation(
    tmp_path: Path,
    release_document: dict[str, object],
    raw_mutation: Callable[[str], str],
    message: str,
) -> None:
    canonical = _canonical_test_json(release_document)
    duplicated = raw_mutation(canonical)
    assert json.loads(duplicated) == release_document
    path = tmp_path / "manifest.json"
    path.write_text(duplicated, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_manifest(path)


@pytest.mark.parametrize(
    "raw_mutation",
    [
        lambda encoded: encoded.removesuffix("\n"),
        lambda encoded: encoded + "\n",
        lambda encoded: " " + encoded,
        lambda encoded: (
            json.dumps(json.loads(encoded), indent=2, sort_keys=True) + "\n"
        ),
        lambda encoded: encoded.replace('"noise_std":0.001', '"noise_std":1e-3'),
    ],
)
def test_load_manifest_rejects_noncanonical_json_bytes(
    tmp_path: Path,
    release_document: dict[str, object],
    raw_mutation: Callable[[str], str],
) -> None:
    canonical = _canonical_test_json(release_document)
    noncanonical = raw_mutation(canonical)
    assert noncanonical != canonical
    assert json.loads(noncanonical) == release_document
    path = tmp_path / "manifest.json"
    path.write_text(noncanonical, encoding="utf-8")

    with pytest.raises(ValueError, match="canonical UTF-8"):
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


def test_write_manifest_fsyncs_the_file_then_replaces_then_fsyncs_the_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Durability needs both fsyncs, in that order.

    Flushing the temporary file makes its *contents* durable; the rename that
    publishes it lives in the parent directory and survives power loss only if
    that directory is fsynced too. Ordering matters: syncing the directory
    before the replace would make a rename that had not happened yet durable.
    """
    path = tmp_path / "manifest.json"
    events: list[str] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def recording_fsync(descriptor: int) -> None:
        events.append("fsync-directory" if os.path.isdir(descriptor) else "fsync-file")
        real_fsync(descriptor)

    def recording_replace(source: object, destination: object) -> None:
        events.append("replace")
        real_replace(source, destination)  # type: ignore[arg-type]

    monkeypatch.setattr(reliability.os, "fsync", recording_fsync)
    monkeypatch.setattr(reliability.os, "replace", recording_replace)
    write_manifest(path, generate_release_manifest())

    assert events == ["fsync-file", "replace", "fsync-directory"]
    assert load_manifest(path)["semantic_sha256"] == TEST_APPROVED_SEMANTIC_SHA256


def test_observation_and_inference_streams_use_the_declared_bit_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The manifest declares PCG64, so nothing may fall back to default_rng.

    ``numpy.random.default_rng`` documents its bit generator as an
    implementation detail, so archived seeds would stop reproducing archived
    observations if NumPy ever changed it -- silently, and only for runs made
    after the change.
    """
    constructed: list[object] = []
    real_pcg64 = np.random.PCG64

    def recording_pcg64(seed: object = None) -> np.random.BitGenerator:
        constructed.append(seed)
        return real_pcg64(seed)

    def reject_default_rng(*args: object, **kwargs: object) -> None:
        raise AssertionError("study streams must construct PCG64 explicitly")

    monkeypatch.setattr(rng_module.np.random, "PCG64", recording_pcg64)
    monkeypatch.setattr(np.random, "default_rng", reject_default_rng)

    result = scenarios.run_echolocation_case(
        truth_position=np.array([4.5, -3.75, 3.5]),
        truth_mass=0.065,
        observation_seed=101,
        inference_seed=202,
        n_particles=4,
        n_observations=1,
    )

    assert rng_module.BIT_GENERATOR_NAME == "PCG64"
    assert reliability._GENERATOR_METADATA["bit_generator"] == "PCG64"
    # Exactly the two declared streams, seeded with exactly the two given seeds.
    assert constructed == [101, 202]
    assert result.observation_seed == 101
    assert result.inference_seed == 202
