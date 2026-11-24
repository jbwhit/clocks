# Replay Data (Pull Request 1): Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Commit, for each of the seven demos, an archive of what its packaged run produced, with tests that pin the archive to the code by its inputs and by bounded summaries, and a CI report that compares it in full without gating anything.

**Architecture:** A private module `src/clocks/_replay.py` holds the archive format (a validated `.npz` with a JSON `meta` string), the display-sample rule, a recorder that reads a particle filter's public history, and a full comparison. Each demo module gains `scenario()` and `record()` beside its untouched command. `scripts/record_replays.py` writes the archives and the report. Tests gate on scenario inputs and on posterior summaries after observations 1, 8 and last; a separate workflow reports full reproduction from three runner allocations.

**Tech Stack:** Python 3.12+, NumPy, SciPy, pytest, uv, ruff, GitHub Actions.

**Status:** Approved for implementation. Codex xhigh round 2, on `dd818c2`: "SOUND ENOUGH TO IMPLEMENT", with small notes that this revision folds in (the empty-file case, the scripted double's literals, the scale test's form, three echolocation artists, and the hand-over list). Round 1, on `0312f71`, returned "NOT SOUND ENOUGH TO IMPLEMENT" with seven findings, all folded in. Corrected after merge: clock positions are compared at `rtol=1e-12, atol=1e-12`, not exactly (`a108783`).

**Spec:** `docs/plans/2026-11-19-replay-and-animation-design.md`, sections 1 to 3 and item 1 of section 11. Read them before any task. Where this plan and the spec disagree, the spec wins and the disagreement is a plan bug to report. This plan covers pull request 1 only; nothing here draws a figure that is merged.

## Global Constraints

- Python only through uv: `uv run pytest`, `uv run ruff format .`, `uv run ruff check .`, `uv run python ...`. Never bare `python` or `pip`.
- **Every task runs the same cycle**, and its steps say only what is particular to it: write the tests of Step 1; run them and see them fail for the stated reason; implement; run them and see them pass; run the full local gate (`uv run ruff format --check .`, `uv run ruff check .`, `uv run pytest`, `uv run pytest -m slow`) and read its output; then, in a separate command, commit with the task's message and push. Do not open a pull request (the coordinator does).
- Commit messages are passed with a quoted heredoc (`<<'MSG'`) and end with a `Co-Authored-By:` trailer naming the model that wrote the commit.
- Nothing under `assets/` or `site/assets/` changes except the seven new files `assets/replays/<demo>.npz`. `tests/test_generated_assets.py` stays green.
- Every command keeps its present path and output. In each `src/clocks/_demos/demo_*.py`, `build_parser`, `_run_demo` and `main` are not edited: the demos still write their GIFs, `demo-model-comparison` still runs its comparison twice, and `demo-density` keeps its shared generator. New code goes in new functions.
- The seven demo names, in this order everywhere: `demo_1d`, `demo_2d`, `demo_multi_mass`, `demo_multi_mass_2d`, `demo_model_comparison`, `demo_density`, `demo_echolocation_3d`.
- Format name `"clocks-replay"`, version `1`. Arrays are float64 except `cloud` (float32) and the integer arrays (int64). At most 500 particles are displayed: `cloud_size = min(500, n_particles)`.
- The display-sample offset is φ = (√5 − 1) / 2, with `searchsorted(..., side="left")` on a cumulative sum divided by its last entry.
- Row 0 of every per-observation array is the prior. Row-0 sentinels are written as exact values: `log_evidence[0] = 0.0`, `ess[0] = float(n_particles)`, and for model comparison `model_log_evidence[0] = 0.0`, `model_ess[0] = float(n_particles)`, `model_posterior[0] = model_prior`.
- Two tolerance regimes. The **gate** compares computed inputs (`clock_positions`, `true_rates`, `observations`, `filter_observations`) with `rtol=1e-12, atol=1e-12` and other inputs and `meta` exactly. The **report's** full comparison uses `rtol=1e-9, atol=1e-12` for every float64 array, `rtol=1e-5, atol=1e-6` for `cloud`, and exact equality for integers and `meta`.
- `created_with` is provenance. No test or comparison reads its values.
- The CI test job (`.github/workflows/ci.yml`) is not edited.

## Review Focus

Conditions the spec implies but does not spell out a test for, most likely to bite first. Each has a test in the task that owns the code.

1. **`np.savez_compressed` appends `.npz` to a path without that suffix** (measured), so `save_replay(Path("x"))` would write `x.npz` and report success. `save_replay` must raise `ValueError` for any other suffix. Task 1.
2. **The no-name command line.** On Python 3.12 `nargs="*"` with `choices` rejects a tuple default as one invalid choice; 3.14 accepts it (measured on 3.12.12 and 3.14.0). Task 9 and the report workflow both run the script with no names. The script applies the default after parsing. Task 8.
3. **A float32 dot outside a float64 bound.** `np.float32(0.15)` is 0.15000000596, above 0.15, so a particle just under the mass bound fails a comparison against `np.float64(0.15)` once stored (measured). The check uses float32 endpoint arrays, and is tested one float32 step either side. Task 9.
4. **A candidate whose posterior underflows to exactly 0.** At the end of the packaged run P(K=1) is 0.0. The evidence check works in log space and must give 0 there with no warning and no NaN. Task 9.
5. **The report must be written when its surroundings are broken**: no git, no committed archive for a demo, a platform different from the archive's. Task 8.

---

### Task 1: The archive format

**Files:**
- Create: `src/clocks/_replay.py`
- Test: `tests/test_replay.py`

**Interfaces:**
- Produces:
  - `FORMAT = "clocks-replay"`, `VERSION = 1`, `CLOUD_SIZE = 500`, `DEMOS: tuple[str, ...]` (the seven names in order), and `SPATIAL_DIMENSIONS: dict[str, int]` = 1 for `demo_1d`, `demo_multi_mass`, `demo_model_comparison`, `demo_density`; 2 for `demo_2d`, `demo_multi_mass_2d`; 3 for `demo_echolocation_3d`.
  - `Replay`: `@dataclass(frozen=True, eq=False)` with `meta: Mapping[str, object]` and `arrays: Mapping[str, NDArray]`. (`eq=False` because comparing dicts of arrays raises.)
  - `validate_replay(replay: Replay, source: str = "<memory>") -> None`, raising `ValueError`.
  - `save_replay(path: Path, replay: Replay) -> None` and `load_replay(path: Path) -> Replay`.
  - `provenance() -> dict[str, str]` with keys `python` (`platform.python_version()`), `numpy`, `scipy`, `platform` (`platform.platform()`).

- [ ] **Step 1: Write the failing tests** in `tests/test_replay.py`. Three literal fixtures, each a function returning `(meta, arrays)`; every key and dtype of the format appears in one of them.

```python
F8, I8 = np.float64, np.int64
CONTROLS = {"ess_target": 0.8, "rejuvenation_steps": 2, "proposal_scale": 2.38,
            "resampling": "systematic"}
CREATED = {"python": "3.14.0", "numpy": "2.4.2", "scipy": "1.17.1", "platform": "test"}
BASE_META = {"format": "clocks-replay", "version": 1, "seed": 7, "n_particles": 4,
             "noise_std": 0.005, "controls": CONTROLS, "created_with": CREATED}


def filter_fixture():  # T = 2, C = 2 clocks on a line (D = 1), P = 2, N = 4, m = 3
    meta = {**BASE_META, "demo": "demo_1d", "kind": "filter", "n_observations": 2,
            "track_offset": 1.0, "param_names": ["x", "M"],
            "prior_ranges": [[-8.0, 8.0], [0.005, 0.15]], "cloud_size": 3}
    arrays = {
        "clock_positions": np.array([[-1.0], [1.0]], F8),
        "true_params": np.array([0.5, 0.05], F8),
        "true_rates": np.array([0.97, 0.98], F8),
        "observations": np.array([[0.971, 0.979], [0.969, 0.981]], F8),
        "mean": np.array([[0.0, 0.08], [0.4, 0.06], [0.5, 0.05]], F8),
        "cov": np.array([[[21.0, 0.0], [0.0, 0.002]],
                         [[0.3, 0.001], [0.001, 0.0002]],
                         [[0.1, 0.0005], [0.0005, 0.0001]]], F8),
        "ess": np.array([4.0, 3.5, 3.8], F8),
        "log_evidence": np.array([0.0, 5.0, 11.0], F8),
        "tempering_stages": np.array([3, 1], I8),
        "mh_proposals": np.array([24, 8], I8),
        "mh_acceptances": np.array([6, 3], I8),
        "cloud": np.array([[[-4.0, 0.02], [0.0, 0.08], [4.0, 0.14]],
                           [[0.2, 0.05], [0.4, 0.06], [0.6, 0.07]],
                           [[0.45, 0.049], [0.5, 0.05], [0.55, 0.051]]], np.float32),
        "predicted_mean": np.array([[0.96, 0.97], [0.972, 0.979], [0.97, 0.98]], F8),
        "predicted_std": np.array([[0.02, 0.02], [0.004, 0.003], [0.002, 0.002]], F8),
    }
    return meta, arrays


def echo_fixture():  # T = 1, C = 2 clocks in space (D = 3), P = 4, N = 4, m = 3
    meta = {**BASE_META, "demo": "demo_echolocation_3d", "kind": "filter",
            "n_observations": 1, "track_offset": 0.0,
            "param_names": ["x", "y", "z", "M"],
            "prior_ranges": [[-16.0, 16.0]] * 3 + [[0.005, 0.15]], "cloud_size": 3}
    arrays = {
        "clock_positions": np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], F8),
        "true_params": np.array([1.0, 1.5, 3.0, 0.08], F8),
        "true_rates": np.array([0.98, 0.97], F8),
        "observations": np.array([[0.004, -0.004]], F8),
        "filter_observations": np.array([[0.005656854249492381]], F8),
        "mean": np.array([[0.0, 0.0, 0.0, 0.08], [1.1, 1.6, 3.2, 0.09]], F8),
        "cov": np.stack([np.diag([85.0, 85.0, 85.0, 0.0018]),
                         np.diag([0.03, 0.04, 0.07, 0.0004])]).astype(F8),
        "ess": np.array([4.0, 3.2], F8),
        "log_evidence": np.array([0.0, 7.5], F8),
        "tempering_stages": np.array([2], I8),
        "mh_proposals": np.array([8], I8),
        "mh_acceptances": np.array([3], I8),
        "cloud": np.array([[[-9.0, 2.0, 5.0, 0.03], [0.5, -7.0, 1.0, 0.09],
                            [8.0, 4.0, -3.0, 0.14]],
                           [[1.0, 1.5, 3.1, 0.085], [1.1, 1.6, 3.2, 0.09],
                            [1.2, 1.7, 3.3, 0.095]]], np.float32),
        "predicted_mean": np.array([[0.0, 0.0], [0.0041, -0.0041]], F8),
        "predicted_std": np.array([[0.01, 0.01], [0.0005, 0.0005]], F8),
        "view_bounds": np.array([-1.3, 4.8], F8),
        "weight_in_view": np.array([0.25, 1.0], F8),
    }
    return meta, arrays


def model_fixture():  # T = 1, C = 2, candidates K = 1 and 2, N = 16
    meta = {**BASE_META, "demo": "demo_model_comparison", "kind": "model_comparison",
            "n_observations": 1, "n_particles": 16, "track_offset": 1.0,
            "param_names": ["x₁", "x₂", "M₁", "M₂"],
            "prior_ranges": [[-8.0, 8.0], [0.005, 0.15]], "true_k": 2}
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
```

`FIXTURES = {"demo_1d": filter_fixture, "demo_echolocation_3d": echo_fixture, "demo_model_comparison": model_fixture}`; "each fixture" means parametrized over its three keys, with the key as the test ID. "Written plainly" means `np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)`.

- `test_literal_archive_loads` (each fixture): written plainly, `load_replay(path)` returns `meta` equal to the literal dict and, for every key, an array `np.array_equal` to the literal with the same `dtype`.
- `test_saved_archive_reads_back_as_literals` (each fixture): after `save_replay(path, Replay(meta, arrays))`, `np.load(path, allow_pickle=False)` has `set(z.files) == {"meta", *arrays}`, `json.loads(z["meta"].item()) == meta`, and every array equal with the same dtype. (`z["meta"]` is a 0-d `<U` array and `.item()` gives the `str`: measured.)

Rejected arrays: each of the next twelve raises `ValueError`, and where a test names a key the message contains it.

- `test_missing_key_is_rejected` (each fixture × each array key): also contains `missing`.
- `test_extra_key_is_rejected` (each fixture): `"surplus": np.zeros(1)` added; names `surplus`.
- `test_wrong_dtype_is_rejected` (each fixture × each array key): float64 cast to float32, float32 to float64, int64 to int32; also contains the expected dtype name.
- `test_wrong_rank_is_rejected` (each fixture × each array key): `array[None]`.
- `test_wrong_length_is_rejected` (each fixture × each array key except `clock_positions` and `model_ks`, which define C, D and K × each axis): the array with its last entry along that axis removed.
- `test_wrong_coordinate_width_is_rejected` (each fixture): `clock_positions` with a column of zeros appended; names `clock_positions`. (Nothing else in a filter archive depends on D, so without `SPATIAL_DIMENSIONS` a `demo_2d` archive could lose a coordinate unnoticed.)
- `test_clock_count_must_agree` (each fixture): a third clock row appended raises.
- `test_candidate_count_must_agree`: the model fixture with `model_ks = [1, 2, 3]` and nothing else changed raises.
- `test_non_finite_value_is_rejected` (each fixture × each floating key): one entry set to `np.nan`.
- `test_descending_model_ks_is_rejected`: `model_ks = [2, 1]`; names `model_ks`.
- `test_cloud_size_must_match_meta`: `"cloud_size": 4` in the filter fixture; names `cloud`.
- `test_cloud_size_is_capped_at_500`: the filter fixture with `n_particles = 600`, `cloud_size = 501` and `cloud = np.zeros((3, 501, 2), np.float32)`; names `meta.cloud_size`. The fixtures themselves, with three dots from four particles and two clocks, stay valid: neither the clock count nor `cloud_size == min(500, n_particles)` is checked by the loader.

- `test_meta_keys_are_exact` (each fixture × `missing`, `extra`): removing `"seed"`, or adding `"surplus": 1`, raises naming `meta` and the key.
- `test_wrong_format_or_version_is_rejected`: `"format": "other"` and `"version": 2` each raise.
- `test_unknown_demo_is_rejected`: `"demo": "demo_9d"` raises naming the value.
- `test_meta_must_decode_to_an_object` (IDs `null`, `list`, `number`): the filter fixture written plainly with `meta=np.array("null")`, `np.array("[1]")`, `np.array("3")` raises `ValueError` naming `meta`. (`json.loads("null")` succeeds and returns `None`: measured.)
- `test_meta_must_be_one_string` (IDs `integers`, `two_strings`): `meta=np.array([1, 2])` and `meta=np.array(["a", "b"])` raise naming `meta`.
- `test_an_npy_payload_is_a_value_error`: a file written with `np.save` raises `ValueError` containing its path. (`np.load` returns a bare ndarray there, which a `with` statement turns into `TypeError`: measured.)
- `test_an_unreadable_file_is_a_value_error` (IDs `not_a_zip`, `empty`): a file holding `b"not a zip"`, and a zero-byte file, each raise `ValueError` containing the path. (`np.load` raises `EOFError` on a zero-byte file: measured in review.)
- `test_an_unreadable_member_is_a_value_error`: the filter fixture written plainly with `mean` replaced by `np.array([{"a": 1}], dtype=object)` raises `ValueError` containing the path and `mean`. (Reading that member raises only when it is accessed: measured.)
- `test_save_requires_an_npz_path`: `save_replay(tmp_path / "x", ...)` raises `ValueError` and `list(tmp_path.iterdir()) == []`.
- `test_save_refuses_an_invalid_replay`: a fixture with `mean` removed raises before any file exists.

- [ ] **Step 2: Run** `uv run pytest tests/test_replay.py -q`; it fails with `ModuleNotFoundError: clocks._replay`.

- [ ] **Step 3: Implement.** Decisions the tests do not settle:

- `meta` is serialized with `json.dumps(meta, ensure_ascii=False, sort_keys=True)` into `np.array(text)` under the key `meta`.
- `load_replay` calls `np.load(path, allow_pickle=False)`, requires the result to be an `NpzFile`, requires a `meta` member that is a 0-d array of kind `U` decoding to a `dict`, reads every member into memory and closes the file before returning. Failures to open (`ValueError`, `OSError`, `EOFError`, `zipfile.BadZipFile`) become `ValueError(f"{path}: not a replay archive: {error}")`; a failure reading one member becomes `ValueError(f"{path}: {key}: unreadable: {error}")`.
- Every validation message has the form `f"{source}: {key}: expected {expected}, got {got}"`, where `key` is an array key or `meta.<field>`.
- Required `meta` keys, exactly: `format`, `version`, `demo`, `kind`, `seed`, `n_observations`, `n_particles`, `noise_std`, `track_offset`, `param_names`, `prior_ranges`, `controls`, `created_with`; plus `cloud_size` for kind `filter` and `true_k` for kind `model_comparison`. `kind` is `model_comparison` for `demo_model_comparison` and `filter` for the other six.
- Sizes: `C, D = clock_positions.shape` with `D == SPATIAL_DIMENSIONS[demo]`, checked before anything is derived from it; `T = n_observations`; `P = len(param_names)`; `m = cloud_size`, which must equal `cloud.shape[1]` and be at most `min(500, n_particles)`; `K = len(model_ks)`. `prior_ranges` has `P` pairs for kind `filter` and two (positions, masses) for `model_comparison`, each with `lo < hi`.
- Arrays (float64 unless stated). A demo's key set is exactly the rows that apply to it.

| Key | Shape | Present in |
|---|---|---|
| `clock_positions` | (C, D) | all |
| `true_params` | (P,) | all |
| `true_rates` | (C,) | all |
| `observations` | (T, C) | all |
| `mean` | (T+1, P) | kind `filter` |
| `cov` | (T+1, P, P) | kind `filter` |
| `ess`, `log_evidence` | (T+1,) | kind `filter` |
| `tempering_stages`, `mh_proposals`, `mh_acceptances` | (T,) int64 | kind `filter` |
| `cloud` | (T+1, m, P) float32 | kind `filter` |
| `predicted_mean`, `predicted_std` | (T+1, C) | `demo_1d`, `demo_multi_mass`, `demo_echolocation_3d` |
| `filter_observations` | (T, C−1) | `demo_echolocation_3d` |
| `view_bounds` | (2,) | `demo_echolocation_3d` |
| `weight_in_view` | (T+1,) | `demo_echolocation_3d` |
| `model_ks` | (K,) int64, strictly ascending | `demo_model_comparison` |
| `model_prior` | (K,) | `demo_model_comparison` |
| `model_posterior`, `model_log_evidence`, `model_ess` | (T+1, K) | `demo_model_comparison` |
| `final_mean_k<k>`, `final_std_k<k>` for each k in `model_ks` | (k·(D+1),) | `demo_model_comparison` |

- `save_replay` validates first, then requires `path.suffix == ".npz"`, then writes.

- [ ] **Step 4: Tests pass; full gate; commit and push** as `Add the replay archive format with a validating loader`.

---

### Task 2: The display sample and the filter recorder

**Files:**
- Modify: `src/clocks/_replay.py`
- Test: `tests/test_replay.py`

**Interfaces:**
- Consumes: `Replay`, `CLOUD_SIZE` from Task 1.
- Produces:
  - `thin(particles: NDArray, weights: NDArray, size: int) -> NDArray[np.float32]`, with `min(size, N)` rows.
  - `record_filter(pf, observations: Sequence[Observation], *, meta: Mapping[str, object], arrays: Mapping[str, NDArray], predict: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None) -> Replay`. It returns `meta` unchanged and `arrays` plus `mean`, `cov`, `ess`, `log_evidence`, `tempering_stages`, `mh_proposals`, `mh_acceptances`, `cloud`, `observations` (the stacked readings it fed the filter, unless `arrays` already holds that key), and, when `predict` is given, `predicted_mean` and `predicted_std`. It does not validate; the caller does, after adding anything else.
  - `record_filter` reads only this public surface of `pf`: `state.observations_seen`, `n_particles`, `update(observation)`, `history`, `diagnostics_history`, `log_evidence_history`. A test double with those six works.

- [ ] **Step 1: Write the failing tests**, added to `tests/test_replay.py`.

The moment tests use a double, `ScriptedFilter(states, diagnostics, log_evidence)`, with exactly the surface `record_filter` may read: `n_particles` (the length of the first state's weights); `history`, a list that starts as `[states[0]]`; `state`, a property returning `history[-1]`; `diagnostics_history` and `log_evidence_history`, lists that start empty; and `update(observation)`, which appends the next prepared state, diagnostics entry and log evidence and returns the new state. It has no `rng`.

The scripted history has N = 5 particles, P = 2 and three `ParticleState`s with `observations_seen` 0, 1 and 2: the prior with uniform weights, then weights `[0.1, 0.2, 0.3, 0.25, 0.15]` and `[0.05, 0.4, 0.05, 0.3, 0.2]`. Their particles, in order, are `[[-4.0, 0.02], [-2.0, 0.05], [0.0, 0.08], [2.0, 0.11], [4.0, 0.14]]`, `[[0.1, 0.050], [0.2, 0.055], [0.4, 0.060], [0.5, 0.066], [0.7, 0.070]]` and `[[0.42, 0.0480], [0.47, 0.0495], [0.50, 0.0500], [0.53, 0.0512], [0.58, 0.0520]]`. Diagnostics are `UpdateDiagnostics(3, 24, 6)` and `UpdateDiagnostics(1, 8, 3)`; log evidence `[5.0, 11.0]`; two `Observation`s with two clocks. "The real run" below is two filters from `build_particle_filter` with the `demo_1d` constants at 64 particles, both seeded 42, and three observations from `simulate`; one goes through `record_filter` with `predict=pf.forward_model_batch`, the other is updated in a plain loop.

- `test_thin_picks` (IDs `unequal`, `uniform_7_3`, `uniform_4_4`, `tie`), with `particles = np.arange(N, dtype=float)[:, None]` so a row's value is its index. Each result is float32 with shape `(len(expected), 1)`. All four measured.
  - weights `[0.1, 0.2, 0.0, 0.3, 0.4]`, `size=5` → `[1, 3, 3, 4, 4]` (offset 0.5 gives `[0, 1, 3, 4, 4]`);
  - seven uniform weights, `size=3` → `[1, 3, 6]` (offset 0.5 gives `[1, 3, 5]`);
  - four uniform weights, `size=4` → `[0, 1, 2, 3]`;
  - weights `[φ/2, 0.0, 1 − φ/2]`, `size=2` → `[0, 2]`: the first target equals the first step exactly, and `side="right"` gives `[2, 2]`.
- `test_thin_ignores_the_scale_of_the_weights`: for the first case's weights `w`, `thin(particles, 0.5 * w, 5)` is `np.array_equal` to `thin(particles, w, 5)`. The test compares the two calls and states no expected picks, so a wrong offset does not fail it. (Without the division, indexing raises: measured.)
- `test_thin_is_capped_by_the_particle_count`: four particles, `size=500` → four rows.
- `test_thin_rejects_unusable_weights` (IDs `negative`, `all_zero`): `ValueError` for `[0.5, -0.1, 0.6]` and `[0.0, 0.0, 0.0]`.
- `test_moments_match_an_independent_calculation`: for every row, `mean` and the whole of `cov` equal values the test computes with explicit sums (`m = Σ wᵢxᵢ / Σ wᵢ`; `cov = Σ wᵢ (xᵢ−m)(xᵢ−m)ᵀ / Σ wᵢ`), `rtol=1e-12`; and `abs(cov[1][0, 1]) > 0`.
- `test_row_zero_holds_sentinels`: `ess[0] == 5.0` and `log_evidence[0] == 0.0` exactly; `mean[0]` equals the prior's moments; `cloud[0]` equals `thin(prior particles, prior weights, 500)`. (1 / Σw² for five uniform weights is 4.999999999999999: measured.)
- `test_ess_rows_follow_the_weights`: rows 1 and 2 equal `1 / Σ wᵢ²`, `rtol=1e-12`.
- `test_diagnostics_and_evidence_are_copied`: the three integer arrays are int64 and equal `[3, 1]`, `[24, 8]`, `[6, 3]`; `log_evidence` equals `[0.0, 5.0, 11.0]`.
- `test_predictions_are_moments_of_predictions`: with `predict = lambda p: np.column_stack([p[:, 0] ** 2, np.exp(p[:, 1])])`, `predicted_mean` and `predicted_std` equal the weighted mean and standard deviation of `predict(particles)` per row, `rtol=1e-12`; and `predicted_mean[1]` differs from `predict(mean[1][None])[0]` by more than 1e-6.
- `test_observations_are_what_the_filter_consumed`: `observations` equals the stacked `rates`; when `arrays` already holds `"observations"`, that array is kept.
- `test_recording_does_not_change_the_run`: first assert the plain run did real work (its `mh_acceptances` sum to more than 0, and its generator state differs from a fresh filter's; measured: 213, 92 and 53 acceptances). Then every history row has `np.array_equal` particles and weights in the two runs, and `pf.rng.bit_generator.state` is equal at the end.
- `test_final_row_matches_the_filter_estimate`: in the real run, `mean[-1]`, `sqrt(diag(cov[-1]))` and `ess[-1]` equal `pf.estimate()`'s `mean`, `std` and `ess`, `rtol=1e-12`; and `cloud.shape == (4, 64, 2)` with dtype float32.
- `test_a_used_filter_is_refused`: a real filter after one update raises `RuntimeError` matching `fresh`; afterwards `pf.state.observations_seen == 1` and `len(pf.history) == 2`.

- [ ] **Step 2: Run them; they fail** on the missing names.

- [ ] **Step 3: Implement.**

- `thin`: cumulative sum in float64 divided by its last entry; targets `(np.arange(m) + φ) / m` with `m = min(size, len(weights))`; `np.searchsorted(c, targets, side="left")`; `particles[indices].astype(np.float32)`. Negative weights or a non-positive total raise `ValueError`. No random numbers.
- `record_filter`: raise `RuntimeError(f"record_filter needs a fresh filter; this one has seen {n} observations")` before any update. Build every per-row array from `pf.history` after the loop. `predict` is called once per row on that row's particles, never on a mean: the real forward models raise at the weighted prior mean (measured for `demo_1d` and echolocation). Draw nothing from any generator.

- [ ] **Step 4: Tests pass; full gate; commit and push** as `Record a filter's history: summaries from the whole cloud, a thinned display sample`.

---

### Task 3: Full comparison of two replays

**Files:**
- Modify: `src/clocks/_replay.py`
- Test: `tests/test_replay.py`

**Interfaces:**
- Produces: `Difference` (`NamedTuple` with `key: str`, `row: int | None`, `largest: float`) and `first_difference(reference: Replay, other: Replay) -> Difference | None`.

- [ ] **Step 1: Write the failing tests**, using `filter_fixture()` and `model_fixture()` as the replays.

- `test_equal_replays_have_no_difference`: `None`, also when every `created_with` value differs.
- `test_meta_difference_is_reported_first`: `seed` and `mean` both changed → `Difference("meta", None, ...)`.
- `test_key_set_difference`: a key removed from one side → `key == "keys"`.
- `test_shape_difference` and `test_dtype_difference`: `mean` with a row removed, and `mean.astype(np.float32)` → `key == "mean"`, `row is None`.
- `test_integer_arrays_compare_exactly`: `mh_proposals[1] += 1` → `Difference("mh_proposals", 1, 1.0)`.
- `test_float_tolerance` (IDs `inside`, `outside`): `mean[2, 0]` scaled by `1 + 1e-10` → `None`; by `1 + 1e-8` → `key == "mean"`, `row == 2`.
- `test_inputs_use_the_same_tolerance` (IDs `inside`, `outside`): `observations[1, 0]` scaled by `1 + 1e-10` → `None`; by `1 + 1e-8` → `key == "observations"`, `row == 1`. (The spec's report has one float64 tolerance; the tighter one for computed inputs belongs to the gate.)
- `test_cloud_tolerance` (IDs `inside`, `outside`): `cloud[1, 0, 0]` scaled by `1 + 1e-6` → `None`; by `1 + 1e-4` → `key == "cloud"`, `row == 1`.
- `test_the_earliest_row_is_reported`: `mean` changed in rows 1 and 2 → `row == 1`, and `largest` is the largest absolute difference anywhere in `mean`.
- `test_keys_are_compared_in_a_fixed_order`: `cov` and `mean` both changed → `key == "cov"`; `observations` and `cov` both changed → `key == "observations"`.

- [ ] **Step 2: Run them; they fail** on the missing names.

- [ ] **Step 3: Implement.** Order: `meta` without `created_with`; the key sets; then arrays, inputs first in the order `clock_positions`, `true_params`, `true_rates`, `observations`, `filter_observations`, `model_ks`, `model_prior`, then every other key alphabetically. A shape or dtype mismatch is a difference with `row=None`. Tolerances are the report's, from Global Constraints. `row` is the smallest index along axis 0 holding an element outside tolerance.

- [ ] **Step 4: Tests pass; full gate; commit and push** as `Compare two replays in full and name the first difference`.

---

### Task 4: `scenario()` and `record()` for the four point-mass demos

**Files:**
- Modify: `src/clocks/_replay.py`, `src/clocks/_demos/demo_1d.py`, `demo_2d.py`, `demo_multi_mass.py`, `demo_multi_mass_2d.py`
- Create: `tests/test_demo_records.py`

**Interfaces:**
- Consumes: `Replay`, `validate_replay`, `record_filter`, `provenance`, `CLOUD_SIZE`.
- Produces, in `_replay.py`:
  - `Scenario`: `@dataclass(frozen=True, eq=False)` with `demo: str`, `clock_array: ClockArray`, `true_params: NDArray[np.float64]`, `true_rates: NDArray[np.float64]`, `param_names: tuple[str, ...]`, `prior_ranges: tuple[tuple[float, float], ...]`, `seed: int`, `noise_std: float`, `n_observations: int`, `n_particles: int`.
  - `scenario_meta(scenario: Scenario, *, kind: str, n_observations: int, n_particles: int, controls: Mapping[str, object], **extra: object) -> dict[str, object]`: the complete `meta`. `param_names` becomes a list and `prior_ranges` a list of two-element lists, both levels, so `meta` equals itself after a JSON round trip. `track_offset` comes from the clock array; kind `filter` gets `cloud_size = min(CLOUD_SIZE, n_particles)`; `created_with = provenance()`; `extra` is merged in.
  - `filter_controls(pf) -> dict[str, object]`: `ess_target`, `rejuvenation_steps`, `proposal_scale`, `resampling` read from the filter.
- Produces, in each of the four demo modules: `scenario() -> Scenario` (packaged defaults, reading the module's constants when called) and `record(*, n_observations: int = N_OBSERVATIONS, n_particles: int = N_PARTICLES) -> Replay`, validated. **`record()` takes its clocks, truth, noise level and seed from `scenario()`** and from nowhere else, so the archive and the scenario test share one source.

- [ ] **Step 1: Write the failing tests** in `tests/test_demo_records.py`, parametrized over the four module names. `SMALL = {"n_observations": 3, "n_particles": 64}`. Arrays are compared with `np.array_equal` (never `==` in an `assert`), and readings as `np.stack([o.rates for o in sim.observations])`.

- `test_scenario_values`: `demo` equals the module name; `seed`, `noise_std`, `n_observations`, `n_particles` equal the module's (for `demo_multi_mass_2d`, `_scenarios`') constants; `param_names` is `("x", "M")`, `("x", "y", "M")`, `("x₁", "x₂", "M₁", "M₂")`, `("x₁", "y₁", "x₂", "y₂", "M₁", "M₂")`; `prior_ranges` is `(-8.0, 8.0)` for each position and `(0.005, 0.15)` for each mass; `true_params` equals an array of the module's truth constants in that order; `true_rates` equals `clock_rates(MassConfig(...), clock_array)` for that truth.
- `test_two_mass_2d_clocks_come_from_the_seed`: `demo_multi_mass_2d.scenario().clock_array.positions` equals `generate_random_clocks(N_CLOCKS, np.random.default_rng(11), exclude=[tuple(p) for p in TRUE_POSITIONS])`.
- `test_record_is_a_valid_replay`: `record(**SMALL)` passes `validate_replay`; `meta["demo"]`; `meta["n_observations"] == 3`, `meta["n_particles"] == 64`, `meta["cloud_size"] == 64`; `meta["controls"] == {"ess_target": 0.8, "rejuvenation_steps": 2, "proposal_scale": 2.38, "resampling": "systematic"}`; `json.loads(json.dumps(meta)) == meta`; `predicted_mean` is present for `demo_1d` and `demo_multi_mass` and absent for the other two.
- `test_record_inputs_match_the_scenario`: `clock_positions`, `true_params` and `true_rates` equal the scenario's; `observations` equals the stacked readings of `simulate(SimulationConfig(..., n_observations=3, seed=SEED))`.
- `test_record_agrees_with_the_public_api`: `infer(those observations, InferenceConfig(..., n_particles=64, n_masses=..., seed=SEED))` gives `posterior_mean` and `posterior_std` equal to `mean[-1]` and `sqrt(diag(cov[-1]))`, `rtol=1e-12`.
- `test_record_defaults_are_the_packaged_run`: `inspect.signature(module.record)` has defaults equal to the scenario's `n_observations` and `n_particles`.
- `test_record_handles_one_observation`: `record(n_observations=1, n_particles=16)` validates and has two rows.

- [ ] **Step 2: Run them; they fail** on the missing names.

- [ ] **Step 3: Implement.** Each `record()` simulates and builds its filter as that module's `_run_demo` does (`simulate(SimulationConfig(..., seed=...))`, `build_particle_filter(InferenceConfig(..., seed=...))`). `demo_1d` and `demo_multi_mass` pass `predict=pf.forward_model_batch`, which returns tick rates at the clocks, shape (N, C) (measured); the 2D demos pass none. Read `git diff main -- src/clocks/_demos` and confirm no changed line lies inside `build_parser`, `_run_demo` or `main`; `uv run pytest tests/test_cli.py -q` confirms the GIFs.

- [ ] **Step 4: Tests pass; full gate; commit and push** as `Give the four point-mass demos a scenario and a recorder`.

---

### Task 5: `record()` for echolocation and density

**Files:**
- Modify: `src/clocks/_demos/demo_echolocation_3d.py`, `src/clocks/_demos/demo_density.py`
- Test: `tests/test_demo_records.py`

**Interfaces:**
- Consumes: `Scenario`, `scenario_meta`, `filter_controls`, `record_filter`, `validate_replay`; `study_generator` from `clocks._rng`.
- Produces: `scenario()` and `record()` in both modules, as in Task 4; in `demo_echolocation_3d`, `close_view(truth_xyz: NDArray, mean_xyz: NDArray, sd_xyz: NDArray) -> tuple[float, float]` and `weight_in_view(states: Sequence[ParticleState], bounds: tuple[float, float]) -> NDArray[np.float64]`.

- [ ] **Step 1: Write the failing tests** (`record(n_observations=3, n_particles=64)` unless stated; arrays compared with `np.array_equal`).

Echolocation:

- `test_echolocation_scenario_values`: `seed == DEMO_SEED`; `param_names == ("x", "y", "z", "M")`; `prior_ranges` is `(-16.0, 16.0)` three times then `(0.005, 0.15)`; `true_params` equals `np.append(echo_mass_position(DEMO_RANGE_R), ECHO_M_TRUE)`; `noise_std == ECHO_NOISE_STD`; defaults `ECHO_N_OBSERVATIONS` and `ECHO_N_PARTICLES`.
- `test_echolocation_record_is_valid`: validates; has `filter_observations`, `view_bounds`, `weight_in_view`, `predicted_mean`, `predicted_std`; `meta["controls"] == {"ess_target": 0.9, "rejuvenation_steps": 1, "proposal_scale": 1.5, "resampling": "systematic"}`.
- `test_echolocation_inputs_are_the_packaged_readings`: with `_, display, contrasts = make_echo_observations(DEMO_SEED, DEMO_RANGE_R, n_observations=3)`, `observations` equals the stacked `display` rates and `filter_observations` the stacked `contrasts` rates.
- `test_echolocation_channels_are_centred_contrasts`: `np.testing.assert_allclose(observations.sum(axis=1), 0.0, rtol=0, atol=1e-12)` (measured sums reach 4e-15); `filter_observations` equals `observations @ contrast_matrix(27).T` with `rtol=1e-12, atol=1e-12`.
- `test_echolocation_predictions_are_centred`: `predicted_mean.sum(axis=1)` is zero with `rtol=0, atol=1e-12`.
- `test_echolocation_record_agrees_with_an_unrecorded_filter`: a second `build_echolocation_filter(DEMO_SEED, n_particles=64)` updated with the same contrasts has `estimate()["mean"]` equal to `mean[-1]`, `rtol=1e-12`.
- `test_close_view` (IDs `packaged`, `cloud_below_the_head`, `rounding_up`, `rounding_down`), compared with `==`, all four measured:
  - truth `[0.99, 1.48, 2.97]`, mean `[[1.174, 1.810, 3.379]]`, sd `[[0.164, 0.212, 0.264]]` → `(-1.3, 4.8)`;
  - truth `[0.0, 0.0, 0.5]`, mean `[[0.0, 0.0, -2.0]]`, sd `[[0.1, 0.1, 0.5]]` → `(-4.3, 1.3)`;
  - truth `[0.0, 0.0, 1.1]`, mean `[[0.0, 0.0, 1.1]]`, sd zeros → `(-1.3, 1.4)`; a bare `ceil` gives 1.5;
  - truth `[0.0, 0.0, -1.1]`, mean `[[0.0, 0.0, -1.1]]`, sd zeros → `(-1.4, 1.3)`; a bare `floor` gives −1.5.
- `test_weight_in_view`: two states of four particles (columns x, y, z, M) with weights `[0.1, 0.2, 0.3, 0.4]`, bounds `(-1.0, 1.0)`, and **every mass set to 5.0**, outside the bounds. State one has positions `(0, 0, 0)`, `(2, 0, 0)`, `(0.5, 0.5, 0.5)`, `(0, 0, -3)`; state two has `(1.0, 0, 0)`, `(0, 0, 0)`, `(0, 1.5, 0)`, `(0, 0, 0)`. The result is `pytest.approx([0.4, 0.7])`: a particle on a bound is inside, and an implementation that tests the mass column gives zeros.
- `test_echolocation_view_holds_the_head_the_truth_and_the_cloud`: `view_bounds[0] <= -1.0`; `view_bounds[1] >= max(1.0, true_params[:3].max())`; every `weight_in_view` entry is in `[0, 1 + 1e-12]`; `weight_in_view[-1] > 0.9`.

Density (under a second at this size):

- `test_density_scenario_values`: `param_names == ("μ", "σ", "A")`; `prior_ranges == (MU_RANGE, SIGMA_RANGE, AMPLITUDE_RANGE)`; `true_params` equals `np.array([TRUE_MU, TRUE_SIGMA, TRUE_AMPLITUDE])`; `true_rates` equals `clock_rates_density_gaussian(true_params, clock_array)`, and calling `scenario()` draws nothing.
- `test_density_record_is_valid`: validates; has no `predicted_mean`; controls are Task 4's defaults.
- `test_density_readings_come_from_their_own_generator`: `observations` equals three successive `add_clock_noise(true_rates, NOISE_STD, rng)` draws from one `study_generator(SEED)`, stacked.
- `test_density_readings_do_not_depend_on_particles`: `record(n_observations=3, n_particles=32)` and `record(n_observations=3, n_particles=64)` have equal `observations`.
- `test_density_record_agrees_with_an_unrecorded_filter`: `build_density_filter(clock_array, study_generator(SEED), n_particles=64)` updated with those readings has `estimate()["mean"]` equal to `mean[-1]`, `rtol=1e-12`.

- [ ] **Step 2: Run them; they fail** on the missing names.

- [ ] **Step 3: Implement.**

- Echolocation `record()`: readings from `make_echo_observations(DEMO_SEED, DEMO_RANGE_R, n_observations=...)`; filter from `build_echolocation_filter(DEMO_SEED, n_particles=...)`; the filter consumes the contrasts. Pass `record_filter` the arrays `clock_positions`, `true_params`, `true_rates` (the simulation's, not centred), `observations` (the centred display readings) and `filter_observations`. `predict` returns each particle's tick rates at the 27 clocks with that particle's own mean over clocks removed, as `_animate.predicted_centered` does today. Then compute `view_bounds` and `weight_in_view` from the recorded `mean`, `cov` and `pf.history`, add them, and validate.
- `close_view`: `lo_raw = min(-1.0, truth.min(), (mean - 4*sd).min()) - 0.3` and `hi_raw = max(1.0, truth.max(), (mean + 4*sd).max()) + 0.3`, over the rows for k ≥ 1. Then `lo = floor(round(lo_raw * 10, 6)) / 10` and `hi = ceil(round(hi_raw * 10, 6)) / 10`: **the six-decimal rounding is applied to the product, after multiplying by 10**, not to the coordinate.
- `weight_in_view`: per state, the sum of the weights of particles whose first three columns all lie in `[lo, hi]`, bounds included.
- Density `record()`: all readings drawn first from `study_generator(SEED)` with `add_clock_noise`; the filter from `build_density_filter(clock_array, study_generator(SEED), n_particles=...)`; no `predict`. `_run_demo` keeps its one shared `np.random.default_rng(SEED)` and is not edited.

- [ ] **Step 4: Tests pass; full gate; commit and push** as `Record the echolocation and density runs; give density's readings their own generator`.

---

### Task 6: `record()` for model comparison

**Files:**
- Modify: `src/clocks/_demos/demo_model_comparison.py`
- Test: `tests/test_demo_records.py`

**Interfaces:**
- Consumes: `Scenario`, `scenario_meta`, `filter_controls`, `Replay`, `validate_replay`.
- Produces: `scenario()` and `record()` as in Task 4, and the constant `MODEL_KS = (1, 2, 3)`.

- [ ] **Step 1: Write the failing tests** (`record(n_observations=3, n_particles=64)` unless stated). `result = infer(observations, InferenceConfig(..., n_particles=64, n_masses=(1, 2, 3), seed=SEED))`.

- `test_model_scenario_values`: `param_names == ("x₁", "x₂", "M₁", "M₂")`; `prior_ranges == ((-8.0, 8.0), (0.005, 0.15))`; `true_params` equals `np.array([TRUE_X1, TRUE_X2, TRUE_M1, TRUE_M2])`.
- `test_model_record_is_valid`: validates; `meta["kind"] == "model_comparison"`; `meta["true_k"] == 2`; `model_ks` equals `np.array([1, 2, 3])` with dtype int64; `model_prior` equals `np.full(3, 1.0 / 3)`.
- `test_model_record_row_zero_is_the_prior`: with `n_particles=80`, `model_posterior[0]` equals `model_prior`; `model_log_evidence[0]` is all `0.0`; `model_ess[0]` is all `80.0` exactly. (1 / Σw² for 80 uniform weights is 79.99999999999997: measured.)
- `test_model_record_agrees_with_the_public_api`: `result.history` is a tuple of mappings keyed by candidate count, so project it first: `expected = np.array([[row[k] for k in (1, 2, 3)] for row in result.history])`. Then `model_posterior[1:]` equals `expected`, `rtol=1e-12` (measured exactly equal for this run), and its last row equals `[result.posterior_by_model[k] for k in (1, 2, 3)]`.
- `test_model_record_final_estimates_match_the_public_api`: for each k, `final_mean_k<k>` and `final_std_k<k>` equal `result.result_by_model[k].posterior_mean` and `.posterior_std`, `rtol=1e-12`; `final_mean_k3` has six entries whose first three are strictly increasing (positions, then masses: measured).
- `test_model_posterior_follows_the_evidence`: every row of `model_posterior` equals `exp(l − logsumexp(l))` with `l = log(model_prior) + model_log_evidence[row]`, `rtol=1e-9, atol=1e-12`.
- `test_model_record_inputs_match_the_scenario`: as Task 4's input test.

- [ ] **Step 2: Run them; they fail** on the missing names.

- [ ] **Step 3: Implement.** `record()` builds `build_model_comparison(InferenceConfig(..., n_masses=MODEL_KS, seed=...))` once from `scenario()` and updates it observation by observation, reading `comparison.evidence()` after each for the posterior and log evidence and each filter's `estimate()["ess"]` for `model_ess`. `model_prior` comes from `comparison.model_prior` in `comparison.k_values` order; the final estimates from each filter's `estimate()`; controls from `filter_controls(comparison.filters[1])`. No `cloud`, no `cloud_size`. `_run_demo` still calls `infer` and then animates a second comparison; it is not edited.

- [ ] **Step 4: Tests pass; full gate; commit and push** as `Record the model-comparison run in one pass`.

---

### Task 7: Smoke-draw every panel and probe the 0.10 run

No tracked file changes. This task finds out, before any archive is committed, whether version 1 of the format holds what the later drawings need. Its result is a gate: Task 9 does not start until the coordinator has accepted it. Work in the scratch directory the coordinator names; write throwaway code there.

**Interfaces:**
- Consumes: every demo's `record()`; `save_replay`, `load_replay`; `clocks.viz.style` and `THEMES` for the look.

- [ ] **Step 1: Record candidates.** Call each demo's `record()` at its defaults and save the seven archives under the scratch directory (density takes about 35 s; foreground, `timeout: 600000`).

- [ ] **Step 2: Draw every panel of spec sections 5 and 6 that an archive feeds**, at k = 0, 1, 8 and last, light theme, one contact sheet per archive. The drawing script loads each archive with `load_replay` and imports nothing else from `clocks` except `clocks.viz`; in particular nothing from `clocks.physics`, `clocks.inference`, `clocks.api`, `clocks._scenarios` or `clocks._demos`. Layout does not matter; presence does.

| Archive | Panel | Artists, and what each is populated from |
|---|---|---|
| `demo_1d` | cloud in (x, M) | dots `cloud[k]`; truth mark `true_params`; ellipse `mean[k]`, `cov[k]`; axes `prior_ranges`; zoom inset window `true_params`, `mean[-1]`, `cov[-1]` |
| | clock strip | squares `clock_positions` |
| | slowing at each clock | all three marks are slowings, `1 − rate`: true tick `1 − true_rates`; predicted bar `1 − predicted_mean[k]` with `predicted_std[k]`; `1 −` the running mean of `observations[:k]` with bar `meta.noise_std / √k` |
| | traces x, M | line and band `mean`, `cov` diagonal; truth line `true_params`; header numbers from the same |
| `demo_2d` | map | dots; truth; ellipse; inset; clock squares `clock_positions` filled by the running mean of `1 − observations[:k]` |
| | traces x, y, M | as above |
| `demo_multi_mass` | cloud in (x, M), both masses | two dot sets from the columns of `cloud[k]` named in `param_names`; two truth marks; two ellipses from the matching blocks of `cov[k]`; one inset per mass |
| | clock strip, slowing panel | as `demo_1d` |
| | four traces | as above |
| `demo_multi_mass_2d` | map, both masses | two dot sets; two truth marks; **both** ellipses; ten filled clocks |
| | six traces | as above |
| `demo_model_comparison` | posterior lines | one line per `model_ks` entry from `model_posterior[:k+1]`, starting at row 0; a label gutter with a connector from each line's head to its label; "(true model)" on the `meta.true_k` line |
| `demo_echolocation_3d` | close view | cube wireframe from `view_bounds`; clocks `clock_positions`; dots of `cloud[k]` inside `view_bounds`; truth ring `true_params` with its drop-line to the cube floor (`view_bounds[0]`) |
| | whole prior box | all dots of `cloud[k]` on x and z over `prior_ranges`; square `view_bounds` |
| | weight in view | text from `weight_in_view[k]` |
| | traces x, y, z, M | as above |
| | observed against fitted | 27 points: running mean of `observations[:k]` against `predicted_mean[k]`; the equal-values diagonal, with limits from those two arrays; empty of points at k = 0 |
| `demo_density` | profile | true curve from `true_params`; pointwise median and 68% band of A·exp(−(x−μ)²/2σ²) over `cloud[-1]`; clock squares `clock_positions` |
| | traces μ, σ, A | as above |

**Acceptance check, one per row:** every artist listed is drawn, at every k where it exists, from the arrays and `meta` named, with no value typed in by hand and no call into the library beyond `load_replay` and the style. A row passes or fails as a whole.

- [ ] **Step 3: Probe the 2D run at M = 0.10.** In a scratch script set `clocks._demos.demo_2d.TRUE_M = 0.10` and call `record()`. Record the posterior mean and standard deviation after observations 1, 8 and 50. The spec measured the end as (1.4169, −1.2223, 0.1000) ± (0.1111, 0.1148, 0.0012). Save nothing in the repository.

- [ ] **Step 4: Return to the coordinator**, and stop:
  - the seven contact sheets and the drawing script, as paths;
  - the table above with a pass or fail per row and, for each fail, the quantity that was missing or had to come from outside the archive;
  - the probe's six rows of numbers and whether the end agrees with the spec to the digits quoted;
  - the seven archive sizes.

  **What the coordinator looks for:** every row passes; each sheet shows every artist of its rows (dots at all four k, nothing but the prior before the first reading, the echolocation scatter empty at k = 0); the script's imports are as stated; the predicted bars in the slowing panel close on the true ticks by the last frame, which a mis-centred or mis-ordered prediction would not do. If any row fails, the format changes before any archive is committed; that decision is the coordinator's, and the affected tasks are revised first.

---

### Task 8: The recording script

**Files:**
- Create: `scripts/record_replays.py`, `tests/test_record_replays_script.py`

**Interfaces:**
- Consumes: `DEMOS`, `save_replay`, `load_replay`, `first_difference`, `provenance`; each demo's `record()`.
- Produces: `record_demo(name: str) -> Replay` (imports `clocks._demos.<name>` and calls its `record()`; the tests' seam) and `main(argv: Sequence[str] | None = None) -> int`, with `--out DIR` (required), optional demo names, `--compare DIR2` and `--report FILE`. `main` returns 0 when nothing was compared or every compared demo reproduced, and 1 otherwise; the file ends `if __name__ == "__main__": raise SystemExit(main())`. The script defines no dataclass at module level: the tests import it by path without registering it, and a dataclass then fails at import (spec, M14).

- [ ] **Step 1: Write the failing tests**, importing the script as `tests/test_site_prose_contract.py` imports `check_site_links`. **Every argument is a string: `str(path)`.** A `Path` in the list makes `argparse` raise `TypeError` before anything runs (measured). Tests that name a demo record only `demo_1d` (a quarter of a second).

- `test_records_the_named_demo`: `main(["--out", str(out), "demo_1d"]) == 0`; `out` is created and holds exactly `demo_1d.npz`, which `load_replay` accepts.
- `test_no_names_records_all_seven`: with `record_demo` monkeypatched to a stub that appends its argument to a list and returns one small valid replay (`demo_1d.record(n_observations=1, n_particles=16)`, made once), `main(["--out", str(out)]) == 0`; the stub was called with the seven names in `DEMOS` order; `out` holds the seven `<name>.npz` files.
- `test_an_unknown_name_is_a_usage_error`: `SystemExit` with code 2.
- `test_compare_reports_a_reproduced_demo`: record into `a`; then `main(["--out", str(b), "--compare", str(a), "--report", str(report), "demo_1d"]) == 0` and the report contains the line `demo_1d: reproduced`.
- `test_compare_names_the_first_difference`: record into `a`; load `a/demo_1d.npz`, add `1e-6` to `mean[3, 0]`, save it back; the same command returns 1 and the report contains `demo_1d: differs at mean, row 3`.
- `test_compare_ignores_created_with`: record into `a`; rewrite the archive with every `created_with` value replaced by `"elsewhere"`; the command returns 0 and reports `reproduced`.
- `test_a_missing_or_unreadable_archive_is_reported` (IDs `missing`, `unreadable`): `--compare` pointing at an empty directory, or at one whose `demo_1d.npz` holds `b"not a zip"`, returns 1; the report contains `demo_1d: no committed archive` or `demo_1d: committed archive unreadable`; `b/demo_1d.npz` was still written.
- `test_report_header_records_where_it_ran`: with `GITHUB_RUN_ID=123` and `REPLAY_HEAD_SHA=abc123` set, the report contains `run: 123`, `head: abc123`, a `checkout:` line, and lines for `python:` (`platform.python_version()`), `numpy:`, `scipy:`, `blas:` and `platform:`.
- `test_report_survives_a_missing_git`: with the script's `subprocess.run` monkeypatched to raise `FileNotFoundError`, the report is written and contains `checkout: unknown`.
- `test_report_is_also_printed`: the report text appears on standard output.
- `test_the_command_exits_with_the_comparison_status`: `subprocess.run([sys.executable, "scripts/record_replays.py", "--out", str(b), "--compare", str(empty), "demo_1d"])` has return code 1.

- [ ] **Step 2: Run them; they fail** on the missing script.

- [ ] **Step 3: Implement.**

- The positional is `nargs="*"` with `choices=DEMOS` and **no default**; after parsing, `names = args.demos or DEMOS`. A tuple default is rejected as one invalid choice on Python 3.12 and accepted on 3.14 (measured), and the project supports both.
- Record each name with `record_demo` and save `<out>/<name>.npz`. For `--compare`, load `<DIR2>/<name>.npz` and call `first_difference(committed, recorded)`; one line per demo: `reproduced`, `differs at <key>, row <row> (largest difference <largest:.3g>)` (no row when it is `None`), `no committed archive`, or `committed archive unreadable: <error>`.
- Header lines: `run:` (`GITHUB_RUN_ID`, else `local`), `head:` (`REPLAY_HEAD_SHA`, else `unknown`), `checkout:` (`git rev-parse HEAD`, else `unknown`), then `provenance()` and `blas:` from `np.show_config(mode="dicts")["Build Dependencies"]["blas"]` (on this Mac, `accelerate unknown`: measured). Without `--report` the text goes to standard output only.

- [ ] **Step 4: Tests pass; full gate; commit and push** as `Add the script that records replays and reports how they compare`.

---

### Task 9: The seven archives, their invariants and the gate

**Files:**
- Create: `assets/replays/demo_1d.npz`, `demo_2d.npz`, `demo_multi_mass.npz`, `demo_multi_mass_2d.npz`, `demo_model_comparison.npz`, `demo_density.npz`, `demo_echolocation_3d.npz`; `tests/test_replays.py`

**Interfaces:**
- Consumes: `scripts/record_replays.py`; `load_replay`, `DEMOS`; each demo's `scenario()` and `record()`.

- [ ] **Step 0: Confirm the gate.** The dispatch for this task must say that the coordinator accepted Task 7's contact sheets and table. If it does not, stop and ask; do not record or commit an archive.

- [ ] **Step 1: Write the failing tests** in `tests/test_replays.py`. `ARCHIVES = ROOT / "assets/replays"`. "Each demo" is parametrized over `DEMOS` with the name as ID. Tests that call `record()` take the fresh replay from one cached helper (`functools.lru_cache`), so each demo is recorded once per session, and mark `demo_multi_mass_2d`, `demo_density` and `demo_echolocation_3d` with `pytest.mark.slow` through `pytest.param`. Checks stated as absolute use `np.testing.assert_allclose(..., rtol=0, atol=...)`.

Helpers in the test module, each usable on in-memory arrays so it can be tried on a corrupted copy:

- `_inside_prior(cloud, prior_ranges) -> bool`: with `lo = np.asarray([r[0] for r in prior_ranges], dtype=np.float32)` and `hi` likewise, `bool(np.all((cloud >= lo) & (cloud <= hi)))`, the comparison done in float32.
- `_posterior_from_evidence(model_prior, model_log_evidence) -> NDArray`: `l = np.log(model_prior) + model_log_evidence`; `np.exp(l - logsumexp(l, axis=1, keepdims=True))`.
- `_checkpoints(n_observations) -> tuple[int, ...]`: `sorted({1, min(8, n_observations), n_observations})`.

Invariants on the committed archives (no inference; all seven in the default suite):

- `test_archives_are_exactly_the_seven`: the `.npz` stems in `ARCHIVES` equal `set(DEMOS)`.
- `test_archive_matches_its_scenario` (each demo): `meta` fields `demo`, `seed`, `noise_std`, `n_observations`, `n_particles`, `track_offset`, `param_names`, `prior_ranges` equal the scenario's (tuples converted to lists at both levels); `true_params` is `np.array_equal` to the scenario's; `clock_positions` and `true_rates` match with `rtol=1e-12, atol=1e-12`.
- `test_archive_truth_is_physical` (each demo): `np.max(np.abs(true_rates**2 - 1.0)) <= 0.08`; and x₁ < x₂, which is `true_params[0] < true_params[1]` for `demo_multi_mass` and `demo_model_comparison` and `true_params[0] < true_params[2]` for `demo_multi_mass_2d`.
- `test_archive_sentinels_and_ranges` (the six filter demos): `log_evidence[0] == 0.0`; `ess[0] == n_particles` exactly; `0 < ess` and `ess <= n_particles * (1 + 1e-12)` in every row; for every row, `cov` equals its transpose with `rtol=0, atol=1e-12 * abs(cov).max()` and `np.linalg.eigvalsh(cov).min() >= -1e-12 * np.trace(cov)`; `mh_acceptances <= mh_proposals`; `tempering_stages >= 1`.
- `test_archive_dots_agree_with_summaries` (the six filter demos): `_inside_prior(cloud, prior_ranges)`; in every row and column, `abs(cloud.mean(axis=1) - mean) <= 6 * sd / sqrt(m)` and `0.6 <= cloud.std(axis=1) / sd <= 1.5`, with `sd = sqrt(diag(cov))` and `cloud` cast to float64 for the moments. (The spec's candidates reach 3.0 and 0.78 to 1.29; if a fresh archive fails this, report it, do not widen it.)
- `test_float32_endpoints` (IDs `just_inside`, `one_step_above`, `one_step_below`), against `prior_ranges = [[0.005, 0.15]]`: `np.float32(np.nextafter(0.15, 0.0))` is inside although it exceeds the float64 0.15; `np.nextafter(np.float32(0.15), np.float32(1))` is outside; `np.nextafter(np.float32(0.005), np.float32(0))` is outside. (All three measured. The first alone would also pass an implementation that compares against Python floats, which NumPy rounds to float32 silently; the other two pin the endpoints.)
- `test_echolocation_archive_channels`: `observations.sum(axis=1)` is zero with `rtol=0, atol=1e-12`; `filter_observations` equals `observations @ contrast_matrix(27).T` with `rtol=1e-12, atol=1e-12`; `view_bounds[0] <= -1.0`; `view_bounds[1] >= max(1.0, true_params[:3].max())`; `0 <= weight_in_view <= 1 + 1e-12`.
- `test_model_archive_rows`: `model_log_evidence[0]` is all 0.0; `model_ess[0]` is all `n_particles`; `model_posterior[0]` is `np.array_equal` to `model_prior`; `model_posterior.sum(axis=1)` is 1 with `rtol=0, atol=1e-12`; `model_ks` equals `[1, 2, 3]`; `meta["true_k"] == 2`.
- `test_model_posterior_follows_evidence`: `model_posterior` equals `_posterior_from_evidence(...)` with `rtol=1e-9, atol=1e-12`.
- `test_a_posterior_row_that_keeps_its_sum_is_caught`: a copy of the archive's `model_posterior` with its last row set to `[0.0, 0.99, 0.01]` no longer matches `_posterior_from_evidence(...)` at that tolerance, while still summing to 1.
- `test_an_underflowed_candidate_gives_zero`: `_posterior_from_evidence(np.array([0.5, 0.5]), np.array([[-1000.0, 0.0]]))` equals `[[0.0, 1.0]]` under `warnings.simplefilter("error")`.

The gate (each demo; the three slow ones marked), against `fresh = record()`:

- `test_gate_inputs`: the key sets are equal; every array has the same shape and dtype; `meta` is equal after dropping `created_with` from both; `true_params` and, for model comparison, `model_ks` and `model_prior` are `np.array_equal`; `clock_positions`, `true_rates`, `observations` and, for echolocation, `filter_observations` match with `rtol=1e-12, atol=1e-12`.
- `test_gate_summaries`: at each row in `_checkpoints(n_observations)`, for each parameter column, `abs(fresh_mean - archive_mean) <= bound` and `abs(fresh_sd - archive_sd) <= bound`, with `sd = sqrt(diag(cov))`. Position columns are those whose name starts with `x`, `y` or `z`; mass columns those starting with `M`. For `demo_model_comparison`: `model_posterior[row, 0] <= 1e-5` at row 1 and `<= 1e-20` at row 8 and the last row, in both the archive and the fresh run; and `abs(fresh − archive) <= 0.45` for `model_posterior[1, 1]`.

```python
# (mean positions, mean masses, sd positions, sd masses) after observation 1, 8, last
BOUNDS = {
    "demo_1d": ((0.095, 0.0025, 0.063, 0.0027), (0.024, 0.00091, 0.015, 0.00079),
                (0.0085, 0.00034, 0.0045, 0.00026)),
    "demo_2d": ((0.48, 0.0054, 0.28, 0.0020), (0.11, 0.00059, 0.062, 0.00052),
                (0.025, 0.00025, 0.018, 0.00014)),
    "demo_multi_mass": ((0.87, 0.0095, 0.19, 0.0065), (0.060, 0.0013, 0.063, 0.00079),
                        (0.0079, 0.00015, 0.0058, 0.00012)),
    "demo_multi_mass_2d": ((1.6, 0.011, 0.85, 0.0071), (4.2, 0.026, 2.9, 0.0096),
                           (0.46, 0.0027, 0.47, 0.0014)),
    "demo_echolocation_3d": ((0.28, 0.021, 0.043, 0.0029), (0.012, 0.00077, 0.0097, 0.00059),
                             (0.0073, 0.00047, 0.0044, 0.00027)),
}
# density: (mean μ, σ, A), (sd μ, σ, A) after observation 1, 8, last
DENSITY_BOUNDS = (((0.49, 0.38, 0.00050), (0.38, 0.15, 0.00069)),
                  ((0.067, 0.22, 0.0016), (0.039, 0.20, 0.0031)),
                  ((0.019, 0.027, 0.00011), (0.014, 0.021, 0.000083)))
```

- `test_bounds_cover_every_filter_demo`: `set(BOUNDS) | {"demo_density"}` equals the six filter demos.

- [ ] **Step 2: Run them; they fail**, the archive tests on missing files.

- [ ] **Step 3: Record the archives**, in the foreground with `timeout: 600000`: `uv run python scripts/record_replays.py --out assets/replays`. Report the seven sizes; the spec's candidates totalled 1.8 MB, and a total above 3 MB is a finding to report before committing. Then `uv run python scripts/record_replays.py --out <scratch dir> --compare assets/replays` must print seven `reproduced` lines, and `uv run pytest tests/test_replays.py -q -m ''` must pass. If a gate test fails against an archive recorded minutes earlier on the same machine, the recorder is not repeatable: stop and report, do not widen a bound. If an invariant fails, report the demo, row and value.

- [ ] **Step 4: Full gate; confirm `git status --short` lists only the seven archives and the test file; commit and push** as `Commit the seven replay archives with their invariants and gate`. The commit body lists each archive's size and the final estimate each recording gives.

---

### Task 10: The report workflow

**Files:**
- Create: `.github/workflows/replay-report.yml`, `tests/test_replay_report_workflow.py`

**Interfaces:**
- Consumes: `scripts/record_replays.py --out ... --compare ... --report ...`, and its use of `REPLAY_HEAD_SHA`.

- [ ] **Step 1: Write the failing tests**, reading the workflow as text (no YAML library is a declared dependency).

- `test_every_step_may_fail`: split the text after `steps:` into items (lines beginning with six spaces and `- `); every item contains `continue-on-error: true`.
- `test_the_report_job_is_a_three_way_matrix`: the text contains `sample: [1, 2, 3]` and `fail-fast: false`.
- `test_triggers_are_filtered_to_what_can_change_a_replay`: each of `src/clocks/**`, `assets/replays/**`, `scripts/record_replays.py`, `uv.lock`, `pyproject.toml` and the workflow's own path appears exactly twice as a quoted list entry (a line that is `- "<path>"` after stripping), once per trigger. Count the list entries, not the bare text: the script's path also appears in the record step's command.
- `test_drafts_are_skipped_and_jobs_time_out`: the text contains the draft condition and `timeout-minutes: 10`.
- `test_the_recording_uses_the_locked_environment`: the record step runs `uv run --locked`.
- `test_artifacts_are_named_per_attempt_and_sample`: the artifact name contains `github.run_id`, `github.run_attempt` and `matrix.sample`.
- `test_the_report_never_touches_the_test_workflow`: `.github/workflows/ci.yml` does not mention `record_replays`.
- `test_the_last_step_shows_or_warns` (IDs `shown`, `empty_report`, `no_report`, `upload_failed`): take the lines after the file's last `run: |`, `textwrap.dedent` them, and run them with `bash -c` in a `tmp_path` with `GITHUB_STEP_SUMMARY` pointing at a file there, `SAMPLE=2` and `UPLOAD_OUTCOME` set.
  - a non-empty `replay-report.md` and `success`: the summary file holds the report and standard output has no `::warning`;
  - an empty `replay-report.md`, or none, and `success`: standard output contains `no report was written for sample 2` and the summary file is empty or absent;
  - a non-empty report and `failure`: standard output contains `was not uploaded (failure)`.

- [ ] **Step 2: Run them; they fail** on the missing file.

- [ ] **Step 3: Write the workflow.** The paths are written out under both triggers; do not use a YAML anchor.

```yaml
name: Replay report

# Reports whether the runner reproduces each committed replay value for value.
# Nothing here can fail a check; the tests that gate a merge are in ci.yml.
on:
  push:
    branches: [main]
    paths:
      - "src/clocks/**"
      - "assets/replays/**"
      - "scripts/record_replays.py"
      - "uv.lock"
      - "pyproject.toml"
      - ".github/workflows/replay-report.yml"
  pull_request:
    branches: [main]
    types: [opened, synchronize, reopened, ready_for_review]
    paths:
      - "src/clocks/**"
      - "assets/replays/**"
      - "scripts/record_replays.py"
      - "uv.lock"
      - "pyproject.toml"
      - ".github/workflows/replay-report.yml"

concurrency:
  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.run_id }}-${{ github.event.pull_request.draft && 'draft' || 'ready' }}
  cancel-in-progress: true

jobs:
  report:
    if: github.event_name == 'push' || !github.event.pull_request.draft
    runs-on: ubuntu-latest
    timeout-minutes: 10
    permissions:
      contents: read
    strategy:
      fail-fast: false
      matrix:
        sample: [1, 2, 3]
    env:
      REPLAY_HEAD_SHA: ${{ github.event.pull_request.head.sha || github.sha }}
    steps:
      - uses: actions/checkout@v6
        continue-on-error: true
      - uses: astral-sh/setup-uv@v8.2.0
        continue-on-error: true
      - name: Install dependencies
        continue-on-error: true
        run: uv sync --locked
      - name: Record and compare
        continue-on-error: true
        run: uv run --locked python scripts/record_replays.py --out recorded --compare assets/replays --report replay-report.md
      - name: Upload the report
        id: upload
        continue-on-error: true
        uses: actions/upload-artifact@v4
        with:
          name: replay-report-${{ github.run_id }}-${{ github.run_attempt }}-${{ matrix.sample }}
          path: replay-report.md
          if-no-files-found: error
      - name: Show the report, or say what is missing
        continue-on-error: true
        if: always()
        env:
          SAMPLE: ${{ matrix.sample }}
          UPLOAD_OUTCOME: ${{ steps.upload.outcome }}
        run: |
          if [ -s replay-report.md ]; then
            cat replay-report.md >> "$GITHUB_STEP_SUMMARY"
          else
            echo "::warning title=Replay report::no report was written for sample $SAMPLE"
          fi
          if [ "$UPLOAD_OUTCOME" != "success" ]; then
            echo "::warning title=Replay report::the report for sample $SAMPLE was not uploaded ($UPLOAD_OUTCOME)"
          fi
```

`uv run --locked` refuses a lock that does not match, so a tolerated install failure cannot lead to a report from a freshly resolved environment. `steps.upload.outcome` is the raw result, before `continue-on-error` is applied.

- [ ] **Step 4: Tests pass; full gate; commit and push** as `Report on every push whether the runner reproduces the replays, without gating`.

- [ ] **Step 5: Hand over to the coordinator.** The coordinator opens the pull request, marks it ready, and for one attempt of the run on the reviewed head verifies three artifacts, each a non-empty report whose `head:` line is that head, and no warning annotation. The three reports, the macOS comparison (`uv run python scripts/record_replays.py --out <scratch> --compare assets/replays`), Task 7's contact sheets with its pass-or-fail table, and the 0.10 probe's numbers go into the pull request's description, as the spec's sequencing section requires.

---

### Task 11: Mutation checks

No files change. This task proves the new tests can fail. Work in the task's own worktree; set `PYTHONDONTWRITEBYTECODE=1`; delete every `__pycache__` before each mutant; confirm `git status --porcelain` is empty before and after each; apply each mutant and restore it in a `try`/`finally` (an archive mutant with `git checkout -- assets/replays`); run the whole suite including the slow tests (`uv run pytest -q -m ''`) for each mutant, in the foreground with a 600000 ms timeout. Archive mutants rewrite the file with `load_replay`, an edit to one array, and `save_replay`.

| Mutation | Tests that must fail, and no others |
|---|---|
| `thin` uses offset 0.5 in place of φ | `test_thin_picks[unequal]`, `test_thin_picks[uniform_7_3]` |
| `thin` uses `side="right"` | `test_thin_picks[tie]` |
| `thin` does not divide the cumulative sum by its last entry | `test_thin_ignores_the_scale_of_the_weights` |
| `record_filter` zeroes the off-diagonal entries of `cov` | `test_moments_match_an_independent_calculation`, `test_row_zero_holds_sentinels` (it checks the prior row's `cov`, and the scripted prior is correlated) |
| `record_filter` computes `ess[0]` as 1 / Σw² | `test_row_zero_holds_sentinels` |
| `record_filter` computes `predicted_mean` as `predict` at the mean particle, **only when the filter is the scripted double** (`not hasattr(pf, "rng")`) | `test_predictions_are_moments_of_predictions` |
| `record_filter` draws one number from the filter's generator after the last update, when the filter has one (`getattr(pf, "rng", None)`) | `test_recording_does_not_change_the_run` |
| `record_filter` drops its freshness check | `test_a_used_filter_is_refused` |
| `validate_replay` skips the dtype check | every `test_wrong_dtype_is_rejected` case |
| `validate_replay` does not check D against `SPATIAL_DIMENSIONS` | every `test_wrong_coordinate_width_is_rejected` case |
| `validate_replay` accepts descending `model_ks` | `test_descending_model_ks_is_rejected` |
| `load_replay` opens the file in a `with` statement without checking for an `NpzFile` | `test_an_npy_payload_is_a_value_error` |
| `first_difference` skips `cloud` | `test_cloud_tolerance[outside]` |
| `close_view` uses bare `floor` and `ceil` | `test_close_view[rounding_up]`, `test_close_view[rounding_down]` |
| `weight_in_view` tests all four columns against the bounds | `test_weight_in_view` |
| `demo_model_comparison.record()` writes each candidate's mean under `final_std_k<k>` as well | `test_model_record_final_estimates_match_the_public_api` |
| **An early trajectory change.** In `_replay.record_point_masses` (the helper the four point-mass `record()` functions call), before the call to `record_filter` and only `if scenario.demo == "demo_1d"`, with `NOISE_STD = scenario.noise_std`: `real = pf.update`, then `pf.update = lambda o: (setattr(pf, "noise_std", 2 * NOISE_STD if pf.state.observations_seen < 3 else NOISE_STD), real(o))[1]` | `test_gate_summaries[demo_1d]`, `test_record_agrees_with_the_public_api[demo_1d]` |
| **A stale final result.** In `assets/replays/demo_1d.npz`, add 0.05 to `mean[-1, 0]` | `test_gate_summaries[demo_1d]`, `test_archive_dots_agree_with_summaries[demo_1d]` |
| **A stale early row.** In `assets/replays/demo_2d.npz`, multiply `cov[1]` by 9 | `test_gate_summaries[demo_2d]`, `test_archive_dots_agree_with_summaries[demo_2d]` |
| In `assets/replays/demo_model_comparison.npz`, set the last `model_posterior` row to `[0.0, 0.99, 0.01]` | `test_model_posterior_follows_evidence` |
| In `assets/replays/demo_echolocation_3d.npz`, add `1e-6` to every entry of `observations` | `test_echolocation_archive_channels`, `test_gate_inputs[demo_echolocation_3d]` |
| `demo_1d.NOISE_STD = 0.010` | `test_archive_matches_its_scenario[demo_1d]`, `test_gate_inputs[demo_1d]`, `test_gate_summaries[demo_1d]` |
| `demo_density.record()` draws each reading from the filter's generator between updates | `test_density_readings_come_from_their_own_generator`, `test_density_readings_do_not_depend_on_particles`, `test_density_record_agrees_with_an_unrecorded_filter`, `test_gate_inputs[demo_density]`, `test_gate_summaries[demo_density]` |

The prediction mutant is scoped to the scripted double because the real forward models raise a weak-field error at the weighted prior mean (measured for `demo_1d` at 16, 64 and 1,000 particles and for echolocation at 64 and 6,000); unscoped, it would break every test that records `demo_1d`, `demo_multi_mass` or echolocation and show nothing about the moment test. Several expectations rest on measurement and arithmetic, not on a run of this code: the early change fails the first checkpoint and passes the last (spec, M15); the stale final mean moves 0.05 against a bound of 0.0085 and a dot tolerance of about 0.010; the stale early row triples a standard deviation of about 0.58 against a bound of 0.28; shared density draws moved the final μ by 0.127 against 0.019. The no-default parser has no row: its mutant, a tuple default, fails `test_no_names_records_all_seven` on Python 3.12 and passes on 3.14.

- [ ] **Step 1: Run each mutation** and record, for each, the exact set of failing test IDs.
- [ ] **Step 2: Report** the table of results. Where the failing set differs from the expected set, say which tests were extra or missing; do not change tests or code to make the table match. A difference is a finding for the coordinator.
