# Replay Data (Pull Request 1): Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Commit, for each of the seven demos, an archive of what its packaged run produced, with tests that pin the archive to the code by its inputs and by bounded summaries, and a CI report that compares it in full without gating anything.

**Architecture:** A private module `src/clocks/_replay.py` holds the archive format (a validated `.npz` with a JSON `meta` string), the display-sample rule, a recorder that reads a particle filter's public history, and a full comparison. Each demo module gains `scenario()` and `record()` beside its untouched command. `scripts/record_replays.py` writes the archives and the report. Tests gate on scenario inputs and on posterior summaries after observations 1, 8 and last; a separate workflow reports full reproduction from three runner allocations.

**Tech Stack:** Python 3.12+, NumPy, SciPy, pytest, uv, ruff, GitHub Actions.

**Status:** Draft, in review.

**Spec:** `docs/plans/2026-11-19-replay-and-animation-design.md`, sections 1 to 3 and item 1 of section 11. Read them before any task. Where this plan and the spec disagree, the spec wins and the disagreement is a plan bug to report. This plan covers pull request 1 only; nothing here draws a figure that is merged.

## Global Constraints

- Python only through uv: `uv run pytest`, `uv run ruff format .`, `uv run ruff check .`, `uv run python ...`. Never bare `python` or `pip`.
- Before every commit run the full local gate: `uv run ruff format --check .`, `uv run ruff check .`, `uv run pytest`, and `uv run pytest -m slow`. Run the checks and the commit in separate commands, and read the output before committing.
- Commit messages are passed with a quoted heredoc (`<<'MSG'`) and end with a `Co-Authored-By:` trailer naming the model that wrote the commit. Commit after each task and push; do not open a pull request (the coordinator does).
- Nothing under `assets/` or `site/assets/` changes except the seven new files `assets/replays/<demo>.npz`. `tests/test_generated_assets.py` stays green.
- Every command keeps its present path and output. In each `src/clocks/_demos/demo_*.py` the diff contains additions only: `build_parser`, `_run_demo` and `main` are not edited, the demos still write their GIFs, and `demo-density` keeps its shared generator.
- The seven demo names, in this order everywhere: `demo_1d`, `demo_2d`, `demo_multi_mass`, `demo_multi_mass_2d`, `demo_model_comparison`, `demo_density`, `demo_echolocation_3d`.
- Format name `"clocks-replay"`, version `1`. Arrays are float64 except `cloud` (float32) and the integer arrays (int64). At most 500 particles are displayed: `cloud_size = min(500, n_particles)`.
- The display-sample offset is φ = (√5 − 1) / 2, with `searchsorted(..., side="left")` on a cumulative sum divided by its last entry.
- Row 0 of every per-observation array is the prior. Row-0 sentinels are written as exact values: `log_evidence[0] = 0.0`, `ess[0] = float(n_particles)`, and for model comparison `model_log_evidence[0] = 0.0`, `model_ess[0] = float(n_particles)`, `model_posterior[0] = model_prior`.
- Tolerances: computed inputs (`true_rates`, `observations`, `filter_observations`) `rtol=1e-12, atol=1e-12`; full comparison of other float64 arrays `rtol=1e-9, atol=1e-12`; `cloud` `rtol=1e-5, atol=1e-6`; integers and `meta` (without `created_with`) exactly.
- `created_with` is provenance. No test or comparison reads its values.
- The CI test job (`.github/workflows/ci.yml`) is not edited.

## Review Focus

Conditions the spec implies but does not spell out a test for, most likely to bite first. Each has a test in the task that owns the code.

1. **`np.savez_compressed` appends `.npz` to a path without that suffix** (measured), so `save_replay(Path("x"))` would write `x.npz` and report success. `save_replay` must raise `ValueError` for any other suffix. Task 1.
2. **Weights that do not sum to 1.** `thin` divides the cumulative sum by its last entry; without that, weights scaled by 0.5 send the upper targets past the end and index out of range. Task 2.
3. **A float32 dot outside a float64 bound.** `np.float32(0.15)` is 0.15000000596, above 0.15 (measured), so a particle just under the mass bound fails a naive in-range check once stored. The check rounds the bounds to float32 first. Task 9.
4. **A candidate whose posterior underflows to exactly 0.** At the end of the packaged run P(K=1) is 0.0. The evidence check works in log space and must give 0 there with no warning and no NaN. Task 9.
5. **The report must be written when its surroundings are broken**: no git, no committed archive for a demo, a platform different from the archive's. Task 8.

---

### Task 1: The archive format

**Files:**
- Create: `src/clocks/_replay.py`
- Test: `tests/test_replay.py`

**Interfaces:**
- Produces:
  - `FORMAT = "clocks-replay"`, `VERSION = 1`, `CLOUD_SIZE = 500`, `DEMOS: tuple[str, ...]` (the seven names in order).
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

`FIXTURES = {"demo_1d": filter_fixture, "demo_echolocation_3d": echo_fixture, "demo_model_comparison": model_fixture}`; "each fixture" below means parametrized over its three keys, with the key as the test ID.

- `test_literal_archive_loads` (each fixture): write with plain NumPy, `np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)`; `load_replay(path)` returns `meta` equal to the literal dict, and for every key an array `np.array_equal` to the literal with the same `dtype`.
- `test_saved_archive_reads_back_as_literals` (each fixture): `save_replay(path, Replay(meta, arrays))`; read with `np.load(path, allow_pickle=False)`; `set(z.files) == {"meta", *arrays}`; `json.loads(z["meta"].item()) == meta`; every array equal with the same dtype. (`z["meta"]` is a 0-d `<U` array and `.item()` gives the `str`: measured.)
- `test_missing_key_is_rejected` (each fixture × each array key): `ValueError` whose message contains the key and `missing`.
- `test_extra_key_is_rejected` (each fixture): adding `"surplus": np.zeros(1)` raises `ValueError` naming `surplus`.
- `test_wrong_dtype_is_rejected` (each fixture × each array key): float64 cast to float32, float32 to float64, int64 to int32; the message contains the key and the expected dtype name.
- `test_wrong_rank_is_rejected` (each fixture × each array key): `array[None]` raises, naming the key.
- `test_wrong_length_is_rejected` (each fixture × each array key except `clock_positions` × each axis): the array with its last entry along that axis removed raises, naming the key.
- `test_clock_count_must_agree` (each fixture): `clock_positions` with a third clock row appended raises `ValueError`.
- `test_meta_keys_are_exact` (each fixture × `missing`, `extra`): removing `"seed"`, or adding `"surplus": 1`, raises naming `meta` and the key.
- `test_wrong_format_or_version_is_rejected`: `"format": "other"` and `"version": 2` each raise.
- `test_unknown_demo_is_rejected`: `"demo": "demo_9d"` raises naming the value.
- `test_non_finite_value_is_rejected` (each fixture × each floating key): one entry set to `np.nan` raises naming the key.
- `test_descending_model_ks_is_rejected`: `model_ks = [2, 1]` raises naming `model_ks`.
- `test_cloud_size_must_match_meta`: `"cloud_size": 4` in the filter fixture raises naming `cloud`.
- `test_save_requires_an_npz_path`: `save_replay(tmp_path / "x", ...)` raises `ValueError` and writes nothing (`list(tmp_path.iterdir()) == []`).
- `test_save_refuses_an_invalid_replay`: a fixture with `mean` removed raises before any file exists.
- `test_unreadable_file_is_a_value_error`: a file holding `b"not a zip"` raises `ValueError` whose message contains its path. (`np.load` raises `ValueError` there, with a misleading message about pickles: measured.)

- [ ] **Step 2: Run them and confirm they fail** with `ModuleNotFoundError: clocks._replay`.

Run: `uv run pytest tests/test_replay.py -q`

- [ ] **Step 3: Implement.** Decisions the tests do not settle:

- `meta` is serialized with `json.dumps(meta, ensure_ascii=False, sort_keys=True)` into `np.array(text)` under the key `meta`.
- `load_replay` reads every array into memory inside `with np.load(path, allow_pickle=False) as archive:` and returns after the file is closed. `ValueError`, `OSError` and `zipfile.BadZipFile` from opening, and `json.JSONDecodeError`, become `ValueError(f"{path}: not a replay archive: {error}")`.
- Every validation message has the form `f"{source}: {key}: expected {expected}, got {got}"`, where `key` is an array key or `meta.<field>`, and `source` is the path for files.
- Required `meta` keys, exactly: `format`, `version`, `demo`, `kind`, `seed`, `n_observations`, `n_particles`, `noise_std`, `track_offset`, `param_names`, `prior_ranges`, `controls`, `created_with`; plus `cloud_size` for kind `filter` and `true_k` for kind `model_comparison`.
- Sizes: `C, D = clock_positions.shape`; `T = n_observations`; `P = len(param_names)`; `m = cloud_size`; `K = len(model_ks)`. `prior_ranges` has `P` pairs for kind `filter` and two pairs (positions, masses) for `model_comparison`; each pair has `lo < hi`.
- Array table (all float64 unless stated):

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

  The key set for a demo is exactly the rows that apply to it. `kind` must be `model_comparison` for `demo_model_comparison` and `filter` for the other six.
- `cloud_size` must equal `cloud.shape[1]` and be at most `n_particles`. The loader does not check that it is `min(500, n_particles)`; the recorder guarantees that.
- `save_replay` validates first, then requires `path.suffix == ".npz"`, then writes with `np.savez_compressed`.

- [ ] **Step 4: Run the tests and confirm they pass.** Then run the full gate.

- [ ] **Step 5: Commit and push** with message `Add the replay archive format with a validating loader`.

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

A scripted double used by the moment tests:

```python
class ScriptedFilter:
    """Replays prepared states; has exactly the surface record_filter may read."""

    def __init__(self, states, diagnostics, log_evidence):
        self._states, self._diagnostics, self._log_evidence = states, diagnostics, log_evidence
        self.n_particles = len(states[0].weights)
        self.history = [states[0]]
        self.diagnostics_history, self.log_evidence_history = [], []

    @property
    def state(self):
        return self.history[-1]

    def update(self, observation):
        step = len(self.history)
        self.history.append(self._states[step])
        self.diagnostics_history.append(self._diagnostics[step - 1])
        self.log_evidence_history.append(self._log_evidence[step - 1])
        return self.history[-1]
```

The scripted history has N = 5 particles, P = 2 and three states: the prior with uniform weights, then weights `[0.1, 0.2, 0.3, 0.25, 0.15]` and `[0.05, 0.4, 0.05, 0.3, 0.2]`. Particles are literals whose second column is correlated with the first (for example column 1 = 0.5 × column 0 plus a small literal offset per particle), different in each state. Diagnostics are `UpdateDiagnostics(3, 24, 6)` and `UpdateDiagnostics(1, 8, 3)`; log evidence `[5.0, 11.0]`. Observations are two `Observation`s with two clocks.

- `test_thin_picks` (IDs `unequal`, `uniform_7_3`, `uniform_4_4`), with `particles = np.arange(N, dtype=float)[:, None]` so a row's value is its index:
  - weights `[0.1, 0.2, 0.0, 0.3, 0.4]`, `size=5` → `[1, 3, 3, 4, 4]`;
  - seven uniform weights, `size=3` → `[1, 3, 6]`;
  - four uniform weights, `size=4` → `[0, 1, 2, 3]`.
  Each result has dtype float32 and shape `(len(expected), 1)`. (All three measured; offset 0.5 gives `[0, 1, 3, 4, 4]` and `[1, 3, 5]`.)
- `test_thin_ignores_the_scale_of_the_weights`: the first case with weights multiplied by 0.5 gives the same picks.
- `test_thin_is_capped_by_the_particle_count`: four particles, `size=500` → four rows.
- `test_moments_match_an_independent_calculation`: for every row, `mean` and the whole of `cov` equal values the test computes itself with explicit sums (`m = Σ wᵢxᵢ / Σ wᵢ`; `cov = Σ wᵢ (xᵢ−m)(xᵢ−m)ᵀ / Σ wᵢ`), `rtol=1e-12`; and `abs(cov[1][0, 1]) > 0`, so a diagonal-only implementation cannot pass.
- `test_row_zero_holds_sentinels`: `ess[0] == 5.0` and `log_evidence[0] == 0.0` exactly; `mean[0]` equals the prior's moments; `cloud[0]` equals `thin(prior particles, prior weights, 500)`. (1 / Σw² for five uniform weights is 4.999999999999999, so a computed row 0 fails: measured.)
- `test_ess_rows_follow_the_weights`: rows 1 and 2 equal `1 / Σ wᵢ²`, `rtol=1e-12`.
- `test_diagnostics_and_evidence_are_copied`: the three integer arrays are int64 and equal `[3, 1]`, `[24, 8]`, `[6, 3]`; `log_evidence` equals `[0.0, 5.0, 11.0]`.
- `test_predictions_are_moments_of_predictions`: with `predict = lambda p: np.column_stack([p[:, 0] ** 2, np.exp(p[:, 1])])`, `predicted_mean` and `predicted_std` equal the weighted mean and weighted standard deviation of `predict(particles)` per row, `rtol=1e-12`; and `predicted_mean[1]` differs from `predict(mean[1][None])[0]` by more than 1e-6.
- `test_observations_are_what_the_filter_consumed`: `observations` equals the stacked `rates`; when `arrays` already holds `"observations"`, that array is kept unchanged.
- `test_recording_does_not_change_the_run`: two filters from `build_particle_filter` with the `demo_1d` constants at 64 particles, both seeded 42, and three observations from `simulate`. One runs through `record_filter` with `predict=pf.forward_model_batch`; the other is updated in a plain loop. First assert the plain run did real work: its three `mh_acceptances` sum to more than 0 and its generator state differs from a fresh filter's. Then: every history row has `np.array_equal` particles and weights, and `pf.rng.bit_generator.state` is equal at the end. (Measured for this configuration: 213, 92 and 53 acceptances.)
- `test_final_row_matches_the_filter_estimate`: for that real run, `mean[-1]`, `sqrt(diag(cov[-1]))` and `ess[-1]` equal `pf.estimate()`'s `mean`, `std` and `ess`, `rtol=1e-12`.
- `test_cloud_is_float32_and_thinned`: for that run `cloud.shape == (4, 64, 2)` and dtype float32.
- `test_a_used_filter_is_refused`: a real filter after one update raises `RuntimeError` matching `fresh`; afterwards `pf.state.observations_seen == 1` and `len(pf.history) == 2`.

- [ ] **Step 2: Run them and confirm they fail** on the missing names.

- [ ] **Step 3: Implement.**

- `thin`: `c = np.cumsum(weights, dtype=np.float64)`, `c /= c[-1]`; targets `(np.arange(m) + φ) / m` with `m = min(size, len(weights))`; indices from `np.searchsorted(c, targets, side="left")`; return `particles[indices].astype(np.float32)`. No random numbers. Negative weights or a non-positive total raise `ValueError`.
- `record_filter`: raise `RuntimeError(f"record_filter needs a fresh filter; this one has seen {n} observations")` before any update. Feed the observations in order. Build every per-row array from `pf.history` after the loop, weighting by each state's weights; row 0 of `ess` and `log_evidence` are the sentinels of Global Constraints; rows from 1 are `1 / Σw²` and `pf.log_evidence_history`. `cloud` uses `CLOUD_SIZE`. `predict` is called once per row on that row's particles. Draw nothing from any generator.

- [ ] **Step 4: Run the tests and confirm they pass.** Then run the full gate.

- [ ] **Step 5: Commit and push** with message `Record a filter's history: summaries from the whole cloud, a thinned display sample`.

---

### Task 3: Full comparison of two replays

**Files:**
- Modify: `src/clocks/_replay.py`
- Test: `tests/test_replay.py`

**Interfaces:**
- Consumes: `Replay`.
- Produces: `Difference` (`NamedTuple` with `key: str`, `row: int | None`, `largest: float`) and `first_difference(reference: Replay, other: Replay) -> Difference | None`.

- [ ] **Step 1: Write the failing tests**, using `filter_fixture()` and `model_fixture()` as the replays.

- `test_equal_replays_have_no_difference`: `None`, also when the two `created_with` blocks differ in every value.
- `test_meta_difference_is_reported_first`: `seed` changed, and `mean` also changed → `Difference("meta", None, ...)`.
- `test_key_set_difference`: a key removed from one side → `key == "keys"`.
- `test_shape_or_dtype_difference`: `mean` with a row removed (not validated here) → `key == "mean"`, `row is None`.
- `test_integer_arrays_compare_exactly`: `mh_proposals[1] += 1` → `Difference("mh_proposals", 1, 1.0)`.
- `test_float_tolerance` (IDs `inside`, `outside`): `mean[2, 0]` scaled by `1 + 1e-10` → `None`; by `1 + 1e-8` → `key == "mean"`, `row == 2`.
- `test_cloud_tolerance` (IDs `inside`, `outside`): `cloud[1, 0, 0]` scaled by `1 + 1e-6` → `None`; by `1 + 1e-4` → `key == "cloud"`, `row == 1`.
- `test_computed_inputs_use_the_tight_tolerance`: `observations[1, 0] += 1e-10` → `key == "observations"`, `row == 1`.
- `test_the_earliest_row_is_reported`: `mean` changed in rows 1 and 2 → `row == 1`, and `largest` is the largest absolute difference anywhere in `mean`.
- `test_keys_are_compared_in_a_fixed_order`: `cov` and `mean` both changed → `key == "cov"`; `observations` and `cov` both changed → `key == "observations"`.

- [ ] **Step 2: Run them and confirm they fail.**

- [ ] **Step 3: Implement.** Order of comparison: `meta` without `created_with`; the key sets; then arrays, inputs first in the order `clock_positions`, `true_params`, `true_rates`, `observations`, `filter_observations`, `model_ks`, `model_prior`, then every other key alphabetically. For each array: a shape or dtype mismatch is a difference with `row=None`; integers compare exactly; `true_rates`, `observations` and `filter_observations` with `rtol=1e-12, atol=1e-12`; `cloud` with `rtol=1e-5, atol=1e-6`; every other array with `rtol=1e-9, atol=1e-12`. `row` is the smallest index along axis 0 holding an element outside tolerance.

- [ ] **Step 4: Run the tests and confirm they pass.** Then run the full gate.

- [ ] **Step 5: Commit and push** with message `Compare two replays in full and name the first difference`.

---

### Task 4: `scenario()` and `record()` for the four point-mass demos

**Files:**
- Modify: `src/clocks/_replay.py`, `src/clocks/_demos/demo_1d.py`, `demo_2d.py`, `demo_multi_mass.py`, `demo_multi_mass_2d.py`
- Create: `tests/test_demo_records.py`

**Interfaces:**
- Consumes: `Replay`, `validate_replay`, `record_filter`, `provenance`, `CLOUD_SIZE`.
- Produces, in `_replay.py`:
  - `Scenario`: `@dataclass(frozen=True, eq=False)` with `demo: str`, `clock_array: ClockArray`, `true_params: NDArray[np.float64]`, `true_rates: NDArray[np.float64]`, `param_names: tuple[str, ...]`, `prior_ranges: tuple[tuple[float, float], ...]`, `seed: int`, `noise_std: float`, `n_observations: int`, `n_particles: int`.
  - `scenario_meta(scenario: Scenario, *, kind: str, n_observations: int, n_particles: int, controls: Mapping[str, object], **extra: object) -> dict[str, object]`: the complete `meta`, with `param_names` and `prior_ranges` as lists, `track_offset` from the clock array, `cloud_size = min(CLOUD_SIZE, n_particles)` for kind `filter`, `created_with = provenance()`, and `extra` merged in.
  - `filter_controls(pf) -> dict[str, object]`: `ess_target`, `rejuvenation_steps`, `proposal_scale`, `resampling` read from the filter.
- Produces, in each of the four demo modules: `scenario() -> Scenario` (packaged defaults, reading the module's constants when called) and `record(*, n_observations: int = N_OBSERVATIONS, n_particles: int = N_PARTICLES) -> Replay`, validated.

- [ ] **Step 1: Write the failing tests** in `tests/test_demo_records.py`, parametrized over the four module names. `SMALL = {"n_observations": 3, "n_particles": 64}`.

- `test_scenario_values`: `demo` equals the module name; `seed`, `noise_std`, `n_observations`, `n_particles` equal the module's (for `demo_multi_mass_2d`, `_scenarios`') constants; `param_names` is `("x", "M")`, `("x", "y", "M")`, `("x₁", "x₂", "M₁", "M₂")`, `("x₁", "y₁", "x₂", "y₂", "M₁", "M₂")`; `prior_ranges` is `(-8.0, 8.0)` for each position and `(0.005, 0.15)` for each mass; `true_params` equals the module's truth constants in that order; `true_rates` equals `clock_rates(MassConfig(...), clock_array)` for that truth.
- `test_two_mass_2d_clocks_come_from_the_seed`: `demo_multi_mass_2d.scenario().clock_array.positions` equals `generate_random_clocks(N_CLOCKS, np.random.default_rng(11), exclude=[tuple(p) for p in TRUE_POSITIONS])`.
- `test_record_is_a_valid_replay`: `record(**SMALL)` passes `validate_replay`; `meta["demo"]`, `meta["n_observations"] == 3`, `meta["n_particles"] == 64`, `meta["cloud_size"] == 64`; `meta["controls"] == {"ess_target": 0.8, "rejuvenation_steps": 2, "proposal_scale": 2.38, "resampling": "systematic"}`; `predicted_mean` is present for `demo_1d` and `demo_multi_mass` and absent for the other two.
- `test_record_inputs_match_the_scenario`: `clock_positions`, `true_params` and `true_rates` are `np.array_equal` to the scenario's; `observations` is `np.array_equal` to the three readings of `simulate(SimulationConfig(..., n_observations=3, seed=SEED))`.
- `test_record_agrees_with_the_public_api`: `infer(those observations, InferenceConfig(..., n_particles=64, n_masses=..., seed=SEED))` gives `posterior_mean` and `posterior_std` equal to `mean[-1]` and `sqrt(diag(cov[-1]))`, `rtol=1e-12`.
- `test_record_defaults_are_the_packaged_run`: `inspect.signature(module.record)` has defaults equal to the scenario's `n_observations` and `n_particles`.
- `test_record_handles_one_observation`: `record(n_observations=1, n_particles=16)` validates and has two rows.

- [ ] **Step 2: Run them and confirm they fail.**

- [ ] **Step 3: Implement.** Each `record()` builds its simulation and filter exactly as that module's `_run_demo` does (`simulate(SimulationConfig(..., seed=SEED))`, `build_particle_filter(InferenceConfig(..., seed=SEED))`) and passes `record_filter` the scenario arrays `clock_positions`, `true_params`, `true_rates`. `demo_1d` and `demo_multi_mass` pass `predict=pf.forward_model_batch`, which returns tick rates at the clocks, shape (N, C) (measured); the 2D demos pass no `predict`. `_run_demo`, `build_parser` and `main` are not touched: after the change, `git diff main -- src/clocks/_demos | grep '^-[^-]'` prints nothing.

- [ ] **Step 4: Run the tests and confirm they pass.** Run `uv run pytest tests/test_cli.py -q` to confirm the commands still write their GIFs. Then run the full gate.

- [ ] **Step 5: Commit and push** with message `Give the four point-mass demos a scenario and a recorder`.

---

### Task 5: `record()` for echolocation and density

**Files:**
- Modify: `src/clocks/_demos/demo_echolocation_3d.py`, `src/clocks/_demos/demo_density.py`
- Test: `tests/test_demo_records.py`

**Interfaces:**
- Consumes: Task 4's `Scenario`, `scenario_meta`, `filter_controls`; `record_filter`; `validate_replay`; `study_generator` from `clocks._rng`.
- Produces: `scenario()` and `record()` in both modules, as in Task 4; in `demo_echolocation_3d`, `close_view(truth_xyz: NDArray, mean_xyz: NDArray, sd_xyz: NDArray) -> tuple[float, float]` and `weight_in_view(states: Sequence[ParticleState], bounds: tuple[float, float]) -> NDArray[np.float64]`.

- [ ] **Step 1: Write the failing tests.**

Echolocation (`record(n_observations=3, n_particles=64)` unless stated):

- `test_echolocation_scenario_values`: `seed == DEMO_SEED`; `param_names == ("x", "y", "z", "M")`; `prior_ranges` is `(-16.0, 16.0)` three times then `(0.005, 0.15)`; `true_params` equals `np.append(echo_mass_position(DEMO_RANGE_R), ECHO_M_TRUE)`; `noise_std == ECHO_NOISE_STD`; defaults `ECHO_N_OBSERVATIONS` and `ECHO_N_PARTICLES`.
- `test_echolocation_record_is_valid`: validates; has `filter_observations`, `view_bounds`, `weight_in_view`, `predicted_mean`, `predicted_std`; `meta["controls"] == {"ess_target": 0.9, "rejuvenation_steps": 1, "proposal_scale": 1.5, "resampling": "systematic"}`.
- `test_echolocation_inputs_are_the_packaged_readings`: with `_, display, contrasts = make_echo_observations(DEMO_SEED, DEMO_RANGE_R, n_observations=3)`, `observations` equals the stacked `display` rates and `filter_observations` the stacked `contrasts` rates (`np.array_equal`).
- `test_echolocation_channels_are_centred_contrasts`: each `observations` row sums to zero within an absolute 1e-12 (measured sums are up to 4e-15); `filter_observations` equals `contrast_matrix(27) @ row` with `rtol=1e-12, atol=1e-12`.
- `test_echolocation_predictions_are_centred`: every `predicted_mean` row sums to zero within 1e-12.
- `test_echolocation_record_agrees_with_an_unrecorded_filter`: a second `build_echolocation_filter(DEMO_SEED, n_particles=64)` updated with the same contrasts has `estimate()["mean"]` equal to `mean[-1]`, `rtol=1e-12`.
- `test_close_view` (IDs `packaged`, `cloud_below_the_head`, `rounding_up`, `rounding_down`):
  - truth `[0.99, 1.48, 2.97]`, mean `[[1.174, 1.810, 3.379]]`, sd `[[0.164, 0.212, 0.264]]` → `(-1.3, 4.8)`;
  - truth `[0.0, 0.0, 0.5]`, mean `[[0.0, 0.0, -2.0]]`, sd `[[0.1, 0.1, 0.5]]` → `(-4.3, 1.3)`;
  - truth `[0.0, 0.0, 1.1]`, mean `[[0.0, 0.0, 1.1]]`, sd zeros → `(-1.3, 1.4)`;
  - truth `[0.0, 0.0, -1.1]`, mean `[[0.0, 0.0, -1.1]]`, sd zeros → `(-1.4, 1.3)`.
  Each compared with `==`. (The last two guard a float trap: `(1.1 + 0.3) * 10` is 14.000000000000002, which a bare `ceil` turns into 1.5: measured.)
- `test_weight_in_view`: two states of four particles with weights `[0.1, 0.2, 0.3, 0.4]`; with bounds `(-1.0, 1.0)` and the second and fourth particles outside on one coordinate, the result is `[0.4, ...]` as computed by hand for each state; a particle exactly on a bound counts as inside.
- `test_echolocation_view_holds_the_head_the_truth_and_the_cloud`: `view_bounds[0] <= -1.0`, `view_bounds[1] >= max(1.0, true_params[:3].max())`; every `weight_in_view` entry is in `[0, 1 + 1e-12]`; `weight_in_view[-1] > 0.9`.

Density (`record(n_observations=3, n_particles=64)` unless stated; this takes under a second):

- `test_density_scenario_values`: `param_names == ("μ", "σ", "A")`; `prior_ranges == (MU_RANGE, SIGMA_RANGE, AMPLITUDE_RANGE)`; `true_params == (TRUE_MU, TRUE_SIGMA, TRUE_AMPLITUDE)`; `true_rates` equals `clock_rates_density_gaussian(true_params, clock_array)`.
- `test_density_record_is_valid`: validates; has no `predicted_mean`; controls are the defaults of Task 4.
- `test_density_readings_come_from_their_own_generator`: `observations` is `np.array_equal` to three successive `add_clock_noise(true_rates, NOISE_STD, rng)` draws from one `study_generator(SEED)`.
- `test_density_readings_do_not_depend_on_particles`: `record(n_observations=3, n_particles=32)` and `record(n_observations=3, n_particles=64)` have `np.array_equal` observations.
- `test_density_record_agrees_with_an_unrecorded_filter`: `build_density_filter(clock_array, study_generator(SEED), n_particles=64)` updated with those readings has `estimate()["mean"]` equal to `mean[-1]`, `rtol=1e-12`.
- `test_density_true_rates_need_no_readings`: `scenario().true_rates` is computed without drawing: calling `scenario()` twice gives equal arrays, and equal to the forward model at the truth.

- [ ] **Step 2: Run them and confirm they fail.**

- [ ] **Step 3: Implement.**

- Echolocation `record()`: readings from `make_echo_observations(DEMO_SEED, DEMO_RANGE_R, n_observations=...)`; filter from `build_echolocation_filter(DEMO_SEED, n_particles=...)`; the filter consumes the contrasts. Pass `record_filter` the arrays `clock_positions`, `true_params`, `true_rates` (the simulation's, not centred), `observations` (the centred display readings) and `filter_observations`. `predict` returns each particle's tick rates at the 27 clocks with that particle's own mean over clocks removed (`clock_rates_batch(p[:, :3], p[:, 3], clock_array)`, as `_animate.predicted_centered` does today). Afterwards compute `view_bounds` and `weight_in_view` from the recorded `mean`, `cov` and `pf.history`, add them, and validate.
- `close_view`: `lo_raw = min(-1.0, truth.min(), (mean - 4*sd).min()) - 0.3`; `hi_raw = max(1.0, truth.max(), (mean + 4*sd).max()) + 0.3`, where `mean` and `sd` are the rows for k ≥ 1; `lo = floor(round(lo_raw * 10, 6)) / 10`, `hi = ceil(round(hi_raw * 10, 6)) / 10`.
- `weight_in_view`: per state, the sum of the weights of particles whose three coordinates all lie in `[lo, hi]`, bounds included.
- Density `record()`: clocks and truth as in `_run_demo`; all readings drawn first from `study_generator(SEED)` with `add_clock_noise`; the filter from `build_density_filter(clock_array, study_generator(SEED), n_particles=...)`. `_run_demo` keeps `np.random.default_rng(SEED)` shared between prior, readings and filter; it is not edited. No `predict`.

- [ ] **Step 4: Run the tests and confirm they pass.** Then run the full gate.

- [ ] **Step 5: Commit and push** with message `Record the echolocation and density runs; give density's readings their own generator`.

---

### Task 6: `record()` for model comparison

**Files:**
- Modify: `src/clocks/_demos/demo_model_comparison.py`
- Test: `tests/test_demo_records.py`

**Interfaces:**
- Consumes: `Scenario`, `scenario_meta`, `filter_controls`, `Replay`, `validate_replay`.
- Produces: `scenario()` and `record()` as in Task 4, and the constant `MODEL_KS = (1, 2, 3)`.

- [ ] **Step 1: Write the failing tests** (`record(n_observations=3, n_particles=64)` unless stated).

- `test_model_scenario_values`: `param_names == ("x₁", "x₂", "M₁", "M₂")`; `prior_ranges == ((-8.0, 8.0), (0.005, 0.15))`; `true_params == (TRUE_X1, TRUE_X2, TRUE_M1, TRUE_M2)`.
- `test_model_record_is_valid`: validates; `meta["kind"] == "model_comparison"`; `meta["true_k"] == 2`; `model_ks` equals `[1, 2, 3]` with dtype int64; `model_prior` equals `[1/3, 1/3, 1/3]` exactly as the library computes it (`1.0 / 3`).
- `test_model_record_row_zero_is_the_prior`: with `n_particles=80`, `model_posterior[0]` is `np.array_equal` to `model_prior`; `model_log_evidence[0]` is all `0.0`; `model_ess[0]` is all `80.0` exactly. (1 / Σw² for 80 uniform weights is 79.99999999999997, so a computed row fails: measured.)
- `test_model_record_agrees_with_the_public_api`: `result = infer(observations, InferenceConfig(..., n_particles=64, n_masses=(1, 2, 3), seed=SEED))`; `model_posterior[-1]` equals `[result.posterior_by_model[k] for k in (1, 2, 3)]`, `rtol=1e-12`; and `model_posterior[1:]` equals `result.history`, row by row.
- `test_model_record_final_estimates_match_the_public_api`: for each k, `final_mean_k<k>` and `final_std_k<k>` equal `result.result_by_model[k].posterior_mean` and `.posterior_std`, `rtol=1e-12`; `final_mean_k3` has six entries whose first three are strictly increasing (positions first, then masses: measured from the prior sampler).
- `test_model_posterior_follows_the_evidence`: every row of `model_posterior` equals `exp(l − logsumexp(l))` with `l = log(model_prior) + model_log_evidence[row]`, `rtol=1e-9, atol=1e-12`. (The library's own posterior reproduces this with zero difference: measured.)
- `test_model_record_inputs_match_the_scenario`: as Task 4's input test, with `observations` equal to `simulate(...)`'s.

- [ ] **Step 2: Run them and confirm they fail.**

- [ ] **Step 3: Implement.** `record()` builds `build_model_comparison(InferenceConfig(..., n_masses=MODEL_KS, seed=SEED))` once and updates it observation by observation, reading `comparison.evidence()` after each for the posterior and log evidence and each filter's `estimate()["ess"]` for `model_ess`. `model_prior` comes from `comparison.model_prior` in `comparison.k_values` order. The final estimates come from each filter's `estimate()`. Controls are `filter_controls(comparison.filters[1])`. There is no `cloud` and no `cloud_size`. `_run_demo` still calls `infer` and then animates a second comparison; it is not edited.

- [ ] **Step 4: Run the tests and confirm they pass.** Then run the full gate.

- [ ] **Step 5: Commit and push** with message `Record the model-comparison run in one pass`.

---

### Task 7: Smoke-draw the undrawn panels and probe the 0.10 run

No tracked file changes. This task finds out, before any archive is committed, whether version 1 of the format holds what the later drawings need. Work in the scratch directory the coordinator names in the dispatch; write throwaway code there, not in the repository.

**Interfaces:**
- Consumes: every demo's `record()`; `save_replay`, `load_replay`; `clocks.viz.style` and `THEMES` for the look.

- [ ] **Step 1: Record candidates.** Call each demo's `record()` at its defaults and save the seven archives under the scratch directory. The density run takes about 35 s; run it in the foreground with `timeout: 600000`.

- [ ] **Step 2: Draw, from the saved archives only** (load each with `load_replay`; import nothing from `clocks.physics` or `clocks.inference`), at k = 0, 1, 8 and last, in the light theme:

| Panel (spec section 5 or 6) | Archive | Must be drawable from |
|---|---|---|
| Slowing at each clock: true value, predicted mean ± s.d., running mean of readings ± σ/√k | `demo_1d`, `demo_multi_mass` | `true_rates`, `predicted_mean`, `predicted_std`, `observations`, `meta.noise_std`, `clock_positions` |
| (x, M) cloud with both masses, truth marks, covariance ellipses | `demo_multi_mass` | `cloud`, `true_params`, `mean`, `cov`, `prior_ranges` |
| Map with both clouds and six traces | `demo_multi_mass_2d` | `cloud`, `mean`, `cov`, `true_params`, `clock_positions`, `observations` |
| Posterior lines with a label gutter and "(true model)" | `demo_model_comparison` | `model_ks`, `model_posterior`, `meta.true_k` |
| Density profile: truth, pointwise median and 68% band | `demo_density` | `cloud[-1]`, `true_params`, `prior_ranges` |
| Whole prior box with the close view marked; weight in view | `demo_echolocation_3d` | `cloud`, `view_bounds`, `weight_in_view` |

These are rough drawings. Their purpose is the right-hand column, not their layout.

- [ ] **Step 3: Probe the 2D run at M = 0.10.** In a scratch script set `clocks._demos.demo_2d.TRUE_M = 0.10` and call `record()`. Report the posterior mean and standard deviation after observations 1, 8 and 50. The spec measured the end as (1.4169, −1.2223, 0.1000) ± (0.1111, 0.1148, 0.0012); say whether this run agrees to those digits. Save nothing in the repository.

- [ ] **Step 4: Report to the coordinator**: one contact sheet per panel row (ImageMagick `magick montage`, or matplotlib), the paths, the probe's numbers, and a list of anything a panel needed that the archive did not hold. If that list is not empty, stop: the format changes before Task 9, and that is the coordinator's decision. The coordinator inspects the sheets and attaches them to the pull request.

---

### Task 8: The recording script

**Files:**
- Create: `scripts/record_replays.py`, `tests/test_record_replays_script.py`

**Interfaces:**
- Consumes: `DEMOS`, `save_replay`, `load_replay`, `first_difference`, `provenance`; each demo's `record()`.
- Produces: `main(argv: Sequence[str] | None = None) -> int` with `--out DIR` (required), optional demo names (default all seven; `argparse` `choices`), `--compare DIR2` and `--report FILE`. It returns 0 when nothing was compared or every compared demo reproduced, and 1 otherwise. The script defines no dataclass at module level, because the tests import it by path without registering it and a dataclass then fails at import (spec, M14).

- [ ] **Step 1: Write the failing tests**, importing the script as `tests/test_site_prose_contract.py` imports `check_site_links`. Every test records only `demo_1d` (a quarter of a second).

- `test_records_the_named_demo`: `main(["--out", str(out), "demo_1d"]) == 0`; `out` is created; it holds exactly `demo_1d.npz`, which `load_replay` accepts.
- `test_an_unknown_name_is_a_usage_error`: `SystemExit` with code 2.
- `test_compare_reports_a_reproduced_demo`: record into `a`; then `main(["--out", b, "--compare", a, "--report", report, "demo_1d"]) == 0` and the report contains the line `demo_1d: reproduced`.
- `test_compare_names_the_first_difference`: record into `a`; load `a/demo_1d.npz`, add `1e-6` to `mean[3, 0]`, save it back; the same command returns 1 and the report contains `demo_1d: differs at mean, row 3`.
- `test_compare_ignores_created_with`: record into `a`; rewrite the archive with every `created_with` value replaced by `"elsewhere"`; the command returns 0 and reports `reproduced`.
- `test_a_missing_archive_is_reported`: `--compare` pointing at an empty directory returns 1; the report contains `demo_1d: no committed archive`; `b/demo_1d.npz` was still written.
- `test_an_unreadable_archive_is_reported`: `a/demo_1d.npz` holding `b"not a zip"` returns 1; the report contains `demo_1d: committed archive unreadable`.
- `test_report_header_records_where_it_ran`: with `GITHUB_RUN_ID=123` and `REPLAY_HEAD_SHA=abc123` set, the report contains `run: 123`, `head: abc123`, a `checkout:` line, and lines for `python:` (`platform.python_version()`), `numpy:`, `scipy:`, `blas:` and `platform:`.
- `test_report_survives_a_missing_git`: with `subprocess.run` monkeypatched to raise `FileNotFoundError`, the report is written and contains `checkout: unknown`.
- `test_report_is_also_printed`: the report text appears on standard output.

- [ ] **Step 2: Run them and confirm they fail.**

- [ ] **Step 3: Implement.** Record each demo with its default `record()` and save `<out>/<name>.npz`. For `--compare`, load `<DIR2>/<name>.npz` and call `first_difference(committed, recorded)`; one line per demo: `reproduced`, `differs at <key>, row <row> (largest difference <largest:.3g>)` (omit the row when it is `None`), `no committed archive`, or `committed archive unreadable: <error>`. The header lines are `run:` (`GITHUB_RUN_ID`, else `local`), `head:` (`REPLAY_HEAD_SHA`, else `unknown`), `checkout:` (`git rev-parse HEAD`, else `unknown`), then `provenance()` and `blas:` from `np.show_config(mode="dicts")["Build Dependencies"]["blas"]` (name and version; on this Mac that is `accelerate unknown`: measured). Without `--report` the text goes to standard output only.

- [ ] **Step 4: Run the tests and confirm they pass.** Then run the full gate.

- [ ] **Step 5: Commit and push** with message `Add the script that records replays and reports how they compare`.

---

### Task 9: The seven archives, their invariants and the gate

**Files:**
- Create: `assets/replays/demo_1d.npz`, `demo_2d.npz`, `demo_multi_mass.npz`, `demo_multi_mass_2d.npz`, `demo_model_comparison.npz`, `demo_density.npz`, `demo_echolocation_3d.npz`; `tests/test_replays.py`

**Interfaces:**
- Consumes: `scripts/record_replays.py`; `load_replay`, `DEMOS`; each demo's `scenario()` and `record()`.

- [ ] **Step 1: Write the failing tests** in `tests/test_replays.py`. `ARCHIVES = ROOT / "assets/replays"`. "Each demo" is parametrized over `DEMOS` with the name as ID. Tests that call `record()` take the fresh replay from one cached helper (`functools.lru_cache`) so each demo is recorded once per session, and mark `demo_multi_mass_2d`, `demo_density` and `demo_echolocation_3d` with `pytest.mark.slow` through `pytest.param`.

Helpers in the test module, each usable on an in-memory `Replay` so it can be tried on a corrupted copy:

- `_inside_prior(cloud, prior_ranges) -> bool`: every column of every row lies between `np.float32(lo)` and `np.float32(hi)`, compared in float32.
- `_posterior_from_evidence(model_prior, model_log_evidence) -> NDArray`: `l = np.log(model_prior) + model_log_evidence`; `np.exp(l - logsumexp(l, axis=1, keepdims=True))`.
- `_checkpoints(n_observations) -> tuple[int, ...]`: `sorted({1, min(8, n_observations), n_observations})`.

Invariants on the committed archives (no inference; all seven in the default suite):

- `test_archives_are_exactly_the_seven`: the `.npz` stems in `ARCHIVES` equal `set(DEMOS)`.
- `test_archive_matches_its_scenario` (each demo): `meta` fields `demo`, `seed`, `noise_std`, `n_observations`, `n_particles`, `track_offset`, `param_names`, `prior_ranges` equal the scenario's (lists against tuples compared as lists); `clock_positions` and `true_params` are `np.array_equal` to the scenario's; `true_rates` matches with `rtol=1e-12, atol=1e-12`.
- `test_archive_truth_is_physical` (each demo): `np.max(np.abs(true_rates**2 - 1.0)) <= 0.08`; for `demo_multi_mass`, `demo_multi_mass_2d` and `demo_model_comparison`, `true_params[0] < true_params[1]` or, in 2D, `true_params[0] < true_params[2]` (x₁ < x₂).
- `test_archive_sentinels_and_ranges` (the six filter demos): `log_evidence[0] == 0.0`; `ess[0] == n_particles` exactly; `0 < ess` and `ess <= n_particles * (1 + 1e-12)` in every row; for every row, `cov` equals its transpose within `1e-12 * abs(cov).max()` and `np.linalg.eigvalsh(cov).min() >= -1e-12 * np.trace(cov)`; `mh_acceptances <= mh_proposals`; `tempering_stages >= 1`.
- `test_archive_dots_agree_with_summaries` (the six filter demos): `_inside_prior(cloud, prior_ranges)`; in every row and column, `abs(cloud.mean(axis=1) - mean) <= 6 * sd / sqrt(m)` and `0.6 <= cloud.std(axis=1) / sd <= 1.5`, with `sd = sqrt(diag(cov))` and `cloud` cast to float64 for the moments. (The spec's candidates reach 3.0 and 0.78 to 1.29.)
- `test_a_float32_dot_at_the_bound_is_inside`: `_inside_prior(np.array([[[np.nextafter(0.15, 0.0)]]], np.float32), [[0.005, 0.15]])` is true, although the stored value is greater than the float64 0.15.
- `test_echolocation_archive_channels`: every `observations` row sums to zero within an absolute 1e-12; `filter_observations` equals `contrast_matrix(27) @ row` with `rtol=1e-12, atol=1e-12`; `view_bounds[0] <= -1.0`; `view_bounds[1] >= max(1.0, true_params[:3].max())`; `0 <= weight_in_view <= 1 + 1e-12`.
- `test_model_archive_rows`: `model_log_evidence[0]` is all 0.0; `model_ess[0]` is all `n_particles`; `model_posterior[0]` is `np.array_equal` to `model_prior`; every posterior row sums to 1 within 1e-12; `model_ks` equals `[1, 2, 3]`; `meta["true_k"] == 2`.
- `test_model_posterior_follows_evidence`: `model_posterior` equals `_posterior_from_evidence(...)` with `rtol=1e-9, atol=1e-12`.
- `test_a_posterior_row_that_keeps_its_sum_is_caught`: a copy of the archive's `model_posterior` with its last row set to `[0.0, 0.99, 0.01]` no longer matches `_posterior_from_evidence(...)` at that tolerance, while still summing to 1.
- `test_an_underflowed_candidate_gives_zero`: `_posterior_from_evidence(np.array([0.5, 0.5]), np.array([[-1000.0, 0.0]]))` equals `[[0.0, 1.0]]` under `warnings.simplefilter("error")`.

The gate (each demo; the three slow ones marked):

- `test_gate_inputs`: against `fresh = record()`: the key sets are equal; every array has the same shape and dtype; `meta` is equal after dropping `created_with` from both; `clock_positions`, `true_params` and, for model comparison, `model_ks` and `model_prior` are `np.array_equal`; `true_rates`, `observations` and, for echolocation, `filter_observations` match with `rtol=1e-12, atol=1e-12`.
- `test_gate_summaries`: at each row in `_checkpoints(n_observations)`, for each parameter column, `abs(fresh_mean - archive_mean) <= bound` and `abs(fresh_sd - archive_sd) <= bound`, with the bounds below, where `sd = sqrt(diag(cov))`. Position columns are those of `x`, `y`, `z`; mass columns are those of `M`. For `demo_model_comparison`, at those rows: `model_posterior[row, 0] <= 1e-5` at row 1 and `<= 1e-20` at row 8 and the last row, in both the archive and the fresh run; and `abs(fresh − archive) <= 0.45` for `model_posterior[1, 1]`.

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

- `test_bounds_cover_every_filter_demo`: `set(BOUNDS) | {"demo_density"}` equals the six filter demos, so a demo cannot be left ungated by omission.

- [ ] **Step 2: Run them and confirm they fail**, the archive tests on missing files.

- [ ] **Step 3: Record the archives.** In the foreground with `timeout: 600000`:

Run: `uv run python scripts/record_replays.py --out assets/replays`

Then `ls -l assets/replays` and report the seven sizes; the spec's candidates totalled 1.8 MB, and a total above 3 MB is a finding to report before committing. Run `uv run python scripts/record_replays.py --out <scratch dir> --compare assets/replays` and confirm seven `reproduced` lines: a second run on the same machine must match the first.

- [ ] **Step 4: Run the tests** and confirm they pass: `uv run pytest tests/test_replays.py -q` and `uv run pytest tests/test_replays.py -q -m slow`. If a gate test fails against an archive recorded minutes earlier on the same machine, the recorder is not repeatable; stop and report, do not widen a bound. If an invariant fails, report the demo, row and value.

- [ ] **Step 5: Run the full gate**, confirm `git status --short` lists only the seven archives and the test file, then **commit and push** with message `Commit the seven replay archives with their invariants and gate`. The commit body lists each archive's size and the final estimate each demo's recording gives (read from `mean[-1]` and `sqrt(diag(cov[-1]))`, or `model_posterior[-1]`).

---

### Task 10: The report workflow

**Files:**
- Create: `.github/workflows/replay-report.yml`, `tests/test_replay_report_workflow.py`

**Interfaces:**
- Consumes: `scripts/record_replays.py --out ... --compare ... --report ...`, and its use of `REPLAY_HEAD_SHA`.

- [ ] **Step 1: Write the failing tests**, reading the workflow as text (no YAML library is a declared dependency).

- `test_every_step_may_fail`: split the `steps:` list into its items (lines beginning with six spaces and `- `); every item contains `continue-on-error: true`.
- `test_the_report_job_is_a_three_way_matrix`: the text contains `sample: [1, 2, 3]` and `fail-fast: false`.
- `test_the_report_never_touches_the_test_workflow`: `.github/workflows/ci.yml` does not mention `record_replays`.
- `test_triggers_are_filtered_to_what_can_change_a_replay`: both the `push` and the `pull_request` trigger list the paths `src/clocks/**`, `assets/replays/**`, `scripts/record_replays.py`, `uv.lock`, `pyproject.toml` and the workflow's own path.
- `test_drafts_are_skipped_and_jobs_time_out`: the text contains the draft condition and `timeout-minutes: 10`.

- [ ] **Step 2: Run them and confirm they fail.**

- [ ] **Step 3: Write the workflow.**

```yaml
name: Replay report

# Records every demo on the runner and reports whether each reproduces its
# committed archive value for value. Nothing here can fail a check: the tests
# that gate a merge are in ci.yml and do not depend on reproducing a
# trajectory. Three jobs sample three runner allocations from one tree.
on:
  push:
    branches: [main]
    paths: &replay_paths
      - "src/clocks/**"
      - "assets/replays/**"
      - "scripts/record_replays.py"
      - "uv.lock"
      - "pyproject.toml"
      - ".github/workflows/replay-report.yml"
  pull_request:
    branches: [main]
    types: [opened, synchronize, reopened, ready_for_review]
    paths: *replay_paths

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
        run: uv run python scripts/record_replays.py --out recorded --compare assets/replays --report replay-report.md
      - name: Upload the report
        continue-on-error: true
        uses: actions/upload-artifact@v4
        with:
          name: replay-report-${{ github.run_id }}-${{ matrix.sample }}
          path: replay-report.md
      - name: Show the report, or say it is missing
        continue-on-error: true
        if: always()
        run: |
          if [ -f replay-report.md ]; then
            cat replay-report.md >> "$GITHUB_STEP_SUMMARY"
          else
            echo "::warning title=Replay report::no report was written for sample ${{ matrix.sample }}"
          fi
```

GitHub's workflow parser may not accept YAML anchors. If `gh workflow view` or the first run rejects the file, write the six paths out under both triggers; the tests read the text and pass either way.

- [ ] **Step 4: Run the tests and confirm they pass.** Then run the full gate.

- [ ] **Step 5: Commit and push** with message `Report on every push whether the runner reproduces the replays, without gating`.

- [ ] **Step 6: Tell the coordinator** the workflow is in place. The coordinator opens the pull request, marks it ready, waits for the run, and copies the three reports and the macOS comparison (`uv run python scripts/record_replays.py --out <scratch> --compare assets/replays`) into its description.

---

### Task 11: Mutation checks

No files change. This task proves the new tests can fail. Work in the task's own worktree; set `PYTHONDONTWRITEBYTECODE=1`; delete every `__pycache__` before each mutant; confirm `git status --porcelain` is empty before and after each; apply each mutant and restore it in a `try`/`finally` (for an archive mutant, restore with `git checkout -- assets/replays`); run the whole suite including the slow tests (`uv run pytest -q -m ''`) for each mutant, in the foreground with a 600000 ms timeout. Archive mutants rewrite the file with `load_replay`, an edit to one array, and `save_replay`.

| Mutation | Tests that must fail, and no others |
|---|---|
| `thin` uses offset 0.5 in place of φ | `test_thin_picks[unequal]`, `test_thin_picks[uniform_7_3]` |
| `thin` does not divide the cumulative sum by its last entry | `test_thin_ignores_the_scale_of_the_weights` |
| `record_filter` zeroes the off-diagonal entries of `cov` | `test_moments_match_an_independent_calculation` |
| `record_filter` computes `ess[0]` as 1 / Σw² | `test_row_zero_holds_sentinels` |
| `record_filter` computes `predicted_mean` as `predict` at the mean particle | `test_predictions_are_moments_of_predictions` |
| `record_filter` draws one number from `pf.rng` after the last update | `test_recording_does_not_change_the_run` |
| `record_filter` drops its freshness check | `test_a_used_filter_is_refused` |
| `load_replay` skips the dtype check | every `test_wrong_dtype_is_rejected` case |
| `load_replay` accepts descending `model_ks` | `test_descending_model_ks_is_rejected` |
| `first_difference` skips `cloud` | `test_cloud_tolerance[outside]` |
| `close_view` uses bare `floor` and `ceil` | `test_close_view[rounding_up]`, `test_close_view[rounding_down]` |
| `demo_model_comparison.record()` writes the winning candidate's estimate under every `final_mean_k<k>` | `test_model_record_final_estimates_match_the_public_api` |
| **An early trajectory change.** In `demo_1d.record()`, before the call to `record_filter`: `real = pf.update`, then `pf.update = lambda o: (setattr(pf, "noise_std", 2 * NOISE_STD if pf.state.observations_seen < 3 else NOISE_STD), real(o))[1]` | `test_gate_summaries[demo_1d]`, `test_record_agrees_with_the_public_api[demo_1d]` |
| **A stale final result.** In `assets/replays/demo_1d.npz`, add 0.05 to `mean[-1, 0]` | `test_gate_summaries[demo_1d]`, `test_archive_dots_agree_with_summaries[demo_1d]` |
| **A stale early row.** In `assets/replays/demo_2d.npz`, multiply `cov[1]` by 4 | `test_gate_summaries[demo_2d]`, `test_archive_dots_agree_with_summaries[demo_2d]` |
| In `assets/replays/demo_model_comparison.npz`, set the last `model_posterior` row to `[0.0, 0.99, 0.01]` | `test_model_posterior_follows_evidence` |
| In `assets/replays/demo_echolocation_3d.npz`, add `1e-6` to every entry of `observations` | `test_echolocation_archive_channels`, `test_gate_inputs[demo_echolocation_3d]` |
| `demo_1d.NOISE_STD = 0.010` | `test_archive_matches_its_scenario[demo_1d]`, `test_gate_inputs[demo_1d]`, `test_gate_summaries[demo_1d]` |
| `demo_density.record()` draws each reading from the filter's generator between updates | `test_density_readings_come_from_their_own_generator`, `test_density_readings_do_not_depend_on_particles`, `test_density_record_agrees_with_an_unrecorded_filter`, `test_gate_inputs[demo_density]`, `test_gate_summaries[demo_density]` |

Three of these expectations rest on measurements made for the spec and not on a run of this code: the early change fails the first checkpoint and passes the last (spec, M15); the last `test_gate_summaries[demo_density]` assumes a different realisation of 60 readings moves a summary by more than its bound, which is likely and not certain; and the model-posterior mutant passes the gate because only the first checkpoint bounds the split.

- [ ] **Step 1: Run each mutation** and record, for each, the exact set of failing test IDs.
- [ ] **Step 2: Report** the table of results. Where the failing set differs from the expected set, say which tests were extra or missing; do not change tests or code to make the table match. A difference is a finding for the coordinator.
