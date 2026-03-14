# Library API Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a stable public library API for simulation and end-to-end inference, including multi-mass and model-comparison workflows, without requiring users to copy demo code.

**Architecture:** Introduce a thin orchestration layer in `src/clocks/api.py`, supported by explicit public config/result dataclasses in `src/clocks/config.py` and `src/clocks/results.py`. Keep existing physics and particle-filter internals intact, adapt demo orchestration to call the new API, and expose the new public surface from `src/clocks/__init__.py`.

**Tech Stack:** Python 3.12+, NumPy, SciPy, Matplotlib, pytest, Ruff, dataclasses

---

## File Structure

Create or modify these files:

- Create: `src/clocks/config.py`
- Create: `src/clocks/results.py`
- Create: `src/clocks/api.py`
- Modify: `src/clocks/__init__.py`
- Modify: `scripts/demo_multi_mass.py`
- Modify: `scripts/demo_model_comparison.py`
- Modify: `README.md`
- Create: `tests/test_api.py`

Responsibilities:

- `src/clocks/config.py`: Public config dataclasses and validation helpers for inference and simulation.
- `src/clocks/results.py`: Public result dataclasses plus serialization helpers like `to_dict()`.
- `src/clocks/api.py`: Public workflow functions `infer`, `simulate`, and `simulate_and_infer`.
- `src/clocks/__init__.py`: Re-export the supported public API.
- `scripts/demo_multi_mass.py`: Consume the new public API instead of rebuilding orchestration inline.
- `scripts/demo_model_comparison.py`: Consume the new public API for simulation/inference flow.
- `README.md`: Document the new library entry points with short examples.
- `tests/test_api.py`: Public API regression tests and validation tests.

## Chunk 1: Public Types And Simulation Result

### Task 1: Add failing tests for public config and simulation result shapes

**Files:**
- Create: `tests/test_api.py`
- Create: `src/clocks/config.py`
- Create: `src/clocks/results.py`

- [ ] **Step 1: Write the failing tests**

```python
from clocks.config import InferenceConfig, NoiseConfig, SimulationConfig
from clocks.results import SimulationResult


def test_simulation_result_exposes_ground_truth() -> None:
    ...


def test_inference_config_rejects_nonpositive_particles() -> None:
    InferenceConfig(n_particles=0, ...)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api.py::test_simulation_result_exposes_ground_truth tests/test_api.py::test_inference_config_rejects_nonpositive_particles -v`
Expected: FAIL with import errors or missing symbols

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass(frozen=True)
class NoiseConfig:
    observation_std: float


@dataclass(frozen=True)
class InferenceConfig:
    n_particles: int
    noise: NoiseConfig
    ...

    def __post_init__(self) -> None:
        if self.n_particles <= 0:
            raise ValueError("n_particles must be > 0")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_api.py::test_simulation_result_exposes_ground_truth tests/test_api.py::test_inference_config_rejects_nonpositive_particles -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_api.py src/clocks/config.py src/clocks/results.py
git commit -m "Add public config and result dataclasses"
```

### Task 2: Add serialization tests for result objects

**Files:**
- Modify: `tests/test_api.py`
- Modify: `src/clocks/results.py`

- [ ] **Step 1: Write the failing test**

```python
def test_simulation_result_to_dict_serializes_arrays() -> None:
    payload = result.to_dict()
    assert payload["ground_truth"]["masses"] == [0.6, 0.4]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api.py::test_simulation_result_to_dict_serializes_arrays -v`
Expected: FAIL because `to_dict()` is missing or returns raw NumPy arrays

- [ ] **Step 3: Write minimal implementation**

```python
def to_dict(self) -> dict[str, object]:
    return {
        "ground_truth": {...},
        "observations": [...],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_api.py::test_simulation_result_to_dict_serializes_arrays -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_api.py src/clocks/results.py
git commit -m "Add result serialization helpers"
```

## Chunk 2: Public Simulation And Inference Entry Points

### Task 3: Add failing tests for `simulate(...)`

**Files:**
- Modify: `tests/test_api.py`
- Create: `src/clocks/api.py`

- [ ] **Step 1: Write the failing test**

```python
from clocks.api import simulate


def test_simulate_returns_observations_and_ground_truth() -> None:
    result = simulate(sim_config)
    assert len(result.observations) == 5
    assert result.ground_truth.masses.shape == (2,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api.py::test_simulate_returns_observations_and_ground_truth -v`
Expected: FAIL because `clocks.api` or `simulate` does not exist

- [ ] **Step 3: Write minimal implementation**

```python
def simulate(config: SimulationConfig) -> SimulationResult:
    true_rates = clock_rates(config.ground_truth, config.clock_array)
    observations = [...]
    return SimulationResult(...)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_api.py::test_simulate_returns_observations_and_ground_truth -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_api.py src/clocks/api.py
git commit -m "Add public simulation API"
```

### Task 4: Add failing tests for fixed-K multi-mass `infer(...)`

**Files:**
- Modify: `tests/test_api.py`
- Modify: `src/clocks/api.py`
- Modify: `src/clocks/results.py`

- [ ] **Step 1: Write the failing test**

```python
from clocks.api import infer


def test_infer_multi_mass_returns_summary_history() -> None:
    result = infer(observations, config)
    assert result.posterior_mean.shape == (4,)
    assert len(result.history) == len(observations)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api.py::test_infer_multi_mass_returns_summary_history -v`
Expected: FAIL because `infer` does not exist or result shape is incomplete

- [ ] **Step 3: Write minimal implementation**

```python
def infer(observations: list[Observation], config: InferenceConfig) -> InferenceResult:
    pf = _build_particle_filter(config)
    for obs in observations:
        pf.update(obs)
    return _result_from_particle_filter(pf)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_api.py::test_infer_multi_mass_returns_summary_history -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_api.py src/clocks/api.py src/clocks/results.py
git commit -m "Add public infer API for fixed-K workflows"
```

### Task 5: Add failing tests for model comparison in `infer(...)`

**Files:**
- Modify: `tests/test_api.py`
- Modify: `src/clocks/api.py`
- Modify: `src/clocks/results.py`

- [ ] **Step 1: Write the failing test**

```python
def test_infer_model_comparison_returns_model_probabilities() -> None:
    result = infer(observations, config_with_candidate_models)
    assert set(result.posterior_by_model) == {1, 2, 3}
    assert result.best_model == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api.py::test_infer_model_comparison_returns_model_probabilities -v`
Expected: FAIL because model-comparison dispatch or result shaping is missing

- [ ] **Step 3: Write minimal implementation**

```python
if isinstance(config.n_masses, tuple):
    mc = _build_model_comparison(config)
    ...
    return ModelComparisonInferenceResult(...)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_api.py::test_infer_model_comparison_returns_model_probabilities -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_api.py src/clocks/api.py src/clocks/results.py
git commit -m "Add model comparison support to public infer API"
```

### Task 6: Add failing tests for `simulate_and_infer(...)`

**Files:**
- Modify: `tests/test_api.py`
- Modify: `src/clocks/api.py`
- Modify: `src/clocks/results.py`

- [ ] **Step 1: Write the failing test**

```python
from clocks.api import simulate_and_infer


def test_simulate_and_infer_preserves_simulation_output() -> None:
    result = simulate_and_infer(sim_config, inference_config)
    assert result.simulation.ground_truth.masses.shape == (2,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api.py::test_simulate_and_infer_preserves_simulation_output -v`
Expected: FAIL because wrapper or attached simulation payload is missing

- [ ] **Step 3: Write minimal implementation**

```python
def simulate_and_infer(...):
    simulation = simulate(sim_config)
    inference = infer(simulation.observations, inference_config)
    return inference.with_simulation(simulation)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_api.py::test_simulate_and_infer_preserves_simulation_output -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_api.py src/clocks/api.py src/clocks/results.py
git commit -m "Add simulate and infer convenience wrapper"
```

## Chunk 3: Public Exports, Demos, And Docs

### Task 7: Add failing tests for top-level exports

**Files:**
- Modify: `tests/test_api.py`
- Modify: `src/clocks/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
def test_public_api_is_exported_from_package() -> None:
    import clocks

    assert clocks.infer is not None
    assert clocks.SimulationConfig is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api.py::test_public_api_is_exported_from_package -v`
Expected: FAIL because the package does not re-export the new API

- [ ] **Step 3: Write minimal implementation**

```python
from clocks.api import infer, simulate, simulate_and_infer
from clocks.config import InferenceConfig, NoiseConfig, SimulationConfig
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_api.py::test_public_api_is_exported_from_package -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_api.py src/clocks/__init__.py
git commit -m "Export public library API from package"
```

### Task 8: Refactor demos to use the public API

**Files:**
- Modify: `scripts/demo_multi_mass.py`
- Modify: `scripts/demo_model_comparison.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write the failing smoke tests**

```python
def test_demo_configs_can_run_through_public_api() -> None:
    result = infer(observations, config)
    assert result.posterior_mean.shape[0] == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api.py::test_demo_configs_can_run_through_public_api -v`
Expected: FAIL because the current helpers do not cover demo-shaped configuration

- [ ] **Step 3: Write minimal implementation**

```python
# demo_multi_mass.py
simulation = simulate(sim_config)
result = infer(simulation.observations, inference_config)
```

- [ ] **Step 4: Run targeted tests to verify they pass**

Run: `uv run pytest tests/test_api.py::test_demo_configs_can_run_through_public_api -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_api.py scripts/demo_multi_mass.py scripts/demo_model_comparison.py
git commit -m "Refactor demos to use public API"
```

### Task 9: Document the public API in the README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write the documentation update**

```python
from clocks import infer, simulate, InferenceConfig, SimulationConfig
```

- [ ] **Step 2: Verify examples remain aligned with real symbols**

Run: `uv run pytest tests/test_api.py::test_public_api_is_exported_from_package -v`
Expected: PASS, confirming README symbols exist

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Document public library API"
```

## Chunk 4: Final Verification

### Task 10: Run the full verification suite

**Files:**
- Test: `tests/test_api.py`
- Test: `tests/test_inference.py`
- Test: `tests/test_physics.py`

- [ ] **Step 1: Run targeted API tests**

Run: `uv run pytest tests/test_api.py -v`
Expected: PASS

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest`
Expected: PASS

- [ ] **Step 3: Run lint**

Run: `uv run ruff check src/ tests/ scripts/`
Expected: PASS

- [ ] **Step 4: Commit any final cleanup**

```bash
git add src/clocks tests scripts README.md
git commit -m "Finalize public library API"
```
