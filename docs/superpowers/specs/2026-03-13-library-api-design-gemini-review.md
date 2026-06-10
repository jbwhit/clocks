# Library API Design Review (Gemini Feedback)

This document contains Gemini's review and proposed refinements to the initial API design.

## Key Recommendations

### 1. Unified Functional Entry Points
Instead of separate observation/simulation functions, we recommend:
- `clocks.api.infer(observations, config)`: Single entry point for all inference.
- `clocks.api.simulate(config)`: Standalone synthetic data generation.
- `clocks.api.simulate_and_infer(...)`: Convenience wrapper for synthetic benchmarks.

### 2. Pydantic v2 for Config & Results
Shift from standard dataclasses to Pydantic v2.
- **Why:** Robust validation (e.g., ensuring `n_particles > 0`), easy JSON serialization for downstream tools, and better performance in validation-heavy codebases.
- **Structure:** Use nested models (e.g., `InferenceConfig` contains `NoiseConfig` and `PriorConfig`).

### 3. Integrated Visualization
Add `.plot()` methods directly to `InferenceResult` and `SimulationResult`.
- **Why:** Dramatic improvement for notebook users. `result.plot()` provides instant feedback without requiring the user to import `clocks.viz` and pass objects manually.

### 4. Explicit Ground Truth
`SimulationResult` must explicitly include the "Ground Truth" state (`MassConfig`) to enable easy accuracy metrics.

## Proposed API Refinement

### Data Models (Pydantic)

```python
class InferenceConfig(BaseModel):
    noise: NoiseConfig
    prior: PriorConfig
    n_particles: int = 1000
    n_masses: int | list[int]  # List triggers model comparison

class InferenceResult(BaseModel):
    posterior_mean: NDArray
    posterior_std: NDArray
    history: list[Snapshot]
    model_probs: NDArray | None = None

    def plot(self):
        """Integrated visualization via clocks.viz"""
        ...
```

### Core Functions

```python
def infer(observations: list[Observation], config: InferenceConfig) -> InferenceResult:
    """Run unified inference orchestration."""
    ...

def simulate(config: SimulationConfig) -> SimulationResult:
    """Generate observations and ground truth."""
    ...
```

## Testing & Validation Strategy

- **Pydantic Validation:** Test that invalid configs raise clear, actionable errors.
- **Notebook Compatibility:** Ensure `.plot()` returns Matplotlib objects correctly.
- **Serialization:** Verify `result.model_dump_json()` preserves all data for external storage.

## Feedback Summary
The original design is a strong foundation. These refinements focus on making the library feel "modern" (Pydantic) and "batteries-included" (Integrated Plotting).
