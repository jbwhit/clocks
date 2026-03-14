# Library API Design

## Goal

Make `clocks` easier to embed in other Python projects by adding stable, end-to-end public entry points for simulation and inference. A secondary goal is to make repeatable script and notebook workflows easier without forcing callers to understand the current demo-oriented orchestration.

## Scope

This design covers:

- public batch APIs for single-mass, fixed-K multi-mass, and model-comparison workflows
- synthetic-data generation as a first-class public workflow
- typed public config and result objects
- serialization-friendly result access for notebooks and downstream tools

This design does not cover:

- a stateful incremental/session API
- slider-driven interactive UI work
- major rewrites of the current physics or particle-filter internals

## Current Problem

The repo already has capable lower-level building blocks:

- forward simulation in `physics.py`
- noise modeling in `noise.py`
- particle filtering in `inference.py`
- demo orchestration in `scripts/` and `_cli.py`

What is missing is a stable public layer that answers common library-use questions:

- How do I run end-to-end inference without reproducing demo code?
- How do I do model comparison from Python code?
- How do I simulate data and infer against it with one supported interface?
- What output shape can downstream code rely on?

Right now, the practical path for users is to inspect demos and assemble the workflow themselves. That is acceptable for examples, but weak for embedding.

## Recommended Approach

Add a thin public API layer with a small, unified workflow surface:

- `infer(...)`
- `simulate(...)`
- `simulate_and_infer(...)` as an optional convenience wrapper

`infer(...)` should be the main entry point for all inference against already
materialized observations. `simulate(...)` should be the standalone synthetic
data generator. `simulate_and_infer(...)` should exist only to simplify common
benchmark and notebook workflows.

These entry points should share the same underlying config normalization,
orchestration, and result shaping. This keeps the public story simple while
avoiding duplicate implementations.

The first release should remain batch-oriented. It should be designed so a future stateful `InferenceSession` can reuse the same configs and result types, but that session API should not be introduced yet.

## Alternatives Considered

### 1. Unified batch API plus shared public types

Recommended.

Pros:

- simplest path for library adoption
- supports scripts and notebooks immediately
- minimizes disruption to current internals
- leaves a clean path to future incremental APIs

Cons:

- does not directly support streaming updates
- some advanced users will still want lower-level hooks

### 2. Stateful session API first

Pros:

- aligns more directly with future interactive use
- naturally supports incremental observations

Cons:

- heavier API design burden up front
- forces decisions around mutable state, lifecycle, and reset semantics early
- risks overfitting to future UI ideas before the basic library contract is stable

### 3. Experiment-spec or config-driven façade only

Pros:

- good for reproducible studies
- easy to save and rerun experiments

Cons:

- weaker fit for embedding in another Python codebase
- pushes the public API toward config plumbing instead of usable functions

## Public API Shape

The public layer should expose one primary inference entry point and one
primary simulation entry point.

### Inference

For callers who already have clock observations:

- `infer(...)`

This should support:

- single-mass inference
- fixed-K multi-mass inference
- model comparison across candidate values of `K`

### Simulation

For callers doing synthetic experiments:

- `simulate(...)`

This should:

- accept simulation inputs describing clocks, true mass configuration, noise, and observation count
- generate observations using the existing forward model
- return a `SimulationResult` that includes both the generated observations and
  the explicit ground-truth mass configuration

### Convenience wrapper

For common end-to-end synthetic benchmarks:

- `simulate_and_infer(...)`

This should:

- call `simulate(...)`
- run the same inference/model-comparison workflows as the observation-driven path
- return the same inference result family, with simulation output attached or
  otherwise easily accessible

## Proposed Module Layout

One reasonable public layout:

```text
src/clocks/
    api.py          Public end-to-end entry points
    config.py       Public config dataclasses
    results.py      Public result dataclasses and exporters
```

The package `__init__.py` can re-export the stable public surface, while current low-level modules remain available for advanced use.

## Public Config Types

The public API should use explicit dataclasses rather than loosely structured dictionaries.

Candidate public config types:

- `InferenceConfig`
- `SimulationConfig`
- `ModelComparisonConfig`
- `PriorConfig`
- `NoiseConfig`

These should cover:

- clock geometry and dimensionality
- particle count
- resampling and jitter settings
- observation-noise parameters
- prior bounds or prior configuration
- number of masses for fixed-K inference
- candidate models for model comparison
- simulation parameters such as true mass configuration and observation length

The config layer should normalize user-friendly inputs into the lower-level structures already expected by the existing internals.

## Public Result Types

Results should be stable, typed, and easy to inspect.

Base expectations:

- posterior summaries
- uncertainty summaries
- time history of estimates
- model-specific or mass-specific metadata
- optional access to raw arrays for advanced users

Candidate result types:

- `InferenceResult`
- `MultiMassInferenceResult`
- `ModelComparisonResult`
- `SimulationResult`

For model comparison specifically, results should include:

- per-model posterior probability history
- evidence or normalized score history
- best-model summary
- per-model nested results where practical

Results should also provide convenience exporters such as:

- `to_dict()`
- `to_numpy()` or array-oriented accessors

The primary API should not force callers to depend on internal particle-state layouts.

Plotting should not be a required method on result objects in the first version.
If notebook ergonomics need improvement, add plotting helpers that accept the
public result types, likely in `clocks.viz` or a small public plotting module.
That keeps result objects focused on stable data contracts instead of coupling
them immediately to Matplotlib-facing APIs.

## Execution Flow

All public workflows should converge on one orchestration path:

1. validate and normalize config objects
2. construct observations via `simulate(...)` or accept them via `infer(...)`
3. dispatch to fixed-K inference or model comparison
4. collect histories and summaries from the underlying filter runs
5. map internal outputs into stable public result objects

This orchestration should stay thin. The design goal is not to rewrite `physics.py` or `inference.py`, but to make them easier to consume safely.

## Compatibility With Future Incremental APIs

The batch API should be designed as the first layer of a larger interface, not a dead end.

To preserve that path:

- use public config objects that can later initialize a session object unchanged
- use result types that can represent either final batch outputs or snapshots of an evolving session
- avoid exposing mutable internal state as the primary contract
- keep orchestration logic separate from CLI/demo code

A future `InferenceSession` could then reuse:

- config validation
- result shaping
- model-comparison bookkeeping

without breaking callers of the batch API.

## Testing Strategy

Tests should validate public behavior, not internal implementation details.

Required coverage:

- `infer(...)` returns stable result objects for single-mass, multi-mass, and model-comparison workflows
- `simulate(...)` generates observations and returns explicit ground truth alongside them
- `simulate_and_infer(...)` returns the same inference result family while preserving access to simulation outputs
- controlled synthetic cases recover the correct model in model-comparison tests
- result exporters produce serialization-friendly structures
- invalid config combinations fail with clear public-facing errors

Testing should continue to treat lower-level modules independently, but new tests should center on the new public API layer.

## Migration Strategy

Initial rollout should be additive:

- keep current CLI scripts and demos working
- refactor demo orchestration to use the new public API where it reduces duplication
- document the new library entry points in the README with short embedding examples
- keep plotting integration additive by adapting `clocks.viz` to public result objects rather than embedding plotting methods into the first release of those objects

This avoids breaking current users while making the package usable as a library.

## Success Criteria

The design is successful if a caller can:

- import a small, documented public API
- run multi-mass inference or model comparison without copying demo code
- choose either real observations or synthetic simulation from supported entry points
- receive stable result objects suitable for Python applications and notebooks
- avoid depending on private/internal module details
