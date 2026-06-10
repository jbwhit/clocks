# Deep Clean Implementation Plan (Spec Phases 1–3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land all correctness fixes, consolidate the animation code, port the demos to the public API, and finish repo hygiene, per `docs/superpowers/specs/2026-06-09-deep-clean-and-website-design.md` Phases 1–3.

**Architecture:** Small sequential commits, TDD where behavior changes. The viz consolidation extracts one private dashboard-animation driver and splits `viz.py` into private `_panels.py` (static plotting primitives) + `_animate.py` (animators) behind an unchanged `clocks.viz` facade. Public API gains exactly one name: `build_particle_filter`.

**Tech Stack:** Python 3.12, numpy/scipy/matplotlib, uv, pytest, ruff. Run everything with `uv run`. The repo pre-commit hook runs ruff format/lint automatically.

**Verification gate for every task:** `uv run pytest -q` (all pass; 102 tests before this plan, more after) and `uv run ruff check src/ tests/ scripts/`.

---

### Task 1: Fix log-evidence accumulation

The per-step marginal-likelihood estimate is `log(sum(prev_weights * likelihoods))`. `ParticleFilter.update` already starts `log_weights` from the normalized previous weights, so the current extra `- np.log(self.n_particles)` biases `log_evidence` by `-log(N)` per observation.

**Files:**
- Modify: `src/clocks/inference.py:152`
- Test: `tests/test_inference.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_inference.py` inside `class TestParticleFilter` (match the file's existing import style; it already imports `ParticleFilter`, `ClockArray`, `MassConfig`, `Observation`, `clock_rates`, `add_clock_noise` — add `from scipy.special import logsumexp` and `from clocks.noise import log_likelihood_gaussian` if not present):

```python
def test_log_evidence_matches_direct_computation(self) -> None:
    """Each update's log-evidence increment is log(sum(prev_w * L))."""
    clock_array = ClockArray(
        positions=np.array([[-5.0], [0.0], [5.0]]), track_offset=1.0
    )
    true_rates = clock_rates(
        MassConfig(positions=np.array([[2.0]]), masses=np.array([0.5])),
        clock_array,
    )

    def prior_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        return np.column_stack(
            [rng.uniform(-8, 8, n), rng.uniform(0.1, 2.0, n)]
        )

    def forward_model(params: np.ndarray) -> np.ndarray:
        return clock_rates(
            MassConfig(
                positions=np.array([[params[0]]]),
                masses=np.array([params[1]]),
            ),
            clock_array,
        )

    # resample_threshold=0 → never resample → previous weights stay
    # NONUNIFORM after the first update, which is the case the bias
    # claim is about.
    pf = ParticleFilter(
        n_particles=50,
        prior_sampler=prior_sampler,
        forward_model=forward_model,
        noise_std=0.005,
        resample_threshold=0.0,
        rng=np.random.default_rng(1),
    )

    obs_rng = np.random.default_rng(0)
    expected = 0.0
    for t in range(3):
        obs = Observation(
            rates=add_clock_noise(true_rates, 0.005, obs_rng), time=float(t)
        )
        prev_weights = pf.state.weights.copy()
        log_w = np.log(prev_weights) + np.array(
            [
                log_likelihood_gaussian(obs.rates, forward_model(p), 0.005)
                for p in pf.state.particles
            ]
        )
        expected += logsumexp(log_w)
        pf.update(obs)
        assert pf.log_evidence == pytest.approx(expected, rel=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_inference.py::TestParticleFilter::test_log_evidence_matches_direct_computation -v`
Expected: FAIL — `pf.log_evidence` is lower than `expected` by `(t+1) * log(50)`.

- [ ] **Step 3: Apply the fix**

In `src/clocks/inference.py`, change:

```python
self.log_evidence += max_lw + np.log(weights.sum()) - np.log(self.n_particles)
```

to:

```python
self.log_evidence += max_lw + np.log(weights.sum())
```

- [ ] **Step 4: Run the new test, then the full suite**

Run: `uv run pytest tests/test_inference.py::TestParticleFilter::test_log_evidence_matches_direct_computation -v` → PASS.
Run: `uv run pytest -q` → all pass. The model-comparison regression required by the spec already exists: `tests/test_api.py::test_infer_model_comparison_returns_model_probabilities` asserts `result.best_model == 2`; it must stay green (the removed term was a constant shared across K). If any existing test pinned an absolute `log_evidence` value, update that pinned value to the new unbiased number and note it in the commit message.

- [ ] **Step 5: Commit**

```bash
git add src/clocks/inference.py tests/test_inference.py
git commit -m "Fix log-evidence bias: drop spurious -log(N) per update"
```

---

### Task 2: Document jitter_std semantics

**Files:**
- Modify: `src/clocks/inference.py` (ParticleFilter docstring)
- Modify: `src/clocks/config.py` (InferenceConfig docstring)

- [ ] **Step 1: Update the ParticleFilter docstring**

In `src/clocks/inference.py`, replace the parameter lines:

```python
    jitter_std : Std of Gaussian jitter applied after resampling.
```

with:

```python
    jitter_std : Scale of the post-resampling jitter. With ``jitter="fixed"``
        this is an absolute standard deviation applied isotropically; with
        ``jitter="covariance"`` it scales the Cholesky factor of the weighted
        empirical covariance, so 0.02 means "2% of the cloud's own spread
        along its correlation structure".
```

- [ ] **Step 2: Update the InferenceConfig docstring**

In `src/clocks/config.py`, replace:

```python
@dataclass(frozen=True)
class InferenceConfig:
    """Top-level config for end-to-end inference."""
```

with:

```python
@dataclass(frozen=True)
class InferenceConfig:
    """Top-level config for end-to-end inference.

    ``jitter_std`` scales the post-resampling jitter: an absolute standard
    deviation when ``jitter="fixed"``, or a fraction of the particle cloud's
    weighted covariance when ``jitter="covariance"``.
    """
```

- [ ] **Step 3: Verify and commit**

Run: `uv run pytest -q` → all pass.

```bash
git add src/clocks/inference.py src/clocks/config.py
git commit -m "Document jitter_std semantics for fixed vs covariance modes"
```

---

### Task 3: Fail fast on empty observations

**Files:**
- Modify: `src/clocks/api.py` (top of `infer`)
- Test: `tests/test_api.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_api.py` (it already has `_make_inference_config(...)` helpers — reuse them):

```python
def test_infer_rejects_empty_observations() -> None:
    for n_masses in (1, (1, 2)):
        config = _make_inference_config(n_masses=n_masses)
        with pytest.raises(ValueError, match="observations must not be empty"):
            infer([], config)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api.py::test_infer_rejects_empty_observations -v`
Expected: FAIL — fixed-K mode returns a prior-only result instead of raising; model-comparison mode raises `UnboundLocalError`.

- [ ] **Step 3: Add the guard before mode dispatch**

In `src/clocks/api.py`, at the top of `infer`:

```python
def infer(
    observations: list[Observation], config: InferenceConfig
) -> InferenceResult | ModelComparisonInferenceResult:
    """Run inference against a list of observations."""
    if not observations:
        raise ValueError("observations must not be empty")
    if isinstance(config.n_masses, tuple):
        return _infer_model_comparison(observations, config)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_api.py -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add src/clocks/api.py tests/test_api.py
git commit -m "Reject empty observation lists in infer()"
```

---

### Task 4: Raise instead of NaN when every particle is invalid

**Files:**
- Modify: `src/clocks/inference.py` (`ParticleFilter.update`)
- Test: `tests/test_inference.py`

- [ ] **Step 1: Write the failing test**

Add inside `class TestParticleFilter` (reuse the prior-sampler/forward-model pattern from Task 1's test):

```python
def test_update_raises_when_all_particles_have_zero_weight(self) -> None:
    clock_array = ClockArray(positions=np.array([[0.0]]), track_offset=1.0)

    pf = ParticleFilter(
        n_particles=10,
        prior_sampler=lambda rng, n: rng.uniform(-1, 1, (n, 1)),
        forward_model=lambda params: np.array([1.0]),
        noise_std=0.01,
        log_prior=lambda particles: np.full(particles.shape[0], -np.inf),
        rng=np.random.default_rng(0),
    )
    obs = Observation(rates=np.array([1.0]), time=0.0)

    with pytest.raises(RuntimeError, match="zero weight"):
        pf.update(obs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_inference.py::TestParticleFilter::test_update_raises_when_all_particles_have_zero_weight -v`
Expected: FAIL — no exception; NaN weights propagate silently.

- [ ] **Step 3: Add the guard**

In `ParticleFilter.update`, immediately after the log-prior addition and before the normalization block:

```python
        # Normalize weights (log-sum-exp for numerical stability)
        max_lw = np.max(log_weights)
        if not np.isfinite(max_lw):
            raise RuntimeError(
                "All particles have zero weight (every log-weight is -inf); "
                "the prior or forward model is inconsistent with the "
                "observations"
            )
        log_weights -= max_lw
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add src/clocks/inference.py tests/test_inference.py
git commit -m "Raise RuntimeError when all particle weights collapse to zero"
```

---

### Task 5: Fix animate_model_comparison for sparse k_values

`animate_model_comparison` builds `k_values = list(range(1, k_max + 1))`, which KeyErrors when `ModelComparison` was constructed with non-contiguous `k_values` such as `(2, 3)`.

**Files:**
- Modify: `src/clocks/viz.py:794-795`
- Test: `tests/test_viz.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_viz.py` (match its existing import style; it already exercises animators with tiny inputs and `tmp_path`):

```python
def test_animate_model_comparison_with_sparse_k_values(tmp_path) -> None:
    clock_array = ClockArray(
        positions=np.array([[-5.0], [0.0], [5.0]]), track_offset=1.0
    )
    mass_config = MassConfig(positions=np.array([[2.0]]), masses=np.array([0.5]))
    true_rates = clock_rates(mass_config, clock_array)
    rng = np.random.default_rng(0)
    observations = [
        Observation(rates=add_clock_noise(true_rates, 0.005, rng), time=float(t))
        for t in range(2)
    ]
    mc = ModelComparison(
        clock_array=clock_array,
        noise_std=0.005,
        k_values=(2, 3),
        n_particles=20,
        rng=np.random.default_rng(1),
    )
    output = tmp_path / "comparison.gif"

    animate_model_comparison(clock_array, mass_config, observations, mc, output)

    assert output.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_viz.py::test_animate_model_comparison_with_sparse_k_values -v`
Expected: FAIL with `KeyError: 1` (posterior dict only has keys {2, 3}).

- [ ] **Step 3: Apply the fix**

In `src/clocks/viz.py`, in `animate_model_comparison`, replace:

```python
    k_max = model_comparison.k_max
    k_values = list(range(1, k_max + 1))
```

with:

```python
    k_values = sorted(model_comparison.filters)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_viz.py -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add src/clocks/viz.py tests/test_viz.py
git commit -m "Animate model comparison over actual k_values, not 1..k_max"
```

---

### Task 6: Extract the dashboard animation driver

Behavior-preserving refactor: one driver owns the drive-filter → render → `FuncAnimation` → save lifecycle; the four fixed-K animators become panel wiring. `animate_model_comparison` keeps its own loop (1×2 layout, drives a `ModelComparison`, not a `ParticleFilter`) — the spec explicitly allows this.

**Files:**
- Modify: `src/clocks/viz.py`
- Test: existing `tests/test_viz.py` (no new tests — covered behavior must not change)

- [ ] **Step 1: Add the driver and the shared 2D-rates renderer factory**

Add to `src/clocks/viz.py` (above `animate_inference`):

```python
def _animate_filter_dashboard(
    fig: Figure,
    axes: dict[str, Axes],
    pf: ParticleFilter,
    observations: list[Observation],
    output_path: Path,
    fps: int,
    render_particles: Callable[[Axes, ParticleState], None],
    render_rates: Callable[[Axes, Observation, int], None],
    render_history: Callable[
        [Axes, NDArray[np.floating], NDArray[np.floating]], None
    ],
) -> None:
    """Drive a particle filter through observations on the 2x2 dashboard.

    Render callables own their panel completely, including ``ax.clear()``
    and any artist lifecycle (e.g. colorbars).
    """
    means: list[NDArray[np.floating]] = []
    stds: list[NDArray[np.floating]] = []

    def update(frame: int) -> None:
        obs = observations[frame]
        pf.update(obs)
        est = pf.estimate()
        means.append(est["mean"])
        stds.append(est["std"])
        render_particles(axes["particles"], pf.state)
        render_rates(axes["rates"], obs, frame)
        render_history(axes["history"], np.array(means), np.array(stds))

    anim = animation.FuncAnimation(
        fig, update, frames=len(observations), repeat=False
    )
    _save_animation(anim, fig, output_path, fps)


def _make_rates_renderer_2d(
    clock_array: ClockArray,
    true_rates: NDArray[np.floating],
    xylim: tuple[float, float],
) -> Callable[[Axes, Observation, int], None]:
    """Per-frame renderer for the 2D observed-rates panel (owns its colorbar)."""
    cbar_state: list = []

    def render(ax: Axes, obs: Observation, frame: int) -> None:
        if cbar_state:
            cbar_state[0].remove()
            cbar_state.clear()
        ax.clear()
        cx = clock_array.positions[:, 0]
        cy = clock_array.positions[:, 1]
        sc = ax.scatter(
            cx,
            cy,
            c=obs.rates,
            cmap="coolwarm",
            s=120,
            marker="s",
            vmin=min(true_rates) - 0.002,
            vmax=max(true_rates) + 0.002,
            zorder=5,
            edgecolors="black",
            linewidths=0.5,
        )
        for x, y, r in zip(cx, cy, obs.rates):
            ax.annotate(
                f"{r:.4f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, -14),
                ha="center",
                fontsize=7,
            )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.set_xlim(xylim)
        ax.set_ylim(xylim)
        ax.set_title(f"Observed Rates (t={frame + 1})")
        cbar_state.append(plt.colorbar(sc, ax=ax, label="Rate", shrink=0.8))

    return render
```

Add `Callable` to the existing `collections.abc` import (or add `from collections.abc import Callable` if absent) and confirm `Figure` is imported (it is, for `_save_animation`).

- [ ] **Step 2: Rewrite `animate_inference` against the driver**

Replace the body of `animate_inference` (keep the signature and docstring exactly as today):

```python
    true_params = np.array(
        [mass_config.positions[0, 0], mass_config.masses[0]]
    )
    true_rates = clock_rates(mass_config, clock_array)

    fig, axes = create_inference_dashboard()
    plot_clock_setup(axes["setup"], clock_array, mass_config)
    axes["setup"].set_xlim(xlim)

    def render_particles(ax: Axes, state: ParticleState) -> None:
        ax.clear()
        plot_particle_cloud(ax, state, true_params)
        ax.set_xlim(xlim)
        ax.set_ylim(mlim)

    def render_rates(ax: Axes, obs: Observation, frame: int) -> None:
        ax.clear()
        plot_clock_rates(ax, true_rates, clock_array, label="True", color="lightcoral")
        plot_clock_rates(
            ax, obs.rates, clock_array, label="Observed", color="steelblue"
        )

    def render_history(
        ax: Axes, means: NDArray[np.floating], stds: NDArray[np.floating]
    ) -> None:
        ax.clear()
        steps = np.arange(1, len(means) + 1)
        _plot_convergence(
            ax, steps, means, stds, true_params, ["tab:blue", "tab:orange"], ["x", "M"]
        )

    _animate_filter_dashboard(
        fig, axes, pf, observations, output_path, fps,
        render_particles, render_rates, render_history,
    )
```

- [ ] **Step 3: Run the viz tests**

Run: `uv run pytest tests/test_viz.py -q` → all pass.

- [ ] **Step 4: Rewrite `animate_inference_2d` against the driver**

Replace its body (signature and docstring unchanged):

```python
    true_params = np.array(
        [
            mass_config.positions[0, 0],
            mass_config.positions[0, 1],
            mass_config.masses[0],
        ]
    )
    true_rates = clock_rates(mass_config, clock_array)

    fig, axes = create_inference_dashboard(figsize=(13, 10))
    plot_clock_setup_2d(axes["setup"], clock_array, mass_config)
    axes["setup"].set_xlim(xylim)
    axes["setup"].set_ylim(xylim)

    def render_particles(ax: Axes, state: ParticleState) -> None:
        ax.clear()
        plot_particle_cloud_2d(ax, state, true_params[:2])
        ax.set_xlim(xylim)
        ax.set_ylim(xylim)
        ax.set_aspect("equal")

    render_rates = _make_rates_renderer_2d(clock_array, true_rates, xylim)

    def render_history(
        ax: Axes, means: NDArray[np.floating], stds: NDArray[np.floating]
    ) -> None:
        ax.clear()
        steps = np.arange(1, len(means) + 1)
        _plot_convergence(
            ax,
            steps,
            means,
            stds,
            true_params,
            ["tab:blue", "tab:green", "tab:orange"],
            ["x", "y", "M"],
        )

    _animate_filter_dashboard(
        fig, axes, pf, observations, output_path, fps,
        render_particles, render_rates, render_history,
    )
```

- [ ] **Step 5: Rewrite `animate_inference_multi_1d` against the driver**

Replace its body (signature and docstring unchanged):

```python
    true_params = np.array(
        [
            mass_config.positions[0, 0],
            mass_config.positions[1, 0],
            mass_config.masses[0],
            mass_config.masses[1],
        ]
    )
    true_rates = clock_rates(mass_config, clock_array)

    fig, axes = create_inference_dashboard()
    plot_clock_setup(axes["setup"], clock_array, mass_config)
    axes["setup"].set_xlim(xlim)

    def render_particles(ax: Axes, state: ParticleState) -> None:
        ax.clear()
        plot_particle_cloud_multi_1d(ax, state, true_params[:2])
        ax.set_xlim(xlim)
        ax.set_ylim(xlim)

    def render_rates(ax: Axes, obs: Observation, frame: int) -> None:
        ax.clear()
        plot_clock_rates(ax, true_rates, clock_array, label="True", color="lightcoral")
        plot_clock_rates(
            ax, obs.rates, clock_array, label="Observed", color="steelblue"
        )

    def render_history(
        ax: Axes, means: NDArray[np.floating], stds: NDArray[np.floating]
    ) -> None:
        ax.clear()
        steps = np.arange(1, len(means) + 1)
        _plot_convergence(
            ax,
            steps,
            means,
            stds,
            true_params,
            _MULTI_COLORS,
            _MULTI_LABELS,
            legend_kwargs={"fontsize": 7, "ncol": 2},
        )

    _animate_filter_dashboard(
        fig, axes, pf, observations, output_path, fps,
        render_particles, render_rates, render_history,
    )
```

- [ ] **Step 6: Rewrite `animate_inference_multi_2d` against the driver**

Replace its body (signature and docstring unchanged):

```python
    true_params = np.array(
        [
            mass_config.positions[0, 0],
            mass_config.positions[0, 1],
            mass_config.positions[1, 0],
            mass_config.positions[1, 1],
            mass_config.masses[0],
            mass_config.masses[1],
        ]
    )
    true_rates = clock_rates(mass_config, clock_array)

    fig, axes = create_inference_dashboard(figsize=(13, 10))
    plot_clock_setup_2d(axes["setup"], clock_array, mass_config)
    axes["setup"].set_xlim(xylim)
    axes["setup"].set_ylim(xylim)

    def render_particles(ax: Axes, state: ParticleState) -> None:
        ax.clear()
        plot_particle_cloud_multi_2d(ax, state, true_params[:2])
        ax.set_xlim(xylim)
        ax.set_ylim(xylim)
        ax.set_aspect("equal")

    render_rates = _make_rates_renderer_2d(clock_array, true_rates, xylim)

    def render_history(
        ax: Axes, means: NDArray[np.floating], stds: NDArray[np.floating]
    ) -> None:
        ax.clear()
        steps = np.arange(1, len(means) + 1)
        _plot_convergence(
            ax,
            steps,
            means,
            stds,
            true_params,
            _MULTI_2D_COLORS,
            _MULTI_2D_LABELS,
            legend_kwargs={"fontsize": 6, "ncol": 3},
        )

    _animate_filter_dashboard(
        fig, axes, pf, observations, output_path, fps,
        render_particles, render_rates, render_history,
    )
```

- [ ] **Step 7: Run the full suite and check size**

Run: `uv run pytest -q` → all pass.
Run: `wc -l src/clocks/viz.py` → expect roughly 600 lines (down from 841).

- [ ] **Step 8: Commit**

```bash
git add src/clocks/viz.py
git commit -m "Extract dashboard animation driver; dedupe 2D rates panel"
```

---

### Task 7: Split viz.py into _panels.py and _animate.py behind a facade

`viz.py` remains above the 500-line guideline after deduplication, so apply the spec's split. Import path `clocks.viz.<name>` stays valid for all 15 public names.

**Files:**
- Create: `src/clocks/_panels.py`
- Create: `src/clocks/_animate.py`
- Modify: `src/clocks/viz.py` (becomes a facade)
- Test: existing suite (no new tests)

- [ ] **Step 1: Create `src/clocks/_panels.py`**

Move from `viz.py`, unchanged: module docstring + imports they need, the color/label constants (`_MULTI_COLORS`, `_MULTI_LABELS`, `_MULTI_2D_COLORS`, `_MULTI_2D_LABELS`, `_POSTERIOR_COLORS`), `plot_clock_setup`, `plot_particle_cloud`, `plot_clock_rates`, `plot_clock_setup_2d`, `plot_particle_cloud_2d`, `plot_mass_histogram`, `plot_clock_rates_2d`, `create_inference_dashboard`, `plot_particle_cloud_multi_1d`, `plot_particle_cloud_multi_2d`, and `_plot_convergence`. Top-of-file docstring: `"""Static plotting primitives for clock-inference dashboards."""`

- [ ] **Step 2: Create `src/clocks/_animate.py`**

Move from `viz.py`, unchanged: `_save_animation`, `_animate_filter_dashboard`, `_make_rates_renderer_2d`, `animate_inference`, `animate_inference_2d`, `animate_inference_multi_1d`, `animate_inference_multi_2d`, `animate_model_comparison`. It imports the panel functions and constants from `clocks._panels`. Top-of-file docstring: `"""Animation drivers for clock-inference dashboards."""`

- [ ] **Step 3: Reduce `src/clocks/viz.py` to the facade**

```python
"""Matplotlib plotting and animation (public facade).

Implementations live in ``clocks._panels`` (static primitives) and
``clocks._animate`` (animation drivers).
"""

from clocks._animate import (
    animate_inference,
    animate_inference_2d,
    animate_inference_multi_1d,
    animate_inference_multi_2d,
    animate_model_comparison,
)
from clocks._panels import (
    create_inference_dashboard,
    plot_clock_rates,
    plot_clock_rates_2d,
    plot_clock_setup,
    plot_clock_setup_2d,
    plot_mass_histogram,
    plot_particle_cloud,
    plot_particle_cloud_2d,
    plot_particle_cloud_multi_1d,
    plot_particle_cloud_multi_2d,
)

__all__ = [
    "animate_inference",
    "animate_inference_2d",
    "animate_inference_multi_1d",
    "animate_inference_multi_2d",
    "animate_model_comparison",
    "create_inference_dashboard",
    "plot_clock_rates",
    "plot_clock_rates_2d",
    "plot_clock_setup",
    "plot_clock_setup_2d",
    "plot_mass_histogram",
    "plot_particle_cloud",
    "plot_particle_cloud_2d",
    "plot_particle_cloud_multi_1d",
    "plot_particle_cloud_multi_2d",
]
```

- [ ] **Step 4: Verify**

Run: `uv run pytest -q` → all pass (test_viz imports from `clocks.viz`).
Run: `wc -l src/clocks/_panels.py src/clocks/_animate.py src/clocks/viz.py` → each under 500.
Run: `uv run ruff check src/` → clean.

- [ ] **Step 5: Commit**

```bash
git add src/clocks/viz.py src/clocks/_panels.py src/clocks/_animate.py
git commit -m "Split viz into _panels and _animate behind unchanged facade"
```

---

### Task 8: Promote build_particle_filter to the public API

**Files:**
- Modify: `src/clocks/api.py` (rename `_build_particle_filter`)
- Modify: `src/clocks/__init__.py` (export)
- Modify: `scripts/demo_multi_mass.py` (already imports the private name — update in the same commit)
- Test: `tests/test_api.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_api.py`:

```python
def test_build_particle_filter_is_public_and_runs() -> None:
    simulation = simulate(_make_simulation_config(n_observations=3, seed=7))
    pf = build_particle_filter(_make_inference_config(n_masses=2, seed=7))

    assert pf.n_particles == _make_inference_config(n_masses=2).n_particles
    for obs in simulation.observations:
        pf.update(obs)
    assert pf.state.observations_seen == 3

    with pytest.raises(TypeError, match="fixed-K"):
        build_particle_filter(_make_inference_config(n_masses=(1, 2)))
```

Also extend the existing `test_public_api_is_exported_from_package` list with `"build_particle_filter"`. Import `build_particle_filter` at the top of the test file alongside the existing `infer`/`simulate` imports.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api.py::test_build_particle_filter_is_public_and_runs -v`
Expected: FAIL with `ImportError: cannot import name 'build_particle_filter'`.

- [ ] **Step 3: Rename and export**

In `src/clocks/api.py`: rename `_build_particle_filter` → `build_particle_filter`, update its one call site in `infer`, and give it a docstring:

```python
def build_particle_filter(config: InferenceConfig) -> ParticleFilter:
    """Construct the ParticleFilter that infer() would use for fixed-K config.

    Use this when you need to drive the filter observation-by-observation
    (e.g. for animation); ``infer()`` covers the run-to-completion case.
    """
    if isinstance(config.n_masses, tuple):
        raise TypeError("expected int for n_masses in fixed-K mode")
```

(keep the rest of the body unchanged). In `src/clocks/__init__.py`: add `build_particle_filter` to the `from clocks.api import ...` line and to `__all__` (keep the list sorted).

In `scripts/demo_multi_mass.py`: change the import

```python
from clocks.api import (
    _build_particle_filter,
    _inference_result_from_particle_filter,
    simulate,
)
```

to

```python
from clocks.api import build_particle_filter, simulate
```

rename its `_build_particle_filter(...)` call accordingly, and replace any use of `_inference_result_from_particle_filter(pf)` with `pf.estimate()` for the printed summary (print `est["mean"]`, `est["std"]`, `est["ess"]`).

- [ ] **Step 4: Run tests and the affected demo**

Run: `uv run pytest -q` → all pass.
Run: `uv run demo-multi-mass` → completes, writes `output/demo_multi_mass.gif`, printed estimates near truth (x≈[-2.0, 4.5], M≈[0.6, 0.4]).

- [ ] **Step 5: Commit**

```bash
git add src/clocks/api.py src/clocks/__init__.py tests/test_api.py scripts/demo_multi_mass.py
git commit -m "Promote build_particle_filter to the public API"
```

---

### Task 9: Port demo_1d.py to the public API

**Files:**
- Modify: `scripts/demo_1d.py` (full rewrite below)

- [ ] **Step 1: Rewrite the script**

```python
"""End-to-end 1D gravitational time dilation demo.

3 clocks along a track, one hidden mass. The particle filter
deduces position and magnitude from noisy clock readings.
"""

from pathlib import Path

import numpy as np

from clocks import (
    ClockArray,
    InferenceConfig,
    MassConfig,
    NoiseConfig,
    PriorConfig,
    SimulationConfig,
    build_particle_filter,
    simulate,
)
from clocks.viz import animate_inference

# --- Configuration ---
TRUE_X = 2.5
TRUE_M = 0.8
CLOCK_POSITIONS = [-5.0, 0.0, 5.0]
TRACK_OFFSET = 1.0
N_OBSERVATIONS = 50
NOISE_STD = 0.005
N_PARTICLES = 1000
JITTER_STD = 0.02
SEED = 42
OUTPUT_PATH = Path("output/demo_1d.gif")


def main() -> None:
    mass_config = MassConfig(
        positions=np.array([[TRUE_X]]), masses=np.array([TRUE_M])
    )
    clock_array = ClockArray(
        positions=np.array([[x] for x in CLOCK_POSITIONS]),
        track_offset=TRACK_OFFSET,
    )

    simulation = simulate(
        SimulationConfig(
            clock_array=clock_array,
            ground_truth=mass_config,
            noise=NoiseConfig(observation_std=NOISE_STD),
            n_observations=N_OBSERVATIONS,
            seed=SEED,
        )
    )
    print(f"True mass: x={TRUE_X}, M={TRUE_M}")
    print(f"True rates: {simulation.true_rates}")

    pf = build_particle_filter(
        InferenceConfig(
            clock_array=clock_array,
            noise=NoiseConfig(observation_std=NOISE_STD),
            prior=PriorConfig(position_range=(-8.0, 8.0), mass_range=(0.1, 2.0)),
            n_particles=N_PARTICLES,
            n_masses=1,
            jitter_std=JITTER_STD,
            seed=SEED,
        )
    )

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    print(f"Animating {N_OBSERVATIONS} observations...")
    animate_inference(
        clock_array, mass_config, simulation.observations, pf, OUTPUT_PATH
    )

    est = pf.estimate()
    print(f"Estimate: x={est['mean'][0]:.3f}±{est['std'][0]:.3f} (true {TRUE_X})")
    print(f"          M={est['mean'][1]:.3f}±{est['std'][1]:.3f} (true {TRUE_M})")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
```

Before replacing, diff against the current script: if the existing file prints extra diagnostics or uses different animation kwargs (`fps`, `xlim`), preserve those exact values.

- [ ] **Step 2: Run the demo end-to-end (the spec's GIF-parity gate)**

Run: `uv run demo-1d`
Expected: completes in a few minutes, writes `output/demo_1d.gif`, estimates near x=2.5, M=0.8. Open the GIF and compare against `assets/demo_1d.gif` — same 2×2 dashboard, converging cloud.

- [ ] **Step 3: Run suite and commit**

Run: `uv run pytest -q` → all pass.

```bash
git add scripts/demo_1d.py
git commit -m "Port demo_1d to the public simulate/build_particle_filter API"
```

---

### Task 10: Port demo_2d.py to the public API

**Files:**
- Modify: `scripts/demo_2d.py`

- [ ] **Step 1: Rewrite the wiring**

Same transformation as Task 9 (full pattern shown there). Keep this script's own constants and docstring: `TRUE_X=1.5, TRUE_Y=-1.0, TRUE_M=0.5, TRACK_OFFSET=3.0, N_OBSERVATIONS=50, NOISE_STD=0.005, N_PARTICLES=2000, JITTER_STD=0.02, SEED=42`, the 8 hand-placed clock positions array, and `OUTPUT_PATH = Path("output/demo_2d.gif")`. Replace the manual observation loop / `prior_sampler` / `forward_model` / `ParticleFilter` construction with:

```python
    simulation = simulate(
        SimulationConfig(
            clock_array=clock_array,
            ground_truth=mass_config,
            noise=NoiseConfig(observation_std=NOISE_STD),
            n_observations=N_OBSERVATIONS,
            seed=SEED,
        )
    )
    pf = build_particle_filter(
        InferenceConfig(
            clock_array=clock_array,
            noise=NoiseConfig(observation_std=NOISE_STD),
            prior=PriorConfig(position_range=(-8.0, 8.0), mass_range=(0.1, 2.0)),
            n_particles=N_PARTICLES,
            n_masses=1,
            jitter_std=JITTER_STD,
            seed=SEED,
        )
    )
```

with `mass_config = MassConfig(positions=np.array([[TRUE_X, TRUE_Y]]), masses=np.array([TRUE_M]))`, then `animate_inference_2d(clock_array, mass_config, simulation.observations, pf, OUTPUT_PATH)` and the final `pf.estimate()` printout for x, y, M. Imports mirror Task 9 plus `animate_inference_2d`.

- [ ] **Step 2: Smoke-run and commit**

Run: `uv run pytest -q` → all pass. (Full `uv run demo-2d` regeneration is optional — Task 9 already satisfies the spec's regenerate-one-GIF gate; run it only if you want to eyeball the output.)

```bash
git add scripts/demo_2d.py
git commit -m "Port demo_2d to the public simulate/build_particle_filter API"
```

---

### Task 11: Port demo_multi_mass_2d.py to the public API

**Files:**
- Modify: `scripts/demo_multi_mass_2d.py`

- [ ] **Step 1: Rewrite the wiring**

Keep: module docstring, all constants (`TRUE_X1, TRUE_Y1 = -3.0, 2.0`; `TRUE_X2, TRUE_Y2 = 4.0, -1.0`; `TRUE_M1=0.6, TRUE_M2=0.4, N_CLOCKS=10, TRACK_OFFSET=3.0, MIN_SEPARATION=1.5, N_OBSERVATIONS=80, NOISE_STD=0.005, N_PARTICLES=4000, JITTER_STD=0.02, SEED=42`), and the `generate_random_clocks` helper unchanged. Delete `enforce_ordering` and the manual prior/forward/constraint wiring — `build_particle_filter` installs the equivalent sort-based ordering constraint and log-prior for `n_masses=2`.

New `main()`:

```python
def main() -> None:
    rng = np.random.default_rng(SEED)

    mass_config = MassConfig(
        positions=np.array([[TRUE_X1, TRUE_Y1], [TRUE_X2, TRUE_Y2]]),
        masses=np.array([TRUE_M1, TRUE_M2]),
    )
    clock_positions = generate_random_clocks(
        N_CLOCKS, rng, exclude=[(TRUE_X1, TRUE_Y1), (TRUE_X2, TRUE_Y2)]
    )
    clock_array = ClockArray(positions=clock_positions, track_offset=TRACK_OFFSET)

    simulation = simulate(
        SimulationConfig(
            clock_array=clock_array,
            ground_truth=mass_config,
            noise=NoiseConfig(observation_std=NOISE_STD),
            n_observations=N_OBSERVATIONS,
            seed=SEED,
        )
    )
    print(f"True masses: ({TRUE_X1},{TRUE_Y1}) M={TRUE_M1}; "
          f"({TRUE_X2},{TRUE_Y2}) M={TRUE_M2}")
    print(f"Clocks: {N_CLOCKS} random positions")

    pf = build_particle_filter(
        InferenceConfig(
            clock_array=clock_array,
            noise=NoiseConfig(observation_std=NOISE_STD),
            prior=PriorConfig(position_range=(-8.0, 8.0), mass_range=(0.1, 2.0)),
            n_particles=N_PARTICLES,
            n_masses=2,
            jitter_std=JITTER_STD,
            seed=SEED,
        )
    )

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    print(f"Animating {N_OBSERVATIONS} observations...")
    animate_inference_multi_2d(
        clock_array, mass_config, simulation.observations, pf, OUTPUT_PATH
    )

    est = pf.estimate()
    labels = ["x1", "y1", "x2", "y2", "M1", "M2"]
    truths = [TRUE_X1, TRUE_Y1, TRUE_X2, TRUE_Y2, TRUE_M1, TRUE_M2]
    for i, (label, truth) in enumerate(zip(labels, truths)):
        print(f"  {label} = {est['mean'][i]:.3f} ± {est['std'][i]:.3f} (true {truth})")
    print(f"Saved: {OUTPUT_PATH}")
```

Imports mirror Task 9 plus `animate_inference_multi_2d`. Before replacing, diff against the current script and preserve any print/kwargs details not covered above.

- [ ] **Step 2: Verify and commit**

Run: `uv run pytest -q` → all pass.

```bash
git add scripts/demo_multi_mass_2d.py
git commit -m "Port demo_multi_mass_2d to the public API"
```

---

**Note on the remaining demos:** `scripts/demo_model_comparison.py` already uses the public `simulate`/`infer`/`ModelComparison` API — no port needed. `scripts/demo_density.py` keeps its custom density forward model per the spec; its only change is Task 12.

### Task 12: Give demo_density a figure

**Files:**
- Modify: `scripts/demo_density.py`
- Modify: `src/clocks/_cli.py` (no change needed — entry point already exists; verify only)

- [ ] **Step 1: Add the figure**

Keep the script's existing filter wiring (custom density forward model — per spec it does NOT move to `build_particle_filter`). After the final estimate, add a 3-panel static figure. Add imports `from pathlib import Path` and `import matplotlib` / `matplotlib.use("Agg")` before `import matplotlib.pyplot as plt`, and define `OUTPUT_PATH = Path("output/demo_density.png")` with the other constants. Append to `main()` after the final-estimate print:

```python
    # --- Static summary figure ---
    est = pf.estimate()
    fig, (ax_density, ax_rates, ax_conv) = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: true vs inferred density profile
    xs = np.linspace(-8, 8, 400)
    true_density = TRUE_AMPLITUDE * np.exp(-0.5 * ((xs - TRUE_MU) / TRUE_SIGMA) ** 2)
    mu_hat, sigma_hat, amp_hat = est["mean"]
    est_density = amp_hat * np.exp(-0.5 * ((xs - mu_hat) / sigma_hat) ** 2)
    ax_density.plot(xs, true_density, color="lightcoral", label="True")
    ax_density.plot(xs, est_density, color="steelblue", ls="--", label="Inferred")
    ax_density.set_xlabel("x")
    ax_density.set_ylabel("mass density")
    ax_density.set_title("Density profile")
    ax_density.legend()

    # Panel 2: true vs final predicted clock rates
    predicted = clock_rates_density_gaussian(est["mean"], clock_array)
    positions = clock_array.positions[:, 0]
    ax_rates.plot(positions, true_rates, "o-", color="lightcoral", label="True")
    ax_rates.plot(positions, predicted, "s--", color="steelblue", label="Inferred")
    ax_rates.set_xlabel("clock position")
    ax_rates.set_ylabel("tick rate")
    ax_rates.set_title("Clock rates")
    ax_rates.legend()

    # Panel 3: convergence of the three parameters
    history = pf.history[1:]
    means = np.array(
        [np.average(s.particles, weights=s.weights, axis=0) for s in history]
    )
    stds = np.array(
        [
            np.sqrt(
                np.average(
                    (s.particles - np.average(s.particles, weights=s.weights, axis=0))
                    ** 2,
                    weights=s.weights,
                    axis=0,
                )
            )
            for s in history
        ]
    )
    steps = np.arange(1, len(history) + 1)
    for j, (label, truth, color) in enumerate(
        [
            ("mu", TRUE_MU, "tab:blue"),
            ("sigma", TRUE_SIGMA, "tab:green"),
            ("A", TRUE_AMPLITUDE, "tab:orange"),
        ]
    ):
        ax_conv.plot(steps, means[:, j], color=color, label=f"{label} est")
        ax_conv.fill_between(
            steps, means[:, j] - stds[:, j], means[:, j] + stds[:, j],
            alpha=0.15, color=color,
        )
        ax_conv.axhline(truth, color=color, ls="--", alpha=0.5)
    ax_conv.set_xlabel("Observation #")
    ax_conv.set_title("Convergence")
    ax_conv.legend(fontsize=8)

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")
```

- [ ] **Step 2: Run it**

Run: `uv run demo-density`
Expected: existing text output unchanged, plus `output/demo_density.png` written — three panels, inferred curves near true ones.

- [ ] **Step 3: Copy the figure into assets and commit**

```bash
cp output/demo_density.png assets/demo_density.png
uv run pytest -q
git add scripts/demo_density.py assets/demo_density.png
git commit -m "Add static summary figure to demo_density"
```

---

### Task 13: README refresh

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Apply the edits**

1. Replace the test-count line:

```
uv run pytest                # 87 tests
```

with:

```
uv run pytest
```

2. In "Project structure", replace the `src/clocks/` block with:

```
src/clocks/
    types.py       Data structures (MassConfig, ClockArray, Observation, ParticleState)
    config.py      Public config dataclasses (SimulationConfig, InferenceConfig, ...)
    results.py     Public result dataclasses (SimulationResult, InferenceResult, ...)
    api.py         End-to-end entry points (simulate, infer, build_particle_filter)
    physics.py     Forward model: mass config → clock tick rates
    noise.py       Gaussian noise model and log-likelihood
    inference.py   Particle filter (SMC with systematic resampling)
    viz.py         Plotting and animation facade (_panels.py, _animate.py)
    _cli.py        Entry points for demo scripts
```

and add `test_api.py` to the tests line:

```
tests/
    test_api.py, test_physics.py, test_inference.py, test_noise.py, test_viz.py
```

3. After the "Use as a library" code block (after the "For fixed-K inference..." line), add:

```markdown
To drive the filter observation-by-observation (e.g. for custom animation),
build the same filter `infer` uses internally:

```python
from clocks import build_particle_filter

pf = build_particle_filter(config)   # fixed-K InferenceConfig
for obs in simulation.observations:
    pf.update(obs)
print(pf.estimate())
```
```

4. In the "Gaussian density" demo section, replace `# text output only, no GIF` with `# → output/demo_density.png` and add the image embed below the command block:

```markdown
![Gaussian density demo](assets/demo_density.png)
```

- [ ] **Step 2: Verify claims and commit**

Run: `uv run pytest -q` and confirm every command named in the README exists (`uv run demo-1d` … `uv run demo-density`, `uv run ruff check src/ tests/ scripts/`).

```bash
git add README.md
git commit -m "Refresh README: drop stale test count, document full structure and build_particle_filter"
```

---

### Task 14: LICENSE and package metadata

**Files:**
- Create: `LICENSE`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the MIT license**

Create `LICENSE` with the standard MIT text, copyright line:

```
MIT License

Copyright (c) 2026 Jonathan Whitmore

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 2: Complete pyproject metadata**

In `pyproject.toml` `[project]`, after the `description` line add:

```toml
readme = "README.md"
license = "MIT"
license-files = ["LICENSE"]
```

and after the `[project.scripts]` table add:

```toml
[project.urls]
Repository = "https://github.com/jbwhit/clocks"
```

(The site URL gets added in the website plan once it is live.) If `uv sync` rejects the PEP 639 `license = "MIT"` string under the pinned hatchling, fall back to `license = { text = "MIT" }` and drop `license-files`.

- [ ] **Step 3: Verify the metadata builds**

Run: `uv sync` → succeeds (rebuilds the editable install with new metadata).
Run: `uv run pytest -q` → all pass.

- [ ] **Step 4: Commit**

```bash
git add LICENSE pyproject.toml
git commit -m "Add MIT license and complete package metadata"
```

(If `uv sync` regenerated `uv.lock`, leave the lockfile uncommitted — it is deliberately committed with the website plan's Jupyter dependency task, per the spec.)

---

### Task 15: Resolve scratch files

**Files:**
- Create: `docs/someday-maybe.md`
- Modify: `.gitignore`
- Delete (untracked): `next.md`, `gemini-convo.md`
- Commit (untracked): `docs/superpowers/specs/2026-03-13-library-api-design-gemini-review.md`

- [ ] **Step 1: Create `docs/someday-maybe.md`**

```markdown
# Someday / Maybe

Ideas considered and deliberately not implemented yet. (Sources: scratch
notes and an external Gemini review, 2026-02; updated 2026-06.)

## Inference

- **MCMC rejuvenation step.** Add a Metropolis-Hastings accept/reject after
  the post-resampling jitter, turning the filter into a rigorous SMC sampler
  that exactly preserves the posterior. Today's jitter slightly distorts it.
- **Neural-net amortized inference.** Train a network on simulated
  (clock rates → mass parameters) pairs and compare its speed/accuracy
  against the particle filter.

## Physics

- **Special-relativity velocity term.** Give masses or clocks velocities and
  use the combined dilation factor sqrt(1 + 2*Phi - v^2) (c = 1), so SR and
  GR effects compete the way they do for real GPS satellites.
- **Mass placement scenarios.** Masses inside vs outside a ring of clocks in
  2D; which geometries make the inverse problem ill-posed?

## Deployment

- **In-browser interactivity.** The inference engine is pure numpy/scipy,
  which Pyodide supports — a live-slider demo could run client-side on the
  website without a server. (The website ships static GIFs by deliberate
  scope choice; see docs/superpowers/specs/2026-06-09-deep-clean-and-website-design.md.)
```

- [ ] **Step 2: Update `.gitignore`**

Add under the existing `# Gemini conversation` block:

```
.gemini/
```

- [ ] **Step 3: Delete scratch files, stage everything, verify clean status**

```bash
rm next.md gemini-convo.md
git add docs/someday-maybe.md .gitignore docs/superpowers/specs/2026-03-13-library-api-design-gemini-review.md
git status --short
```

Expected status: only the staged files plus the known deliberately-uncommitted `uv.lock` delta.

- [ ] **Step 4: Commit and push everything**

```bash
git commit -m "Absorb scratch notes into someday-maybe; track gemini review spec"
git push
```

---

## Final verification

- [ ] `uv run pytest -q` → all pass (expect 107: 102 + 5 new from Tasks 1, 3, 4, 5, 8; confirm the actual count and do NOT write it into the README).
- [ ] `uv run ruff check src/ tests/ scripts/` → clean.
- [ ] `git log --oneline` shows one commit per task; `git status` shows only the intentional `uv.lock` delta.
- [ ] `output/demo_1d.gif` regenerated and visually equivalent to `assets/demo_1d.gif` (Task 9).
