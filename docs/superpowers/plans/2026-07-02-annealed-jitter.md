# Annealed Jitter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `"annealed"` post-resampling jitter mode (exponential decay from prior scale to a floor), make it the library default, and validate it with a committed seed-scan harness at ≥ 10/12 holdout seeds on the multi-mass-2D problem.

**Architecture:** The jitter schedule lives in `ParticleFilter` and keys off `observations_seen`; the initial particle cloud's per-parameter std supplies the prior scale so no prior bounds are plumbed in. A one-shot reject-and-stay support-repair step (all jitter modes) keeps the public particle state inside prior support. A shared scenario module `clocks._scenarios` feeds the demo, the scan script, and the slow acceptance test. The animation drivers are fixed to precompute filter states (they currently double-process frame 0).

**Tech Stack:** Python 3.12, numpy/scipy/matplotlib, uv, pytest, ruff, Quarto (site).

**Spec:** `docs/superpowers/specs/2026-07-02-annealed-jitter-design.md` (Approved, Codex xhigh 4 rounds → SOUND ENOUGH TO IMPLEMENT).

**Status:** Approved — Codex xhigh, 3 rounds → SOUND ENOUGH TO IMPLEMENT (round 1: 2 Critical / 2 Important / 2 Minor applied, 1 pushback verified correct; round 2: 2 residuals applied; round 3: clean).

## Global Constraints

- Work on a feature branch `annealed-jitter`; commit after every task; push after every commit (repo workflow).
- Before every commit: `uv run ruff format --check .` AND `uv run ruff check .` AND `uv run pytest` all green (a pre-commit hook enforces the first two).
- Never run bare `python`; always `uv run ...`.
- Schedule: `sigma_t = floor + (init − floor) · exp(−t / tau)`, `t` = 0-based observation index (the state's `observations_seen` **before** the current update increments it).
- `init` clamp: `np.maximum(initial_cloud_std, jitter_std)` per parameter — the schedule never anneals upward.
- Pass rule (one scan seed): every |posterior_mean − truth| ≤ 0.5 for the 4 position components and ≤ 0.1 for the 2 masses.
- Tuning seeds 0–11; holdout seeds 100–111; acceptance = ≥ 10/12 on **holdout**. If holdout fails and the design changes, holdout moves to seeds 200–211 (burn rule).
- Winner selection (tuning seeds, total order): most passes → lowest median max-parameter abs error → smaller tau → smaller floor.
- Provisional defaults until Task 7 finalizes them from the scan: `jitter_tau = 15.0`; `jitter_std` defaults unchanged (`0.01` ParticleFilter, `0.02` InferenceConfig/ModelComparison).
- Out of scope: likelihood tempering, MCMC resample-move, annealing combined with `covariance` mode.

---

### Task 1: Post-jitter support repair + initial-cloud constraint

Pre-existing bug fix, independent of the annealed mode: particles jittered out of prior support currently enter the public state with uniform weight (`log_prior` runs before resampling, `src/clocks/inference.py:150`).

**Files:**
- Modify: `src/clocks/inference.py` (module-level helper + `__init__` + `_resample`)
- Test: `tests/test_inference.py` (new `TestSupportRepair` class)

**Interfaces:**
- Produces: `_repair_support(proposals, parents, log_prior, rng) -> NDArray` — module-level pure function in `clocks.inference`; `ParticleFilter.__init__` applies `constraint_fn` to the initial cloud before storing it.
- Consumes: nothing from other tasks.

- [ ] **Step 1: Write the failing tests.** First add `_repair_support` to the **existing** `from clocks.inference import ...` line at the top of `tests/test_inference.py` (do NOT add a new import mid-file — that trips ruff E402/F811). Then append:

```python
def _interval_log_prior(particles: np.ndarray) -> np.ndarray:
    """Support: every component in [-1, 1]."""
    lp = np.zeros(particles.shape[0])
    lp[np.any(np.abs(particles) > 1.0, axis=1)] = -np.inf
    return lp


class TestSupportRepair:
    def test_valid_proposals_pass_through_unchanged(self) -> None:
        proposals = np.array([[0.5], [-0.5]])
        parents = np.array([[0.1], [0.2]])
        out = _repair_support(
            proposals, parents, _interval_log_prior, np.random.default_rng(0)
        )
        assert np.array_equal(out, proposals)

    def test_rejected_proposals_revert_to_parent(self) -> None:
        proposals = np.array([[0.7], [1.5]])  # second is out of support
        parents = np.array([[0.5], [0.6]])
        out = _repair_support(
            proposals, parents, _interval_log_prior, np.random.default_rng(0)
        )
        assert out[0, 0] == 0.7  # valid proposal kept (reject-and-stay)
        assert out[1, 0] == 0.6  # invalid proposal reverted to parent

    def test_invalid_parent_replaced_from_valid_particles(self) -> None:
        proposals = np.array([[3.0], [0.4]])  # both rows: proposal 0 invalid
        parents = np.array([[2.0], [0.5]])  # ... and its parent is too
        out = _repair_support(
            proposals, parents, _interval_log_prior, np.random.default_rng(0)
        )
        assert out[0, 0] == 0.4  # safety net: drawn from the only valid particle
        assert out[1, 0] == 0.4

    def test_raises_when_no_valid_particles_exist(self) -> None:
        proposals = np.array([[3.0], [4.0]])
        parents = np.array([[2.0], [5.0]])
        with pytest.raises(RuntimeError, match="no valid particles"):
            _repair_support(
                proposals, parents, _interval_log_prior, np.random.default_rng(0)
            )

    def test_public_state_stays_in_support_after_resample(self) -> None:
        """Huge jitter + log_prior: no out-of-support particle may survive."""
        rng = np.random.default_rng(42)
        pf = ParticleFilter(
            n_particles=200,
            prior_sampler=lambda r, n: r.uniform(-1.0, 1.0, (n, 1)),
            forward_model=lambda p: p,
            noise_std=0.1,
            resample_threshold=1.1,  # force a resample every update
            jitter_std=5.0,  # most proposals leave [-1, 1]
            jitter="fixed",
            log_prior=_interval_log_prior,
            rng=rng,
        )
        pf.update(Observation(rates=np.array([0.0]), time=0.0))
        assert np.all(np.abs(pf.state.particles) <= 1.0)

    def test_initial_cloud_constraint_applied(self) -> None:
        """Unconstrained prior sampler + sorting constraint: stored initial
        particles must already satisfy the constraint."""

        def sort_rows(particles: np.ndarray) -> np.ndarray:
            return np.sort(particles, axis=1)

        pf = ParticleFilter(
            n_particles=50,
            prior_sampler=lambda r, n: r.uniform(-1.0, 1.0, (n, 2)),
            forward_model=lambda p: p,
            noise_std=0.1,
            constraint_fn=sort_rows,
            rng=np.random.default_rng(7),
        )
        p = pf.state.particles
        assert np.all(p[:, 0] <= p[:, 1])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_inference.py::TestSupportRepair -v`
Expected: FAIL / ERROR with `ImportError: cannot import name '_repair_support'`

- [ ] **Step 3: Implement** in `src/clocks/inference.py`:

Add after the `_JITTER_MODES` line:

```python
def _repair_support(
    proposals: NDArray[np.floating],
    parents: NDArray[np.floating],
    log_prior: Callable[[NDArray[np.floating]], NDArray[np.floating]],
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    """One-shot reject-and-stay support repair for post-jitter proposals.

    Proposals with -inf log-prior revert to their resampled parent's value.
    Parents that are themselves invalid (e.g. a zero-weight CDF plateau
    selected by left-sided searchsorted) are replaced by a uniform draw
    from the valid repaired particles. Deliberately NOT retry-until-valid:
    retrying samples a parent-dependent truncated proposal that biases
    particles away from support boundaries.
    """
    repaired = proposals.copy()
    invalid = np.isneginf(log_prior(repaired))
    if not invalid.any():
        return repaired
    repaired[invalid] = parents[invalid]
    still_invalid = np.isneginf(log_prior(repaired))
    if still_invalid.any():
        valid_idx = np.flatnonzero(~still_invalid)
        if valid_idx.size == 0:
            raise RuntimeError(
                "Support repair failed: no valid particles remain after "
                "reverting to parents; prior support and proposals are "
                "fully disjoint"
            )
        donors = rng.choice(valid_idx, size=int(still_invalid.sum()))
        repaired[np.flatnonzero(still_invalid)] = repaired[donors]
    return repaired
```

In `__init__`, replace the initial-cloud block:

```python
        particles = prior_sampler(self.rng, n_particles)
        if constraint_fn is not None:
            # Parents must satisfy the constraint for reject-and-stay
            # reversion to be sound on the very first resample.
            particles = constraint_fn(particles)
```

In `_resample`, keep a `parents` reference and add the repair step after `constraint_fn`:

```python
        parents = particles[indices]
        new_particles = parents.copy()
        # ... existing jitter branches unchanged ...
        if self.constraint_fn is not None:
            new_particles = self.constraint_fn(new_particles)
        if self.log_prior is not None:
            new_particles = _repair_support(
                new_particles, parents, self.log_prior, self.rng
            )
        new_weights = np.ones(n) / n
        return new_particles, new_weights
```

(`particles[indices]` fancy-indexing already copies, so `parents` is independent of the jittered `new_particles`.)

- [ ] **Step 4: Run the new tests** — `uv run pytest tests/test_inference.py::TestSupportRepair -v` → PASS

- [ ] **Step 5: Deliberately update the two call-counting tests** (their expected counts change by design — this is the spec's "updated deliberately, not silently"):

In `test_constraint_fn_applied` (`tests/test_inference.py:201`): the constraint now also runs once on the initial cloud, so with 5 forced resamples the count is 6. Replace the assertion:

```python
        # 1 call on the initial cloud (support-repair soundness) + 1 per
        # resample; resample_threshold=1.0 forces one per update.
        assert call_count[0] == 6, (
            f"Constraint: 1 init + 5 resamples expected, got {call_count[0]}"
        )
```

In `test_log_prior_called_each_update` (`tests/test_inference.py:446`): the repair step calls `log_prior` once per resample (all-valid particles return early after the first check). Make the resample count deterministic by adding `resample_threshold=1.0,` to the `ParticleFilter(...)` construction in this test, update the docstring to "once per update to reweight, plus once per resample for support repair", and replace the assertion:

```python
        # 5 reweight calls + 5 support-repair calls (threshold forces a
        # resample every update; all particles valid => one check each).
        assert call_count[0] == 10, (
            f"log_prior: 5 reweight + 5 repair calls expected, got {call_count[0]}"
        )
```

- [ ] **Step 6: Run the full suite** — `uv run pytest` → all pass. Repair changes particle trajectories for filters that pass `log_prior` (the ModelComparison tests); if one fails, investigate the assertion — thresholds must NOT be loosened; report instead of patching.
- [ ] **Step 7: Commit** — `git add src/clocks/inference.py tests/test_inference.py && git commit -m "Enforce prior support after resampling jitter (reject-and-stay repair)"`

---

### Task 2: Annealed jitter mode in ParticleFilter (opt-in for now)

**Files:**
- Modify: `src/clocks/inference.py` (`_JITTER_MODES`, `__init__`, new `_annealed_std`, `_resample`)
- Test: `tests/test_inference.py` (new `TestAnnealedJitter` class)

**Interfaces:**
- Consumes: Task 1's constrained initial cloud (`init` stds are captured post-constraint).
- Produces: `ParticleFilter(jitter="annealed", jitter_tau=...)`; `pf._annealed_std(t) -> NDArray` (per-parameter sigma at observation index t); attribute `pf._jitter_init: NDArray`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_inference.py`:

```python
class TestAnnealedJitter:
    def _make_pf(self, **kwargs: object) -> ParticleFilter:
        defaults: dict = dict(
            n_particles=500,
            prior_sampler=lambda r, n: r.uniform(-8.0, 8.0, (n, 2)),
            forward_model=lambda p: p,
            noise_std=0.1,
            jitter="annealed",
            jitter_std=0.01,
            jitter_tau=5.0,
            rng=np.random.default_rng(0),
        )
        defaults.update(kwargs)
        return ParticleFilter(**defaults)

    def test_schedule_starts_at_initial_cloud_scale(self) -> None:
        pf = self._make_pf()
        expected = np.maximum(pf.state.particles.std(axis=0), 0.01)
        assert np.allclose(pf._annealed_std(0), expected)

    def test_schedule_decays_to_floor(self) -> None:
        pf = self._make_pf()
        assert np.allclose(pf._annealed_std(10_000), 0.01)

    def test_schedule_never_anneals_upward(self) -> None:
        # Tight prior (std ~0.001) with a larger floor: constant at floor.
        pf = self._make_pf(
            prior_sampler=lambda r, n: r.uniform(-0.001, 0.001, (n, 2)),
            jitter_std=0.5,
        )
        assert np.allclose(pf._annealed_std(0), 0.5)
        assert np.allclose(pf._annealed_std(100), 0.5)

    @pytest.mark.parametrize("bad_tau", [0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_jitter_tau_raises(self, bad_tau: float) -> None:
        with pytest.raises(ValueError, match="jitter_tau"):
            self._make_pf(jitter_tau=bad_tau)

    @pytest.mark.parametrize("bad_std", [-0.1, float("nan"), float("inf")])
    def test_invalid_jitter_std_raises(self, bad_std: float) -> None:
        with pytest.raises(ValueError, match="jitter_std"):
            self._make_pf(jitter_std=bad_std)

    def test_annealed_mode_converges_1d(self) -> None:
        """Annealed jitter recovers a single 1D mass (standard scenario)."""
        rng = np.random.default_rng(3)
        true_params = np.array([2.0, 0.5])
        positions = np.linspace(-6, 6, 8).reshape(-1, 1)
        ca = ClockArray(positions=positions, track_offset=3.0)
        mc = MassConfig(
            positions=true_params[:1].reshape(1, 1), masses=true_params[1:]
        )
        rates = clock_rates(mc, ca)

        def forward(params: np.ndarray) -> np.ndarray:
            m = MassConfig(
                positions=params[:1].reshape(1, 1), masses=params[1:]
            )
            return clock_rates(m, ca)

        pf = ParticleFilter(
            n_particles=2000,
            prior_sampler=lambda r, n: np.column_stack(
                [r.uniform(-8, 8, n), r.uniform(0.1, 2.0, n)]
            ),
            forward_model=forward,
            noise_std=0.005,
            jitter="annealed",
            jitter_std=0.02,
            jitter_tau=5.0,
            rng=rng,
        )
        for t in range(60):
            noisy = rates + rng.normal(0, 0.005, size=rates.shape)
            pf.update(Observation(rates=noisy, time=float(t)))
        est = pf.estimate()
        assert abs(est["mean"][0] - 2.0) < 0.5
        assert abs(est["mean"][1] - 0.5) < 0.1
```

(`ClockArray`, `MassConfig`, `Observation`, `clock_rates` are already imported at the top of `tests/test_inference.py`; check and add any missing import.)

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_inference.py::TestAnnealedJitter -v` → FAIL (`unexpected keyword argument 'jitter_tau'` / `Unknown jitter mode`)

- [ ] **Step 3: Implement** in `src/clocks/inference.py`:

```python
import math  # top of file

_JITTER_MODES = {"fixed", "covariance", "annealed"}
```

`__init__`: add parameter `jitter_tau: float = 15.0` (after `jitter`), then validation with the other checks:

```python
        if not math.isfinite(jitter_std) or jitter_std < 0:
            msg = f"jitter_std must be finite and >= 0, got {jitter_std}"
            raise ValueError(msg)
        if not math.isfinite(jitter_tau) or jitter_tau <= 0:
            msg = f"jitter_tau must be finite and > 0, got {jitter_tau}"
            raise ValueError(msg)
```

store `self.jitter_tau = jitter_tau`, and after the (constrained) initial cloud exists:

```python
        # Prior scale for the annealed schedule: the initial cloud is a
        # prior sample. Clamped so the schedule never anneals upward.
        self._jitter_init = np.maximum(particles.std(axis=0), jitter_std)
```

Add the method:

```python
    def _annealed_std(self, t: float) -> NDArray[np.floating]:
        """Scheduled per-parameter jitter std at 0-based observation index t."""
        decay = np.exp(-t / self.jitter_tau)
        return self.jitter_std + (self._jitter_init - self.jitter_std) * decay
```

`_resample`: insert an `elif` between the covariance branch and the isotropic fallback:

```python
        if self.jitter == "covariance" and ess >= 2.0:
            # ... unchanged ...
        elif self.jitter == "annealed":
            # observations_seen has not been incremented for the update in
            # progress, so this is the 0-based index of the current one.
            sigma = self._annealed_std(self._state.observations_seen)
            z = self.rng.normal(0.0, 1.0, size=new_particles.shape)
            new_particles += z * sigma
        else:
            # ... unchanged fixed/fallback branch ...
```

Update the class docstring: document `jitter="annealed"` (axis-aligned diagonal Gaussian whose per-parameter scale decays from the initial cloud's std to the `jitter_std` floor with time constant `jitter_tau` observations) and `jitter_tau`.

- [ ] **Step 4: Run the tests** — `uv run pytest tests/test_inference.py -v` → PASS (all, including Task 1's)
- [ ] **Step 5: Commit** — `git commit -am "Add annealed jitter mode to ParticleFilter (opt-in)"`

---

### Task 3: Default flip + config/API plumbing + behavioral regressions

**Files:**
- Modify: `src/clocks/inference.py` (`ParticleFilter` and `ModelComparison` defaults), `src/clocks/config.py`, `src/clocks/api.py`
- Test: `tests/test_api.py` (new tests), `tests/test_inference.py` (defaults test)

**Interfaces:**
- Consumes: Task 2 (`jitter_tau`, `"annealed"`).
- Produces: `InferenceConfig(jitter_tau=...)` field; `jitter="annealed"` default in `ParticleFilter`, `ModelComparison`, `InferenceConfig`; `build_particle_filter`/`_infer_model_comparison` forward `jitter_tau`.

- [ ] **Step 1: Write the failing tests.** In `tests/test_inference.py`:

```python
class TestAnnealedDefaults:
    def test_particle_filter_default_jitter_is_annealed(self) -> None:
        pf = ParticleFilter(
            n_particles=10,
            prior_sampler=lambda r, n: r.uniform(-1, 1, (n, 1)),
            forward_model=lambda p: p,
            noise_std=0.1,
        )
        assert pf.jitter == "annealed"

    def test_model_comparison_default_jitter_is_annealed(self) -> None:
        ca = ClockArray(
            positions=np.linspace(-5, 5, 6).reshape(-1, 1), track_offset=3.0
        )
        mc = ModelComparison(clock_array=ca, noise_std=0.01, k_max=2)
        assert all(pf.jitter == "annealed" for pf in mc.filters.values())

    def test_model_comparison_jitter_tau_plumbs_through(self) -> None:
        ca = ClockArray(
            positions=np.linspace(-5, 5, 6).reshape(-1, 1), track_offset=3.0
        )
        mc = ModelComparison(
            clock_array=ca, noise_std=0.01, k_max=2, jitter_tau=7.0
        )
        assert all(pf.jitter_tau == 7.0 for pf in mc.filters.values())
```

In `tests/test_api.py` (uses the existing helpers/imports at the top of that file — `ClockArray`, `InferenceConfig`, `MassConfig`, `NoiseConfig`, `PriorConfig`, `SimulationConfig`, `build_particle_filter`, `infer`, `simulate`, `np`, `pytest`):

```python
class TestAnnealedDefaultsAPI:
    def _config(self, **kwargs: object) -> InferenceConfig:
        ca = ClockArray(
            positions=np.linspace(-5, 5, 6).reshape(-1, 1), track_offset=3.0
        )
        defaults: dict = dict(
            clock_array=ca,
            noise=NoiseConfig(observation_std=0.01),
            prior=PriorConfig(position_range=(-8.0, 8.0), mass_range=(0.1, 2.0)),
            n_particles=100,
            n_masses=1,
        )
        defaults.update(kwargs)
        return InferenceConfig(**defaults)

    def test_inference_config_default_jitter_is_annealed(self) -> None:
        assert self._config().jitter == "annealed"

    def test_jitter_tau_plumbs_through_build(self) -> None:
        pf = build_particle_filter(self._config(jitter_tau=7.0))
        assert pf.jitter_tau == 7.0
        assert pf.jitter == "annealed"

    @pytest.mark.parametrize("bad_tau", [0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_jitter_tau_raises(self, bad_tau: float) -> None:
        with pytest.raises(ValueError, match="jitter_tau"):
            self._config(jitter_tau=bad_tau)

    @pytest.mark.parametrize("bad_std", [-0.1, float("nan"), float("inf")])
    def test_invalid_jitter_std_raises(self, bad_std: float) -> None:
        with pytest.raises(ValueError, match="jitter_std"):
            self._config(jitter_std=bad_std)


class TestDefaultFlipRecovery:
    """Numerical recovery at the new defaults (spec: default-flip regressions).

    Single-mass 1D recovery and correct-K model comparison already exist;
    these add the missing single-mass 2D and multi-mass 1D coverage.
    """

    def test_single_mass_2d_recovery(self) -> None:
        rng = np.random.default_rng(3)
        ca = ClockArray(
            positions=rng.uniform(-5.0, 5.0, (8, 2)), track_offset=3.0
        )
        truth = MassConfig(
            positions=np.array([[1.5, -2.0]]), masses=np.array([0.5])
        )
        sim = simulate(
            SimulationConfig(
                clock_array=ca,
                ground_truth=truth,
                noise=NoiseConfig(observation_std=0.005),
                n_observations=60,
                seed=3,
            )
        )
        result = infer(
            sim.observations,
            InferenceConfig(
                clock_array=ca,
                noise=NoiseConfig(observation_std=0.005),
                prior=PriorConfig(
                    position_range=(-8.0, 8.0), mass_range=(0.1, 2.0)
                ),
                n_particles=2000,
                n_masses=1,
                seed=3,
            ),
        )
        error = np.abs(result.posterior_mean - np.array([1.5, -2.0, 0.5]))
        assert np.all(error <= np.array([0.5, 0.5, 0.1]))

    def test_multi_mass_1d_recovery(self) -> None:
        ca = ClockArray(
            positions=np.linspace(-6.0, 6.0, 10).reshape(-1, 1),
            track_offset=3.0,
        )
        truth = MassConfig(
            positions=np.array([[-3.0], [4.5]]), masses=np.array([0.6, 0.4])
        )
        sim = simulate(
            SimulationConfig(
                clock_array=ca,
                ground_truth=truth,
                noise=NoiseConfig(observation_std=0.005),
                n_observations=80,
                seed=5,
            )
        )
        result = infer(
            sim.observations,
            InferenceConfig(
                clock_array=ca,
                noise=NoiseConfig(observation_std=0.005),
                prior=PriorConfig(
                    position_range=(-8.0, 8.0), mass_range=(0.1, 2.0)
                ),
                n_particles=4000,
                n_masses=2,
                seed=5,
            ),
        )
        truth_vec = np.array([-3.0, 4.5, 0.6, 0.4])
        error = np.abs(result.posterior_mean - truth_vec)
        assert np.all(error <= np.array([0.5, 0.5, 0.1, 0.1]))
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_api.py::TestAnnealedDefaultsAPI tests/test_inference.py::TestAnnealedDefaults -v` → FAIL (`unexpected keyword argument 'jitter_tau'` on config / default is `"fixed"`)

- [ ] **Step 3: Implement.**

`src/clocks/inference.py`: `ParticleFilter.__init__` signature → `jitter: str = "annealed"`. `ModelComparison.__init__` → `jitter: str = "annealed"`, add `jitter_tau: float = 15.0`, and pass `jitter_tau=jitter_tau` in its `ParticleFilter(...)` construction.

`src/clocks/config.py`, in `InferenceConfig` (add `import math` at top of file):

```python
    jitter: str = "annealed"
    jitter_tau: float = 15.0
```

and in `__post_init__`:

```python
        if not math.isfinite(self.jitter_std) or self.jitter_std < 0:
            raise ValueError(
                f"jitter_std must be finite and >= 0, got {self.jitter_std}"
            )
        if not math.isfinite(self.jitter_tau) or self.jitter_tau <= 0:
            raise ValueError(
                f"jitter_tau must be finite and > 0, got {self.jitter_tau}"
            )
```

Update the docstring: `jitter_std` is the absolute std for `"fixed"`, the covariance fraction for `"covariance"`, and the **floor** (late-run asymptote) for `"annealed"`; `jitter_tau` is the anneal time constant in observations.

`src/clocks/api.py`: add `jitter_tau=config.jitter_tau` to the `ParticleFilter(...)` call in `build_particle_filter` and to the `ModelComparison(...)` call in `_infer_model_comparison`.

- [ ] **Step 4: Run the new tests** — the Step-2 command plus `uv run pytest tests/test_api.py::TestDefaultFlipRecovery -v` → PASS. If a recovery test misses the tolerance, try seeds 4 then 5 for the failing scenario (deterministic pin selection); if none recover, STOP — that is exactly the regression the spec is guarding against; report it.
- [ ] **Step 5: Run the full suite** — `uv run pytest` → all pass. Existing default-sensitive tests (single-mass-1D recovery, correct-K in `tests/test_inference.py:760` and `tests/test_api.py:158`) must pass with **thresholds unchanged**. A failure here is the spec's recorded fallback trigger: report it; the fallback (do NOT implement without agreement) is `ModelComparison` keeping `jitter="fixed"` + an `InferenceConfig.jitter: str | None = None` sentinel resolved at build time.
- [ ] **Step 6: Commit** — `git commit -am "Make annealed jitter the library default; plumb jitter_tau through config and API"`

---

### Task 4: Animation frame-0 fix (precompute states)

`FuncAnimation` with a state-mutating callback and no `init_func` processes frame 0 twice (81 filter updates for 80 observations), which would also shift the anneal schedule.

**Files:**
- Modify: `src/clocks/_animate.py` (`_animate_filter_dashboard`, `animate_model_comparison`)
- Test: `tests/test_viz.py`

**Interfaces:**
- Consumes: nothing new; signatures of all public `animate_*` functions unchanged.
- Produces: animations that leave `pf.state.observations_seen == len(observations)`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_viz.py` (reuse the file's existing fixtures/imports; it already imports `animate_inference`, `animate_model_comparison`, and builds small scenarios):

```python
class TestAnimationProcessesObservationsOnce:
    def test_dashboard_animation_observation_count(
        self, tmp_path: Path, clock_array_1d: ClockArray, mass_config_1d: MassConfig
    ) -> None:
        rng = np.random.default_rng(0)
        true_rates = clock_rates(mass_config_1d, clock_array_1d)
        observations = [
            Observation(
                rates=true_rates + rng.normal(0, 0.01, true_rates.shape),
                time=float(t),
            )
            for t in range(4)
        ]
        pf = ParticleFilter(
            n_particles=50,
            prior_sampler=lambda r, n: np.column_stack(
                [r.uniform(-8, 8, n), r.uniform(0.1, 2.0, n)]
            ),
            forward_model=lambda p: clock_rates(
                MassConfig(positions=p[:1].reshape(1, 1), masses=p[1:]),
                clock_array_1d,
            ),
            noise_std=0.01,
        )
        animate_inference(
            clock_array_1d,
            mass_config_1d,
            observations,
            pf,
            tmp_path / "anim.gif",
        )
        assert pf.state.observations_seen == len(observations)

    def test_model_comparison_animation_observation_count(
        self, tmp_path: Path, clock_array_1d: ClockArray, mass_config_1d: MassConfig
    ) -> None:
        rng = np.random.default_rng(0)
        true_rates = clock_rates(mass_config_1d, clock_array_1d)
        observations = [
            Observation(
                rates=true_rates + rng.normal(0, 0.01, true_rates.shape),
                time=float(t),
            )
            for t in range(3)
        ]
        mc = ModelComparison(
            clock_array=clock_array_1d, noise_std=0.01, k_max=2, n_particles=50
        )
        animate_model_comparison(
            clock_array_1d, mass_config_1d, observations, mc, tmp_path / "mc.gif"
        )
        for pf in mc.filters.values():
            assert pf.state.observations_seen == len(observations)
```

Add any imports `tests/test_viz.py` is missing (`ParticleFilter`, `ModelComparison`, `Observation`, `clock_rates`, `Path`).

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_viz.py::TestAnimationProcessesObservationsOnce -v`
Expected: FAIL with `observations_seen == 5` (or 4) vs `len(observations) == 4` (or 3) — the double-processed frame 0.

- [ ] **Step 3: Implement.** In `_animate_filter_dashboard`, replace the mutating callback with a precompute pass:

```python
    states: list[ParticleState] = []
    means: list[NDArray[np.floating]] = []
    stds: list[NDArray[np.floating]] = []
    for obs in observations:
        state = pf.update(obs)
        est = pf.estimate()
        states.append(state)
        means.append(est["mean"])
        stds.append(est["std"])
    if pf.state.observations_seen != len(observations):
        raise RuntimeError(
            f"Animation expected a fresh filter: saw "
            f"{pf.state.observations_seen} observations for "
            f"{len(observations)} frames"
        )

    def render(frame: int) -> None:
        render_particles(axes["particles"], states[frame])
        render_rates(axes["rates"], observations[frame], frame)
        render_history(
            axes["history"],
            np.array(means[: frame + 1]),
            np.array(stds[: frame + 1]),
        )

    anim = animation.FuncAnimation(fig, render, frames=len(observations), repeat=False)
    _save_animation(anim, fig, output_path, fps)
```

In `animate_model_comparison`, same pattern: loop `model_comparison.update(obs)` collecting `evidence()["posterior"]` per observation into `posteriors_seq`, check every `filters.values()` count, and have `update(frame)` read `observations[frame]` / `posteriors_seq[frame]` instead of mutating.

- [ ] **Step 4: Run viz tests** — `uv run pytest tests/test_viz.py -v` → PASS
- [ ] **Step 5: Run full suite** — `uv run pytest` → PASS
- [ ] **Step 6: Commit** — `git commit -am "Fix animation double-processing of frame 0 by precomputing filter states"`

---

### Task 5: Shared scenario module `clocks._scenarios` + demo refactor

**Files:**
- Create: `src/clocks/_scenarios.py`
- Modify: `scripts/demo_multi_mass_2d.py` (import shared pieces)
- Test: `tests/test_scenarios.py` (fast tests only)

**Interfaces:**
- Consumes: `infer`, `simulate`, configs (existing API); Task 3's defaults.
- Produces (used by Tasks 6–7):
  - constants `TRUTH: NDArray` (`[-3.0, 2.0, 4.0, -1.0, 0.6, 0.4]`), `PASS_TOLERANCE: NDArray` (`[0.5]*4 + [0.1]*2`), `N_CLOCKS=10`, `TRACK_OFFSET=3.0`, `MIN_SEPARATION=1.5`, `N_OBSERVATIONS=80`, `NOISE_STD=0.005`, `N_PARTICLES=4000`, `POSITION_RANGE=(-8.0, 8.0)`, `MASS_RANGE=(0.1, 2.0)`, `TRUE_POSITIONS`, `TRUE_MASSES`
  - `generate_random_clocks(n, rng, *, bounds=(-6.0, 6.0), min_sep=MIN_SEPARATION, exclude=None) -> NDArray`
  - `passes(posterior_mean: NDArray) -> bool`
  - `run_multi_mass_2d(seed: int, *, jitter: str = "annealed", jitter_std: float = 0.02, jitter_tau: float = 15.0) -> RunResult` where `RunResult` is a TypedDict with keys `seed, passed, mean, std, max_abs_error, covered_3sigma, max_posterior_std, residual_over_noise`

- [ ] **Step 1: Write the failing tests** — create `tests/test_scenarios.py`:

```python
"""Fast tests for the shared multi-mass-2D scenario module."""

import numpy as np

from clocks._scenarios import (
    MIN_SEPARATION,
    PASS_TOLERANCE,
    TRUTH,
    generate_random_clocks,
    passes,
)


class TestPassRule:
    def test_truth_passes(self) -> None:
        assert passes(TRUTH)

    def test_position_error_at_tolerance_passes(self) -> None:
        assert passes(TRUTH + np.array([0.5, 0, 0, 0, 0, 0]))

    def test_position_error_beyond_tolerance_fails(self) -> None:
        assert not passes(TRUTH + np.array([0.51, 0, 0, 0, 0, 0]))

    def test_mass_error_beyond_tolerance_fails(self) -> None:
        assert not passes(TRUTH + np.array([0, 0, 0, 0, 0.11, 0]))

    def test_tolerance_values(self) -> None:
        assert np.array_equal(
            PASS_TOLERANCE, np.array([0.5, 0.5, 0.5, 0.5, 0.1, 0.1])
        )


class TestClockPlacement:
    def test_respects_min_separation_and_exclusions(self) -> None:
        rng = np.random.default_rng(11)
        exclude = [(-3.0, 2.0), (4.0, -1.0)]
        clocks = generate_random_clocks(10, rng, exclude=exclude)
        assert clocks.shape == (10, 2)
        for i in range(10):
            for j in range(i + 1, 10):
                assert np.linalg.norm(clocks[i] - clocks[j]) >= MIN_SEPARATION
            for p in exclude:
                assert np.linalg.norm(clocks[i] - np.array(p)) >= MIN_SEPARATION
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_scenarios.py -v` → FAIL (`ModuleNotFoundError: clocks._scenarios`)

- [ ] **Step 3: Create `src/clocks/_scenarios.py`:**

```python
"""Shared multi-mass-2D scenario: demo, scan harness, and acceptance test.

This is the problem instance whose premature-collapse failure motivated
the annealed jitter mode (spec:
docs/superpowers/specs/2026-07-02-annealed-jitter-design.md). It lives in
the package (not scripts/) because the demo console-scripts launch via
runpy and pytest imports from the repo root; neither puts scripts/ on
sys.path.
"""

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from clocks.api import infer, simulate
from clocks.config import InferenceConfig, NoiseConfig, PriorConfig, SimulationConfig
from clocks.physics import clock_rates
from clocks.results import InferenceResult
from clocks.types import ClockArray, MassConfig

TRUE_POSITIONS = np.array([[-3.0, 2.0], [4.0, -1.0]])
TRUE_MASSES = np.array([0.6, 0.4])
TRUTH = np.array([-3.0, 2.0, 4.0, -1.0, 0.6, 0.4])
N_CLOCKS = 10
TRACK_OFFSET = 3.0
MIN_SEPARATION = 1.5
N_OBSERVATIONS = 80
NOISE_STD = 0.005
N_PARTICLES = 4000
POSITION_RANGE = (-8.0, 8.0)
MASS_RANGE = (0.1, 2.0)
# Pass rule (see spec §3): abs posterior-mean error per parameter. These
# tolerances reproduce the June 2026 ad-hoc scan's implicit criterion.
PASS_TOLERANCE = np.array([0.5, 0.5, 0.5, 0.5, 0.1, 0.1])


class RunResult(TypedDict):
    """One scan run: gate result plus non-gating diagnostics."""

    seed: int
    passed: bool
    mean: NDArray[np.floating]
    std: NDArray[np.floating]
    max_abs_error: float
    covered_3sigma: bool
    max_posterior_std: float
    residual_over_noise: float


def generate_random_clocks(
    n: int,
    rng: np.random.Generator,
    *,
    bounds: tuple[float, float] = (-6.0, 6.0),
    min_sep: float = MIN_SEPARATION,
    exclude: list[tuple[float, float]] | None = None,
) -> NDArray[np.floating]:
    """Place n clocks on a 2D plane via rejection sampling.

    Keeps clocks at least min_sep apart from each other and from any
    positions listed in exclude (e.g. true mass locations).
    """
    placed: list[NDArray[np.floating]] = []
    blocked = [np.array(p) for p in (exclude or [])]
    while len(placed) < n:
        candidate = rng.uniform(bounds[0], bounds[1], 2)
        too_close = any(
            np.linalg.norm(candidate - p) < min_sep for p in placed + blocked
        )
        if not too_close:
            placed.append(candidate)
    return np.array(placed)


def passes(posterior_mean: NDArray[np.floating]) -> bool:
    """Gate: every parameter within its absolute tolerance of truth."""
    return bool(np.all(np.abs(posterior_mean - TRUTH) <= PASS_TOLERANCE))


def run_multi_mass_2d(
    seed: int,
    *,
    jitter: str = "annealed",
    jitter_std: float = 0.02,
    jitter_tau: float = 15.0,
) -> RunResult:
    """One end-to-end run: seed drives clocks, sim noise, and filter rng."""
    rng = np.random.default_rng(seed)
    mass_config = MassConfig(positions=TRUE_POSITIONS, masses=TRUE_MASSES)
    clock_positions = generate_random_clocks(
        N_CLOCKS,
        rng,
        exclude=[tuple(p) for p in TRUE_POSITIONS],
    )
    clock_array = ClockArray(positions=clock_positions, track_offset=TRACK_OFFSET)
    sim = simulate(
        SimulationConfig(
            clock_array=clock_array,
            ground_truth=mass_config,
            noise=NoiseConfig(observation_std=NOISE_STD),
            n_observations=N_OBSERVATIONS,
            seed=seed,
        )
    )
    result = infer(
        sim.observations,
        InferenceConfig(
            clock_array=clock_array,
            noise=NoiseConfig(observation_std=NOISE_STD),
            prior=PriorConfig(
                position_range=POSITION_RANGE, mass_range=MASS_RANGE
            ),
            n_particles=N_PARTICLES,
            n_masses=2,
            jitter=jitter,
            jitter_std=jitter_std,
            jitter_tau=jitter_tau,
            seed=seed,
        ),
    )
    assert isinstance(result, InferenceResult)  # fixed-K mode
    mean, std = result.posterior_mean, result.posterior_std
    error = np.abs(mean - TRUTH)
    predicted = clock_rates(
        MassConfig(positions=mean[:4].reshape(2, 2), masses=mean[4:]),
        clock_array,
    )
    return RunResult(
        seed=seed,
        passed=passes(mean),
        mean=mean,
        std=std,
        max_abs_error=float(error.max()),
        covered_3sigma=bool(np.all(error <= 3.0 * std)),
        max_posterior_std=float(std.max()),
        residual_over_noise=float(
            np.max(np.abs(predicted - sim.true_rates)) / NOISE_STD
        ),
    )
```

- [ ] **Step 4: Run the tests** — `uv run pytest tests/test_scenarios.py -v` → PASS

- [ ] **Step 5: Refactor `scripts/demo_multi_mass_2d.py`** to import the shared pieces instead of defining them. Replace the constants block and local `generate_random_clocks` with:

```python
from clocks._scenarios import (
    MIN_SEPARATION,
    N_CLOCKS,
    N_OBSERVATIONS,
    N_PARTICLES,
    NOISE_STD,
    TRACK_OFFSET,
    TRUE_MASSES,
    TRUE_POSITIONS,
    generate_random_clocks,
)

JITTER_STD = 0.02  # floor for the annealed default; finalized by the scan
SEED = 11
OUTPUT_PATH = Path("output/demo_multi_mass_2d.gif")
```

and update `main()` to use `TRUE_POSITIONS`/`TRUE_MASSES` (keep the printed labels by indexing them, e.g. `TRUE_POSITIONS[0][0]` where the old `TRUE_X1` appeared). Keep `MIN_SEPARATION` only if still referenced. Delete the now-duplicated local definitions. Run `uv run python -c "import runpy; runpy.run_path('scripts/demo_multi_mass_2d.py')"` is NOT needed — just verify import wiring by the console script below.

- [ ] **Step 6: Smoke-run the demo via the console script** (exercises the runpy path): `uv run demo-multi-mass-2d` → completes, writes `output/demo_multi_mass_2d.gif`, prints estimates. (~1–2 min.)
- [ ] **Step 7: Full gate** — `uv run pytest && uv run ruff format --check . && uv run ruff check .` → green
- [ ] **Step 8: Commit** (explicit adds — `-am` misses new files) — `git add src/clocks/_scenarios.py tests/test_scenarios.py scripts/demo_multi_mass_2d.py && git commit -m "Extract shared multi-mass-2D scenario into clocks._scenarios"`

---

### Task 6: Scan script + slow acceptance test + pytest marker

**Files:**
- Create: `scripts/scan_multi_mass_2d.py`, `tests/test_acceptance_multi_mass_2d.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `clocks._scenarios.run_multi_mass_2d`, `RunResult`.
- Produces: `uv run scripts/scan_multi_mass_2d.py [--taus ...] [--floors ...] [--baseline] [--holdout] [--workers N]`; `@pytest.mark.slow` acceptance test on holdout seeds.

- [ ] **Step 1: Register the `slow` marker** in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-m 'not slow'"
markers = [
    "slow: long-running acceptance scans (run with `uv run pytest -m slow`)",
]
```

- [ ] **Step 2: Write the acceptance test** — create `tests/test_acceptance_multi_mass_2d.py`:

```python
"""Slow acceptance scan: annealed-jitter defaults on the holdout seeds.

Deterministic regression pin (same seeds + same code => same result), not
a population reliability estimate. Excluded from default runs; execute
with `uv run pytest -m slow`. Rerun whenever inference defaults change.
"""

import pytest

from clocks._scenarios import run_multi_mass_2d

HOLDOUT_SEEDS = tuple(range(100, 112))


@pytest.mark.slow
def test_annealed_defaults_pass_holdout_scan() -> None:
    results = [run_multi_mass_2d(seed) for seed in HOLDOUT_SEEDS]
    failed = [r["seed"] for r in results if not r["passed"]]
    assert len(HOLDOUT_SEEDS) - len(failed) >= 10, (
        f"holdout acceptance below 10/12; failing seeds: {failed}"
    )
```

- [ ] **Step 3: Verify the marker excludes it** — `uv run pytest --collect-only -q | tail -3` → the acceptance test is deselected; `uv run pytest tests/test_acceptance_multi_mass_2d.py -m slow --collect-only -q` → selected. (Do not run it yet; defaults aren't final.)

- [ ] **Step 4: Write the scan script** — create `scripts/scan_multi_mass_2d.py`:

```python
"""Seed-scan harness for the multi-mass-2D scenario.

Tunes (jitter_tau, floor) on seeds 0-11, prints a per-cell pass table and
the winner under the spec's total order, and re-measures the fixed-jitter
baseline post-support-repair. See
docs/superpowers/specs/2026-07-02-annealed-jitter-design.md §3.

Usage:
    uv run scripts/scan_multi_mass_2d.py                 # tuning grid
    uv run scripts/scan_multi_mass_2d.py --baseline      # fixed-jitter baseline
    uv run scripts/scan_multi_mass_2d.py --holdout --taus 15 --floors 0.02
"""

import argparse
import statistics
from multiprocessing import Pool

from clocks._scenarios import RunResult, run_multi_mass_2d

TUNING_SEEDS = tuple(range(12))
HOLDOUT_SEEDS = tuple(range(100, 112))


def _run(job: tuple[int, str, float, float]) -> tuple[tuple, RunResult]:
    seed, jitter, floor, tau = job
    if jitter == "fixed":
        # tau is a display/sort key only here: jitter_tau must validate
        # (> 0) even when unused, so don't pass the 0.0 placeholder.
        result = run_multi_mass_2d(seed, jitter="fixed", jitter_std=floor)
    else:
        result = run_multi_mass_2d(
            seed, jitter="annealed", jitter_std=floor, jitter_tau=tau
        )
    return (jitter, tau, floor), result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taus", type=float, nargs="+", default=[5, 10, 15, 25, 40])
    parser.add_argument(
        "--floors", type=float, nargs="+", default=[0.01, 0.02, 0.05]
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="fixed-jitter baseline over --floors instead of the annealed grid",
    )
    parser.add_argument(
        "--holdout", action="store_true", help="use holdout seeds 100-111"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--per-run",
        action="store_true",
        help="print per-run diagnostics (3-sigma coverage, max std, residual)",
    )
    args = parser.parse_args()

    seeds = HOLDOUT_SEEDS if args.holdout else TUNING_SEEDS
    if args.baseline:
        jobs = [("fixed", 0.0, floor) for floor in args.floors]
    else:
        jobs = [("annealed", tau, floor) for tau in args.taus for floor in args.floors]
    runs = [
        (seed, jitter, floor, tau)
        for (jitter, tau, floor) in jobs
        for seed in seeds
    ]

    with Pool(args.workers) as pool:
        results = pool.map(_run, runs)

    cells: dict[tuple, list[RunResult]] = {}
    for key, result in results:
        cells.setdefault(key, []).append(result)

    header = (
        f"{'mode':>9} {'tau':>6} {'floor':>6} {'pass':>6}"
        f" {'med|err|':>9} {'resid':>7}"
    )
    print(header)
    ranked = []
    for (jitter, tau, floor), cell in sorted(cells.items()):
        n_pass = sum(r["passed"] for r in cell)
        med_err = statistics.median(r["max_abs_error"] for r in cell)
        med_resid = statistics.median(r["residual_over_noise"] for r in cell)
        ranked.append(((-n_pass, med_err, tau, floor), (jitter, tau, floor), n_pass))
        print(
            f"{jitter:>9} {tau:>6g} {floor:>6g} {n_pass:>4}/{len(cell)}"
            f" {med_err:>9.3f} {med_resid:>7.1f}"
        )
        if args.per_run:
            for r in sorted(cell, key=lambda r: r["seed"]):
                print(
                    f"    seed {r['seed']:>3}  pass={int(r['passed'])}"
                    f"  max|err|={r['max_abs_error']:.3f}"
                    f"  3sig={int(r['covered_3sigma'])}"
                    f"  maxstd={r['max_posterior_std']:.3f}"
                    f"  resid/noise={r['residual_over_noise']:.1f}"
                )

    if not args.baseline:
        ranked.sort(key=lambda item: item[0])
        _, (jitter, tau, floor), n_pass = ranked[0]
        seed_kind = "holdout" if args.holdout else "tuning"
        print(
            f"\nwinner: tau={tau:g} floor={floor:g}"
            f" ({n_pass}/{len(seeds)} on {seed_kind} seeds)"
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Format and smoke-test** — `uv run ruff format scripts/scan_multi_mass_2d.py tests/test_acceptance_multi_mass_2d.py` (the plan's code blocks are not guaranteed formatter-clean), then `uv run scripts/scan_multi_mass_2d.py --taus 15 --floors 0.02 --workers 4` (12 runs, ~1–3 min) → prints one table row + winner line, no traceback.
- [ ] **Step 6: Full gate** — `uv run pytest && uv run ruff format --check . && uv run ruff check .` → green (acceptance test stays deselected).
- [ ] **Step 7: Commit** (explicit adds — `-am` misses new files) — `git add pyproject.toml scripts/scan_multi_mass_2d.py tests/test_acceptance_multi_mass_2d.py && git commit -m "Add multi-mass-2D seed-scan harness and slow holdout acceptance test"`

---

### Task 7: Run the scan, finalize defaults, certify holdout [DECISION GATE]

**Files:**
- Modify (winner values): `src/clocks/inference.py` (`jitter_tau` defaults ×2), `src/clocks/config.py` (`jitter_tau` default), `src/clocks/_scenarios.py` (`run_multi_mass_2d` keyword defaults), `scripts/demo_multi_mass_2d.py` (`JITTER_STD`), `docs/superpowers/specs/2026-07-02-annealed-jitter-design.md` (status line records shipped defaults + measured numbers)

**Interfaces:**
- Consumes: Task 6's script and test.
- Produces: final shipped `jitter_tau` / demo floor; recorded post-repair baseline and holdout result.

- [ ] **Step 1: Baseline** — `uv run scripts/scan_multi_mass_2d.py --baseline` → record the post-repair fixed-jitter pass counts for floors 0.01/0.02/0.05 (June's pre-repair numbers 1/12, 5/12, 7/12 are provenance only).
- [ ] **Step 2: Tuning grid** — `uv run scripts/scan_multi_mass_2d.py` (full 15-cell grid × 12 seeds, ~5–15 min with 8 workers) → note the winner (total order is built into the script).
- [ ] **Step 3: DECISION GATE** — if the winner has < 10/12 on tuning seeds, STOP: do not flip anything further, report the table. (Spec fallback: hybrid schedule with a posterior-std-scaled lower bound — needs explicit approval, and holdout burns to seeds 200–211.)
- [ ] **Step 4: Holdout certification** — `uv run scripts/scan_multi_mass_2d.py --holdout --taus <winner_tau> --floors <winner_floor>` → must be ≥ 10/12. If not: STOP, same rule as Step 3.
- [ ] **Step 5: Write the winner into the defaults** — update `jitter_tau` defaults in `ParticleFilter.__init__`, `ModelComparison.__init__`, `InferenceConfig`, and `run_multi_mass_2d`; set `scripts/demo_multi_mass_2d.py::JITTER_STD` and `run_multi_mass_2d`'s `jitter_std` default to the winning floor. Update the spec's Status line: shipped defaults + baseline table + tuning winner + holdout score.

Then add a **fast** default-pinning test to `tests/test_acceptance_multi_mass_2d.py` (the slow holdout test calls `run_multi_mass_2d`, which forwards its *own* defaults — it cannot catch a stale `jitter_tau` in `ParticleFilter`/`InferenceConfig`/`ModelComparison`). Fill in the certified values:

```python
import inspect

from clocks._scenarios import run_multi_mass_2d
from clocks.config import InferenceConfig
from clocks.inference import ModelComparison, ParticleFilter

CERTIFIED_TAU = 15.0  # <- replace with the scan winner's tau
CERTIFIED_FLOOR = 0.02  # <- replace with the scan winner's floor


def test_shipped_defaults_match_certified_cell() -> None:
    """Fast guard: every shipped jitter_tau default equals the certified
    scan winner (spec §3). Runs in regular CI (not marked slow)."""
    field = InferenceConfig.__dataclass_fields__["jitter_tau"]
    assert field.default == CERTIFIED_TAU
    for fn in (ParticleFilter.__init__, ModelComparison.__init__):
        params = inspect.signature(fn).parameters
        assert params["jitter_tau"].default == CERTIFIED_TAU
    runner = inspect.signature(run_multi_mass_2d).parameters
    assert runner["jitter_tau"].default == CERTIFIED_TAU
    assert runner["jitter_std"].default == CERTIFIED_FLOOR
```

(Move the existing `import pytest` / `run_multi_mass_2d` import lines into one import block at the top of the file.)
- [ ] **Step 6: Run the acceptance test as shipped** — `uv run pytest -m slow -v` → PASS (≥ 10/12 through the runner) and `uv run pytest tests/test_acceptance_multi_mass_2d.py -v` → the fast default-pinning test passes (proves the library defaults match the certified cell).
- [ ] **Step 7: Full gate** — `uv run pytest && uv run ruff format --check . && uv run ruff check .` → green
- [ ] **Step 8: Commit** — `git commit -am "Finalize annealed-jitter defaults from seed scan; certify holdout acceptance"`

---

### Task 8: Regenerate all committed demo artifacts

**Files:**
- Modify (binaries): `assets/*.gif`, `site/assets/*.gif`, `assets/demo_density.png`, `site/assets/demo_density.png` (check actual density filename with `ls assets/`)

**Interfaces:**
- Consumes: Tasks 1–7 all merged into the branch (artifacts must reflect final defaults).

- [ ] **Step 1: Regenerate** (each writes to `output/`; ~10–20 min total):

```bash
uv run demo-1d && uv run demo-2d && uv run demo-multi-mass \
  && uv run demo-multi-mass-2d && uv run demo-model-comparison && uv run demo-density
```

- [ ] **Step 2: Sanity-check the multi-mass-2D printout** — the final estimates printed by `demo-multi-mass-2d` (seed 11) must be near truth (−3, 2, 4, −1, 0.6, 0.4) within the pass tolerances; if seed 11 fails under the new defaults, pick the lowest-numbered tuning seed that passes, set `SEED` in the demo accordingly, note it in the commit message, and re-run.
- [ ] **Step 3: Copy and verify byte-equality**:

```bash
ARTIFACTS="demo_1d.gif demo_2d.gif demo_multi_mass.gif demo_multi_mass_2d.gif demo_model_comparison.gif demo_density.png"
for f in $ARTIFACTS; do
  cp "output/$f" assets/ && cp "output/$f" site/assets/ || exit 1
done
for f in $ARTIFACTS; do
  cmp "assets/$f" "site/assets/$f" || exit 1
done
echo "all artifacts copied and byte-identical"
```

Expected: the final echo line, exit 0. Explicit names only — `output/` is gitignored and may hold stale unrelated files; do not glob it. (Confirm the density filename first with `ls output/`; the demo writes `output/demo_density.png`.)

- [ ] **Step 4: Commit** — `git add assets site/assets scripts/demo_multi_mass_2d.py && git commit -m "Regenerate all demo artifacts under annealed-jitter defaults"` (the demo script is included in case Step 2 changed `SEED`).

---

### Task 9: Site text, full site re-render, someday-maybe closeout

**Files:**
- Modify: `site/method/the-particle-filter.qmd`, `docs/someday-maybe.md`; check `site/story/two-hidden-masses.qmd`, `site/story/into-the-plane.qmd`, `site/reproduce/*.qmd` for stale jitter phrasing

**Interfaces:**
- Consumes: Task 7's shipped defaults and measured numbers (fill them into the prose — no placeholders may survive).

- [ ] **Step 1: `site/method/the-particle-filter.qmd`** — the list intro at line ~39 reads "… randomizes the remainder. Two jitter modes:" — change "Two" to "Three". Then add a third bullet to the list:

```markdown
- **annealed** (the default) — axis-aligned Gaussian whose per-parameter
  scale starts at the initial particle cloud's spread (prior scale) and
  decays exponentially, with time constant `jitter_tau` observations,
  to the `jitter_std` floor. Early on the cloud can still escape a wrong
  mode; late in the run the jitter is as gentle as fixed jitter.
```

and rewrite the particle-impoverishment failure bullet (~line 108): the freeze under too-weak fixed jitter is why the jitter now anneals from prior scale — cite the shipped scan numbers (post-repair fixed baseline X/12 at the same floor vs annealed Y/12 tuning, Z/12 holdout; exact values from Task 7).

- [ ] **Step 2: Check the other pages** — `grep -rn -i "jitter" site --include="*.qmd"`; update `two-hidden-masses.qmd` line ~22 ("resampling jitter") and any `into-the-plane.qmd`/`reproduce/` phrasing only if it asserts fixed jitter or a stale default; leave physics prose alone.
- [ ] **Step 3: `docs/someday-maybe.md`** — replace the "Adaptive or annealed jitter" item body with a short done-note: shipped (date + spec path — the PR number doesn't exist until Task 10; don't reference it), pointer to `scripts/scan_multi_mass_2d.py`, `tests/test_acceptance_multi_mass_2d.py`, shipped defaults, and the measured baseline → holdout numbers. Keep the MCMC-rejuvenation item (still open).
- [ ] **Step 4: Re-render the whole site** — `cd site && uv run --frozen quarto render` → no errors. `site/_output/` is **gitignored**, so `git status` cannot tell you which rendered pages changed. Instead, list the pages that execute inference directly — `grep -rln "InferenceConfig\|build_particle_filter\|ModelComparison" site --include="*.qmd"` — and inspect each one's rendered HTML under `site/_output/` (open in a browser or read the printed estimate lines): every such page prints slightly different numbers now — confirm none shows a *wrong* result (estimates far from that page's stated truth).
- [ ] **Step 5: Full gate** — `uv run pytest && uv run ruff format --check . && uv run ruff check .` → green
- [ ] **Step 6: Commit** — `git commit -am "Update site and someday-maybe for the annealed-jitter default"`

---

### Task 10: Final verification, PR, and Codex review

- [ ] **Step 1:** `uv run pytest -v` (full fast suite) and `uv run pytest -m slow -v` (acceptance) → all green; `uv run ruff format --check . && uv run ruff check .` → clean.
- [ ] **Step 2:** Re-read the spec top to bottom; confirm every §1–§5 item maps to a landed commit. The one spec item that is *conditional* — the `InferenceConfig.jitter` sentinel fallback — lands only if Task 3 Step 5 triggered it.
- [ ] **Step 3:** Push the branch, open a PR titled "Annealed jitter: prior-scale-to-floor schedule as the library default" with a body summarizing: mechanism, support repair, animation fix, scan results table (baseline vs annealed, tuning + holdout), artifact regeneration, site updates.
- [ ] **Step 4:** Verify CI green on the PR (`gh pr checks`).
- [ ] **Step 5:** Codex xhigh PR review rounds per the repo protocol (review → triage → fix → re-review until "READY TO MERGE"), posting each round to the PR via `gh`. Codex–Claude agreement is merge approval.
