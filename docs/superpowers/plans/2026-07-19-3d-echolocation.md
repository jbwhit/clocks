# 3D Gravitational Echolocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A 3D exterior-mass demo (rotating-camera GIF), a resolution-vs-range study with certification protocol, and a "Gravitational Echolocation" site page — per spec `docs/superpowers/specs/2026-07-19-3d-echolocation-design.md`.

**Architecture:** A new shared scenario (`run_echolocation_3d` in `clocks._scenarios`) builds a 27-clock cubic lattice and a raw `ParticleFilter` with a mean-centered (differential) measurement model. New viz module `clocks._panels3d` supplies the hero-3D dashboard; a new `animate_echolocation` driver in `clocks._animate` rotates the camera per frame. Study helpers live in `clocks._echo_study` (importable by tests and the thin scan script). No changes to physics, inference, noise, or the public API.

**Tech Stack:** Python 3.12+, uv, numpy, matplotlib (Agg / pillow GIF writer), pytest, Quarto site.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-19-3d-echolocation-design.md` — consult on any ambiguity.
- **uv only**: `uv run pytest`, `uv run ruff …`; never bare `python`/`pip`.
- Local gate before every commit: `uv run ruff format --check .` AND `uv run ruff check .` AND `uv run pytest` (fast suite; slow tests only where a step says so).
- **No core library changes**: `physics.py`, `inference.py`, `noise.py`, `api.py`, `config.py`, `types.py`, `results.py` must be untouched. Everything is additive.
- Annealed jitter shipped defaults are used as-is: `jitter="annealed"`, `jitter_tau=15.0`, `jitter_std=0.02` floor.
- Seed protocol (spec §3a): tuning seeds 0–11, certification seeds 300–311 run exactly once. Never run certification seeds during development or tuning.
- Range unit is the head **circumradius** `R_head = sqrt(3)`; masses stay exterior with weak-field clearance `d_min ≥ 10 × M_true` (validated, not assumed).
- Commit messages: imperative, ≤72-char subject, end body with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Work on branch `echolocation-3d`; PR to `main` at the end (Task 12). First task creates the branch.

---

### Task 1: Scenario geometry — lattice, mass placement, validation

**Files:**
- Modify: `src/clocks/_scenarios.py` (append after `run_multi_mass_2d`)
- Test: `tests/test_scenarios.py` (append)

**Interfaces:**
- Consumes: `clocks.types.ClockArray`, `clocks.types.MassConfig` (existing).
- Produces (used by Tasks 2, 4, 5, 8, 11):
  - constants `ECHO_R_HEAD: float`, `ECHO_DIRECTION: NDArray` (unit 3-vector), `ECHO_M_TRUE = 0.15`, `ECHO_NOISE_STD = 0.005`, `ECHO_N_OBSERVATIONS = 80`, `ECHO_N_PARTICLES = 6000`, `ECHO_MASS_RANGE = (0.05, 2.0)`, `ECHO_MIN_RANGE_R = 2.0`, `ECHO_POSITION_HALFWIDTH = 16.0`, `ECHO_SWEEP_RANGES = (2.0, 2.6, 3.5, 4.6, 6.1, 8.0)`
  - `build_head_lattice() -> ClockArray`
  - `echo_mass_position(range_r: float) -> NDArray` (shape (3,))
  - `echo_mass_config(range_r: float, m_true: float = ECHO_M_TRUE) -> MassConfig`
  - `validate_echo_geometry(range_r: float, m_true: float, clock_array: ClockArray) -> None` (raises `ValueError`)

- [ ] **Step 1: Create the branch**

```bash
git checkout -b echolocation-3d
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_scenarios.py`:

```python
import pytest

from clocks._scenarios import (
    ECHO_DIRECTION,
    ECHO_M_TRUE,
    ECHO_MIN_RANGE_R,
    ECHO_R_HEAD,
    build_head_lattice,
    echo_mass_config,
    echo_mass_position,
    validate_echo_geometry,
)


class TestEchoGeometry:
    def test_lattice_is_3x3x3_cube(self) -> None:
        head = build_head_lattice()
        assert head.positions.shape == (27, 3)
        assert head.track_offset == 0.0
        # Every coordinate is exactly -1, 0, or 1; all 27 cells distinct.
        assert set(np.unique(head.positions)) == {-1.0, 0.0, 1.0}
        assert len({tuple(p) for p in head.positions}) == 27

    def test_circumradius_is_sqrt_3(self) -> None:
        head = build_head_lattice()
        radii = np.linalg.norm(head.positions, axis=1)
        assert np.isclose(radii.max(), ECHO_R_HEAD)
        assert np.isclose(ECHO_R_HEAD, np.sqrt(3.0))

    def test_direction_is_unit_and_off_axis(self) -> None:
        assert np.isclose(np.linalg.norm(ECHO_DIRECTION), 1.0)
        # No zero component (off-axis) and components distinct (off-diagonal).
        assert np.all(np.abs(ECHO_DIRECTION) > 0.1)
        assert len(set(np.round(np.abs(ECHO_DIRECTION), 6))) == 3

    def test_mass_position_at_requested_range(self) -> None:
        pos = echo_mass_position(3.0)
        assert pos.shape == (3,)
        assert np.isclose(np.linalg.norm(pos), 3.0 * ECHO_R_HEAD)
        config = echo_mass_config(3.0)
        assert config.positions.shape == (1, 3)
        assert np.isclose(config.masses[0], ECHO_M_TRUE)

    def test_validate_rejects_interior_range(self) -> None:
        head = build_head_lattice()
        with pytest.raises(ValueError, match="exterior"):
            validate_echo_geometry(ECHO_MIN_RANGE_R - 0.5, ECHO_M_TRUE, head)

    def test_validate_rejects_weak_field_violation(self) -> None:
        head = build_head_lattice()
        # Heavy mass at minimum range: d_min ~ 2.0 < 10 * 0.5.
        with pytest.raises(ValueError, match="weak-field"):
            validate_echo_geometry(ECHO_MIN_RANGE_R, 0.5, head)

    def test_validate_accepts_shipped_defaults(self) -> None:
        head = build_head_lattice()
        validate_echo_geometry(ECHO_MIN_RANGE_R, ECHO_M_TRUE, head)
```

(`numpy as np` is already imported at the top of `test_scenarios.py`; do not re-import.)

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_scenarios.py -v -k TestEchoGeometry`
Expected: FAIL with `ImportError: cannot import name 'ECHO_DIRECTION'`

- [ ] **Step 4: Implement**

Append to `src/clocks/_scenarios.py` (add `from itertools import product` to the imports block):

```python
# --- 3D echolocation scenario (spec 2026-07-19-3d-echolocation-design) ---

# The head: 3x3x3 cubic lattice, spacing 1.0, centered on the origin.
# Circumradius (center to corner clocks) — the unit for range_r.
ECHO_R_HEAD = float(np.sqrt(3.0))
# Fixed exterior direction: exact unit vector, off-axis and off-diagonal
# so no projection or lattice symmetry hides the mass.
ECHO_DIRECTION = np.array([2.0, 3.0, 6.0]) / 7.0
ECHO_M_TRUE = 0.15
ECHO_NOISE_STD = 0.005
ECHO_N_OBSERVATIONS = 80
ECHO_N_PARTICLES = 6000
ECHO_MASS_RANGE = (0.05, 2.0)
ECHO_MIN_RANGE_R = 2.0  # circumradii; exterior means exterior, with clearance
ECHO_POSITION_HALFWIDTH = 16.0  # prior box covers max swept range 8*R_head~13.9
ECHO_SWEEP_RANGES = (2.0, 2.6, 3.5, 4.6, 6.1, 8.0)  # log-ish, circumradii


def build_head_lattice() -> ClockArray:
    """The 27-clock head: 3x3x3 grid over {-1, 0, 1}^3."""
    grid = (-1.0, 0.0, 1.0)
    positions = np.array(list(product(grid, grid, grid)))
    return ClockArray(positions=positions, track_offset=0.0)


def echo_mass_position(range_r: float) -> NDArray[np.floating]:
    """Exterior mass position at range_r circumradii along ECHO_DIRECTION."""
    return ECHO_DIRECTION * range_r * ECHO_R_HEAD


def echo_mass_config(range_r: float, m_true: float = ECHO_M_TRUE) -> MassConfig:
    return MassConfig(
        positions=echo_mass_position(range_r).reshape(1, 3),
        masses=np.array([m_true]),
    )


def validate_echo_geometry(
    range_r: float, m_true: float, clock_array: ClockArray
) -> None:
    """Fail fast on interior masses and weak-field violations (spec section 1)."""
    if range_r < ECHO_MIN_RANGE_R:
        raise ValueError(
            f"range_r={range_r} is below the exterior minimum "
            f"{ECHO_MIN_RANGE_R} circumradii: exterior means exterior"
        )
    d_min = float(
        np.min(
            np.linalg.norm(clock_array.positions - echo_mass_position(range_r), axis=1)
        )
    )
    if d_min < 10.0 * m_true:
        raise ValueError(
            f"weak-field constraint violated: min clock-mass distance "
            f"{d_min:.3f} < 10*M_true={10.0 * m_true:.3f} "
            f"(range_r={range_r}, M_true={m_true})"
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_scenarios.py -v`
Expected: all PASS (existing tests plus the new class)

- [ ] **Step 6: Gate and commit**

```bash
uv run ruff format --check . && uv run ruff check . && uv run pytest
git add src/clocks/_scenarios.py tests/test_scenarios.py
git commit -m "Add echolocation head lattice, mass placement, and geometry validation"
```

---

### Task 2: Centered measurement model, filter builder, and scenario runner

**Files:**
- Modify: `src/clocks/_scenarios.py` (append)
- Test: `tests/test_scenarios.py` (append)

**Interfaces:**
- Consumes (Task 1): `build_head_lattice`, `echo_mass_config`, `echo_mass_position`, `validate_echo_geometry`, `ECHO_*` constants.
- Consumes (existing): `ParticleFilter` (raw constructor with `forward_model_batch`, `log_prior`, `support_bounds`), `simulate`, `SimulationConfig`, `NoiseConfig`, `clock_rates`, `clock_rates_batch`, `Observation`, `SimulationResult`.
- Produces (used by Tasks 4, 5, 8, 10):
  - `ECHO_PASS_POS_TOL = 1.0`, `ECHO_PASS_MASS_TOL = 0.075` (provisional; frozen in Task 9)
  - `class EchoRunResult(TypedDict)`: `seed: int`, `range_r: float`, `passed: bool`, `mean: NDArray`, `std: NDArray`, `position_error: float`, `mass_error: float`, `pos_std: float`, `mass_std: float`, `covered_3sigma: bool`, `residual_over_noise: float`
  - `make_echo_observations(seed, range_r, *, n_observations=ECHO_N_OBSERVATIONS, noise_std=ECHO_NOISE_STD) -> tuple[SimulationResult, list[Observation]]` (result, centered observations)
  - `build_echolocation_filter(seed, *, n_particles=ECHO_N_PARTICLES, noise_std=ECHO_NOISE_STD) -> ParticleFilter`
  - `run_echolocation_3d(seed, range_r, *, n_particles=ECHO_N_PARTICLES, n_observations=ECHO_N_OBSERVATIONS) -> EchoRunResult`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scenarios.py` (extend the existing import from `clocks._scenarios` with the new names):

```python
from clocks._scenarios import (  # noqa: E501 — merge into the existing import
    EchoRunResult,
    build_echolocation_filter,
    make_echo_observations,
    run_echolocation_3d,
)


class TestEchoMeasurementModel:
    def test_centered_observations_have_zero_mean(self) -> None:
        _, centered = make_echo_observations(seed=0, range_r=2.0)
        for obs in centered:
            assert np.isclose(obs.rates.mean(), 0.0, atol=1e-12)

    def test_centering_removes_constant_offset(self) -> None:
        # A uniform offset (the M/R common mode) must vanish under centering.
        rates = np.full(27, 0.997)
        assert np.allclose(rates - rates.mean(), 0.0)

    def test_forward_model_batch_rows_are_centered(self) -> None:
        pf = build_echolocation_filter(seed=0, n_particles=50)
        assert pf.forward_model_batch is not None
        predicted = pf.forward_model_batch(pf.state.particles)
        assert predicted.shape == (50, 27)
        assert np.allclose(predicted.mean(axis=1), 0.0, atol=1e-12)

    def test_scalar_forward_model_matches_batch(self) -> None:
        pf = build_echolocation_filter(seed=1, n_particles=8)
        assert pf.forward_model_batch is not None
        batch = pf.forward_model_batch(pf.state.particles)
        for i in range(8):
            single = pf.forward_model(pf.state.particles[i])
            assert np.allclose(single, batch[i])

    def test_prior_support_bounds_match_log_prior(self) -> None:
        pf = build_echolocation_filter(seed=0, n_particles=100)
        assert pf.support_bounds is not None
        assert pf.log_prior is not None
        lower, upper = pf.support_bounds
        # In-support particles get 0; nudged-outside particles get -inf.
        inside = pf.state.particles
        assert np.all(pf.log_prior(inside) == 0.0)
        outside = inside.copy()
        outside[:, 3] = upper[3] + 0.1
        assert np.all(np.isneginf(pf.log_prior(outside)))


class TestEchoRunResult:
    def test_small_run_populates_fields(self) -> None:
        result: EchoRunResult = run_echolocation_3d(
            seed=0, range_r=2.0, n_particles=400, n_observations=15
        )
        assert result["seed"] == 0
        assert result["range_r"] == 2.0
        assert result["mean"].shape == (4,)
        assert result["std"].shape == (4,)
        assert result["position_error"] >= 0.0
        assert result["mass_error"] >= 0.0
        assert result["pos_std"] > 0.0
        assert result["mass_std"] > 0.0
        assert isinstance(result["passed"], bool)
        assert isinstance(result["covered_3sigma"], bool)
        assert result["residual_over_noise"] >= 0.0

    def test_run_rejects_invalid_geometry(self) -> None:
        with pytest.raises(ValueError, match="exterior"):
            run_echolocation_3d(seed=0, range_r=1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_scenarios.py -v -k "TestEchoMeasurementModel or TestEchoRunResult"`
Expected: FAIL with `ImportError: cannot import name 'EchoRunResult'`

- [ ] **Step 3: Implement**

Append to `src/clocks/_scenarios.py`. Extend the existing imports: add `clock_rates_batch` to the `clocks.physics` import, `ParticleFilter` from `clocks.inference`, `Observation` from `clocks.types`, `SimulationResult` from `clocks.results`, and `Callable` from `collections.abc`.

```python
# Provisional pass tolerances at the closest swept range; frozen after the
# tuning sweep (spec section 3a) — the tuning task records final values.
ECHO_PASS_POS_TOL = 1.0
ECHO_PASS_MASS_TOL = 0.075


class EchoRunResult(TypedDict):
    """One echolocation run: gate result plus study metrics (spec section 1)."""

    seed: int
    range_r: float
    passed: bool
    mean: NDArray[np.floating]
    std: NDArray[np.floating]
    position_error: float
    mass_error: float
    pos_std: float
    mass_std: float
    covered_3sigma: bool
    residual_over_noise: float


def _center(rates: NDArray[np.floating]) -> NDArray[np.floating]:
    """Remove the across-clock mean: the head has no external reference."""
    return rates - rates.mean()


def _make_echo_forward_models(
    clock_array: ClockArray,
) -> tuple[
    Callable[[NDArray[np.floating]], NDArray[np.floating]],
    Callable[[NDArray[np.floating]], NDArray[np.floating]],
]:
    """Scalar and batch forward models emitting mean-centered rates.

    Centering the noisy data correlates its noise (covariance
    sigma^2 (I - 11^T/N)); we keep the iid Gaussian likelihood because on
    centered residuals its quadratic form matches the projected Gaussian
    up to a parameter-independent constant — particle weights are exact,
    only the (unused) log-evidence normalization differs.
    """

    def forward(params: NDArray[np.floating]) -> NDArray[np.floating]:
        rates = clock_rates(
            MassConfig(positions=params[:3].reshape(1, 3), masses=params[3:4]),
            clock_array,
        )
        return _center(rates)

    def forward_batch(particles: NDArray[np.floating]) -> NDArray[np.floating]:
        rates = clock_rates_batch(particles[:, :3], particles[:, 3], clock_array)
        return rates - rates.mean(axis=1, keepdims=True)

    return forward, forward_batch


def make_echo_observations(
    seed: int,
    range_r: float,
    *,
    n_observations: int = ECHO_N_OBSERVATIONS,
    noise_std: float = ECHO_NOISE_STD,
) -> tuple[SimulationResult, list[Observation]]:
    """Simulate absolute rates, then center each observation across clocks."""
    clock_array = build_head_lattice()
    sim = simulate(
        SimulationConfig(
            clock_array=clock_array,
            ground_truth=echo_mass_config(range_r),
            noise=NoiseConfig(observation_std=noise_std),
            n_observations=n_observations,
            seed=seed,
        )
    )
    centered = [
        Observation(rates=_center(obs.rates), time=obs.time)
        for obs in sim.observations
    ]
    return sim, centered


def build_echolocation_filter(
    seed: int,
    *,
    n_particles: int = ECHO_N_PARTICLES,
    noise_std: float = ECHO_NOISE_STD,
) -> ParticleFilter:
    """Raw ParticleFilter for the (x, y, z, M) exterior-mass problem.

    Built directly (not via InferenceConfig) because the public API cannot
    express a centered measurement model; support_bounds are identical to
    the log-prior support so reflected annealed jitter always lands inside
    it (spec section 1).
    """
    clock_array = build_head_lattice()
    hw = ECHO_POSITION_HALFWIDTH
    lower = np.array([-hw, -hw, -hw, ECHO_MASS_RANGE[0]])
    upper = np.array([hw, hw, hw, ECHO_MASS_RANGE[1]])

    def prior_sampler(rng: np.random.Generator, n: int) -> NDArray[np.floating]:
        return rng.uniform(lower, upper, size=(n, 4))

    def log_prior(particles: NDArray[np.floating]) -> NDArray[np.floating]:
        lp = np.zeros(particles.shape[0])
        outside = np.any((particles < lower) | (particles > upper), axis=1)
        lp[outside] = -np.inf
        return lp

    forward, forward_batch = _make_echo_forward_models(clock_array)
    return ParticleFilter(
        n_particles=n_particles,
        prior_sampler=prior_sampler,
        forward_model=forward,
        noise_std=noise_std,
        jitter_std=0.02,
        rng=np.random.default_rng(seed),
        forward_model_batch=forward_batch,
        jitter="annealed",
        jitter_tau=15.0,
        log_prior=log_prior,
        support_bounds=(lower, upper),
    )


def run_echolocation_3d(
    seed: int,
    range_r: float,
    *,
    n_particles: int = ECHO_N_PARTICLES,
    n_observations: int = ECHO_N_OBSERVATIONS,
) -> EchoRunResult:
    """One end-to-end echolocation run at a given range (in circumradii)."""
    clock_array = build_head_lattice()
    validate_echo_geometry(range_r, ECHO_M_TRUE, clock_array)
    sim, centered_obs = make_echo_observations(
        seed, range_r, n_observations=n_observations
    )
    pf = build_echolocation_filter(seed, n_particles=n_particles)
    for obs in centered_obs:
        pf.update(obs)

    est = pf.estimate()
    mean, std = est["mean"], est["std"]
    truth = np.append(echo_mass_position(range_r), ECHO_M_TRUE)
    error = np.abs(mean - truth)
    position_error = float(np.linalg.norm(mean[:3] - truth[:3]))
    mass_error = float(error[3])
    predicted_centered = _make_echo_forward_models(clock_array)[0](mean)
    residual = float(
        np.max(np.abs(predicted_centered - _center(sim.true_rates))) / ECHO_NOISE_STD
    )
    return EchoRunResult(
        seed=seed,
        range_r=range_r,
        passed=bool(
            position_error <= ECHO_PASS_POS_TOL and mass_error <= ECHO_PASS_MASS_TOL
        ),
        mean=mean,
        std=std,
        position_error=position_error,
        mass_error=mass_error,
        pos_std=float(np.linalg.norm(std[:3])),
        mass_std=float(std[3]),
        covered_3sigma=bool(np.all(error <= 3.0 * std)),
        residual_over_noise=residual,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_scenarios.py -v`
Expected: all PASS (the small-N run takes a few seconds)

- [ ] **Step 5: Gate and commit**

```bash
uv run ruff format --check . && uv run ruff check . && uv run pytest
git add src/clocks/_scenarios.py tests/test_scenarios.py
git commit -m "Add centered echolocation measurement model and scenario runner"
```

---

### Task 3: 3D core coverage tests (physics batch equivalence + API recovery)

**Files:**
- Test: `tests/test_physics.py` (append)
- Test: `tests/test_api.py` (append)

**Interfaces:**
- Consumes (existing): `clock_rates`, `clock_rates_batch`, `MassConfig`, `ClockArray`, `simulate`, `infer`, `SimulationConfig`, `InferenceConfig`, `NoiseConfig`, `PriorConfig`, `InferenceResult`.
- Produces: nothing downstream — these pin that the dimension-agnostic core actually works in 3D (spec §5). They should pass without production changes; if one fails, that is a genuine core bug: stop and fix it in core with its own commit.

- [ ] **Step 1: Write the physics test**

Append to `tests/test_physics.py` (reuse the file's existing imports; add any missing name to them rather than re-importing):

```python
class TestBatchEquivalence3D:
    def test_clock_rates_batch_matches_loop_in_3d(self) -> None:
        rng = np.random.default_rng(7)
        clock_array = ClockArray(
            positions=rng.uniform(-2, 2, size=(9, 3)), track_offset=0.0
        )
        mass_positions = rng.uniform(3, 8, size=(20, 3))
        masses = rng.uniform(0.05, 0.5, size=20)
        batch = clock_rates_batch(mass_positions, masses, clock_array)
        assert batch.shape == (20, 9)
        for i in range(20):
            single = clock_rates(
                MassConfig(
                    positions=mass_positions[i].reshape(1, 3),
                    masses=masses[i : i + 1],
                ),
                clock_array,
            )
            assert np.allclose(batch[i], single)
```

- [ ] **Step 2: Write the API test**

Append to `tests/test_api.py` (again extending existing imports as needed — in particular the file currently imports only `SimulationResult` from `clocks.results`, so extend that line):

```python
from clocks.results import InferenceResult, SimulationResult  # replaces the existing import line


class TestInference3D:
    def test_single_mass_3d_recovery(self) -> None:
        """(x, y, z, M) inference works end-to-end through the public API."""
        rng = np.random.default_rng(3)
        clock_array = ClockArray(
            positions=rng.uniform(-3, 3, size=(12, 3)), track_offset=0.0
        )
        truth = MassConfig(
            positions=np.array([[1.0, -1.5, 0.5]]), masses=np.array([0.5])
        )
        sim = simulate(
            SimulationConfig(
                clock_array=clock_array,
                ground_truth=truth,
                noise=NoiseConfig(observation_std=0.005),
                n_observations=40,
                seed=3,
            )
        )
        result = infer(
            sim.observations,
            InferenceConfig(
                clock_array=clock_array,
                noise=NoiseConfig(observation_std=0.005),
                prior=PriorConfig(position_range=(-5.0, 5.0), mass_range=(0.1, 2.0)),
                n_particles=2000,
                n_masses=1,
                seed=3,
            ),
        )
        assert isinstance(result, InferenceResult)
        expected = np.array([1.0, -1.5, 0.5, 0.5])
        assert np.all(np.abs(result.posterior_mean - expected) < 0.5)
```

- [ ] **Step 3: Run the new tests**

Run: `uv run pytest tests/test_physics.py::TestBatchEquivalence3D tests/test_api.py::TestInference3D -v`
Expected: PASS. (These are characterization tests of existing behavior — if either FAILS, a genuine 3D core bug exists: diagnose it, fix core in a separate commit with this test as the regression pin, then continue.)

- [ ] **Step 4: Gate and commit**

```bash
uv run ruff format --check . && uv run ruff check . && uv run pytest
git add tests/test_physics.py tests/test_api.py
git commit -m "Pin 3D batch-forward equivalence and public-API 3D recovery"
```

---

### Task 4: Study helpers — `clocks._echo_study`

**Files:**
- Create: `src/clocks/_echo_study.py`
- Test: `tests/test_echo_study.py`

**Interfaces:**
- Consumes (Tasks 1–2): `EchoRunResult`, `build_head_lattice`, `echo_mass_config`, `ECHO_M_TRUE`, `ECHO_NOISE_STD`; existing `clock_rates`.
- Produces (used by Tasks 5, 10, 11):
  - `snr_table(ranges: Sequence[float], *, m_true=ECHO_M_TRUE, noise_std=ECHO_NOISE_STD) -> list[dict]` — dicts with keys `range_r`, `signal`, `signal_over_noise`
  - `save_study(path: Path, seed_block: int, results: list[EchoRunResult]) -> None`
  - `load_study(path: Path) -> dict` — `{"seed_block": int, "results": [dict, ...]}`
  - `summarize(results: list[dict]) -> list[dict]` — per range: `range_r`, `n_pass`, `n_runs`, `med_position_error`, `med_mass_error`, `med_pos_std`, `med_mass_std`
  - `write_summary_figure(study: dict, png_path: Path) -> None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_echo_study.py`:

```python
"""Tests for the echolocation study helpers (pure, no inference runs)."""

from pathlib import Path

import matplotlib
import numpy as np

from clocks._echo_study import (
    load_study,
    save_study,
    snr_table,
    summarize,
    write_summary_figure,
)
from clocks._scenarios import ECHO_NOISE_STD, EchoRunResult

matplotlib.use("Agg")


def _fake_result(seed: int, range_r: float, err: float) -> EchoRunResult:
    return EchoRunResult(
        seed=seed,
        range_r=range_r,
        passed=err < 1.0,
        mean=np.array([1.0, 1.5, 3.0, 0.15]),
        std=np.array([0.2, 0.2, 0.4, 0.03]),
        position_error=err,
        mass_error=err / 10.0,
        pos_std=0.5 * err + 0.1,
        mass_std=0.03,
        covered_3sigma=True,
        residual_over_noise=1.2,
    )


def test_snr_table_decreases_with_range() -> None:
    table = snr_table([2.0, 4.0, 8.0])
    assert [row["range_r"] for row in table] == [2.0, 4.0, 8.0]
    signals = [row["signal"] for row in table]
    assert signals[0] > signals[1] > signals[2] > 0.0
    assert np.isclose(
        table[0]["signal_over_noise"], table[0]["signal"] / ECHO_NOISE_STD
    )


def test_save_load_round_trip(tmp_path: Path) -> None:
    results = [_fake_result(s, 2.0, 0.3) for s in range(3)]
    path = tmp_path / "study.json"
    save_study(path, seed_block=0, results=results)
    study = load_study(path)
    assert study["seed_block"] == 0
    assert len(study["results"]) == 3
    loaded = study["results"][0]
    assert loaded["position_error"] == 0.3
    assert loaded["mean"] == [1.0, 1.5, 3.0, 0.15]  # arrays become lists


def test_summarize_medians_per_range(tmp_path: Path) -> None:
    results = [_fake_result(s, 2.0, e) for s, e in enumerate([0.1, 0.3, 0.5])]
    results += [_fake_result(s, 8.0, e) for s, e in enumerate([2.0, 4.0, 6.0])]
    path = tmp_path / "study.json"
    save_study(path, seed_block=0, results=results)
    summary = summarize(load_study(path)["results"])
    assert [row["range_r"] for row in summary] == [2.0, 8.0]
    assert summary[0]["med_position_error"] == 0.3
    assert summary[0]["n_pass"] == 3
    assert summary[1]["med_position_error"] == 4.0
    assert summary[1]["n_pass"] == 0


def test_write_summary_figure_creates_png(tmp_path: Path) -> None:
    results = [_fake_result(s, r, 0.2 * r) for r in (2.0, 4.0) for s in range(3)]
    json_path = tmp_path / "study.json"
    save_study(json_path, seed_block=0, results=results)
    png_path = tmp_path / "study.png"
    write_summary_figure(load_study(json_path), png_path)
    assert png_path.exists()
    assert png_path.stat().st_size > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_echo_study.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clocks._echo_study'`

- [ ] **Step 3: Implement**

Create `src/clocks/_echo_study.py`:

```python
"""Reporting helpers for the echolocation resolution-vs-range study.

Lives in the package (not scripts/) so tests and the Quarto page can
import it; the scan script stays a thin CLI (same reasoning as
clocks._scenarios).
"""

import json
import statistics
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from clocks._scenarios import (
    ECHO_M_TRUE,
    ECHO_NOISE_STD,
    EchoRunResult,
    build_head_lattice,
    echo_mass_config,
)
from clocks.physics import clock_rates


def snr_table(
    ranges: Sequence[float],
    *,
    m_true: float = ECHO_M_TRUE,
    noise_std: float = ECHO_NOISE_STD,
) -> list[dict]:
    """Noise-free centered signal magnitude per range (spec section 3).

    Pure forward-model evaluation — the same computation the site page's
    falloff cell shows.
    """
    head = build_head_lattice()
    table = []
    for range_r in ranges:
        rates = clock_rates(echo_mass_config(range_r, m_true), head)
        signal = float(np.max(np.abs(rates - rates.mean())))
        table.append(
            {
                "range_r": float(range_r),
                "signal": signal,
                "signal_over_noise": signal / noise_std,
            }
        )
    return table


def save_study(path: Path, seed_block: int, results: list[EchoRunResult]) -> None:
    """Write sweep results to JSON (numpy arrays become lists)."""
    serializable = [
        {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in result.items()
        }
        for result in results
    ]
    path.write_text(
        json.dumps({"seed_block": seed_block, "results": serializable}, indent=2)
    )


def load_study(path: Path) -> dict:
    return json.loads(path.read_text())


def summarize(results: list[dict]) -> list[dict]:
    """Per-range medians and pass counts, sorted by range."""
    by_range: dict[float, list[dict]] = {}
    for result in results:
        by_range.setdefault(result["range_r"], []).append(result)
    summary = []
    for range_r in sorted(by_range):
        cell = by_range[range_r]
        summary.append(
            {
                "range_r": range_r,
                "n_pass": sum(r["passed"] for r in cell),
                "n_runs": len(cell),
                "med_position_error": statistics.median(
                    r["position_error"] for r in cell
                ),
                "med_mass_error": statistics.median(r["mass_error"] for r in cell),
                "med_pos_std": statistics.median(r["pos_std"] for r in cell),
                "med_mass_std": statistics.median(r["mass_std"] for r in cell),
            }
        )
    return summary


def write_summary_figure(study: dict, png_path: Path) -> None:
    """Two aligned subplots: position and mass error vs range, with the
    filter's own claimed uncertainty (posterior std) as dashed medians."""
    results = study["results"]
    summary = summarize(results)
    ranges = [row["range_r"] for row in summary]

    fig, (ax_pos, ax_mass) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True, constrained_layout=True
    )
    for result in results:
        ax_pos.plot(
            result["range_r"], result["position_error"],
            "o", color="steelblue", alpha=0.35, markersize=4,
        )
        ax_mass.plot(
            result["range_r"], result["mass_error"],
            "o", color="steelblue", alpha=0.35, markersize=4,
        )
    ax_pos.plot(
        ranges, [r["med_position_error"] for r in summary],
        "-o", color="tab:blue", label="median error",
    )
    ax_pos.plot(
        ranges, [r["med_pos_std"] for r in summary],
        "--s", color="tab:orange", label="median posterior std",
    )
    ax_mass.plot(
        ranges, [r["med_mass_error"] for r in summary],
        "-o", color="tab:blue", label="median error",
    )
    ax_mass.plot(
        ranges, [r["med_mass_std"] for r in summary],
        "--s", color="tab:orange", label="median posterior std",
    )
    ax_pos.set_yscale("log")
    ax_mass.set_yscale("log")
    ax_pos.set_ylabel("position error")
    ax_mass.set_ylabel("mass error")
    ax_mass.set_xlabel("range (circumradii)")
    ax_pos.legend(fontsize=8)
    ax_mass.legend(fontsize=8)
    ax_pos.set_title("Echolocation resolution vs range")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_echo_study.py -v`
Expected: PASS

- [ ] **Step 5: Gate and commit**

```bash
uv run ruff format --check . && uv run ruff check . && uv run pytest
git add src/clocks/_echo_study.py tests/test_echo_study.py
git commit -m "Add echolocation study helpers: SNR table, JSON IO, summary figure"
```

---

### Task 5: Scan harness — `scripts/scan_echolocation_range.py`

**Files:**
- Create: `scripts/scan_echolocation_range.py`

**Interfaces:**
- Consumes (Tasks 1, 2, 4): `run_echolocation_3d`, `ECHO_SWEEP_RANGES`, `EchoRunResult`, `snr_table`, `save_study`, `load_study`, `write_summary_figure`, `summarize`.
- Produces: CLI producing `output/echolocation_range_study.json` and `output/echolocation_range_study.png`; `--seed-block N` selects seeds N…N+11 (default 0 = tuning; 300 = certification, run exactly once in Task 10).

- [ ] **Step 1: Write the script**

Create `scripts/scan_echolocation_range.py`:

```python
"""Resolution-vs-range sweep for the 3D echolocation scenario.

Runs the shared scenario over a range grid x 12 seeds and writes the
study JSON + summary figure. See
docs/superpowers/specs/2026-07-19-3d-echolocation-design.md section 3.

Usage:
    uv run scripts/scan_echolocation_range.py                  # tuning seeds 0-11
    uv run scripts/scan_echolocation_range.py --seed-block 300 # initial certification block (300, 400, ... per spec 3a)
    uv run scripts/scan_echolocation_range.py --figure-only    # re-render PNG
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

from clocks._echo_study import (
    load_study,
    save_study,
    snr_table,
    summarize,
    write_summary_figure,
)
from clocks._scenarios import (
    ECHO_M_TRUE,
    ECHO_SWEEP_RANGES,
    EchoRunResult,
    build_head_lattice,
    run_echolocation_3d,
    validate_echo_geometry,
)

JSON_PATH = Path("output/echolocation_range_study.json")
PNG_PATH = Path("output/echolocation_range_study.png")


def _run(job: tuple[int, float]) -> EchoRunResult:
    seed, range_r = job
    return run_echolocation_3d(seed, range_r)


def _print_snr_table(ranges: list[float]) -> None:
    print(f"{'range':>7} {'signal':>10} {'signal/noise':>13}")
    for row in snr_table(ranges):
        print(
            f"{row['range_r']:>7g} {row['signal']:>10.2e}"
            f" {row['signal_over_noise']:>13.2f}"
        )


def _print_summary(results: list[dict]) -> None:
    header = (
        f"{'range':>7} {'pass':>6} {'med pos err':>12} {'med M err':>10}"
        f" {'med pos std':>12} {'med M std':>10}"
    )
    print(header)
    for row in summarize(results):
        print(
            f"{row['range_r']:>7g} {row['n_pass']:>4}/{row['n_runs']}"
            f" {row['med_position_error']:>12.3f} {row['med_mass_error']:>10.4f}"
            f" {row['med_pos_std']:>12.3f} {row['med_mass_std']:>10.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ranges", type=float, nargs="+", default=list(ECHO_SWEEP_RANGES)
    )
    parser.add_argument(
        "--seed-block",
        type=int,
        default=0,
        help=(
            "first seed of the 12-seed block: 0 = tuning; certification "
            "blocks are 300, 400, ... (spec section 3a; the Status history "
            "records which block certified)"
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--per-run", action="store_true")
    parser.add_argument(
        "--figure-only",
        action="store_true",
        help="re-render the PNG from the existing JSON without sweeping",
    )
    parser.add_argument(
        "--snr-only",
        action="store_true",
        help="print the SNR sanity table and exit without sweeping",
    )
    args = parser.parse_args()

    if args.figure_only:
        write_summary_figure(load_study(JSON_PATH), PNG_PATH)
        print(f"Figure written to {PNG_PATH}")
        return

    # Validate at the boundary (spec sections 3a and 5).
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.seed_block != 0 and (args.seed_block < 300 or args.seed_block % 100 != 0):
        parser.error(
            "--seed-block must be 0 (tuning) or a certification block "
            "(300, 400, ...); see spec section 3a"
        )
    head = build_head_lattice()
    for range_r in args.ranges:
        validate_echo_geometry(range_r, ECHO_M_TRUE, head)
    if args.seed_block >= 300:
        print(
            f"CERTIFICATION RUN (seed block {args.seed_block}): "
            "run exactly once; results are final."
        )

    print("Noise-free centered signal vs range (SNR sanity gate):")
    _print_snr_table(args.ranges)
    if args.snr_only:
        return

    seeds = range(args.seed_block, args.seed_block + 12)
    jobs = [(seed, range_r) for range_r in args.ranges for seed in seeds]
    with Pool(args.workers) as pool:
        results = pool.map(_run, jobs)

    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_study(JSON_PATH, args.seed_block, results)
    study = load_study(JSON_PATH)
    write_summary_figure(study, PNG_PATH)

    print(f"\nSweep on seed block {args.seed_block} ({len(jobs)} runs):")
    _print_summary(study["results"])
    if args.per_run:
        for r in sorted(study["results"], key=lambda r: (r["range_r"], r["seed"])):
            print(
                f"    range {r['range_r']:>4g} seed {r['seed']:>3}"
                f" pass={int(r['passed'])} pos_err={r['position_error']:.3f}"
                f" M_err={r['mass_error']:.4f} pos_std={r['pos_std']:.3f}"
                f" 3sig={int(r['covered_3sigma'])}"
                f" resid/noise={r['residual_over_noise']:.1f}"
            )
    print(f"\nWrote {JSON_PATH} and {PNG_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the CLI cheaply**

Run: `uv run scripts/scan_echolocation_range.py --ranges 2.0 --workers 4 --seed-block 0`
Expected: SNR table prints; 12 runs complete (a few minutes); JSON + PNG appear under `output/`; summary table prints. This is tuning-seed usage, which is unrestricted.

- [ ] **Step 3: Gate and commit**

```bash
uv run ruff format --check . && uv run ruff check . && uv run pytest
git add scripts/scan_echolocation_range.py
git commit -m "Add echolocation range-sweep harness with SNR gate and seed blocks"
```

---

### Task 6: 3D dashboard panels — `clocks._panels3d`

**Files:**
- Create: `src/clocks/_panels3d.py`
- Modify: `src/clocks/viz.py` (facade exports)
- Test: `tests/test_viz.py` (append)

**Interfaces:**
- Consumes (existing): `ClockArray`, `MassConfig`, `ParticleState`.
- Produces (used by Task 7):
  - `create_echolocation_dashboard(figsize=(14.0, 8.0)) -> tuple[Figure, dict[str, Axes]]` with keys `"scene"` (3D), `"history"`, `"mass"`, `"rates"`
  - `plot_scene_3d(ax, clock_array, mass_config, particle_state, *, azim: float, elev: float = 18.0, max_particles: int = 1500) -> None`
  - `plot_centered_rates(ax, observed: NDArray, predicted_centered: NDArray) -> None` — observed vs the filter's *current prediction* (spec §2), both centered
  - `_ECHO_COLORS = ["tab:blue", "tab:green", "tab:purple", "tab:orange"]`, `_ECHO_LABELS = ["x", "y", "z", "M"]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_viz.py` (extend the existing `clocks.viz` import with the three new names):

```python
# -- Echolocation 3D panels --


@pytest.fixture()
def head_state() -> ParticleState:
    rng = np.random.default_rng(0)
    particles = np.column_stack(
        [rng.uniform(-5, 5, size=(200, 3)), rng.uniform(0.05, 2.0, size=(200, 1))]
    )
    return ParticleState(
        particles=particles,
        weights=np.ones(200) / 200,
        observations_seen=5,
    )


class TestEcholocationDashboard:
    def test_dashboard_has_expected_axes(self) -> None:
        fig, axes = create_echolocation_dashboard()
        assert set(axes) == {"scene", "history", "mass", "rates"}
        assert axes["scene"].name == "3d"
        plt.close(fig)

    def test_scene_renders_without_error(self, head_state: ParticleState) -> None:
        from clocks._scenarios import build_head_lattice, echo_mass_config

        fig, axes = create_echolocation_dashboard()
        plot_scene_3d(
            axes["scene"],
            build_head_lattice(),
            echo_mass_config(4.0),
            head_state,
            azim=30.0,
        )
        assert len(axes["scene"].collections) > 0
        plt.close(fig)

    def test_centered_rates_panel(self) -> None:
        fig, axes = create_echolocation_dashboard()
        rng = np.random.default_rng(1)
        predicted_centered = rng.normal(0, 0.005, 27)
        observed = predicted_centered + rng.normal(0, 0.005, 27)
        plot_centered_rates(axes["rates"], observed, predicted_centered)
        assert len(axes["rates"].patches) == 54  # two bar sets, 27 clocks each
        plt.close(fig)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_viz.py -v -k Echolocation`
Expected: FAIL with `ImportError: cannot import name 'create_echolocation_dashboard'`

- [ ] **Step 3: Implement**

Create `src/clocks/_panels3d.py`:

```python
"""3D plotting primitives for the echolocation dashboard (hero layout)."""

from itertools import combinations, product

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from clocks.types import ClockArray, MassConfig, ParticleState

# Parameter labels/colors for the (x, y, z, M) convergence trace.
_ECHO_COLORS = ["tab:blue", "tab:green", "tab:purple", "tab:orange"]
_ECHO_LABELS = ["x", "y", "z", "M"]


def create_echolocation_dashboard(
    figsize: tuple[float, float] = (14.0, 8.0),
) -> tuple[Figure, dict[str, Axes]]:
    """Hero layout: 3D scene at ~2/3 width, three stacked diagnostics right.

    Returns (fig, axes) with keys: 'scene' (3D), 'history', 'mass', 'rates'.
    """
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(3, 3)
    axes = {
        "scene": fig.add_subplot(gs[:, :2], projection="3d"),
        "history": fig.add_subplot(gs[0, 2]),
        "mass": fig.add_subplot(gs[1, 2]),
        "rates": fig.add_subplot(gs[2, 2]),
    }
    return fig, axes


def _draw_head_wireframe(ax: Axes, half_width: float = 1.0) -> None:
    """The 12 edges of the head cube, so the lattice reads as an object."""
    corners = np.array(list(product((-half_width, half_width), repeat=3)))
    for start, end in combinations(corners, 2):
        if np.count_nonzero(start != end) == 1:  # edge: differs in one axis
            ax.plot3D(*zip(start, end), color="gray", alpha=0.4, linewidth=0.8)


def plot_scene_3d(
    ax: Axes,
    clock_array: ClockArray,
    mass_config: MassConfig,
    particle_state: ParticleState,
    *,
    azim: float,
    elev: float = 18.0,
    max_particles: int = 1500,
) -> None:
    """The hero panel: head lattice, exterior mass, particle cloud."""
    ax.clear()
    _draw_head_wireframe(ax)
    cp = clock_array.positions
    ax.scatter(
        cp[:, 0], cp[:, 1], cp[:, 2],
        marker="s", s=15, color="steelblue", label="Clocks",
    )
    mp = mass_config.positions[0]
    ax.scatter(
        mp[0], mp[1], mp[2],
        marker="*", s=250, color="red", label="Mass (true)",
    )
    particles = particle_state.particles
    weights = particle_state.weights
    if len(particles) > max_particles:
        idx = np.linspace(0, len(particles) - 1, max_particles).astype(int)
        particles, weights = particles[idx], weights[idx]
    ax.scatter(
        particles[:, 0], particles[:, 1], particles[:, 2],
        c=weights, cmap="viridis", s=3, alpha=0.3,
    )
    lim = float(np.max(np.abs(mass_config.positions))) + 2.0
    ax.set_xlim3d(-lim, lim)
    ax.set_ylim3d(-lim, lim)
    ax.set_zlim3d(-lim, lim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=elev, azim=azim)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(
        f"Head lattice + particle cloud "
        f"(n_obs={particle_state.observations_seen})"
    )


def plot_centered_rates(
    ax: Axes,
    observed: NDArray[np.floating],
    predicted_centered: NDArray[np.floating],
) -> None:
    """Centered (differential) rates by clock index: prediction vs observed.

    ``predicted_centered`` is the forward model at the filter's current
    estimate, centered — so the panel shows the fit improving over frames.
    """
    idx = np.arange(len(observed))
    ax.bar(
        idx,
        predicted_centered,
        width=0.8,
        color="lightcoral",
        alpha=0.7,
        label="Predicted",
    )
    ax.bar(idx, observed, width=0.4, color="steelblue", alpha=0.7, label="Observed")
    ax.axhline(0.0, color="gray", linewidth=0.5)
    ax.set_xlabel("Clock index")
    ax.set_ylabel("Centered rate")
    ax.legend(fontsize=7)
    ax.set_title("Differential Rates")
```

Add to `src/clocks/viz.py`: import `create_echolocation_dashboard`, `plot_centered_rates`, `plot_scene_3d` from `clocks._panels3d` and add all three to `__all__` (keep both lists alphabetized as they are now).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_viz.py -v`
Expected: all PASS

- [ ] **Step 5: Gate and commit**

```bash
uv run ruff format --check . && uv run ruff check . && uv run pytest
git add src/clocks/_panels3d.py src/clocks/viz.py tests/test_viz.py
git commit -m "Add 3D echolocation dashboard panels behind the viz facade"
```

---

### Task 7: Animation driver — `animate_echolocation`

**Files:**
- Modify: `src/clocks/_animate.py` (extract precompute helper + new driver)
- Modify: `src/clocks/viz.py` (facade export)
- Test: `tests/test_viz.py` (append)

**Interfaces:**
- Consumes (Task 6): `create_echolocation_dashboard`, `plot_scene_3d`, `plot_centered_rates`, `_ECHO_COLORS`, `_ECHO_LABELS`; (existing) `plot_mass_histogram`, `_plot_convergence`, `_save_animation`, `clock_rates`.
- Produces (used by Task 8): `animate_echolocation(clock_array, mass_config, observations, pf, output_path, fps=4) -> None` — `observations` are the **centered** observations; `pf` must be fresh.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_viz.py` (add `animate_echolocation` to the `clocks.viz` import):

```python
class TestAnimateEcholocation:
    def test_creates_gif_and_processes_all_observations(
        self, tmp_path: Path
    ) -> None:
        from clocks._scenarios import (
            build_echolocation_filter,
            build_head_lattice,
            echo_mass_config,
            make_echo_observations,
        )

        _, centered = make_echo_observations(seed=0, range_r=2.0, n_observations=4)
        pf = build_echolocation_filter(seed=0, n_particles=300)
        out = tmp_path / "echo.gif"
        animate_echolocation(
            clock_array=build_head_lattice(),
            mass_config=echo_mass_config(2.0),
            observations=centered,
            pf=pf,
            output_path=out,
            fps=2,
        )
        assert out.exists()
        assert pf.state.observations_seen == 4  # frame-0 fix invariant
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_viz.py::TestAnimateEcholocation -v`
Expected: FAIL with `ImportError: cannot import name 'animate_echolocation'`

- [ ] **Step 3: Implement**

In `src/clocks/_animate.py`:

3a. Extract the precompute block (currently lines 65–79 of `_animate_filter_dashboard`) into a module-level helper, and call it from `_animate_filter_dashboard`:

```python
def _precompute_filter_states(
    pf: ParticleFilter,
    observations: list[Observation],
) -> tuple[
    list[ParticleState], list[NDArray[np.floating]], list[NDArray[np.floating]]
]:
    """Run the filter through all observations up front (frame-0 fix)."""
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
    return states, means, stds
```

In `_animate_filter_dashboard`, replace the extracted block with:

```python
    states, means, stds = _precompute_filter_states(pf, observations)
```

3b. Add the new driver (imports: extend the `clocks._panels` import with `plot_mass_histogram`; add a new import block for `clocks._panels3d`):

```python
from clocks._panels3d import (
    _ECHO_COLORS,
    _ECHO_LABELS,
    create_echolocation_dashboard,
    plot_centered_rates,
    plot_scene_3d,
)


def animate_echolocation(
    clock_array: ClockArray,
    mass_config: MassConfig,
    observations: list[Observation],
    pf: ParticleFilter,
    output_path: Path,
    fps: int = 4,
) -> None:
    """Animate the 3D echolocation filter with a slowly orbiting camera.

    ``observations`` must be the centered observations the filter consumes
    (the head has no external reference). One full azimuth orbit spans the
    whole animation. Particles have 4 columns: [x, y, z, M].
    """
    true_params = np.append(mass_config.positions[0], mass_config.masses[0])

    fig, axes = create_echolocation_dashboard()
    states, means, stds = _precompute_filter_states(pf, observations)
    n_frames = len(observations)

    def predicted_centered(frame: int) -> NDArray[np.floating]:
        """Centered forward model at the frame's posterior mean (spec §2)."""
        mean = means[frame]
        rates = clock_rates(
            MassConfig(positions=mean[:3].reshape(1, 3), masses=mean[3:4]),
            clock_array,
        )
        return rates - rates.mean()

    def render(frame: int) -> None:
        azim = -60.0 + 360.0 * frame / n_frames
        plot_scene_3d(
            axes["scene"], clock_array, mass_config, states[frame], azim=azim
        )
        axes["history"].clear()
        steps = np.arange(1, frame + 2)
        _plot_convergence(
            axes["history"],
            steps,
            np.array(means[: frame + 1]),
            np.array(stds[: frame + 1]),
            true_params,
            _ECHO_COLORS,
            _ECHO_LABELS,
            legend_kwargs={"fontsize": 7, "ncol": 2},
        )
        axes["mass"].clear()
        plot_mass_histogram(axes["mass"], states[frame], float(true_params[3]))
        axes["rates"].clear()
        plot_centered_rates(
            axes["rates"], observations[frame].rates, predicted_centered(frame)
        )

    anim = animation.FuncAnimation(fig, render, frames=n_frames, repeat=False)
    _save_animation(anim, fig, output_path, fps)
```

Add `animate_echolocation` to the `clocks._animate` import and `__all__` in `src/clocks/viz.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_viz.py -v`
Expected: all PASS, including all pre-existing animation tests (the extraction must not change behavior)

- [ ] **Step 5: Gate and commit**

```bash
uv run ruff format --check . && uv run ruff check . && uv run pytest
git add src/clocks/_animate.py src/clocks/viz.py tests/test_viz.py
git commit -m "Add rotating-camera echolocation animation driver"
```

---

### Task 8: Demo script and CLI entry point

**Files:**
- Create: `scripts/demo_echolocation_3d.py`
- Modify: `src/clocks/_cli.py` (append)
- Modify: `pyproject.toml` (`[project.scripts]`)

**Interfaces:**
- Consumes (Tasks 1, 2, 7): scenario builders and `animate_echolocation` via `clocks.viz`.
- Produces: `uv run demo-echolocation-3d` → `output/demo_echolocation_3d.gif`.

- [ ] **Step 1: Write the demo script**

Create `scripts/demo_echolocation_3d.py`:

```python
"""3D gravitational echolocation demo: exterior mass, differential sensing.

A 3x3x3 lattice of 27 clocks (the "head") senses a single point mass
placed outside it. The head has no external time reference, so the filter
sees only mean-centered (differential) rates. The camera orbits once over
the animation.

Demo seed and range are curated for visual clarity (disclosed in README
and on the site page); the range study carries the quantitative argument.
"""

from pathlib import Path

import numpy as np

from clocks._scenarios import (
    ECHO_M_TRUE,
    ECHO_N_OBSERVATIONS,
    ECHO_N_PARTICLES,
    build_echolocation_filter,
    build_head_lattice,
    echo_mass_config,
    echo_mass_position,
    make_echo_observations,
    validate_echo_geometry,
)
from clocks.viz import animate_echolocation

# --- Configuration (curated; see module docstring) ---
DEMO_RANGE_R = 4.0  # circumradii — mid-range: converges, but visibly works
DEMO_SEED = 0
OUTPUT_PATH = Path("output/demo_echolocation_3d.gif")


def main() -> None:
    clock_array = build_head_lattice()
    validate_echo_geometry(DEMO_RANGE_R, ECHO_M_TRUE, clock_array)
    mass_config = echo_mass_config(DEMO_RANGE_R)
    truth = np.append(echo_mass_position(DEMO_RANGE_R), ECHO_M_TRUE)
    print(f"True mass: M={ECHO_M_TRUE} at {truth[:3].round(2)} "
          f"({DEMO_RANGE_R} circumradii)")

    _, centered_obs = make_echo_observations(DEMO_SEED, DEMO_RANGE_R)
    pf = build_echolocation_filter(DEMO_SEED)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating animation → {OUTPUT_PATH}")
    animate_echolocation(
        clock_array=clock_array,
        mass_config=mass_config,
        observations=centered_obs,
        pf=pf,
        output_path=OUTPUT_PATH,
        fps=4,
    )

    est = pf.estimate()
    print(f"\nFinal estimate after {ECHO_N_OBSERVATIONS} observations:")
    for i, label in enumerate(["x", "y", "z", "M"]):
        print(
            f"  {label} = {est['mean'][i]:.3f} ± {est['std'][i]:.3f}"
            f"  (true: {truth[i]:.3f})"
        )
    print(f"  ESS = {est['ess']:.0f} / {ECHO_N_PARTICLES}")
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Wire the entry point**

Append to `src/clocks/_cli.py`:

```python
def demo_echolocation_3d() -> None:
    _run_script("demo_echolocation_3d.py")
```

Add to `[project.scripts]` in `pyproject.toml` (after `demo-density`):

```toml
demo-echolocation-3d = "clocks._cli:demo_echolocation_3d"
```

- [ ] **Step 3: Run the demo**

Run: `uv run demo-echolocation-3d`
Expected: prints the true parameters, generates `output/demo_echolocation_3d.gif` (expect a few minutes — 80 frames × 6000 particles), prints final estimates. Open the GIF and check: camera orbits smoothly, the head cube and star are visible, the cloud visibly contracts toward the star. If the cloud does not converge with seed 0, this is the disclosed-curation knob: try seeds 1, 2, … and update `DEMO_SEED` (nothing else) — record the chosen seed in the commit message.

- [ ] **Step 4: Copy the GIF to tracked asset locations (byte-identical policy)**

`output/` is gitignored, so publishable artifacts are committed from `assets/` and `site/assets/` the moment they are produced — later tasks must never need `output/`:

```bash
cp output/demo_echolocation_3d.gif assets/
cp output/demo_echolocation_3d.gif site/assets/
cmp output/demo_echolocation_3d.gif assets/demo_echolocation_3d.gif
cmp output/demo_echolocation_3d.gif site/assets/demo_echolocation_3d.gif
```

Expected: both `cmp` commands silent.

- [ ] **Step 5: Gate and commit**

```bash
uv run ruff format --check . && uv run ruff check . && uv run pytest
git add scripts/demo_echolocation_3d.py src/clocks/_cli.py pyproject.toml uv.lock assets/demo_echolocation_3d.gif site/assets/demo_echolocation_3d.gif
git commit -m "Add demo-echolocation-3d entry point and rotating 3D demo"
```

(`uv.lock` changes only if uv re-locks on the scripts change; include it only if modified.)

---

### Task 9: Tuning sweep — freeze parameters and thresholds (operational)

**Files:**
- Modify (only if tuning demands it): `ECHO_*` constants in `src/clocks/_scenarios.py`, `DEMO_RANGE_R`/`DEMO_SEED` in `scripts/demo_echolocation_3d.py`
- Modify: `docs/superpowers/specs/2026-07-19-3d-echolocation-design.md` (status history — record frozen values)

**Interfaces:**
- Consumes: the full pipeline (Tasks 1–8).
- Produces: frozen `ECHO_PASS_POS_TOL`, `ECHO_PASS_MASS_TOL`, `ECHO_SWEEP_RANGES`, scenario parameters, and the honest-uncertainty factor `ECHO_FAR_STD_FACTOR` used by Task 10. **Tuning seeds only (0–11). Never touch seeds 300–311 here.**

- [ ] **Step 0: SNR preflight — before burning any runs**

Run: `uv run scripts/scan_echolocation_range.py --snr-only`
Expected: the SNR table only (no sweep). Check criterion 1 of Step 2 now: at the farthest swept range, `signal_over_noise` ≥ 0.1. If violated, shrink the top of `ECHO_SWEEP_RANGES` (and re-run this preflight) before sweeping. (For reference, the review-time probe measured `signal_over_noise` ≈ 0.28 at range 8.0 with the starting defaults — expected to pass.)

- [ ] **Step 1: Run the full tuning sweep**

Run: `uv run scripts/scan_echolocation_range.py --per-run`
Expected: SNR table + 72 runs (6 ranges × 12 seeds) + summary table, `output/echolocation_range_study.json` and `.png` written.

- [ ] **Step 2: Evaluate against the freeze criteria**

The frozen configuration must satisfy, on tuning seeds:

1. **SNR gate:** at the farthest swept range, `signal_over_noise` ≥ 0.1 (spec §3: not more than ~10× below the noise floor). If violated, shrink the top of `ECHO_SWEEP_RANGES`.
2. **Close-range recovery:** at `ECHO_SWEEP_RANGES[0]`, ≥ 10/12 runs pass. If not, adjust in this order, re-running the sweep after each single change: double `ECHO_N_PARTICLES` (6000 → 12000); then raise `ECHO_PASS_POS_TOL` toward 1.5 **only** if the failures are near-misses (inspect `--per-run` output); then reconsider `ECHO_NOISE_STD` (halve to 0.0025) as a last resort.
3. **Degradation trend:** `med_position_error` increases with range, and `med_pos_std` at the farthest range ≥ 2× its value at the closest range. Record the actual measured ratio; set `ECHO_FAR_STD_FACTOR` (Task 10 constant) to half the measured ratio, rounded down to one decimal, minimum 2.0 — a pin with headroom, not a tautology.
4. **Honest uncertainty:** at the farthest range, ≥ 8/12 runs have `covered_3sigma` true. If far-range runs are confidently wrong (low coverage), that is a finding to *report on the page*, not tune away — but check first that it isn't a premature-collapse artifact by re-running one failing seed with `n_particles=12000`; if the doubled-particle run recovers coverage, adopt the larger particle count.

- [ ] **Step 3: Freeze**

Apply any constant changes from Step 2 (each with the sweep re-run confirming the criteria), then record in the spec's Status history block:

```
Tuning freeze (seeds 0-11, <date>): ECHO_SWEEP_RANGES=<values>,
ECHO_N_PARTICLES=<value>, ECHO_PASS_POS_TOL=<value>,
ECHO_PASS_MASS_TOL=<value>, ECHO_FAR_STD_FACTOR=<value>,
close-range pass <n>/12, far-range med_pos_std ratio <value>.
```

- [ ] **Step 4: Regenerate the demo GIF if constants changed**

If **any** constant affecting the demo changed in Step 2–3 — `ECHO_M_TRUE`, `ECHO_NOISE_STD`, `ECHO_N_OBSERVATIONS`, `ECHO_N_PARTICLES`, `ECHO_MASS_RANGE`, `ECHO_POSITION_HALFWIDTH`, `ECHO_DIRECTION`, or the demo range/seed (in short: anything that alters observations, truth, prior, particle count, or frame count):
Run: `uv run demo-echolocation-3d`, re-verify the GIF visually, and refresh the tracked copies:

```bash
cp output/demo_echolocation_3d.gif assets/
cp output/demo_echolocation_3d.gif site/assets/
cmp output/demo_echolocation_3d.gif assets/demo_echolocation_3d.gif
cmp output/demo_echolocation_3d.gif site/assets/demo_echolocation_3d.gif
```

- [ ] **Step 5: Gate and commit**

```bash
uv run ruff format --check . && uv run ruff check . && uv run pytest
git add -A
git commit -m "Freeze echolocation tuning: parameters, tolerances, sweep bounds"
```

---

### Task 10: Acceptance test and one-shot certification

**Files:**
- Create: `tests/test_acceptance_echolocation_3d.py`
- Create (copies): `assets/echolocation_range_study.json`, `assets/echolocation_range_study.png`, `site/assets/echolocation_range_study.png`
- Modify: `docs/superpowers/specs/2026-07-19-3d-echolocation-design.md` (record certification outcome)

**Interfaces:**
- Consumes (Tasks 2, 9): `run_echolocation_3d`, frozen `ECHO_SWEEP_RANGES`, `ECHO_PASS_*`, `ECHO_FAR_STD_FACTOR` value from the tuning freeze.
- Produces: the slow certification pin; certified artifacts committed as `assets/echolocation_range_study.{json,png}` and `site/assets/echolocation_range_study.png` from seed block 300 (Task 11 consumes only these tracked copies — never `output/`).

- [ ] **Step 1: Write the acceptance test (before running certification)**

Create `tests/test_acceptance_echolocation_3d.py`. Replace `ECHO_FAR_STD_FACTOR = 2.0` with the value frozen in Task 9 Step 3:

```python
"""Slow acceptance pin: echolocation scenario on certification seeds.

Deterministic re-execution of the certified runs (same seeds + same code
=> same result) — a regression pin, not a re-certification and not a
population reliability estimate. The exactly-once rule (spec section 3a)
bars using these seeds for tuning or selection; re-executing the frozen
configuration is permitted, exactly as test_acceptance_multi_mass_2d.py
re-executes its holdout. Excluded from default runs; execute with
`uv run pytest -m slow`. Rerun whenever inference defaults or the
scenario change.
"""

import inspect
import statistics

import numpy as np
import pytest

from clocks._scenarios import (
    ECHO_DIRECTION,
    ECHO_M_TRUE,
    ECHO_MASS_RANGE,
    ECHO_N_OBSERVATIONS,
    ECHO_N_PARTICLES,
    ECHO_NOISE_STD,
    ECHO_PASS_MASS_TOL,
    ECHO_PASS_POS_TOL,
    ECHO_POSITION_HALFWIDTH,
    ECHO_SWEEP_RANGES,
    build_echolocation_filter,
    run_echolocation_3d,
)

# Certified configuration, pinned as LITERALS (frozen in Task 9): the pin
# must not follow later edits to the live scenario constants. The fast
# guard below asserts the live constants still match, so drift fails
# loudly instead of silently redefining the certified run. On a burned
# block (spec section 3a), update CERT_SEED_BLOCK — nothing else here
# encodes the block.
CERT_SEED_BLOCK = 300
CERT_SEEDS = tuple(range(CERT_SEED_BLOCK, CERT_SEED_BLOCK + 12))
CERT_CLOSE_RANGE = 2.0
CERT_FAR_RANGE = 8.0
CERT_POS_TOL = 1.0
CERT_MASS_TOL = 0.075
CERT_SWEEP_RANGES = (2.0, 2.6, 3.5, 4.6, 6.1, 8.0)
CERT_M_TRUE = 0.15
CERT_NOISE_STD = 0.005
CERT_N_PARTICLES = 6000
CERT_N_OBSERVATIONS = 80
CERT_MASS_RANGE = (0.05, 2.0)
CERT_POSITION_HALFWIDTH = 16.0
CERT_DIRECTION = (2.0 / 7.0, 3.0 / 7.0, 6.0 / 7.0)
# Far-range posterior std must be at least this multiple of close-range.
ECHO_FAR_STD_FACTOR = 2.0


def _cert_passed(result: dict) -> bool:
    """Pass gate against the CERTIFIED tolerances (not the live ones)."""
    return (
        result["position_error"] <= CERT_POS_TOL
        and result["mass_error"] <= CERT_MASS_TOL
    )


@pytest.mark.slow
def test_certified_gates_hold_on_certification_seeds() -> None:
    """Both certified gates in one pass so each run executes exactly once."""
    close = [run_echolocation_3d(seed, CERT_CLOSE_RANGE) for seed in CERT_SEEDS]
    far = [run_echolocation_3d(seed, CERT_FAR_RANGE) for seed in CERT_SEEDS]

    failed = [r["seed"] for r in close if not _cert_passed(r)]
    assert len(CERT_SEEDS) - len(failed) >= 10, (
        f"close-range acceptance below 10/12; failing seeds: {failed}"
    )

    med_close = statistics.median(r["pos_std"] for r in close)
    med_far = statistics.median(r["pos_std"] for r in far)
    assert med_far >= ECHO_FAR_STD_FACTOR * med_close, (
        f"far-range posterior std {med_far:.3f} not >= "
        f"{ECHO_FAR_STD_FACTOR}x close-range {med_close:.3f}"
    )


def test_scenario_matches_certified_configuration() -> None:
    """Fast guard: live scenario constants equal the certified pins."""
    params = inspect.signature(run_echolocation_3d).parameters
    assert params["n_particles"].default == ECHO_N_PARTICLES
    assert params["n_observations"].default == ECHO_N_OBSERVATIONS
    assert ECHO_N_PARTICLES == CERT_N_PARTICLES
    assert ECHO_N_OBSERVATIONS == CERT_N_OBSERVATIONS
    assert ECHO_SWEEP_RANGES == CERT_SWEEP_RANGES
    assert ECHO_SWEEP_RANGES[0] == CERT_CLOSE_RANGE
    assert ECHO_SWEEP_RANGES[-1] == CERT_FAR_RANGE
    assert ECHO_PASS_POS_TOL == CERT_POS_TOL
    assert ECHO_PASS_MASS_TOL == CERT_MASS_TOL
    assert ECHO_M_TRUE == CERT_M_TRUE
    assert ECHO_NOISE_STD == CERT_NOISE_STD
    assert ECHO_MASS_RANGE == CERT_MASS_RANGE
    assert ECHO_POSITION_HALFWIDTH == CERT_POSITION_HALFWIDTH
    assert np.allclose(ECHO_DIRECTION, CERT_DIRECTION)


def test_filter_construction_matches_certified_configuration() -> None:
    """Fast guard: the built filter is certified too, not just constants."""
    pf = build_echolocation_filter(seed=0, n_particles=10)
    assert pf.jitter == "annealed"
    assert pf.jitter_tau == 15.0
    assert pf.jitter_std == 0.02
    assert pf.noise_std == CERT_NOISE_STD
    assert pf.support_bounds is not None
    lower, upper = pf.support_bounds
    hw = CERT_POSITION_HALFWIDTH
    assert np.allclose(lower, [-hw, -hw, -hw, CERT_MASS_RANGE[0]])
    assert np.allclose(upper, [hw, hw, hw, CERT_MASS_RANGE[1]])
```

Before running certification, replace every `CERT_*` literal and `ECHO_FAR_STD_FACTOR` with the values frozen in Task 9 (they are correct as written only if Task 9 froze the starting defaults unchanged).

- [ ] **Step 2: Run the certification sweep — exactly once**

Run: `uv run scripts/scan_echolocation_range.py --seed-block 300 --per-run`
Expected: the CERTIFICATION RUN banner prints; 72 runs on seeds 300–311; JSON records `"seed_block": 300`; summary shows close-range pass ≥ 10/12 and the honest-widening trend. **Do not re-run this command after this step** (the deterministic pytest pin in Step 4 is the only permitted re-execution). If the gates FAIL: per spec §3a, block 300 is burned — record the failure and diagnosis in the spec Status history, return to Task 9 (tuning seeds only), re-freeze, then certify once on `--seed-block 400`. On the burn path, the block number is parameterized in exactly one place per artifact: update `CERT_SEED_BLOCK` in the acceptance test, and substitute the new block for `300` in this task's scan command, Status-history record, and commit message (Steps 2, 5, 6) **and in every downstream mention of the certified block** — the site page's reproduce callout and study paragraph (Task 11), the reproduce page's scan comment (Task 11), and the PR body (Task 12), all of which say "seed-block 300" as written. The spec Status history is the source of truth for which block certified.

- [ ] **Step 3: Commit the certified artifacts to tracked locations**

`output/` is gitignored and the certification cannot be re-run, so the certified JSON and figure are committed immediately (byte-identical policy; Task 11 consumes only these tracked copies):

```bash
cp output/echolocation_range_study.png assets/
cp output/echolocation_range_study.png site/assets/
cp output/echolocation_range_study.json assets/
cmp output/echolocation_range_study.png assets/echolocation_range_study.png
cmp output/echolocation_range_study.png site/assets/echolocation_range_study.png
cmp output/echolocation_range_study.json assets/echolocation_range_study.json
```

Expected: all three `cmp` commands silent.

- [ ] **Step 4: Run the slow acceptance pin**

Run: `uv run pytest -m slow tests/test_acceptance_echolocation_3d.py -v`
Expected: PASS (deterministic re-execution of the certified runs; ~10–20 min). Also run the fast guard: `uv run pytest tests/test_acceptance_echolocation_3d.py -v -k certified_configuration` → PASS.

- [ ] **Step 5: Record certification in the spec**

Append to the spec Status history:

```
Certification (seeds <block>-<block+11>, run once, <date>): close-range
pass <n>/12, med pos_std ratio far/close = <value>; certified artifacts
committed as assets/echolocation_range_study.{json,png} and
site/assets/echolocation_range_study.png (seed_block <block>).
```

(`<block>` is 300, or the replacement block on the burn path.)

- [ ] **Step 6: Gate and commit**

```bash
uv run ruff format --check . && uv run ruff check . && uv run pytest
git add tests/test_acceptance_echolocation_3d.py docs/superpowers/specs/2026-07-19-3d-echolocation-design.md assets/echolocation_range_study.json assets/echolocation_range_study.png site/assets/echolocation_range_study.png
git commit -m "Certify echolocation on seeds 300-311; add slow acceptance pin"
```

---

### Task 11: Site page, sidebar, and published artifacts

**Files:**
- Create: `site/story/gravitational-echolocation.qmd`
- Modify: `site/_quarto.yml` (sidebar), `site/reproduce/getting-started.qmd` (demo list)

**Interfaces:**
- Consumes: **tracked** artifacts committed by earlier tasks — `assets/demo_echolocation_3d.gif` + `site/assets/demo_echolocation_3d.gif` (Task 8/9) and `assets/echolocation_range_study.{json,png}` + `site/assets/echolocation_range_study.png` (Task 10); `build_head_lattice`, `echo_mass_config`, `ECHO_*` constants for the in-page falloff cell. This task never reads `output/`.
- Produces: the published page. Fill `<N-CLOSE>`, `<USABLE-RANGE>`, `<FAR-BEHAVIOR>` from the certified study (`assets/echolocation_range_study.json` / the spec's certification record) before committing — grep the page for `<` to confirm nothing remains.
- **Conditional rewrite (verdict honesty):** the page template below assumes the certified far range showed *honest* widening (`covered_3sigma` mostly true — check the certified JSON). If certification instead showed the filter **confidently wrong** at far range (poor coverage), rewrite `<FAR-BEHAVIOR>`, the "Verdict" section, and the caption claims accordingly: report that the sense degrades *without* knowing it, drop the "knows when it is failing" framing, and state the coverage numbers plainly. The study's finding is whatever certification measured — the page adapts to it, never the reverse.

- [ ] **Step 1: Verify the tracked artifacts exist**

```bash
git ls-files --error-unmatch assets/demo_echolocation_3d.gif site/assets/demo_echolocation_3d.gif assets/echolocation_range_study.json assets/echolocation_range_study.png site/assets/echolocation_range_study.png
```

Expected: all five paths print (exit 0). If any is missing, stop — the producing task (8/9/10) did not complete its artifact-commit step.

- [ ] **Step 2: Write the page**

Create `site/story/gravitational-echolocation.qmd`. Fill the three `<placeholders>` from the certification summary (Task 10 Step 2 output). **Then reconcile every hardcoded number in the prose with the frozen constants** — the template below is written for the *starting* defaults, and Task 9 may have changed them: "four circumradii" ↔ `DEMO_RANGE_R`, "eighty differential observations" ↔ `ECHO_N_OBSERVATIONS`, "six thousand hypotheses" ↔ `ECHO_N_PARTICLES`, "two to eight circumradii" ↔ `ECHO_SWEEP_RANGES` endpoints, "twelve independent runs" ↔ the seed-block size, and "seed-block 300" in the reproduce callout ↔ the certified block per the spec Status history ("twenty-seven clocks" and "3×3×3" are fixed by the spec and never change). Cross-check against `src/clocks/_scenarios.py` and `scripts/demo_echolocation_3d.py` as committed:

````markdown
---
title: "Gravitational Echolocation"
---

## A sense you weren't born with

Imagine someone born with a lattice of atomic clocks embedded in their
head — twenty-seven of them, arranged in a neat 3×3×3 grid. Every mass in
the world dents spacetime, and clocks deeper in the dent tick slower. Could
such a person *feel* the masses around them, the way echolocation feels
surfaces?

One constraint makes this premise honest: the head carries no outside
reference. There is no distant master clock to compare against — only the
twenty-seven clocks against each other. Whatever is sensed must live in
the *differences* between their tick rates, never in their absolute values.

That is a well-posed physics question, and everything on this site so far
— the forward model, the particle filter, the degeneracy analysis — is
already dimension-agnostic. So: how far can this sense reach?

## What an exterior mass shows the head

Every previous page hid the mass *inside* the sensor array. An exterior
mass is a harder target: it shifts all twenty-seven clocks by nearly the
same amount (a common-mode offset, ∝ M/R), and that shared offset is
exactly what the head cannot perceive. Subtract the mean and what remains
is the *gradient* of the potential across the head — falling off as 1/R²
— plus the still-fainter curvature that distinguishes "small mass nearby"
from "large mass far away." Direction is cheap; range is expensive. This
is the [mass–distance degeneracy](one-clock-is-not-enough.qmd) in its
purest form.

```{python}
#| code-fold: true
#| fig-cap: "The differential signal an exterior mass leaves on the head, vs range. The dashed line is the per-observation noise floor; the sense must work in the shrinking gap above it."
import matplotlib.pyplot as plt
import numpy as np
from clocks import clock_rates
from clocks._scenarios import (
    ECHO_NOISE_STD,
    build_head_lattice,
    echo_mass_config,
)

head = build_head_lattice()
ranges = np.linspace(2.0, 10.0, 60)  # from the validated exterior minimum
signal = []
for r in ranges:
    rates = clock_rates(echo_mass_config(float(r)), head)
    signal.append(np.max(np.abs(rates - rates.mean())))

fig, ax = plt.subplots()
ax.semilogy(ranges, signal, color="steelblue", label="differential signal")
ax.axhline(
    ECHO_NOISE_STD, color="lightcoral", linestyle="--", label="noise floor"
)
ax.set_xlabel("range (circumradii)")
ax.set_ylabel("max centered rate difference")
ax.legend()
plt.close(fig)
fig
```

One technical footnote. Subtracting the mean makes the observation noise
slightly correlated — every centered rate shares the subtracted average.
The filter nonetheless keeps the simpler independent-noise likelihood,
which is harmless here: on centered residuals the two likelihoods differ
only by a constant that cancels out of the particle weights. (Only the
absolute evidence normalization shifts, and nothing on this page uses it.)

## Watching the head lock on

The demo places the mass at four circumradii along a fixed off-axis
direction and lets the filter work through eighty differential
observations. The camera orbits once while the particle cloud — six
thousand hypotheses spread through a box far larger than the head —
collapses onto the true mass. (This demo run is curated for visual
clarity; the study below is not.)

![The echolocation demo. Left: the head lattice, the true exterior mass (star), and the particle cloud, camera orbiting. Right, top to bottom: parameter convergence, mass marginal, differential rates by clock.](../assets/demo_echolocation_3d.gif)

## How far does the sense reach?

The quantitative version: sweep the range from two to eight circumradii,
twelve independent runs per range (certification seeds, run exactly once —
no cherry-picking), and track both the *actual* error and the filter's own
*claimed* uncertainty.

![Resolution vs range. Top: position error; bottom: mass error. Dots are individual runs, solid lines are medians, dashed lines are the filter's claimed uncertainty (posterior std).](../assets/echolocation_range_study.png)

At the closest range the head localizes the mass in <N-CLOSE> of twelve
runs. Past roughly <USABLE-RANGE> circumradii the errors grow into the
range itself — and, crucially, <FAR-BEHAVIOR>. A sense that *knows when it
is failing* is a coherent sense; one that hallucinates confident answers
is not.

## Verdict

Physically coherent, with a short horizon. The differential signal an
exterior mass leaves on a head-sized lattice dies as 1/R², and separating
mass from range leans on a curvature term dying as 1/R³ — so the sense is
sharp out to a few head-radii and dissolves into honest uncertainty beyond
that. Our imagined synesthete would feel furniture, doorframes, a passing
truck — as presences with direction and rough heft — while the far world
stays dark. Echolocation is the right metaphor: a near-field sense,
written in spacetime instead of sound.

::: {.callout-note title="Reproduce"}
`uv run demo-echolocation-3d` regenerates the demo;
`uv run scripts/scan_echolocation_range.py` reruns the sweep (tuning
seeds; the published figure used `--seed-block 300`). See
[Getting Started](../reproduce/getting-started.qmd).
:::
````

- [ ] **Step 3: Wire the sidebar and reproduce page**

In `site/_quarto.yml`, append to the Part 1 contents after the "Beyond Point Masses" entry:

```yaml
          - text: "Gravitational Echolocation"
            href: story/gravitational-echolocation.qmd
```

In `site/reproduce/getting-started.qmd`, update the demo count sentence ("Five animated demos" → "Six animated demos") and add to the demo list:

```bash
uv run demo-echolocation-3d     # → output/demo_echolocation_3d.gif (~5 min)
```

Also add, after the demo list (spec §4 requires the scan command on the reproduce pages):

````markdown
The echolocation range study behind the site page:

```bash
uv run scripts/scan_echolocation_range.py   # tuning seeds; published figure used --seed-block 300
```
````

- [ ] **Step 4: Build the site**

Run: `cd site && quarto render && cd ..`
Expected: build succeeds; `site/_output/story/gravitational-echolocation.html` exists; the falloff cell executed (figure present in the HTML). Open the page and check images render and no `<placeholder>` text remains: `grep -c "N-CLOSE\|USABLE-RANGE\|FAR-BEHAVIOR" site/story/gravitational-echolocation.qmd` → 0.

- [ ] **Step 5: Gate and commit**

```bash
uv run ruff format --check . && uv run ruff check . && uv run pytest
git add site/ assets/
git commit -m "Add Gravitational Echolocation story page with demo and range study"
```

---

### Task 12: README, someday-maybe, final gate, and PR

**Files:**
- Modify: `README.md` (demo section + project structure), `docs/someday-maybe.md` (mark shipped)

**Interfaces:**
- Consumes: everything above.
- Produces: the merged feature.

- [ ] **Step 1: Update README**

In `README.md`, add after the Gaussian-density demo block:

````markdown
**3D echolocation** — a 3×3×3 "head" of 27 clocks senses a single
*exterior* mass from differential (mean-centered) rates only — the head
has no outside time reference. Demo seed/range are curated for clarity;
`scripts/scan_echolocation_range.py` runs the uncurated resolution-vs-range
study behind the site page:

```bash
uv run demo-echolocation-3d    # → output/demo_echolocation_3d.gif
```

![3D echolocation demo](assets/demo_echolocation_3d.gif)
````

Update the "Project structure" listing: add `_scenarios.py`, `_panels3d.py`, `_echo_study.py` lines under `src/clocks/` (note: `_scenarios.py` exists but is currently missing from the README listing — add it now) and `demo_echolocation_3d.py`, `scan_echolocation_range.py`, `scan_multi_mass_2d.py` under `scripts/` (the two scan scripts are also currently missing).

- [ ] **Step 2: Update someday-maybe**

In `docs/someday-maybe.md`, rewrite the "3D and exterior masses: gravitational echolocation" bullet to:

```markdown
- **3D and exterior masses: gravitational echolocation.** Shipped
  2026-07-19 — spec
  `docs/superpowers/specs/2026-07-19-3d-echolocation-design.md`.
  Demo: `uv run demo-echolocation-3d`; study:
  `scripts/scan_echolocation_range.py`; site page
  `site/story/gravitational-echolocation.qmd`. Deferred follow-ons still
  open: lattice-geometry comparisons (field of view per clock),
  angular-resolution measurement, multiple/moving exterior masses.
```

- [ ] **Step 3: Full local gate, including slow tests**

```bash
uv run ruff format --check . && uv run ruff check .
uv run pytest
uv run pytest -m slow
```

Expected: all green.

- [ ] **Step 4: Commit, push, open PR**

```bash
git add README.md docs/someday-maybe.md
git commit -m "Document echolocation demo; mark someday-maybe item shipped"
git push -u origin echolocation-3d
gh pr create --title "3D gravitational echolocation: demo, range study, site page" --body "Implements docs/superpowers/specs/2026-07-19-3d-echolocation-design.md (approved: Codex xhigh, 3 rounds).

- Centered (differential) measurement model on a raw ParticleFilter — the head has no external reference
- 27-clock head lattice scenario with weak-field validation
- Rotating-camera 3D demo GIF (demo-echolocation-3d)
- Resolution-vs-range study: tuning seeds 0-11, certified once on seeds 300-311
- Slow acceptance pin + 3D core coverage tests
- New site story page with falloff cell, demo GIF, and study figure

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

- [ ] **Step 5: CI green, Codex PR review, merge**

1. `gh pr checks <PR> --watch` → all green (never merge on local-green alone).
2. Run the Codex xhigh review of the final PR diff per the global Review & Approval Protocol; post each round's findings and responses as PR comments; iterate until "READY TO MERGE".
3. Merge; confirm CI stays green on `main`; delete the branch.

---

## Self-Review (completed)

- **Spec coverage:** scenario+geometry (T1–2), measurement model §1a (T2), 3D core tests §5 (T3), study helpers+SNR gate §3 (T4–5), seed protocol §3a (T5, 9, 10), dashboard §2 (T6–7), demo+CLI §2 (T8), tuning freeze (T9), acceptance+certification §5 (T10), site page+assets §4 (T11), README/someday-maybe §4 (T12). Deferred items (§ out-of-scope) have no tasks — correct.
- **Placeholder scan:** the only intentional fill-ins are Task 11's three page placeholders and Task 9/10's frozen-value recordings, each with explicit fill instructions and a grep check.
- **Type consistency:** `EchoRunResult` field names match between T2 (producer), T4 (fake in tests + summarize keys), T5 (per-run printout), T10 (acceptance asserts). `ECHO_SWEEP_RANGES` indices used by T10 match T1's tuple. Facade export names match between T6/T7 and their tests.
