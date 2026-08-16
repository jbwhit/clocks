# Correctness and Rigor Remediation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the repository's invalid physics and inference behavior with strict weak-field contracts and target-preserving tempered SMC, while fixing resampling, packaging, public validation, and incorrect prose.

**Architecture:** Keep the streaming `ParticleFilter.update()` interface, but implement each update as adaptive likelihood tempering with resampling and symmetric random-walk Metropolis-Hastings moves. Treat configured bounds, multi-mass ordering, and `|2 Phi| <= 0.1` as the actual conditional prior support; make direct physics calls fail loudly outside that domain. Package demos beneath `src/clocks`, expose exact diagnostics, and regenerate all claims and assets from the corrected implementation.

**Tech Stack:** Python 3.12+, NumPy, SciPy, pytest, Ruff, uv/Hatchling, Quarto, Matplotlib.

---

## Execution rules

- Work in `/Users/jonathan/projects/clocks/.worktrees/codex-correctness-rigor` on branch `codex-correctness-rigor`.
- Read `docs/plans/2026-08-16-correctness-rigor-design.md` before editing.
- Use @superpowers:test-driven-development for every behavior change,
  @superpowers:systematic-debugging for any unexpected failure, and
  @superpowers:verification-before-completion before each success claim.
- Never tune against a certification seed block. Development seeds are `0-11`.
  The old `200-211` and `300-311` blocks are burned; the new one-shot
  certification block is `400-411`.
- Run verification and commit as separate shell calls. After each logical
  commit, push without asking; the branch's first push uses `-u`.
- Do not add compatibility aliases for `jitter`, `jitter_std`, `jitter_tau`,
  `constraint_fn`, `support_bounds`, or the old `_cli` dispatcher.

## Task 1: Make public data and configuration objects enforce invariants

**Files:**

- Create: `src/clocks/_validation.py`
- Create: `tests/test_types.py`
- Modify: `src/clocks/types.py:1-70`
- Modify: `src/clocks/config.py:1-110`
- Modify: `tests/test_api.py`

**Step 1: Write failing contract tests**

Add tests for array coercion, ranks, empty arrays, non-finite values, negative
masses/offsets, mismatched counts, immutable stored arrays, invalid weights,
and non-finite configuration values. Representative tests:

```python
@pytest.mark.parametrize(
    ("positions", "masses", "message"),
    [
        (np.empty((0, 1)), np.empty(0), "nonempty"),
        (np.zeros((2, 1)), np.ones((2, 1)), "masses must be 1-D"),
        (np.array([[np.nan]]), np.array([1.0]), "finite"),
        (np.array([[0.0]]), np.array([-1.0]), "nonnegative"),
    ],
)
def test_mass_config_rejects_invalid_arrays(positions, masses, message):
    with pytest.raises(ValueError, match=message):
        MassConfig(positions=positions, masses=masses)


def test_public_arrays_are_defensive_read_only_copies():
    source = np.array([[1.0]])
    config = MassConfig(source, np.array([0.1]))
    source[0, 0] = 9.0
    assert config.positions[0, 0] == 1.0
    with pytest.raises(ValueError, match="read-only"):
        config.positions[0, 0] = 2.0


def test_observation_requires_one_dimensional_finite_rates():
    with pytest.raises(ValueError, match="rates must be 1-D"):
        Observation(np.ones((1, 3)), time=0.0)
    with pytest.raises(ValueError, match="time must be finite"):
        Observation(np.ones(3), time=np.inf)
```

For `ParticleState`, require `(N, D)` particles, `(N,)` weights, nonnegative
finite weights summing to one within `1e-12`, and nonnegative integer
`observations_seen`. Add `NoiseConfig(np.nan)`, infinite prior endpoints,
invalid ESS/proposal controls, and `bool`-as-count cases to `tests/test_api.py`.

**Step 2: Run the focused tests and confirm failure**

Run:

```bash
uv run pytest tests/test_types.py tests/test_api.py -q
```

Expected: failures showing missing rank/finiteness/immutability checks and the
old jitter configuration surface.

**Step 3: Add reusable validators**

Implement these private helpers in `src/clocks/_validation.py`:

```python
def finite_float(name: str, value: object) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def finite_float_array(
    name: str,
    value: object,
    *,
    ndim: int,
    nonempty: bool = True,
) -> NDArray[np.float64]:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {array.shape}")
    if nonempty and 0 in array.shape:
        raise ValueError(f"{name} must be nonempty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array
```

Keep the documented 1-D position shorthand and scalar mass shorthand by
normalizing those two cases before calling the exact-rank helper.

**Step 4: Replace the inference configuration fields**

Change `InferenceConfig` to:

```python
@dataclass(frozen=True)
class InferenceConfig:
    clock_array: ClockArray
    noise: NoiseConfig
    prior: PriorConfig
    n_particles: int
    n_masses: int | tuple[int, ...]
    resampling: str = "systematic"
    ess_target: float = 0.8
    rejuvenation_steps: int = 2
    proposal_scale: float = 2.38
    seed: int | None = None
```

Validate `0 < ess_target < 1`, positive integer counts excluding `bool`, a
finite positive `proposal_scale`, known resampling names, finite increasing
prior ranges, and a positive finite observation standard deviation. Make the
`PriorConfig` docstring state that both ranges are true prior support.

**Step 5: Implement and run the contract tests**

Run:

```bash
uv run pytest tests/test_types.py tests/test_api.py -q
uv run ruff check src/clocks/_validation.py src/clocks/types.py src/clocks/config.py tests/test_types.py tests/test_api.py
```

Expected: contract tests pass. Some unrelated API tests may still fail because
callers still pass removed jitter fields; do not restore those fields.

**Step 6: Commit and push**

```bash
git add src/clocks/_validation.py src/clocks/types.py src/clocks/config.py tests/test_types.py tests/test_api.py
git commit -m "refactor: enforce public data contracts"
git push -u origin codex-correctness-rigor
```

## Task 2: Replace physics clamps with a strict weak-field domain

**Files:**

- Modify: `src/clocks/physics.py:1-230`
- Modify: `src/clocks/__init__.py:17-26,82-107`
- Modify: `tests/test_physics.py`

**Step 1: Write failing domain and shape tests**

Replace `test_black_hole_guard` with strict tests:

```python
@pytest.mark.parametrize("potential", [-0.051, 0.001, np.nan])
def test_time_dilation_rejects_outside_model_domain(potential):
    with pytest.raises(PhysicsDomainError):
        time_dilation_factor(np.array([potential]))


def test_time_dilation_accepts_validity_boundary_without_clamping():
    result = time_dilation_factor(np.array([-0.05, 0.0]))
    np.testing.assert_allclose(result, np.sqrt([0.9, 1.0]))


def test_scalar_and_batch_reject_same_invalid_state():
    clocks = ClockArray(np.array([[0.0]]), track_offset=1.0)
    mass = MassConfig(np.array([[0.0]]), np.array([0.051]))
    with pytest.raises(PhysicsDomainError, match="weak-field"):
        clock_rates(mass, clocks)
    with pytest.raises(PhysicsDomainError, match="weak-field"):
        clock_rates_batch(np.array([[0.0]]), np.array([0.051]), clocks)
```

Add exact-rank and compatible-dimension tests for all scalar and batch paths.
Add density tests for parameter shape `(3,)`, batch shape `(N, 3)`,
`sigma > 0`, `amplitude >= 0`, `track_offset > 0`, positive finite integration
limit, and integer `n_quad >= 2`.

**Step 2: Confirm the old implementation fails**

Run:

```bash
uv run pytest tests/test_physics.py -q
```

Expected: the clamp test returns a finite floor and malformed inputs broadcast
instead of raising.

**Step 3: Implement the strict domain**

In `physics.py`, add and export:

```python
WEAK_FIELD_LIMIT = 0.1


class PhysicsDomainError(ValueError):
    """A state lies outside the documented gravitational model."""


def _validate_potential(potential: NDArray[np.float64]) -> None:
    if not np.all(np.isfinite(potential)):
        raise PhysicsDomainError("potential must be finite")
    if np.any(potential > 0.0):
        raise PhysicsDomainError("potential must be nonpositive")
    strength = np.abs(2.0 * potential)
    if np.any(strength > WEAK_FIELD_LIMIT):
        raise PhysicsDomainError(
            f"weak-field policy requires |2*Phi| <= {WEAK_FIELD_LIMIT}"
        )
```

Delete `_EPS`. Reject zero distance only in columns whose mass is positive.
Compute rates exactly as `np.sqrt(1.0 + 2.0 * potential)` after validation.
Create a private `_point_mass_potential_batch(...)` shared by strict batch
functions and API prior-support evaluation; it returns raw potentials without
clipping. Candidate support evaluation must use
`np.errstate(divide="ignore", invalid="ignore")` around raw division and mark
zero-distance/positive-mass pairs invalid before thresholding; an impossible
candidate is normal control flow and must not emit a warning. Batch public
functions call the same validation as scalar functions.

**Step 4: Make density evaluation strict**

Validate parameters before integration. Both quadrature paths must call the
same exact `time_dilation_factor`; do not insert a density-specific floor.
Require `track_offset > 0` because the line-density integral diverges at a
coincident clock when density is nonzero.

**Step 5: Run focused verification**

```bash
uv run pytest tests/test_physics.py -q
uv run ruff format --check src/clocks/physics.py tests/test_physics.py
uv run ruff check src/clocks/physics.py tests/test_physics.py
```

Expected: all physics tests pass and a repository search finds no radicand or
distance floors:

```bash
rg -n "maximum\(.*argument|maximum\(.*distance|_EPS|clamp" src/clocks/physics.py
```

Expected search result: empty.

**Step 6: Commit and push**

```bash
git add src/clocks/physics.py src/clocks/__init__.py tests/test_physics.py
git commit -m "fix: enforce the weak-field physics domain"
git push
```

## Task 3: Correct and isolate all resampling algorithms

**Files:**

- Modify: `src/clocks/inference.py:1-120,319-345`
- Modify: `tests/test_inference.py`

**Step 1: Add a regression test for the exact residual bug**

Expose private module-level `_systematic_indices`, `_stratified_indices`, and
`_residual_indices` helpers with `(weights, n_draws, rng)` signatures. Test:

```python
def test_residual_resampling_does_not_clip_to_remainder_count():
    weights = np.array([0.20, 0.19, 0.21, 0.20, 0.20])
    rng = np.random.default_rng(1234)
    counts = np.zeros(5, dtype=int)
    repetitions = 20_000
    for _ in range(repetitions):
        draw = _residual_indices(weights, 5, rng)
        counts += np.bincount(draw, minlength=5)
    expected = repetitions * 5 * weights
    sigma = np.sqrt(repetitions * 5 * weights * (1.0 - weights))
    assert np.all(np.abs(counts - expected) < 6.0 * sigma)
    assert counts[1] > counts[0] * 0.8
```

Also test non-unit-sum rejection, negative/NaN weights, `n_draws <= 0`, exact
output length, index bounds, and deterministic copies when residual counts
consume the entire draw.

**Step 2: Run and observe the biased result**

```bash
uv run pytest tests/test_inference.py -k "residual_resampling_does_not_clip" -q
```

Expected: failure with almost all residual draws incorrectly sent to index 0.

**Step 3: Implement the corrected primitives**

Use source population size for clipping:

```python
def _systematic_indices(weights, n_draws, rng):
    _validate_resampling_inputs(weights, n_draws)
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    positions = (rng.random() + np.arange(n_draws)) / n_draws
    return np.clip(
        np.searchsorted(cumulative, positions, side="right"),
        0,
        len(weights) - 1,
    ).astype(np.intp)
```

Apply the same `len(weights) - 1` rule to stratified draws. Residual sampling
uses `floor(n_draws * weights)`, normalizes only the fractional remainder, and
passes that full source-length vector to `_systematic_indices`.

**Step 4: Run the resampling tests**

```bash
uv run pytest tests/test_inference.py -k "resampl" -q
```

Expected: all focused tests pass, including the empirical regression.

**Step 5: Commit and push**

```bash
git add src/clocks/inference.py tests/test_inference.py
git commit -m "fix: remove residual resampling index bias"
git push
```

## Task 4: Build the Gaussian sufficient-statistics target evaluator

**Files:**

- Rewrite: `src/clocks/inference.py:102-469`
- Modify: `tests/test_inference.py`

**Step 1: Remove jitter-era tests and write target tests first**

Delete tests whose asserted behavior is unconditional jitter, reflection,
support repair, clone-aware ESS, or annealed defaults. Add tests that compare
the sufficient-statistics log likelihood with direct likelihood summation:

```python
def test_sufficient_statistics_match_direct_gaussian_sum():
    observations = np.array([[0.2, -0.1], [0.3, 0.0], [0.4, 0.2]])
    predicted = np.array([[0.25, 0.05], [0.5, -0.2]])
    stats = GaussianObservationStats.empty(2)
    for row in observations:
        stats = stats.add(row)
    actual = stats.log_likelihood(predicted, noise_std=0.3)
    expected = np.array([
        sum(log_likelihood_gaussian(row, mu, 0.3) for row in observations)
        for mu in predicted
    ])
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
```

Add a fractional-current-observation case verifying `beta=0`, `0.37`, and
`1.0`, plus tests that invalid prior rows become `-inf` without being sent to
the strict forward model.

**Step 2: Confirm failure**

```bash
uv run pytest tests/test_inference.py -k "sufficient_statistics or invalid_prior_rows" -q
```

Expected: `GaussianObservationStats` and the new target evaluator do not exist.

**Step 3: Implement immutable sufficient statistics**

Add a private frozen dataclass with `n`, `sum_y`, and `sum_y2`. Its batch
formula must be:

```python
quadratic = (
    self.sum_y2
    - 2.0 * predicted @ self.sum_y
    + self.n * np.sum(predicted**2, axis=1)
)
normalizer = self.n * n_channels * math.log(noise_std * math.sqrt(2.0 * math.pi))
return -normalizer - quadratic / (2.0 * noise_std**2)
```

Represent fractional current data by adding `beta` to `n`, `beta * rates` to
`sum_y`, and `beta * dot(rates, rates)` to `sum_y2` without mutating completed
statistics.

**Step 4: Implement support-aware prediction**

`ParticleFilter._log_prior(particles)` returns a vector, validates its shape,
and treats any non-finite value other than `-inf` as an error. The target
evaluator calls scalar/batch forward models only on rows with finite prior
density, validates prediction shape `(n_valid, n_channels)`, and fills every
other row with `-inf` target density.

The constructor contract becomes:

```python
ParticleFilter(
    n_particles,
    prior_sampler,
    forward_model,
    noise_std,
    *,
    log_prior_density,
    forward_model_batch=None,
    resampling="systematic",
    ess_target=0.8,
    rejuvenation_steps=2,
    proposal_scale=2.38,
    rng=None,
)
```

Require every initial particle to have finite prior density; the sampler must
already draw from the represented prior.

**Step 5: Run focused tests**

```bash
uv run pytest tests/test_inference.py -k "sufficient_statistics or target or prior_rows" -q
```

Expected: exact equality tests pass.

**Step 6: Commit and push**

```bash
git add src/clocks/inference.py tests/test_inference.py
git commit -m "refactor: evaluate static Gaussian SMC targets exactly"
git push
```

## Task 5: Implement adaptive tempering and evidence accumulation

**Files:**

- Modify: `src/clocks/inference.py`
- Modify: `src/clocks/types.py`
- Modify: `tests/test_inference.py`

**Step 1: Add analytic posterior and evidence tests**

Use the conjugate scalar problem
`theta ~ Normal(0, tau^2)`, `y_t | theta ~ Normal(theta, sigma^2)`.
Compute the exact posterior and log evidence with:

```python
posterior_var = 1.0 / (1.0 / tau**2 + len(y) / sigma**2)
posterior_mean = posterior_var * y.sum() / sigma**2
cov = sigma**2 * np.eye(len(y)) + tau**2 * np.ones((len(y), len(y)))
expected_log_evidence = scipy.stats.multivariate_normal.logpdf(
    y, mean=np.zeros(len(y)), cov=cov
)
```

With `100_000` particles and a fixed seed, assert posterior mean within `0.01`,
posterior standard deviation within `0.01`, and log evidence within `0.04`.
Use `ess_target=0.8`, sharp enough data to require more than one tempering stage,
and assert one history state per observation.

Add a separate identity test for each incremental evidence update:

```python
expected = logsumexp(np.log(old_weights) + delta_beta * log_likelihood)
assert update_log_increment == pytest.approx(expected, abs=1e-12)
```

**Step 2: Run and confirm failure**

```bash
uv run pytest tests/test_inference.py -k "analytic or tempering or evidence_increment" -q
```

Expected: posterior/evidence mismatch under the old unconditional-jitter path.

**Step 3: Implement stable tempering helpers**

Add `_normalize_log_weights`, `_effective_sample_size`, and `_next_beta`.
`_next_beta` returns `1.0` if the full remaining likelihood keeps ESS at or
above `ess_target * N`; otherwise use 60 iterations of bisection to find the
largest beta at that ESS. Do not multiply `0 * -inf`; beta increments are
strictly positive.

For each increment:

```python
candidate_log_weights = np.log(weights) + delta_beta * observation_ll
log_increment = logsumexp(candidate_log_weights)
self.log_evidence += log_increment
weights = np.exp(candidate_log_weights - log_increment)
beta = next_beta
```

Whenever ESS is at or below the target, resample and reset weights to `1/N`,
including after the final increment reaches `beta=1`; Task 6 adds the move
immediately afterward. This guarantees the next observation never starts below
the target with no bisection root. When beta reaches one, add the observation
to completed sufficient statistics and append exactly one immutable
`ParticleState`.

**Step 4: Add update diagnostics**

Add a frozen `UpdateDiagnostics` type with `tempering_stages`, `mh_proposals`,
and `mh_acceptances`. Expose `ParticleFilter.last_diagnostics`; for now MH
counts remain zero. Validate observations and prediction channel count on the
first update and require all later observations to match it.

**Step 5: Run focused and full inference tests**

```bash
uv run pytest tests/test_inference.py -k "analytic or tempering or evidence or observation" -q
uv run pytest tests/test_inference.py -q
```

Expected: all retained inference tests pass; convergence fixtures may need only
the weak-field numeric retuning scheduled in Task 8.

**Step 6: Commit and push**

```bash
git add src/clocks/inference.py src/clocks/types.py tests/test_inference.py
git commit -m "feat: add adaptive tempered SMC updates"
git push
```

## Task 6: Add target-preserving Metropolis-Hastings rejuvenation

**Files:**

- Modify: `src/clocks/inference.py`
- Modify: `tests/test_inference.py`

**Step 1: Write MH correctness tests**

Add tests that:

- force at least one resample-move stage and assert nonzero proposals and
  acceptances;
- use a bounded prior and assert out-of-support proposals stay at their current
  particle rather than being reflected or sorted;
- freeze a proposal covariance and verify that reversing a proposed displacement
  has the same Gaussian log proposal density; and
- rerun the conjugate test with multiple rejuvenation steps, preserving the
  analytic posterior and evidence tolerances.

```python
def test_mh_rejects_support_crossing_without_repair():
    pf = bounded_scalar_filter(proposal_scale=50.0, rejuvenation_steps=1)
    current = np.array([[0.99]])
    moved, diagnostics = pf._metropolis_move(current, beta=1.0, observation=OBS)
    np.testing.assert_array_equal(moved, current)
    assert diagnostics.proposals == 1
    assert diagnostics.acceptances == 0
```

Use a stub RNG in the focused boundary test so the crossing proposal is
deterministic; do not assert a random acceptance count.

**Step 2: Confirm failure**

```bash
uv run pytest tests/test_inference.py -k "mh_ or rejuvenation" -q
```

Expected: missing move implementation and zero diagnostics.

**Step 3: Implement a frozen symmetric proposal per stage**

Before resampling, compute the weighted covariance of the current target cloud.
Regularize it with the initial prior scale:

```python
ridge = 1e-6 * np.diag(np.maximum(self._initial_scale, 1e-12) ** 2)
proposal_cov = (self.proposal_scale**2 / n_params) * (weighted_cov + ridge)
proposal_chol = np.linalg.cholesky(proposal_cov)
```

Freeze this Cholesky factor for all particles and all configured move steps in
that tempering stage. Each proposal is `current + z @ proposal_chol.T`.
Calculate `log_alpha = target(proposal) - target(current)` and accept iff
`log(U) < min(0, log_alpha)`. Never reflect, sort, redraw-until-valid, or copy a
different donor. A rejected proposal remains exactly at its current value.

**Step 4: Wire moves immediately after every resample**

Resample first, set equal weights, then run `rejuvenation_steps` MH sweeps
against `pi[t,beta]`. Carry the moved particles into the next beta stage and
accumulate exact proposal/acceptance diagnostics. The move must never change
`log_evidence`.

**Step 5: Verify posterior, evidence, and support**

```bash
uv run pytest tests/test_inference.py -k "mh_ or rejuvenation or analytic" -q
uv run pytest tests/test_inference.py -q
```

Expected: all tests pass; the analytic log evidence remains within its prior
tolerance with rejuvenation enabled.

**Step 6: Commit and push**

```bash
git add src/clocks/inference.py tests/test_inference.py
git commit -m "feat: add target-preserving SMC rejuvenation"
git push
```

## Task 7: Make the API sample and preserve the actual conditional prior

**Files:**

- Rewrite: `src/clocks/api.py:1-260`
- Rewrite: `src/clocks/inference.py:472-700`
- Modify: `src/clocks/__init__.py`
- Modify: `src/clocks/results.py`
- Modify: `tests/test_api.py`
- Modify: `tests/test_inference.py`

**Step 1: Write failing prior-support tests**

Test all initial and moved particles against position bounds, mass bounds,
strict first-coordinate ordering, and physical validity:

```python
def test_api_prior_ranges_are_actual_support():
    pf = build_particle_filter(make_config(n_masses=2, n_particles=2_000))
    for state in [pf.state, *(pf.update(obs) for obs in observations[:5])]:
        particles = state.particles
        assert np.all((-8.0 <= particles[:, :2]) & (particles[:, :2] <= 8.0))
        assert np.all((0.005 <= particles[:, 2:]) & (particles[:, 2:] <= 0.15))
        assert np.all(particles[:, 0] < particles[:, 1])
        assert np.all(api_physical_support_mask(particles, ...))
```

Add a test whose geometry/mass range has no valid conditional volume and assert
a `ValueError` naming the prior and weak-field policy. Add an observation length
mismatch test at `infer()` and `ParticleFilter.update()` boundaries.

Add model-comparison tests where filters have known fixed log evidence and a
uniform model prior, checking normalized posterior probabilities exactly.

**Step 2: Confirm failure**

```bash
uv run pytest tests/test_api.py tests/test_inference.py -k "support or prior_range or model_comparison or channel" -q
```

Expected: masses exceed `mass_range`, sorting mutates proposals, and API calls
still use removed jitter parameters.

**Step 3: Build one support predicate per fixed-K model**

The required `log_prior_density` contract has no `None`/implicit-flat mode.
The private API support predicate must combine:

```python
inside_positions = np.all((lo_x <= positions) & (positions <= hi_x), axis=(1, 2))
inside_masses = np.all((lo_m <= masses) & (masses <= hi_m), axis=1)
ordered = np.all(np.diff(positions[:, :, 0], axis=1) > 0.0, axis=1)
potentials = _point_mass_potential_batch(positions, masses, clock_array)
physical = np.all(np.abs(2.0 * potentials) <= WEAK_FIELD_LIMIT, axis=1)
valid = inside_positions & inside_masses & ordered & physical
```

For `K=1`, `ordered` is all true. `log_prior_density` is zero on this support
and `-inf` elsewhere.

**Step 4: Rejection-sample the normalized conditional prior**

Draw rectangular candidates in batches of `max(1024, 2 * remaining)`, sort
positions and their associated masses together only during initial sampling,
filter through the support predicate, and continue until `N` samples are
collected. Stop after 1,000 batches and raise a diagnostic `ValueError` showing
accepted/drawn counts and the weak-field limit. Do not sort during MH.

**Step 5: Remove duplicate model construction**

Change `ModelComparison` to accept a nonempty `dict[int, ParticleFilter]` and
an optional normalized `model_prior`; use uniform prior by default. Add public
`build_model_comparison(config)` and make `infer()` call it for tuple-valued
`n_masses`. Derive independent deterministic RNG streams with
`np.random.SeedSequence(config.seed).spawn(len(k_values))`.

Add `log_evidence` and per-update diagnostics to `InferenceResult` and
`HistoryEntry`, including `to_dict()` output. Preserve one result/history entry
per observation.

**Step 6: Verify API behavior**

```bash
uv run pytest tests/test_api.py tests/test_inference.py -q
uv run ruff check src/clocks/api.py src/clocks/inference.py src/clocks/results.py tests/test_api.py tests/test_inference.py
```

Expected: actual support, observation lengths, fixed-K inference, and model
comparison all pass.

**Step 7: Commit and push**

```bash
git add src/clocks/api.py src/clocks/inference.py src/clocks/results.py src/clocks/__init__.py tests/test_api.py tests/test_inference.py
git commit -m "refactor: make configured priors mathematically real"
git push
```

## Task 8: Retune scenarios into the weak field and make echolocation likelihood exact

**Files:**

- Modify: `src/clocks/_scenarios.py`
- Modify: `src/clocks/_animate.py:506-559`
- Modify: `tests/test_scenarios.py`
- Modify: `tests/test_viz.py`
- Modify: `tests/test_acceptance_multi_mass_2d.py`
- Modify: `tests/test_acceptance_echolocation_3d.py`
- Modify: `scripts/scan_multi_mass_2d.py`
- Modify: `scripts/scan_echolocation_range.py`

**Step 1: Add scenario-audit and contrast tests**

Add a helper assertion that every ground truth has
`max(abs(2 * potential)) <= 0.08`, leaving margin below the public `0.1`
boundary. Add tests for an orthonormal contrast matrix `Q`:

```python
def test_contrast_matrix_is_orthonormal_and_removes_common_mode():
    q = contrast_matrix(27)
    assert q.shape == (26, 27)
    np.testing.assert_allclose(q @ q.T, np.eye(26), atol=1e-14)
    np.testing.assert_allclose(q @ np.ones(27), 0.0, atol=1e-14)


def test_contrast_noise_retains_iid_variance():
    rng = np.random.default_rng(0)
    draws = rng.normal(0.0, 0.001, size=(100_000, 27))
    contrasts = draws @ contrast_matrix(27).T
    np.testing.assert_allclose(np.cov(contrasts, rowvar=False), 1e-6 * np.eye(26), rtol=0.04, atol=2e-8)
```

Test that filter observations have 26 channels while display observations stay
centered in 27 labeled clock channels.

**Step 2: Confirm the old scenarios fail the audit**

```bash
uv run pytest tests/test_scenarios.py -k "weak_field or contrast" -q
```

Expected: point-mass and density scenarios exceed the bound; the contrast
representation does not exist.

**Step 3: Apply initial physically valid constants**

Use these predeclared development values before any seed scan:

- one-mass 1-D truth `M=0.10`;
- one-mass 2-D truth `M=0.15`;
- two-mass 1-D/model-comparison truths `M=[0.045, 0.030]`;
- multi-mass 2-D truth `M=[0.050, 0.030]`;
- density truth `amplitude=0.010`;
- density prior `mu ~ U(-8, 8)`, `sigma ~ U(0.1, 5.0)`, and
  `amplitude ~ U(0.001, 0.030)`, further conditioned on weak-field validity;
- echolocation truth `M=0.080`, noise standard deviation `0.001`, mass prior
  `(0.005, 0.15)`; and
- general point-mass priors `(0.005, 0.15)` unless a narrower documented range
  is appropriate.

Keep positions and clock geometry initially unchanged. Update mass tolerances
proportionally but do not inspect certification results while choosing them.
Make `validate_echo_geometry` enforce the shared `WEAK_FIELD_LIMIT`, replacing
its independent `10*M` rule.

**Step 4: Replace centering-as-likelihood with contrasts**

Use `scipy.linalg.helmert(n_clocks, full=False)` for the `(C-1, C)` orthonormal
matrix. `make_echo_observations` returns simulation data, centered display
observations, and contrast-space filter observations. Echo forward models
return `Q @ rates` or `rates_batch @ Q.T`.

Extend `animate_echolocation` with a required keyword-only
`filter_observations`; use it for `_precompute_filter_states` and keep centered
`observations` for the plotted labeled channels. Assert both lists have the
same nonzero length.

**Step 5: Replace old tuning arguments and pins**

Change `run_multi_mass_2d` and `build_echolocation_filter` to accept/pass
`ess_target`, `rejuvenation_steps`, and `proposal_scale`. Update the scan grid
to compare only declared SMC controls, for example:

```text
ess_target:         0.7, 0.8, 0.9
rejuvenation_steps: 1, 2, 4
proposal_scale:     1.5, 2.38, 3.0
```

Keep development seeds `0-11`. Replace `--holdout` with a validated
`--seed-block` option (`0` for development; unseen multiples of 100 from 400
onward for certification), and change certification comments to the unseen
`400-411` block. Delete old jitter certification literals and clone-freeze
prose from tests and module docstrings.

**Step 6: Run fast scenario and visualization tests**

```bash
uv run pytest tests/test_scenarios.py tests/test_viz.py tests/test_acceptance_multi_mass_2d.py tests/test_acceptance_echolocation_3d.py -m "not slow" -q
```

Expected: all fast support, contrast, animation, and default-pin checks pass.

**Step 7: Commit and push**

```bash
git add src/clocks/_scenarios.py src/clocks/_animate.py tests/test_scenarios.py tests/test_viz.py tests/test_acceptance_multi_mass_2d.py tests/test_acceptance_echolocation_3d.py scripts/scan_multi_mass_2d.py scripts/scan_echolocation_range.py
git commit -m "fix: move shipped scenarios into the rigorous model"
git push
```

## Task 9: Package every demo command inside the wheel

**Files:**

- Create: `src/clocks/_demos/__init__.py`
- Create: `src/clocks/_demos/demo_1d.py`
- Create: `src/clocks/_demos/demo_2d.py`
- Create: `src/clocks/_demos/demo_multi_mass.py`
- Create: `src/clocks/_demos/demo_multi_mass_2d.py`
- Create: `src/clocks/_demos/demo_model_comparison.py`
- Create: `src/clocks/_demos/demo_density.py`
- Create: `src/clocks/_demos/demo_echolocation_3d.py`
- Rewrite: `scripts/demo_1d.py`
- Rewrite: `scripts/demo_2d.py`
- Rewrite: `scripts/demo_multi_mass.py`
- Rewrite: `scripts/demo_multi_mass_2d.py`
- Rewrite: `scripts/demo_model_comparison.py`
- Rewrite: `scripts/demo_density.py`
- Rewrite: `scripts/demo_echolocation_3d.py`
- Delete: `src/clocks/_cli.py`
- Modify: `pyproject.toml:16-24`
- Create: `tests/test_cli.py`
- Create: `scripts/check_wheel_entrypoints.py`
- Modify: `.github/workflows/ci.yml`

**Step 1: Write command dispatch tests**

Each packaged demo module must provide `build_parser()` and `main(argv: Sequence[str] | None = None)`.
Test every parser's help path without running a simulation:

```python
@pytest.mark.parametrize("module_name", DEMO_MODULES)
def test_demo_help_exits_without_running(module_name, capsys):
    module = importlib.import_module(module_name)
    with pytest.raises(SystemExit) as exc:
        module.main(["--help"])
    assert exc.value.code == 0
    assert "usage:" in capsys.readouterr().out
```

Add a test parsing `--output` and a small `--observations`/`--particles`
override for each command; this makes smoke paths testable without changing
default published configurations.

**Step 2: Confirm installed-layout failure**

```bash
uv build
uv run python -c "import zipfile, glob; z=zipfile.ZipFile(glob.glob('dist/*.whl')[0]); assert not any(n.startswith('scripts/') for n in z.namelist())"
uv run pytest tests/test_cli.py -q
```

Expected: packaged demo modules are missing and the old fallback still depends
on `scripts`.

**Step 3: Move implementations and leave thin wrappers**

Move each full implementation under `clocks._demos`, update it to the new SMC
configuration and Task 8 constants, and add the common parser flags. A source
wrapper contains only:

```python
from clocks._demos.demo_1d import main

if __name__ == "__main__":
    main()
```

Point `[project.scripts]` directly at `clocks._demos.<name>:main` and delete
`clocks._cli`. Each packaged module must select Matplotlib's `Agg` backend
before any `pyplot` import so default and smoke execution remain headless-safe.
The density module must rejection-sample the Task 8 conditional density prior
and provide its matching required `log_prior_density`; it must never pass an
out-of-domain row to the strict batch forward model.

**Step 4: Add an installed-wheel smoke script**

`scripts/check_wheel_entrypoints.py` must:

1. accept the wheel path and a Python executable;
2. create a `TemporaryDirectory`;
3. run `uv venv --python <python> <tmp>/venv`;
4. run `uv pip install --python <tmp>/venv/bin/python <wheel>`; and
5. invoke all seven installed executables with `--help`, requiring exit 0 and
   `usage:` in stdout.

Use `Scripts` rather than `bin` on Windows. Add CI steps:

```yaml
- name: Build wheel
  run: uv build
- name: Smoke-test installed entry points
  run: uv run python scripts/check_wheel_entrypoints.py dist/*.whl python3
```

**Step 5: Verify source and installed commands**

```bash
uv run pytest tests/test_cli.py -q
uv build
uv run python scripts/check_wheel_entrypoints.py dist/*.whl python3
```

Expected: every command prints help from the fresh environment and exits zero.

**Step 6: Commit and push**

```bash
git add pyproject.toml .github/workflows/ci.yml src/clocks/_demos scripts/demo_*.py scripts/check_wheel_entrypoints.py tests/test_cli.py
git rm src/clocks/_cli.py
git commit -m "fix: package all demo entry points"
git push
```

## Task 10: Update prose to match the corrected mathematics and commands

**Files:**

- Modify: `README.md`
- Modify: `docs/someday-maybe.md`
- Modify: `site/index.qmd`
- Modify: `site/method/the-particle-filter.qmd`
- Modify: `site/method/units-and-scales.qmd`
- Modify: `site/method/notation-and-glossary.qmd`
- Modify: `site/story/into-the-plane.qmd`
- Modify: `site/story/the-search-in-one-dimension.qmd`
- Modify: `site/story/two-hidden-masses.qmd`
- Modify: `site/story/how-many-masses.qmd`
- Modify: `site/story/beyond-point-masses.qmd`
- Modify: `site/story/gravitational-echolocation.qmd`
- Modify: `site/reproduce/getting-started.qmd`
- Modify: `site/reproduce/reproducibility.qmd`
- Modify: `site/reproduce/architecture.qmd`
- Modify: `scripts/check_site_links.py`

**Step 1: Add prose contract checks**

Extend `scripts/check_site_links.py` or add a focused pytest that rejects stale
tokens from current-facing prose:

```python
FORBIDDEN_CURRENT_CLAIMS = {
    "jitter_tau": "tempered SMC replaced annealed jitter",
    "jitter_std": "tempered SMC replaced annealed jitter",
    "mass_range shapes the initial sample only": "mass range is support",
    "perfectly symmetric ring would leave mirror-image": "labeled channels break this claim",
    "deliberately deep in the relativistic regime": "scenarios are weak-field",
}
```

Exclude historical design documents under `docs/plans` and
`docs/superpowers`; they are records, not current documentation.

**Step 2: Confirm stale claims are detected**

```bash
uv run python scripts/check_site_links.py
```

Expected: failure listing the old jitter, strong-field, prior, and mirror claims.

**Step 3: Rewrite the method and units explanations**

Document the exact `pi[t,beta]` target, ESS-selected beta increments,
pre-resampling evidence increments, symmetric MH acceptance rule, conditional
uniform support, and diagnostics. State that `|2 Phi| <= 0.1` is the project's
conservative policy and that `sqrt(1+2 Phi)` plus summed Newtonian potentials is
a pedagogical weak-field surrogate.

Do not claim SMC evidence is deterministic or exact at finite particle count;
call it a consistent Monte Carlo normalizing-constant estimate with sampling
error. Do not call fixed-seed acceptance scans population reliability evidence.

**Step 4: Correct geometry and echolocation prose**

Replace the mirror statement with: reflection permutes readings on a symmetric
layout; labeled channels generally distinguish the vectors. Explain that
echolocation uses `C-1` orthonormal contrasts, preserving iid variance and the
correct likelihood normalization after discarding common mode.

Update all executable examples to the new fields and weak-field constants.
Update the architecture diagram so console commands point to `_demos` and
source scripts are thin wrappers.

**Step 5: Render and check prose**

```bash
uv run python scripts/check_site_links.py
cd site && uv run --frozen quarto render
```

Expected: link/prose contract passes and all 15 pages render.

**Step 6: Commit and push**

```bash
git add README.md docs/someday-maybe.md site scripts/check_site_links.py
git commit -m "docs: align claims with rigorous physics and SMC"
git push
```

## Task 11: Calibrate on development seeds, freeze once, and regenerate evidence

**Files:**

- Modify: `src/clocks/_scenarios.py`
- Modify: `tests/test_acceptance_multi_mass_2d.py`
- Modify: `tests/test_acceptance_echolocation_3d.py`
- Modify: `assets/demo_1d.gif`
- Modify: `assets/demo_2d.gif`
- Modify: `assets/demo_multi_mass.gif`
- Modify: `assets/demo_multi_mass_2d.gif`
- Modify: `assets/demo_model_comparison.gif`
- Modify: `assets/demo_density.png`
- Modify: `assets/demo_echolocation_3d.gif`
- Modify: `assets/echolocation_range_study.json`
- Modify: `assets/echolocation_range_study.png`
- Modify: corresponding files under `site/assets/`

**Step 1: Run only development-seed scans**

Run the declared SMC grid on seeds `0-11`:

```bash
uv run scripts/scan_multi_mass_2d.py --workers 8 --per-run
uv run scripts/scan_echolocation_range.py --seed-block 0 --workers 8 --per-run
```

Record the complete results in a new dated section of the implementation PR or
a generated JSON report. Choose defaults by the predeclared order: highest pass
count, then lowest median normalized parameter error, then fewest forward-model
evaluations, then lower `rejuvenation_steps`. Do not inspect block `400` yet.

**Step 2: Freeze constants and fast guards**

Update scenario defaults and literal fast guards to the selected cell. Fix
acceptance tolerances before certification from scientific utility and the
development distribution; do not loosen them after seeing certification.
Commit this freeze before running block `400`:

```bash
git add src/clocks/_scenarios.py tests/test_acceptance_multi_mass_2d.py tests/test_acceptance_echolocation_3d.py
git commit -m "test: freeze rigorous SMC acceptance configuration"
git push
```

**Step 3: Run one-shot certification**

Run each unseen configuration exactly once on seeds `400-411`:

```bash
uv run scripts/scan_multi_mass_2d.py --seed-block 400 --workers 8 --per-run
uv run scripts/scan_echolocation_range.py --seed-block 400 --workers 8 --per-run
```

If a predeclared gate fails, report the failure; do not retune on this block.
Any new design starts a new development cycle and reserves block `500-511`.

**Step 4: Pin and run the slow acceptance tests**

Write the block-400 literal pins and expected gates, then run:

```bash
uv run pytest -m slow -q
```

Expected: both acceptance studies meet their frozen gates. The tests are
deterministic regression replays, not re-certification claims.

**Step 5: Regenerate all committed evidence**

Use the packaged commands with their default fixed seeds:

```bash
uv run demo-1d
uv run demo-2d
uv run demo-multi-mass
uv run demo-multi-mass-2d
uv run demo-model-comparison
uv run demo-density
uv run demo-echolocation-3d
uv run scripts/scan_echolocation_range.py --seed-block 400 --figure-only
```

Copy generated outputs byte-for-byte into both `assets/` and `site/assets/`.
Verify paired hashes:

```bash
for f in assets/*; do base=$(basename "$f"); test ! -e "site/assets/$base" || cmp "$f" "site/assets/$base"; done
```

**Step 6: Commit and push generated evidence**

```bash
git add assets site/assets tests/test_acceptance_*.py
git commit -m "chore: regenerate corrected scientific evidence"
git push
```

## Task 12: Run the complete verification matrix and independent review

**Files:**

- Modify only files required by verified failures or actionable review findings.

**Step 1: Run formatting and static checks**

```bash
uv run ruff format --check .
uv run ruff check .
git diff --check main...HEAD
```

Expected: all commands exit zero.

**Step 2: Run all automated tests**

```bash
uv run pytest -q
uv run pytest -m slow -q
```

Expected: default and slow suites pass with no unexpected deselections.

**Step 3: Verify lock, build, installed commands, and site**

```bash
uv lock --check
uv build
uv run python scripts/check_wheel_entrypoints.py dist/*.whl python3
uv run python scripts/check_site_links.py
cd site && uv run --frozen quarto render
```

Expected: lock/build/smoke/link/render commands all exit zero.

**Step 4: Run exact defect probes**

Run focused probes that would reproduce the original bugs:

```bash
uv run pytest tests/test_physics.py -k "domain or clamp or weak_field" -q
uv run pytest tests/test_inference.py -k "residual_resampling_does_not_clip or analytic or rejuvenation" -q
uv run pytest tests/test_cli.py -q
rg -n "jitter_tau|jitter_std|constraint_fn|support_bounds|np\.maximum\(argument|scripts\.demo" src tests site README.md
```

Expected: focused tests pass; the search is empty except any explicitly
allowlisted historical explanation in a regression test.

**Step 5: Request independent AGY review**

Use @superpowers:requesting-code-review and the repository's AGY protocol.
Create a clean detached review worktree at `HEAD`, verify its status and commit,
and ask Gemini to review:

- physical validity and scenario audit;
- SMC target, evidence increments, and MH invariance;
- prior normalization and model comparison;
- residual sampling statistics;
- installed-wheel behavior; and
- prose/implementation agreement.

Treat findings as leads: reproduce each one before changing code. An AGY
`LGTM`/`READY TO MERGE` counts as Jonathan's merge approval, but does not waive
local verification.

**Step 6: Address verified findings test-first**

For each real defect, add a failing focused test, make the minimal correction,
rerun the focused test, then rerun the complete matrix from Steps 1-3. Commit
review fixes separately:

```bash
git add <reviewed-files>
git commit -m "fix: address independent correctness review"
git push
```

**Step 7: Final clean-state evidence**

```bash
git status --short
git log --oneline main..HEAD
git rev-parse HEAD
```

Expected: empty status, intentional commit series, and the reviewed HEAD hash.
Do not merge or deploy without the AGY approval record.
