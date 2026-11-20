# Replay Data and Rebuilt Animations: Design

**Date:** 2026-11-19

**Status:** Approved for implementation. Codex xhigh round 3, on `75ef763`:
"SOUND ENOUGH TO IMPLEMENT", with three small findings that this revision folds
in.

**Source:** findings V1, V5 and C6 of the 2026-11-17 audit
(`docs/audits/2026-11-17/README.md`), the visual reviewer's critique
(`docs/audits/2026-11-17/reviewers/visual.md`, sections 2 and 3) and prototype
(`docs/audits/2026-11-17/prototype/`), and the owner's decisions of 2026-11-18
in the audit's "Decisions" section. Everything an implementer needs is restated
here.

Claims are marked. **Measured** means a command was run for this spec and its
output is in "Measurements" (M1 to M15). **Read** means it rests on reading
code or a document. **Believed** means neither.

## Purpose

Six demo GIFs and two PNGs are committed under `assets/`, mirrored under
`site/assets/` (33 MB in the tree), and pinned by SHA-256. The pin proves the
files have not changed, not that they are what the code produces, and it turns
every restyle into a change of "evidence". The animations also hide the
search: each opens after the first observation, when the cloud has already
collapsed; axes rescale every frame; position and mass share an axis; three
panels show the same plane; the weight colouring encodes nothing.

This track commits **replay data** instead: what each demo run produced, in
numbers. Tests pin the data to the code by its inputs and by bounded summaries
of what a reader sees. The site draws its animations and two static figures
from the data when it is built, in the figure style of the first track
(`docs/plans/2026-11-18-figure-style-design.md`), in both themes. The
animations start from the prior, keep every axis fixed, give each parameter its
own panel and dwell on the first updates.

## Scope

In scope: a replay archive per demo, its recorder and tests; the six animations
rebuilt and rendered at build time as H.264 video with a poster; the density
and range-study figures redrawn at build time; the `demo-*` commands, README
and Getting Started; the 2D demo's true mass, 0.15 to 0.10 (audit C6); the
provenance prose and its tests; the rendered-site check and source guard, with
two small fixes already owed; removal of the committed GIFs and PNGs.

Out of scope, each its own track: what the model-comparison page shows (a seed
spread, a one-mass case); the population-reliability study; the story rewrite,
page order, titles and URLs; any interactive or scroll-driven component; page
CSS beyond what video needs; audit findings C3 and T1 to T4. On story pages
this track changes only media embeds, their captions and descriptions, and
sentences that describe an animation's layout or, on one page, the 2D result.

## Design

### 1. Replay format

**One compressed NumPy archive per demo**, `assets/replays/<demo>.npz`, written
with `numpy.savez_compressed` and read with `allow_pickle=False`. The seven
names are the demo module names. The files sit beside
`assets/echolocation_range_study.json`, outside the package, and are not
mirrored under `site/`.

Each archive holds `meta`, a JSON document stored as a string (measured to
round-trip without pickle, M2), and arrays.

```json
{
  "format": "clocks-replay", "version": 1, "demo": "demo_2d", "kind": "filter",
  "seed": 42, "n_observations": 50, "n_particles": 2000,
  "noise_std": 0.005, "track_offset": 3.0, "param_names": ["x", "y", "M"],
  "prior_ranges": [[-8.0, 8.0], [-8.0, 8.0], [0.005, 0.15]],
  "controls": {"ess_target": 0.8, "rejuvenation_steps": 2,
               "proposal_scale": 2.38, "resampling": "systematic"},
  "cloud_size": 500,
  "created_with": {"python": "3.14.0", "numpy": "2.4.2", "scipy": "1.17.1",
                   "platform": "macOS-27.2.1-arm64"}
}
```

`kind` is `"filter"` or `"model_comparison"`. The model-comparison `meta` also
carries `true_k`. `created_with` records the platform, the resolved Python
version and the NumPy and SciPy versions of whatever machine ran the recorder.
It is provenance and no test reads it.

**The truth lives in the `true_params` array, not in `meta`.** Parameter order,
which is also the column order of `mean`, `cov` and `cloud`:

| Demo | `param_names` |
|---|---|
| `demo_1d` | x, M |
| `demo_2d` | x, y, M |
| `demo_multi_mass` | x₁, x₂, M₁, M₂, with x₁ < x₂ |
| `demo_multi_mass_2d` | x₁, y₁, x₂, y₂, M₁, M₂ |
| `demo_density` | μ, σ, A |
| `demo_echolocation_3d` | x, y, z, M |
| `demo_model_comparison` | `true_params` is x₁, x₂, M₁, M₂ of the K = 2 truth; `final_mean_k<K>` is x₁…x_K, M₁…M_K |

Arrays, with T observations, C clocks, P parameters, m displayed particles.
Every array is consumed by a drawing, by the command's printed output, or by
the gate or report of section 3.

| Key | Shape, type | Meaning | Demos |
|---|---|---|---|
| `clock_positions` | (C, D) float64 | clock coordinates | all |
| `true_params` | (P,) float64 | the truth | all |
| `true_rates` | (C,) float64 | noise-free tick rates | all |
| `observations` | (T, C) float64 | readings; centred for echolocation | all |
| `filter_observations` | (T, C-1) float64 | the contrasts the likelihood consumed | echolocation |
| `mean` | (T+1, P) float64 | weighted posterior mean; row 0 is the prior | filter |
| `cov` | (T+1, P, P) float64 | weighted covariance, no bias correction | filter |
| `ess` | (T+1,) float64 | 1 / sum of squared weights; row 0 is exactly N | filter |
| `log_evidence` | (T+1,) float64 | cumulative; row 0 is exactly 0 | filter |
| `tempering_stages`, `mh_proposals`, `mh_acceptances` | (T,) int64 | per-update diagnostics | filter |
| `cloud` | (T+1, m, P) float32 | equal-weight display sample | filter |
| `predicted_mean`, `predicted_std` | (T+1, C) float64 | weighted mean and s.d., over all particles, of each particle's predicted reading at each clock | `demo_1d`, `demo_multi_mass`, echolocation (centred) |
| `view_bounds` | (2,) float64 | the close view's cube | echolocation |
| `weight_in_view` | (T+1,) float64 | posterior weight inside that cube, from all particles | echolocation |
| `model_ks` | (K,) int64 | candidate counts, ascending | model comparison |
| `model_prior` | (K,) float64 | prior over counts, aligned with `model_ks` | model comparison |
| `model_posterior` | (T+1, K) float64 | columns aligned with `model_ks`; row 0 equals `model_prior` | model comparison |
| `model_log_evidence`, `model_ess` | (T+1, K) float64 | row 0 is exactly 0 and N | model comparison |
| `final_mean_k<K>`, `final_std_k<K>` | (P_K,) float64 | each candidate's final estimate | model comparison |

Decisions:

- **Row 0 is the prior.** `ParticleFilter.history[0]` holds it (read). Today
  `_precompute_filter_states` drops it, which is why every animation opens
  after the first observation. Row-0 sentinels (`ess`, `log_evidence`, the
  model rows) are written as exact values, not computed: 1 / Σw² for 3,000
  uniform weights is 3000.0000000000005 (measured, M5).
- **Predictions are recorded at the clocks only.** A dense tick-rate profile
  along the track cannot be computed: the prior is valid at the clocks, not
  between them, and on a 161-point grid 451 of 1,000 one-mass and 1,319 of
  3,000 two-mass prior particles leave the weak-field domain, so the strict
  forward model raises (measured, M3). The 1D clips draw at the clocks.
- **Summaries come from the whole cloud, in float64**; the square root of
  `cov`'s diagonal equals `ParticleFilter.estimate()["std"]` (read).
- **Clouds are thinned to at most 500 particles, in float32.** Whole clouds
  compress to 30.7 MB (float64) or 13.0 MB (float32); thinned, the seven files
  total 1.8 MB (measured, M2).
- **The model-comparison archive has no clouds** and stores each candidate's
  final estimate, so the command can print the winning model's estimate, as it
  does today. **The density archive has no predictions**; its forward model is
  a quadrature and inference alone takes 33 s (M2).
- **Renderers never call the clock-rate forward model.** They may evaluate the
  density parameterisation A·exp(-(x-μ)²/2σ²), which defines the parameters.
- **One file, not two.** A JSON file of summaries beside a binary file of
  clouds would diff readably but could fall out of step. A later in-browser
  replay can be fed by an export the build makes from the archive.

**The display sample.** For weights w (float64, non-negative, summing to
about 1) and m = min(500, N):

1. c = cumsum(w) / cumsum(w)[-1], so c is non-decreasing and ends at exactly 1;
2. targets t_j = (j + φ) / m for j = 0..m-1, with φ = (√5 - 1) / 2;
3. index_j = `searchsorted(c, t_j, side="left")`: the first particle whose
   cumulative weight reaches the target. A target exactly on a step selects the
   particle that ends there.

A particle of weight w is selected about m·w times, so heavy particles repeat;
one of zero weight is never selected; N need not divide by m; the result
depends on particle order, which is the filter's own. The irrational offset
matters: with offset 0.5 and uniform weights the targets land exactly on steps
and rounding decides each pick, where φ gives one stride with at least 5% of a
weight to spare (measured, M5). The sample uses no random numbers but is still
a discrete choice: moving 5.55e-17 of weight changed a pick in a constructed
case (M5), one reason it is reported on and not gated (section 3).

**Loading validates**: format and version; the exact key set for the `kind` and
demo, with no extra keys; each key's dtype and number of dimensions; row counts
of T+1 and T; inner dimensions against C, P, m and K; finite values; ascending
`model_ks`. A failure raises `ValueError` naming the file, the key and what was
expected.

**A literal model-comparison fixture** fixes what the table leaves open. With
one observation, two clocks and candidates K = 1 and 2:

```python
meta = {..., "kind": "model_comparison", "n_observations": 1,
        "n_particles": 16, "true_k": 2,
        "prior_ranges": [[-8.0, 8.0], [0.005, 0.15]]}
model_ks = [1, 2]
model_prior = [0.5, 0.5]
model_posterior = [[0.5, 0.5], [0.2, 0.8]]
model_log_evidence = [[0.0, 0.0], [-3.0, -1.6137056388801094]]
model_ess = [[16.0, 16.0], [12.5, 14.0]]
true_params = [-2.0, 3.0, 0.045, 0.030]          # x1, x2, M1, M2
final_mean_k1 = [0.4, 0.07]                      # x1, M1
final_mean_k2 = [-2.1, 3.2, 0.044, 0.031]        # x1, x2, M1, M2
```

`prior_ranges` here is the position range and the mass range, shared by every
candidate. The format test builds this archive from these literals.

### 2. Recording

A new private module `src/clocks/_replay.py` holds `Replay`, a frozen pair of
`meta` and `arrays`, and:

```python
def record_filter(pf, observations, *, meta, arrays, predict=None) -> Replay: ...
def thin(particles, weights, size: int) -> NDArray[np.float32]: ...
def validate_replay(replay: Replay, source: str = "<memory>") -> None: ...
def save_replay(path: Path, replay: Replay) -> None: ...
def load_replay(path: Path) -> Replay: ...
def first_difference(reference: Replay, other: Replay) -> Difference | None: ...
```

`arrays` carries what the scenario supplies (clocks, truth, true rates, and for
echolocation the two forms of the readings); the recorder adds the rest.

- `record_filter` raises before touching the filter if it has already seen an
  observation. `predict` maps particles to predicted readings and receives no
  generator.
- Each demo module gains two functions. `scenario()` is the packaged builder:
  it returns the clocks, the truth and the settings, and is the one source of
  geometry. The two-mass 2D demo generates its clocks from its seed there, so
  no module constant holds them. `record(*, n_observations=..., n_particles=...)
  -> Replay` runs the scenario and draws nothing.
- The model-comparison `record()` runs its comparison once. The command keeps
  its present path, which runs it twice and whose animator advances the
  filters itself, until pull request 3.
- **The density demo's readings get their own generator.** Today each reading
  is drawn from the generator the filter is using, between updates, so the
  data depend on how many random numbers the sampler consumed: drawing the
  readings first changes the second by 0.018 (measured, M4). Such readings are
  not scenario inputs and could not be compared across machines. The demo will
  draw its readings first from `study_generator(SEED)` and give the filter a
  second `study_generator(SEED)`, as the other six demos do (read). This
  changes the density run: its final estimate moves from (1.375, 1.985,
  0.0101) to (1.502, 2.136, 0.0095), against a truth of (1.5, 2.0, 0.010)
  (M4). No prose quotes the old numbers (read), and their figure is redrawn.
- `scripts/record_replays.py --out DIR` records all seven demos at their
  packaged defaults. Run with `--out assets/replays`, on whatever machine, it
  is how the committed archives are made. With `--compare DIR2` it also writes
  a report (section 3).

### 3. What tests pin

The byte hashes go. A fresh recording is **not** required to match the
committed archive value for value, because that cannot be relied on. The CI
runner printed one posterior in two variants 1.1e-13 apart across five builds
with nothing changed, and this Mac prints a third (measured, M1). The filter
and the display sample make discrete choices, and one ulp of weight can change
a resampled or displayed index (the reviewer's probe for both rules;
reproduced for the display sample, M5), after which two runs differ at the
level of Monte Carlo noise. So the gate uses quantities that do not depend on
which valid trajectory a machine takes, and full comparison is reported.

**The gate**, the same on every platform. A test calls the demo's `record()`
and compares it with the committed archive:

1. **Schema and inputs, exactly.** Key set, shapes and dtypes; `meta` except
   `created_with`; `clock_positions`, `true_params`, `model_ks` and
   `model_prior`. The inputs computed in floating point (`true_rates`,
   `observations`, `filter_observations`) are compared with `rtol=1e-12` and
   `atol=1e-12`: a library's last digit is not the scenario, and centred and
   contrast values near zero need the absolute term.
2. **Summaries a reader sees, within bounds**, after observations 1 and 8 and
   at the end: each component of the posterior mean, and each posterior
   standard deviation, within a stated absolute distance of the archive's. The
   early checkpoints are there so a materially different animation cannot hide
   behind a matching final frame.

Bounds on |fresh − archive|, per component:

| Demo | After | Mean: positions | Mean: masses | S.d.: positions | S.d.: masses |
|---|---|---|---|---|---|
| `demo_1d` | 1 | 0.095 | 0.0025 | 0.063 | 0.0027 |
| | 8 | 0.024 | 0.00091 | 0.015 | 0.00079 |
| | last | 0.0085 | 0.00034 | 0.0045 | 0.00026 |
| `demo_2d` | 1 | 0.48 | 0.0054 | 0.28 | 0.0020 |
| | 8 | 0.11 | 0.00059 | 0.062 | 0.00052 |
| | last | 0.025 | 0.00025 | 0.018 | 0.00014 |
| `demo_multi_mass` | 1 | 0.87 | 0.0095 | 0.19 | 0.0065 |
| | 8 | 0.060 | 0.0013 | 0.063 | 0.00079 |
| | last | 0.0079 | 0.00015 | 0.0058 | 0.00012 |
| `demo_multi_mass_2d` | 1 | 1.6 | 0.011 | 0.85 | 0.0071 |
| | 8 | 4.2† | 0.026† | 2.9 | 0.0096 |
| | last | 0.46 | 0.0027 | 0.47 | 0.0014 |
| `demo_echolocation_3d` | 1 | 0.28† | 0.021† | 0.043 | 0.0029 |
| | 8 | 0.012 | 0.00077 | 0.0097 | 0.00059 |
| | last | 0.0073 | 0.00047 | 0.0044 | 0.00027 |
| `demo_density` | 1 | μ 0.49, σ 0.38 | A 0.00050 | μ 0.38, σ 0.15 | A 0.00069 |
| | 8 | μ 0.067, σ 0.22 | A 0.0016† | μ 0.039, σ 0.20 | A 0.0031† |
| | last | μ 0.019, σ 0.027 | A 0.00011 | μ 0.014, σ 0.021 | A 0.000083 |

For `demo_model_comparison`: P(K=1) ≤ 1e-5 after observation 1 and ≤ 1e-20
after observation 8 and at the end; |P(K=2) − archive| ≤ 0.45 after
observation 1. The split between two and three masses is not bounded later.
The model-comparison archive's per-candidate final estimates
(`final_mean_k<K>`, `final_std_k<K>`) are checked for shape and finiteness
only, and are compared with the public API only at recording time, by
`tests/test_demo_records.py`.

Five invariants tie the arrays the drawings use to the rest of each archive.

- The echolocation close view equals `close_view` of the recorded rows.
- In every row, the weight in view is within max(0.01, 6√(p(1 − p)/m)) of
  the fraction p of displayed dots inside the view: six sampling errors of
  that fraction. The committed archive's worst gap is 0.03 of its bound
  (measured). Setting the prior row's 0.45% to 4% fails.
- In every row of `demo_1d`, `demo_multi_mass` and echolocation, the forward
  model is evaluated at each displayed dot (centred over clocks for
  echolocation, as recorded). At each clock the dots' mean reading is within
  6 `predicted_std`/√m of `predicted_mean`, and their standard deviation is
  0.6 to 1.5 times `predicted_std`. The worst rows reach 3.4 and 0.89 to 1.11
  (measured). The exception is echolocation's prior row, which is bounded at
  0.01 to 2.2: there the ratio runs from 0.019 to 1.43, because one of the
  6,000 prior particles, inside the head, carries 99.8% of the centre clock's
  predicted variance and no 500-dot sample holds it (measured). That row is
  not drawn. Shifting the first update's predictions by 0.02, or multiplying
  the final predicted standard deviations by ten, fails. The forward model is
  defined at every prior dot.
- For the same three demos, the final predictions are within half a
  `predicted_std` of the forward model at the final mean. This ties the
  predictions to the displayed estimate, which the dots check cannot:
  lowering echolocation's final mean mass by 0.00045 passes every other
  invariant and is 1.19 predicted s.d. here (measured).
- The covariance's correlations are within 0.4 of the dots'.

The last is partial: a reversed tilt is caught only where |r| exceeds about
0.2 (an axis ratio below about 0.82), so the near-round final ellipses of
`demo_1d` and `demo_2d` are not tied. A tighter bound was tried and does not
separate an edited row from the two-mass 2D archive's own unedited rows
(measured: Fisher z of 6.13 for the edit against 6.00 for the worst unedited
row).

- **Calibration.** Each bound is eight times the standard deviation of that
  quantity over inference seeds with the data fixed, the largest over the
  components it covers, rounded up: 20 seeds, 8 for density (measured, M6). A
  different seed is what a diverged trajectory looks like. No seed deviated
  from the demo's own by more than half a bound.
- **The `demo_2d` row holds at both truths**: it is calibrated at M = 0.10,
  where every spread is the larger.
- **† marks a bound no tighter than the posterior's own width there.** The
  two-mass 2D cloud wanders between modes early, the echolocation cloud is
  still settling after one reading, and the density's σ–A ridge is unresolved
  after eight. There the gate says little and the other checkpoints carry it.
- **Model comparison is bounded only where it is stable.** Over 20 seeds
  P(K=2) ran from 0.27 to 0.45 after one observation and from 0.18 to 0.84 at
  the end (M6; audit finding C2). One mass is excluded at every checkpoint in
  every seed.
- **What the early checkpoints buy.** With the likelihood's noise doubled for
  the first three updates of `demo_1d` only, the final summaries move by
  2.3e-4 against a bound of 0.0085 and pass; the standard deviation after
  observation 1 moves by 0.86 against 0.063 and fails (measured, M15).

**Invariants on the committed archive**, with no inference:

- the files are exactly the seven names; each loads; `meta` and the geometry
  equal the demo's `scenario()`;
- `true_rates` equals the forward model at `true_params` (1e-12). This holds
  for the density demo too, and involves no readings;
- each echolocation reading sums to zero within an absolute 1e-12, and
  `filter_observations = Q @ observations` with `rtol=1e-12, atol=1e-12`;
- every `model_posterior` row equals `model_prior × exp(model_log_evidence)`
  normalized, computed in log space, with `rtol=1e-9, atol=1e-12`. That is the
  relation the inference code defines (read), and it catches an archive whose
  probabilities disagree with its evidence while still summing to 1;
- the truth meets the existing 0.08 weak-field margin at the clocks, and
  x₁ < x₂ where there are two masses;
- sentinels, exactly: `log_evidence[0] = 0`, `ess[0] = N`,
  `model_log_evidence[0] = 0`, `model_ess[0] = N`, `model_posterior[0] =
  model_prior`;
- with roundoff allowed: 0 < `ess` ≤ N(1 + 1e-12); `cov` symmetric within
  1e-12 of its largest entry, with no eigenvalue below −1e-12 times its trace;
  `model_posterior` rows sum to 1 within 1e-12; `weight_in_view` in
  [0, 1 + 1e-12]; acceptances ≤ proposals;
- the dots agree with the summaries: in every row, the display sample lies
  inside `prior_ranges` with the endpoints rounded to float32 first (0.15
  rounds up as float32, and rounding is monotone), its mean is within 6
  s.d./√m of `mean`, and its
  standard deviation is within 0.6 to 1.5 of the recorded one. The candidates
  reach 3.0 and 0.78 to 1.29 (measured, M6).

**The report, which never fails anything.** A separate workflow,
`replay-report.yml`, runs on pull requests and pushes to `main` that touch the
library, the lock, the archives or the recorder. Three jobs from one matrix
check out the same tree on three runner allocations, so one run gives three
samples without repeating the test suite. Each job records all seven demos,
compares each with the committed archive in full, and uploads a short report:

```yaml
- name: Replay report
  continue-on-error: true
  run: uv run python scripts/record_replays.py --out recorded --compare assets/replays --report replay-report.md
- uses: actions/upload-artifact@v4
  continue-on-error: true
  with: {name: "replay-report-${{ github.run_id }}-${{ matrix.sample }}", path: replay-report.md}
```

Every step may fail without failing the job, the upload included. A last step
adds the report to the job summary, or a warning annotation if no report was
written. For each demo the report says "reproduced" (integers equal, float64
arrays within `rtol=1e-9`, `cloud` within 1e-5) or gives the first differing
key and row; `created_with` is ignored, so a change of platform still reaches
the arrays. It records the run ID, the pull request's head SHA, the SHA
checked out, the platform, the resolved Python version and the NumPy, SciPy
and BLAS versions. The CI test job is not touched and keeps the Python it has
(3.14.4 today; M13).

Pull request 1's description carries the three reports of one run and the same
comparison from macOS. If a demo reproduces in all of them, a later change may
promote its full comparison to a gate. This spec does not.

**Cost.** Inference takes 0.2 to 4.1 s for six demos and 33 s for density
(measured, M2). The four cheapest gate tests run in the default suite, which
takes 32 s today (M13); the other three are marked `slow`. Each report
job takes about a minute beyond its setup.

**Against the old hashes.** They caught any change to the committed bytes
(corruption, a wrong file under a right name, drift between the two copies)
and nothing about the code. Now git holds the bytes, the second copy is gone, a
wrong archive under a right name fails the input comparison, and a stale or
edited archive fails the bounds when a reader could see the change. An edit
smaller than seed-to-seed variation passes the gate and shows in the report.
No loss that matters to a reader of the site was found.

The calibration report keeps its hash table as the record of the August release
and gains a dated note that the files were removed and are in git history. The
row for `echolocation_range_study.json` is unaffected. `EXPECTED_VISUAL_HASHES`
loses its two PNG rows in pull request 2 and the rest in pull request 3.

### 4. How and when media is rendered

**A pre-render script.** Quarto runs `site/prerender.py`, three lines over a
library function, before any page. It writes clips and posters to `site/media/`
(git-ignored). `_quarto.yml` names it under `project.pre-render` and lists
`media/*.mp4` and `media/*.png` under `project.resources`.

Measured with Quarto 1.9.32 (M11): the script runs in the project directory
under the `uv run` Python, sees `QUARTO_PROJECT_RENDER_ALL=1` on a full render,
and its outputs are copied to the site. Files written by a cell are not copied
on a clean build even when `resources` names them, which rules out rendering
in a cell.

**Rebuild on change.** A full render rebuilds every file and deletes anything
else in `site/media/`. A single-page render or preview rebuilds a clip when its
output is missing or older than its archive or than any file under
`src/clocks/`. The script prints the encoder's path and version.

**Encoder.** H.264 through `imageio-ffmpeg`, added to the `dev` group in pull
request 3. The Ubuntu 26.04 and 24.04 runner images' software lists do not
mention FFmpeg (read, M13). Version 0.8.1 has a `manylinux_2_28_x86_64` wheel
and its macOS binary is FFmpeg 9.0 with `libx264` (measured, M10); the Linux
binary was not run. The library looks for `imageio_ffmpeg`, then `ffmpeg` on
`PATH`, then raises with instructions; it does not depend on either.

`src/clocks/_video.py` pipes raw frames to the encoder: 20 frames per second,
`libx264`, `yuv420p`, CRF 20, preset `slow`, `+faststart`, no audio. A frame is
drawn once and written as many times as its dwell requires.

**Build time and size.** A full local render takes 38 s today; the same step
took 43 s and 61 s on the runner (measured, M13). Prototype clips at 1500
pixels wide: 2D, 2.1 s per theme and 330 kB; echolocation in the round 1
layout, 12.5 s per theme and 1.18 MB (M9). All six in both themes should take
about a minute here and under two on the runner, against a 10-minute limit.
Rendering is serial; pull request 3 measures the cold runner first.

**Pixel sizes.** `yuv420p` needs even dimensions, and a figure 4.6 inches tall
at 200 dpi is 919 pixels (measured, M9). Canvas heights come from those
measured to be exact (4.2, 4.5, 5.0, 5.4, 5.5 inches); a test asserts each.

**Markup: native controls, no script.** Each embed is raw HTML in the page:

```html
<figure class="replay">
<video class="light-content" aria-label="Animation: the 2D search" aria-describedby="demo-2d-text" src="../media/demo_2d.light.mp4" poster="../media/demo_2d.light.png" width="750" height="450" controls playsinline preload="none"></video>
<video class="dark-content" aria-label="Animation: the 2D search" aria-describedby="demo-2d-text" src="../media/demo_2d.dark.mp4" poster="../media/demo_2d.dark.png" width="750" height="450" controls playsinline preload="none"></video>
<figcaption>The caption, once.</figcaption>
</figure>

::: {#demo-2d-text}
Before any reading the dots cover the whole search area. … It ends at
x = 1.42 ± 0.11, …
:::
```

- Nothing plays until the reader presses play: no `autoplay`, no `loop`, no
  script, so reduced motion needs no special case. The poster is the final
  frame.
- The theme swap reuses Quarto's classes, whose rule is the plain selector
  `body.quarto-dark .light-content { display: none !important }`; the static
  HTML's body class is `quarto-light` (measured, M11).
- In headless Chrome 157 with this markup (M12): both posters are requested,
  no video file is requested with JavaScript on or off, the box is 750 by 450
  before any media loads, and switching theme swaps the visible video.
- **One visible description per clip**, in the element both videos name with
  `aria-describedby`. It says what the prior looks like, the change that
  matters in the first readings, where the run ends, and what the final
  uncertainty means. For echolocation it names the collapse at the first
  reading and says the fitted-against-observed panel shows agreement with the
  readings used in the fit. On "How Many Masses?" the paragraph already under
  the figure is the description: it tells the sequence, and the archive agrees
  with it (the lead passes to two masses at observation 10 and back to three
  at 19; M6; the test derives the lead changes and the ending from the
  archive). Numbers in a description are checked against that clip's archive
  and nothing else. The test reads only the described element, so the numbers
  of a page's executed cell, a different run, are not touched.
- Seven embeds are written by hand and verified by the rendered-site check.
  Markdown image syntax was tried and offers no poster.
- `site/styles.css` gains `figure.replay video { max-width: 100%; height:
  auto; }`. Without it a narrow page scrolls sideways (measured at 500 pixels,
  M12).

### 5. The rebuilt animations

Shared conventions:

- **Canvas** 750 CSS pixels wide, Source Sans 3 through `style(theme)`,
  constrained layout off. Every axes rectangle and limit is computed once from
  the archive and fixed for the clip.
- **Status line** top left: "Before any observation", then "After k
  observations". Direct labels, no legend boxes.
- **Colour roles.** Particles are `belief` dots of one size and opacity. A
  second mass is `belief_alt`, drawn with a different marker and labelled, so
  overlapping clouds stay apart without relying on hue. The truth is `truth`.
  Clocks and observed data are `ink`. No clip has a threshold, so none has a
  dashed line.
- **Traces.** One panel per parameter: mean, a band of one standard deviation,
  the truth as a solid line, x from 0 to T. The header gives the name, "true
  1.50" and "1.42 ± 0.11". The y-limits are the smallest interval holding the
  truth and the mean ± one standard deviation for k ≥ 1, widened 8% and clipped
  to the prior range. **The prior row is left out of the limits on purpose**:
  its band is clipped, its values are in the header, and the cloud panel is
  where the collapse is seen.
- **Ellipse.** Drawn from `cov` at two standard deviations and labelled
  "covariance, 2 s.d.". It is a summary of spread, not a region of stated
  probability, and the label does not say otherwise.
- **Zoom inset** for one-mass clouds: a fixed window centred on the truth, with
  half-width 1.25 × (final offset from the truth + two final standard
  deviations) per axis, labelled with its magnification.
- **Pacing** at 20 frames per second: the prior holds 1.2 s; observations 1 to 3
  hold 0.9 s; 4 to 8, 0.5 s; 9 to 20, 0.25 s; later ones 0.1 s; the last 2.5 s.
  That is 14.8 s for 50 observations and 17.8 s for 80.

Only the 2D clip and the echolocation frames have been drawn. Pull request 1
smoke-draws every other panel from candidate archives before the format is
fixed; pull request 3 draws and times every clip, in both themes at full width
and 390 pixels, before any layout number here is treated as final.

**`demo_1d`, 750 by 450.** Left, upper: the cloud in the (x, M) plane on the
prior ranges, with the truth marker, ellipse and inset. Beneath it, on the same
x scale, a strip of `ink` squares marks the clocks. Left, lower: at each clock,
on the same x scale, the slowing (1 - rate, labelled so): the true value as a
`truth` tick; the posterior-predicted mean with a ± one standard deviation bar
in `belief`; and the running mean of the readings with a ± σ/√k bar in `ink`,
the three side by side. Before any observation the `ink` marks are absent and
the `belief` bars show the prior's prediction. Right: traces for x and M.

**`demo_2d`, 750 by 450.** Left: one square map on the prior range with the
cloud, clocks, truth marker, ellipse and inset. Each clock's fill runs from
`background` to `ink` with the running mean of its slowing, on a scale from 0
to the largest running mean in the whole run (two early clock-frames exceed 1.1
times the final maximum; M8), with the key "stronger fill = slower clock";
clocks are hollow before any observation. Right: traces for x, y and M. The
audit prototype's orange ramp is not used: the palette has none, orange is the
truth's hue, and observed data are `ink`.

**`demo_multi_mass`, 750 by 500.** As `demo_1d`, with both clouds in the
(x, M) panel, each truth labelled "mass 1" or "mass 2" (the masses are ordered
by x), one inset per mass, and four stacked traces.

**`demo_multi_mass_2d`, 750 by 550.** One map with both clouds, both labelled
truths, both ellipses and the ten clocks; six traces in three rows and two
columns, one column per mass. No inset: the final standard deviations are 0.2
to 0.8 on a 16-wide axis (M2). The cloud passes through a second mode, which an
ellipse cannot show; the dots do.

**`demo_model_comparison`, 750 by 420.** One panel: posterior probability of
each count against observations, three lines growing from the model prior at
k = 0, in `ordered[0..2]`. Labels sit in a fixed gutter to the right, in a
fixed order, each joined to its line's head by a thin connector, so crossing
lines cannot stack them. The true count's label reads "(true model)" and comes
from `true_k`. Whether this page keeps an animation is the third track's
decision; meanwhile the same run stays one and only its form changes.

**`demo_echolocation_3d`, 750 by 550.** Drawn for this revision at k = 0, 1, 8
and 80 in both themes, and at 390 pixels (measured, M7).

- **Close view**, a 3D scene on one fixed cube: the smallest cube holding the
  head, the truth, and each coordinate's mean ± four standard deviations for
  every k ≥ 1, padded by 0.3 and rounded outward to 0.1. The value is multiplied by
  10 and that product rounded to six decimals before the floor or ceiling, so
  that 14.000000000000002 stays 14. For this run that is -1.3 to 4.8. It shows the head wireframe (`muted`), clocks (`ink`), the cloud
  (`belief`) and the truth as a ring with a drop-line to the floor. The ring is
  open and several times wider than the final cloud, which is visible inside
  it in the drawn frames.
- **Whole prior box**, a small fixed panel at upper left: the display sample
  projected on x and z over the prior's ±16, with a square marking the close
  view. Here the collapse is visible: the box is full at k = 0 and holds one
  small cluster at k = 1.
- **Weight in view**, under the status line, from `weight_in_view`: "0.4% of
  the prior lies inside the close view", then "100.0% of the cloud's weight
  lies inside the close view" from the first observation on (measured, M7).
- **Traces** for x, y, z and M, stacked at right.
- **Observed against fitted**, a small square panel on equal scales: for each
  of the 27 clocks, the running mean of its centred reading against
  `predicted_mean`, with the diagonal drawn. It is empty at k = 0 and says "no
  readings yet". Its axes are labelled "observed mean" and "fitted", and the
  caption calls it agreement with the readings used in the fit, not a test on
  new data.
- **Camera.** This is the one clip drawn on every video frame; the others draw
  each observation once and repeat it. With F video frames before the final
  hold, frame f has azimuth -75° + 40° × (1 - cos(π f / (F - 1))) / 2 and
  elevation 18°; the camera is still during the hold. Frames were drawn for a
  60-degree turn first; its last frame looks nearly along the x-axis and
  flattens the head.

### 6. The two static figures

Both become plots drawn by an executed cell, like the nine existing plots:
`renderings: [light, dark]`, a two-entry caption, `style(theme)`, `show(fig)`.
The drawing is in the library because a command and a script draw the same
figures. A cell's working directory is its page's directory (measured, M11).

**Density**, 750 by 540, from `assets/replays/demo_density.npz` through
`density_figure(replay, colors)`. Top: the true profile in `truth`; in
`belief`, the median and central 68% band of the 500 sampled profiles of the
last frame, evaluated point by point along the track and labelled "pointwise
median and 68% band"; clocks as `ink` squares. Below: traces for μ, σ and A.
Declared changes: a median with a band where there was the profile at the
posterior mean; no clock-rate panel; one axis per parameter; site-wide colours.

**Range study**, 750 by 540, from `assets/echolocation_range_study.json`
through `range_study_figure(study, colors)`. Two stacked log panels, position
error above mass error. Measured errors are `ink`: each run a faint dot, the
median a line with circles. The filter's own claim, the median posterior
standard deviation, is a solid `belief` line with squares, labelled once. Each
panel's frozen tolerance, read from the study, is a dashed `threshold` line.
Pass counts computed from the records run along the top, labelled "runs within
both limits". No title.

`write_summary_figure(study, png_path)` stays as a thin wrapper over
`range_study_figure`. Both scan paths, `tests/test_echo_study.py` and a
monkeypatch in `tests/test_scenarios.py` use that name (read); keeping it
leaves all four untouched.

### 7. The demo commands and public names

Each `uv run demo-*` command records its run and prints its truth and
estimates from the replay. `uv run demo-2d` writes `output/demo_2d.npz` and
`output/demo_2d.png`; `--video` adds `output/demo_2d.mp4`.

The PNG is the final frame (for `demo-density`, the density figure) and needs
only Matplotlib. `--video` needs an encoder, which `uv sync` provides in a
checkout; without one the command stops and says how to get one.
`demo-density` has no `--video`. `--output-dir` replaces `--output`; `--theme`
is new. Video is opt-in so the default works from a bare wheel install.

**This is a break, and it is documented.** `clocks.viz` calls itself a public
facade and exports the old names, although none is in `clocks.__all__` and only
the demos and `tests/test_viz.py` use them (read). Pull request 3 bumps the
version to 0.2.0 and adds a short "Upgrading from 0.1" note to the README:

- removed from `clocks.viz`: the six `animate_*` functions, the eleven `plot_*`
  functions and the two `create_*_dashboard` functions; added: `load_replay`,
  `render_clip`, `density_figure`, `range_study_figure`;
- `--output PATH` becomes `--output-dir DIR`;
- demos write a replay and a PNG where they wrote a GIF; MP4 is opt-in.

`scripts/check_wheel_entrypoints.py` gains one run: from the installed wheel,
with no encoder and outside a checkout, `demo-1d` with two observations writes
its archive and PNG, and with `--video` exits non-zero naming the encoder.

README and Getting Started show the new commands; the README's GIF and its
paragraph about committed media are replaced by a link to the site and two
sentences on replay data. The Architecture page's module list and
`docs/someday-maybe.md`'s remark about static GIFs are corrected.

### 8. The 2D truth change

`TRUE_M` in `demo_2d.py` goes from 0.15 to 0.10, in pull request 3. It touches:

- `assets/replays/demo_2d.npz`, re-recorded, and the clip on the landing page
  and "Into the Plane". Pull request 1 records the 0.10 run as a probe and
  reports it without committing it;
- the cell at `site/story/into-the-plane.qmd` line 49. Measured (M8): its
  posterior moves from (1.405, -1.077, 0.149), with a largest particle mass of
  exactly 0.1500, to (1.372, -1.098, 0.100) ± (0.172, 0.170, 0.002), with a
  largest particle mass of 0.1068;
- that page's sentence that the cloud collapses "to a tight knot at the truth".
  At the new value the cell's mean is 0.16 from the true position with a spread
  of 0.17, so the sentence is rewritten to say the cloud ends as a small knot
  beside the true position, offset by about its own width;
- `tests/test_shipped_ground_truths.py`, which reads the constant; the
  weak-field strength falls from 0.070 to 0.047;
- the audit's status table.

No other prose quotes 0.15 as this demo's truth (read). In the 50-observation
run the final estimate is x = 1.417 ± 0.111, y = -1.222 ± 0.115, M = 0.1000 ±
0.0012 (M2): y is 1.9 standard deviations from -1.0, as it was at the old value
by the audit's numbers. That is seed 42's noise, and the description states
the numbers.

**A box-margin guard**, new: every shipped continuous truth value lies inside
its prior box by at least 5% of the box's width, including the K = 2 truth of
the model-comparison demo. It guards finding C6 only. Physical support and
ordering keep their own tests (section 3).

### 9. Provenance prose

`site/reproduce/reproducibility.qmd` keeps the anchor `#sec-figures` and the
section is retitled "Where the figures come from".

- The link to the calibration report is labelled as the record of the August
  2026 release, not of the current figures.
- **Replay data** (replaces "Demo animations and the density figure"): each
  animation and the density figure is drawn at build time from
  `assets/replays/<name>.npz`, written by the recorder: configuration, seed,
  observations, a summary of the cloud before any observation and after each,
  and, except for the model-comparison run, a 500-particle sample for display.
  Tests run each demo again, compare its inputs exactly and compare its
  summaries within stated bounds; they do not require the same random path.
  Video files are build products. Each archive is one finite realization.
- **Echolocation animation**: the opening sentences calling the GIF an old
  release whose panel "will change when the assets are next regenerated" are
  deleted; the sentences on two circumradii and the most favourable range stay.
- **Model-comparison runs**: the two sentences below.
- The CI paragraph stops calling the media exceptions.
- **"Run exactly once."** By the audit's decision this becomes a statement of
  protocol where it survives (this page, Getting Started, README,
  `method/the-particle-filter.qmd`): each block is reserved for one run, and
  block 400 to 411 was run under the frozen settings. A sentence about blocks
  100 to 300 is added only with a citation the implementer has checked; this
  spec has only the audit's word for it.

Sentences pinned by `tests/test_generated_assets.py`:

| Today | Becomes |
|---|---|
| "The committed GIF on [How Many Masses?](…) is the packaged default run: 80 observations, 2,000 particles per candidate, and seed 42. " | "The animation on [How Many Masses?](…) replays the packaged default run: …", with the three numbers from `meta` |
| "With K=2 truth, its final estimate is K=2: 0.3990 and K=3: 0.6010. " | the same words, with the true count from `true_k` and each probability looked up by K in `model_ks` |
| the range-study, 25-observation and horizon sentences | unchanged |

The recorded run gives 0.39903515 and 0.60096485 (M2). Each number is checked
in the sentence that names its run; the 25-observation cell's "K=2: 0.740" in
the same section belongs to another run and is left alone. The story pages'
forbidden provenance vocabulary gains "committed replay" and "replay file".

### 10. The rendered-site check and the source guard

`scripts/check_site_figures.py` keeps every check it has (one image per
wrapper, declared size, files inside the site, no nesting, no element with both
theme classes, equal captions) and learns an ordered manifest: for each page, a
tuple of `Fig(kind, height, clip)` in document order.

| Page | Figures, in order (declared height) |
|---|---|
| `index` | clip `demo_2d` (450) |
| `clocks-as-gravimeters`, `one-clock-is-not-enough` | plot (420) |
| `the-search-in-one-dimension` | clip `demo_1d` (450), plot (540) |
| `into-the-plane` | clip `demo_2d` (450), plot (540) |
| `two-hidden-masses` | clips `demo_multi_mass` (500) and `demo_multi_mass_2d` (550), plot (420) |
| `how-many-masses` | clip `demo_model_comparison` (420), plot (420) |
| `beyond-point-masses` | plot (540, the density figure), plot (420) |
| `gravitational-echolocation` | plot (420), clip `demo_echolocation_3d` (550), plot (540, the range study) |
| `the-particle-filter` | plot (540) |

`Fig` is a `NamedTuple`: the tests import the script by path without
registering it, and a dataclass with postponed annotations then fails at import
(measured, M14).

- A page's themed elements, in document order, must be exactly its manifest. A
  themed `<video>` inside `<figure class="replay">` is a clip rendering;
  anything else is a plot wrapper. Every unlisted page must have none (the six
  pages outside today's list have none; M14).
- **Plots:** wrappers alternate light, dark; each pair is checked as today, and
  the two image files must differ in content, not only in name.
- **Clips:** one video per theme and one non-empty `<figcaption>`; declared 750
  by the listed height; `controls`, `playsinline`, `preload="none"`, a non-empty
  `aria-label`, and an `aria-describedby` naming a non-empty element on the
  page, the same for both themes; no `autoplay`; `src` and `poster` resolving
  inside the site to `<clip>.<theme>.mp4` and `.png`; the two themes' files
  differing in content; the poster's pixel size twice the declared size. The
  rendered stylesheet must contain the responsive rule.
- **The video stream is decoded** with the encoder: H.264, `yuv420p`, the
  frame's pixel size, 20 frames per second, the frame count the dwell schedule
  gives, and `moov` before `mdat`. A file cut in half keeps its `ftyp` box and
  box order but decodes to 97 of 296 frames (measured, M10), so the count is
  what catches truncation.
- **Folded in (pull request 2):** a path whose escapes decode to a NUL makes
  `_resolve_source` raise (measured, M14); it will return a failure message.

`tests/test_site_figure_style.py`:

- **`data=` by position (pull request 2).** `ax.plot(x, "rate", "r-",
  data=frame)` passes today although its third argument is a format string
  (measured, M14). With `data=` only the first two positional arguments may be
  strings; without it, none.
- **Plot cells** are cells that declare `renderings` *or* call a plot producer:
  `plt.subplots`, `plt.figure`, `show`, or a `clocks.viz` function named
  `*_figure`. Every such cell must meet the cell contract.
- **The manifest matches the source:** for each listed page, the order of plot
  cells and replay figures in the `.qmd` equals the manifest's.
- **Library drawing code** passes the same syntax-tree colour check as page
  cells. A colour passed through a variable escapes it; the artist tests cover
  that.
- **Retired assets.** No page or README links to or embeds a retired file
  under `assets/` or `site/assets/`: the two PNGs from pull request 2, the six
  GIFs from pull request 3. A bare file name such as the command's
  `output/demo_density.png` is not a link and is allowed.

The site-check workflow's path filter gains `assets/replays/**` and the study
JSON.

### 11. Sequencing

Three pull requests. Each leaves the site correct, and none merges a renderer
that nothing calls.

1. **Replay data.** `_replay.py`; `scenario()` and `record()` in each demo;
   `scripts/record_replays.py`; the seven archives at today's truths; the gate
   and invariants; the report workflow. Every command keeps its present path
   and output: the demos still write GIFs, and `demo-density` keeps its shared
   generator until pull request 2, so each committed file still matches its
   command. Before the format is fixed, every undrawn panel is
   smoke-drawn from candidate archives with throwaway code, and the sheet goes
   in the pull request with the reports of section 3 and a probe recording of
   the 2D run at 0.10.
2. **The two static figures.** `density_figure` and `range_study_figure` with
   their page cells and captions; `demo-density` switching to `record()`, the
   separate generator and the new figure in the same change as the hash rows
   and provenance sentences; the check and guard learning several plots per page,
   with the two owed fixes (NUL, `data=`), each with its own test; removal of
   the two PNGs, their mirror and hash rows; the ban on links to them; the
   provenance sentences about them.
3. **The clips and the switch.** First on the branch: every clip drawn and
   timed, a clean build on Quarto 1.11.9, and cold-runner timings, all in the
   pull request. Then `_video.py`, the clip builders and the encoder
   dependency; the demo commands and migration note; the 2D truth with its
   archive, clip, story cell and the box-margin guard; the pre-render script,
   embeds and descriptions; clip support in the check and guard; the rest of
   the provenance prose; removal of the GIFs, `_animate.py`, the old panel
   functions and their tests.

If drawing shows the archive lacks something, the format version is raised and
the archives re-recorded in that pull request.

## Testing

Each negative case asserts its specific message, so a checker that rejects
everything, or nothing, fails.

**Format** (`tests/test_replay.py`)

- The model-comparison fixture of section 1 and a filter fixture are written by
  the test with plain NumPy from literals and load to those values; written by
  `save_replay`, they read back with plain NumPy equal to the literals.
- For every key: a wrong dtype, number of dimensions or inner dimension raises,
  as do a missing key, an extra key, a wrong version, a non-finite value and
  descending `model_ks`.

**Recorder**

- On a hand-built history with correlated parameters and unequal weights in
  every row, `mean`, the whole of `cov` and `ess` equal moments computed
  independently in the test, row by row; row 0 holds the sentinels.
- With a non-linear `predict`, the recorded moments equal the independently
  computed weighted moments of the predictions, and differ from the prediction
  at the mean particle.
- A recorded run and the same run without the recorder, on a filter whose
  updates demonstrably resample and accept moves, give the same particles and
  weights in every history row and the same final generator state.
- A filter that has seen an observation raises, and its state is unchanged.
- The density demo's readings are identical at two particle counts.
- `thin`: weights (0.1, 0.2, 0, 0.3, 0.4) with m = 5 give indices 1, 3, 3, 4,
  4; seven uniform weights with m = 3 give 1, 3, 6; four with m = 4 give 0, 1,
  2, 3 (measured, M5). Offset 0.5 gives 0, 1, 3, 4, 4 and 1, 3, 5 on the first
  two, and a plain stride gives neither.
- `first_difference` returns nothing for equal replays and the earliest key
  and row for a perturbed one.

**Committed data** (`tests/test_replays.py`): the gate and the invariants of
section 3, and from pull request 3 the box-margin guard.

**Figures** (pull request 2)

- On synthetic inputs: the plotted median-error and posterior-spread lines
  equal the values from `summarize`; tolerance lines sit at the study's
  tolerances; pass counts equal those computed from the records; the density
  band's edges equal independently computed percentiles.
- Changing one input value moves the corresponding artist.
- Every visible artist's colour is a palette colour for the theme.

**Clips** (pull request 3), on small synthetic replays and the committed ones

- Every axes has the same rectangle and limits on every frame, and the frame
  is 1500 by twice the declared height.
- On sampled frames including middle ones, each cloud artist is visible, with
  non-zero size and opacity and no opaque artist over it, x on x and y on y.
  In map and (x, M) panels and the echolocation overview it holds the whole
  archive row; in the echolocation close view, exactly the row's points inside
  `view_bounds`. Frame 0 shows row 0.
- The dwell schedule equals the literal list for 50 and for 80 observations,
  and the encoded frame count equals its total times 20.
- Truth, belief and clock artists carry their role colours in both themes;
  every visible artist's colour is a palette colour or a `background`-to-`ink`
  blend.
- The echolocation azimuth equals the stated function at sampled frames and is
  constant during the hold.
- A one-observation replay renders a frame with more than a set number of
  distinct colours.
- **Trace mutation:** changing `mean[-1]` in a copy of a committed archive
  changes the trace's last point and the header text. **Cloud mutation:**
  changing a middle `cloud` row changes that frame's dots and no other
  frame's; changing the last row changes the poster's pixels when the site
  media are rendered from the copy.
- The encode test decodes its output as section 10 describes, and fails when no
  encoder is found.

**Site**

- `tests/test_check_site_figures.py`: positive fixtures for two plots and for a
  clip; negative fixtures, each with its message, for a missing dark video, a
  missing poster, a wrong clip name, a wrong height, `autoplay`, a missing
  `aria-label`, a description missing or different between themes, identical
  light and dark content, a themed element out of order and on an unlisted
  page, a truncated video, and an encoded NUL.
- `tests/test_cli.py`: default outputs; the archive written has the requested
  demo name and observation count; `--video` on a short run.
- The pinned sentences and the descriptions, each against its own run.

### Mutation checks

The protocol is the first track's: each mutation in turn, bytecode caching off,
tree confirmed clean between mutants, the whole suite run, the plan naming the
exact tests expected to fail.

| Test | Catches | Does not catch |
|---|---|---|
| format, both directions | a writer that scales an array; a loader accepting float32 `mean`; `model_prior` in another order | a value wrong before it reaches the writer |
| recorder moments | an unweighted mean; a diagonal-only `cov`; history from index 1; a computed row-0 `ess`; prediction at the mean particle | an error shared with the test's arithmetic |
| recorded against unrecorded | thinning with the filter's generator | a recorder that only reads state |
| density readings | drawing readings from the filter's generator again | a different but still separate generator |
| `thin` cases | offset 0.5; a plain stride; special-casing uniform weights | another scheme with the same picks on these cases |
| gate, inputs | a changed seed, control, noise level or truth without re-recording; one demo's archive under another's name | a sampler change that leaves every input alone |
| gate, early bounds | **an early trajectory change:** likelihood noise doubled for the first three updates (measured to pass the final bounds and fail at observation 1, M15); the first update skipped | a different valid trajectory, by design; rows other than 1, 8 and last |
| gate, final bounds | **a stale final result:** an archive recorded before a change to the forward model or the likelihood | a change smaller than seed-to-seed spread |
| invariants | a constant offset on the echolocation channels; a truth outside physical support; dots taken from another row; a model-posterior row changed to (0, 0.99, 0.01), which keeps its sum | summaries wrong but consistent with their dots |
| box margin | `TRUE_M = 0.15` | a truth 6% inside the box |
| clip frames | autoscaling in a middle frame; x and y swapped; an invisible cloud on frame 0; uncropped points in the close view; two middle dwells swapped; constant camera speed | a fixed limit badly chosen; labels that collide |
| trace and cloud mutations | a renderer reading a stale or different archive, for traces and for dots separately | a panel neither test perturbs |
| decode check | truncation; another codec or pixel format; a wrong frame count | whether Safari plays it |
| rendered-site check | each negative fixture above | whether a clip looks right |
| source guard | `ax.plot(x, "rate", "r-", data=frame)`; an unthemed plot cell; a link to a retired asset | a literal colour passed through a variable (the artist tests catch its effect) |
| prose | a stale sentence; a number differing from its own run's archive | a well-formed sentence that misleads |

### Checks by eye

- Contact sheets of every clip (k = 0, 1, 8, last) and both figures, both
  themes, full width and 390 pixels, committed under
  `docs/plans/2026-11-19-replay-and-animation/`; the six trace headers of the
  two-mass 2D clip are read at 390 pixels.
- Each description against its clip: prior, early change and ending.
- Each clip in Chrome, Firefox and Safari: poster and controls with JavaScript
  off; play, pause and seek from the keyboard; the theme switch.
- The phone harness from PR #21 at 390 pixels.

## Measurements

Run on 2026-11-18 and 19 in the worktree, on macOS arm64, Python 3.14.0, NumPy
2.4.2, SciPy 1.17.1, Matplotlib 3.10.8, Quarto 1.9.32, through `uv run
--frozen`. Probe scripts were in a session scratch directory and are not kept.

- **M1. The runner computes one run two ways.** From the Pages artifacts of
  five deploy builds, all on image ubuntu-26.04 20261111.196.1 with Python
  3.12.16, the same lock and no change to numerical code, the Getting Started
  cell printed `2: 0.6086294618644883` three times and `2: 0.6086294618645575`
  twice. This Mac prints the first with another last digit in the third entry.
  None diverged. The cause was not identified.
- **M2. Candidate archives.** A scratch recorder ran each demo's default
  inference. With 500 displayed particles the seven files total 1.80 MB; whole
  clouds are 30.7 MB (float64) or 13.0 MB (float32). Inference seconds: 0.2,
  0.4, 0.8, 2.3, 1.4, 33.0 (density), 4.1. The 2D run at M = 0.10 ended at
  (1.4169, -1.2223, 0.1000) ± (0.1111, 0.1148, 0.0012); model comparison at
  0.39903515 and 0.60096485. A JSON string with non-ASCII text round-tripped
  through `savez_compressed` without pickle. The candidates predate this
  format.
- **M3. Dense profiles fail.** On a 161-point grid at each demo's track offset,
  451 of 1,000 `demo_1d` and 1,319 of 3,000 `demo_multi_mass` prior particles
  exceed the weak-field limit somewhere; the strict forward model raised
  `PhysicsDomainError` for both.
- **M4. Density generators.** With the demo's builder at 100 particles, drawing
  readings before any update against between updates: the first equal, the
  second differing by 0.01824. The default run as it is today ended at
  (1.3751, 1.9847, 0.0101); with readings from their own `study_generator(42)`
  it ended at (1.5023, 2.1358, 0.0095) ± (0.083, 0.116, 0.0005).
- **M5. Thinning and roundoff.** Uniform weights, N from 1000 to 6000, m = 500:
  offset 0.5 gave mixed strides with targets exactly on steps; φ gave one
  stride, the nearest target 0.056 to 0.47 of a weight from a step. The three
  small cases in "Testing" gave the indices stated. With two particles and a
  target on the step, moving 5.55e-17 of weight changed the picks from 0, 1 to
  1, 1. 1 / Σw² for N uniform weights gave 999.9999999999998,
  3000.0000000000005 and 6000.000000000001 for N = 1000, 3000 and 6000.
- **M6. Spread over inference seeds**, data fixed, at observations 1, 8 and
  last: 20 seeds for six demos (2D at both truths), 8 for density with separate
  generators. Section 3's bounds are eight times the standard deviations found.
  The largest deviation of any seed from the demo's own was 0.48 of its bound.
  Model comparison: P(K=1) at most 9.1e-7, 4.4e-41 and 0 at the three
  checkpoints; P(K=2) in 0.269 to 0.448, 0.379 to 0.696 and 0.178 to 0.835; in
  the demo's own run three masses lead for observations 1 to 9, two for 10 to
  18, three from 19 on. In the candidate archives the display sample's mean
  was at most 3.0 s.d./√m from the recorded mean and its standard deviation
  0.78 to 1.29 of the recorded one, over all rows.
- **M7. Echolocation.** From all 6,000 particles: the round 1 cube held 0.25%
  of the prior weight and 77.8% after one observation; the cube of section 5,
  -1.3 to 4.8, holds 0.45% and 100.00% at every k ≥ 1. Spatial standard
  deviations: 9.2 to 9.3 (prior), 0.16 to 0.26 (k = 1), 0.017 to 0.030 (final).
  Frames at k = 0, 1, 8, 80 were drawn at 1500 by 1100 in both themes and
  looked at, in two layouts; the second, of section 5, has no overlapping
  text. At 390 pixels the final frame keeps its layout, with 5-pixel labels.
- **M8. 2D details.** Clock fills: 2 of 400 clock-frames exceed 1.1 times the
  final maximum. The "Into the Plane" cell gave the numbers in section 8.
- **M9. Rendering.** A scratch renderer with explicit rectangles and raw frames
  piped to FFmpeg. 2D clip: 1500 by 900, 51 draws at 23 ms, 2.1 s per theme,
  330 and 309 kB, 76 kB poster. Echolocation, round 1 layout: 356 draws at 33
  ms, 12.5 s per theme, 1.18 and 1.10 MB. A 7.5 by 4.6 inch canvas gave 1500 by
  919; heights 4.2, 4.5, 5.0, 5.4 and 5.5 gave exact pixels.
- **M10. Encoder.** `imageio-ffmpeg` 0.8.1 in an overlay: its macOS binary is
  FFmpeg 9.0 with `libx264` and no `ffprobe`; PyPI lists a 67 MB
  `manylinux_2_28_x86_64` wheel. Decoding the 2D clip with it reported H.264
  High, `yuv420p`, 1500 by 900, 20 fps, 296 frames, `moov` before `mdat`. Cut
  to half its bytes, the file kept `ftyp` and the box order, exited 0, and
  decoded 97 frames.
- **M11. Quarto 1.9.32**, in a scratch project with the site's themes: the
  pre-render and resource behaviour of section 4; raw `<video>` attributes pass
  through; the static body class is `quarto-light`; the compiled CSS holds the
  plain theme selectors; a cell's working directory is its page's.
- **M12. Browser.** The section 4 markup without the description element, with
  real clips, in headless Chrome 157 against a logging server: both posters
  requested; no video requested, scripts on or off; the visible video 750 by
  450 at `DOMContentLoaded`; the dark video shown after the theme toggle. At a
  requested 390 pixels Chrome gave 500: the document was 776 wide without the
  CSS rule and 500 with it.
- **M13. Environment.** CI run 44473014146 logged image ubuntu-26.04 and
  CPython 3.14.4; the deploy builds logged 3.12.16. The Ubuntu 26.04 and 24.04
  image software lists contain no "ffmpeg". "Render site" took 43 s and 61 s on
  the runner and 38.1 s here. `uv run pytest -q`: 1079 passed in 32.1 s.
- **M14. Guards.** `_style_violations('ax.plot(x, "rate", "r-", data=frame)')`
  returned `[]`. `_resolve_source("a%00b.png", ...)` raised `ValueError`. A
  frozen dataclass in a module with postponed annotations, imported by path
  without registration, raised `AttributeError`; a `NamedTuple` imported.
  Themed elements in the rendered site: 2 on each of the nine plot pages, 0 on
  the other six.
- **M15. An early-only change.** `demo_1d` with the likelihood's noise doubled
  for updates 1 to 3 only, against the unchanged run: after observation 1 the
  standard deviation of x differs by 0.858 (bound 0.063); at the end the mean
  and standard deviation of x differ by 2.3e-4 and 7.2e-4 (bounds 0.0085 and
  0.0045).

## Risks

- **The bounds come from one machine and 20 seeds.** They are eight standard
  deviations wide and no seed came within half of one, but a multi-modal run's
  spread is not Gaussian. If a valid run elsewhere exceeds a bound, it is
  recalibrated with the evidence recorded, not loosened without it.
- **The gate is loose where runs are unstable**: the † cells, the two-mass 2D
  demo generally, and the model-comparison split. Those are pinned by inputs,
  invariants and their other checkpoints.
- **Only three rows are bounded.** A change that altered a run only between
  observation 8 and the last would pass.
- **The report can go unread.** Nothing fails when a demo stops reproducing.
- **The density demo's run changes**, as section 2 declares.
- **Quarto 1.11.9 was not run.** Pull request 3 runs a clean build on it first.
- **Only Chrome was tested, and not at 390 pixels.** Safari needs byte-range
  responses for video, which GitHub Pages is believed to serve.
- **Four clips are undrawn**, and the format may need a second version once
  they are; the smoke-drawing in pull request 1 is meant to find that early.
- **The encoder** is a 22 to 67 MB development dependency, a GPL build invoked
  as a separate program and not redistributed. The Linux binary is untested.
- **Archives are binary.** Review of a re-recording rests on the estimates the
  demo prints and on the report.
- **Text in a clip scales with the video**: about 7 CSS pixels at 390. The
  descriptions carry what small labels cannot.

## Not doing

- Autoplay, looping, play-on-scroll, a scrubber, or in-browser drawing.
- A JSON export of the replay; recording the generating commit in the archive.
- Pinning or committing video, poster or figure bytes; image-comparison tests.
- A GIF fallback; parallel rendering.
- A seed spread or one-mass case for model comparison; a new landing visual.

## Where this design is least sure

1. **The bounds** (section 3). Too tight, and a valid run on another machine
   fails the gate; too loose, and the gate says little, which is already the
   case at the † cells and for the two-mass 2D and model-comparison demos. The
   calibration is 20 seeds on one machine.
2. **The undrawn clips** (section 5): both 1D clips with the per-clock panel,
   both two-mass clips, and the model-comparison gutter. Their layouts, and
   whether the archive holds everything they need, are untested until the
   smoke-drawing in pull request 1.
3. **The density demo's generator** (section 2). Giving its readings their own
   generator changes that demo's run. No ruling asked for it; it follows from
   requiring inputs to be comparable. The alternative keeps the shared
   generator and gates only the first reading.
