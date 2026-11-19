# Replay Data and Rebuilt Animations: Design

**Date:** 2026-11-19

**Status:** Draft, in review. Revised after review round 1 (Codex xhigh on
`5edbcb5`: "NOT SOUND ENOUGH TO IMPLEMENT", nine findings). "Changes since
round 1" at the end lists what moved.

**Source:** findings V1, V5 and C6 of the 2026-11-17 audit
(`docs/audits/2026-11-17/README.md`), the visual reviewer's critique
(`docs/audits/2026-11-17/reviewers/visual.md`, sections 2 and 3) and prototype
(`docs/audits/2026-11-17/prototype/`), and the owner's decisions of 2026-11-18
in the audit's "Decisions" section. Everything an implementer needs is restated
here.

Claims are marked. **Measured** means a command was run for this spec and its
output is in "Measurements" (M1 to M14). **Read** means it rests on reading
code or a document. **Believed** means neither.

## Purpose

Six demo GIFs and two PNGs are committed under `assets/`, mirrored under
`site/assets/` (33 MB in the tree), and pinned by SHA-256. The pin proves the
files have not changed. It does not prove they are what the code produces, and
it turns every restyle into a change of "evidence".

The animations also hide the search. Each opens after the first observation,
when the cloud has already collapsed; axes rescale every frame; position and
mass share an axis, so mass is a flat line; three panels show the same plane;
the weight colouring encodes nothing after resampling.

This track commits **replay data** instead: what each demo run produced, in
numbers. A reference comparison on the CI runner pins the data to the code. The
site draws its animations and two static figures from the data when it is
built, in the figure style of the first track
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
this track changes only media embeds, their captions and text equivalents, and
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
  "format": "clocks-replay", "version": 1,
  "demo": "demo_2d", "kind": "filter",
  "seed": 42, "n_observations": 50, "n_particles": 2000,
  "noise_std": 0.005, "track_offset": 3.0,
  "param_names": ["x", "y", "M"],
  "prior_ranges": [[-8.0, 8.0], [-8.0, 8.0], [0.005, 0.15]],
  "controls": {"ess_target": 0.8, "rejuvenation_steps": 2,
               "proposal_scale": 2.38, "resampling": "systematic"},
  "cloud_size": 500,
  "created_with": {"python": "3.12.16", "numpy": "2.4.2", "scipy": "1.17.1",
                   "platform": "Linux-...-x86_64", "cpu": "...",
                   "image": "ubuntu-26.04 20261111.196.1"}
}
```

`kind` is `"filter"` or `"model_comparison"`. The model-comparison `meta` also
carries `true_k` and `model_prior`. `created_with` is provenance; only its
platform and Python version are tested (section 3).

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
| `demo_model_comparison` | `true_params` is x₁, x₂, M₁, M₂ of the K = 2 truth |

Arrays, with T observations, C clocks, P parameters, m displayed particles.
Every array is consumed by a drawing, by the command's printed output, or by
the reference comparison.

| Key | Shape, type | Meaning | Demos |
|---|---|---|---|
| `clock_positions` | (C, D) float64 | clock coordinates | all |
| `true_params` | (P,) float64 | the truth | all |
| `true_rates` | (C,) float64 | noise-free tick rates | all |
| `observations` | (T, C) float64 | readings; centred for echolocation | all |
| `filter_observations` | (T, C-1) float64 | the contrasts the likelihood consumed | echolocation |
| `mean` | (T+1, P) float64 | weighted posterior mean; row 0 is the prior | filter |
| `cov` | (T+1, P, P) float64 | weighted covariance, no bias correction | filter |
| `ess` | (T+1,) float64 | 1 / sum of squared weights; row 0 is N | filter |
| `log_evidence` | (T+1,) float64 | cumulative; row 0 is 0 | filter |
| `tempering_stages`, `mh_proposals`, `mh_acceptances` | (T,) int64 | per-update diagnostics | filter |
| `cloud` | (T+1, m, P) float32 | equal-weight display sample | filter |
| `predicted_mean`, `predicted_std` | (T+1, C) float64 | weighted mean and s.d., over all particles, of each particle's predicted reading at each clock | `demo_1d`, `demo_multi_mass`, echolocation (centred) |
| `view_bounds` | (2,) float64 | the close view's cube | echolocation |
| `weight_in_view` | (T+1,) float64 | posterior weight inside that cube, from all particles | echolocation |
| `model_ks` | (K,) int64 | candidate counts | model comparison |
| `model_posterior` | (T+1, K) float64 | row 0 is the model prior | model comparison |
| `model_log_evidence`, `model_ess` | (T+1, K) float64 | row 0 is 0 and N | model comparison |
| `final_mean_k<K>`, `final_std_k<K>` | (P_K,) float64 | each candidate's final estimate | model comparison |

Decisions:

- **Row 0 is the prior.** `ParticleFilter.history[0]` holds it (read). Today
  `_precompute_filter_states` drops it, which is why every animation opens
  after the first observation.
- **Predictions are recorded at the clocks only.** Round 1 proposed a dense
  tick-rate profile along the track for the two 1D clips. That cannot be
  computed: the prior is valid at the clocks, not between them, and on a
  161-point grid 451 of 1,000 one-mass prior particles and 1,319 of 3,000
  two-mass ones leave the weak-field domain, so the strict forward model raises
  `PhysicsDomainError` (measured, M3). The clips draw at the clock positions
  instead (section 5), and the profile arrays are gone.
- **Summaries come from the whole cloud, in float64.** The square root of
  `cov`'s diagonal equals `ParticleFilter.estimate()["std"]` (read: same
  uncorrected weighted variance).
- **Clouds are thinned to at most 500 particles, in float32.** Whole clouds
  compress to 30.7 MB (float64) or 13.0 MB (float32); thinned, the seven files
  total 1.8 MB (measured, M2). A thinned cloud cannot reproduce the summaries,
  which is why they are stored.
- **The model-comparison archive has no clouds** and stores each candidate's
  final estimate, so the command can print the winning model's estimate from
  the replay (it prints one today).
- **The density archive has no predictions.** Its forward model is a
  quadrature; inference alone takes 33 s (measured, M2).
- **Renderers never call the clock-rate forward model.** They may evaluate the
  density parameterisation A·exp(-(x-μ)²/2σ²) for the density figure; that is
  the definition of the parameters, not the physics.

**The display sample.** For weights w (float64, non-negative, summing to
about 1) and m = min(500, N):

1. c = cumsum(w) / cumsum(w)[-1], so c is non-decreasing and ends at exactly 1;
2. targets t_j = (j + φ) / m for j = 0..m-1, with φ = (√5 - 1) / 2;
3. index_j = `searchsorted(c, t_j, side="left")`: the first particle whose
   cumulative weight reaches the target. A target exactly on a step selects the
   particle that ends there.

A particle of weight w is selected about m·w times, so heavy particles repeat;
a particle of zero weight is never selected; N need not divide by m; the result
depends on particle order, which is the filter's own. The irrational offset
matters: with offset 0.5 and uniform weights the targets land exactly on steps
and rounding decides each pick (strides of 4 and 5 mixed for N = 2000), where φ
gives one stride with at least 5% of a weight to spare (measured, M5). The
sample uses no random numbers.

**One file, not two.** A JSON file of summaries beside a binary file of clouds
would diff readably, but could fall out of step; the comparison tool prints a
readable summary of any change instead (section 3). A later in-browser replay
can be fed by an export the build makes from the archive.

**Loading validates**: format and version; the exact key set for the `kind` and
demo, with no extra keys; each key's dtype and number of dimensions; row counts
of T+1 and T; inner dimensions against C, P, m and K; finite values. A failure
raises `ValueError` naming the file, the key and what was expected.

### 2. Recording

A new private module `src/clocks/_replay.py`:

```python
@dataclass(frozen=True)
class Replay:
    meta: Mapping[str, object]
    arrays: Mapping[str, NDArray]

def record_filter(pf, observations: Iterable[Observation], *, meta,
                  predict=None) -> Replay: ...
def thin(particles, weights, size: int) -> NDArray[np.float32]: ...
def save_replay(path: Path, replay: Replay) -> None: ...
def load_replay(path: Path) -> Replay: ...
def first_divergence(reference: Replay, other: Replay) -> Divergence | None: ...
def diff_summary(reference: Replay, other: Replay) -> str: ...
```

- `record_filter` raises before touching the filter if it has already seen an
  observation.
- **Observations are consumed lazily, one per update.** The density demo draws
  each reading from the same generator the filter uses, between updates.
  Drawing the readings first changes them: the second reading differs by up to
  0.018 (measured, M4). The recorder takes an iterator and captures each
  reading as it is consumed, so the existing draw order is kept.
- `predict` maps particles to predicted readings; it receives no generator.
- Each demo module gains `record(*, n_observations=..., n_particles=...) ->
  Replay`, which draws nothing. The model-comparison demo runs its comparison
  once (today it runs twice, once to print and once to animate) and records the
  model prior as row 0.
- `scripts/record_replays.py --out DIR [--compare DIR2]` records all seven
  demos at their packaged defaults into `DIR`. With `--compare` it prints, per
  demo, either "matches" or the first divergence (frame, key, largest
  difference) and a summary of final estimates before and after, writes the
  same text to `DIR/replay-diff.md`, and exits non-zero on any divergence.

### 3. What tests pin

The byte hashes go. Their replacement has three layers.

**Layer 1: a reference comparison on the CI runner.** The committed archives
are recorded on the runner, and every pull request's CI run records them again
and compares:

```yaml
- name: Replay reference
  run: uv run python scripts/record_replays.py --out recorded --compare assets/replays
- uses: actions/upload-artifact@v4
  if: always()
  with: {name: "replays-${{ github.sha }}", path: recorded, retention-days: 7}
```

- **How files get committed.** The step's artifact is the recording. When the
  comparison fails because the code legitimately changed, the author downloads
  it (`gh run download <run> -n replays-<sha>`), commits the archives, and
  pastes `replay-diff.md` into the pull request, where it is reviewed. A lock
  bump that changes the output goes through the same path, so it cannot land
  without that summary. Nothing else writes `assets/replays/`.
- **Why an artifact and not a container.** No container tool is installed on
  the development machine (measured, M13), and an arm64 host could not match
  the x86-64 runner anyway. A manually dispatched workflow was the alternative;
  it cannot be dispatched until it is on `main`, and it would duplicate a job
  that CI must run regardless.
- **The environment is declared.** A committed `.python-version` pins Python
  3.12 for CI, the site build and local work, and `ci.yml` installs it with
  `uv python install` as the site workflows do. Today the CI job uses the
  image's Python 3.14.4 while the site jobs install 3.12.16 (measured, M13). A
  test requires each committed archive's `created_with` to name Linux and
  Python 3.12, so a locally recorded file cannot be committed unnoticed, and a
  text test requires `ci.yml` to contain the step.
- **What is compared.**

  | Compared | How |
  |---|---|
  | key set, shapes, dtypes; `meta` except `created_with` | exactly |
  | integer arrays | exactly |
  | `clock_positions`, `true_params`, `true_rates`, `observations`, `filter_observations` | `rtol=1e-12, atol=1e-12` |
  | other float64 arrays | `rtol=1e-9, atol=1e-12` |
  | `cloud` | `rtol=1e-5, atol=1e-6` |

- **Why a tolerance even there.** The runner is not one numerical environment.
  Five deploy builds on the same image, Python and lock, with no change to
  numerical code, printed the Getting Started posterior in two variants that
  differ by 1.1e-13 relative; this Mac prints a third (measured, M1). None of
  the three diverged. The filter makes discrete decisions (tempering steps,
  resampling indices, accept or reject), so a last-digit difference could flip
  one, after which runs differ at about 1e-3. The tolerance sits four orders
  above the difference observed and six below a divergence. The exact integer
  diagnostics are the sharpest detector.

**Layer 2: portable invariants on the committed archives**, in every
environment, with no inference:

- the files are exactly the seven names and each loads;
- scenario: `meta`, `true_params` and `clock_positions` equal the demo module's
  constants exactly;
- observations: for the six demos that simulate through `simulate()` or the
  echolocation helper, `observations` equals a fresh simulation from the seed
  and `true_rates` equals the forward model at the truth (1e-12); each
  echolocation reading sums to zero and `filter_observations = Q @
  observations`;
- physics: the truth meets the existing 0.08 weak-field margin at the clocks,
  and x₁ < x₂ where there are two masses;
- internal: `log_evidence[0] = 0`; `ess[0] = N` and 0 < `ess` ≤ N; every
  `cloud` row lies inside `prior_ranges`; `cov` is symmetric with no negative
  eigenvalue; acceptances ≤ proposals; `model_posterior` rows sum to 1 and row
  0 equals `model_prior`; `weight_in_view` lies in [0, 1].

**Layer 3: explicit bounds on a fresh recording**, in every environment. The
test calls `record()` and requires the final posterior mean to lie within a
stated distance of the committed one, and each final standard deviation within
a factor of 0.75 to 1.33:

| Demo | Positions | Other |
|---|---|---|
| `demo_1d` | 0.008 | M 0.0003 |
| `demo_2d` | 0.02 | M 0.0002 |
| `demo_multi_mass` | 0.008 | M 0.00015 |
| `demo_multi_mass_2d` | 0.35 | M 0.002 |
| `demo_density` | μ 0.008, σ 0.025 | A 0.0001 |
| `demo_echolocation_3d` | 0.006 | M 0.0004 |
| `demo_model_comparison` | | P(K=1) ≤ 1e-6; P(K=2) and P(K=3) each in [0.05, 0.95] |

Each bound is at least six times the spread of that quantity over inference
seeds with the data held fixed (measured, M6), so a run that takes a different
but valid trajectory passes and a changed scenario, noise level or likelihood
does not. For model comparison the bound is deliberately wide: over 20 seeds
P(K=2) ran from 0.18 to 0.84 (M6; audit finding C2). The existing acceptance
gates are untouched.

**On macOS** layers 2 and 3 gate. Layer 1 is run by hand
(`record_replays.py --compare`), and pull request 1 reports its result per
demo: whether this machine reproduces the runner's recording and, if not, the
first divergence.

**Cost.** Inference takes 0.2 s (1D), 0.4 s (2D), 0.8 s (two masses 1D), 1.4 s
(model comparison), 2.3 s (two masses 2D), 4.1 s (echolocation) and 33 s
(density) (measured, M2). The four cheapest layer-3 tests run in the default
suite, which takes 32 s today (M13); the other three are marked `slow`. The CI
step adds about a minute.

**The old hashes.** The calibration report keeps its asset table as the record
of the August release and gains a dated note that the files it names were
removed and can be found in git history. The row for
`echolocation_range_study.json` is unaffected: that file stays, hash-pinned.
`EXPECTED_VISUAL_HASHES` loses its two PNG rows in pull request 2 and the rest
in pull request 3.

### 4. How and when media is rendered

**A pre-render script.** Quarto runs `site/prerender.py`, three lines over a
library function, before any page. It writes clips and posters to `site/media/`
(git-ignored), declared in `_quarto.yml`:

```yaml
project:
  pre-render: prerender.py
  resources: ["media/*.mp4", "media/*.png"]
```

Measured with Quarto 1.9.32 (M11): the script runs in the project directory
under the `uv run` Python and sees `QUARTO_PROJECT_RENDER_ALL=1` on a full
render; its outputs are copied to the site; on a clean build, files written by
a cell are not copied even when `resources` names them. A cell with
`renderings: [light, dark]` does wrap `<video>` outputs in the theme classes,
but its captions come out as plain paragraphs, and the copying problem rules it
out.

**Rebuild on change.** On a full render every file is rebuilt and anything else
in `site/media/` is deleted. On a single-page render or preview, a clip is
rebuilt when its output is missing or older than its archive or than any file
under `src/clocks/`. The script prints the encoder's path and version.

**Encoder.** H.264 through the `imageio-ffmpeg` package, added to the `dev`
group in pull request 3.

- The software lists of the Ubuntu 26.04 and 24.04 runner images do not mention
  FFmpeg (read, M13). Its absence was not tested on a runner.
- Version 0.8.1 has a `manylinux_2_28_x86_64` wheel; its macOS arm64 binary is
  FFmpeg 9.0 with `libx264` (measured, M10). The Linux binary was not run.
  `uv.lock` fixes the version and hashes.
- The library looks for `imageio_ffmpeg`, then `ffmpeg` on `PATH`, then raises
  with instructions. The library does not depend on it.

`src/clocks/_video.py` pipes raw frames to the encoder. A frame is drawn once
and written as many times as its dwell requires. Settings: 20 frames per
second, `libx264`, `yuv420p`, CRF 20, preset `slow`, `+faststart`, no audio.

**Build time and size.** A full local render takes 38 s today; the same step
took 43 s and 61 s on the runner (measured, M13). Prototype clips at 1500
pixels wide: 2D, 2.1 s per theme and 330 kB; echolocation in the round 1
layout, 12.5 s per theme and 1.18 MB (measured, M9). By frame count all six
clips in both themes should take about a minute here and under two on the
runner, against a 10-minute limit. Rendering is serial. Pull request 3 measures
the cold runner before anything is tuned.

**Pixel sizes.** `yuv420p` needs even dimensions, and a figure 4.6 inches tall
at 200 dpi is 919 pixels, not 920 (measured, M9). Canvas heights are chosen
from those measured to be exact (4.2, 4.5, 5.0, 5.4, 5.5 inches), and a test
asserts each frame size.

**Markup: native controls, no script.** Each embed is raw HTML in the page:

```html
<figure class="replay">
<video class="light-content" aria-label="Animation: the 2D search" src="../media/demo_2d.light.mp4" poster="../media/demo_2d.light.png" width="750" height="450" controls playsinline preload="none"></video>
<video class="dark-content" aria-label="Animation: the 2D search" src="../media/demo_2d.dark.mp4" poster="../media/demo_2d.dark.png" width="750" height="450" controls playsinline preload="none"></video>
<figcaption>The caption, once.</figcaption>
</figure>

In the animation: … It ends at x = 1.42 ± 0.11, …
```

- Nothing plays until the reader presses play. There is no `autoplay`, no
  `loop` and no script, so reduced motion needs no special case.
- The theme swap reuses Quarto's classes. Its rule is the plain selector
  `body.quarto-dark .light-content { display: none !important }` and the static
  HTML's body class is `quarto-light` (measured, M11, M12).
- Measured in headless Chrome 157 with this markup (M12): both posters are
  requested, no video file is requested with JavaScript on or off, the box is
  750 by 450 before any media loads, and switching theme swaps the visible
  video.
- The poster is the final frame, so a reader who never plays sees the result.
- **Text equivalent.** Each clip is followed by one short paragraph saying what
  the prior looks like, what changes in the first few readings, and where the
  run ends, with the final estimates as numbers. A test computes those numbers
  from the archive and requires the sentence in the page, as the range-study
  sentences are bound today.
- Seven embeds are written by hand; the rendered-site check verifies each
  against the files. Markdown image syntax was tried and offers no poster.
- `site/styles.css` gains `figure.replay video { max-width: 100%; height:
  auto; }`. Without it a narrow page scrolls sideways; with it the box scales
  and keeps its aspect ratio (measured at 500 pixels, M12).

### 5. The rebuilt animations

Shared conventions:

- **Canvas** 750 CSS pixels wide, Source Sans 3 through `style(theme)`,
  constrained layout off, every axes at an explicit rectangle.
- **Fixed for the clip:** every axes rectangle and limit, computed once from the
  archive.
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

Only the 2D clip and the echolocation frames have been drawn. Pull request 3
draws and times every clip on its branch, in both themes at full width and 390
pixels, before any layout number here is treated as final.

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
to the largest running mean in the whole run, with the key "stronger fill =
slower clock"; clocks are hollow before any observation. Round 1 scaled to 1.1
times the final maximum, which two early clock-frames exceed (measured, M8).
Right: traces for x, y and M. The audit prototype filled clocks from an orange
ramp; the merged palette has none, orange is the truth's hue, and observed data
are `ink`.

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
from `true_k`. The reviewer of the audit suggested retiring this animation for
a static chart; what the page shows is the third track's decision. Meanwhile
the same run stays an animation and only its form changes, bars to lines.

**`demo_echolocation_3d`, 750 by 550.** Drawn for this revision at k = 0, 1, 8
and 80 in both themes, and at 390 pixels (measured, M7).

- **Close view**, a 3D scene on one fixed cube: the smallest cube holding the
  head, the truth, and each coordinate's mean ± four standard deviations for
  every k ≥ 1, padded by 0.3 and rounded outward to 0.1. For this run that is
  -1.3 to 4.8. It shows the head wireframe (`muted`), clocks (`ink`), the cloud
  (`belief`) and the truth as a ring with a drop-line to the floor. The ring is
  open and several times wider than the final cloud, which is visible inside
  it in the drawn frames.
- **Whole prior box**, a small fixed panel at upper left: the display sample
  projected on x and z over the prior's ±16, with a square marking the close
  view. Here the collapse is visible: the box is full at k = 0 and holds one
  small cluster at k = 1.
- **Weight in view**, under the status line, from `weight_in_view`: "0.4% of
  the prior lies inside the close view", then "100.0% of the cloud's weight
  lies inside the close view". The round 1 cube held 0.25% of the prior and
  only 78% of the weight after the first observation; this one holds 100.0%
  from the first observation on (measured, M7).
- **Traces** for x, y, z and M, stacked at right.
- **Observed against fitted**, a small square panel on equal scales: for each
  of the 27 clocks, the running mean of its centred reading against
  `predicted_mean`, with the diagonal drawn. It is empty at k = 0 and says "no
  readings yet". Its axes are labelled "observed mean" and "fitted", and the
  caption calls it agreement with the readings used in the fit, not a test on
  new data.
- **Camera.** Azimuth turns 40 degrees, eased, at fixed elevation, and is still
  during the final hold. Frames were drawn for a 60-degree turn first; its last
  frame looks nearly along the x-axis and flattens the head.

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

`write_summary_figure(study, png_path)` stays as a thin wrapper that enters the
light style, calls `range_study_figure` and saves. Both scan paths
(`--figure-only` and after a single-cell scan), `tests/test_echo_study.py` and
the monkeypatch in `tests/test_scenarios.py` use that name (read), and keeping
it leaves all four untouched.

### 7. The demo commands and public names

Each `uv run demo-*` command records its run, prints its truth and estimates
from the replay, and writes:

```bash
uv run demo-2d            # output/demo_2d.npz and output/demo_2d.png
uv run demo-2d --video    # also output/demo_2d.mp4
```

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

- `assets/replays/demo_2d.npz`, re-recorded on the runner, and the clip on the
  landing page and "Into the Plane";
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
by the audit's numbers. That is seed 42's noise, and the text equivalent states
the numbers.

**A box-margin guard**, new: every shipped continuous truth value lies inside
its prior box by at least 5% of the box's width, including the K = 2 truth of
the model-comparison demo. It guards finding C6 only. Physical support and
ordering keep their own tests (section 3, layer 2).

### 9. Provenance prose

`site/reproduce/reproducibility.qmd` keeps the anchor `#sec-figures` and the
section is retitled "Where the figures come from".

- The link to the calibration report is labelled as the record of the August
  2026 release, not of the current figures.
- **Replay data** (replaces "Demo animations and the density figure"): each
  animation and the density figure is drawn at build time from
  `assets/replays/<name>.npz`, recorded on the CI runner: configuration, seed,
  observations, a summary of the cloud before any observation and after each,
  and, except for the model-comparison run, a 500-particle sample for display.
  CI records each run again and compares. Video files are build products. Each
  archive is one finite realization.
- **Echolocation animation**: the opening sentences calling the GIF an old
  release whose panel "will change when the assets are next regenerated" are
  deleted; the sentences on two circumradii and the most favourable range stay.
- **Model-comparison runs**: the two sentences below.
- The CI paragraph stops calling the media exceptions.
- **"Run exactly once."** The audit's decision is to state this as protocol and
  disclose earlier blocks. Where the phrase survives (this page, Getting
  Started, README, `method/the-particle-filter.qmd`) it becomes a statement
  that the protocol reserves each block for one run and that block 400 to 411
  was run under the frozen settings. A sentence about blocks 100 to 300 is
  added only with a citation the implementer has checked; this spec has only
  the audit's word for it.

Sentences pinned by `tests/test_generated_assets.py`:

| Today | Becomes |
|---|---|
| "The committed GIF on [How Many Masses?](…) is the packaged default run: 80 observations, 2,000 particles per candidate, and seed 42. " | "The animation on [How Many Masses?](…) replays the packaged default run: …", with the three numbers from `meta` |
| "With K=2 truth, its final estimate is K=2: 0.3990 and K=3: 0.6010. " | the same words, with the true count from `true_k` and each probability looked up by K in `model_ks` |
| the range-study, 25-observation and horizon sentences | unchanged |

The recorded run gives 0.39903515 and 0.60096485 (M2). The test also requires
that no other "K=2: …" figure in the section disagrees. The story pages'
forbidden provenance vocabulary gains "committed replay" and "replay file".

### 10. The rendered-site check and the source guard

`scripts/check_site_figures.py` keeps every check it has (one image per
wrapper, declared size, files inside the site, no nesting, no element with both
theme classes, equal captions) and learns an ordered manifest:

```python
class Fig(NamedTuple):
    kind: str             # "plot" or "clip"
    height: int
    clip: str | None = None

EXPECTED_PAGES: dict[str, tuple[Fig, ...]] = {
    "index.html": (Fig("clip", 450, "demo_2d"),),
    "story/the-search-in-one-dimension.html":
        (Fig("clip", 450, "demo_1d"), Fig("plot", 540)),
    "story/two-hidden-masses.html": (
        Fig("clip", 500, "demo_multi_mass"),
        Fig("clip", 550, "demo_multi_mass_2d"),
        Fig("plot", 420),
    ),
    "story/beyond-point-masses.html": (Fig("plot", 540), Fig("plot", 420)),
    "story/gravitational-echolocation.html": (
        Fig("plot", 420), Fig("clip", 550, "demo_echolocation_3d"),
        Fig("plot", 540),
    ),
    # ... the other five pages likewise
}
```

A `NamedTuple` is used because the tests import the script by path without
registering it, and a dataclass with postponed annotations then fails at import
(measured, M14).

- A page's themed elements, in document order, must be exactly its manifest. A
  themed `<video>` inside `<figure class="replay">` is a clip rendering;
  anything else is a plot wrapper. A page that gains a themed element fails, as
  now. Every unlisted page must have none (the six pages outside today's list
  have none; M14).
- **Plots:** wrappers alternate light, dark; each pair is checked as today, and
  the two image files must now differ in content, not only in name.
- **Clips:** one video per theme and one non-empty `<figcaption>`; declared 750
  by the listed height; `controls`, `playsinline`, `preload="none"` and a
  non-empty `aria-label`; no `autoplay`; `src` and `poster` resolving inside
  the site to `<clip>.<theme>.mp4` and `.png`; the two themes' files differing
  in content; the poster's pixel size, from its PNG header, twice the declared
  size. The rendered stylesheet must contain the responsive rule.
- **The video stream is decoded.** Using the encoder, the check requires H.264,
  `yuv420p`, the frame's pixel size, 20 frames per second and the frame count
  the dwell schedule gives, and that `moov` precedes `mdat`. A file cut in half
  keeps its `ftyp` box and box order but decodes to 97 of 296 frames
  (measured, M10), so the count is what catches truncation.
- **Folded in:** a path whose escapes decode to a NUL makes `_resolve_source`
  raise (measured, M14); it will return a failure message.

`tests/test_site_figure_style.py`:

- **`data=` by position.** `ax.plot(x, "rate", "r-", data=frame)` passes today
  although its third argument is a format string (measured, M14). With `data=`
  only the first two positional arguments may be strings; without it, none.
- **Plot cells** are cells that declare `renderings` *or* call a plot producer:
  `plt.subplots`, `plt.figure`, `show`, or a `clocks.viz` function named
  `*_figure`. Every such cell must meet the cell contract, so an unthemed plot
  cannot slip past by omitting `renderings`.
- **The manifest matches the source.** For each listed page, the order of plot
  cells and replay figures in the `.qmd` equals the manifest's order and clip
  names.
- **Library drawing code** passes the same syntax-tree colour check as page
  cells. A colour passed through a variable escapes that check; the artist
  tests in "Testing" cover it.
- **Retired names.** No page or README mentions any of the eight retired file
  names. References to `assets/replays/` and the study JSON are allowed.

The site-check workflow's path filter gains `assets/replays/**` and the study
JSON.

### 11. Sequencing

Three pull requests. Each leaves the site correct, and none merges a renderer
that nothing calls.

1. **Replay data.** `_replay.py`; `record()` in each demo;
   `scripts/record_replays.py`; `.python-version`; the CI step and artifact; the
   seven archives, recorded on the runner at today's truths; layers 2 and 3; the
   two guard fixes (NUL, `data=`), each with its own test. Demos still write
   GIFs and the site is unchanged. The pull request reports, per demo, whether
   macOS reproduces the runner's recording.
2. **The two static figures.** `density_figure` and `range_study_figure` with
   their page cells and captions; `demo-density` drawing through the new
   function; the check and guard learning several plots per page; removal of
   the two PNGs, their mirror and their hash rows; the provenance sentences
   about those two files.
3. **The clips and the switch.** First on the branch: every clip drawn and
   timed, with contact sheets and runner timings in the pull request. Then
   `_video.py`, the clip builders and the encoder dependency; the demo commands
   and migration note; the 2D truth, its archive and the box-margin guard; the
   pre-render script, page embeds and text equivalents; clip support in the
   check and guard; the rest of the provenance prose; removal of the GIFs,
   `_animate.py`, the old panel functions and their tests.

If drawing in pull request 3 shows the archive lacks something, the format
version is raised and the archives re-recorded there.

## Testing

Each negative case asserts the specific message, so a checker that rejects
everything, or nothing, fails.

**Format** (`tests/test_replay.py`)

- An archive written by the test with plain NumPy, from literal values, loads
  to those values; an archive written by `save_replay` is read back with plain
  NumPy and equals the literals. A value serialized wrongly but consistently
  fails one of the two.
- For every key: a wrong dtype, a wrong number of dimensions and a wrong inner
  dimension each raise, as do a missing key, an extra key, a wrong version and
  a non-finite value.

**Recorder**

- On a hand-built history with correlated parameters and unequal weights in
  every row, `mean`, the whole of `cov` including off-diagonals, and `ess`
  equal moments computed independently in the test, row by row.
- With a non-linear `predict`, `predicted_mean` and `predicted_std` equal the
  independently computed weighted moments of the predictions, and differ from
  the prediction at the mean particle.
- A recorded run and the same run without the recorder, on a filter whose
  updates demonstrably resample and accept moves, give the same observations,
  the same particles and weights in every history row, and the same final
  generator state. Run for the shared-generator density arrangement and with
  `predict` supplied.
- A filter that has seen an observation raises, and its state is unchanged
  afterwards.
- `thin`: weights (0.1, 0.2, 0, 0.3, 0.4) with m = 5 give indices 1, 3, 3, 4,
  4; seven uniform weights with m = 3 give 1, 3, 6; four uniform weights with
  m = 4 give 0, 1, 2, 3 (each measured, M5). Offset 0.5 gives 0, 1, 3, 4, 4 and
  1, 3, 5 on the first two, and a plain stride gives neither.
- `first_divergence` returns nothing for equal replays and the earliest frame
  and key for a perturbed one, at each tolerance class.

**Committed data** (`tests/test_replays.py`): layers 2 and 3 of section 3, the
`created_with` requirement, and the box-margin guard from pull request 3.

**Figures** (pull request 2)

- `density_figure` and `range_study_figure` on synthetic inputs: the plotted
  median error and posterior-spread lines equal the values from `summarize`;
  tolerance lines sit at the study's tolerances; pass counts equal those
  computed from the records; the density band's edges equal independently
  computed percentiles.
- Changing one value in the input moves the corresponding artist.
- Every visible artist's colour is a palette colour for the theme.

**Clips** (pull request 3), on small synthetic replays and the committed ones

- Every axes has the same rectangle and limits on every frame.
- On sampled frames including middle ones, the cloud artist's coordinates equal
  the archive's row, x on x and y on y; the artist is visible, has non-zero
  size and opacity, and is not covered by an opaque artist. Frame 0 shows row
  0.
- The dwell schedule equals the literal list for 50 and for 80 observations,
  and the encoded frame count equals its total times 20.
- Truth, belief and clock artists carry their role colours in both themes, and
  every visible artist's colour is a palette colour or a `background`-to-`ink`
  blend.
- The echolocation azimuth equals the eased function at sampled frames, with a
  smaller first step than middle step, and is constant during the hold.
- A one-observation replay renders a frame with more than a set number of
  distinct colours.
- The frame is 1500 by twice the declared height.
- Changing `mean[-1]` or a `cloud` row in a copy of a committed archive changes
  the trace's last point, the header text and the dots; rendering the site
  media from that copy changes the poster's pixels. This is the test that the
  site draws from the committed data.
- The encode test decodes its output as section 10 describes, and fails when no
  encoder is found.

**Site**

- `tests/test_check_site_figures.py`: positive fixtures for two plots and for a
  clip; negative fixtures, each with its message, for a missing dark video, a
  missing poster, a wrong clip name, a wrong height, `autoplay`, a missing
  `aria-label`, identical light and dark content, a themed element out of order
  and on an unlisted page, a truncated video, and an encoded NUL.
- `tests/test_cli.py`: default outputs; the archive written has the requested
  demo name and observation count; `--video` on a short run.
- The pinned sentences and the text equivalents.

### Mutation checks

The protocol is the first track's: each mutation in turn, bytecode caching off,
tree confirmed clean between mutants, the whole suite run, the plan naming the
exact tests expected to fail.

| Test | Catches | Does not catch |
|---|---|---|
| format, both directions | a writer that scales an array; a loader accepting float32 `mean` | a value wrong before it reaches the writer |
| recorder moments | an unweighted mean; a diagonal-only `cov`; history from index 1; prediction at the mean particle | an error shared with the test's arithmetic |
| recorded against unrecorded | the recorder drawing readings early; thinning with the filter's generator | a recorder that only reads state |
| `thin` cases | offset 0.5; a plain stride; special-casing uniform weights | another valid equal-weight scheme with the same picks on these cases |
| layer 2 | a constant offset on the echolocation channels; a truth outside physical support; stale `meta`; a locally recorded file | wrong summaries in a well-formed archive |
| layer 3 | a changed noise level, truth or likelihood | a changed seed or control, which the reference comparison catches |
| reference comparison (CI) | a changed seed, control or sampler step; a stale archive; changed array content under current names | a recorder formula wrong on both sides, which the moments tests cover; drawing |
| box margin | `TRUE_M = 0.15` | a truth 6% inside the box |
| clip frames | autoscaling in a middle frame; x and y swapped; an invisible cloud on frame 0; two middle dwells swapped; constant camera speed | a fixed limit badly chosen; labels that collide |
| replay-to-artist | a renderer reading a stale or different archive | a panel the test does not perturb |
| decode check | truncation; another codec or pixel format; a wrong frame count | whether Safari plays it |
| rendered-site check | each negative fixture above | whether a clip looks right |
| source guard | `ax.plot(x, "rate", "r-", data=frame)`; an unthemed plot cell; a retired file name | a literal colour passed through a variable (the artist tests catch its effect) |
| prose | a stale sentence; a number differing from the archive; a second, contradicting "K=2" figure | a well-formed sentence that misleads |

### Checks by eye

- Contact sheets of every clip (k = 0, 1, 8, last) and both figures, both
  themes, full width and 390 pixels, committed under
  `docs/plans/2026-11-19-replay-and-animation/`. All six trace headers of the
  two-mass 2D clip are read at 390 pixels.
- Each clip in Chrome, Firefox and Safari: poster and controls with JavaScript
  off; play, pause and seek from the keyboard; the theme switch.
- The phone harness from PR #21 at 390 pixels.
- Each caption and text equivalent against its clip.

## Measurements

Run on 2026-11-18 and 19 in the worktree, on macOS arm64 (Apple M1 Max),
NumPy 2.4.2, SciPy 1.17.1, Matplotlib 3.10.8, Quarto 1.9.32, through
`uv run --frozen`. Probe scripts were in a session scratch directory and are
not kept.

- **M1. The runner is not bitwise repeatable, and nothing diverged.** The
  Getting Started cell (two masses, 40 observations, 1,500 particles, three
  candidates) prints its posterior at full precision. From the Pages artifacts
  of five deploy builds, all on image ubuntu-26.04 20261111.196.1 with Python
  3.12.16, the same lock and no change to numerical code: three print `{1:
  2.7885139471082757e-205, 2: 0.6086294618644883, 3: 0.3913705381355013}` and
  two print `{1: 2.7885139471085926e-205, 2: 0.6086294618645575, 3:
  0.39137053813541234}`. This Mac prints the first with `...5014` as the last
  entry. The cause was not identified; differing runner CPUs are believed.
- **M2. Candidate archives.** A scratch recorder ran each demo's default
  inference. Sizes with 500 displayed particles: 1.80 MB in all; whole clouds
  30.7 MB (float64), 13.0 MB (float32). Inference seconds: 0.2, 0.4, 0.8, 2.3
  (two masses 2D), 1.4 (model comparison), 33.0 (density), 4.1 (echolocation).
  The 2D run at M = 0.10 ended at (1.4169, -1.2223, 0.1000) ± (0.1111, 0.1148,
  0.0012). Model comparison ended at 0.39903515 and 0.60096485. A JSON string
  with non-ASCII text round-tripped through `savez_compressed` without pickle.
  The candidates predate this revision's keys.
- **M3. Dense profiles fail.** On a 161-point grid from -8 to 8 at each demo's
  track offset, 451 of 1,000 `demo_1d` prior particles and 1,319 of 3,000
  `demo_multi_mass` prior particles exceed the weak-field limit somewhere; the
  strict forward model raised `PhysicsDomainError` for both.
- **M4. Density draw order.** With the demo's filter builder at 100 particles,
  drawing three readings before any update against drawing them between
  updates: the first reading equal, the second differing by up to 0.01824, the
  third by 0.00476.
- **M5. Thinning.** Uniform weights, N from 1000 to 6000, m = 500: offset 0.5
  gave mixed strides with targets exactly on steps; φ gave one stride, the
  nearest target 0.056 to 0.47 of a weight from a step. The three small cases
  in "Testing" gave the indices stated there.
- **M6. Spread over inference seeds**, data fixed, 20 seeds (8 for density).
  Standard deviation of the final mean: `demo_1d` x 0.0011, M 4.2e-5;
  `demo_2d` at 0.10, x 0.0027, y 0.0031, M 3.1e-5 (at 0.15: 0.0018, 0.0022,
  2.1e-5); `demo_multi_mass` positions up to 0.0010, masses up to 1.8e-5;
  `demo_multi_mass_2d` positions up to 0.057, masses 0.0003; density μ 0.0012,
  σ 0.0035, A 1.6e-5; echolocation positions up to 0.0009, M 5.8e-5. Final
  standard deviations stayed within 0.80 to 1.13 of the demo seed's. Model
  comparison: P(K=2) from 0.178 to 0.835, standard deviation 0.18; P(K=1) zero
  to printed precision.
- **M7. Echolocation.** From all 6,000 particles: the round 1 cube (-1.6 to
  3.6) held 0.25% of the prior weight, 77.8% after one observation and 94.2%
  after two. The rule in section 5 gives -1.3 to 4.8, holding 0.45% of the
  prior and 100.00% at every k ≥ 1. Spatial standard deviations: 9.2 to 9.3
  (prior), 0.16 to 0.26 (k = 1), 0.017 to 0.030 (final). Frames at k = 0, 1, 8,
  80 were drawn at 1500 by 1100 in both themes and looked at, in two layouts;
  the second, described in section 5, has no overlapping text. The final frame
  was also viewed scaled to 390 pixels: the layout holds and labels are about 5
  pixels tall.
- **M8. 2D details.** Clock fills: 2 of 400 clock-frames exceed 1.1 times the
  final maximum (largest running mean 0.0317 at k = 1 against a scale top of
  0.0267). The "Into the Plane" cell at 0.15 and 0.10 gave the numbers in
  section 8.
- **M9. Rendering.** A scratch renderer with explicit rectangles, artists
  updated in place and raw frames piped to FFmpeg. 2D clip: 1500 by 900, 51
  draws at 23 ms, 2.1 s per theme, 330 and 309 kB, 76 kB poster. Echolocation
  in the round 1 layout: 1500 by 1000, 356 draws at 33 ms, 12.5 s per theme,
  1.18 and 1.10 MB. A 7.5 by 4.6 inch canvas gave 1500 by 919; heights 4.2,
  4.5, 5.0, 5.4 and 5.5 gave exact pixels.
- **M10. Encoder.** `imageio-ffmpeg` 0.8.1 installed in an overlay without
  touching the lock; its macOS binary is FFmpeg 9.0 with `libx264`; PyPI lists
  a 67 MB `manylinux_2_28_x86_64` wheel uploaded 2026-11-03; it ships no
  `ffprobe`. Decoding the 2D clip with it reported H.264 High, `yuv420p`, 1500
  by 900, 20 fps, 296 frames, boxes `ftyp, moov, free, mdat`. The same file cut
  to half its bytes kept `ftyp` and the box order, exited 0, and decoded 97
  frames. The system FFmpeg 9.0.2 produced different bytes for the same frames.
- **M11. Quarto 1.9.32**, in a scratch project with the site's themes: the
  pre-render and resource behaviour in section 4; raw `<video>` attributes pass
  through; Markdown in `<figcaption>` renders; the static body class is
  `quarto-light`; the compiled CSS holds the plain theme selectors; a cell's
  working directory is its page's.
- **M12. Browser.** The section 4 markup with real clips, served locally with
  request logging, in headless Chrome 157: both posters requested; no video
  requested, with scripts on or off; the visible video 750 by 450 at
  `DOMContentLoaded`; after `quartoToggleColorScheme()` the dark video shown.
  At a requested 390 pixels Chrome gave 500: the document was 776 wide without
  the CSS rule, and 500 with it, the video 449 by 269.
- **M13. Environment.** `type -a` finds no docker, podman, colima, lima or
  `container`. CI run 44473014146 logged image ubuntu-26.04 and `CPython
  3.14.4 ... /usr/bin/python3`; the deploy builds logged 3.12.16. The Ubuntu
  26.04 and 24.04 image software lists contain no "ffmpeg". "Render site" took
  43 s and 61 s on the runner and 38.1 s here. `uv run pytest -q`: 1079 passed
  in 32.1 s.
- **M14. Guards.** `_style_violations('ax.plot(x, "rate", "r-", data=frame)')`
  returned `[]`. `_resolve_source("a%00b.png", ...)` raised `ValueError`. A
  module with postponed annotations and a frozen dataclass, imported by path
  without registration, raised `AttributeError`; the same with a `NamedTuple`
  imported. Counting themed elements in the rendered site gave 2 on each of the
  nine plot pages and 0 on the other six.

## Risks

- **A discrete flip on the runner.** M1 shows the runner computing the same run
  two ways. If the two ways ever choose differently at a tempering, resampling
  or acceptance step, the reference comparison fails on some runs and passes on
  others. Nothing measured shows this happening. If it does, the diff summary
  will show which demo; the remedy to consider then is a looser comparison for
  that demo, and it would need its own mutation check.
- **The runner image moves.** `ubuntu-latest` was Ubuntu 26.04 in every run
  inspected, while the image repository's table still maps the label to 24.04
  (read). An image or lock change that alters output fails CI until the
  archives are re-recorded with a reviewed summary. That is intended, and it
  couples such changes to a data commit.
- **Quarto 1.11.9 was not run.** Everything in M11 is from 1.9.32. Pull request
  3's site check runs a clean build on 1.11.9 with real media.
- **Only Chrome was tested, and not at 390 pixels.** Safari needs byte-range
  responses for video, which GitHub Pages is believed to serve.
- **Four clips are still undrawn.** The two-mass 2D layout is the tightest.
  Pull request 3 draws them before fixing anything.
- **The format may need a second version** once those clips are drawn.
- **The encoder** is a 22 to 67 MB development dependency, a GPL build invoked
  as a separate program and not redistributed. The Linux binary is untested.
- **Archives are binary.** Review of a change rests on `replay-diff.md`.
- **Layer 3 is loose where the posterior is unstable**: the two-mass 2D
  position bound is 0.35 and the model-comparison bound is wide. Off the runner
  those two demos are pinned mostly by layer 2.
- **Text in a clip scales with the video**: about 7 CSS pixels at 390, the
  plots' known limit.

## Not doing

- Autoplay, looping, play-on-scroll, a scrubber, or in-browser drawing.
- A JSON export of the replay; recording the generating commit in the archive.
- Pinning or committing video, poster or figure bytes; image-comparison tests.
- A GIF fallback; parallel rendering.
- A seed spread or one-mass case for model comparison; a new landing visual.

## Where this design is least sure

1. **The reference comparison's stability** (section 3). The runner already
   shows two numerical variants of one run. The design bets that they keep
   agreeing through every discrete decision of seven demos, on evidence from
   one cell across five builds and one Mac.
2. **The clips that have not been drawn** (section 5): both 1D clips with the
   new per-clock panel, both two-mass clips, and the model-comparison gutter.
   Their layouts, and whether the archive holds everything they need, are
   untested.
3. **The text equivalents and the echolocation scene's reading.** A paragraph
   per clip with bound numbers is new prose on story pages, in a track that
   otherwise leaves them alone. The echolocation frame now carries six panels
   in 750 by 550 and is legible at full width but small on a phone.

## Changes since round 1

1. Dense track profiles dropped; predictions recorded and drawn at the clocks.
2. Archives recorded on the CI runner and compared there on every pull request;
   invariants and explicit bounds elsewhere; Python pinned to 3.12.
3. Observations consumed lazily; model comparison records `true_k`, the model
   prior row and each candidate's final estimate.
4. Echolocation: wider close view from a stated rule, a prior-scale panel,
   weight in view, an open truth ring, an empty scatter at k = 0; frames drawn.
5. Native click-to-play video; the playback script removed; accessible names
   and text equivalents added.
6. Rebuild on change; decode-based video checks; encoder path printed.
7. Display-sample contract, parameter order, exact key sets; clock-fill scale,
   ellipse and band labels, label gutter.
8. Migration note, version bump and wheel run; `write_summary_figure` kept;
   2D sentence, provenance wording and box-margin guard scoped.
9. Ordered manifest as a `NamedTuple`; plot-producer detection; retired names
   banned instead of `assets/`; three pull requests re-cut so each renderer
   lands with its caller.
