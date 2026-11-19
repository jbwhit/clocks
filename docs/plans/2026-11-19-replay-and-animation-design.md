# Replay Data and Rebuilt Animations: Design

**Date:** 2026-11-19

**Status:** Draft, in review.

**Source:** findings V1, V5 and C6 of the 2026-11-17 audit
(`docs/audits/2026-11-17/README.md`), the visual reviewer's critique
(`docs/audits/2026-11-17/reviewers/visual.md`, sections 2 and 3) and its
prototype (`docs/audits/2026-11-17/prototype/`), and the owner's decisions of
2026-11-18 recorded in the audit's "Decisions" section. Everything an
implementer needs is restated here; the audit is background, not a second
source of truth.

Claims in this document are marked. **Measured** means a command was run for
this spec and its output is quoted in "Measurements" (M1 to M16). **Read**
means it rests on reading the code or a document. **Believed** means neither.

## Purpose

Six demo GIFs and two PNGs are committed under `assets/`, mirrored under
`site/assets/` (33 MB in the tree), and pinned by SHA-256 in
`tests/test_generated_assets.py`. The pin proves only that the files have not
changed. It does not prove they are what the code produces, and it makes every
restyle a change to "evidence".

The animations themselves hide the search. Every one opens after the first
observation, when the cloud has already collapsed. Axes rescale each frame,
position and mass share an axis so mass is a flat line, three panels show the
same plane, and the weight colouring encodes nothing after resampling.

This track replaces the pinned media with committed **replay data**: what each
demo run produced, in numbers. Tests pin that data against the code. The site
draws its animations and the two static figures from the data when it is built,
in the figure style already merged for the build-time plots, in both themes.
The animations are rebuilt around what changes: they start from the prior, keep
every axis fixed, give each parameter its own small panel, and dwell on the
first updates.

It is the second of three figure tracks. The first
(`docs/plans/2026-11-18-figure-style-design.md`) added `src/clocks/_style.py`;
this track uses its palette, type, sizes and light/dark approach unchanged. The
third changes what the model-comparison page shows.

## Scope

In scope:

- a replay file per demo, the code that records it, and the tests that pin it;
- rebuilt animations for the six animated demos, rendered at site build time as
  H.264 video with a poster, in both themes;
- `demo_density.png` and `echolocation_range_study.png` redrawn at build time
  from committed data in the shared style;
- the `uv run demo-*` entry points, the README and Getting Started;
- the 2D demo's true mass, 0.15 to 0.10 (audit finding C6);
- the provenance section of `site/reproduce/reproducibility.qmd` and the tests
  that pin its sentences;
- `scripts/check_site_figures.py` and `tests/test_site_figure_style.py`,
  including two small follow-ups already owed (a NUL in an image path; `data=`
  calls judged by argument position);
- removal of the committed GIFs and PNGs, their mirror, and the code and tests
  that only they needed.

Out of scope, and why:

- **What the model-comparison page shows** (a spread over inference seeds, a
  one-mass case). That is the third track. This track keeps the page's one
  packaged run and changes only how it is drawn.
- **The population-reliability study, the story rewrite, page order, titles and
  URLs.** Each is its own track. On story pages this track changes only the
  media embeds, their captions, and sentences that describe an animation's
  layout.
- **Any interactive or scroll-driven component.** The replay format must not
  preclude a later in-browser replay with a scrubber; "Replay format" says why
  it does not. Nothing here builds one.
- **Page CSS beyond what video needs** (callouts, code blocks, caption colour in
  the dark theme, the unused `.figure-light` rules). Those belong to the CSS
  track.
- **The distinct-particle diagnostic** (audit C3) and audit findings T1 to T4.
  T5 (no evidence file records the versions that produced it) is addressed for
  the new replay files only.

## Design

### 1. Replay format

**One compressed NumPy archive per demo**, `assets/replays/<demo>.npz`, written
with `numpy.savez_compressed` and read with `numpy.load(path,
allow_pickle=False)`. The seven names are the demo module names: `demo_1d`,
`demo_2d`, `demo_multi_mass`, `demo_multi_mass_2d`, `demo_model_comparison`,
`demo_density`, `demo_echolocation_3d`. The files live beside the existing
`assets/echolocation_range_study.json`, outside the package, so the wheel does
not grow. Nothing is mirrored under `site/`: the site reads `assets/` when it is
built.

Each archive holds a `meta` entry, a JSON document stored as a string, and
numeric arrays. `meta` was confirmed to round-trip without pickle, including
non-ASCII parameter names (measured, M5).

```json
{
  "format": "clocks-replay",
  "version": 1,
  "demo": "demo_2d",
  "kind": "filter",
  "seed": 42,
  "n_observations": 50,
  "n_particles": 2000,
  "noise_std": 0.005,
  "track_offset": 3.0,
  "param_names": ["x", "y", "M"],
  "prior_ranges": [[-8.0, 8.0], [-8.0, 8.0], [0.005, 0.15]],
  "controls": {"ess_target": 0.8, "rejuvenation_steps": 2,
               "proposal_scale": 2.38, "resampling": "systematic"},
  "cloud_size": 500,
  "created_with": {"python": "3.14.0", "numpy": "2.4.2", "scipy": "1.17.1",
                   "platform": "macOS-27.2.1-arm64"}
}
```

`kind` is `"filter"` for the six single-filter demos and `"model_comparison"`
for the seventh. `prior_ranges` is the prior box per parameter; it supplies the
fixed axis limits. `created_with` is provenance only and no test compares it.

Arrays, with T observations, C clocks, P parameters, m displayed particles:

| Key | Shape, type | Meaning | Demos |
|---|---|---|---|
| `clock_positions` | (C, D) float64 | clock coordinates | all |
| `true_params` | (P,) float64 | the truth, in `param_names` order | all but model comparison |
| `true_rates` | (C,) float64 | noise-free tick rates | all |
| `observations` | (T, C) float64 | the readings, in display form | all |
| `filter_observations` | (T, C-1) float64 | the contrasts the likelihood consumed | echolocation |
| `mean` | (T+1, P) float64 | weighted posterior mean; row 0 is the prior | filter demos |
| `cov` | (T+1, P, P) float64 | weighted posterior covariance, no bias correction | filter demos |
| `ess` | (T+1,) float64 | effective sample size, 1 / sum of squared weights | filter demos |
| `log_evidence` | (T+1,) float64 | running log evidence; row 0 is 0 | filter demos |
| `tempering_stages`, `mh_proposals`, `mh_acceptances` | (T,) int64 | per-update diagnostics | filter demos |
| `cloud` | (T+1, m, P) float32 | equal-weight display sample of the particles | filter demos |
| `predicted_mean` | (T+1, C) float64 | weighted mean of the particles' predicted centred rates | echolocation |
| `profile_x`, `profile_true` | (G,) float32 | track positions and the true tick rate along the track | `demo_1d`, `demo_multi_mass` |
| `profile_mean`, `profile_std` | (T+1, G) float32 | weighted mean and s.d. of the particles' predicted tick rate along the track | `demo_1d`, `demo_multi_mass` |
| `model_ks` | (K,) int64 | candidate mass counts | model comparison |
| `model_posterior` | (T+1, K) float64 | posterior over counts; row 0 is the model prior | model comparison |
| `model_log_evidence`, `model_ess` | (T+1, K) float64 | per candidate | model comparison |

Decisions, with reasons:

- **Row 0 is always the prior.** `ParticleFilter.history[0]` already holds it
  (read: `inference.py` sets `_history = [self._state]` at construction). Today
  `_precompute_filter_states` in `_animate.py` discards it, which is why every
  animation opens after the first observation.
- **Summaries come from the whole cloud and are stored in float64.** `mean` and
  `cov` are the weighted moments of all particles; the square root of `cov`'s
  diagonal equals `ParticleFilter.estimate()["std"]`, which uses the same
  uncorrected weighted variance (read). `cov` replaces a separate `std` array
  so the two cannot disagree, and supplies the two-standard-deviation ellipse.
- **Clouds are thinned to at most 500 particles and stored in float32.** Whole
  float64 clouds for the six filter demos compress to 30.7 MB, float32 to 13.0
  MB; thinned to 500 the seven files total 1.80 MB (measured, M2). A whole-cloud
  file would add tens of megabytes to the repository's history at every
  regeneration, and a drawing needs neither 6,000 dots nor sixteen digits. A
  thinned cloud cannot reproduce the summaries, which is why they are stored
  separately.
- **The display sample is equal-weight, deterministic, and uses no random
  numbers.** After an update the weights are generally not uniform: the filter
  resamples only when the effective sample size falls to the target (read), and
  the candidate runs end with it between 80% and 97% of the particle count
  (measured, M2). Since the redesign draws every dot alike, the sample must
  already be weighted. For m = min(500, N) the indices are
  `searchsorted(cumsum(weights), (arange(m) + phi) / m)` with
  phi = (sqrt(5) - 1) / 2 and the last cumulative weight set to exactly 1. The
  irrational offset matters. With the obvious offset 0.5 and uniform weights,
  the targets land exactly on steps of the cumulative sum and rounding decides
  each pick: the strides come out irregular (4 and 5 mixed for N = 2000) where
  phi gives a regular stride with at least 5% of one weight to spare (measured,
  M6). Recording must not touch the filter's random generator, so that
  recording a run cannot change it.
- **Renderers read only the replay.** No drawing code calls the forward model.
  That is why the tick-rate profile along the track is stored for the two 1D
  point-mass demos, and why the echolocation file stores the posterior-mean
  prediction. It keeps drawing testable on synthetic data and means a browser
  replay would need no physics.
- **The echolocation file stores both observation forms.** `observations` holds
  the centred clock channels that are displayed; `filter_observations` holds the
  orthonormal contrasts the likelihood used. The invariant
  `filter_observations = Q @ observations` that `animate_echolocation` checks
  today (read) becomes a test on the committed file.
- **The model-comparison file has no clouds.** Nothing drawn from it needs
  them.
- **The density file has no tick-rate profile and no predicted rates.** Its
  forward model is a quadrature; the inference alone takes 33 s (measured, M3),
  and a 161-point profile for every frame would multiply that. The redrawn
  figure needs only the cloud and the summaries.

**One format or two.** A split was considered: a JSON file for configuration and
summaries, which would show as a readable diff in a pull request, plus a binary
file for clouds. One archive was chosen because one file per demo cannot fall
out of step with itself, and because a test that regenerates the run (below)
verifies the numbers by machine instead of by eye. What would change this: the
in-browser replay landing, which would want a browser-readable export. That
export can be produced from the archive by the build without changing what is
committed, so the choice does not preclude it.

**Loading validates** (fail fast): the format name and version; the keys
required for the `kind` and demo; consistent shapes; finite values; `cloud` in
float32; row counts of T+1 and T. A file that fails raises `ValueError` naming
the file, the key and what was expected.

### 2. Recording

A new private module `src/clocks/_replay.py`:

```python
@dataclass(frozen=True)
class Replay:
    meta: Mapping[str, object]
    arrays: Mapping[str, NDArray]

def record_filter(pf, observations, *, meta, predict=None, profile=None) -> Replay: ...
def thin(state: ParticleState, size: int) -> NDArray[np.float32]: ...
def save_replay(path: Path, replay: Replay) -> None: ...
def load_replay(path: Path) -> Replay: ...
```

`record_filter` drives a fresh filter through the observations once, raising if
the filter has already seen data (the guard `_precompute_filter_states` has
today), and builds the arrays from `pf.history`, `pf.diagnostics_history` and
`pf.log_evidence_history`.

Each demo module gains `record(*, n_observations=..., n_particles=...) ->
Replay`, which runs the simulation and inference and draws nothing. The demo's
`main`, the tests and a maintainer script all call it.

`scripts/record_replays.py [name ...]` writes `assets/replays/<name>.npz` from
each demo's packaged defaults. It is the only way the committed files are
produced, it accepts no overrides, and the regeneration test's failure message
names it.

### 3. What tests pin

The byte hashes go. Their replacement has three parts.

1. **The committed file is what the code produces.** A test calls the demo's
   `record()` and compares the result with the committed archive.
2. **The committed file is well formed and matches its demo.** Loading
   validates it; a test checks that the set of files is exactly the seven names
   and that `meta` agrees exactly with the demo module's constants (seed,
   counts, noise, truth, prior ranges).
3. **The site draws from the file.** The pre-render script reads only
   `assets/replays/`, and the rendered-site check requires its outputs on the
   pages.

Video and poster bytes are not pinned anywhere. Two encoders of the same FFmpeg
major version produced different bytes for the same frames (measured, M9).

**Exact or toleranced.** The comparison in part 1:

| Compared | How |
|---|---|
| keys present, shapes, dtypes; `meta` except `created_with` | exactly |
| `tempering_stages`, `mh_proposals`, `mh_acceptances`, `model_ks` | exactly |
| `clock_positions`, `true_params`, `true_rates`, `observations`, `filter_observations` | `rtol=1e-12, atol=1e-12` |
| `mean`, `cov`, `ess`, `log_evidence`, `predicted_mean`, `model_posterior`, `model_log_evidence`, `model_ess` | `rtol=1e-9, atol=1e-12` |
| `cloud`, `profile_*` (float32) | `rtol=1e-5, atol=1e-6` |

What this rests on:

- On one machine the recorder is bitwise repeatable: two runs, and a run under
  Python 3.12 against one under 3.14, gave identical arrays for the six fast
  demos (measured, M4).
- Across platforms there is one data point. The deployed Getting Started page,
  rendered on `ubuntu-latest`, prints a model posterior at full precision; the
  same cell run here on macOS arm64 agrees in every printed digit for two
  entries and differs in the last digit of the third, `...5013` against
  `...5014` (measured, M1). So the platforms are not bitwise identical, and in
  that run, 40 observations through three filters of 1,500 particles, they did
  not diverge.
- The filter makes discrete decisions (resampling indices, Metropolis
  accept/reject). A last-digit difference can in principle flip one, after
  which the two runs differ at the level of Monte Carlo noise, about 1e-3
  relative. The tolerance of 1e-9 sits seven orders above the difference
  observed and six below a divergence, so the test separates "same trajectory"
  from "different trajectory" and nothing in between is expected. The integer
  diagnostics compared exactly are the sharpest detector.

This is the least certain part of the design; see "Risks" for what happens if
the runner diverges.

**Cost.** Inference alone takes 0.2 s (1D), 0.4 s (2D), 0.8 s (two masses 1D),
1.4 s (model comparison), 2.3 s (two masses 2D), 4.1 s (echolocation) and 33 s
(density) (measured, M2 and M3). The default suite takes 32 s today for 1,079
tests (measured, M13). The four cheapest regenerations, about 3 s together, run
in the default suite; the other three carry the existing `slow` marker. CI runs
both suites on every pull request (read: `ci.yml`), so all seven are checked
before a merge either way.

**The old hashes.** `docs/2026-08-16-development-calibration.md` keeps its
"Corrected generated assets" table as the record of the August release, and
gains one dated paragraph beneath it saying that the GIF and PNG files were
removed from the tree by this work, that replay data under `assets/replays/`
superseded them, and that the old files remain in git history. The row for
`echolocation_range_study.json` is unaffected: that file stays, and stays
hash-pinned. In `tests/test_generated_assets.py`, `EXPECTED_VISUAL_HASHES`, the
test that reads the eight files, and the loop asserting each hash appears in the
report are deleted; `ECHO_CERT_SOURCE_HASH` and its test stay. The assertions in
`tests/test_calibration.py` that mention the report's assets heading stay true
and are not changed.

### 4. How and when media is rendered

**A pre-render script, not a cell per page.** Quarto runs
`site/prerender.py` before it renders any page. The script is three lines over
a library function; it renders every clip and poster into `site/media/`, which
is ignored by git and declared in `_quarto.yml`:

```yaml
project:
  pre-render: prerender.py
  resources:
    - "media/*.mp4"
    - "media/*.png"
```

Reasons, each from an experiment with the installed Quarto 1.9.32 (measured,
M10):

- On a clean build, files a cell writes during execution are **not** copied to
  the output directory, even when a `resources` pattern names them. Files
  written by the pre-render script are. A clean build is what the runner does.
- The script runs with the working directory at the project root, under the
  Python of the surrounding `uv run`, and sees `QUARTO_PROJECT_RENDER_ALL=1` on
  a full render. On a single-page render it still runs, without that variable.
- A cell with `renderings: [light, dark]` does wrap two HTML outputs in
  `light-content` and `dark-content`, so the mechanism the plots use would work
  for `<video>`. Its `fig-cap` list comes out as plain paragraphs, not
  `<figcaption>`. This route was not taken because of the first point.

One script also renders each clip once (the 2D clip appears on two pages) and
shows as its own line in the build log.

On a full render the script rebuilds every expected file and deletes anything
else in `site/media/`. On a single-page render or a preview it builds only
files that are missing, so editing a page does not re-encode video each time.

**Committed video files that are not pinned** were the third option. It needs no
encoder on the runner, but the decision is that the site renders from data when
it is built, and unpinned committed media could drift from the data unnoticed.

**Build time.** A full local render takes 38 s today; the same step took 43 s
and 61 s on the runner in the two most recent runs (measured, M12), so the
runner is about 1.1 to 1.6 times slower than this machine. Rendering the 2D
clip takes 2.1 s per theme and the echolocation clip 12.5 s per theme
(measured, M7). The other four clips were not prototyped; by frame count they
should take 2 to 3.5 s per theme each. The estimate is about 55 s here and
about 1.5 minutes on the runner, serially, for a job of roughly 3 minutes
against the 10-minute limit. The script renders serially. If the runner step
ever exceeds 4 minutes, the remedy is a process pool over (clip, theme) pairs,
which are independent.

**Encoder.** H.264 from FFmpeg, obtained through the Python package
`imageio-ffmpeg`, added to the `dev` dependency group, which the site build and
CI already install.

- The documented software list of the `ubuntu-latest` image (Ubuntu 24.04,
  image 20261111.370.1) does not mention FFmpeg (read, M11). That it is absent
  was not tested on a runner.
- `imageio-ffmpeg` 0.8.1 publishes a `manylinux_2_28_x86_64` wheel; its macOS
  arm64 wheel carries FFmpeg 9.0 with `libx264` (measured, M8). The Linux
  binary was not run. The version is fixed by `uv.lock`, so a contributor and
  the runner use the same encoder.
- The library finds an encoder in this order: `imageio_ffmpeg` if importable,
  else `ffmpeg` on `PATH`, else a `RuntimeError` saying how to get one. It is a
  development dependency only; the library does not depend on it and the wheel
  does not ship it.

A new private module `src/clocks/_video.py` holds the lookup and a writer that
pipes raw frames to the encoder. Each distinct frame is drawn once and written
as many times as its dwell requires, so a paced clip costs one draw per
observation, not one per video frame. Settings: 20 frames per second, `libx264`,
`yuv420p`, CRF 20, preset `slow`, `+faststart`, no audio.

**Sizes.** Frames are rendered at twice the display size, 1500 pixels wide.
Measured: the 2D clip is 330 kB (light) and 309 kB (dark) with a 76 kB poster;
the echolocation clip is 1.18 MB and 1.10 MB with a 167 kB poster (M7). At
display size they would be 155 kB and 566 kB, and render time barely changes,
so the sharper size is kept. Both prototyped clips are within the visual
reviewer's targets of 0.5 MB and 1.5 MB. Today's GIFs total 16.6 MB.

**Pixel sizes are a requirement, and one of them is a trap.** H.264 with
`yuv420p` needs even dimensions. A figure 4.6 inches tall at 200 dpi is 919
pixels, not 920, because the product falls just below the integer (measured,
M7). The audit prototype used that height at 160 dpi, where it gives 736. Clip
canvases are therefore chosen from heights that were measured to give exact
pixels at 200 dpi (4.2, 4.5, 5.0, 5.4, 5.5 inches), and a test asserts each
clip's frame size.

**Markup.** Each embed is written in the page source as raw HTML:

```html
<figure class="replay">
<video class="light-content" src="../media/demo_2d.light.mp4" poster="../media/demo_2d.light.png" width="750" height="450" controls muted loop playsinline preload="none"></video>
<video class="dark-content" src="../media/demo_2d.dark.mp4" poster="../media/demo_2d.dark.png" width="750" height="450" controls muted loop playsinline preload="none"></video>
<figcaption>The caption, once.</figcaption>
</figure>
```

Measured with Quarto 1.9.32 (M10): the attributes pass through unchanged;
Markdown inside the `<figcaption>` is processed; and Quarto's theme rule is the
plain class selector `body.quarto-dark .light-content { display: none
!important }`, so it hides a `<video>` as it hides a `<div>`. One figure holds
both videos and a single caption, so the caption is not duplicated as it must
be for the plots.

Markdown image syntax with an `.mp4` was tried: Quarto emits a `<video>` with a
caption and copies the file, but offers no poster, `muted`, `loop` or
`preload`. Raw HTML was chosen over a cell or shortcode that would generate the
markup because there are seven embeds and the rendered-site check verifies each
attribute against the files. What would change this: more embeds, or the check
catching hand-edited attributes drifting more than once.

**Playback, and reduced motion.** The videos carry no `autoplay` attribute. A
small script, `site/includes/replay.html`, included after the body, plays a clip
while at least half of it is on screen and pauses it otherwise, and does
neither when the reader has asked for reduced motion:

```html
<script>
(() => {
  const still = window.matchMedia("(prefers-reduced-motion: reduce)");
  const watch = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      const video = entry.target;
      if (entry.isIntersecting && !still.matches) {
        video.play().catch(() => {});
      } else {
        video.pause();
      }
    }
  }, { threshold: 0.5 });
  document.querySelectorAll("figure.replay video").forEach((video) => watch.observe(video));
})();
</script>
```

Measured in headless Chrome 157 against a page built this way (M14):

- the visible clip starts playing; the clip for the other theme is hidden, never
  intersects, and its video file is **not requested**;
- switching the theme pauses the first clip and fetches and plays the second;
- with reduced motion forced, no video file is requested and nothing plays; the
  reader sees the poster and the controls;
- both posters are requested at load, so the hidden theme costs one extra image
  per clip;
- the video box is 750 by 450 at `DOMContentLoaded`, before any media loads, so
  the page does not shift.

Without JavaScript the reader gets the poster and working controls. The poster
is the clip's final frame, so a reader who never presses play sees the result.

**CSS.** `site/styles.css` gains `figure.replay video { max-width: 100%;
height: auto; }`. Without it a narrow page scrolls sideways; with it the box
scales and keeps its aspect ratio before the media loads (measured at a
500-pixel viewport, M14). The caption is styled to match the plots' captions,
checked by eye.

### 5. The rebuilt animations

Shared conventions, from the visual reviewer's section 3 and the prototype:

- **Canvas.** 750 CSS pixels wide, the text column. Source Sans 3, 10.5 pt
  labels and 9.5 pt ticks, through `style(theme)`. Constrained layout is turned
  off for clips: every axes is placed at an explicit rectangle, so nothing is
  re-solved between frames.
- **Fixed for the whole clip:** every axes rectangle and every axis limit,
  computed once from the replay before the first frame.
- **Status line.** Top left, semibold: "Before any observation", then "After k
  observations". No other title.
- **Colour roles.** Particles are `belief` dots of one size and opacity, with no
  colour map. A second mass is `belief_alt`. The truth is `truth`: a ring and
  crosshair for a position, a solid line for a value. Clocks and observed data
  are `ink`. There are no dashed lines, because no clip shows a threshold.
  Nothing is distinguished by hue alone: truth and belief differ in mark, and
  two masses carry direct labels.
- **Direct labels, no legend boxes.**
- **Parameter traces.** One small panel per parameter: the posterior mean as a
  line, mean plus or minus one standard deviation as a band, the truth as a
  solid line. The x-axis runs from 0 to T. Each panel's header gives the name,
  "true 1.50" in the truth colour and "1.42 ± 0.11" in the belief colour. The
  y-limits are the smallest interval containing the truth and the mean plus or
  minus one standard deviation for every k from 1, widened by 8%, and clipped
  to the prior range; a one-observation replay gets a minimum span. The prior's
  band overflows and is clipped by the axes.
- **Zoom inset.** A one-mass cloud collapses to a speck on prior-range axes. The
  panel carries an inset with a fixed window centred on the truth, whose
  half-width on each axis is 1.25 times (the final mean's distance from the
  truth plus two final standard deviations), and a label giving the
  magnification.
- **Pacing.** Twenty frames per second. The prior holds 1.2 s; observations 1 to
  3 hold 0.9 s each; 4 to 8 hold 0.5 s; 9 to 20 hold 0.25 s; later ones 0.1 s;
  the last holds 2.5 s. A 50-observation clip runs 14.8 s and an 80-observation
  clip 17.8 s. The clip loops.
- **Poster.** The final frame, as a PNG at the frame's pixel size.

The clips, with their display sizes. "Prototyped" means a scratch renderer was
run for this spec on real recorded data; the others are specified from the
reviewer's critique and have not been drawn.

**`demo_1d`, 750 by 450. Not prototyped.**
Left, upper: the cloud in the (x, M) plane on the prior ranges, with the truth
marker, the three clocks as `ink` squares on the x-axis, the two-standard-
deviation ellipse, and the zoom inset. Left, lower, sharing the x-axis: the
tick-rate deficit (1 minus rate) along the track, with the true curve in
`truth`, the posterior-predictive mean and band in `belief`, and the running
mean of the observed deficit at each clock as `ink` dots with bars of the noise
divided by the square root of k; its y-axis runs from 0 to 1.25 times the
largest value it will show. Right: traces for x and M. This replaces the
zero-based rate bars, the separate setup panel and a mass axis of 0 to 2
against a prior that ends at 0.15.

**`demo_2d`, 750 by 450. Prototyped (M7); the audit prototype shows the
target.**
Left: one square map on the prior position range with the cloud, the eight
clocks, the truth marker, the ellipse and the zoom inset. Each clock's fill
runs from `background` to `ink` with the running mean of its observed deficit,
on a fixed scale from 0 to 1.1 times the largest final value, so the pattern
sharpens instead of flickering; before any observation the clocks are hollow.
Right: traces for x, y and M. This merges the three panels that showed the same
plane and drops the colour bar.

The audit prototype filled the clocks from an orange ramp. Here the fill is a
blend of two palette colours because the merged palette has no such ramp,
orange is the truth's hue, and observed data are `ink` by the first track's
rules.

**`demo_multi_mass`, 750 by 500. Not prototyped.**
As `demo_1d`, with both masses' clouds in the one (x, M) panel, mass 1 in
`belief` and mass 2 in `belief_alt`, each truth marker labelled "mass 1" or
"mass 2", and one inset per mass in the upper corners. Right: four stacked
traces, x₁, x₂, M₁, M₂, each in its mass's colour. This replaces the x₁-against-
x₂ scatter, an abstract plane that needed explaining.

**`demo_multi_mass_2d`, 750 by 550. Not prototyped.**
Left: one map with both clouds, both labelled truths, both ellipses and the ten
clocks filled as in `demo_2d`. No inset: the final standard deviations are 0.2
to 0.8 on a 16-wide axis (measured, M2) and stay visible. Right: six traces in
three rows and two columns, one column per mass. Today only mass 1's cloud is
drawn.

**`demo_model_comparison`, 750 by 420. Not prototyped.**
One panel: the posterior probability of each count against observations, as
three lines that grow from the model prior (one third each) at k = 0. Lines are
`ordered[0..2]` for K = 1, 2, 3, each labelled at its head with its count and
current value; the K = 2 label reads "K = 2 (true model)"; labels are kept a
text line apart where the lines cross. The axes are 0 to T and -0.05 to 1.05.

The visual reviewer suggested a line chart in place of the bars and, better
still, retiring the animation for a static figure. What that page shows is the
third track's decision. Meanwhile this track keeps one animation of the same
packaged run, with the same numbers, and changes only its form from bars to
lines, the form the page's own executed plot already uses. The zero-based rate
bars beside it are dropped. A static figure from the same replay would be
simpler still; it was not chosen because it would remove an animation the
page's prose refers to, which is a content change. If the third track retires
the animation, it deletes one clip builder.

**`demo_echolocation_3d`, 750 by 500. Prototyped (M7), crudely.**
Left: the 3D scene. Its three axes share one fixed cube that contains the head
and the true mass with a margin of 0.6, here -1.6 to 3.6, about half the width
of today's box. The panes take the page colour. It shows the head's wireframe
in `muted`, the 27 clocks in `ink`, the truth as a ring with a thin drop-line
to the floor, and the cloud in `belief`. Particles outside the cube are not
drawn (by masking; clipping to 3D limits needs Matplotlib 3.10 and the project
allows 3.9). Right: four stacked traces, x, y, z, M, and beneath them a scatter
of the 27 running-mean centred readings against the posterior-mean prediction,
on fixed symmetric limits with the diagonal drawn. The scatter replaces the 27
bars; the per-parameter traces replace the shared-axis convergence panel and
the mass histogram.

The camera turns 60 degrees in azimuth over the clip, eased at both ends, at a
fixed elevation, and is still during the final hold. It moves on every video
frame before the hold, so this clip is drawn about 300 times where the others
are drawn T+1 times. The prototype, whose camera also moved during the hold,
drew 356 frames in 12.5 s per theme.

One limit is known and accepted. The prior spans plus or minus 16 on each axis
and the first observation collapses it to a standard deviation of about 0.2
(measured, M2). In the fixed cube the prior frame is nearly empty: 5 of the 500
displayed particles fall within even plus or minus 4.5. The collapse is
therefore carried by the traces, whose first frame shows the prior's band, and
the prior frame's status line says the prior extends far beyond the view.

### 6. The two static figures

Both become plots drawn by an executed cell from committed data, in the same
way as the nine existing plots: `renderings: [light, dark]`, a two-entry
caption, `style(theme)`, `show(fig)`. The drawing lives in the library, because
a demo and a script draw the same figures; the cell loads the data and calls
it. A cell's working directory is its page's directory (measured, M10), so the
data path is `../../assets/...`.

**Density** (`story/beyond-point-masses.qmd`), 750 by 540, from
`assets/replays/demo_density.npz` through a new `density_figure(replay,
colors)`. Top: the density profile along the track, with the true profile in
`truth`, and the pointwise median and central 68% band of the profiles of the
500 displayed particles of the last frame in `belief`; the clocks are `ink`
squares on the x-axis. Below: three traces, for μ, σ and A. Declared changes
from today's figure: the inferred curve was the profile at the posterior mean
and becomes a median with a band; the clock-rate panel is dropped; the three
parameters no longer share an axis; truth and inference swap to the site-wide
colours.

**Range study** (`story/gravitational-echolocation.qmd`), 750 by 540, from the
existing `assets/echolocation_range_study.json` through
`range_study_figure(study, colors)`, which replaces `write_summary_figure` in
`src/clocks/_echo_study.py`. Two stacked log-scale panels, position error above
mass error, against range. Individual runs are small `muted` dots; the median
error is an `ink` line with circles; the median posterior standard deviation is
a solid `belief` line with squares, labelled directly once; the frozen pass
tolerance, read from the study's `tolerances`, is a dashed `threshold` line in
each panel; the pass counts per range, computed from the records, run along the
top. There is no title. `scripts/scan_echolocation_range.py` keeps `--figure-
only`, writing a light-theme PNG through the same function.

### 7. The demo entry points

After this change each `uv run demo-*` command runs the simulation and
inference, prints its truth and estimates as it does today, and writes into
`output/`:

```bash
uv run demo-2d            # output/demo_2d.npz and output/demo_2d.png
uv run demo-2d --video    # also output/demo_2d.mp4
```

- `<name>.npz` is the replay, the same format as the committed files.
- `<name>.png` is the final frame: the clip's poster, or for `demo-density` the
  density figure. It needs only Matplotlib.
- `--video` renders the clip as well. It needs an encoder, which `uv sync`
  provides in a checkout; without one the command stops with a message saying
  how to get one. `demo-density` has no `--video`.
- `--output-dir` replaces `--output`; `--theme light|dark` is new and defaults
  to light; `--observations` and `--particles` stay.

Video is opt-in so that the default command works from a bare wheel install,
which has no encoder. An alternative, making `imageio-ffmpeg` a dependency of
the library so that demos always write video, was rejected: it would put an
encoder binary of 22 to 67 MB, depending on platform, behind a physics
library's install.

The old `animate_*`, `plot_*` and `create_*_dashboard` names leave `clocks.viz`.
Nothing outside the demos and `tests/test_viz.py` uses them (read), and none is
in `clocks.__all__`. `clocks.viz` then exports the style names it has today,
plus `load_replay`, `render_clip`, `density_figure` and `range_study_figure`.

**README.** The command list shows the new outputs and the `--video` form. The
embedded GIF is removed, since a README cannot play a build-time video, and the
sentence above it links to the site. The paragraph about committed demo media
and the echolocation panel caveat is replaced by two sentences: replay data for
each demo is committed under `assets/replays/`, and the site draws its
animations from it. The project-structure list is updated.

**Getting Started.** The same command list, and one added sentence: rendering
the site needs no system FFmpeg, because `uv sync` installs the encoder.

**Architecture page and `docs/someday-maybe.md`.** The module diagram and its
text equivalent name the new modules; the sentence about the site shipping
static GIFs is corrected.

### 8. The 2D truth change

`TRUE_M` in `src/clocks/_demos/demo_2d.py` goes from 0.15 to 0.10. What it
touches:

- `assets/replays/demo_2d.npz`, recorded at the new value.
- The clip on the landing page and on "Into the Plane", both drawn from that
  file.
- The executed cell in `site/story/into-the-plane.qmd`, line 49, which states
  the truth again and changes with it. Measured (M3): that cell's posterior
  moves from (1.405, -1.077, 0.149) with standard deviations (0.111, 0.114,
  0.001) and a largest particle mass of exactly 0.1500, the prior bound, to
  (1.372, -1.098, 0.100) with (0.172, 0.170, 0.002) and a largest particle mass
  of 0.1068. The posterior is no longer pressed against the ceiling.
- `tests/test_shipped_ground_truths.py` reads the module constant and needs no
  edit; the weak-field strength falls from 0.070 to 0.047 against the 0.08
  limit (measured, M3).
- No prose quotes 0.15 as this demo's truth (read: a search of the site and the
  README finds only the cell). The page's sentence that the cloud collapses "to
  a tight knot at the truth" is checked against the new figure by eye.
- The audit's status table marks C6 fixed.

A new test guards the finding itself: for every demo, each true parameter lies
inside its prior range by at least 5% of the range's width.

One thing the sharper figure will show. In the full 50-observation run at the
new value the final estimate is x = 1.417 ± 0.111, y = -1.222 ± 0.115 and
M = 0.1000 ± 0.0012 (measured, M2): y sits 1.9 standard deviations from the
truth of -1.0. It was 1.9 standard deviations off at the old value too, by the
visual reviewer's numbers, so this is the noise realization of seed 42, not the
prior bound. The caption says "one simulated run" and claims no more.

### 9. Provenance prose

The section keeps its anchor, `#sec-figures`, which the landing page and the
echolocation page link to, and is retitled "Where the figures come from". It
becomes:

- **Replay data.** Each animation, and the density figure, is drawn when the
  site is built from a committed file `assets/replays/<name>.npz` written by
  that demo's recorder: its configuration and seed, the observations, a summary
  of the particle cloud before any observation and after each one, and a
  500-particle sample for display. Tests regenerate each run from the code and
  compare. The video files are products of the build and no claim is made about
  their bytes. Each file is one named finite realization and none establishes a
  recovery rate. (Replaces "Demo animations and the density figure".)
- **The density paragraph** is unchanged.
- **Echolocation animation.** The sentences about a panel that "will change
  when the assets are next regenerated" are deleted, because this work
  regenerates it: the panel now shows the posterior mean of the particles' own
  predictions. The sentences about the mass at two circumradii and the most
  favourable range stay.
- **Model-comparison runs.** Two pinned sentences change; see below.
- **Echolocation range study.** Unchanged, except that the figure is said to be
  drawn at build time from the certified records.

The CI paragraph above the section stops calling the GIFs and PNGs exceptions:
every figure and animation is drawn at deploy time, the animations and two
figures from committed data described in the next section.

Sentences pinned by `tests/test_generated_assets.py`:

| Today | Becomes |
|---|---|
| "The committed GIF on [How Many Masses?](../story/how-many-masses.qmd) is the packaged default run: 80 observations, 2,000 particles per candidate, and seed 42. " | "The animation on [How Many Masses?](../story/how-many-masses.qmd) replays the packaged default run: 80 observations, 2,000 particles per candidate, and seed 42. " with the three numbers taken from the replay's `meta` |
| "With K=2 truth, its final estimate is K=2: 0.3990 and K=3: 0.6010. " | the same sentence, with both numbers computed from the last row of `model_posterior` |
| the range-study sentences, the 25-observation sentence, the two sentences on horizon and budget | unchanged |

The recorded run gives 0.39903515 and 0.60096485 (measured, M2), so the
sentence's text does not change; it stops being a literal. The provenance
vocabulary the story pages must not use gains "committed replay" and "replay
file". The test that the report contains every asset hash is removed, as in
section 3.

### 10. The rendered-site check and the source guard

`scripts/check_site_figures.py` today expects exactly one light and one dark
plot on nine pages. It keeps that strictness and learns two things: a page may
hold several plots, and a page may hold clips.

```python
@dataclass(frozen=True)
class PageFigures:
    plots: tuple[int, ...] = ()               # declared heights, in page order
    clips: tuple[tuple[str, int], ...] = ()   # (clip name, declared height)

EXPECTED_PAGES = {
    "index.html": PageFigures(clips=(("demo_2d", 450),)),
    "story/clocks-as-gravimeters.html": PageFigures(plots=(420,)),
    "story/one-clock-is-not-enough.html": PageFigures(plots=(420,)),
    "story/the-search-in-one-dimension.html":
        PageFigures(plots=(540,), clips=(("demo_1d", 450),)),
    "story/into-the-plane.html":
        PageFigures(plots=(540,), clips=(("demo_2d", 450),)),
    "story/two-hidden-masses.html": PageFigures(
        plots=(420,),
        clips=(("demo_multi_mass", 500), ("demo_multi_mass_2d", 550)),
    ),
    "story/how-many-masses.html":
        PageFigures(plots=(420,), clips=(("demo_model_comparison", 420),)),
    "story/beyond-point-masses.html": PageFigures(plots=(540, 420)),
    "story/gravitational-echolocation.html":
        PageFigures(plots=(420, 540), clips=(("demo_echolocation_3d", 500),)),
    "method/the-particle-filter.html": PageFigures(plots=(540,)),
}
```

- **Themed elements are classified.** An element carrying a theme class is a
  clip rendering if it is a `<video>` inside `<figure class="replay">`, and a
  plot wrapper otherwise. A themed `<video>` anywhere else is a failure.
- **Counts are exact.** A page must have exactly the listed number of plot
  wrappers per theme and exactly the listed clips, in order. A page that gains
  another themed element still fails.
- **Plots.** Wrappers must alternate light, dark in document order; each
  consecutive pair is checked as today (one image each, declared 750 by the
  listed height, files existing inside the site and different, one caption
  each, the same text).
- **Clips.** Each replay figure must hold exactly one video per theme and one
  non-empty `<figcaption>`. Each video must be declared 750 by the listed
  height; carry `controls`, `muted`, `loop`, `playsinline` and
  `preload="none"`, and no `autoplay`; and have a `src` and a `poster` that
  resolve inside the site to existing files named `<clip>.<theme>.mp4` and
  `.png`. The poster's pixel size, read from its PNG header, must be twice the
  declared size, and the video file must begin with an MP4 `ftyp` box.
- **Every other page** in the output must contain no theme class and no replay
  figure, so a themed element on an unlisted page is caught.

Folded in: `_resolve_source` raises `ValueError` on a path whose percent-escapes
decode to a NUL character, from `Path.resolve()` (measured, M15). It will
return a failure message instead, like every other bad path.

`tests/test_site_figure_style.py`:

- **`data=` by position.** `_style_violations` counts string arguments and
  allows two when `data=` is passed, wherever they sit, so
  `ax.plot(x, "rate", "r-", data=frame)` passes although its third argument is
  a format string (measured, M15). The rule becomes: with `data=`, only the
  first two positional arguments may be strings; without it, none may.
- **Plot cells are found by their `renderings` option**, not by a call to
  `plt.subplots`, since the two new cells draw through library functions. The
  expected number of plot cells is listed per page, and the test that the set
  of plotting pages is exactly the known one stays.
- **Library drawing code obeys the same colour rules.** The syntax-tree check
  runs over the modules that draw clips and figures: no colour keyword holds a
  string literal other than `"none"`, and no `cmap` is passed.
- **Replay figures in page source.** Each `<figure class="replay">` block holds
  a light and a dark video for one known clip, with matching `src` and `poster`
  names. No page or README references a `.gif` or a file under `assets/`.

The site-check workflow's path filter gains `assets/replays/**` and
`assets/echolocation_range_study.json`, since the site now reads them.

### 11. Sequencing

Three pull requests. The first two change nothing the site shows or says; the
third is the switch and cannot usefully be split, because pages, checks, prose
and the removal of the old files have to agree.

1. **Replay data.** `_replay.py`, `record()` in each demo,
   `scripts/record_replays.py`, the seven committed files at today's truths, the
   format, recorder, contract and regeneration tests, and the two owed
   follow-ups. Demos still write GIFs. This goes first because it puts the
   cross-platform comparison in front of the runner before anything depends on
   it.
2. **Renderers.** `_video.py`, the clip builders, `density_figure`,
   `range_study_figure`, the encoder dependency, and their tests, beside the
   old code and unused by the demos or the site. Contact sheets of every clip
   at four frames, in both themes, accompany the pull request, because four of
   the six clips have not been drawn yet.
3. **The switch.** The 2D truth, its re-recorded file and the guard that every
   truth lies inside its prior (which the 2D file fails until then); the demo
   commands; the
   pre-render script, `_quarto.yml`, the playback script and CSS; the page
   embeds, captions and two plot cells; the rendered-site check and source
   guard; the provenance prose and its tests; README, Getting Started,
   Architecture; the calibration report's note; the audit's status table (C6,
   V1 and V5); deletion of the GIFs, PNGs, their mirror, `_animate.py`, the old
   panel functions and their tests.

## Testing

### Replay format and recorder (`tests/test_replay.py`)

- Save then load returns equal `meta` and arrays with the declared dtypes.
- `load_replay` raises `ValueError` for a wrong format name, an unknown
  version, a missing key, a row count that is not T+1, a non-finite value, and
  a float64 `cloud`; each message names the key.
- On a hand-built filter history with unequal weights, `mean`, `cov` and `ess`
  equal moments computed independently in the test; row 0 equals the prior
  state; the square root of the last `cov` diagonal equals
  `pf.estimate()["std"]`.
- `thin`: with uniform weights the picks are a regular stride; with all weight
  on one particle every pick is that particle; the result is float32 with
  `min(size, N)` rows.
- Recording leaves the filter's random generator state unchanged, and raises on
  a filter that has already seen an observation.

### Committed data (`tests/test_replays.py`)

- The files under `assets/replays/` are exactly the seven names, and each
  loads.
- Each file's `meta` equals its demo module's constants.
- The echolocation file satisfies `filter_observations = Q @ observations`.
- Every true parameter is inside its prior range by 5% of the width. This test
  arrives with the 2D truth change, in the third pull request.
- Regeneration: `record()` against the committed file, as in section 3's
  table. Four demos in the default suite, three marked `slow`.

### Clips and figures (`tests/test_viz.py`, rewritten)

Drawn from small synthetic replays (three observations, 40 particles) and, for
limits, from the committed files:

- every axes has the same rectangle and the same limits on the first and last
  frame;
- the frame is exactly 1500 pixels wide and twice the declared height, both
  even;
- the first frame's status line reads "Before any observation" and its cloud
  holds the prior row;
- the dwell schedule starts at 1.2 s, ends at 2.5 s, has every entry a multiple
  of one twentieth of a second, and the number of frames written equals its
  sum times 20;
- the truth marker's colour is the palette's `truth` and the cloud's is
  `belief`, in both themes;
- the echolocation camera's azimuth is non-decreasing, spans 60 degrees and
  does not move during the final hold;
- a one-observation replay renders without error;
- encoding a three-observation clip produces a file that begins with an `ftyp`
  box and a poster of the frame's pixel size. This test fails, and is not
  skipped, when no encoder is found.
- `range_study_figure` on a synthetic study draws its tolerance lines at that
  study's tolerances and its pass counts as computed from that study's records.

### Site

- `tests/test_check_site_figures.py` gains cases for: a page with two plots; a
  page with a clip; a clip missing its dark video; a poster that does not
  exist; a wrong clip name; a wrong height; an `autoplay` attribute; a themed
  element on an unlisted page; a path with an encoded NUL.
- `tests/test_site_figure_style.py` as in section 10.
- `tests/test_cli.py`: the default outputs, `--output-dir`, the fast overrides,
  and `--video` on a one-observation run.
- `tests/test_generated_assets.py`: the pinned sentences as in section 9.

### Mutation checks

The protocol is the first track's: each mutation applied in turn with bytecode
caching off and the tree confirmed clean between mutants, the whole suite run,
and the plan naming the exact test IDs expected to fail. For each kind of test,
one mutation it must catch and one it will not:

| Test | Catches | Does not catch |
|---|---|---|
| format validation | the writer omitting `cov`; saving `cloud` as float64 | wrong values in arrays of the right shape |
| recorder moments | an unweighted mean; taking history from index 1, so row 0 is not the prior | an error made identically in the test's own arithmetic |
| `thin` | the offset 0.5 (irregular strides); a plain stride on a weighted state | another valid equal-weight scheme |
| generator untouched | thinning with the filter's own generator | a recorder that reads state without advancing it |
| `meta` against the module | changing `TRUE_M` without re-recording | a stale file whose `meta` still matches (regeneration catches it) |
| regeneration | a changed default control; a changed seed; a change to the sampler that alters the trajectory; a stale file | any change to drawing; a formula wrong in the recorder, since both sides share it (the moments test covers that); a change below 1e-9 |
| truth inside the prior | `TRUE_M = 0.15` | a truth 6% from the bound |
| fixed axes | autoscaling a trace panel; turning constrained layout back on | a limit that is fixed but badly chosen |
| frame size | a canvas 4.6 inches tall | a size that is even but not the declared one, which the declared-size assertion catches separately |
| prior frame, pacing | dropping row 0; uniform dwell | a dwell schedule that is plausible but different from the spec's, unless its ends or total change |
| role colours | swapping truth and belief | a wrong but distinct shade |
| encode smoke | the encoder missing; an odd frame size reaching the encoder | whether a browser can play the file |
| rendered-site check | a page losing a theme's video; a missing poster; `autoplay`; a third themed element; the wrong clip on a page | whether the clip looks right or plays in Safari |
| `data=` by position | `ax.plot(x, "rate", "r-", data=frame)` | a colour passed through a variable |
| prose | "committed GIF" left in the sentence; a number differing from the replay | a sentence that is well formed and misleading |

### Checks by eye

- Contact sheets of every clip (prior, first, eighth and last frame) and both
  static figures, in both themes, at full width and at 390 pixels, committed
  under `docs/plans/2026-11-19-replay-and-animation/` with the second pull
  request.
- Each clip played through in both themes in Chrome, Firefox and Safari on the
  deployed preview or a local server: it starts, loops, pauses off screen, and
  stays still under reduced motion.
- The phone harness from PR #21 at 390 pixels: no page wider than the viewport.
- The 2D page's "tight knot at the truth" sentence, and each new caption,
  against its figure.

## Measurements

All run on 2026-11-18 in the worktree at `7d40c92`, on macOS arm64 (Apple M1
Max), Python 3.14.0, NumPy 2.4.2, SciPy 1.17.1, Matplotlib 3.10.8, Quarto
1.9.32, through `uv run --frozen`. The probe scripts were in a session scratch
directory and are not kept; each entry says what was run.

- **M1. Cross-platform agreement.** The Getting Started cell (two masses, 40
  observations, 1,500 particles, K = 1, 2, 3, seed 42) run locally printed
  `{1: 2.7885139471082757e-205, 2: 0.6086294618644883, 3:
  0.3913705381355014}`. The deployed page, fetched with `curl`, shows the same
  with `0.3913705381355013`. The deploy workflow installs Python 3.12 (read);
  the local run used 3.14.
- **M2. Candidate replay files.** A scratch recorder ran each demo's default
  inference and wrote a candidate archive with 500 displayed particles. Sizes
  in kB: 73.5, 169.3, 392.3, 366.1, 9.7, 193.9, 548.4 in the order of section
  1's list; 1.80 MB in all. Whole clouds compressed: 30.7 MB in float64, 13.0
  MB in float32. Inference times in seconds: 0.2, 0.4, 0.8, 2.3, 1.4, 40.1
  (density, including its summaries), 4.1. Final effective sample sizes were
  1630 of 2000 to 970 of 1000. The 2D run at M = 0.10 ended at (1.4169,
  -1.2223, 0.1000) with standard deviations (0.1111, 0.1148, 0.0012). The
  model comparison ended at 0.39903515 and 0.60096485. The echolocation prior
  has standard deviation 9.2 to 9.3 per axis; after one observation 0.16 to
  0.26; at the end 0.017 to 0.030; 5 of 500 displayed prior particles lie
  within plus or minus 4.5. The candidates used the offset 0.5 and lacked the
  profile arrays, so the final files will differ slightly in size.
- **M3. Density time and the 2D cell.** Density inference alone: 33.0 s. The
  "Into the Plane" cell at 0.15 and at 0.10: the numbers in section 8, with
  weak-field strengths 0.0702 and 0.0468.
- **M4. Repeatability.** The six fast demos recorded twice, and once more in a
  separate Python 3.12 environment from the same lock: every array bitwise
  equal in all three.
- **M5. `meta` round trip.** A JSON string holding "μ σ" saved with
  `savez_compressed` and read with `allow_pickle=False` came back equal.
- **M6. Thinning offsets.** For uniform weights and N of 1000 to 6000 with m =
  500: offset 0.5 gave strides such as [4, 5] and [11, 12, 13] with a target
  exactly on a cumulative step; offset 0.618 gave one stride each, the nearest
  target 0.056 to 0.47 of a weight from a step.
- **M7. Rendering.** A scratch renderer using `clocks.viz.style`, explicit
  axes rectangles, artists updated in place and raw frames piped to FFmpeg
  9.0.2. 2D clip: 1500 by 900, 14.8 s, 51 draws at 23 ms, 2.0 to 2.1 s per
  theme in all, 330 and 309 kB, posters 76 and 75 kB; at 750 by 450, 1.4 s and
  155 and 145 kB. Echolocation clip: 1500 by 1000, 17.8 s, 356 draws at 33 ms,
  12.2 to 12.8 s per theme, 1177 and 1095 kB, posters 167 and 152 kB; at 750
  by 500, 10.6 to 10.8 s and 566 and 534 kB. A first attempt at 7.5 by 4.6
  inches failed with a frame of 1500 by 919. Canvas heights 4.2, 4.5, 4.75,
  5.0, 5.4 and 5.5 inches gave exact pixels at 100 and 200 dpi; 4.6 inches
  gave 459 and 919, and 736 at 160 dpi.
- **M8. Packaged encoder.** `uv run --with imageio-ffmpeg` installed 0.8.1
  without touching `uv.lock`; its binary reports FFmpeg 9.0 configured with
  `--enable-libx264`. PyPI lists wheels for `manylinux_2_28_x86_64` (67 MB),
  `manylinux_2_28_aarch64`, macOS and Windows, uploaded 2026-11-03.
- **M9. Encoder bytes.** The same clip encoded by the system FFmpeg 9.0.2 and
  by the packaged 9.0 had equal sizes to the kilobyte and different SHA-256.
  `ffprobe` on the latter: H.264 High, `yuv420p`, 1500 by 900, 20 fps, 296
  frames.
- **M10. Quarto 1.9.32.** A scratch website project with the site's theme
  files. A pre-render `.py` script ran in the project directory under the
  `uv run` Python with `QUARTO_PROJECT_RENDER_ALL=1`; on a single-file render
  it ran without that variable. Its `media/` files matched by `resources`
  patterns were copied to the output; files written by a cell and matched by a
  pattern were not. Raw `<video>` markup passed through with all attributes; a
  `<figure>` gained the class `figure`; Markdown in `<figcaption>` rendered.
  `![caption](x.mp4){width= height=}` produced `<video src width height
  controls>` inside a captioned figure and copied the file. A cell with
  `renderings: [light, dark]` and two HTML outputs produced `cell-output-
  display light-content` and `dark-content` wrappers, with `fig-cap` entries as
  `<p>`. A cell printed its working directory as the page's directory. The
  compiled CSS contains `body.quarto-dark .light-content{display:none
  !important}` and its counterpart.
- **M11. Runner image.** `Ubuntu2404-Readme.md` in `actions/runner-images`,
  image version 20261111.370.1, fetched 2026-11-18: no occurrence of "ffmpeg".
  The repository README maps `ubuntu-latest` to Ubuntu 24.04.
- **M12. Site render time.** Local `quarto render` of the current site: 38.1 s.
  `gh run view` step times for "Render site": 43 s (site-check run
  44471577791) and 61 s (deploy run 44473013998).
- **M13. Test suite.** `uv run pytest -q`: 1079 passed, 2 deselected in 32.1 s.
- **M14. Browser.** A page built by the M10 project with the section 4 markup,
  script and real clips, served locally with request logging, opened in
  headless Chrome 157. Default: both posters requested, then the light video;
  the light clip playing at 2.5 s; after `quartoToggleColorScheme()`, the dark
  video requested, the light paused, the dark playing. With `--force-prefers-
  reduced-motion`: no video requested, both paused. Box 750 by 450 at
  `DOMContentLoaded`. At a requested width of 390 Chrome gave a 500-pixel
  viewport: without the CSS rule the document was 776 wide; with it the video
  was 449 by 269 and the document 500.
- **M15. The two follow-ups.** `_style_violations('ax.plot(x, "rate", "r-",
  data=frame)')` returned `[]`. `_resolve_source("a%00b.png", ...)` raised
  `ValueError: lstat: embedded null character in path`.
- **M16. Current check.** After the local render,
  `scripts/check_site_figures.py` printed "Figure contract OK: 9 pages
  checked." Counting elements with a theme class in every rendered page gave 2
  on each of the nine plotting pages and 0 on the other six, so the rule that
  unlisted pages hold none is satisfied today.

## Risks

- **The runner's trajectory may differ from this machine's.** One measurement
  (M1) says the platforms agree to the last digit or so; nothing says they
  always will. The first pull request finds out, because CI runs the
  regeneration tests on Linux against files recorded on macOS. If they fail
  there, the files are not wrong, the comparison is too strict for two
  platforms. The fallback, to be decided then and not built now, is to compare
  trajectories tightly only on the platform named in the file's `created_with`
  and elsewhere to require each posterior mean within a stated fraction of the
  recorded standard deviation. That is a weaker test and its looseness would
  need its own mutation check.
- **A NumPy or SciPy upgrade can change the numbers.** The regeneration tests
  then fail until the files are re-recorded. That is the test doing its job,
  but it couples a lock bump to a data commit.
- **Quarto 1.11.9 was not run.** Every Quarto behaviour above was measured on
  1.9.32. Pre-render scripts, `resources` and the theme classes are
  long-standing features and are believed unchanged. The site check runs on
  1.11.9 when the third pull request is marked ready, and the rendered-site
  check fails if a clip, a poster or the markup is missing.
- **Browsers other than headless Chrome were not tested.** Safari needs
  `playsinline` and `muted` for scripted playback and byte-range responses for
  video; the markup has the first two, and GitHub Pages is believed to serve
  ranges. This is a check by eye on the deployed site.
- **The phone width was not reached.** Headless Chrome would not go below 500
  pixels. The 390-pixel behaviour rests on the same CSS rule and is checked
  with the existing phone harness.
- **Four clips and both static figures are specified, not drawn.** The two-mass
  2D clip, with a map and six traces in 750 by 550, is the tightest layout. The
  second pull request's contact sheets are where a layout that does not fit is
  found, before any page depends on it.
- **The echolocation prior frame is nearly empty**, as section 5 says. If
  review finds that unacceptable, the alternatives are a wider fixed cube,
  which shrinks the head again, or a second, zoomed-out view.
- **The encoder is a 22 to 67 MB development dependency.** It is believed to be
  kept by the runner's uv cache between runs, and it is not a dependency of the
  library. Its binary is a GPL build of FFmpeg; the project invokes it as a
  separate program and does not redistribute it.
- **Replay files are binary.** A re-recording shows in a pull request as changed
  files with no readable diff. The pull request must quote the changed final
  estimates, which the demo command prints.
- **The hidden theme's poster is downloaded**, 75 to 170 kB per clip. Accepted.
- **Text in a clip scales with the video.** At 390 pixels, labels are about 7
  CSS pixels, the same known limit as the plots.

## Not doing

- A scrubber, play-on-scroll, or any in-browser drawing of replay data.
- A JSON or other browser-readable export of the replay.
- Pinning or committing video, poster or figure bytes; image-comparison tests.
- A GIF fallback for machines without an encoder.
- A seed spread or a one-mass case for model comparison.
- A new landing-page visual; the landing page keeps the 2D clip.
- Recording the generating commit inside a replay file. The file's own history
  in git identifies it; `created_with` records versions and platform.
- Parallel rendering, unless the measured runner time calls for it.

## Where this design is least sure

1. **The cross-platform tolerance** (section 3). It rests on one comparison.
   The sequencing exists to test it first.
2. **The echolocation scene and the undrawn layouts** (section 5). The scene
   was prototyped only roughly, its prior frame is nearly empty by
   construction, and the collapsed cloud is small with no inset; the two-mass
   clips and the model-comparison clip have not been drawn at all.
3. **Hand-written video markup with a playback script** (section 4). It was
   measured in one browser and one Quarto version. The choice of raw HTML over
   generated markup, and of script-driven playback over the `autoplay`
   attribute, should be challenged; the second was chosen because an
   `autoplay` video hidden by the theme switch would still load and play.
