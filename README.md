# clocks

Gravitational time dilation simulation and inference library.

**Website:** [jbwhit.github.io/clocks](https://jbwhit.github.io/clocks/) — the full story, from GPS corrections through model comparison, continuous densities, and 3D gravitational echolocation.

Place atomic clocks near a hidden mass — they tick slower in the gravitational well. A particle filter (Sequential Monte Carlo) watches the noisy tick rates and infers the mass's position and magnitude — GPS run in reverse.

## How it works

**Forward model:** In simulation units where G = c = 1, sum the Newtonian
potentials $\Phi_i=-\sum_j M_j/r_{ij}$ and map them to clock rates with
$\sqrt{1+2\Phi_i}$. This is a pedagogical weak-field surrogate, not an exact
strong-field solution; every state must satisfy the conservative policy
$|2\Phi_i|\le0.1$.

**Inverse problem:** Adaptive tempered resample-move SMC maintains a weighted
cloud of hypotheses. ESS-selected likelihood increments control each update;
resampling is followed by symmetric random-walk Metropolis-Hastings moves that
preserve the intermediate target. The prior is the normalized configured box
conditioned on ordering and weak-field validity.

The point-mass forward model and core particle filter are dimension-agnostic, with examples in 1D, 2D, and 3D.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Use as a library

The package exposes stable end-to-end entry points — `simulate`, `infer`,
and `simulate_and_infer` — configured via public dataclasses
(`SimulationConfig`, `InferenceConfig`, ...), plus `build_particle_filter`
for driving the filter observation-by-observation. A complete
simulate-then-infer round trip runs live on the site's
[Getting Started](https://jbwhit.github.io/clocks/reproduce/getting-started.html)
page, and the filter machinery is documented in
[The Particle Filter](https://jbwhit.github.io/clocks/method/the-particle-filter.html).
For accepted stages, exact subnormal weights are rounded nearest-even; a narrow
near-normal strip can also change. Ordinary weights remain approximate. If the
strict precision cap cannot settle a required rounding decision,
`ParticleFilter.update` raises `WeightRoundingUndecided` and rolls the update
back. Do not blindly retry unchanged inputs; correct rounding does not promise
that every finite population completes.

## Run the demos

Seven demos. Each writes a replay archive and a PNG; the six animated
ones also write an MP4 with `--video`:

```bash
uv run demo-1d                  # → output/demo_1d.npz, output/demo_1d.png
uv run demo-2d                  # → output/demo_2d.npz, output/demo_2d.png
uv run demo-multi-mass          # → output/demo_multi_mass.npz, output/demo_multi_mass.png
uv run demo-multi-mass-2d       # → output/demo_multi_mass_2d.npz, output/demo_multi_mass_2d.png
uv run demo-model-comparison    # → output/demo_model_comparison.npz, output/demo_model_comparison.png
uv run demo-density             # → output/demo_density.npz, output/demo_density.png
uv run demo-echolocation-3d     # → output/demo_echolocation_3d.npz, output/demo_echolocation_3d.png
uv run demo-2d --video          # also → output/demo_2d.mp4
```

Each command records its run, prints the truth and its final estimate from
the recording, and writes a replay archive and a PNG: the final frame of the
animation for six of them, the density figure for `demo-density`. The six
animated demos also take `--video`, which writes the animation as an MP4 and
needs an FFmpeg encoder (`uv sync` provides one in a checkout). `--theme dark`
draws the dark version, and `--output-dir` chooses the directory (`output/` by
default). A run of one of the six without `--video` leaves an MP4 from an
earlier run where it is and says that it was not updated.

The site animates six of the demos and draws the density figure, all from
replay data, with commentary:
[jbwhit.github.io/clocks](https://jbwhit.github.io/clocks/). The site's clips
and posters are build products and are not committed. What is committed is
replay data, `assets/replays/<demo>.npz`: what each packaged run produced, in
numbers, tied to the code by tests of its inputs and of bounded summaries. The
echolocation range study behind the site's final page:
`scripts/scan_echolocation_range.py`.

## Upgrading from 0.1

Version 0.2.0 changes what the demos write and removes the old plotting names.
This note covers everything since 0.1.

- Removed from `clocks.viz`: the six `animate_*` functions, the eleven `plot_*`
  functions and the two `create_*_dashboard` functions. Added since 0.1:
  `load_replay`, `render_clip`, `density_figure`, `range_study_figure`,
  `style`, `palette`, `Palette`, `THEMES` and `show`.
- `--output PATH` becomes `--output-dir DIR`; the file names are the demo's,
  and options can no longer be abbreviated.
- Each demo writes a replay archive and a PNG where six of them wrote a GIF; an
  MP4 is opt-in for those six, with `--video`. `demo-density` already wrote a
  PNG and now writes its archive beside it.

## Run tests

`uv run pytest` runs the default non-slow suite; run `uv run pytest -m slow`
for long fixed-seed regression cases. Corrected SMC controls and tolerances
were frozen on development seeds before the reserved certification block,
which by protocol is for one run and was run under those frozen settings. Its
fixed-seed outcomes are deterministic
regression evidence, never population reliability estimates.

```bash
uv run pytest
uv run ruff format --check .
uv run ruff check .
```

## Project structure

```
src/clocks/
    types.py       Data structures (MassConfig, ClockArray, Observation, ParticleState)
    config.py      Public config dataclasses (SimulationConfig, InferenceConfig, ...)
    results.py     Public result dataclasses (SimulationResult, InferenceResult, ...)
    api.py         End-to-end entry points (simulate, infer, build_particle_filter)
    physics.py     Strict weak-field forward models
    noise.py       Gaussian noise model and log-likelihood
    inference.py   Adaptive tempered resample-move SMC
    viz.py         Public facade over _figures, _clips, _echo_study, _replay and _style
    _style.py      Shared figure style: palette, theme context, bundled font (_fonts/)
    _support.py    Conditional prior support and rejection sampling
    _replay.py     Replay archives: record, validate, save, load, compare
    _figures.py    Static figures drawn from a replay, and the trace helpers
    _clips.py      Clips drawn from replays (_clip_*.py hold the six builders)
    _video.py      Raw frames to H.264 through FFmpeg, and the decode check
    _site_media.py The site's clips and posters, drawn before Quarto renders
    _scenarios.py  Shared scenario builders for demos/scan harnesses/tests
    _echo_study.py Reporting helpers for the echolocation range study
    _demos/        Packaged implementations for all seven console commands
scripts/
    demo_*.py                     Thin source-tree wrappers around clocks._demos
    scan_echolocation_range.py    Resolution-vs-range study for 3D echolocation
    scan_multi_mass_2d.py         Seed-scan harness for the multi-mass-2D scenario
tests/
    Unit, scenario, visualization, echo-study, and slow acceptance tests —
    see `tests/` for the current inventory.
```

Dependency structure and the public/private boundary:
[Architecture](https://jbwhit.github.io/clocks/reproduce/architecture.html).
