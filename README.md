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

## Run the demos

Six animated demos and one static figure:

```bash
uv run demo-1d                  # → output/demo_1d.gif
uv run demo-2d                  # → output/demo_2d.gif
uv run demo-multi-mass          # → output/demo_multi_mass.gif
uv run demo-multi-mass-2d       # → output/demo_multi_mass_2d.gif
uv run demo-model-comparison    # → output/demo_model_comparison.gif
uv run demo-density             # → output/demo_density.png
uv run demo-echolocation-3d     # → output/demo_echolocation_3d.gif
```

![2D inference demo](assets/demo_2d.gif)

The committed demo media predates the corrected physics and SMC
implementation. The development controls are frozen, but the media remains
illustrative until one-shot certification and asset regeneration are complete.

Most GIF demos animate the physical setup, the particle cloud converging,
and the estimates' uncertainty; `demo-model-comparison` instead tracks
the posterior probability over candidate mass counts, and `demo-density`
produces a static comparison figure. All seven, with commentary:
[jbwhit.github.io/clocks](https://jbwhit.github.io/clocks/). The
echolocation range study behind the site's final page:
`scripts/scan_echolocation_range.py`.

## Run tests

`uv run pytest` runs the default non-slow suite; run `uv run pytest -m slow`
for long fixed-seed regression cases. Corrected SMC controls and tolerances
were frozen on development seeds before the reserved certification block,
which has not been run; fixed-seed outcomes are never population reliability
estimates.

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
    viz.py         Plotting and animation facade (_panels.py, _panels3d.py, _animate.py)
    _support.py    Conditional prior support and rejection sampling
    _panels3d.py   3D plotting primitives for the echolocation dashboard
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
