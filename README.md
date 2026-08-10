# clocks

Gravitational time dilation simulation and inference library.

**Website:** [jbwhit.github.io/clocks](https://jbwhit.github.io/clocks/) — the full story, from GPS corrections through model comparison, continuous densities, and 3D gravitational echolocation.

Place atomic clocks near a hidden mass — they tick slower in the gravitational well. A particle filter (Sequential Monte Carlo) watches the noisy tick rates and infers the mass's position and magnitude — GPS run in reverse.

## How it works

**Forward model:** Given one or more point masses at positions **x**_j with masses M_j, compute the Newtonian gravitational potential at each clock, then derive the weak-field GR time dilation factor (tick rate). Uses simulation units where G = c = 1.

**Inverse problem:** A particle filter maintains a cloud of hypotheses for the unknown parameters. Each observation (noisy clock rates) reweights particles by likelihood, and resampling with jitter reduces particle degeneracy. In well-conditioned scenarios, the cloud concentrates near the true parameters.

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

Most GIF demos animate the physical setup, the particle cloud converging,
and the estimates' uncertainty; `demo-model-comparison` instead tracks
the posterior probability over candidate mass counts, and `demo-density`
produces a static comparison figure. All seven, with commentary:
[jbwhit.github.io/clocks](https://jbwhit.github.io/clocks/). The
echolocation range study behind the site's final page:
`scripts/scan_echolocation_range.py`.

## Run tests

`uv run pytest` runs the default non-slow suite; run `uv run pytest -m slow` for the long acceptance scans.

```bash
uv run pytest
uv run ruff check src/ tests/ scripts/   # lint
```

## Project structure

```
src/clocks/
    types.py       Data structures (MassConfig, ClockArray, Observation, ParticleState)
    config.py      Public config dataclasses (SimulationConfig, InferenceConfig, ...)
    results.py     Public result dataclasses (SimulationResult, InferenceResult, ...)
    api.py         End-to-end entry points (simulate, infer, build_particle_filter)
    physics.py     Forward model: mass config → clock tick rates
    noise.py       Gaussian noise model and log-likelihood
    inference.py   Particle filter (SMC with systematic/stratified/residual resampling)
    viz.py         Plotting and animation facade (_panels.py, _panels3d.py, _animate.py)
    _panels3d.py   3D plotting primitives for the echolocation dashboard
    _scenarios.py  Shared scenario builders for demos/scan harnesses/tests
    _echo_study.py Reporting helpers for the echolocation range study
    _cli.py        Entry points for demo scripts
scripts/
    demo_1d.py                    1D end-to-end demo
    demo_2d.py                    2D end-to-end demo
    demo_multi_mass.py            Two masses in 1D
    demo_multi_mass_2d.py         Two masses in 2D (random clocks)
    demo_model_comparison.py      Bayesian model comparison (infer K)
    demo_density.py               Gaussian density forward model
    demo_echolocation_3d.py       27-clock 3D echolocation demo (rotating camera)
    scan_echolocation_range.py    Resolution-vs-range study for 3D echolocation
    scan_multi_mass_2d.py         Seed-scan harness for the multi-mass-2D scenario
tests/
    Unit, scenario, visualization, echo-study, and slow acceptance tests —
    see `tests/` for the current inventory.
```

Dependency structure and the public/private boundary:
[Architecture](https://jbwhit.github.io/clocks/reproduce/architecture.html).
