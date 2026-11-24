"""Localize an exterior mass from contrast-space 3-D clock readings."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
from numpy.typing import NDArray

from clocks._demos._common import add_common_arguments
from clocks._replay import (
    Replay,
    Scenario,
    filter_controls,
    record_filter,
    scenario_meta,
    validate_replay,
)
from clocks._scenarios import (
    ECHO_M_TRUE,
    ECHO_MASS_RANGE,
    ECHO_N_OBSERVATIONS,
    ECHO_N_PARTICLES,
    ECHO_NOISE_STD,
    ECHO_POSITION_HALFWIDTH,
    build_echolocation_filter,
    build_head_lattice,
    echo_mass_config,
    echo_mass_position,
    make_echo_observations,
    validate_echo_geometry,
)
from clocks.physics import clock_rates, clock_rates_batch
from clocks.types import ParticleState
from clocks.viz import animate_echolocation

DEMO_RANGE_R = 2.0
DEMO_SEED = 4
OUTPUT_PATH = Path("output/demo_echolocation_3d.gif")
VIEW_HEAD_HALFWIDTH = 1.0  # the head lattice spans [-1, 1] on every axis
VIEW_SPREAD = 4.0  # standard deviations of each coordinate kept in view
VIEW_PADDING = 0.3


def scenario() -> Scenario:
    """The packaged inputs of this demo, read from the module constants."""
    clock_array = build_head_lattice()
    hw = ECHO_POSITION_HALFWIDTH
    return Scenario(
        demo="demo_echolocation_3d",
        clock_array=clock_array,
        true_params=np.append(echo_mass_position(DEMO_RANGE_R), ECHO_M_TRUE),
        true_rates=clock_rates(echo_mass_config(DEMO_RANGE_R), clock_array),
        param_names=("x", "y", "z", "M"),
        prior_ranges=((-hw, hw),) * 3 + (ECHO_MASS_RANGE,),
        seed=DEMO_SEED,
        noise_std=ECHO_NOISE_STD,
        n_observations=ECHO_N_OBSERVATIONS,
        n_particles=ECHO_N_PARTICLES,
    )


def record(
    *,
    n_observations: int = ECHO_N_OBSERVATIONS,
    n_particles: int = ECHO_N_PARTICLES,
) -> Replay:
    """Run the packaged inference once and return it as a validated replay.

    The filter consumes the contrasts; the archive's ``observations`` are the
    centred readings the clip displays.
    """
    packaged = scenario()
    clock_array = packaged.clock_array
    _, display, contrasts = make_echo_observations(
        DEMO_SEED, DEMO_RANGE_R, n_observations=n_observations
    )
    particle_filter = build_echolocation_filter(DEMO_SEED, n_particles=n_particles)

    def predict_centred(particles: NDArray[np.float64]) -> NDArray[np.float64]:
        rates = clock_rates_batch(particles[:, :3], particles[:, 3], clock_array)
        return rates - rates.mean(axis=1, keepdims=True)

    recorded = record_filter(
        particle_filter,
        contrasts,
        meta=scenario_meta(
            packaged,
            kind="filter",
            n_observations=n_observations,
            n_particles=n_particles,
            controls=filter_controls(particle_filter),
        ),
        arrays={
            "clock_positions": clock_array.positions,
            "true_params": packaged.true_params,
            "true_rates": packaged.true_rates,
            "observations": np.array([o.rates for o in display], dtype=np.float64),
            "filter_observations": np.array(
                [o.rates for o in contrasts], dtype=np.float64
            ),
        },
        predict=predict_centred,
    )
    arrays = dict(recorded.arrays)
    sd = np.sqrt(np.diagonal(arrays["cov"], axis1=1, axis2=2))
    bounds = close_view(packaged.true_params[:3], arrays["mean"][1:, :3], sd[1:, :3])
    arrays["view_bounds"] = np.array(bounds, dtype=np.float64)
    arrays["weight_in_view"] = weight_in_view(particle_filter.history, bounds)
    replay = Replay(recorded.meta, arrays)
    validate_replay(replay)
    return replay


def close_view(
    truth_xyz: NDArray, mean_xyz: NDArray, sd_xyz: NDArray
) -> tuple[float, float]:
    """The close view's cube, as one (low, high) pair shared by x, y and z.

    The smallest cube holding the head, the truth and each row's mean ± four
    standard deviations, padded and rounded outward to 0.1. ``mean_xyz`` and
    ``sd_xyz`` hold one row per observation (k >= 1). The tenths are rounded
    to six decimals before the floor or ceiling, so 14.000000000000002 stays 14.
    """
    truth = np.asarray(truth_xyz, dtype=np.float64)
    mean = np.asarray(mean_xyz, dtype=np.float64)
    sd = np.asarray(sd_xyz, dtype=np.float64)
    low = min(
        -VIEW_HEAD_HALFWIDTH, float(truth.min()), float((mean - VIEW_SPREAD * sd).min())
    )
    high = max(
        VIEW_HEAD_HALFWIDTH, float(truth.max()), float((mean + VIEW_SPREAD * sd).max())
    )
    low_tenths = round((low - VIEW_PADDING) * 10, 6)
    high_tenths = round((high + VIEW_PADDING) * 10, 6)
    return math.floor(low_tenths) / 10, math.ceil(high_tenths) / 10


def weight_in_view(
    states: Sequence[ParticleState], bounds: tuple[float, float]
) -> NDArray[np.float64]:
    """Per state, the weight of the particles whose x, y and z lie in ``bounds``.

    Bounds are inclusive; the mass column is not tested.
    """
    low, high = bounds
    weights = []
    for state in states:
        positions = state.particles[:, :3]
        inside = np.all((positions >= low) & (positions <= high), axis=1)
        weights.append(float(state.weights[inside].sum()))
    return np.array(weights, dtype=np.float64)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without running the simulation."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(
        parser,
        output=OUTPUT_PATH,
        observations=ECHO_N_OBSERVATIONS,
        particles=ECHO_N_PARTICLES,
    )
    return parser


def _run_demo(args: argparse.Namespace) -> None:
    clock_array = build_head_lattice()
    validate_echo_geometry(DEMO_RANGE_R, ECHO_M_TRUE, clock_array)
    mass_config = echo_mass_config(DEMO_RANGE_R)
    truth = np.append(echo_mass_position(DEMO_RANGE_R), ECHO_M_TRUE)
    print(
        f"True mass: M={ECHO_M_TRUE} at {truth[:3].round(2)} "
        f"({DEMO_RANGE_R} circumradii)"
    )
    _, display_observations, filter_observations = make_echo_observations(
        DEMO_SEED,
        DEMO_RANGE_R,
        n_observations=args.observations,
    )
    particle_filter = build_echolocation_filter(DEMO_SEED, n_particles=args.particles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating animation → {args.output}")
    animate_echolocation(
        clock_array=clock_array,
        mass_config=mass_config,
        observations=display_observations,
        filter_observations=filter_observations,
        pf=particle_filter,
        output_path=args.output,
        fps=4,
    )
    estimate = particle_filter.estimate()
    print(f"\nFinal estimate after {args.observations} observations:")
    for index, label in enumerate(("x", "y", "z", "M")):
        print(
            f"  {label} = {estimate['mean'][index]:.3f} ± "
            f"{estimate['std'][index]:.3f}  (true: {truth[index]:.3f})"
        )
    print(f"  ESS = {estimate['ess']:.0f} / {args.particles}")
    print(f"\nSaved to {args.output}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the demo using command-line arguments from *argv*."""
    args = build_parser().parse_args(argv)
    _run_demo(args)
    return 0


if __name__ == "__main__":
    main()
