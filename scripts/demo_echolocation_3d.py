"""3D gravitational echolocation demo: exterior mass, differential sensing.

A 3x3x3 lattice of 27 clocks (the "head") senses a single point mass
placed outside it. The head has no external time reference, so the filter
uses orthonormal contrasts while the animation shows centered labeled clock
rates. The camera nearly completes an orbit over the animation.

Demo seed and range are curated for visual clarity (disclosed in README
and on the site page); the range study carries the quantitative argument.

The demo uses the scenario's minimum exterior range for a visually resolvable
example. Quantitative calibration and certification belong to the separate
range study.
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
DEMO_RANGE_R = 2.0  # circumradii — scenario minimum; the range that converges
DEMO_SEED = 4
OUTPUT_PATH = Path("output/demo_echolocation_3d.gif")


def main() -> None:
    clock_array = build_head_lattice()
    validate_echo_geometry(DEMO_RANGE_R, ECHO_M_TRUE, clock_array)
    mass_config = echo_mass_config(DEMO_RANGE_R)
    truth = np.append(echo_mass_position(DEMO_RANGE_R), ECHO_M_TRUE)
    print(
        f"True mass: M={ECHO_M_TRUE} at {truth[:3].round(2)} "
        f"({DEMO_RANGE_R} circumradii)"
    )

    _, centered_obs, filter_obs = make_echo_observations(DEMO_SEED, DEMO_RANGE_R)
    pf = build_echolocation_filter(DEMO_SEED)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating animation → {OUTPUT_PATH}")
    animate_echolocation(
        clock_array=clock_array,
        mass_config=mass_config,
        observations=centered_obs,
        filter_observations=filter_obs,
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
