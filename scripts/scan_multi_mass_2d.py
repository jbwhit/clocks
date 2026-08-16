"""Development-seed scan for rigorous SMC controls in the 2-D scenario."""

import argparse
import math
import statistics
from itertools import product
from multiprocessing import Pool
from numbers import Integral

from clocks._scenarios import (
    MULTI_ESS_TARGET,
    MULTI_PROPOSAL_SCALE,
    MULTI_REJUVENATION_STEPS,
    RunResult,
    run_multi_mass_2d,
)

DEVELOPMENT_ESS_TARGETS = (0.7, 0.8, 0.9)
DEVELOPMENT_REJUVENATION_STEPS = (1, 2, 4)
DEVELOPMENT_PROPOSAL_SCALES = (1.5, 2.38, 3.0)


def _seeds_for_block(seed_block: int) -> tuple[int, ...]:
    """Return a valid development or reserved certification seed block."""
    if seed_block == 0:
        return tuple(range(12))
    if seed_block < 400 or seed_block % 100 != 0:
        raise ValueError(
            "seed block must be 0 (development) or a multiple of 100 from 400"
        )
    return tuple(range(seed_block, seed_block + 12))


def _control_cells(
    seed_block: int,
    ess_targets: list[float] | None,
    steps: list[int] | None,
    scales: list[float] | None,
) -> list[tuple[float, int, float]]:
    """Use the declared grid in development and one frozen cell in certification."""
    selected_ess = ess_targets or (
        list(DEVELOPMENT_ESS_TARGETS) if seed_block == 0 else [MULTI_ESS_TARGET]
    )
    selected_steps = steps or (
        list(DEVELOPMENT_REJUVENATION_STEPS)
        if seed_block == 0
        else [MULTI_REJUVENATION_STEPS]
    )
    selected_scales = scales or (
        list(DEVELOPMENT_PROPOSAL_SCALES) if seed_block == 0 else [MULTI_PROPOSAL_SCALE]
    )
    if any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in selected_ess):
        raise ValueError("ess-target controls must be finite and in (0, 1)")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) or value <= 0
        for value in selected_steps
    ):
        raise ValueError("rejuvenation-step controls must be positive integers")
    if any(not math.isfinite(value) or value <= 0.0 for value in selected_scales):
        raise ValueError("proposal-scale controls must be finite and positive")
    cells = list(product(selected_ess, selected_steps, selected_scales))
    if seed_block >= 400 and len(cells) != 1:
        raise ValueError("certification requires a single control cell")
    return cells


def _run(job: tuple[int, float, int, float]) -> tuple[tuple, RunResult]:
    seed, ess_target, steps, scale = job
    result = run_multi_mass_2d(
        seed,
        ess_target=ess_target,
        rejuvenation_steps=steps,
        proposal_scale=scale,
    )
    return (ess_target, steps, scale), result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ess-targets", type=float, nargs="+")
    parser.add_argument("--steps", type=int, nargs="+")
    parser.add_argument("--scales", type=float, nargs="+")
    parser.add_argument(
        "--seed-block",
        type=int,
        default=0,
        help="0 for development; unseen multiples of 100 from 400 for certification",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--per-run", action="store_true")
    args = parser.parse_args()

    try:
        seeds = _seeds_for_block(args.seed_block)
    except ValueError as error:
        parser.error(str(error))
    try:
        cells = _control_cells(
            args.seed_block, args.ess_targets, args.steps, args.scales
        )
    except ValueError as error:
        parser.error(str(error))
    runs = [(seed, *cell) for cell in cells for seed in seeds]
    with Pool(args.workers) as pool:
        results = pool.map(_run, runs)

    grouped: dict[tuple, list[RunResult]] = {}
    for key, result in results:
        grouped.setdefault(key, []).append(result)

    ranked = []
    for (ess_target, steps, scale), cell in sorted(grouped.items()):
        n_pass = sum(result["passed"] for result in cell)
        median_error = statistics.median(result["normalized_error"] for result in cell)
        median_evaluations = statistics.median(
            result["forward_model_evaluations"] for result in cell
        )
        ranked.append(
            (
                (-n_pass, median_error, median_evaluations, steps),
                (ess_target, steps, scale),
            )
        )
        print(
            f"ess={ess_target:.2f} steps={steps} scale={scale:.2f}: "
            f"{n_pass}/{len(cell)}, median normalized error={median_error:.3f}, "
            f"median forward evaluations={median_evaluations:.0f}"
        )
        if args.per_run:
            for result in sorted(cell, key=lambda item: item["seed"]):
                print(
                    f"    seed={result['seed']:>3} pass={int(result['passed'])} "
                    f"normalized_error={result['normalized_error']:.3f} "
                    f"forward_evaluations={result['forward_model_evaluations']}"
                )
    ranked.sort()
    _, winner = ranked[0]
    print(f"winner: ess={winner[0]:.2f}, steps={winner[1]}, scale={winner[2]:.2f}")


if __name__ == "__main__":
    main()
