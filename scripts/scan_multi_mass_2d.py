"""Development-seed scan for rigorous SMC controls in the 2-D scenario."""

import argparse
import math
import statistics
from itertools import product
from multiprocessing import Pool
from numbers import Integral
from pathlib import Path

from clocks._calibration import (
    DEVELOPMENT_ESS_TARGETS,
    DEVELOPMENT_PROPOSAL_SCALES,
    DEVELOPMENT_REJUVENATION_STEPS,
    build_study_document,
    control_grid_from_cells,
    validate_multi_results,
    write_study,
)
from clocks._scenarios import (
    MULTI_ESS_TARGET,
    MULTI_PROPOSAL_SCALE,
    MULTI_REJUVENATION_STEPS,
    PASS_TOLERANCE,
    RunResult,
    run_multi_mass_2d,
)


def _study_json_path(seed_block: int) -> Path:
    """Keep every seed block in a separately named raw evidence file."""
    return Path(f"output/multi_mass_2d_study_seed_block_{seed_block}.json")


def _write_study(
    path: Path,
    *,
    seed_block: int,
    seeds: tuple[int, ...],
    cells: list[tuple[float, int, float]],
    results: list[RunResult],
) -> None:
    expected_tuples = {
        (ess_target, steps, scale, seed)
        for ess_target, steps, scale in cells
        for seed in seeds
    }
    validate_multi_results(results, expected_tuples=expected_tuples)
    study = build_study_document(
        study="multi_mass_2d",
        seed_block=seed_block,
        seeds=seeds,
        control_grid=control_grid_from_cells(cells),
        tolerances={"absolute_parameter_error": PASS_TOLERANCE},
        results=results,
    )
    write_study(path, study)


def _reject_duplicates(name: str, values: list[float] | list[int]) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{name} controls contain duplicate values")


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
    if seed_block >= 400 and any(
        value is not None for value in (ess_targets, steps, scales)
    ):
        raise ValueError(
            "explicit control overrides are forbidden for protected seed blocks"
        )
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
    _reject_duplicates("ess-target", selected_ess)
    _reject_duplicates("rejuvenation-step", selected_steps)
    _reject_duplicates("proposal-scale", selected_scales)
    cells = list(product(selected_ess, selected_steps, selected_scales))
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

    json_path = _study_json_path(args.seed_block)
    _write_study(
        json_path,
        seed_block=args.seed_block,
        seeds=seeds,
        cells=cells,
        results=[result for _, result in results],
    )

    grouped: dict[tuple, list[RunResult]] = {}
    for key, result in results:
        grouped.setdefault(key, []).append(result)
    expected_runs_per_cell = len(seeds)
    if any(len(cell) != expected_runs_per_cell for cell in grouped.values()):
        raise RuntimeError("scan produced unequal run counts across control cells")

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
    print(f"raw study: {json_path}")


if __name__ == "__main__":
    main()
