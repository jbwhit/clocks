"""Development-seed scan for rigorous SMC controls in the 2-D scenario."""

import argparse
import statistics
from multiprocessing import Pool

from clocks._scenarios import RunResult, run_multi_mass_2d

TUNING_SEEDS = tuple(range(12))
CERTIFICATION_SEEDS = tuple(range(400, 412))


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
    parser.add_argument("--ess-targets", type=float, nargs="+", default=[0.7, 0.8, 0.9])
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--scales", type=float, nargs="+", default=[1.5, 2.0, 2.38])
    parser.add_argument("--certification", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    seeds = CERTIFICATION_SEEDS if args.certification else TUNING_SEEDS
    cells = [
        (ess_target, steps, scale)
        for ess_target in args.ess_targets
        for steps in args.steps
        for scale in args.scales
    ]
    runs = [(seed, *cell) for cell in cells for seed in seeds]
    with Pool(args.workers) as pool:
        results = pool.map(_run, runs)

    grouped: dict[tuple, list[RunResult]] = {}
    for key, result in results:
        grouped.setdefault(key, []).append(result)

    ranked = []
    for (ess_target, steps, scale), cell in sorted(grouped.items()):
        n_pass = sum(result["passed"] for result in cell)
        median_error = statistics.median(result["max_abs_error"] for result in cell)
        ranked.append(((-n_pass, median_error), (ess_target, steps, scale)))
        print(
            f"ess={ess_target:.2f} steps={steps} scale={scale:.2f}: "
            f"{n_pass}/{len(cell)}, median max error={median_error:.3f}"
        )
    ranked.sort()
    _, winner = ranked[0]
    print(f"winner: ess={winner[0]:.2f}, steps={winner[1]}, scale={winner[2]:.2f}")


if __name__ == "__main__":
    main()
