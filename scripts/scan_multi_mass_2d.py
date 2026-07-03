"""Seed-scan harness for the multi-mass-2D scenario.

Tunes (jitter_tau, floor) on seeds 0-11, prints a per-cell pass table and
the winner under the spec's total order, and re-measures the fixed-jitter
baseline post-support-repair. See
docs/superpowers/specs/2026-07-02-annealed-jitter-design.md §3.

Usage:
    uv run scripts/scan_multi_mass_2d.py                 # tuning grid
    uv run scripts/scan_multi_mass_2d.py --baseline      # fixed-jitter baseline
    uv run scripts/scan_multi_mass_2d.py --holdout --taus 15 --floors 0.02
"""

import argparse
import statistics
from multiprocessing import Pool

from clocks._scenarios import RunResult, run_multi_mass_2d

TUNING_SEEDS = tuple(range(12))
HOLDOUT_SEEDS = tuple(range(100, 112))


def _run(job: tuple[int, str, float, float]) -> tuple[tuple, RunResult]:
    seed, jitter, floor, tau = job
    if jitter == "fixed":
        # tau is a display/sort key only here: jitter_tau must validate
        # (> 0) even when unused, so don't pass the 0.0 placeholder.
        result = run_multi_mass_2d(seed, jitter="fixed", jitter_std=floor)
    else:
        result = run_multi_mass_2d(
            seed, jitter="annealed", jitter_std=floor, jitter_tau=tau
        )
    return (jitter, tau, floor), result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taus", type=float, nargs="+", default=[5, 10, 15, 25, 40])
    parser.add_argument("--floors", type=float, nargs="+", default=[0.01, 0.02, 0.05])
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="fixed-jitter baseline over --floors instead of the annealed grid",
    )
    parser.add_argument(
        "--holdout", action="store_true", help="use holdout seeds 100-111"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--per-run",
        action="store_true",
        help="print per-run diagnostics (3-sigma coverage, max std, residual)",
    )
    args = parser.parse_args()

    seeds = HOLDOUT_SEEDS if args.holdout else TUNING_SEEDS
    if args.baseline:
        jobs = [("fixed", 0.0, floor) for floor in args.floors]
    else:
        jobs = [("annealed", tau, floor) for tau in args.taus for floor in args.floors]
    runs = [
        (seed, jitter, floor, tau) for (jitter, tau, floor) in jobs for seed in seeds
    ]

    with Pool(args.workers) as pool:
        results = pool.map(_run, runs)

    cells: dict[tuple, list[RunResult]] = {}
    for key, result in results:
        cells.setdefault(key, []).append(result)

    header = (
        f"{'mode':>9} {'tau':>6} {'floor':>6} {'pass':>6} {'med|err|':>9} {'resid':>7}"
    )
    print(header)
    ranked = []
    for (jitter, tau, floor), cell in sorted(cells.items()):
        n_pass = sum(r["passed"] for r in cell)
        med_err = statistics.median(r["max_abs_error"] for r in cell)
        med_resid = statistics.median(r["residual_over_noise"] for r in cell)
        ranked.append(((-n_pass, med_err, tau, floor), (jitter, tau, floor), n_pass))
        print(
            f"{jitter:>9} {tau:>6g} {floor:>6g} {n_pass:>4}/{len(cell)}"
            f" {med_err:>9.3f} {med_resid:>7.1f}"
        )
        if args.per_run:
            for r in sorted(cell, key=lambda r: r["seed"]):
                print(
                    f"    seed {r['seed']:>3}  pass={int(r['passed'])}"
                    f"  max|err|={r['max_abs_error']:.3f}"
                    f"  3sig={int(r['covered_3sigma'])}"
                    f"  maxstd={r['max_posterior_std']:.3f}"
                    f"  resid/noise={r['residual_over_noise']:.1f}"
                )

    if not args.baseline:
        ranked.sort(key=lambda item: item[0])
        _, (jitter, tau, floor), n_pass = ranked[0]
        seed_kind = "holdout" if args.holdout else "tuning"
        print(
            f"\nwinner: tau={tau:g} floor={floor:g}"
            f" ({n_pass}/{len(seeds)} on {seed_kind} seeds)"
        )


if __name__ == "__main__":
    main()
