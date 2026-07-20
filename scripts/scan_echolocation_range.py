"""Resolution-vs-range sweep for the 3D echolocation scenario.

Runs the shared scenario over a range grid x 12 seeds and writes the
study JSON + summary figure. See
docs/superpowers/specs/2026-07-19-3d-echolocation-design.md section 3.

Usage:
    uv run scripts/scan_echolocation_range.py                  # tuning seeds 0-11
    uv run scripts/scan_echolocation_range.py --seed-block 300 # certification (spec 3a)
    uv run scripts/scan_echolocation_range.py --figure-only    # re-render PNG
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

from clocks._echo_study import (
    load_study,
    save_study,
    snr_table,
    summarize,
    write_summary_figure,
)
from clocks._scenarios import (
    ECHO_M_TRUE,
    ECHO_SWEEP_RANGES,
    EchoRunResult,
    build_head_lattice,
    run_echolocation_3d,
    validate_echo_geometry,
)

JSON_PATH = Path("output/echolocation_range_study.json")
PNG_PATH = Path("output/echolocation_range_study.png")


def _run(job: tuple[int, float]) -> EchoRunResult:
    seed, range_r = job
    return run_echolocation_3d(seed, range_r)


def _print_snr_table(ranges: list[float]) -> None:
    print(f"{'range':>7} {'signal':>10} {'signal/noise':>13}")
    for row in snr_table(ranges):
        print(
            f"{row['range_r']:>7g} {row['signal']:>10.2e}"
            f" {row['signal_over_noise']:>13.2f}"
        )


def _print_summary(results: list[dict]) -> None:
    header = (
        f"{'range':>7} {'pass':>6} {'med pos err':>12} {'med M err':>10}"
        f" {'med pos std':>12} {'med M std':>10}"
    )
    print(header)
    for row in summarize(results):
        print(
            f"{row['range_r']:>7g} {row['n_pass']:>4}/{row['n_runs']}"
            f" {row['med_position_error']:>12.3f} {row['med_mass_error']:>10.4f}"
            f" {row['med_pos_std']:>12.3f} {row['med_mass_std']:>10.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ranges", type=float, nargs="+", default=list(ECHO_SWEEP_RANGES)
    )
    parser.add_argument(
        "--seed-block",
        type=int,
        default=0,
        help=(
            "first seed of the 12-seed block: 0 = tuning; certification "
            "blocks are 300, 400, ... (spec section 3a; the Status history "
            "records which block certified)"
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--per-run", action="store_true")
    parser.add_argument(
        "--figure-only",
        action="store_true",
        help="re-render the PNG from the existing JSON without sweeping",
    )
    parser.add_argument(
        "--snr-only",
        action="store_true",
        help="print the SNR sanity table and exit without sweeping",
    )
    args = parser.parse_args()

    if args.figure_only:
        write_summary_figure(load_study(JSON_PATH), PNG_PATH)
        print(f"Figure written to {PNG_PATH}")
        return

    # Validate at the boundary (spec sections 3a and 5).
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.seed_block != 0 and (args.seed_block < 300 or args.seed_block % 100 != 0):
        parser.error(
            "--seed-block must be 0 (tuning) or a certification block "
            "(300, 400, ...); see spec section 3a"
        )
    head = build_head_lattice()
    for range_r in args.ranges:
        validate_echo_geometry(range_r, ECHO_M_TRUE, head)
    if args.seed_block >= 300:
        print(
            f"CERTIFICATION RUN (seed block {args.seed_block}): "
            "run exactly once; results are final."
        )

    print("Noise-free centered signal vs range (SNR sanity gate):")
    _print_snr_table(args.ranges)
    if args.snr_only:
        return

    seeds = range(args.seed_block, args.seed_block + 12)
    jobs = [(seed, range_r) for range_r in args.ranges for seed in seeds]
    with Pool(args.workers) as pool:
        results = pool.map(_run, jobs)

    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_study(JSON_PATH, args.seed_block, results)
    study = load_study(JSON_PATH)
    write_summary_figure(study, PNG_PATH)

    print(f"\nSweep on seed block {args.seed_block} ({len(jobs)} runs):")
    _print_summary(study["results"])
    if args.per_run:
        for r in sorted(study["results"], key=lambda r: (r["range_r"], r["seed"])):
            print(
                f"    range {r['range_r']:>4g} seed {r['seed']:>3}"
                f" pass={int(r['passed'])} pos_err={r['position_error']:.3f}"
                f" M_err={r['mass_error']:.4f} pos_std={r['pos_std']:.3f}"
                f" 3sig={int(r['covered_3sigma'])}"
                f" resid/noise={r['residual_over_noise']:.1f}"
            )
    print(f"\nWrote {JSON_PATH} and {PNG_PATH}")


if __name__ == "__main__":
    main()
