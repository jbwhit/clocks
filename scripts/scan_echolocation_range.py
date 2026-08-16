"""Resolution-vs-range sweep for the 3D echolocation scenario.

Runs the shared scenario over a range grid x 12 seeds and writes the
study JSON + summary figure. See
docs/superpowers/specs/2026-07-19-3d-echolocation-design.md section 3.

Usage:
    uv run scripts/scan_echolocation_range.py                  # development seeds 0-11
    uv run scripts/scan_echolocation_range.py --seed-block 400 # certification
    uv run scripts/scan_echolocation_range.py --figure-only    # re-render PNG
"""

import argparse
import math
import statistics
from itertools import product
from multiprocessing import Pool
from numbers import Integral
from pathlib import Path

from clocks._echo_study import (
    load_study,
    save_study,
    snr_table,
    summarize,
    write_summary_figure,
)
from clocks._scenarios import (
    ECHO_ESS_TARGET,
    ECHO_M_TRUE,
    ECHO_PROPOSAL_SCALE,
    ECHO_REJUVENATION_STEPS,
    ECHO_SWEEP_RANGES,
    EchoRunResult,
    build_head_lattice,
    run_echolocation_3d,
    validate_echo_geometry,
)

JSON_PATH = Path("output/echolocation_range_study.json")
PNG_PATH = Path("output/echolocation_range_study.png")
DEVELOPMENT_ESS_TARGETS = (0.7, 0.8, 0.9)
DEVELOPMENT_REJUVENATION_STEPS = (1, 2, 4)
DEVELOPMENT_PROPOSAL_SCALES = (1.5, 2.38, 3.0)


def _reject_duplicates(name: str, values: list[float] | list[int]) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate values")


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
        list(DEVELOPMENT_ESS_TARGETS) if seed_block == 0 else [ECHO_ESS_TARGET]
    )
    selected_steps = steps or (
        list(DEVELOPMENT_REJUVENATION_STEPS)
        if seed_block == 0
        else [ECHO_REJUVENATION_STEPS]
    )
    selected_scales = scales or (
        list(DEVELOPMENT_PROPOSAL_SCALES) if seed_block == 0 else [ECHO_PROPOSAL_SCALE]
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
    _reject_duplicates("ess-target controls", selected_ess)
    _reject_duplicates("rejuvenation-step controls", selected_steps)
    _reject_duplicates("proposal-scale controls", selected_scales)
    cells = list(product(selected_ess, selected_steps, selected_scales))
    return cells


def _ranges_for_block(seed_block: int, ranges: list[float] | None) -> list[float]:
    """Use canonical ranges for protected blocks and unique ranges in development."""
    if seed_block >= 400 and ranges is not None:
        raise ValueError(
            "explicit range overrides are forbidden for protected seed blocks"
        )
    selected = list(ECHO_SWEEP_RANGES) if ranges is None else list(ranges)
    _reject_duplicates("ranges", selected)
    return selected


def _run(job: tuple[int, float, float, int, float]) -> EchoRunResult:
    seed, range_r, ess_target, rejuvenation_steps, proposal_scale = job
    return run_echolocation_3d(
        seed,
        range_r,
        ess_target=ess_target,
        rejuvenation_steps=rejuvenation_steps,
        proposal_scale=proposal_scale,
    )


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
    parser.add_argument("--ranges", type=float, nargs="+")
    parser.add_argument(
        "--seed-block",
        type=int,
        default=0,
        help=(
            "first seed of the 12-seed block: 0 = development; certification "
            "blocks are unseen multiples of 100 from 400"
        ),
    )
    parser.add_argument("--ess-targets", type=float, nargs="+")
    parser.add_argument("--steps", type=int, nargs="+")
    parser.add_argument("--scales", type=float, nargs="+")
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
    try:
        ranges = _ranges_for_block(args.seed_block, args.ranges)
    except ValueError as error:
        parser.error(str(error))

    if args.figure_only:
        write_summary_figure(load_study(JSON_PATH), PNG_PATH)
        print(f"Figure written to {PNG_PATH}")
        return

    # Validate at the execution boundary (spec sections 3a and 5).
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    head = build_head_lattice()
    for range_r in ranges:
        validate_echo_geometry(range_r, ECHO_M_TRUE, head)
    if args.seed_block >= 400:
        print(
            f"CERTIFICATION RUN (seed block {args.seed_block}): "
            "run exactly once; results are final."
        )

    print("Noise-free centered signal vs range (SNR sanity gate):")
    _print_snr_table(ranges)
    if args.snr_only:
        return

    jobs = [
        (seed, range_r, ess_target, steps, scale)
        for ess_target, steps, scale in cells
        for range_r in ranges
        for seed in seeds
    ]
    with Pool(args.workers) as pool:
        results = pool.map(_run, jobs)

    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_study(JSON_PATH, args.seed_block, results)
    study = load_study(JSON_PATH)

    print(f"\nSweep on seed block {args.seed_block} ({len(jobs)} runs):")
    grouped: dict[tuple[float, int, float], list[dict]] = {}
    for result in study["results"]:
        key = (
            result["ess_target"],
            result["rejuvenation_steps"],
            result["proposal_scale"],
        )
        grouped.setdefault(key, []).append(result)
    expected_runs_per_cell = len(seeds) * len(ranges)
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
        if len(cells) == 1:
            _print_summary(cell)
    ranked.sort()
    _, winner = ranked[0]
    print(f"winner: ess={winner[0]:.2f}, steps={winner[1]}, scale={winner[2]:.2f}")
    if len(cells) == 1:
        write_summary_figure(study, PNG_PATH)
    if args.per_run:
        for r in sorted(
            study["results"],
            key=lambda r: (
                r["ess_target"],
                r["rejuvenation_steps"],
                r["proposal_scale"],
                r["range_r"],
                r["seed"],
            ),
        ):
            print(
                f"    ess={r['ess_target']:.2f} steps={r['rejuvenation_steps']} "
                f"scale={r['proposal_scale']:.2f} "
                f"range {r['range_r']:>4g} seed {r['seed']:>3}"
                f" pass={int(r['passed'])} pos_err={r['position_error']:.3f}"
                f" M_err={r['mass_error']:.4f} pos_std={r['pos_std']:.3f}"
                f" 3sig={int(r['covered_3sigma'])}"
                f" resid/noise={r['residual_over_noise']:.1f}"
                f" normalized_error={r['normalized_error']:.3f}"
                f" forward_evaluations={r['forward_model_evaluations']}"
            )
    written = f"{JSON_PATH} and {PNG_PATH}" if len(cells) == 1 else str(JSON_PATH)
    print(f"\nWrote {written}")


if __name__ == "__main__":
    main()
