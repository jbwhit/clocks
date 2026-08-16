"""Reporting helpers for the echolocation resolution-vs-range study.

Lives in the package (not scripts/) so tests and the Quarto page can
import it; the scan script stays a thin CLI (same reasoning as
clocks._scenarios).
"""

import statistics
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from clocks._calibration import (
    build_study_document,
    control_grid_from_cells,
    validate_echo_results,
    write_study,
)
from clocks._calibration import (
    load_study as load_study,
)
from clocks._scenarios import (
    ECHO_FAR_STD_FACTOR,
    ECHO_M_TRUE,
    ECHO_NOISE_STD,
    ECHO_PASS_MASS_TOL,
    ECHO_PASS_POS_TOL,
    EchoRunResult,
    build_head_lattice,
    echo_mass_config,
)
from clocks.physics import clock_rates


def snr_table(
    ranges: Sequence[float],
    *,
    m_true: float = ECHO_M_TRUE,
    noise_std: float = ECHO_NOISE_STD,
) -> list[dict]:
    """Noise-free centered signal magnitude per range (spec section 3).

    Pure forward-model evaluation — the same computation the site page's
    falloff cell shows.
    """
    head = build_head_lattice()
    table = []
    for range_r in ranges:
        rates = clock_rates(echo_mass_config(range_r, m_true), head)
        signal = float(np.max(np.abs(rates - rates.mean())))
        table.append(
            {
                "range_r": float(range_r),
                "signal": signal,
                "signal_over_noise": signal / noise_std,
            }
        )
    return table


def save_study(
    path: Path,
    *,
    seed_block: int,
    seeds: Sequence[int],
    control_cells: Sequence[tuple[float, int, float]],
    ranges: Sequence[float],
    results: Sequence[EchoRunResult],
) -> None:
    """Write exact echo scan evidence in the shared deterministic schema."""
    expected_tuples = {
        (ess_target, steps, scale, float(range_r), int(seed))
        for ess_target, steps, scale in control_cells
        for range_r in ranges
        for seed in seeds
    }
    validate_echo_results(results, expected_tuples=expected_tuples)
    study = build_study_document(
        study="echolocation_range",
        seed_block=seed_block,
        seeds=seeds,
        control_grid=control_grid_from_cells(control_cells),
        tolerances={
            "position_error_max": ECHO_PASS_POS_TOL,
            "mass_error_max": ECHO_PASS_MASS_TOL,
            "far_std_factor_min": ECHO_FAR_STD_FACTOR,
        },
        ranges=ranges,
        results=results,
    )
    write_study(path, study)


def summarize(results: list[dict]) -> list[dict]:
    """Per-range medians and pass counts, sorted by range."""
    by_range: dict[float, list[dict]] = {}
    for result in results:
        by_range.setdefault(result["range_r"], []).append(result)
    summary = []
    for range_r in sorted(by_range):
        cell = by_range[range_r]
        summary.append(
            {
                "range_r": range_r,
                "n_pass": sum(r["passed"] for r in cell),
                "n_runs": len(cell),
                "med_position_error": statistics.median(
                    r["position_error"] for r in cell
                ),
                "med_mass_error": statistics.median(r["mass_error"] for r in cell),
                "med_pos_std": statistics.median(r["pos_std"] for r in cell),
                "med_mass_std": statistics.median(r["mass_std"] for r in cell),
            }
        )
    return summary


def write_summary_figure(study: dict, png_path: Path) -> None:
    """Two aligned subplots: position and mass error vs range, with the
    filter's own claimed uncertainty (posterior std) as dashed medians."""
    results = study["results"]
    summary = summarize(results)
    ranges = [row["range_r"] for row in summary]

    fig, (ax_pos, ax_mass) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True, constrained_layout=True
    )
    for result in results:
        ax_pos.plot(
            result["range_r"],
            result["position_error"],
            "o",
            color="steelblue",
            alpha=0.35,
            markersize=4,
        )
        ax_mass.plot(
            result["range_r"],
            result["mass_error"],
            "o",
            color="steelblue",
            alpha=0.35,
            markersize=4,
        )
    ax_pos.plot(
        ranges,
        [r["med_position_error"] for r in summary],
        "-o",
        color="tab:blue",
        label="median error",
    )
    ax_pos.plot(
        ranges,
        [r["med_pos_std"] for r in summary],
        "--s",
        color="tab:orange",
        label="median posterior std",
    )
    ax_mass.plot(
        ranges,
        [r["med_mass_error"] for r in summary],
        "-o",
        color="tab:blue",
        label="median error",
    )
    ax_mass.plot(
        ranges,
        [r["med_mass_std"] for r in summary],
        "--s",
        color="tab:orange",
        label="median posterior std",
    )
    ax_pos.set_yscale("log")
    ax_mass.set_yscale("log")
    ax_pos.set_ylabel("position error")
    ax_mass.set_ylabel("mass error")
    ax_mass.set_xlabel("range (circumradii)")
    ax_pos.legend(fontsize=8)
    ax_mass.legend(fontsize=8)
    ax_pos.set_title("Echolocation resolution vs range")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
