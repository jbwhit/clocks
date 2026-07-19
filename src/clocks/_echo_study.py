"""Reporting helpers for the echolocation resolution-vs-range study.

Lives in the package (not scripts/) so tests and the Quarto page can
import it; the scan script stays a thin CLI (same reasoning as
clocks._scenarios).
"""

import json
import statistics
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from clocks._scenarios import (
    ECHO_M_TRUE,
    ECHO_NOISE_STD,
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


def save_study(path: Path, seed_block: int, results: list[EchoRunResult]) -> None:
    """Write sweep results to JSON (numpy arrays become lists)."""
    serializable = [
        {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in result.items()
        }
        for result in results
    ]
    path.write_text(
        json.dumps({"seed_block": seed_block, "results": serializable}, indent=2)
    )


def load_study(path: Path) -> dict:
    return json.loads(path.read_text())


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
