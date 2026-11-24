"""Reporting helpers for the echolocation resolution-vs-range study.

Lives in the package (not scripts/) so tests and the Quarto page can
import it; the scan script stays a thin CLI (same reasoning as
clocks._scenarios).
"""

import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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
from clocks._style import Palette, save_png, style
from clocks.physics import clock_rates


def snr_table(
    ranges: Sequence[float],
    *,
    m_true: float = ECHO_M_TRUE,
    noise_std: float = ECHO_NOISE_STD,
) -> list[dict]:
    """Noise-free centered signal magnitude per range (spec section 3).

    Pure forward-model evaluation. The signal is the largest absolute
    mean-centred clock rate. The site page's falloff cell plots the largest
    absolute orthonormal contrast (``Q @ rates``) instead, so its numbers
    differ from this table's.
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


def range_study_figure(study: Mapping[str, Any], colors: Palette) -> Figure:
    """Position and mass error against range, with each run and the pass limits.

    Each panel shows every run's error, the median error and the median
    posterior standard deviation at each range, and the study's pass limit; the
    top panel also counts the runs within both limits.

    Draws with the rcParams in force: call it inside ``with style(theme) as
    colors:``. It never shows, saves or closes the figure it returns.
    """
    results = study["results"]
    tolerances = study["tolerances"]
    summary = summarize(results)
    ranges = [row["range_r"] for row in summary]
    fig, (ax_pos, ax_mass) = plt.subplots(2, 1, figsize=(7.5, 5.4), sharex=True)
    panels = (
        (ax_pos, "position-error", "position error", "position_error", "pos_std"),
        (ax_mass, "mass-error", "mass error", "mass_error", "mass_std"),
    )
    for ax, gid, label, key, spread_key in panels:
        ax.set_gid(gid)
        tolerance = tolerances[f"{key}_max"]
        ax.plot(
            [r["range_r"] for r in results],
            [r[key] for r in results],
            color=colors.ink,
            marker="o",
            markersize=4,
            linestyle="none",
            alpha=0.35,
            gid="runs",
        )
        ax.plot(
            ranges,
            [row[f"med_{key}"] for row in summary],
            color=colors.ink,
            marker="o",
            linestyle="-",
            gid="median-error",
        )
        ax.plot(
            ranges,
            [row[f"med_{spread_key}"] for row in summary],
            color=colors.belief,
            marker="s",
            linestyle="-",
            gid="median-spread",
        )
        ax.axhline(
            tolerance,
            color=colors.threshold,
            linestyle="--",
            gid="limit",
        )
        ax.set_yscale("log")
        ax.set_ylabel(label)
        ax.annotate(
            f"limit {tolerance}",
            xy=(0.0, tolerance),
            xycoords=("axes fraction", "data"),
            xytext=(4, 3),
            textcoords="offset points",
            color=colors.threshold,
            ha="left",
            va="bottom",
        )
    ax_mass.set_xlabel("range (circumradii)")
    _label_lines(ax_pos, colors, summary)
    _label_pass_counts(ax_pos, colors, results, tolerances, ranges)
    return fig


def _label_lines(ax: Axes, colors: Palette, summary: list[dict]) -> None:
    """Direct labels at the middle range: the spread up and left of its line,
    the error down and right of its own, where nothing else is drawn."""
    # The middle range: on the committed study the two lines are furthest apart
    # there, so the labels sit clear of both.
    first = summary[max(0, len(summary) // 2 - 1)]
    ax.annotate(
        "median posterior s.d.",
        xy=(first["range_r"], first["med_pos_std"]),
        xytext=(-8, 10),
        textcoords="offset points",
        color=colors.belief,
        ha="right",
        va="bottom",
    )
    ax.annotate(
        "median error",
        xy=(first["range_r"], first["med_position_error"]),
        xytext=(10, -12),
        textcoords="offset points",
        color=colors.ink,
        ha="left",
        va="top",
    )


def _label_pass_counts(
    ax: Axes,
    colors: Palette,
    results: list[dict],
    tolerances: Mapping[str, Any],
    ranges: list[float],
) -> None:
    """Counts of runs within both limits, in a row just above the axes."""
    for range_r in ranges:
        count = sum(
            r["position_error"] <= tolerances["position_error_max"]
            and r["mass_error"] <= tolerances["mass_error_max"]
            for r in results
            if r["range_r"] == range_r
        )
        ax.text(
            range_r,
            1.02,
            str(count),
            transform=ax.get_xaxis_transform(),
            color=colors.muted,
            ha="center",
            va="bottom",
            gid="pass-count",
        )
    ax.text(
        0.0,
        1.13,
        "runs within both limits",
        transform=ax.transAxes,
        color=colors.muted,
        ha="left",
        va="bottom",
        gid="pass-count-label",
    )


def write_summary_figure(study: dict, png_path: Path) -> None:
    """Write the range-study figure, light theme, as the site shows it."""
    fig = None
    try:
        with style("light") as colors:
            fig = range_study_figure(study, colors)
            save_png(fig, png_path)
    finally:
        if fig is not None:
            plt.close(fig)
