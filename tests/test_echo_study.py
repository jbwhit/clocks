"""Tests for the echolocation study helpers (pure, no inference runs)."""

import math
from pathlib import Path

import matplotlib
import numpy as np
import pytest

from clocks._echo_study import (
    load_study,
    save_study,
    snr_table,
    summarize,
    write_summary_figure,
)
from clocks._scenarios import (
    ECHO_M_TRUE,
    ECHO_NOISE_STD,
    EchoRunResult,
    echo_mass_position,
)

matplotlib.use("Agg")


def _fake_result(seed: int, range_r: float, err: float) -> EchoRunResult:
    mean = np.append(echo_mass_position(range_r), ECHO_M_TRUE)
    mean[0] += err
    std = np.array([max(err, 0.1), 0.2, 0.4, 0.03])
    return EchoRunResult(
        seed=seed,
        range_r=range_r,
        passed=err <= 1.0,
        mean=mean,
        std=std,
        position_error=err,
        mass_error=0.0,
        pos_std=float(np.linalg.norm(std[:3])),
        mass_std=0.03,
        covered_3sigma=True,
        residual_over_noise=1.2,
        normalized_error=err / 2.0,
        forward_model_evaluations=123,
        ess_target=0.9,
        rejuvenation_steps=1,
        proposal_scale=1.5,
    )


def test_snr_table_decreases_with_range() -> None:
    table = snr_table([2.0, 4.0, 8.0])
    assert [row["range_r"] for row in table] == [2.0, 4.0, 8.0]
    signals = [row["signal"] for row in table]
    assert signals[0] > signals[1] > signals[2] > 0.0
    assert np.isclose(
        table[0]["signal_over_noise"], table[0]["signal"] / ECHO_NOISE_STD
    )


def test_save_load_round_trip(tmp_path: Path) -> None:
    results = [_fake_result(s, 2.0, 0.3) for s in range(3)]
    path = tmp_path / "study.json"
    save_study(
        path,
        seed_block=0,
        seeds=range(3),
        control_cells=[(0.9, 1, 1.5)],
        ranges=[2.0],
        results=results,
    )
    study = load_study(path)
    assert study["schema_version"] == 1
    assert study["study"] == "echolocation_range"
    assert study["seed_block"] == 0
    assert study["seed_role"] == "development"
    assert study["seeds"] == [0, 1, 2]
    assert study["control_grid"] == {
        "ess_target": [0.9],
        "proposal_scale": [1.5],
        "rejuvenation_steps": [1],
    }
    assert study["ranges"] == [2.0]
    assert study["tolerances"] == {
        "far_std_factor_min": 20.0,
        "mass_error_max": 0.04,
        "position_error_max": 1.0,
    }
    assert len(study["results"]) == 3
    loaded = study["results"][0]
    assert loaded["position_error"] == 0.3
    assert loaded["mean"] == pytest.approx(
        [*echo_mass_position(2.0) + np.array([0.3, 0.0, 0.0]), ECHO_M_TRUE]
    )


def test_save_study_validation_preserves_existing_evidence(tmp_path: Path) -> None:
    result = _fake_result(0, 2.0, 0.3)
    result["position_error"] = math.inf
    path = tmp_path / "study.json"
    path.write_bytes(b"existing evidence\n")

    with pytest.raises(ValueError, match="position_error.*finite"):
        save_study(
            path,
            seed_block=0,
            seeds=[0],
            control_cells=[(0.9, 1, 1.5)],
            ranges=[2.0],
            results=[result],
        )

    assert path.read_bytes() == b"existing evidence\n"


def test_summarize_medians_per_range(tmp_path: Path) -> None:
    results = [_fake_result(s, 2.0, e) for s, e in enumerate([0.1, 0.3, 0.5])]
    results += [_fake_result(s, 8.0, e) for s, e in enumerate([2.0, 4.0, 6.0])]
    path = tmp_path / "study.json"
    save_study(
        path,
        seed_block=0,
        seeds=range(3),
        control_cells=[(0.9, 1, 1.5)],
        ranges=[2.0, 8.0],
        results=results,
    )
    summary = summarize(load_study(path)["results"])
    assert [row["range_r"] for row in summary] == [2.0, 8.0]
    assert summary[0]["med_position_error"] == 0.3
    assert summary[0]["n_pass"] == 3
    assert summary[1]["med_position_error"] == 4.0
    assert summary[1]["n_pass"] == 0


def test_write_summary_figure_creates_png(tmp_path: Path) -> None:
    results = [_fake_result(s, r, 0.2 * r) for r in (2.0, 4.0) for s in range(3)]
    json_path = tmp_path / "study.json"
    save_study(
        json_path,
        seed_block=0,
        seeds=range(3),
        control_cells=[(0.9, 1, 1.5)],
        ranges=[2.0, 4.0],
        results=results,
    )
    png_path = tmp_path / "study.png"
    write_summary_figure(load_study(json_path), png_path)
    assert png_path.exists()
    assert png_path.stat().st_size > 0
