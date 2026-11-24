"""Support contract for the raw Gaussian-density particle filter demo."""

import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from _figure_artists import png_size
from matplotlib.colors import to_hex

from clocks._demos import demo_density
from clocks._demos.demo_density import build_density_filter, main
from clocks._replay import Replay, load_replay
from clocks._style import palette
from clocks._support import density_support_mask
from clocks.physics import clock_rates_density_gaussian
from clocks.types import ClockArray, Observation

ROOT = Path(__file__).resolve().parents[1]
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def test_density_demo_builder_samples_only_its_physical_prior() -> None:
    clocks = ClockArray(
        np.array([[-6.0], [-3.0], [0.0], [3.0], [6.0]]), track_offset=1.0
    )
    particle_filter = build_density_filter(
        clocks, np.random.default_rng(9), n_particles=300
    )

    assert np.all(
        np.isfinite(particle_filter.log_prior_density(particle_filter.state.particles))
    )
    assert np.all(
        density_support_mask(
            particle_filter.state.particles,
            clock_array=clocks,
            mu_range=(-8.0, 8.0),
            sigma_range=(0.1, 5.0),
            amplitude_range=(0.001, 0.030),
        )
    )
    amplitudes = particle_filter.state.particles[:, 2]
    assert np.all(amplitudes >= 0.001)
    assert np.all(amplitudes <= 0.03)


def test_density_demo_never_forwards_an_invalid_candidate() -> None:
    clocks = ClockArray(
        np.array([[-6.0], [-3.0], [0.0], [3.0], [6.0]]), track_offset=1.0
    )
    particle_filter = build_density_filter(
        clocks, np.random.default_rng(9), n_particles=300
    )
    forward_model_batch = particle_filter.forward_model_batch
    assert forward_model_batch is not None

    def checked_forward(particles: np.ndarray) -> np.ndarray:
        valid = density_support_mask(
            particles,
            clock_array=clocks,
            mu_range=(-8.0, 8.0),
            sigma_range=(0.1, 5.0),
            amplitude_range=(0.001, 0.030),
        )
        assert np.all(valid)
        return forward_model_batch(particles)

    particle_filter.forward_model_batch = checked_forward
    rates = clock_rates_density_gaussian(np.array([1.5, 2.0, 0.010]), clocks)
    candidates = np.array([[1.5, 2.0, 0.010], [0.0, 1.0, 1.0]])
    log_likelihood = particle_filter._observation_log_likelihood(
        candidates, Observation(rates, time=0.0)
    )
    assert np.isfinite(log_likelihood[0])
    assert np.isneginf(log_likelihood[1])


def test_extreme_finite_density_candidate_rejects_without_warning() -> None:
    clocks = ClockArray(np.array([[0.0]]), track_offset=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        valid = density_support_mask(
            np.array([[0.0, 1.0, 1e308]]),
            clock_array=clocks,
            mu_range=(-1.0, 1.0),
            sigma_range=(0.1, 2.0),
            amplitude_range=(0.0, 1e308),
        )

    np.testing.assert_array_equal(valid, np.array([False]))


def _estimate_lines(
    replay_arrays: dict,
    row: int,
    n_particles: int,
    truth: tuple[float, float, float] = (1.5, 2.0, 0.01),
) -> list[str]:
    mean = replay_arrays["mean"][row]
    std = np.sqrt(np.diagonal(replay_arrays["cov"][row]))
    ess = replay_arrays["ess"][row]
    return [
        f"  mu    = {mean[0]:.3f} ± {std[0]:.3f}  (true: {truth[0]})",
        f"  sigma = {mean[1]:.3f} ± {std[1]:.3f}  (true: {truth[1]})",
        f"  A     = {mean[2]:.3f} ± {std[2]:.3f}  (true: {truth[2]})",
        f"  ESS   = {ess:.0f} / {n_particles}",
    ]


def test_density_command_prints_and_draws_the_recorded_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "new" / "density.png"

    assert (
        main(["--output", str(output), "--observations", "3", "--particles", "64"]) == 0
    )

    expected = demo_density.record(n_observations=3, n_particles=64)
    lines = capsys.readouterr().out.splitlines()
    start = lines.index("Final estimate:")
    assert lines[start + 1 : start + 5] == _estimate_lines(expected.arrays, -1, 64)
    assert png_size(output) == (1500, 1080)


def test_density_command_prints_progress_from_the_replay(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = load_replay(ROOT / "assets/replays/demo_density.npz")
    edited = archive.arrays["true_rates"].copy()
    edited[0] = 0.5
    edited_truth = np.array([1.25, 2.5, 0.02])
    stub_replay = Replay(
        archive.meta,
        {**archive.arrays, "true_rates": edited, "true_params": edited_truth},
    )
    calls: list[dict] = []

    def stub_record(**keywords: int) -> Replay:
        calls.append(keywords)
        return stub_replay

    monkeypatch.setattr(demo_density, "record", stub_record)

    main(
        [
            "--output",
            str(tmp_path / "d.png"),
            "--observations",
            "40",
            "--particles",
            "16",
        ]
    )

    assert calls == [{"n_observations": 40, "n_particles": 16}]
    out = capsys.readouterr().out
    assert out.startswith("True density: mu=1.25, sigma=2.5, amplitude=0.02")
    assert f"True rates: {edited}" in out
    lines = out.splitlines()
    n_particles = archive.meta["n_particles"]
    for k in (20, 40, 60):
        at = lines.index(f"After {k} observations:")
        assert lines[at + 1 : at + 5] == _estimate_lines(
            archive.arrays, k, n_particles, tuple(edited_truth)
        )
    at = lines.index("Final estimate:")
    assert lines[at + 1 : at + 5] == _estimate_lines(
        archive.arrays, -1, n_particles, tuple(edited_truth)
    )


def test_density_command_saves_the_figure_it_drew(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[tuple] = []
    saved: list[tuple] = []
    real_figure, real_save = demo_density.density_figure, demo_density.save_png

    def figure_spy(replay: Replay, colors: object) -> object:
        face = to_hex(matplotlib.rcParams["figure.facecolor"])
        fig = real_figure(replay, colors)
        built.append((replay, colors, fig, face))
        return fig

    def save_spy(fig: object, path: Path) -> None:
        saved.append((fig, path))
        real_save(fig, path)

    monkeypatch.setattr(demo_density, "density_figure", figure_spy)
    monkeypatch.setattr(demo_density, "save_png", save_spy)

    main(
        [
            "--output",
            str(tmp_path / "d.png"),
            "--observations",
            "2",
            "--particles",
            "16",
        ]
    )

    assert len(built) == 1
    replay, colors, fig, face = built[0]
    assert replay.meta["n_observations"] == 2
    assert replay.meta["n_particles"] == 16
    assert colors == palette("light")
    assert face == palette("light").background
    assert len(saved) == 1
    assert saved[0][0] is fig
    assert saved[0][1] == tmp_path / "d.png"


@pytest.mark.parametrize(
    "target", ["out/d", "out/d.jpg"], ids=["no_suffix", "misleading_suffix"]
)
def test_density_command_writes_png_bytes_at_the_path_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    monkeypatch.chdir(tmp_path)

    main(["--output", target, "--observations", "1", "--particles", "16"])

    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert files == [tmp_path / target]
    assert (tmp_path / target).read_bytes()[:8] == PNG_SIGNATURE


def test_density_command_writes_only_its_png(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    main(["--output", "out/d.png", "--observations", "1", "--particles", "16"])

    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert files == [tmp_path / "out/d.png"]
