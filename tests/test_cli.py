"""Contracts for the packaged demo commands."""

import ast
import importlib
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
from _clip_replays import committed, edited, small
from _figure_artists import png_size
from matplotlib.colors import to_hex
from PIL import Image

import clocks._demos._common as common
import clocks._video
from clocks._clips import CLIP_HEIGHTS
from clocks._figures import trace_decimals
from clocks._replay import DEMOS, load_replay
from clocks._style import palette
from clocks._video import EncoderError, probe

ROOT = Path(__file__).resolve().parents[1]
CLIP_DEMOS = tuple(name for name in DEMOS if name != "demo_density")
FILTER_CLIP_DEMOS = tuple(n for n in CLIP_DEMOS if n != "demo_model_comparison")
HEIGHTS = {**{n: 2 * CLIP_HEIGHTS[n] for n in CLIP_DEMOS}, "demo_density": 1080}
FAST = ["--observations", "1", "--particles", "16"]


@pytest.fixture(autouse=True)
def _in_tmp_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Every test runs in its own empty directory: no run writes into the checkout."""
    monkeypatch.chdir(tmp_path)


def _module(name: str):
    return importlib.import_module(f"clocks._demos.{name}")


def _never(**_: int):
    pytest.fail("the demo was recorded")


def _corner(path: Path) -> str:
    with Image.open(path) as image:
        return to_hex([c / 255 for c in image.convert("RGB").getpixel((0, 0))])


@pytest.mark.parametrize("name", DEMOS)
def test_demo_parser_defaults(name: str) -> None:
    """Only this catches a default count that is not the packaged one; a default
    directory other than `output` it catches with
    `test_the_default_run_writes_into_output`."""
    module = _module(name)
    args = module.build_parser().parse_args([])
    packaged = module.scenario()
    assert args.output_dir == Path("output")
    assert args.theme == "light"
    assert (args.observations, args.particles) == (
        packaged.n_observations,
        packaged.n_particles,
    )
    if name in CLIP_DEMOS:
        assert args.video is False


@pytest.mark.parametrize("name", DEMOS)
def test_output_is_no_longer_an_option(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str
) -> None:
    """Only this catches `--output` still accepted as an alias of `--output-dir`
    (Review Focus 1)."""
    module = _module(name)
    monkeypatch.setattr(module, "record", _never)
    with pytest.raises(SystemExit) as raised:
        module.main(["--output", "x.png", *FAST])
    assert raised.value.code == 2
    assert list(tmp_path.iterdir()) == []


def test_options_cannot_be_abbreviated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only this catches a parser that refuses `--output` by name and still
    takes other prefixes: with the test above it shows the cause."""
    module = _module("demo_1d")
    monkeypatch.setattr(module, "record", _never)
    with pytest.raises(SystemExit) as raised:
        module.main(["--obs", "1", "--particles", "16"])
    assert raised.value.code == 2


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("name", DEMOS)
def test_demo_main_writes_its_archive_and_png(
    tmp_path: Path, name: str, theme: str
) -> None:
    """Catches a command that writes elsewhere than `--output-dir` or hands the
    writer another theme. The Task 7 rows show it alone for `demo_2d`, both
    ways; other tests read some demos' outputs too (`demo_1d`'s in the encoder
    and odd-path tests below, `demo_density`'s in its own test). A run recorded
    with other counts than it was given it catches with
    `test_command_prints_the_recorded_run`, and the PNG drawn outside the
    theme's style with `test_the_png_is_the_figure_drawn_for_the_theme`."""
    given = tmp_path / "given"
    given.mkdir()
    assert _module(name).main(["--output-dir", "given", *FAST, "--theme", theme]) == 0
    assert [p.name for p in tmp_path.iterdir()] == ["given"]
    assert sorted(p.name for p in given.iterdir()) == [f"{name}.npz", f"{name}.png"]
    meta = load_replay(given / f"{name}.npz").meta
    assert (meta["demo"], meta["n_observations"], meta["n_particles"]) == (name, 1, 16)
    assert png_size(given / f"{name}.png") == (1500, HEIGHTS[name])
    assert _corner(given / f"{name}.png") == palette(theme).background


def test_the_default_run_writes_into_output(tmp_path: Path) -> None:
    """One of the two tests that start from a missing directory (decision C15).
    A default output directory other than `output/` it catches with
    `test_demo_parser_defaults` (the Task 7 row for that default)."""
    assert _module("demo_1d").main(FAST) == 0
    assert [p.name for p in tmp_path.iterdir()] == ["output"]
    names = sorted(p.name for p in (tmp_path / "output").iterdir())
    assert names == ["demo_1d.npz", "demo_1d.png"]


@pytest.mark.parametrize("name", CLIP_DEMOS)
def test_png_is_the_final_frame(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Only this catches a clip demo's PNG drawn by another figure function, or
    from another replay than the one archived."""
    module = _module(name)
    archived: list = []
    drawn: list = []
    saved: list = []
    real_archive, real_frame = common.save_replay, module.final_frame
    real_save = common.save_png

    def archive_spy(path, replay) -> None:
        archived.append(replay)
        real_archive(path, replay)

    def frame_spy(replay, colors):
        fig = real_frame(replay, colors)
        drawn.append((replay, fig))
        return fig

    def save_spy(fig, file) -> None:
        saved.append(fig)
        real_save(fig, file)

    monkeypatch.setattr(common, "save_replay", archive_spy)
    monkeypatch.setattr(module, "final_frame", frame_spy)
    monkeypatch.setattr(common, "save_png", save_spy)

    module.main(["--output-dir", ".", *FAST])

    [replay] = archived
    [(drawn_replay, fig)] = drawn
    assert drawn_replay is replay
    assert saved == [fig]


@pytest.mark.parametrize("name", CLIP_DEMOS)
def test_video_writes_the_clip(tmp_path: Path, name: str) -> None:
    """Catches a demo whose `--video` writes no clip because it does not pass
    the encoder on. The Task 7 rows show it alone for `demo_multi_mass`;
    `demo_1d` is also run with `--video` by the encoder and odd-path tests
    below, and the encoder looked up too late they catch as well. It needs
    the encoder and fails, not skips, without one (pull request 3's D11)."""
    args = ["--output-dir", ".", "--video", "--observations", "2", "--particles", "16"]
    assert _module(name).main(args) == 0
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == sorted(f"{name}.{suffix}" for suffix in ("mp4", "npz", "png"))
    info = probe(tmp_path / f"{name}.mp4")
    assert (info.codec, info.pix_fmt) == ("h264", "yuv420p")
    assert (info.frames, info.width, info.height) == (92, 1500, HEIGHTS[name])


@pytest.mark.parametrize("directory", ["-captures", "file:captures"])
def test_video_writes_into_a_directory_ffmpeg_could_misread(
    tmp_path: Path, directory: str
) -> None:
    """A relative output directory that FFmpeg would read as an option (`-…`)
    or a protocol (`file:…`) is passed to it as an absolute path, for the
    encode and for the decode check alike. It needs the encoder. The directory
    exists beforehand: only two tests start from a missing one (decision C15)."""
    out = tmp_path / directory
    out.mkdir()
    args = [f"--output-dir={directory}", "--video", *FAST]
    assert _module("demo_1d").main(args) == 0
    names = sorted(p.name for p in out.iterdir())
    assert names == ["demo_1d.mp4", "demo_1d.npz", "demo_1d.png"]
    assert [p.name for p in tmp_path.iterdir()] == [directory]
    assert probe((out / "demo_1d.mp4").absolute()).frames == 74


def test_density_has_no_video_option(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only this catches `--video` offered by the density demo, which has no clip."""
    module = _module("demo_density")
    monkeypatch.setattr(module, "record", _never)
    with pytest.raises(SystemExit) as raised:
        module.main(["--video", *FAST])
    assert raised.value.code == 2
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", CLIP_DEMOS)
def test_video_without_an_encoder_stops_before_the_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    name: str,
) -> None:
    """Only this catches a missing encoder reaching the user as a traceback
    rather than the usage error and the library's instructions (decision C5)."""
    module = _module(name)
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    monkeypatch.setattr(clocks._video.shutil, "which", lambda _: None)
    monkeypatch.setattr(module, "record", _never)
    with pytest.raises(SystemExit) as raised:
        module.main(["--output-dir", "out", "--video", *FAST])
    assert raised.value.code == 2
    err = capsys.readouterr().err
    assert "imageio-ffmpeg" in err
    assert "ffmpeg on PATH" in err
    assert list(tmp_path.iterdir()) == []


def test_a_failed_encode_ends_with_a_message_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Only this catches an encoder failure's message without the notes that
    say what was left behind; the failure reaching the user as a traceback it
    catches with the test below."""

    def failing(replay, path, *, theme="light", encoder=None):
        error = EncoderError("the encoder failed")
        error.add_note("a-file was left behind: busy")
        raise error

    monkeypatch.setattr(common, "find_encoder", lambda: "the-encoder")
    monkeypatch.setattr(common, "render_clip", failing)
    with pytest.raises(SystemExit) as raised:
        _module("demo_1d").main(["--output-dir", ".", "--video", *FAST])
    assert raised.value.code == 1
    lines = capsys.readouterr().err.splitlines()
    [error] = [i for i, line in enumerate(lines) if ": error: " in line]
    assert lines[error].endswith(": error: the encoder failed")
    assert lines[error + 1].endswith(": note: a-file was left behind: busy")


def test_a_failed_encode_says_that_nothing_was_saved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Only this catches a failed `--video` run that does not say its archive
    and PNG were discarded with the clip, and how to get them."""
    old = {"npz": b"old archive", "png": b"old png", "mp4": b"old clip"}
    for suffix, data in old.items():
        (tmp_path / f"demo_1d.{suffix}").write_bytes(data)

    def failing(replay, path, *, theme="light", encoder=None):
        path.write_bytes(b"partial")
        raise EncoderError("the encoder failed")

    monkeypatch.setattr(common, "find_encoder", lambda: "the-encoder")
    monkeypatch.setattr(common, "render_clip", failing)
    with pytest.raises(SystemExit) as raised:
        _module("demo_1d").main(["--output-dir", ".", "--video", *FAST])
    assert raised.value.code == 1
    lines = capsys.readouterr().err.splitlines()
    assert lines[-2].endswith(": error: the encoder failed")
    assert lines[-1].endswith(
        ": note: nothing was saved: this run's archive and PNG were discarded "
        "with its clip; a run without --video writes them"
    )
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == sorted(f"demo_1d.{suffix}" for suffix in old)
    assert {s: (tmp_path / f"demo_1d.{s}").read_bytes() for s in old} == old


def _plain(name: str) -> str:
    return name.replace("₁", "1").replace("₂", "2")


@pytest.mark.parametrize("name", FILTER_CLIP_DEMOS)
def test_command_prints_the_recorded_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], name: str
) -> None:
    """Only this catches numbers printed from the module's constants or the
    command's arguments rather than the replay, or with the headers' decimals."""
    base = small(name)
    width = float(np.diff(base.meta["prior_ranges"][0])[0])
    rates = base.arrays["true_rates"].copy()
    rates[0] = 0.5
    truth = base.arrays["true_params"].copy()
    truth[0] += 0.125 * width
    mean = base.arrays["mean"].copy()
    mean[-1, 0] += 0.25 * width
    stub = edited(base, true_rates=rates, true_params=truth, mean=mean)
    calls: list[dict] = []

    def record(**keywords: int):
        calls.append(keywords)
        return stub

    module = _module(name)
    monkeypatch.setattr(module, "record", record)

    module.main(["--output-dir", ".", "--observations", "7", "--particles", "23"])

    assert calls == [{"n_observations": 7, "n_particles": 23}]
    out = capsys.readouterr().out
    a, meta = stub.arrays, stub.meta
    names = [_plain(n) for n in meta["param_names"]]
    pairs = ", ".join(f"{n}={v:g}" for n, v in zip(names, a["true_params"]))
    # NumPy wraps a long array over several lines (10 rates for the two-mass
    # 2D demo, 27 for echolocation), so the prefix is compared before splitting.
    assert out.startswith(f"Truth: {pairs}\nTrue rates: {a['true_rates']}\n")
    lines = out.splitlines()
    at = lines.index(f"Final estimate after {meta['n_observations']} observations:")
    sd = np.sqrt(np.diagonal(a["cov"][-1]))
    expected = []
    for j, (n, prior_range) in enumerate(zip(names, meta["prior_ranges"])):
        d = trace_decimals(prior_range) + 1
        expected.append(
            f"  {n} = {a['mean'][-1, j]:.{d}f} ± {sd[j]:.{d}f}"
            f"  (true: {a['true_params'][j]:g})"
        )
    expected.append(f"  ESS = {a['ess'][-1]:.0f} / {meta['n_particles']}")
    assert lines[at + 1 : at + 1 + len(expected)] == expected


def test_model_comparison_command_prints_from_the_replay_and_records_once(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Only this catches the comparison recorded a second time for its printout,
    or its truth printed from the module's constants rather than the replay."""
    base = committed("demo_model_comparison")
    rates = base.arrays["true_rates"].copy()
    rates[0] = 0.5
    truth = base.arrays["true_params"].copy()
    truth[0] += 0.5
    stub = edited(base, true_rates=rates, true_params=truth)
    calls: list[dict] = []

    def record(**keywords: int):
        calls.append(keywords)
        return stub

    module = _module("demo_model_comparison")
    monkeypatch.setattr(module, "record", record)

    module.main(["--output-dir", ".", *FAST])

    assert len(calls) == 1
    out = capsys.readouterr().out
    a, meta = stub.arrays, stub.meta
    names = [_plain(n) for n in meta["param_names"]]
    pairs = ", ".join(f"{n}={v:g}" for n, v in zip(names, a["true_params"]))
    assert out.startswith(
        f"Truth: {pairs}\nTrue rates: {a['true_rates']}\n"
        f"True model: K={meta['true_k']}\n"
    )
    ks = [int(k) for k in a["model_ks"]]
    expected = [f"True model: K={meta['true_k']}"]
    for k in range(20, meta["n_observations"] + 1, 20):
        expected.append(f"After {k} observations:")
        posterior = a["model_posterior"][k]
        expected += [f"  K={K}: posterior={p:.4f}" for K, p in zip(ks, posterior)]
    best = int(np.argmax(a["model_posterior"][-1]))
    K = ks[best]
    expected += [
        f"MAP model: K={K}",
        f"Estimate: {a[f'final_mean_k{K}']}",
        f"Std:      {a[f'final_std_k{K}']}",
        f"ESS:      {a['model_ess'][-1, best]:.0f} / {meta['n_particles']}",
    ]
    assert "\n" + "\n".join(expected) + "\n" in out  # whole lines, wrapped or not


_OLD_DRAWING = {"clocks.viz", "clocks._animate", "clocks._panels", "clocks._panels3d"}
_DEMOS_PACKAGE = ("clocks", "_demos")


def _absolute(node: ast.ImportFrom) -> str:
    """The module a `from` import in `clocks._demos` names, relative or not."""
    if not node.level:
        return node.module or ""
    base = _DEMOS_PACKAGE[: len(_DEMOS_PACKAGE) - node.level + 1]
    return ".".join([*base, *([node.module] if node.module else [])])


def test_no_demo_module_imports_the_facade_or_the_old_drawing_code() -> None:
    """Only this catches a demo importing `clocks.viz` or the GIF code, which
    Task 5 of the commands plan deletes (decision C17)."""
    found: dict[str, list[str]] = {}
    for path in sorted((ROOT / "src" / "clocks" / "_demos").glob("*.py")):
        names: set[str] = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = _absolute(node)
                names.add(module)
                names.update(f"{module}.{alias.name}" for alias in node.names)
        if names & _OLD_DRAWING:
            found[path.name] = sorted(names & _OLD_DRAWING)
    assert found == {}


@pytest.mark.parametrize("name", DEMOS)
def test_demo_help_exits_without_running(
    name: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module(name)

    def fail_if_run(*args, **kwargs):
        raise AssertionError("--help must exit before starting the demo")

    monkeypatch.setattr(module, "_run_demo", fail_if_run)
    with pytest.raises(SystemExit) as exc:
        module.main(["--help"])

    assert exc.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_project_scripts_point_directly_to_packaged_demo_modules() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    expected = {name.replace("_", "-"): f"clocks._demos.{name}:main" for name in DEMOS}
    assert pyproject["project"]["scripts"] == expected


@pytest.mark.parametrize("name", DEMOS)
def test_demo_selects_agg_before_importing_pyplot(name: str) -> None:
    check = """
import importlib.abc
import sys

class PyplotGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != "matplotlib.pyplot":
            return None
        matplotlib = sys.modules.get("matplotlib")
        if matplotlib is None or str(matplotlib.get_backend()).lower() != "agg":
            raise RuntimeError("matplotlib.pyplot imported before selecting Agg")
        sys.meta_path.remove(self)
        return None

sys.meta_path.insert(0, PyplotGuard())
__import__(sys.argv[1])
"""
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "svg"
    result = subprocess.run(
        [sys.executable, "-c", check, f"clocks._demos.{name}"],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", DEMOS)
def test_source_demo_is_only_a_thin_wrapper(name: str) -> None:
    source = (ROOT / "scripts" / f"{name}.py").read_text()
    assert source == (
        f"from clocks._demos.{name} import main\n\n"
        'if __name__ == "__main__":\n'
        "    main()\n"
    )
