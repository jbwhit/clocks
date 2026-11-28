"""The demo commands' writer: what it writes, and what it leaves alone."""

import os
import re
from pathlib import Path

import matplotlib
import pytest
from _clip_replays import small
from _figure_artists import png_size
from matplotlib.colors import to_hex
from PIL import Image

import clocks._demos._common as common
from clocks._clips import CLIP_HEIGHTS, final_frame
from clocks._demos._common import STAGED, write_outputs
from clocks._figures import density_figure
from clocks._replay import first_difference, load_replay
from clocks._style import palette
from clocks._video import EncoderError, StreamInfo

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
OLD = {"npz": b"old archive", "png": b"old png", "mp4": b"old clip"}


def _existing_set(directory: Path) -> None:
    for suffix, data in OLD.items():
        (directory / f"demo_1d.{suffix}").write_bytes(data)


def _contents(directory: Path) -> dict[str, bytes | None]:
    paths = {suffix: directory / f"demo_1d.{suffix}" for suffix in OLD}
    return {s: p.read_bytes() if p.exists() else None for s, p in paths.items()}


def _staged_names(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.name.startswith(STAGED))


def _corner(path: Path) -> str:
    with Image.open(path) as image:
        return to_hex([c / 255 for c in image.convert("RGB").getpixel((0, 0))])


def _clip_stand_in(
    log: list, data: bytes, error: BaseException | None = None, sent: int = 92
):
    """Stands in for ``render_clip``, with its defaults; reports ``sent`` frames."""

    def render(replay, path, *, theme="light", encoder=None):
        log.append((replay, path, theme, encoder))
        path.write_bytes(data)
        if error is not None:
            raise error
        return sent

    return render


def _probe_stand_in(log: list, frames: int = 92, error: Exception | None = None):
    """Stands in for ``probe``: reports ``frames`` decoded, or raises ``error``."""

    def probe(path, encoder=None):
        log.append((path, encoder))
        if error is not None:
            raise error
        return StreamInfo("h264", "yuv420p", 1500, 900, 20.0, frames, True)

    return probe


@pytest.mark.parametrize(
    ("demo", "figure", "height"),
    [
        pytest.param("demo_1d", final_frame, 2 * CLIP_HEIGHTS["demo_1d"], id="demo_1d"),
        pytest.param("demo_density", density_figure, 1080, id="demo_density"),
    ],
)
def test_write_outputs_writes_the_archive_and_png_into_a_new_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], demo, figure, height
) -> None:
    """One of the two tests that start from a missing directory (decision C15),
    and the only one below the commands."""
    replay = small(demo)
    directory = tmp_path / "new" / "out"

    written = write_outputs(replay, directory, figure)

    assert written == [directory / f"{demo}.npz", directory / f"{demo}.png"]
    assert sorted(p.name for p in directory.iterdir()) == [f"{demo}.npz", f"{demo}.png"]
    assert first_difference(replay, load_replay(written[0])) is None
    assert png_size(written[1]) == (1500, height)
    assert capsys.readouterr().out.splitlines() == [f"Saved: {p}" for p in written]


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_the_png_is_the_figure_drawn_for_the_theme(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, theme: str
) -> None:
    """Only this catches the figure handed the light palette while the theme's
    style is in force."""
    replay = small("demo_1d")
    drawn: list = []
    saved: list = []
    real_save = common.save_png

    def figure(given, colors):
        fig = final_frame(given, colors)
        face = to_hex(matplotlib.rcParams["figure.facecolor"])
        drawn.append((given, colors, face, fig))
        return fig

    def save_spy(fig, file) -> None:
        saved.append((fig, file.name))
        real_save(fig, file)

    monkeypatch.setattr(common, "save_png", save_spy)

    write_outputs(replay, tmp_path, figure, theme=theme)

    background = palette(theme).background
    [(given, colors, face, fig)] = drawn
    assert given is replay
    assert colors == palette(theme)
    assert face == background
    assert saved == [(fig, str(tmp_path / f"{STAGED}demo_1d.png"))]
    assert _corner(tmp_path / "demo_1d.png") == background


def test_the_clip_is_drawn_from_the_replay_in_the_theme(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Only this catches the clip drawn in the default theme whatever was asked,
    or a `--video` run that does not print one `Saved:` line per output, or
    calls the clip it has just replaced `Kept:`."""
    _existing_set(tmp_path)
    replay = small("demo_1d")
    log: list = []
    probed: list = []
    monkeypatch.setattr(common, "render_clip", _clip_stand_in(log, b"new clip"))
    monkeypatch.setattr(common, "probe", _probe_stand_in(probed))

    written = write_outputs(
        replay, tmp_path, final_frame, theme="dark", encoder="the-encoder"
    )

    [(given, path, theme, encoder)] = log
    assert given is replay
    assert path == tmp_path / f"{STAGED}demo_1d.mp4"
    assert (theme, encoder) == ("dark", "the-encoder")
    assert probed == [(tmp_path / f"{STAGED}demo_1d.mp4", "the-encoder")]
    assert written[2] == tmp_path / "demo_1d.mp4"
    assert written[2].read_bytes() == b"new clip"
    assert _staged_names(tmp_path) == []
    assert capsys.readouterr().out.splitlines() == [f"Saved: {p}" for p in written]


def test_a_failed_clip_leaves_every_existing_output_as_it_was(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches the PNG or the archive moved into place before the clip, the
    last output staged, is made (Review Focus 2). Other failure tests catch it
    too, each where its failure comes after that output is staged; the Task 7
    rows list which."""
    _existing_set(tmp_path)
    failing = _clip_stand_in([], b"partial", EncoderError("the encoder failed"))
    monkeypatch.setattr(common, "render_clip", failing)

    with pytest.raises(EncoderError, match="the encoder failed"):
        write_outputs(small("demo_1d"), tmp_path, final_frame, encoder="the-encoder")

    assert _contents(tmp_path) == OLD
    assert _staged_names(tmp_path) == []
    assert "Saved:" not in capsys.readouterr().out


def test_a_failed_png_leaves_every_existing_output_as_it_was(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches a failure other than the encoder's leaving staged files behind
    (here a PNG save that wrote part of its file), with the rename and the
    interruption tests below."""
    _existing_set(tmp_path)

    def failing_save(fig, file) -> None:
        file.write(b"partial")
        raise OSError("disk")

    monkeypatch.setattr(common, "save_png", failing_save)

    with pytest.raises(OSError, match="disk"):
        write_outputs(small("demo_1d"), tmp_path, final_frame)

    assert _contents(tmp_path) == OLD
    assert _staged_names(tmp_path) == []


@pytest.mark.parametrize(
    "case", ["earlier_clip", "dangling_clip_link", "clip_directory", "no_clip"]
)
def test_a_run_without_a_clip_keeps_an_earlier_one_and_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    """Only this catches an earlier clip, or a dangling link at its name,
    removed without being asked, kept without a word or relinked, or a
    directory at its name announced as a clip (Review Focus 3). A clip
    announced when there is none it catches with the density test and the
    new-directory test."""
    clip = tmp_path / "demo_1d.mp4"
    if case == "earlier_clip":
        _existing_set(tmp_path)
    elif case == "dangling_clip_link":
        try:
            os.symlink(tmp_path / "gone.mp4", clip)
        except OSError:
            pytest.skip("this system refuses symbolic links")
    elif case == "clip_directory":
        clip.mkdir()
        (clip / "kept").write_bytes(b"inside")

    written = write_outputs(small("demo_1d"), tmp_path, final_frame)

    out = capsys.readouterr().out.splitlines()
    kept = [line for line in out if line.startswith("Kept: ")]
    assert written == [tmp_path / "demo_1d.npz", tmp_path / "demo_1d.png"]
    if case in ("earlier_clip", "dangling_clip_link"):
        assert clip.is_symlink() == (case == "dangling_clip_link")
        if case == "earlier_clip":
            assert clip.read_bytes() == b"old clip"
        else:
            assert os.readlink(clip) == str(tmp_path / "gone.mp4")
        assert kept == [
            f"Kept: {clip} is from an earlier run and was not updated "
            "(--video writes a new one)"
        ]
    elif case == "clip_directory":
        assert (clip / "kept").read_bytes() == b"inside"
        assert kept == []
    else:
        assert not clip.exists()
        assert kept == []


def _encode_like_stand_in(data: bytes):
    """Stands in for ``render_clip`` as ``encode`` writes: a temporary file of
    its own, moved onto the staged name with ``os.replace``."""

    def render(replay, path, *, theme="light", encoder=None):
        temporary = path.with_name(f"{path.name}.stand-in.part")
        temporary.write_bytes(data)
        os.replace(temporary, path)
        return 92

    return render


@pytest.mark.parametrize(
    "case",
    [
        "final_png_link",
        "final_archive_link",
        "final_clip_link",
        "staged_png_link",
        "staged_png_file",
        "staged_clip_link",
        "staged_clip_file",
    ],
)
def test_no_output_is_written_through_a_link_or_onto_a_leftover(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str
) -> None:
    """Only this catches a staged file left by an interrupted run stopping the
    next one, or the clip written through a link at its name (Review Focus 6);
    the archive or PNG written through one it catches with the rename and
    link-to-a-directory tests. The clip cases run with an encoder, the staged
    clip written as ``encode`` writes it, by a replace onto its name, so a
    stale staged clip is replaced whether or not the run unlinks it first."""
    target = tmp_path / "elsewhere"
    target.write_bytes(b"target")
    name = {
        "final_png_link": "demo_1d.png",
        "final_archive_link": "demo_1d.npz",
        "final_clip_link": "demo_1d.mp4",
        "staged_png_link": f"{STAGED}demo_1d.png",
        "staged_png_file": f"{STAGED}demo_1d.png",
        "staged_clip_link": f"{STAGED}demo_1d.mp4",
        "staged_clip_file": f"{STAGED}demo_1d.mp4",
    }[case]
    if case.endswith("_file"):
        (tmp_path / name).write_bytes(b"left over")
    else:
        try:
            os.symlink(target, tmp_path / name)
        except OSError:
            pytest.skip("this system refuses symbolic links")
    encoder = "the-encoder" if "clip" in case else None
    monkeypatch.setattr(common, "render_clip", _encode_like_stand_in(b"new clip"))
    monkeypatch.setattr(common, "probe", _probe_stand_in([]))

    write_outputs(small("demo_1d"), tmp_path, final_frame, encoder=encoder)

    assert target.read_bytes() == b"target"
    assert not (tmp_path / "demo_1d.npz").is_symlink()
    assert not (tmp_path / "demo_1d.png").is_symlink()
    assert load_replay(tmp_path / "demo_1d.npz").meta["demo"] == "demo_1d"
    assert (tmp_path / "demo_1d.png").read_bytes()[:8] == PNG_SIGNATURE
    if encoder is not None:
        assert not (tmp_path / "demo_1d.mp4").is_symlink()
        assert (tmp_path / "demo_1d.mp4").read_bytes() == b"new clip"
    assert _staged_names(tmp_path) == []


@pytest.mark.parametrize("suffix", ["npz", "png", "mp4"])
def test_a_directory_at_a_staged_name_stops_the_run_before_staging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    suffix: str,
) -> None:
    """Catches the run not unlinking an output's staged name (the archive's,
    the PNG's or the clip's; alone for the archive's and the clip's, and with
    the leftover test for the PNG's, per the Task 7 rows), or unlinking one
    only after an
    earlier output has been staged (the PNG's after the archive, the clip's
    after the PNG): a directory there must stop the run before anything is
    staged or drawn (each of those steps is a spy that fails the test). Only a
    directory shows it for the archive and the clip: a stale file or link at
    their staged names is replaced anyway, by the ``os.replace`` that
    ``save_replay`` and ``encode`` end with. The PNG's staged file is created
    with ``open("xb")``, which a stale file or link would make fail; that is
    the leftover test's case, and the directory here is this one's."""
    _existing_set(tmp_path)
    staged_dir = tmp_path / f"{STAGED}demo_1d.{suffix}"
    staged_dir.mkdir()
    (staged_dir / "kept").write_bytes(b"inside")

    def never(what: str):
        def step(*args, **kwargs):
            pytest.fail(f"{what} before the staged names were cleared")

        return step

    monkeypatch.setattr(common, "save_replay", never("the archive was staged"))
    monkeypatch.setattr(common, "_stage_png", never("the PNG was staged"))
    monkeypatch.setattr(common, "render_clip", never("the clip was drawn"))

    with pytest.raises(OSError):
        write_outputs(small("demo_1d"), tmp_path, final_frame, encoder="the-encoder")

    assert _contents(tmp_path) == OLD
    assert (staged_dir / "kept").read_bytes() == b"inside"
    assert _staged_names(tmp_path) == [staged_dir.name]
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("case", ["short", "over", "undecodable", "no_decoder"])
def test_a_clip_that_does_not_decode_to_every_frame_is_not_published(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str
) -> None:
    """Only this catches a decode error or a decoder that cannot be started
    reaching the command as other than an `EncoderError` naming the clip. A
    clip published when it decodes to more frames than were sent it catches
    with the decode-count test's `[mismatch]`, and one published with no
    decode check at all with that case and the themed-clip test (Review
    Focus 4)."""
    _existing_set(tmp_path)
    monkeypatch.setattr(common, "render_clip", _clip_stand_in([], b"short clip"))
    probe = {
        "short": _probe_stand_in([], frames=91),
        "over": _probe_stand_in([], frames=93),
        "undecodable": _probe_stand_in([], error=ValueError("the decoder failed")),
        "no_decoder": _probe_stand_in([], error=FileNotFoundError("no such file")),
    }[case]
    monkeypatch.setattr(common, "probe", probe)
    expected = {
        "short": "to 91 frames, but 92 were sent",
        "over": "to 93 frames, but 92 were sent",
        "undecodable": re.escape(f"{STAGED}demo_1d.mp4 does not decode"),
        "no_decoder": re.escape(
            f"{STAGED}demo_1d.mp4 could not be checked: no such file"
        ),
    }[case]

    with pytest.raises(EncoderError, match=expected):
        write_outputs(small("demo_1d"), tmp_path, final_frame, encoder="the-encoder")

    assert _contents(tmp_path) == OLD
    assert _staged_names(tmp_path) == []


@pytest.mark.parametrize(
    ("decoded", "published"),
    [pytest.param(74, True, id="match"), pytest.param(92, False, id="mismatch")],
)
def test_the_decode_check_counts_against_the_frames_this_clip_sent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, decoded: int, published: bool
) -> None:
    """Catches the decode check comparing against a fixed count (the 92 frames
    every other stand-in case sends) instead of what ``render_clip`` reported:
    a one-observation clip sends 74. Of the tests that reach the decode check,
    only this and the real-encoder run into a misreadable directory send
    another count."""
    _existing_set(tmp_path)
    monkeypatch.setattr(common, "render_clip", _clip_stand_in([], b"new clip", sent=74))
    monkeypatch.setattr(common, "probe", _probe_stand_in([], frames=decoded))

    if published:
        write_outputs(small("demo_1d"), tmp_path, final_frame, encoder="the-encoder")
        assert (tmp_path / "demo_1d.mp4").read_bytes() == b"new clip"
    else:
        with pytest.raises(EncoderError, match="to 92 frames, but 74 were sent"):
            write_outputs(
                small("demo_1d"), tmp_path, final_frame, encoder="the-encoder"
            )
        assert _contents(tmp_path) == OLD
    assert _staged_names(tmp_path) == []


@pytest.mark.parametrize("suffix", ["npz", "png", "mp4"])
def test_a_directory_at_an_output_name_stops_the_run_before_anything_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    suffix: str,
) -> None:
    """Only this catches a directory collision, at the archive's, the PNG's or
    the clip's name, found by a rename part-way through publication, after
    the archive had been replaced, or found only once something has been
    staged (the archive's and PNG's staging and the clip's drawing are spies
    that fail the test)."""
    _existing_set(tmp_path)
    collision = tmp_path / f"demo_1d.{suffix}"
    collision.unlink()
    collision.mkdir()
    (collision / "kept").write_bytes(b"inside")

    def never(what: str):
        def step(*args, **kwargs):
            pytest.fail(f"{what} before the preflight")

        return step

    monkeypatch.setattr(common, "save_replay", never("the archive was staged"))
    monkeypatch.setattr(common, "_stage_png", never("the PNG was staged"))
    monkeypatch.setattr(common, "render_clip", never("the clip was drawn"))

    with pytest.raises(IsADirectoryError, match="is a directory"):
        write_outputs(small("demo_1d"), tmp_path, final_frame, encoder="the-encoder")

    others = {s: data for s, data in OLD.items() if s != suffix}
    assert {s: (tmp_path / f"demo_1d.{s}").read_bytes() for s in others} == others
    assert (collision / "kept").read_bytes() == b"inside"
    assert _staged_names(tmp_path) == []
    assert capsys.readouterr().out == ""


def test_a_failed_rename_removes_the_staged_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Only this catches a rename failure leaving staged files behind; it pins
    decision C6's stated limit: the archive renamed before it stays."""
    _existing_set(tmp_path)
    real_replace = os.replace

    def replace(source, destination) -> None:
        if Path(destination).name == "demo_1d.png":
            raise OSError("rename")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", replace)

    with pytest.raises(OSError, match="rename"):
        write_outputs(small("demo_1d"), tmp_path, final_frame)

    assert load_replay(tmp_path / "demo_1d.npz").meta["demo"] == "demo_1d"
    assert (tmp_path / "demo_1d.png").read_bytes() == OLD["png"]
    assert (tmp_path / "demo_1d.mp4").read_bytes() == OLD["mp4"]
    assert _staged_names(tmp_path) == []
    assert "Saved:" not in capsys.readouterr().out


def test_a_closed_output_stream_does_not_stop_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches a closed standard output stopping publication part-way (a
    `Saved:` line printed between renames), with the rename test."""
    _existing_set(tmp_path)

    def closed(*args, **kwargs) -> None:
        raise BrokenPipeError("standard output is closed")

    monkeypatch.setattr(common, "print", closed, raising=False)

    with pytest.raises(BrokenPipeError):
        write_outputs(small("demo_1d"), tmp_path, final_frame)

    assert load_replay(tmp_path / "demo_1d.npz").meta["demo"] == "demo_1d"
    assert (tmp_path / "demo_1d.png").read_bytes()[:8] == PNG_SIGNATURE
    assert (tmp_path / "demo_1d.mp4").read_bytes() == OLD["mp4"]
    assert _staged_names(tmp_path) == []


def test_an_interrupted_run_leaves_every_existing_output_as_it_was(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only this catches staged files left behind by an interruption that is
    not an `Exception` (here Ctrl-C while the clip is being written)."""
    _existing_set(tmp_path)
    interrupted = _clip_stand_in([], b"partial", KeyboardInterrupt())
    monkeypatch.setattr(common, "render_clip", interrupted)

    with pytest.raises(KeyboardInterrupt):
        write_outputs(small("demo_1d"), tmp_path, final_frame, encoder="the-encoder")

    assert _contents(tmp_path) == OLD
    assert _staged_names(tmp_path) == []


def test_a_failed_cleanup_keeps_the_error_and_removes_the_rest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches a staged file that cannot be removed replacing the run's own
    error, or stopping the removal of the staged files after it (here the
    PNG, staged second of three); the Task 7 rows name the tests that share
    each edit."""
    _existing_set(tmp_path)
    staged_png = f"{STAGED}demo_1d.png"

    def failing(replay, path, *, theme="light", encoder=None):
        path.write_bytes(b"partial")
        png = path.with_name(staged_png)
        png.unlink()
        png.mkdir()  # removing a directory with unlink() raises OSError
        raise EncoderError("the encoder failed")

    monkeypatch.setattr(common, "render_clip", failing)

    with pytest.raises(EncoderError, match="the encoder failed") as raised:
        write_outputs(small("demo_1d"), tmp_path, final_frame, encoder="the-encoder")

    notes = getattr(raised.value, "__notes__", [])
    assert len(notes) == 1
    assert staged_png in notes[0] and "left behind" in notes[0]
    assert _staged_names(tmp_path) == [staged_png]
    assert _contents(tmp_path) == OLD


def test_a_cleanup_note_that_cannot_be_added_keeps_the_error_and_removes_the_rest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only this catches a removal's note replacing the run's own error when
    that error's ``__notes__`` is not a list, or stopping the removal of the
    staged files after it (here the PNG, staged second of three)."""
    _existing_set(tmp_path)
    staged_png = f"{STAGED}demo_1d.png"
    error = EncoderError("the encoder failed")
    error.__notes__ = ()

    def failing(replay, path, *, theme="light", encoder=None):
        path.write_bytes(b"partial")
        png = path.with_name(staged_png)
        png.unlink()
        png.mkdir()  # removing a directory with unlink() raises OSError
        raise error

    monkeypatch.setattr(common, "render_clip", failing)

    with pytest.raises(EncoderError) as raised:
        write_outputs(small("demo_1d"), tmp_path, final_frame, encoder="the-encoder")

    assert raised.value is error
    assert error.__notes__ == ()
    assert _staged_names(tmp_path) == [staged_png]
    assert _contents(tmp_path) == OLD


def test_an_interrupted_removal_keeps_the_error_and_removes_the_rest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only this catches an interrupt while a staged file is removed replacing
    the run's own error, or stopping the removal of the staged files after it
    (here the PNG's, second of three); the interrupt becomes the note."""
    _existing_set(tmp_path)
    error = EncoderError("the encoder failed")
    failing = _clip_stand_in([], b"partial", error)
    real_unlink = Path.unlink
    calls: list[str] = []

    def interrupted(self: Path, missing_ok: bool = False) -> None:
        if self.name.startswith(STAGED):
            calls.append(self.name)
            if self.name == f"{STAGED}demo_1d.png" and len(calls) > 3:
                raise KeyboardInterrupt("during the removal")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(common, "render_clip", failing)
    monkeypatch.setattr(Path, "unlink", interrupted)
    # BaseException: an escaped interrupt would end the session.
    with pytest.raises(BaseException) as raised:
        write_outputs(small("demo_1d"), tmp_path, final_frame, encoder="the-encoder")
    monkeypatch.undo()

    assert raised.value is error
    staged_png = tmp_path / f"{STAGED}demo_1d.png"
    assert error.__notes__ == [
        f"{staged_png} may have been left behind: KeyboardInterrupt: during the removal"
    ]
    assert _staged_names(tmp_path) == [staged_png.name]
    assert _contents(tmp_path) == OLD


def test_a_link_to_a_directory_at_an_output_name_is_replaced(tmp_path: Path) -> None:
    """Only this catches the directory preflight following a link, which would
    refuse a link the rename can replace (decision C6)."""
    target = tmp_path / "a_directory"
    target.mkdir()
    (target / "kept").write_bytes(b"inside")
    try:
        os.symlink(target, tmp_path / "demo_1d.png", target_is_directory=True)
    except OSError:
        pytest.skip("this system refuses symbolic links")

    write_outputs(small("demo_1d"), tmp_path, final_frame)

    assert not (tmp_path / "demo_1d.png").is_symlink()
    assert (tmp_path / "demo_1d.png").read_bytes()[:8] == PNG_SIGNATURE
    assert (target / "kept").read_bytes() == b"inside"


def test_the_density_demo_says_nothing_of_a_clip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Only this catches the `Kept:` line, which names `--video`, printed for
    the density demo, which has no clip and no `--video`."""
    (tmp_path / "demo_density.mp4").write_bytes(b"not ours")

    write_outputs(small("demo_density"), tmp_path, density_figure)

    assert capsys.readouterr().out.splitlines() == [
        f"Saved: {tmp_path / 'demo_density.npz'}",
        f"Saved: {tmp_path / 'demo_density.png'}",
    ]
    assert (tmp_path / "demo_density.mp4").read_bytes() == b"not ours"
