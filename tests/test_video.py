import ast
import io
import os
import signal
import struct
import subprocess
import sys
import threading
import time
import tomllib
import types
from collections.abc import Iterator
from pathlib import Path
from typing import IO

import numpy as np
import pytest

import clocks._video
from clocks._video import (
    EncoderError,
    EncoderNotFound,
    StreamInfo,
    _feed,
    _run_encoder,
    box_order,
    dwell_frames,
    encode,
    encoder_version,
    find_encoder,
    parse_progress,
    parse_stream_line,
    probe,
)

ROOT = Path(__file__).resolve().parents[1]

WIDTH, HEIGHT = 64, 48
FRAME_BYTES = WIDTH * HEIGHT * 4
COLOURS = ((200, 40, 30), (30, 160, 60), (20, 50, 210))
COUNTS = (2, 1, 4)

DESIGN_ARGUMENTS = [
    "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
    "-f", "rawvideo", "-pix_fmt", "rgba", "-s", "64x48", "-r", "20",
    "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
    "-crf", "20", "-preset", "slow", "-movflags", "+faststart", "-f", "mp4",
]  # fmt: skip

Q2_LINE = (
    "  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), "
    "yuv420p(progressive), 1500x900, 12 kb/s, 20 fps, 20 tbr, 10240 tbn (default)"
)


def _solid(rgb: tuple[int, int, int]) -> bytes:
    return bytes([*rgb, 255]) * (WIDTH * HEIGHT)


def _small_clip() -> list[tuple[bytes, int]]:
    return [(_solid(rgb), count) for rgb, count in zip(COLOURS, COUNTS, strict=True)]


def _names(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


class _StandInEncoder:
    """Replaces ``_run_encoder``: keeps the command, takes every frame."""

    def __init__(self, result: tuple[int, str] = (0, ""), *, write_frames=False):
        self.result = result
        self.write_frames = write_frames
        self.calls = 0
        self.command: list[str] = []
        self.frames: list[bytes] = []

    def __call__(self, command, frames) -> tuple[int, str]:
        self.calls += 1
        self.command = list(command)
        with open(self.command[-1], "wb") as output:
            if not self.write_frames:
                output.write(b"partial")
            for frame in frames:
                self.frames.append(frame)
                if self.write_frames:
                    output.write(frame)
        return self.result


# ------------------------------------------------------------------ dwell


_DWELL_TOTALS = {50: 296, 80: 356}


@pytest.mark.parametrize(
    ("n_observations", "expected"),
    [
        pytest.param(
            50, [24] + [18] * 3 + [10] * 5 + [5] * 12 + [2] * 29 + [50], id="50"
        ),
        pytest.param(
            80, [24] + [18] * 3 + [10] * 5 + [5] * 12 + [2] * 59 + [50], id="80"
        ),
        pytest.param(1, [24, 50], id="1"),
        pytest.param(2, [24, 18, 50], id="2"),
        pytest.param(4, [24, 18, 18, 18, 50], id="4"),
        pytest.param(9, [24, 18, 18, 18, 10, 10, 10, 10, 10, 50], id="9"),
        pytest.param(21, [24] + [18] * 3 + [10] * 5 + [5] * 12 + [50], id="21"),
    ],
)
def test_dwell_frames(n_observations: int, expected: list[int]) -> None:
    schedule = dwell_frames(n_observations)
    assert schedule == expected
    if n_observations in _DWELL_TOTALS:
        assert sum(schedule) == _DWELL_TOTALS[n_observations]


def test_dwell_frames_needs_an_observation() -> None:
    with pytest.raises(ValueError, match="at least one observation"):
        dwell_frames(0)


# ------------------------------------------------------------ the encoder


def _package(get_ffmpeg_exe) -> types.ModuleType:
    module = types.ModuleType("imageio_ffmpeg")
    module.get_ffmpeg_exe = get_ffmpeg_exe
    return module


def _no_binary() -> str:
    raise RuntimeError("no ffmpeg exe could be found")


@pytest.mark.parametrize(
    ("module", "on_path", "expected"),
    [
        pytest.param(
            _package(lambda: "from-package"),
            "from-path",
            "from-package",
            id="package_first",
        ),
        pytest.param(None, "from-path", "from-path", id="path_second"),
        pytest.param(
            _package(_no_binary),
            "from-path",
            "from-path",
            id="package_without_a_binary",
        ),
        pytest.param(None, None, None, id="neither"),
    ],
)
def test_find_encoder_order(
    monkeypatch: pytest.MonkeyPatch, module, on_path, expected
) -> None:
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", module)
    monkeypatch.setattr(clocks._video.shutil, "which", lambda name: on_path)
    if expected is None:
        with pytest.raises(EncoderNotFound) as raised:
            find_encoder()
        assert "imageio-ffmpeg" in str(raised.value)
        assert "ffmpeg on PATH" in str(raised.value)
    else:
        assert find_encoder() == expected


def _module_level_imports(tree: ast.Module) -> Iterator[str]:
    def visit(node: ast.AST) -> Iterator[str]:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            return
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            yield node.module or ""
        for child in ast.iter_child_nodes(node):
            yield from visit(child)

    yield from visit(tree)


def test_video_module_imports_only_the_standard_library() -> None:
    """The one module of the package it imports, ``clocks._notes``, imports
    only the standard library at module level, so a wheel without the encoder
    still imports it."""
    source = (ROOT / "src" / "clocks" / "_video.py").read_text()
    names = list(_module_level_imports(ast.parse(source)))
    assert names, "no module-level import was found"
    outside = [n for n in names if n.split(".")[0] not in sys.stdlib_module_names]
    assert outside == ["clocks._notes"]
    notes = (ROOT / "src" / "clocks" / "_notes.py").read_text()
    imported = list(_module_level_imports(ast.parse(notes)))
    assert [n for n in imported if n.split(".")[0] not in sys.stdlib_module_names] == []


def test_the_encoder_command_is_the_designs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stand_in = _StandInEncoder(write_frames=True)
    monkeypatch.setattr(clocks._video, "_run_encoder", stand_in)

    written = encode(
        _small_clip(), tmp_path / "clip.mp4", width=64, height=48, encoder="the-encoder"
    )

    assert written == 7
    command = stand_in.command
    assert command[0] == "the-encoder"
    assert command[1:-1] == DESIGN_ARGUMENTS
    temporary = Path(command[-1])
    assert temporary.parent == tmp_path
    assert temporary.name.startswith("clip.mp4.")
    assert temporary.name.endswith(".part")
    first, second, third = (_solid(rgb) for rgb in COLOURS)
    assert stand_in.frames == [first] * 2 + [second] + [third] * 4
    assert _names(tmp_path) == ["clip.mp4"]
    assert (tmp_path / "clip.mp4").read_bytes() == b"".join(stand_in.frames)
    assert len((tmp_path / "clip.mp4").read_bytes()) == 7 * 12_288


def test_encode_then_probe(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"

    written = encode(_small_clip(), path, width=64, height=48)

    assert written == 7
    assert _names(tmp_path) == ["clip.mp4"]
    assert probe(path) == StreamInfo("h264", "yuv420p", 64, 48, 20.0, 7, True)


@pytest.mark.parametrize(
    ("width", "height"),
    [pytest.param(63, 48, id="width"), pytest.param(64, 47, id="height")],
)
def test_encode_refuses_odd_sizes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, width: int, height: int
) -> None:
    stand_in = _StandInEncoder()
    monkeypatch.setattr(clocks._video, "_run_encoder", stand_in)

    with pytest.raises(ValueError, match="even"):
        encode(
            _small_clip(),
            tmp_path / "clip.mp4",
            width=width,
            height=height,
            encoder="the-encoder",
        )

    assert stand_in.calls == 0
    assert _names(tmp_path) == []


def test_encode_refuses_a_frame_of_the_wrong_length(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log: list[object] = []
    pairs = _small_clip()
    pairs[1] = (pairs[1][0][:-1], pairs[1][1])

    def asked() -> Iterator[tuple[bytes, int]]:
        for n, pair in enumerate(pairs, start=1):
            log.append(("asked", n))
            yield pair

    def stand_in(command, frames) -> tuple[int, str]:
        Path(command[-1]).write_bytes(b"partial")
        for _ in frames:
            log.append("took")
        return 0, ""

    monkeypatch.setattr(clocks._video, "_run_encoder", stand_in)

    with pytest.raises(ValueError, match="frame 2 has 12287 bytes, expected 12288"):
        encode(asked(), tmp_path / "clip.mp4", width=64, height=48, encoder="e")

    assert _names(tmp_path) == []
    assert log == [("asked", 1), "took", "took", ("asked", 2)]


def test_a_failed_encode_leaves_the_old_clip_and_nothing_else(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "clip.mp4").write_bytes(b"old")
    monkeypatch.setattr(clocks._video, "_run_encoder", _StandInEncoder((3, "boom")))

    with pytest.raises(EncoderError) as raised:
        encode(_small_clip(), tmp_path / "clip.mp4", width=64, height=48, encoder="e")

    message = str(raised.value)
    assert "boom" in message
    assert "status 3" in message
    assert "clip.mp4" in message
    assert _names(tmp_path) == ["clip.mp4"]
    assert (tmp_path / "clip.mp4").read_bytes() == b"old"


def test_run_encoder_reports_a_failing_encoder_without_raising(
    tmp_path: Path,
) -> None:
    output = tmp_path / "missing" / "clip.mp4"
    command = [find_encoder(), *DESIGN_ARGUMENTS, str(output)]
    frames = [_solid(COLOURS[0])] * 40

    status, text = _run_encoder(command, frames)

    assert status != 0
    assert text.strip() != ""


@pytest.mark.parametrize("kind", ["missing", "not_executable"])
def test_an_encoder_that_cannot_start_is_an_encoder_error(
    tmp_path: Path, kind: str
) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    encoder = tools / "ffmpeg"
    if kind == "not_executable":
        encoder.write_text("not a program\n")
        encoder.chmod(0o644)
    clips = tmp_path / "clips"
    clips.mkdir()

    with pytest.raises(EncoderError) as raised:
        encode(
            _small_clip(), clips / "clip.mp4", width=64, height=48, encoder=str(encoder)
        )

    assert str(encoder) in str(raised.value)
    assert isinstance(raised.value.__cause__, OSError)
    assert _names(clips) == []


def test_an_error_while_drawing_leaves_no_clip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def drawing() -> Iterator[tuple[bytes, int]]:
        yield _solid(COLOURS[0]), 1
        raise ValueError("drawing failed")

    monkeypatch.setattr(clocks._video, "_run_encoder", _StandInEncoder())

    with pytest.raises(ValueError, match="drawing failed"):
        encode(drawing(), tmp_path / "clip.mp4", width=64, height=48, encoder="e")

    assert _names(tmp_path) == []


class _Pipe:
    def __init__(self, calls: list[str], break_write: bool):
        self.calls = calls
        self.break_write = break_write
        self.writes = 0

    def write(self, data: bytes) -> int:
        self.calls.append("write")
        self.writes += 1
        if self.break_write and self.writes == 2:
            raise BrokenPipeError("the encoder has exited")
        return len(data)

    def close(self) -> None:
        self.calls.append("close")


def _process_class(calls: list[str], *, status: int, break_write: bool):
    class Process:
        def __init__(self, *args, **kwargs) -> None:
            self.stdin = _Pipe(calls, break_write)

        def wait(self, timeout: float | None = None) -> int:
            calls.append("wait")
            return status

        def terminate(self) -> None:
            calls.append("terminate")

        def kill(self) -> None:
            calls.append("kill")

        def poll(self) -> int:
            return status

    return Process


_DRAWING_FAILED = ValueError("drawing failed")
_CALLERS_OWN = BrokenPipeError("the caller's own")


def _frames_then(error: BaseException | None) -> Iterator[bytes]:
    if error is None:
        yield from [b"a", b"b", b"c"]
        return
    yield b"a"
    raise error


@pytest.mark.parametrize(
    ("break_write", "error"),
    [
        pytest.param(False, None, id="success"),
        pytest.param(True, None, id="write_breaks"),
        pytest.param(False, _DRAWING_FAILED, id="producer_raises"),
        pytest.param(False, _CALLERS_OWN, id="producer_raises_broken_pipe"),
    ],
)
def test_run_encoder_reaps_the_process(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    break_write: bool,
    error: BaseException | None,
) -> None:
    """The input's close cannot break the pipe: it is unbuffered, so the
    close flushes nothing (a broken pipe surfaces on a write)."""
    case = request.node.callspec.id
    calls: list[str] = []
    status = 0 if case == "success" else 1
    process = _process_class(calls, status=status, break_write=break_write)
    monkeypatch.setattr(clocks._video.subprocess, "Popen", process)

    if error is not None:
        with pytest.raises(type(error)) as raised:
            _run_encoder(["the-encoder"], _frames_then(error))
        assert raised.value is error
        assert "kill" in calls
        assert calls.index("kill") < calls.index("wait")
    else:
        result_status, _ = _run_encoder(["the-encoder"], _frames_then(None))
        assert result_status == status
        if case == "success":
            assert calls == ["write", "write", "write", "close", "wait"]
        else:
            assert calls.index("close") < calls.index("wait")
        if break_write:
            assert calls.count("write") == 2
    assert calls.count("wait") == 1
    assert calls[-1] == "wait"


@pytest.mark.parametrize("break_write", [pytest.param(True, id="write_breaks")])
def test_an_encoder_that_stops_reading_early_is_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, break_write: bool
) -> None:
    """Only this catches a partial clip moved into place by an encoder that
    exited 0 after its input broke on a write. The close cannot break it: the
    input is unbuffered, so the close flushes nothing."""
    (tmp_path / "clip.mp4").write_bytes(b"old")
    calls: list[str] = []
    base = _process_class(calls, status=0, break_write=break_write)

    class PartialWriter(base):
        def __init__(self, command, *args, **kwargs) -> None:
            super().__init__(command, *args, **kwargs)
            Path(command[-1]).write_bytes(b"partial")

    monkeypatch.setattr(clocks._video.subprocess, "Popen", PartialWriter)

    with pytest.raises(EncoderError) as raised:
        encode(_small_clip(), tmp_path / "clip.mp4", width=64, height=48, encoder="e")

    message = str(raised.value)
    assert "status 0" in message
    assert "before reading all of its input" in message
    assert "clip.mp4" in message
    assert _names(tmp_path) == ["clip.mp4"]
    assert (tmp_path / "clip.mp4").read_bytes() == b"old"


def test_a_failed_cleanup_keeps_the_encoders_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches the cleanup's own error replacing the encoder's, or a note that
    does not name what was left behind (the Task 7 rows name the tests that
    share each edit)."""
    made: list[Path] = []

    def stand_in(command, frames) -> tuple[int, str]:
        temporary = Path(command[-1])
        temporary.mkdir()  # removing a directory with unlink() raises OSError
        made.append(temporary)
        for _ in frames:
            pass
        return 3, "boom"

    monkeypatch.setattr(clocks._video, "_run_encoder", stand_in)

    with pytest.raises(EncoderError, match="status 3") as raised:
        encode(_small_clip(), tmp_path / "clip.mp4", width=64, height=48, encoder="e")

    notes = getattr(raised.value, "__notes__", [])
    assert any(made[0].name in note and "was left behind" in note for note in notes)


def test_a_cleanup_note_that_cannot_be_added_keeps_the_encoders_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only this catches `encode`'s cleanup note replacing the error when that
    error's ``__notes__`` is not a list (``add_note`` raises)."""
    error = ValueError("drawing failed")
    error.__notes__ = ()

    def stand_in(command, frames) -> tuple[int, str]:
        Path(command[-1]).mkdir()  # removing a directory with unlink() raises
        raise error

    monkeypatch.setattr(clocks._video, "_run_encoder", stand_in)

    with pytest.raises(ValueError) as raised:
        encode(_small_clip(), tmp_path / "clip.mp4", width=64, height=48, encoder="e")

    assert raised.value is error
    assert error.__notes__ == ()


def test_an_interrupted_cleanup_keeps_the_encoders_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only this catches an interrupt while `encode` removes its temporary
    file replacing the error it is cleaning up after; the interrupt becomes
    the note instead."""
    error = ValueError("drawing failed")

    def stand_in(command, frames) -> tuple[int, str]:
        Path(command[-1]).write_bytes(b"partial")
        raise error

    real_unlink = Path.unlink

    def interrupted(self: Path, missing_ok: bool = False) -> None:
        if self.name.endswith(".part"):
            raise KeyboardInterrupt("during the unlink")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(clocks._video, "_run_encoder", stand_in)
    monkeypatch.setattr(Path, "unlink", interrupted)
    # BaseException: an escaped interrupt would end the session.
    with pytest.raises(BaseException) as raised:
        encode(_small_clip(), tmp_path / "clip.mp4", width=64, height=48, encoder="e")
    monkeypatch.undo()

    assert raised.value is error
    [note] = error.__notes__
    assert note.endswith(".part was left behind: during the unlink")


# Reads its input to the end, then stays alive: an encoder still finishing.
_LINGERING_CHILD = "import sys, time; sys.stdin.buffer.read(); time.sleep(60)"


def test_an_interrupt_while_waiting_reaps_the_encoder_before_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only this catches an interrupt during the encoder's normal wait leaving
    the child running while `encode` removes its temporary file, which a child
    that opens its output late could then write after the cleanup."""
    started: list[subprocess.Popen] = []
    at_cleanup: list[list[int | None]] = []
    interrupt = KeyboardInterrupt()

    class InterruptedOnFirstWait(subprocess.Popen):
        def __init__(self, args, **kwargs) -> None:
            self.interrupted = False
            super().__init__([sys.executable, "-c", _LINGERING_CHILD], **kwargs)
            started.append(self)

        def wait(self, timeout: float | None = None) -> int:
            if not self.interrupted:
                self.interrupted = True
                raise interrupt
            return super().wait(timeout)

    real_unlink = Path.unlink

    def unlink_spy(self: Path, missing_ok: bool = False) -> None:
        at_cleanup.append([process.returncode for process in started])
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(clocks._video.subprocess, "Popen", InterruptedOnFirstWait)
    monkeypatch.setattr(Path, "unlink", unlink_spy)
    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            encode(
                _small_clip(), tmp_path / "clip.mp4", width=64, height=48, encoder="e"
            )
    finally:
        for process in started:
            if process.returncode is None:
                process.kill()
                process.wait()

    assert raised.value is interrupt
    [process] = started
    assert process.interrupted
    assert at_cleanup == [[-signal.SIGKILL]]
    assert _names(tmp_path) == []


# Ignores SIGTERM, says so by creating the file named in its argument, and
# never reads its input: only a kill ends it, and a full pipe stays full.
_DEAF_CHILD = (
    "import signal, sys, time, pathlib; "
    "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
    "pathlib.Path(sys.argv[1]).touch(); time.sleep(60)"
)
_SLEEPING_CHILD = "import time; time.sleep(60)"


def _fill(fd: int) -> None:
    """Write to the pipe ``fd`` until it takes no more."""
    os.set_blocking(fd, False)
    try:
        for size in (65536, 4096, 1):
            try:
                while True:
                    os.write(fd, b"\0" * size)
            except BlockingIOError:
                pass
    finally:
        os.set_blocking(fd, True)


def test_a_cancelled_encode_with_a_full_pipe_ends_promptly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches the cancellation leaving the encoder's input open, or the input
    buffered (a close would then flush into the full pipe); the Task 7 rows
    name the tests that share each edit.
    Here the pipe is full, the child never reads and ignores SIGTERM, and the
    producer then fails: the encoder is killed and reaped within a few
    seconds (a watchdog, after 10 s, must not have to), with no note, since
    nothing in the cleanup failed."""
    ready = tmp_path / "ready"
    started: list[subprocess.Popen] = []
    error = ValueError("drawing failed")

    class Recorded(subprocess.Popen):
        def __init__(self, args, **kwargs) -> None:
            super().__init__(args, **kwargs)
            started.append(self)

    def frames() -> Iterator[bytes]:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        _fill(started[0].stdin.fileno())
        raise error
        yield b""  # a generator

    killed_by_watchdog: list[int] = []

    def watchdog() -> None:
        for process in started:
            if process.returncode is None:
                killed_by_watchdog.append(process.pid)
                os.kill(process.pid, signal.SIGKILL)

    timer = threading.Timer(10, watchdog)
    monkeypatch.setattr(clocks._video.subprocess, "Popen", Recorded)
    command = [sys.executable, "-c", _DEAF_CHILD, str(ready)]
    timer.start()
    begun = time.monotonic()
    try:
        with pytest.raises(ValueError) as raised:
            _run_encoder(command, frames())
        elapsed = time.monotonic() - begun
    finally:
        timer.cancel()
        for process in started:
            if process.returncode is None:
                os.kill(process.pid, signal.SIGKILL)
                process.wait()

    assert raised.value is error
    assert getattr(error, "__notes__", []) == []
    assert killed_by_watchdog == []
    assert elapsed < 5
    [process] = started
    assert process.returncode == -signal.SIGKILL
    assert process.stdin.closed
    # Unbuffered, so no close can flush (and block on) frames left in a buffer.
    assert type(process.stdin) is io.FileIO


class _CloseRaises:
    """The encoder's input, whose ``close`` closes it and then raises."""

    def __init__(self, pipe: IO[bytes], error: BaseException) -> None:
        self.pipe, self.error = pipe, error

    def write(self, data) -> int | None:
        return self.pipe.write(data)

    def close(self) -> None:
        self.pipe.close()
        raise self.error


class _Unprintable(Exception):
    """An exception whose description itself raises."""

    def __str__(self) -> str:
        raise RuntimeError("cannot describe it")


_CANCEL_CASES = (
    "kill_fails",
    "kill_interrupted",
    "kill_raises",
    "kill_error_unprintable",
    "close_raises",
    "close_interrupted",
    "reap_times_out",
    "second_interrupt",
    "notes_not_a_list",
)


@pytest.mark.parametrize("case", _CANCEL_CASES)
def test_a_failing_cleanup_step_never_replaces_the_cancelling_error(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """Catches a cleanup step's own exception replacing the error that
    cancelled the encode, a failed step blocking the cleanup or skipping the
    steps after it, or a note that does not say what its step leaves. Each
    case fails one step: ``kill()`` raising without killing (the reap then
    times out, within a bound), raising an interrupt, a non-OSError, or an
    exception whose description itself raises; the close raising a
    non-OSError or an interrupt; the reap timing out; a second interrupt while
    reaping; and a note that cannot be added (``__notes__`` not a list). The
    input is closed in every case. The reap waits ``_REAP_WAIT`` seconds,
    patched to 0.2. The Task 7 rows name the cases each edit fails."""
    real_wait = subprocess.Popen.wait  # the patch below replaces the class
    started: list[subprocess.Popen] = []
    close_error = {
        "close_raises": RuntimeError("close failed"),
        "close_interrupted": KeyboardInterrupt("during the close"),
    }.get(case)

    class Failing(subprocess.Popen):
        def __init__(self, args, **kwargs) -> None:
            super().__init__(args, **kwargs)
            started.append(self)
            if close_error is not None:
                self.stdin = _CloseRaises(self.stdin, close_error)

        def kill(self) -> None:
            if case in ("kill_fails", "reap_times_out"):
                if case == "kill_fails":
                    raise PermissionError("kill refused")
                return  # reports success, sends nothing
            super().kill()
            if case == "kill_interrupted":
                raise KeyboardInterrupt("during the kill")
            if case in ("kill_raises", "notes_not_a_list"):
                raise RuntimeError("kill failed")
            if case == "kill_error_unprintable":
                raise _Unprintable()

        def wait(self, timeout: float | None = None) -> int:
            if case == "second_interrupt":
                raise KeyboardInterrupt("the second")
            return super().wait(timeout)

    if case == "second_interrupt":
        error: BaseException = KeyboardInterrupt("the first")
    else:
        error = ValueError("drawing failed")
    if case == "notes_not_a_list":
        error.__notes__ = ()

    def frames() -> Iterator[bytes]:
        yield b"x" * 16
        raise error

    monkeypatch.setattr(clocks._video, "_REAP_WAIT", 0.2)
    monkeypatch.setattr(clocks._video.subprocess, "Popen", Failing)
    begun = time.monotonic()
    try:
        # BaseException: a step's own interrupt that escaped would otherwise
        # end the whole pytest session rather than fail this test.
        with pytest.raises(BaseException) as raised:
            _run_encoder([sys.executable, "-c", _SLEEPING_CHILD], frames())
        elapsed = time.monotonic() - begun
        reaped = [process.returncode for process in started]
        closed = [getattr(p.stdin, "pipe", p.stdin).closed for p in started]
    finally:
        for process in started:
            if process.returncode is None:
                os.kill(process.pid, signal.SIGKILL)
                real_wait(process)

    assert raised.value is error
    assert elapsed < 5
    assert closed == [True]

    def note(step: str, cause: str, leaves: str) -> str:
        return f"cancelling the encoder, could not {step}: {cause}; {leaves}"

    running = "it may still be running"
    timed_out = f"cancelling the encoder, it was not reaped within 0.2 s: {running}"
    expected = {
        "kill_fails": (
            [note("kill it", "PermissionError: kill refused", running), timed_out],
            None,
        ),
        "kill_interrupted": (
            [note("kill it", "KeyboardInterrupt: during the kill", running)],
            -signal.SIGKILL,
        ),
        "kill_raises": (
            [note("kill it", "RuntimeError: kill failed", running)],
            -signal.SIGKILL,
        ),
        "kill_error_unprintable": (None, -signal.SIGKILL),
        "close_raises": (
            [
                note(
                    "close its input",
                    "RuntimeError: close failed",
                    "its input may still be open",
                )
            ],
            -signal.SIGKILL,
        ),
        "close_interrupted": (
            [
                note(
                    "close its input",
                    "KeyboardInterrupt: during the close",
                    "its input may still be open",
                )
            ],
            -signal.SIGKILL,
        ),
        "reap_times_out": ([timed_out], None),
        "second_interrupt": (
            [
                note(
                    "reap it",
                    "KeyboardInterrupt: the second",
                    "it may be left unreaped",
                )
            ],
            None,
        ),
        "notes_not_a_list": ((), -signal.SIGKILL),
    }[case]
    assert getattr(error, "__notes__", None) == expected[0]
    assert reaped == [expected[1]]


def test_the_default_reap_wait_is_a_short_bound() -> None:
    """Only this catches the reap's default limit set so long that a
    cancelled encode could wait for minutes (the other tests patch it)."""
    assert 0 < clocks._video._REAP_WAIT <= 10


def test_feed_writes_every_byte_of_a_frame_across_partial_writes() -> None:
    """Only this catches a frame written with one ``write`` call, whose
    partial write would drop the rest of the frame."""

    class Trickle:
        def __init__(self) -> None:
            self.received = bytearray()
            self.closed = False

        def write(self, data) -> int:
            taken = bytes(data[:5])
            self.received += taken
            return len(taken)

        def close(self) -> None:
            self.closed = True

    pipe = Trickle()
    frames = [b"0123456789abcdefghijk", b"xyz"]

    assert _feed(pipe, frames) is True
    assert bytes(pipe.received) == b"".join(frames)
    assert pipe.closed


@pytest.mark.parametrize("taken", [None, 0])
def test_feed_refuses_a_write_that_takes_nothing(taken: int | None) -> None:
    """Only this catches a write that takes no bytes being passed over, which
    would drop the frame or loop for ever."""

    class Stuck:
        def write(self, data) -> int | None:
            return taken

        def close(self) -> None:
            pass

    with pytest.raises(EncoderError, match="took no bytes"):
        _feed(Stuck(), [b"frame"])


class _CutOff:
    """Takes 5 bytes of the first write, then finds the read end closed."""

    def __init__(self) -> None:
        self.writes = 0
        self.closed = False

    def write(self, data) -> int:
        self.writes += 1
        if self.writes == 1:
            return 5
        raise BrokenPipeError("the read end closed")

    def close(self) -> None:
        self.closed = True


def test_a_frame_cut_off_by_a_broken_pipe_is_not_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches a broken pipe part-way through a frame ending only that frame
    (writing would go on with the next), or reported as complete; and an
    encoder that exits 0 after it being trusted."""
    pipe = _CutOff()

    assert _feed(pipe, [b"0123456789", b"abc"]) is False
    assert (pipe.writes, pipe.closed) == (2, True)

    class ExitsZero:
        def __init__(self, *args, **kwargs) -> None:
            self.stdin = _CutOff()

        def wait(self, timeout: float | None = None) -> int:
            return 0

    monkeypatch.setattr(clocks._video.subprocess, "Popen", ExitsZero)
    with pytest.raises(EncoderError, match="before reading all of its input"):
        _run_encoder(["the-encoder", "clip.mp4"], iter([b"0123456789"]))


def test_encode_looks_for_the_encoder_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def missing() -> str:
        raise EncoderNotFound("none")

    def frames() -> Iterator[tuple[bytes, int]]:
        pytest.fail("a frame was asked for")
        yield b"", 0

    monkeypatch.setattr(clocks._video, "find_encoder", missing)

    with pytest.raises(EncoderNotFound):
        encode(frames(), tmp_path / "clip.mp4", width=64, height=48)


# ------------------------------------------------------------ the decoder


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        pytest.param(Q2_LINE, ("h264", "yuv420p", 1500, 900, 20.0), id="ffmpeg_9"),
        pytest.param(
            "  Stream #0:0(und): Video: h264 (High) (avc1 / 0x31637661), "
            "yuv420p(tv, bt709, progressive), 1500x1100 [SAR 1:1 DAR 15:11], "
            "330 kb/s, 20 fps, 20 tbr, 10240 tbn (default)",
            ("h264", "yuv420p", 1500, 1100, 20.0),
            id="with_colour_and_aspect",
        ),
        pytest.param(
            "  Stream #0:0[0x1](und): Video: mpeg4 (Simple Profile) "
            "(mp4v / 0x7634706D), yuv420p, 64x48 [SAR 1:1 DAR 4:3], 5 kb/s, "
            "20 fps, 20 tbr, 10240 tbn (default)",
            ("mpeg4", "yuv420p", 64, 48, 20.0),
            id="mpeg4",
        ),
        pytest.param(
            "  Stream #0:0[0x1](und): Video: h264 (High 4:4:4 Predictive) "
            "(avc1 / 0x31637661), yuv444p(progressive), 64x48, 19 kb/s, 20 fps, "
            "20 tbr, 10240 tbn (default)",
            ("h264", "yuv444p", 64, 48, 20.0),
            id="yuv444p",
        ),
        pytest.param(
            Q2_LINE.replace("20 fps", "29.97 fps"),
            ("h264", "yuv420p", 1500, 900, 29.97),
            id="fractional_rate",
        ),
        pytest.param(
            Q2_LINE + "\n  Stream #0:0(und): Video: wrapped_avframe, "
            "yuv420p(progressive), 1500x900, q=2-31, 200 kb/s, 20 fps, 20 tbn "
            "(default)",
            ("h264", "yuv420p", 1500, 900, 20.0),
            id="two_video_lines",
        ),
    ],
)
def test_parse_stream_line(line: str, expected: tuple) -> None:
    assert parse_stream_line(line) == expected


@pytest.mark.parametrize(
    ("text", "match"),
    [
        pytest.param(
            "  Stream #0:0: Audio: aac, 44100 Hz", "no video stream", id="audio_only"
        ),
        pytest.param(Q2_LINE.replace(", 20 fps", ""), None, id="no_rate"),
    ],
)
def test_parse_stream_line_refuses_other_text(text: str, match: str | None) -> None:
    with pytest.raises(ValueError) as raised:
        parse_stream_line(text)
    assert (match or text.strip()) in str(raised.value)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("frame=45\nfps=0.00\nprogress=end\n", 45, id="finished"),
        pytest.param(
            "frame=30\nprogress=continue\nframe=92\nprogress=end\n",
            92,
            id="two_records",
        ),
        pytest.param("frame=0\nprogress=end\n", 0, id="zero_frames"),
        pytest.param(
            "frame=45\nprogress=continue\n", "did not finish", id="unfinished"
        ),
        pytest.param("", "did not finish", id="empty"),
        pytest.param("fps=0.00\nprogress=end\n", "no frame count", id="no_frame_line"),
        pytest.param(
            "frame=7\nprogress=continue\nfps=0\nprogress=end\n",
            "no frame count",
            id="count_only_in_an_earlier_record",
        ),
        pytest.param("frame=N/A\nprogress=end\n", "no frame count", id="not_a_number"),
        pytest.param("frame=-1\nprogress=end\n", "no frame count", id="negative"),
        pytest.param(
            "frame=N/A\nprogress=continue\nframe=7\nprogress=end\n",
            7,
            id="malformed_earlier_record",
        ),
    ],
)
def test_parse_progress(text: str, expected: int | str) -> None:
    if isinstance(expected, str):
        with pytest.raises(ValueError, match=expected):
            parse_progress(text)
    else:
        assert parse_progress(text) == expected


@pytest.mark.parametrize(
    ("result", "match"),
    [
        pytest.param(
            (1, "frame=7\nprogress=end\n", Q2_LINE),
            r"the decoder failed \(exit status 1\)",
            id="failed",
        ),
        pytest.param(
            (0, "frame=5\nprogress=continue\n", Q2_LINE),
            "did not finish",
            id="unfinished",
        ),
        pytest.param(
            (0, "frame=7\nprogress=end\n", ""), "no video stream", id="no_stream"
        ),
    ],
)
def test_probe_requires_a_finished_decode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    result: tuple[int, str, str],
    match: str,
) -> None:
    path = tmp_path / "clip.mp4"
    encode(_small_clip(), path, width=64, height=48)
    monkeypatch.setattr(clocks._video, "_run_decoder", lambda command: result)

    with pytest.raises(ValueError, match=match):
        probe(path)


def _box(kind: bytes, payload: bytes = b"", size: int | None = None) -> bytes:
    stated = 8 + len(payload) if size is None else size
    return struct.pack(">I4s", stated, kind) + payload


def _large_box(kind: bytes, large_size: int, payload: bytes = b"") -> bytes:
    return struct.pack(">I4s", 1, kind) + struct.pack(">Q", large_size) + payload


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(
            _box(b"ftyp", b"isom")
            + _box(b"moov", b"x" * 12)
            + _box(b"free")
            + _box(b"mdat", b"y" * 5),
            ["ftyp", "moov", "free", "mdat"],
            id="four_boxes",
        ),
        pytest.param(
            _large_box(b"mdat", 24, b"z" * 8) + _box(b"moov", b"x" * 4),
            ["mdat", "moov"],
            id="large_size",
        ),
        pytest.param(
            _box(b"ftyp", b"isom") + _box(b"mdat", b"y" * 4, size=1000),
            ["ftyp", "mdat"],
            id="runs_past_the_end",
        ),
        pytest.param(
            _large_box(b"mdat", 0, b"z" * 8) + _box(b"moov", b"x" * 4),
            ["mdat"],
            id="large_size_zero",
        ),
        pytest.param(
            _large_box(b"mdat", 8, b"z" * 8) + _box(b"moov", b"x" * 4),
            ["mdat"],
            id="large_size_under_its_header",
        ),
        pytest.param(
            _box(b"mdat", b"z" * 8, size=4) + _box(b"moov", b"x" * 4),
            ["mdat"],
            id="size_under_its_header",
        ),
        pytest.param(
            _box(b"ftyp", b"isom") + struct.pack(">I4s", 1, b"free") + b"\0" * 4,
            ["ftyp", "free"],
            id="large_size_cut_short",
        ),
    ],
)
def test_box_order(tmp_path: Path, data: bytes, expected: list[str]) -> None:
    path = tmp_path / "boxes.mp4"
    path.write_bytes(data)
    assert box_order(path) == expected


def test_probe_of_a_truncated_clip_is_an_error_or_a_short_count(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(0)

    def noise() -> Iterator[tuple[bytes, int]]:
        for _ in range(60):
            frame = rng.integers(0, 256, (48, 64, 4), np.uint8)
            frame[..., 3] = 255
            yield frame.tobytes(), 1

    path = tmp_path / "clip.mp4"
    encode(noise(), path, width=64, height=48)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])

    try:
        info = probe(path)
    except ValueError:
        return
    assert info.codec == "h264"
    assert info.moov_first is True
    assert 0 < info.frames < 60


def test_probe_sees_moov_after_mdat(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    other = tmp_path / "other.mp4"
    encode(_small_clip(), clip, width=64, height=48)
    command = [find_encoder(), "-loglevel", "error", "-y", "-i", str(clip)]
    subprocess.run([*command, "-c", "copy", str(other)], check=True)

    info = probe(other)

    assert info.moov_first is False
    assert info.frames == 7


def test_probe_refuses_a_file_that_is_not_video(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    path.write_text("this is not a video\n")
    with pytest.raises(ValueError, match="the decoder failed"):
        probe(path)


_DECODE_ERRORS = ("error while decoding", "concealing", "corrupt decoded frame")


def _mdat_payload(data: bytes) -> tuple[int, int]:
    """The offset and length of the top-level ``mdat`` box's payload."""
    offset = 0
    while offset + 8 <= len(data):
        size, kind = struct.unpack(">I4s", data[offset : offset + 8])
        if kind == b"mdat":
            return offset + 8, size - 8
        offset += size
    raise AssertionError("the clip has no mdat box")


def _lenient_decode(path: Path) -> tuple[int, bool]:
    """Decode ``path`` without ``-xerror``: its frame count, and whether it erred."""
    command = [
        find_encoder(), "-hide_banner", "-nostdin", "-i", str(path),
        "-map", "0:v:0", "-f", "null", "-", "-progress", "pipe:1", "-nostats",
    ]  # fmt: skip
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    erred = any(marker in result.stderr for marker in _DECODE_ERRORS)
    return parse_progress(result.stdout), erred


def test_probe_refuses_a_clip_with_a_corrupt_frame_of_full_length(
    tmp_path: Path,
) -> None:
    """Zeroed bytes inside a frame fail ``probe`` even when no frame goes missing.

    Encoded bytes differ between FFmpeg builds, so the damage is tried at many
    offsets of the ``mdat`` payload; the copies that matter are those a lenient
    decode (the decoder without ``-xerror``) reports errors in while still
    counting every frame. At least one such copy must exist, and ``probe`` must
    refuse each of them.
    """
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(20):
        frame = rng.integers(0, 256, (HEIGHT, WIDTH, 4), np.uint8)
        frame[..., 3] = 255
        frames.append((frame.tobytes(), 1))
    clip = tmp_path / "clip.mp4"
    encode(frames, clip, width=WIDTH, height=HEIGHT)
    data = clip.read_bytes()
    start, length = _mdat_payload(data)

    damaged = 0
    for fraction in np.linspace(0.02, 0.98, 49):
        at = start + int((length - 32) * fraction)
        copy = tmp_path / f"damaged-{at}.mp4"
        copy.write_bytes(data[:at] + bytes(32) + data[at + 32 :])
        frames_decoded, erred = _lenient_decode(copy)
        if not (erred and frames_decoded == 20):
            continue
        damaged += 1
        with pytest.raises(ValueError, match="the decoder failed"):
            probe(copy)

    assert damaged > 0, "no damaged copy kept its length while decoding with errors"


def test_encoder_version_is_the_first_line() -> None:
    assert encoder_version(find_encoder()).startswith("ffmpeg version ")


@pytest.mark.parametrize(
    ("result", "expected", "not_said"),
    [
        pytest.param(
            (1, "ffmpeg version 9.0\n", "bad"),
            "e -version exited with status 1: bad",
            "version line",
            id="non_zero_exit",
        ),
        pytest.param(
            (0, "\n", "quiet"),
            "e -version printed no version line: quiet",
            "exited with status",
            id="no_version_line",
        ),
    ],
)
def test_encoder_version_names_the_cause_it_saw(
    monkeypatch: pytest.MonkeyPatch, result, expected: str, not_said: str
) -> None:
    """Only this catches a non-zero `-version` reported as a missing version
    line, or the reverse."""
    monkeypatch.setattr(clocks._video, "_run_decoder", lambda command: result)
    with pytest.raises(EncoderError) as raised:
        encoder_version("e")
    assert str(raised.value) == expected
    assert not_said not in str(raised.value)


def test_the_encoder_is_a_development_dependency_only() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    development = project["dependency-groups"]["dev"]
    assert any(entry.startswith("imageio-ffmpeg") for entry in development)
    required = project["project"]["dependencies"]
    assert not any(entry.startswith("imageio") for entry in required)
