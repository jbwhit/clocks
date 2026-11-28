"""Raw frames to an H.264 clip, and a check that reads one back.

A clip is drawn once per row and each drawing is written as many times as its
dwell asks, at ``FPS`` frames a second. The encoder is FFmpeg: the binary that
``imageio-ffmpeg`` bundles (a development dependency, never a requirement of the
library), else ``ffmpeg`` on ``PATH``. ``probe`` runs the same binary as a
decoder, since the bundle has no ``ffprobe``, and reads its report.

Only the standard library, and ``clocks._notes`` (which imports only the
standard library), are imported at module level, so ``clocks.viz`` still
imports from a wheel installed without the encoder. The two functions
``_run_encoder`` and ``_run_decoder`` are the only places a process is
started; tests replace them to check arguments and failure handling.
"""

import os
import re
import secrets
import shutil
import struct
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Iterator, Sequence
from functools import partial
from pathlib import Path
from typing import IO, NamedTuple

from clocks._notes import attempt, left_behind

FPS = 20

_TAIL_LINES = 20

# Seconds a cancelled encoder, already sent SIGKILL, is given to be reaped.
_REAP_WAIT = 5.0
_RUNNING = "it may still be running"


class EncoderNotFound(RuntimeError):
    """Neither ``imageio-ffmpeg`` nor an ``ffmpeg`` on ``PATH`` was found."""


class EncoderError(RuntimeError):
    """The encoder failed, or its clip cannot be trusted.

    Raised when the encoder exits with a non-zero status; exits 0 after a
    write to it broke the pipe; takes no bytes of a write (``_feed``); or
    cannot be started at all. Asked for ``-version`` (``encoder_version``), it is raised
    when the binary exits non-zero, or exits 0 and prints no version line.
    The demo commands raise it too (``clocks._demos._common._check_clip``) for
    a clip whose decoder cannot be started, that does not decode, or that
    decodes to a frame count other than the frames sent.
    """


class StreamInfo(NamedTuple):
    """What a decode of a clip reports."""

    codec: str  # "h264"
    pix_fmt: str  # "yuv420p"
    width: int
    height: int
    fps: float
    frames: int  # frames the encoder decoded, not a count read from the container
    moov_first: bool  # the moov box comes before mdat


def dwell_frames(n_observations: int) -> list[int]:
    """Return how many video frames each of the T + 1 rows is held for.

    The prior holds 1.2 s; observations 1 to 3, 0.9 s; 4 to 8, 0.5 s; 9 to 20,
    0.25 s; later ones 0.1 s; the last row 2.5 s, all at ``FPS`` frames a second.
    """
    if n_observations < 1:
        raise ValueError(f"a clip needs at least one observation, got {n_observations}")
    middle = [_dwell(k) for k in range(1, n_observations)]
    return [24, *middle, 50]


def _dwell(k: int) -> int:
    if k <= 3:
        return 18
    if k <= 8:
        return 10
    if k <= 20:
        return 5
    return 2


def find_encoder() -> str:
    """Return the FFmpeg to use: ``imageio-ffmpeg``'s, else ``ffmpeg`` on ``PATH``."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        pass  # no package, or a package without a binary: try PATH
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    raise EncoderNotFound(
        "no FFmpeg encoder found: install imageio-ffmpeg "
        "(in a checkout, `uv sync` provides it) or put ffmpeg on PATH"
    )


def encoder_version(encoder: str) -> str:
    """Return the first line ``encoder -version`` prints, for a log."""
    status, out, err = _run_decoder([encoder, "-version"])
    if status != 0:
        raise EncoderError(
            f"{encoder} -version exited with status {status}: {_tail(err)}"
        )
    lines = out.splitlines()
    if not lines or not lines[0].strip():
        raise EncoderError(f"{encoder} -version printed no version line: {_tail(err)}")
    return lines[0].strip()


def encode(
    frames: Iterable[tuple[bytes, int]],
    path: Path,
    *,
    width: int,
    height: int,
    encoder: str | None = None,
) -> int:
    """Encode ``(frame, count)`` pairs of raw RGBA to an H.264 clip at ``path``.

    Each frame, ``width * height * 4`` bytes, is written ``count`` times. The
    pairs are taken one at a time, so a clip is never held in memory. The clip
    is written beside ``path`` under a temporary name and moved into place only
    when the encoder exits 0 and no write to it failed with a broken pipe; on
    any failure ``path`` is left as it was.
    Returns the number of frames sent with no broken pipe seen, which is not a
    count of the frames in the file.

    The temporary name is absolute, on the command line and for the rename and
    cleanup alike: FFmpeg reads a relative name beginning with ``-`` as an
    option and one beginning with ``file:`` (or another protocol) as a URL.
    """
    if width % 2 or height % 2:
        raise ValueError(
            f"a yuv420p clip needs an even width and height, got {width}x{height}"
        )
    if encoder is None:
        encoder = find_encoder()
    target = path.absolute()
    temporary = target.with_name(f"{target.name}.{secrets.token_hex(8)}.part")
    command = [
        encoder, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{width}x{height}",
        "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "20", "-preset", "slow", "-movflags", "+faststart", "-f", "mp4",
        str(temporary),
    ]  # fmt: skip
    expected = width * height * 4
    written = 0

    def stream() -> Iterator[bytes]:
        nonlocal written
        for n, (frame, count) in enumerate(frames, start=1):
            if len(frame) != expected:
                raise ValueError(
                    f"frame {n} has {len(frame)} bytes, expected {expected}"
                )
            for _ in range(count):
                written += 1
                yield frame

    try:
        status, errors = _run_encoder(command, stream())
        if status != 0:
            raise EncoderError(
                f"{encoder} exited with status {status} while writing {path}: "
                f"{_tail(errors)}"
            )
        os.replace(temporary, target)
    except BaseException as error:
        attempt(
            error,
            lambda: temporary.unlink(missing_ok=True),
            partial(left_behind, temporary),
        )
        raise
    return written


def parse_stream_line(text: str) -> tuple[str, str, int, int, float]:
    """Read codec, pixel format, size and rate from the first ``Video:`` line."""
    line = next((x for x in text.splitlines() if "Video:" in x), None)
    if line is None:
        raise ValueError(f"no video stream in the decoder's report: {_tail(text)}")
    head = re.search(r"Video:\s*(\w+)[^,]*,\s*(\w+)", line)
    size = head and re.search(r"(\d+)x(\d+)", line[head.end() :])
    rate = size and re.search(r"(\d+(?:\.\d+)?) fps\b", line[head.end() :])
    if not (head and size and rate):
        raise ValueError(f"cannot read the decoder's video stream line: {line}")
    codec, pix_fmt = head.groups()
    width, height = (int(x) for x in size.groups())
    return codec, pix_fmt, width, height, float(rate.group(1))


def parse_progress(text: str) -> int:
    """Return the frame count of the last record of a finished ``-progress`` report."""
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if not lines or lines[-1] != "progress=end":
        last = lines[-1] if lines else "nothing"
        raise ValueError(
            f"the decoder's progress report did not finish: its last line is {last!r}"
        )
    earlier = [i for i, x in enumerate(lines[:-1]) if x.startswith("progress=")]
    record = lines[earlier[-1] + 1 if earlier else 0 : -1]
    counts = [x.removeprefix("frame=") for x in record if x.startswith("frame=")]
    if not counts or not re.fullmatch(r"[0-9]+", counts[-1]):
        raise ValueError(
            f"the decoder's progress report has no frame count in its last record: "
            f"{record}"
        )
    return int(counts[-1])


def box_order(path: Path) -> list[str]:
    """Return the four-character types of the file's top-level boxes, in order."""
    end = path.stat().st_size
    boxes: list[str] = []
    offset = 0
    with path.open("rb") as file:
        while offset + 8 <= end:
            file.seek(offset)
            size, kind = struct.unpack(">I4s", file.read(8))
            boxes.append(kind.decode("latin-1"))
            header = 8
            if size == 1:
                if offset + 16 > end:
                    break
                (size,) = struct.unpack(">Q", file.read(8))
                header = 16
            # Each rule ends the walk on its own; together they make every
            # step advance, so the walk ends.
            if size == 0 or size < header or offset + size > end:
                break
            offset += size
    return boxes


def probe(path: Path, encoder: str | None = None) -> StreamInfo:
    """Decode ``path`` to nothing and report its stream, frame count and box order.

    Raises ``ValueError`` when the decoder fails or its report cannot be read.
    The decoder runs with ``-xerror``, so a frame it cannot decode cleanly fails
    the probe even when the stream keeps its length (FFmpeg otherwise conceals
    the damage, exits 0 and counts the frame). It decodes on one thread: with
    frame threads, an error in one of the last frames can reach ``-xerror`` too
    late and the decoder exits 0 anyway (measured with FFmpeg 9.0). A file cut
    short may decode with fewer frames; callers compare the count. ``path`` is
    given to the decoder as an absolute name, for the reason ``encode`` gives.
    """
    if encoder is None:
        encoder = find_encoder()
    command = [
        encoder, "-hide_banner", "-nostdin", "-xerror", "-threads", "1",
        "-i", str(path.absolute()), "-map", "0:v:0",
        "-f", "null", "-", "-progress", "pipe:1", "-nostats",
    ]  # fmt: skip
    status, out, err = _run_decoder(command)
    if status != 0:
        raise ValueError(
            f"{path}: the decoder failed (exit status {status}): {_tail(err)}"
        )
    try:
        codec, pix_fmt, width, height, fps = parse_stream_line(err)
        frames = parse_progress(out)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
    boxes = box_order(path)
    moov_first = "moov" in boxes and (
        "mdat" not in boxes or boxes.index("moov") < boxes.index("mdat")
    )
    return StreamInfo(codec, pix_fmt, width, height, fps, frames, moov_first)


def _tail(text: str) -> str:
    return "\n".join(text.strip().splitlines()[-_TAIL_LINES:])


def _run_encoder(command: Sequence[str], frames: Iterable[bytes]) -> tuple[int, str]:
    """Pipe ``frames`` to ``command``; return its exit status and standard error.

    The encoder's input is an unbuffered pipe: every byte is written by
    ``_feed`` itself, so closing it flushes nothing and can neither block nor
    report a broken pipe. A write that observes a broken pipe has found the
    read end closed; the encoder may still be running, and its exit status
    need not say why. Writing stops, the input is closed, the encoder is
    waited for, and its status is returned. An encoder that exits 0 after a
    broken pipe raises ``EncoderError``; one that exits 0 having taken every
    byte without a broken pipe is trusted here, and only a decode of the file
    can say whether it holds every frame. An encoder that cannot be started
    at all raises ``EncoderError`` naming it. Standard error goes to a file,
    so a chatty encoder cannot block.

    An exception from ``frames`` itself, or one raised while feeding or
    waiting (an interrupt, say), cancels the encode with ``_cancel`` and then
    propagates; see ``_cancel`` for what that does and does not guarantee.
    The teardown of the standard-error file (its ``with`` block) is outside
    that guarantee: an exception from it chains in the ordinary way, with the
    original as its ``__context__``.
    """
    with tempfile.TemporaryFile() as errors:
        try:
            process = subprocess.Popen(
                list(command), stdin=subprocess.PIPE, stderr=errors, bufsize=0
            )
        except OSError as error:
            raise EncoderError(
                f"the encoder {command[0]} could not be started: {error}"
            ) from error
        try:
            complete = _feed(process.stdin, frames)
            status = process.wait()
        except BaseException as error:
            _cancel(process, error)
            raise
        errors.seek(0)
        text = errors.read().decode(errors="replace")
    if status == 0 and not complete:
        raise EncoderError(
            f"{command[0]} exited with status 0 before reading all of its input, "
            f"writing {command[-1]}: {_tail(text)}"
        )
    return status, text


def _cancel(process: subprocess.Popen, error: BaseException) -> None:
    """Kill ``process``, close its input and reap it, before ``error`` propagates.

    The encode is abandoned and its clip discarded, so the encoder is killed
    outright. Nothing here can block without limit: the input is unbuffered,
    so its close flushes nothing, and the reap waits at most ``_REAP_WAIT``
    seconds (only a describer that itself blocks could delay it). Each step
    runs through ``clocks._notes.attempt``, with its describer built at
    import, so nothing is called before each step: an exception raised by a
    step (an interrupt included), or while its note is built or added, never
    replaces ``error`` and never stops the steps after it, and is noted on it,
    with what it leaves, when a note can be added. An asynchronous interrupt
    delivered between the steps, outside those calls, is not excluded: it
    propagates, with ``error`` as its ``__context__``. If the kill fails, the
    encoder is left running; if the close fails, its input may stay open; if
    the reap times out or fails, it is left unreaped. A surviving encoder may
    still create its temporary file after ``encode`` has tried to remove it.
    """
    attempt(error, process.kill, _KILL_NOTE)
    attempt(error, process.stdin.close, _CLOSE_NOTE)
    attempt(error, lambda: process.wait(timeout=_REAP_WAIT), _reap_note)


def _step_note(what: str, leaves: str) -> Callable[[BaseException], str]:
    """Describe a failed cancellation step, and what it leaves."""

    def describe(failure: BaseException) -> str:
        return (
            f"cancelling the encoder, {what}: {type(failure).__name__}: {failure}; "
            f"{leaves}"
        )

    return describe


def _reap_note(failure: BaseException) -> str:
    if isinstance(failure, subprocess.TimeoutExpired):
        return (
            f"cancelling the encoder, it was not reaped within {_REAP_WAIT:g} s: "
            f"{_RUNNING}"
        )
    return _REAP_FAILED_NOTE(failure)


# Built once, at import: in _cancel nothing is called before each attempt.
_KILL_NOTE = _step_note("could not kill it", _RUNNING)
_CLOSE_NOTE = _step_note("could not close its input", "its input may still be open")
_REAP_FAILED_NOTE = _step_note("could not reap it", "it may be left unreaped")


def _feed(pipe: IO[bytes], frames: Iterable[bytes]) -> bool:
    """Write every byte of each frame to the unbuffered ``pipe``, then close it;
    return whether every write went through with no broken pipe seen.

    ``True`` does not show that the encoder read the frames: a process that
    exits without reading can still let every write succeed. A write may take
    only part of a frame, so each frame is written until none of it is left;
    a write that takes nothing (``None`` or 0) raises ``EncoderError``. A write
    that observes a broken pipe has found the read end closed: writing stops
    and ``False`` is returned. Only a write can observe one here: the close
    flushes nothing, and an exception raised while taking the next frame, a
    ``BrokenPipeError`` included, is the caller's.
    """
    complete = True
    for frame in frames:
        view = memoryview(frame)
        try:
            while view:
                taken = pipe.write(view)
                if not taken:
                    raise EncoderError(
                        f"the encoder's input took no bytes of a frame ({taken!r})"
                    )
                view = view[taken:]
        except BrokenPipeError:
            complete = False
            break  # the read end closed; the exit status need not say why
    pipe.close()
    return complete


def _run_decoder(command: Sequence[str]) -> tuple[int, str, str]:
    """Run ``command`` to its end; return its status, standard output and error."""
    result = subprocess.run(
        list(command),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return result.returncode, result.stdout, result.stderr
