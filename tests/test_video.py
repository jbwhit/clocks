import ast
import struct
import subprocess
import sys
import tomllib
import types
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

import clocks._video
from clocks._video import (
    EncoderError,
    EncoderNotFound,
    StreamInfo,
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
    source = (ROOT / "src" / "clocks" / "_video.py").read_text()
    names = list(_module_level_imports(ast.parse(source)))
    assert names, "no module-level import was found"
    outside = [n for n in names if n.split(".")[0] not in sys.stdlib_module_names]
    assert outside == []


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
    def __init__(self, calls: list[str], break_write: bool, break_close: bool):
        self.calls = calls
        self.break_write = break_write
        self.break_close = break_close
        self.writes = 0

    def write(self, data: bytes) -> int:
        self.calls.append("write")
        self.writes += 1
        if self.break_write and self.writes == 2:
            raise BrokenPipeError("the encoder has exited")
        return len(data)

    def close(self) -> None:
        self.calls.append("close")
        if self.break_close:
            raise BrokenPipeError("the encoder has exited")


def _process_class(calls: list[str], *, status: int, break_write, break_close):
    class Process:
        def __init__(self, *args, **kwargs) -> None:
            self.stdin = _Pipe(calls, break_write, break_close)

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
    ("break_write", "break_close", "error"),
    [
        pytest.param(False, False, None, id="success"),
        pytest.param(True, False, None, id="write_breaks"),
        pytest.param(False, True, None, id="close_breaks"),
        pytest.param(True, True, None, id="both_break"),
        pytest.param(False, False, _DRAWING_FAILED, id="producer_raises"),
        pytest.param(False, False, _CALLERS_OWN, id="producer_raises_broken_pipe"),
    ],
)
def test_run_encoder_reaps_the_process(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    break_write: bool,
    break_close: bool,
    error: BaseException | None,
) -> None:
    case = request.node.callspec.id
    calls: list[str] = []
    status = 0 if case == "success" else 1
    process = _process_class(
        calls, status=status, break_write=break_write, break_close=break_close
    )
    monkeypatch.setattr(clocks._video.subprocess, "Popen", process)

    if error is not None:
        with pytest.raises(type(error)) as raised:
            _run_encoder(["the-encoder"], _frames_then(error))
        assert raised.value is error
        assert "terminate" in calls
        assert calls.index("terminate") < calls.index("wait")
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


@pytest.mark.parametrize(
    ("break_write", "break_close"),
    [
        pytest.param(True, False, id="write_breaks"),
        pytest.param(False, True, id="close_breaks"),
    ],
)
def test_an_encoder_that_stops_reading_early_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    break_write: bool,
    break_close: bool,
) -> None:
    """Only this catches a partial clip moved into place by an encoder that
    exited 0 after its input broke, on a write or only on the final close."""
    (tmp_path / "clip.mp4").write_bytes(b"old")
    calls: list[str] = []
    base = _process_class(
        calls, status=0, break_write=break_write, break_close=break_close
    )

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
    """Only this catches the cleanup's own error replacing the encoder's."""
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


def test_the_encoder_is_a_development_dependency_only() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    development = project["dependency-groups"]["dev"]
    assert any(entry.startswith("imageio-ffmpeg") for entry in development)
    required = project["project"]["dependencies"]
    assert not any(entry.startswith("imageio") for entry in required)
