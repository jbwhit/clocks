"""The site's clips and posters, drawn into ``site/media/`` before Quarto renders.

``site/prerender.py`` calls ``render_site_media`` as Quarto's pre-render step.
The media directory belongs wholly to this step: nothing else writes there, it
is git-ignored, and two runs at once in one directory are not supported.

A clip and its poster are a pair, rebuilt together. Both are first written under
staged names (the final names with ``STAGED`` in front); only then are the two
old files removed and the two new ones moved in. A failure while staging leaves
the old pair exactly as it was; an interruption during publication leaves a
file missing, which makes the pair stale on the next run, never a new clip
beside an old poster. A render that is not full keeps every other clip and
poster in the directory, current or not: only a full render followed by the
rendered-site check establishes the published set.

Three inputs are refused rather than trusted. A media path that is itself a
symbolic link is refused before anything is created or removed, since the
cleanup would follow it into another directory. An archive whose replay is of
another demo is refused before anything is drawn. And each archive is read
once, its replay loaded from the bytes that were hashed; the file is hashed
again before each of its pairs is published and again after. If it differs or
cannot be read, every clip and poster of that demo is removed and
``ArchiveChanged`` is raised; and when anything else fails while a pair is
rebuilt, the archive is checked the same way before the error propagates. So
no pair drawn from old content can look fresh beside a new archive, whatever
its modification time says. An archive replaced after the last check of a run
keeping an older modification time than the media is not seen: freshness is
judged by time.
"""

import functools
import hashlib
import os
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

from clocks._clips import final_frame, render_clip
from clocks._replay import DEMOS, Replay, load_replay
from clocks._style import THEMES, save_png, style
from clocks._video import encoder_version, find_encoder

CLIP_DEMOS: tuple[str, ...] = tuple(name for name in DEMOS if name != "demo_density")
PACKAGE = Path(__file__).resolve().parent
STAGED = ".staged-"

_MEDIA_SUFFIXES = frozenset({".mp4", ".png"})


class ArchiveChanged(RuntimeError):
    """An archive changed while a pair drawn from it was being rendered."""


def media_paths(media: Path, demo: str, theme: str) -> tuple[Path, Path]:
    """The clip and the poster of ``demo`` in ``theme``, under ``media``."""
    return media / f"{demo}.{theme}.mp4", media / f"{demo}.{theme}.png"


def render_site_media(
    archives: Path,
    media: Path,
    *,
    full: bool,
    demos: Sequence[str] = CLIP_DEMOS,
    sources: Path = PACKAGE,
    report: Callable[[str], None] = print,
) -> list[Path]:
    """Draw each demo's clip and poster, in every theme, into ``media``.

    A ``media`` that is a symbolic link raises ``ValueError`` first. The encoder
    is found next: when there is none, ``EncoderNotFound`` propagates before
    anything is created or removed. Every run removes the staging leftovers of
    an interrupted one. A pair is rebuilt when ``full`` is true, when either
    file is missing, or when the older of the two is not strictly newer than
    ``archives / f"{demo}.npz"`` and the newest file under ``sources``
    (``__pycache__`` directories skipped). A full render then removes every
    other ``.mp4`` and ``.png`` directly inside ``media``. Returns the paths
    written, clip then poster, in the order written.

    An archive holding another demo's replay raises ``ValueError`` before that
    demo is drawn. An archive that changes or cannot be read while one of its
    pairs renders or is published raises ``ArchiveChanged`` after removing all
    of that demo's media; any other error while a pair is rebuilt also removes
    them when the archive has changed, then propagates.
    """
    _refuse_a_linked_media(media)
    encoder = find_encoder()
    report(f"encoder: {encoder} ({encoder_version(encoder)})")
    media.mkdir(parents=True, exist_ok=True)
    _remove_leftovers(media)
    newest_source = max(
        (p.stat().st_mtime_ns for p in _source_files(sources)), default=0
    )
    written: list[Path] = []
    for demo in demos:
        archive = archives / f"{demo}.npz"
        inputs_time = max(archive.stat().st_mtime_ns, newest_source)
        loaded: tuple[Replay, str] | None = None
        for theme in THEMES:
            clip, poster = media_paths(media, demo, theme)
            if not (full or _is_stale(clip, poster, inputs_time)):
                report(f"kept {clip.name}")
                continue
            if loaded is None:
                loaded = _load_demo(archive, demo)
            replay, digest = loaded
            unchanged = functools.partial(
                _require_unchanged, archive, digest, media, demo
            )
            started = time.perf_counter()
            try:
                frames = _rebuild_pair(replay, theme, clip, poster, encoder, unchanged)
                unchanged()
            except BaseException:
                if _has_changed(archive, digest):
                    _remove_demo_media(media, demo)
                raise
            seconds = time.perf_counter() - started
            report(f"rendered {clip.name} ({frames} frames, {seconds:.1f} s)")
            written += [clip, poster]
    if full:
        _remove_other_media(media, set(written), report)
    return written


def _rebuild_pair(
    replay: Replay,
    theme: str,
    clip: Path,
    poster: Path,
    encoder: str,
    unchanged: Callable[[], None],
) -> int:
    """Stage, check the archive is ``unchanged``, retire, publish; return frames."""
    staged_clip = clip.with_name(STAGED + clip.name)
    staged_poster = poster.with_name(STAGED + poster.name)
    try:
        frames = render_clip(replay, staged_clip, theme=theme, encoder=encoder)
        with style(theme) as colors:
            save_png(final_frame(replay, colors), staged_poster)
        unchanged()
    except BaseException:
        staged_clip.unlink(missing_ok=True)
        staged_poster.unlink(missing_ok=True)
        raise
    clip.unlink(missing_ok=True)
    poster.unlink(missing_ok=True)
    os.replace(staged_clip, clip)
    os.replace(staged_poster, poster)
    return frames


def _refuse_a_linked_media(media: Path) -> None:
    """Refuse a ``media`` whose last component is a symbolic link, dangling or not."""
    if media.is_symlink() or media.resolve() != media.parent.resolve() / media.name:
        raise ValueError(
            f"the media directory {media} is a symbolic link; the pre-render step "
            "removes files there, so it must be a real directory (remove the link)"
        )


def _load_demo(archive: Path, demo: str) -> tuple[Replay, str]:
    """Read ``archive`` once; return the replay of those bytes and their digest.

    Refuses the archive unless its replay is of ``demo``.
    """
    data = archive.read_bytes()
    replay = load_replay(archive, data)
    found = replay.meta["demo"]
    if found != demo:
        raise ValueError(
            f"{archive} holds a replay of {found!r}, not {demo!r}: "
            f"restore {demo}'s own archive (it may have been copied over)"
        )
    return replay, hashlib.sha256(data).hexdigest()


def _has_changed(archive: Path, digest: str) -> bool:
    """Whether ``archive`` no longer reads as the bytes ``digest`` names."""
    try:
        data = archive.read_bytes()
    except OSError:
        return True
    return hashlib.sha256(data).hexdigest() != digest


def _remove_demo_media(media: Path, demo: str) -> None:
    for theme in THEMES:
        for path in media_paths(media, demo, theme):
            path.unlink(missing_ok=True)


def _require_unchanged(archive: Path, digest: str, media: Path, demo: str) -> None:
    """Raise ``ArchiveChanged``, removing ``demo``'s media, if ``archive`` changed
    or cannot be read."""
    if not _has_changed(archive, digest):
        return
    _remove_demo_media(media, demo)
    raise ArchiveChanged(
        f"{archive} changed or could not be read while a {demo} pair was "
        f"rendering; every {demo} clip and poster was removed so the next run "
        "rebuilds them from the archive as it then is"
    )


def _is_stale(clip: Path, poster: Path, inputs_time: int) -> bool:
    """Either file missing, or the older not strictly newer than the inputs."""
    if not (clip.is_file() and poster.is_file()):
        return True
    older = min(clip.stat().st_mtime_ns, poster.stat().st_mtime_ns)
    return older <= inputs_time


def _source_files(sources: Path) -> Iterator[Path]:
    """Every regular file under ``sources``, outside ``__pycache__`` directories."""
    for directory, subdirectories, files in sources.walk():
        subdirectories[:] = [d for d in subdirectories if d != "__pycache__"]
        for name in files:
            path = directory / name
            if path.is_file():
                yield path


def _remove_leftovers(media: Path) -> None:
    """Remove the regular files directly in ``media`` named with ``STAGED``."""
    for path in media.iterdir():
        if path.name.startswith(STAGED) and path.is_file():
            path.unlink()


def _remove_other_media(
    media: Path, keep: set[Path], report: Callable[[str], None]
) -> None:
    """Remove each ``.mp4`` and ``.png`` directly in ``media`` not in ``keep``."""
    for path in sorted(media.iterdir()):
        if path.suffix in _MEDIA_SUFFIXES and path.is_file() and path not in keep:
            path.unlink()
            report(f"removed {path.name}")
