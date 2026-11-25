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

Each published pair has a record beside it, ``<demo>.<theme>.sha256``: the
SHA-256 of the archive bytes it was drawn from, written after the pair is in
place. Quarto publishes only the ``.mp4`` and ``.png`` files of the directory,
so the records stay out of the site. A pair whose record is missing or names
other bytes than the archive's is stale, whatever the times say.

Three inputs are refused rather than trusted. A media path that is itself a
symbolic link is refused before anything is created or removed, since the
cleanup would follow it into another directory. An archive whose replay is of
another demo is refused before anything is drawn. And each archive is read
once per run, before any of its pairs is judged: its time and digest decide
freshness, and any rebuilt pair is drawn from those same bytes. The file is
hashed again before each rebuilt pair is published and again after. If it
differs or cannot be read, every clip, poster and record of that demo is
removed and ``ArchiveChanged`` is raised; and when anything else fails while a
pair is rebuilt, the archive is checked the same way before the error
propagates. So no pair drawn from other content than the archive's survives to
look fresh: one replaced after a run's last check is caught by the records on
the next run.
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

_MEDIA_SUFFIXES = frozenset({".mp4", ".png", ".sha256"})


class ArchiveChanged(RuntimeError):
    """An archive changed while a pair drawn from it was being rendered."""


def media_paths(media: Path, demo: str, theme: str) -> tuple[Path, Path]:
    """The clip and the poster of ``demo`` in ``theme``, under ``media``."""
    return media / f"{demo}.{theme}.mp4", media / f"{demo}.{theme}.png"


def record_path(media: Path, demo: str, theme: str) -> Path:
    """The record of the archive digest ``demo``'s ``theme`` pair was drawn from."""
    return media / f"{demo}.{theme}.sha256"


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
    file or its record is missing, when the record names other bytes than the
    archive's, or when the older of the two files is not strictly newer than
    ``archives / f"{demo}.npz"`` and the newest file under ``sources``
    (``__pycache__`` directories skipped). A full render then removes every
    other ``.mp4``, ``.png`` and ``.sha256`` directly inside ``media``. Returns
    the paths written, clip then poster, in the order written.

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
        archive_time = archive.stat().st_mtime_ns
        data = archive.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        inputs_time = max(archive_time, newest_source)
        replay: Replay | None = None
        unchanged = functools.partial(_require_unchanged, archive, digest, media, demo)
        for theme in THEMES:
            clip, poster = media_paths(media, demo, theme)
            record = record_path(media, demo, theme)
            if not (full or _is_stale(clip, poster, record, inputs_time, digest)):
                report(f"kept {clip.name}")
                continue
            if replay is None:
                replay = _load_demo(archive, data, demo)
            started = time.perf_counter()
            try:
                frames = _rebuild_pair(replay, theme, clip, poster, encoder, unchanged)
                _write_record(record, digest)
                unchanged()
            except BaseException:
                if _has_changed(archive, digest):
                    _remove_demo_media(media, demo)
                raise
            seconds = time.perf_counter() - started
            report(f"rendered {clip.name} ({frames} frames, {seconds:.1f} s)")
            written += [clip, poster]
    if full:
        records = {path.with_suffix(".sha256") for path in written}
        _remove_other_media(media, set(written) | records, report)
    return written


def _rebuild_pair(
    replay: Replay,
    theme: str,
    clip: Path,
    poster: Path,
    encoder: str,
    unchanged: Callable[[], None],
) -> int:
    """Stage, check the archive is ``unchanged``, retire, publish; return frames.

    The pair's record is removed before the old pair is retired, so an
    interruption from then on leaves the pair stale.
    """
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
    clip.with_suffix(".sha256").unlink(missing_ok=True)
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


def _load_demo(archive: Path, data: bytes, demo: str) -> Replay:
    """The replay in ``data``, ``archive``'s bytes; refused unless of ``demo``."""
    replay = load_replay(archive, data)
    found = replay.meta["demo"]
    if found != demo:
        raise ValueError(
            f"{archive} holds a replay of {found!r}, not {demo!r}: "
            f"restore {demo}'s own archive (it may have been copied over)"
        )
    return replay


def _write_record(record: Path, digest: str) -> None:
    """Write ``digest`` to ``record`` whole: staged, then moved into place."""
    staged = record.with_name(STAGED + record.name)
    staged.write_text(f"{digest}\n")
    os.replace(staged, record)


def _has_changed(archive: Path, digest: str) -> bool:
    """Whether ``archive`` no longer reads as the bytes ``digest`` names."""
    try:
        data = archive.read_bytes()
    except OSError:
        return True
    return hashlib.sha256(data).hexdigest() != digest


def _remove_demo_media(media: Path, demo: str) -> None:
    for theme in THEMES:
        paths = (*media_paths(media, demo, theme), record_path(media, demo, theme))
        for path in paths:
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


def _is_stale(
    clip: Path, poster: Path, record: Path, inputs_time: int, digest: str
) -> bool:
    """Either file or the record missing, the record naming other bytes than
    ``digest``, or the older file not strictly newer than the inputs."""
    if not (clip.is_file() and poster.is_file() and record.is_file()):
        return True
    if record.read_text(errors="replace").strip() != digest:
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
    """Remove each ``.mp4``, ``.png`` and ``.sha256`` directly in ``media`` not
    in ``keep``."""
    for path in sorted(media.iterdir()):
        if path.suffix in _MEDIA_SUFFIXES and path.is_file() and path not in keep:
            path.unlink()
            report(f"removed {path.name}")
