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

Freshness is judged by content, never by time. Each run reads every demo's
archive once and every source file under ``sources`` once, and hashes them
into the demo's inputs digest. Each published pair has a record beside it,
``<demo>.<theme>.sha256``, holding the inputs digest it was drawn from,
written after the pair is in place and removed before an old pair is retired.
A pair is fresh only when its clip, poster and record all exist and the record
holds exactly the run's digest; anything else rebuilds it, from the same
archive bytes that were hashed. Quarto publishes only the ``.mp4`` and
``.png`` files of the directory, so the records stay out of the site.

The sources' snapshot must be of the code that runs. ``site/prerender.py``
takes it before importing ``clocks``, writes it to a private temporary
directory, imports the renderer from that copy and passes the snapshot as
``sources_snapshot``: the code that draws is the snapshot, whatever the live
tree or bytecode cached beside it does meanwhile. A caller that passes none
gets a snapshot taken when the call starts, after its own imports. Before a
rebuilt pair replaces the old one, and again before its record is written,
the live tree is read again; a difference raises ``SourcesChanged``, so a
pair drawn while the code changed is not certified: before publication the
old pair stays as it was, after it the pair is left without a record, stale.

The limit: an archive replaced while the step runs is caught, for each pair
this run rebuilds, by the check that runs again after that pair's final
rename; for a pair this run kept, it is caught by the next run, whose digest
no longer matches the record.

Three inputs are refused rather than trusted. A media path whose last
component is a symbolic link, or that resolves elsewhere than its parent's
entry of that name (``linked/..``, say), is refused before anything is created
or removed, since the cleanup would follow it into another directory. An
archive whose replay is of another demo is refused before that demo is drawn.
And an archive that differs from its snapshot, or cannot be read, when a
rebuilt pair is checked (before and after its publication, or when anything
fails while it is rebuilt) removes every clip, poster and record of that demo;
outside a failure it raises ``ArchiveChanged``.

No file of the step is written through a link: the staged poster and record
are created exclusively, the clip is moved into its staged name, and an output
that is a link is stale and replaced rather than kept.
"""

import functools
import hashlib
import os
import re
import shutil
import stat
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from clocks._clips import final_frame, render_clip
from clocks._replay import DEMOS, Replay, load_replay
from clocks._sources import read_sources
from clocks._style import THEMES, save_png, style
from clocks._video import encoder_version, find_encoder

CLIP_DEMOS: tuple[str, ...] = tuple(name for name in DEMOS if name != "demo_density")
PACKAGE = Path(__file__).resolve().parent
STAGED = ".staged-"

_MEDIA_SUFFIXES = frozenset({".mp4", ".png"})
# A record's reserved name: <demo>.<theme>.sha256, the theme one of THEMES.
_RECORD_NAME = re.compile(
    r"[^.]+\.(?:" + "|".join(map(re.escape, THEMES)) + r")\.sha256", re.ASCII
)


_LINKED_MEDIA = (
    "the media directory {media} is {where}; the pre-render step removes "
    "files there, so it must be a real directory named directly"
)


class SourcesChanged(RuntimeError):
    """The source tree differs from the snapshot its records would name."""


class ArchiveChanged(RuntimeError):
    """An archive changed while a pair drawn from it was being rendered."""


def media_paths(media: Path, demo: str, theme: str) -> tuple[Path, Path]:
    """The clip and the poster of ``demo`` in ``theme``, under ``media``."""
    return media / f"{demo}.{theme}.mp4", media / f"{demo}.{theme}.png"


def record_path(media: Path, demo: str, theme: str) -> Path:
    """The record of the inputs digest ``demo``'s ``theme`` pair was drawn from."""
    return media / f"{demo}.{theme}.sha256"


def render_site_media(
    archives: Path,
    media: Path,
    *,
    full: bool,
    demos: Sequence[str] = CLIP_DEMOS,
    sources: Path = PACKAGE,
    sources_snapshot: Sequence[tuple[str, bytes]] | None = None,
    check_code: Callable[[], None] | None = None,
    report: Callable[[str], None] = print,
) -> list[Path]:
    """Draw each demo's clip and poster, in every theme, into ``media``.

    A ``media`` that is a symbolic link, or resolves elsewhere, raises
    ``ValueError`` first. The encoder
    is found next: when there is none, ``EncoderNotFound`` propagates before
    anything is created or removed. Every run removes every entry whose name
    starts with ``STAGED``, a dangling link included. A pair is rebuilt when
    ``full`` is true or when it is not fresh: its clip, poster or record
    missing, or the record not holding exactly the demo's inputs digest (the
    SHA-256 of ``archives / f"{demo}.npz"`` and of every file under ``sources``
    outside ``__pycache__`` directories, with their relative paths). A full
    render then removes every other ``.mp4`` and ``.png`` directly inside
    ``media``, and every record of the reserved ``<demo>.<theme>.sha256`` form
    it did not just write. Returns the paths written, clip then poster, in the
    order written.

    ``sources_snapshot``, when given, is what ``read_sources(sources)`` returned
    before anything of ``clocks`` was imported; the records name it, and the
    tree must still read the same before each rebuilt pair is published and
    before its record is written (``SourcesChanged`` otherwise). A source
    tree that cannot be read raises the ``OSError``.
    ``check_code``, when given, is called for each rebuilt pair twice, and
    whatever it raises propagates (``site/prerender.py`` checks there that
    every clocks module loaded came from its copy of the snapshot). The first
    call comes once the pair is staged, before anything of the old pair is
    replaced: a failure there leaves the old pair and its record as they were.
    The second comes once the new pair is in place, before its record is
    written: a failure there leaves the new pair without a record, so the
    next run rebuilds it.

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
    if sources_snapshot is None:
        source_items = read_sources(sources)
    else:
        source_items = list(sources_snapshot)
    written: list[Path] = []
    for demo in demos:
        archive = archives / f"{demo}.npz"
        data = archive.read_bytes()
        archive_digest = hashlib.sha256(data).hexdigest()
        inputs = inputs_digest(data, source_items)
        replay: Replay | None = None
        unchanged = functools.partial(
            _require_unchanged, archive, archive_digest, media, demo
        )
        checks = functools.partial(
            _before_publishing, unchanged, sources, source_items, check_code
        )
        for theme in THEMES:
            clip, poster = media_paths(media, demo, theme)
            record = record_path(media, demo, theme)
            if not (full or _is_stale(clip, poster, record, inputs)):
                report(f"kept {clip.name}")
                continue
            if replay is None:
                replay = _load_demo(archive, data, demo)
            started = time.perf_counter()
            try:
                frames = _rebuild_pair(replay, theme, clip, poster, encoder, checks)
                _require_sources(sources, source_items)
                if check_code is not None:
                    check_code()
                _write_record(record, inputs)
                unchanged()
            except BaseException:
                if _has_changed(archive, archive_digest):
                    _remove_demo_media(media, demo)
                raise
            seconds = time.perf_counter() - started
            report(f"rendered {clip.name} ({frames} frames, {seconds:.1f} s)")
            written += [clip, poster]
    if full:
        records = {path.with_suffix(".sha256") for path in written}
        _remove_other_media(media, set(written) | records, report)
    return written


def inputs_digest(archive: bytes, sources: Sequence[tuple[str, bytes]]) -> str:
    """The SHA-256 of an archive's bytes and each source's relative path and bytes.

    Every part is framed by its length, so no two different inputs feed the
    hash the same stream.
    """
    digest = hashlib.sha256()
    for name, data in (("archive", archive), *sources):
        label = name.encode()
        digest.update(f"{len(label)}:".encode() + label)
        digest.update(f"{len(data)}:".encode() + data)
    return digest.hexdigest()


def _before_publishing(
    unchanged: Callable[[], None],
    sources: Path,
    source_items: list[tuple[str, bytes]],
    check_code: Callable[[], None] | None,
) -> None:
    """Every check a staged pair must pass before it replaces the old one."""
    unchanged()
    _require_sources(sources, source_items)
    if check_code is not None:
        check_code()


def _require_sources(sources: Path, snapshot: list[tuple[str, bytes]]) -> None:
    """Raise ``SourcesChanged`` unless ``sources`` still reads as ``snapshot``."""
    if read_sources(sources) != snapshot:
        raise SourcesChanged(
            f"the source tree under {sources} changed during the run, so the code "
            "that drew this pair is not known; it is not certified, and the next "
            "run rebuilds it"
        )


def _rebuild_pair(
    replay: Replay,
    theme: str,
    clip: Path,
    poster: Path,
    encoder: str,
    checks: Callable[[], None],
) -> int:
    """Stage, run the ``checks``, retire, publish; return frames.

    The pair's record is removed before the old pair is retired, so an
    interruption from then on leaves the pair stale.
    """
    staged_clip = clip.with_name(STAGED + clip.name)
    staged_poster = poster.with_name(STAGED + poster.name)
    try:
        frames = render_clip(replay, staged_clip, theme=theme, encoder=encoder)
        with style(theme) as colors, staged_poster.open("xb") as file:
            save_png(final_frame(replay, colors), file)
        checks()
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
    """Refuse, with ``ValueError``, a ``media`` whose last component is a
    symbolic link (read with ``lstat``, dangling or looping or not), whose path
    cannot be resolved or resolves to something that cannot be followed (a
    link loop anywhere on it, on any Python), or which resolves elsewhere than
    its parent's entry of its name (``linked/..`` names the parent of the
    link's target). A ``media`` not made yet is fine."""
    try:
        if stat.S_ISLNK(os.lstat(media).st_mode):
            where = f"a symbolic link to {os.readlink(media)}"
            raise ValueError(_LINKED_MEDIA.format(media=media, where=where))
    except FileNotFoundError:
        pass
    except OSError as error:
        raise ValueError(_unfollowable(media)) from error
    try:
        resolved = media.resolve()
        expected = media.parent.resolve() / media.name
    except (OSError, RuntimeError) as error:  # RuntimeError: 3.12 on a loop
        raise ValueError(_unfollowable(media)) from error
    try:
        os.stat(resolved)  # 3.13 and later resolve past a loop without raising
    except FileNotFoundError:
        pass
    except OSError as error:
        raise ValueError(_unfollowable(media)) from error
    if resolved != expected:
        where = f"a path that resolves elsewhere ({resolved})"
        raise ValueError(_LINKED_MEDIA.format(media=media, where=where))


def _unfollowable(media: Path) -> str:
    return _LINKED_MEDIA.format(media=media, where="a path that cannot be resolved")


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
    """Write ``digest`` to ``record`` whole: staged, then moved into place.

    The staged file is created exclusively, so it is never written through a
    link (or any other entry) already at its name.
    """
    staged = record.with_name(STAGED + record.name)
    with staged.open("x", encoding="utf-8") as file:
        file.write(f"{digest}\n")
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


def _is_stale(clip: Path, poster: Path, record: Path, inputs: str) -> bool:
    """Either file or the record missing or a link, or the record not exactly
    ``inputs``."""
    paths = (clip, poster, record)
    if any(path.is_symlink() or not path.is_file() for path in paths):
        return True
    return record.read_text(encoding="utf-8", errors="replace") != f"{inputs}\n"


def _remove_leftovers(media: Path) -> None:
    """Remove every entry directly in ``media`` named with ``STAGED``.

    A link is removed itself, dangling or not, and never followed; a directory
    is removed with what it holds.
    """
    for path in media.iterdir():
        if not path.name.startswith(STAGED):
            continue
        if not path.is_symlink() and path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _remove_other_media(
    media: Path, keep: set[Path], report: Callable[[str], None]
) -> None:
    """Remove each ``.mp4`` and ``.png``, and each record of the reserved form,
    directly in ``media`` and not in ``keep``: a regular file, or a link of such
    a name, dangling or not, removed itself. A directory is left alone."""
    for path in sorted(media.iterdir()):
        if path in keep or not (path.is_symlink() or path.is_file()):
            continue
        if path.suffix in _MEDIA_SUFFIXES or _RECORD_NAME.fullmatch(path.name):
            path.unlink()
            report(f"removed {path.name}")
