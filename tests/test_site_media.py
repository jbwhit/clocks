"""The site's pre-render step: clips and posters staged, then published as pairs."""

import ast
import functools
import importlib.util
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.image
import numpy as np
import pytest
from _clip_replays import edited, first_pixel, shortest
from _figure_artists import one, png_size
from matplotlib.colors import to_hex, to_rgba

import clocks._site_media
from clocks._clips import CLIP_HEIGHTS, final_frame, render_clip
from clocks._demos import demo_2d
from clocks._replay import DEMOS, Replay, load_replay, save_replay
from clocks._site_media import CLIP_DEMOS, render_site_media
from clocks._sources import read_sources
from clocks._style import THEMES, palette, save_png
from clocks._video import EncoderError, EncoderNotFound, StreamInfo, probe

ROOT = Path(__file__).resolve().parents[1]
DEMO = "demo_2d"
OLD = ("old clip", "old poster")

PINNED = '''"""Quarto pre-render step: draw the site's clips and posters into media/.

The package's source tree is read once, before any clocks module is imported,
and written to a private temporary directory; the renderer is imported from
that copy. So the code that draws is exactly the snapshot each pair's record
names, whatever happens to the live tree, or to bytecode cached beside it,
meanwhile. The live tree is read again before each record is written, so an
edit that persists is caught too.
"""

import importlib.util
import os
import sys
import tempfile
import types
from collections.abc import Callable
from pathlib import Path

import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")

SITE = Path(__file__).resolve().parent


def main(
    archives: Path, media: Path, *, full: bool, demos: tuple[str, ...] | None = None
) -> None:
    """Render the media from a private copy of the package; restore what it set.

    Refuses to run when any ``clocks`` module is already imported: that code,
    not the snapshot, would draw.
    """
    loaded = sorted(n for n in sys.modules if n == "clocks" or n.startswith("clocks."))
    if loaded:
        raise RuntimeError(
            f"clocks is already imported ({', '.join(loaded)}): run the "
            "pre-render step in a fresh process, so the snapshot is what draws"
        )
    spec = importlib.util.find_spec("clocks")  # locates the package, imports nothing
    package = Path(spec.submodule_search_locations[0]).resolve()
    saved_path = list(sys.path)
    saved_fonts = list(font_manager.fontManager.ttflist)
    with tempfile.TemporaryDirectory(prefix="clocks-snapshot-") as root:
        try:
            snapshot = _snapshot(package)
            _write_copy(Path(root) / "clocks", snapshot)
            render_site_media = _import_renderer(Path(root))
            options = {} if demos is None else {"demos": demos}
            render_site_media(
                archives,
                media,
                full=full,
                sources=package,
                sources_snapshot=snapshot,
                **options,
            )
        finally:
            for name in [n for n in sys.modules if n.split(".")[0] == "clocks"]:
                del sys.modules[name]
            sys.path[:] = saved_path
            _restore_fonts(saved_fonts)


def _restore_fonts(saved: list[font_manager.FontEntry]) -> None:
    """Forget the fonts the copy registered: their files are about to go."""
    font_manager.fontManager.ttflist[:] = saved
    cached = getattr(font_manager.fontManager, "_findfont_cached", None)
    if cached is not None:  # lookups made while the copy's fonts were listed
        cached.cache_clear()


def _snapshot(package: Path) -> list[tuple[str, bytes]]:
    """Read the package with its own ``_sources.py``, compiled here from its
    bytes: no bytecode cache is consulted, and no clocks module is imported."""
    path = package / "_sources.py"
    module = types.ModuleType("clocks_sources")
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module.read_sources(package)


def _write_copy(copy: Path, snapshot: list[tuple[str, bytes]]) -> None:
    for name, data in snapshot:
        target = copy / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def _import_renderer(root: Path) -> Callable[..., object]:
    """Import ``render_site_media`` from the copy under ``root``; check every
    clocks module it loaded came from there."""
    sys.path.insert(0, str(root))
    from clocks._site_media import render_site_media

    copy = root / "clocks"
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] != "clocks":
            continue
        origin = Path(getattr(module, "__file__", None) or "")
        if not origin.is_relative_to(copy):
            raise RuntimeError(f"{name} was imported from {origin}, not the snapshot")
    return render_site_media


if __name__ == "__main__":
    main(
        SITE.parent / "assets" / "replays",
        SITE / "media",
        full=os.environ.get("QUARTO_PROJECT_RENDER_ALL") == "1",
    )
'''


@functools.cache
def _recorded() -> Replay:
    return demo_2d.record(n_observations=2, n_particles=16)


@dataclass
class Site:
    """An archive directory, a media directory, a source tree and the report."""

    archives: Path
    media: Path
    sources: Path
    lines: list[str] = field(default_factory=list)

    @property
    def archive(self) -> Path:
        return self.archives / f"{DEMO}.npz"

    @property
    def source(self) -> Path:
        return self.sources / "a.py"

    def path(self, name: str) -> Path:
        return self.media / name

    def outputs(self) -> list[Path]:
        """The four outputs, clip then poster, light then dark."""
        return [path for theme in THEMES for path in self.pair(theme)]

    def pair(self, theme: str) -> list[Path]:
        return [self.path(f"{DEMO}.{theme}.mp4"), self.path(f"{DEMO}.{theme}.png")]

    def record(self, theme: str) -> Path:
        """The unpublished note of the archive digest the pair was drawn from."""
        return self.path(f"{DEMO}.{theme}.sha256")

    def records(self) -> list[Path]:
        return [self.record(theme) for theme in THEMES]

    def render(self, *, full: bool) -> list[Path]:
        return render_site_media(
            self.archives,
            self.media,
            full=full,
            demos=(DEMO,),
            sources=self.sources,
            report=self.lines.append,
        )

    def digest(self) -> str:
        """The inputs digest of the fixture's archive and source tree, as read now."""
        return clocks._site_media.inputs_digest(
            self.archive.read_bytes(), [("a.py", self.source.read_bytes())]
        )

    def put_old_pair(self) -> None:
        """An old light pair with a record of the current inputs: fresh."""
        self.media.mkdir(parents=True, exist_ok=True)
        clip, poster = self.pair("light")
        clip.write_bytes(OLD[0].encode())
        poster.write_bytes(OLD[1].encode())
        self.record("light").write_text(f"{self.digest()}\n")

    def old_pair_is_fresh(self) -> bool:
        return self.record("light").read_text() == f"{self.digest()}\n"

    def old_pair(self) -> tuple[str, str]:
        clip, poster = self.pair("light")
        return clip.read_text(), poster.read_text()

    def staged_names(self) -> list[str]:
        return sorted(p.name for p in self.media.iterdir() if p.name.startswith("."))


@pytest.fixture
def site(tmp_path: Path) -> Site:
    archives = tmp_path / "archives"
    archives.mkdir()
    save_replay(archives / f"{DEMO}.npz", _recorded())
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "a.py").write_text("x = 1\n")
    return Site(archives, tmp_path / "media", sources)


def _spy(
    monkeypatch: pytest.MonkeyPatch, name: str, real: Callable | None = None
) -> list[dict[str, object]]:
    """Wrap ``clocks._site_media.<name>``; record each call's arguments and result.

    Each record maps the parameter names to the arguments bound to them, and
    ``"returned"`` to what the call returned.
    """
    target = real or getattr(clocks._site_media, name)
    signature = inspect.signature(target)
    calls: list[dict[str, object]] = []

    def spy(*args, **kwargs):
        call = dict(signature.bind(*args, **kwargs).arguments)
        calls.append(call)
        call["returned"] = target(*args, **kwargs)
        return call["returned"]

    monkeypatch.setattr(clocks._site_media, name, spy)
    return calls


def _background_bytes(theme: str) -> np.ndarray:
    return np.array([round(255 * c) for c in to_rgba(palette(theme).background)[:3]])


def _poster_pixels(path: Path) -> np.ndarray:
    return np.round(255 * matplotlib.image.imread(path)).astype(int)


def test_a_full_render_writes_each_clip_and_poster(site: Site) -> None:
    written = site.render(full=True)
    assert written == site.outputs()
    assert sorted(site.media.iterdir()) == sorted(site.outputs() + site.records())
    for record in site.records():
        assert record.read_text() == f"{site.digest()}\n"
    for theme in THEMES:
        clip, poster = site.pair(theme)
        background = _background_bytes(theme)
        assert probe(clip) == StreamInfo("h264", "yuv420p", 1500, 900, 20.0, 92, True)
        assert png_size(poster) == (1500, 900)
        assert np.array_equal(_poster_pixels(poster)[0, 0, :3], background)
        decoded = np.frombuffer(first_pixel(clip), dtype=np.uint8).astype(int)
        assert np.all(np.abs(decoded - background) <= 16), (theme, decoded)


def test_a_full_render_rewrites_fresh_media(
    site: Site, monkeypatch: pytest.MonkeyPatch
) -> None:
    site.render(full=True)
    calls = _spy(monkeypatch, "render_clip")
    assert site.render(full=True) == site.outputs()
    assert len(calls) == 2


def test_a_full_render_removes_other_media_and_nothing_else(site: Site) -> None:
    site.media.mkdir()
    for name in (
        "old.mp4",
        "old.png",
        "demo_retired.dark.sha256",
        "checksums.sha256",
        "demo_2d.sepia.sha256",
        "notes.txt",
        "other.tmp",
    ):
        site.path(name).write_text(name)
    site.path("sub").mkdir()
    site.path("sub/keep.mp4").write_text("keep")
    site.render(full=True)
    left = {p.relative_to(site.media).as_posix() for p in site.media.rglob("*")}
    assert left == {
        *(p.name for p in site.outputs() + site.records()),
        "checksums.sha256",
        "demo_2d.sepia.sha256",
        "notes.txt",
        "other.tmp",
        "sub",
        "sub/keep.mp4",
    }
    assert "removed old.mp4" in site.lines
    assert "removed old.png" in site.lines
    assert "removed demo_retired.dark.sha256" in site.lines


@pytest.mark.parametrize("full", [True, False], ids=["full", "partial"])
def test_staging_leftovers_are_removed_on_any_render(site: Site, full: bool) -> None:
    site.render(full=True)
    leftovers = [
        site.path(".staged-demo_2d.light.mp4"),
        site.path(".staged-demo_2d.light.mp4.ab12.part"),
    ]
    for path in leftovers:
        path.write_bytes(b"left over")
    site.render(full=full)
    assert not any(path.exists() for path in leftovers)
    assert site.staged_names() == []


def test_a_partial_render_keeps_fresh_media(
    site: Site, monkeypatch: pytest.MonkeyPatch
) -> None:
    site.render(full=True)
    calls = _spy(monkeypatch, "render_clip")
    site.lines.clear()
    assert site.render(full=False) == []
    assert "kept demo_2d.light.mp4" in site.lines
    assert "kept demo_2d.dark.mp4" in site.lines
    assert calls == []


@contextmanager
def _time_held(path: Path) -> Iterator[None]:
    """Change ``path`` inside the block; its modification time is put back after."""
    before = path.stat()
    yield
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))


def _archive_touched(site: Site) -> None:
    os.utime(site.archive, (4_000_000_000, 4_000_000_000))


def _source_touched(site: Site) -> None:
    os.utime(site.source, (4_000_000_000, 4_000_000_000))


def _archive_replaced(site: Site) -> None:
    _replace_archive(site)


def _source_rewritten(site: Site) -> None:
    with _time_held(site.source):
        site.source.write_text("x = 2\n")


def _source_renamed(site: Site) -> None:
    site.source.rename(site.sources / "b.py")


def _poster_missing(site: Site) -> None:
    site.pair("light")[1].unlink()


def _media_older(site: Site) -> None:
    for path in site.outputs():
        os.utime(path, (500, 500))


def _stray_file(site: Site) -> None:
    site.path("old.mp4").write_text("stray")


@pytest.mark.parametrize(
    ("change", "rebuilt"),
    [
        (_archive_touched, ()),
        (_source_touched, ()),
        (_media_older, ()),
        (_archive_replaced, ("light", "dark")),
        (_source_rewritten, ("light", "dark")),
        (_source_renamed, ("light", "dark")),
        (_poster_missing, ("light",)),
        (_stray_file, ()),
    ],
    ids=[
        "archive_touched_kept",
        "source_touched_kept",
        "media_older_kept",
        "archive_replaced_same_time",
        "source_rewritten_same_time",
        "source_renamed",
        "poster_missing",
        "stray_file_kept",
    ],
)
def test_a_partial_render_rebuilds_what_is_stale(
    site: Site, change: Callable[[Site], None], rebuilt: tuple[str, ...]
) -> None:
    """Freshness is the record's digest of the inputs' bytes: times never count."""
    site.render(full=True)
    change(site)
    written = site.render(full=False)
    assert written == [path for theme in rebuilt for path in site.pair(theme)]
    if change is _stray_file:
        assert site.path("old.mp4").read_text() == "stray"


def test_a_cached_bytecode_file_does_not_make_media_stale(site: Site) -> None:
    site.render(full=True)
    cached = site.sources / "__pycache__" / "a.pyc"
    cached.parent.mkdir()
    cached.write_bytes(b"cached")
    assert site.render(full=False) == []


def test_no_encoder_stops_before_anything_is_touched(
    site: Site, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_encoder() -> str:
        raise EncoderNotFound("none")

    monkeypatch.setattr(clocks._site_media, "find_encoder", no_encoder)
    with pytest.raises(EncoderNotFound, match="none"):
        site.render(full=True)
    assert not site.media.exists()

    site.put_old_pair()
    site.path(".staged-x").write_text("x")
    with pytest.raises(EncoderNotFound, match="none"):
        site.render(full=True)
    assert site.old_pair() == OLD
    assert site.path(".staged-x").read_text() == "x"


def test_a_failed_encode_leaves_a_fresh_old_pair_alone(
    site: Site, monkeypatch: pytest.MonkeyPatch
) -> None:
    site.put_old_pair()

    def failing(replay, path, *, theme="light", encoder=None):
        Path(path).write_bytes(b"partial")
        raise EncoderError("the encoder failed")

    monkeypatch.setattr(clocks._site_media, "render_clip", failing)
    with pytest.raises(EncoderError, match="the encoder failed"):
        site.render(full=True)
    assert site.old_pair() == OLD
    assert site.old_pair_is_fresh()
    assert site.staged_names() == []


def test_a_failed_poster_leaves_the_old_pair_alone(
    site: Site, monkeypatch: pytest.MonkeyPatch
) -> None:
    site.put_old_pair()

    def failing(fig, file):
        file.write(b"partial")
        raise OSError("disk")

    monkeypatch.setattr(clocks._site_media, "save_png", failing)
    with pytest.raises(OSError, match="disk"):
        site.render(full=True)
    assert site.old_pair() == OLD
    assert site.old_pair_is_fresh()
    assert site.staged_names() == []


def test_an_interrupted_publication_leaves_a_file_missing(
    site: Site, monkeypatch: pytest.MonkeyPatch
) -> None:
    site.put_old_pair()
    light_clip, light_poster = site.pair("light")
    real_replace = os.replace
    destinations: list[Path] = []

    def interrupted(source, destination, *args, **kwargs):
        destinations.append(Path(destination))
        if Path(destination) == light_poster:
            raise OSError("interrupted")
        return real_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(clocks._site_media.os, "replace", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        site.render(full=True)
    monkeypatch.setattr(clocks._site_media.os, "replace", real_replace)

    assert light_clip in destinations
    assert destinations.index(light_clip) < destinations.index(light_poster)
    assert probe(light_clip).frames == 92
    assert not light_poster.exists()
    assert site.render(full=False) == site.outputs()


@pytest.mark.parametrize("dangling", [False, True], ids=["to_a_directory", "dangling"])
def test_a_symlinked_media_directory_is_refused_before_anything_is_touched(
    site: Site, tmp_path: Path, dangling: bool
) -> None:
    """A link at the media path could send the cleanup into another directory."""
    target = tmp_path / "shared"
    if not dangling:
        target.mkdir()
        for name in ("unrelated.png", "unrelated.mp4", ".staged-x"):
            (target / name).write_text(name)
    site.media.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link") as raised:
        site.render(full=True)

    assert str(site.media) in str(raised.value)
    assert site.media.is_symlink()
    if dangling:
        assert not target.exists()
    else:
        assert sorted(p.name for p in target.iterdir()) == [
            ".staged-x",
            "unrelated.mp4",
            "unrelated.png",
        ]
        assert (target / "unrelated.png").read_text() == "unrelated.png"


def test_a_media_directory_reached_through_a_linked_parent_is_accepted(
    site: Site, tmp_path: Path
) -> None:
    """Only the media path's own last component is refused when it is a link."""
    real = tmp_path / "real-site"
    real.mkdir()
    linked = tmp_path / "linked-site"
    linked.symlink_to(real, target_is_directory=True)
    written = render_site_media(
        site.archives,
        linked / "media",
        full=True,
        demos=(DEMO,),
        sources=site.sources,
        report=site.lines.append,
    )
    assert [p.name for p in written] == [p.name for p in site.outputs()]
    assert (real / "media" / f"{DEMO}.light.mp4").is_file()


def test_an_archive_of_another_demo_is_refused_before_drawing(
    site: Site, monkeypatch: pytest.MonkeyPatch
) -> None:
    """demo_1d's archive saved under demo_2d's name draws nothing for demo_2d."""
    save_replay(site.archive, shortest("demo_1d"))
    site.put_old_pair()
    clips = _spy(monkeypatch, "render_clip", render_clip)

    with pytest.raises(ValueError, match="'demo_1d'.*'demo_2d'") as raised:
        site.render(full=True)

    assert str(site.archive) in str(raised.value)
    assert clips == []
    assert site.old_pair() == OLD
    assert site.staged_names() == []


@pytest.mark.parametrize("during", [0, 1], ids=["light_pair", "dark_pair"])
def test_an_archive_replaced_while_a_pair_renders_leaves_the_demo_stale(
    site: Site, monkeypatch: pytest.MonkeyPatch, during: int
) -> None:
    """No pair of the demo survives to look fresh after its archive changes.

    The replacement keeps the old archive's modification time: only the
    content shows the change.
    """
    site.render(full=True)
    base = load_replay(site.archive)
    changed = edited(base, cloud=np.array(base.arrays["cloud"]) + 0.5)
    calls = 0

    def replacing(replay, path, *, theme="light", encoder=None):
        nonlocal calls
        if calls == during:
            with _time_held(site.archive):
                save_replay(site.archive, changed)
        calls += 1
        return render_clip(replay, path, theme=theme, encoder=encoder)

    monkeypatch.setattr(clocks._site_media, "render_clip", replacing)
    with pytest.raises(clocks._site_media.ArchiveChanged, match="changed") as raised:
        site.render(full=True)
    monkeypatch.setattr(clocks._site_media, "render_clip", render_clip)

    assert str(site.archive) in str(raised.value)
    assert calls == during + 1
    assert [p for p in site.outputs() if p.exists()] == []
    assert site.staged_names() == []
    assert site.render(full=False) == site.outputs()


def _replace_archive(site: Site) -> None:
    """Save a different replay at the archive, keeping its modification time."""
    base = load_replay(site.archive)
    with _time_held(site.archive):
        save_replay(
            site.archive, edited(base, cloud=np.array(base.arrays["cloud"]) + 0.5)
        )


def _nothing_kept(site: Site) -> None:
    """No output of the demo is left for a partial render to keep as fresh."""
    assert [p for p in site.outputs() + site.records() if p.exists()] == []
    assert site.staged_names() == []
    assert site.render(full=False) == site.outputs()


@pytest.mark.parametrize("renamed", ["poster", "record"])
@pytest.mark.parametrize("theme", THEMES)
def test_an_archive_replaced_at_a_final_rename_leaves_the_demo_stale(
    site: Site, monkeypatch: pytest.MonkeyPatch, theme: str, renamed: str
) -> None:
    """Replaced as the pair's poster is moved into place, after every check
    that runs before publication, the archive still marks the demo stale.

    Only ``theme``'s pair is stale and the render is partial, so the other
    pair is kept and no later check of this run can catch the replacement:
    only the check after the final rename can.
    """
    site.render(full=True)
    _, poster = site.pair(theme)
    poster.unlink()
    target = poster if renamed == "poster" else site.record(theme)
    real_replace = os.replace

    def replacing(source, destination, *args, **kwargs):
        if Path(destination) == target:
            _replace_archive(site)
        return real_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(clocks._site_media.os, "replace", replacing)
    with pytest.raises(clocks._site_media.ArchiveChanged) as raised:
        site.render(full=False)
    monkeypatch.setattr(clocks._site_media.os, "replace", real_replace)

    assert str(site.archive) in str(raised.value)
    _nothing_kept(site)


@pytest.mark.parametrize("theme", THEMES)
def test_an_archive_gone_at_the_recheck_leaves_the_demo_stale(
    site: Site, monkeypatch: pytest.MonkeyPatch, theme: str
) -> None:
    """The archive disappears once the pair is staged, just before the check
    that precedes publication, which must stop the pair being published; it
    then comes back unchanged."""
    site.render(full=True)
    kept = site.archive.read_bytes()
    real_replace = os.replace
    published: list[Path] = []

    def vanishing(fig, file):
        save_png(fig, file)
        if Path(file.name).name == f".staged-{DEMO}.{theme}.png":
            site.archive.unlink()

    def recording(source, destination, *args, **kwargs):
        published.append(Path(destination))
        return real_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(clocks._site_media, "save_png", vanishing)
    monkeypatch.setattr(clocks._site_media.os, "replace", recording)
    with pytest.raises(clocks._site_media.ArchiveChanged) as raised:
        site.render(full=True)
    monkeypatch.setattr(clocks._site_media, "save_png", save_png)
    monkeypatch.setattr(clocks._site_media.os, "replace", real_replace)

    assert not set(site.pair(theme)) & set(published), published
    assert str(site.archive) in str(raised.value)
    site.archive.write_bytes(kept)
    _nothing_kept(site)


@pytest.mark.parametrize("during", [0, 1], ids=["light_pair", "dark_pair"])
def test_an_encoder_failure_after_a_replacement_leaves_the_demo_stale(
    site: Site, monkeypatch: pytest.MonkeyPatch, during: int
) -> None:
    """The encoder's error propagates, and no output of the old content stays."""
    site.render(full=True)
    calls = 0

    def failing(replay, path, *, theme="light", encoder=None):
        nonlocal calls
        calls += 1
        if calls <= during:
            return render_clip(replay, path, theme=theme, encoder=encoder)
        _replace_archive(site)
        raise EncoderError("the encoder failed")

    monkeypatch.setattr(clocks._site_media, "render_clip", failing)
    with pytest.raises(EncoderError, match="the encoder failed"):
        site.render(full=True)
    monkeypatch.setattr(clocks._site_media, "render_clip", render_clip)

    _nothing_kept(site)


def test_the_replay_drawn_is_the_bytes_checked(
    site: Site, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The archive is swapped for another replay while it is loaded and swapped
    back, at its old time, before any check: the pair must show the first."""
    original = site.archive.read_bytes()
    base = load_replay(site.archive)
    other = edited(base, cloud=np.array(base.arrays["cloud"]) + 0.5)
    real_load = clocks._site_media.load_replay
    drawn: list[np.ndarray] = []

    def swapping(*args, **kwargs):
        with _time_held(site.archive):
            save_replay(site.archive, other)
        try:
            return real_load(*args, **kwargs)
        finally:
            with _time_held(site.archive):
                site.archive.write_bytes(original)

    def reading(fig, file):
        (cloud,) = [ax for ax in fig.axes if ax.get_gid() == "map"]
        drawn.append(np.array(one(cloud, "dots").get_xydata()))
        save_png(fig, file)

    monkeypatch.setattr(clocks._site_media, "load_replay", swapping)
    monkeypatch.setattr(clocks._site_media, "save_png", reading)
    site.render(full=True)

    expected = base.arrays["cloud"][-1][:, :2]
    assert len(drawn) == len(THEMES)
    for found in drawn:
        np.testing.assert_allclose(found, expected, rtol=1e-12, atol=1e-12)


def test_an_archive_replaced_after_a_pair_is_kept_leaves_the_demo_stale(
    site: Site,
) -> None:
    """The light pair is kept; the archive is then replaced at its old time,
    before the stale dark pair is drawn. No mixture of the two may survive."""
    site.render(full=True)
    site.pair("dark")[1].unlink()

    def report(line: str) -> None:
        site.lines.append(line)
        if line == f"kept {DEMO}.light.mp4":
            _replace_archive(site)

    with pytest.raises(clocks._site_media.ArchiveChanged) as raised:
        render_site_media(
            site.archives,
            site.media,
            full=False,
            demos=(DEMO,),
            sources=site.sources,
            report=report,
        )

    assert str(site.archive) in str(raised.value)
    _nothing_kept(site)


def test_a_pair_drawn_from_another_archive_version_is_not_kept(
    site: Site, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replaced before a partial render at its old time, with the dark pair
    stale: the light pair, drawn from the old bytes, is rebuilt too, from the
    new ones."""
    site.render(full=True)
    _replace_archive(site)
    site.pair("dark")[1].unlink()
    dots: list[np.ndarray] = []

    def reading(fig, file):
        (cloud,) = [ax for ax in fig.axes if ax.get_gid() == "map"]
        dots.append(np.array(one(cloud, "dots").get_xydata()))
        save_png(fig, file)

    monkeypatch.setattr(clocks._site_media, "save_png", reading)
    assert site.render(full=False) == site.outputs()

    expected = load_replay(site.archive).arrays["cloud"][-1][:, :2]
    assert len(dots) == len(THEMES)
    for found in dots:
        np.testing.assert_allclose(found, expected, rtol=1e-12, atol=1e-12)
    assert [r.read_text() for r in site.records()] == [f"{site.digest()}\n"] * 2
    assert site.render(full=False) == []


@pytest.mark.parametrize(
    "spoil",
    [lambda path: path.unlink(), lambda path: path.write_text("0" * 64 + "\n")],
    ids=["missing", "other_digest"],
)
def test_a_pair_without_its_archive_record_is_stale(
    site: Site, spoil: Callable[[Path], None]
) -> None:
    site.render(full=True)
    spoil(site.record("light"))
    assert site.render(full=False) == site.pair("light")


@pytest.mark.parametrize("full", [True, False], ids=["full", "partial"])
def test_a_dangling_staging_link_is_removed_and_never_followed(
    site: Site, tmp_path: Path, full: bool
) -> None:
    """A link at a staged name, pointing outside ``media``, writes nothing there."""
    site.render(full=True)
    site.pair("light")[1].unlink()
    outside = tmp_path / "outside" / "record"
    outside.parent.mkdir()
    for name in (f".staged-{DEMO}.light.sha256", ".staged-anything"):
        site.path(name).symlink_to(outside)

    site.render(full=full)

    assert sorted(outside.parent.iterdir()) == []
    assert site.staged_names() == []
    assert site.record("light").read_text() == f"{site.digest()}\n"


def test_a_staged_record_is_created_exclusively(
    site: Site, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Something already at the staged record's name stops the write."""
    site.render(full=True)
    site.pair("light")[1].unlink()
    outside = tmp_path / "outside"
    real_rebuild = clocks._site_media._rebuild_pair

    def planting(replay, theme, clip, poster, encoder, unchanged):
        frames = real_rebuild(replay, theme, clip, poster, encoder, unchanged)
        clip.with_name(f".staged-{clip.stem}.sha256").symlink_to(outside)
        return frames

    monkeypatch.setattr(clocks._site_media, "_rebuild_pair", planting)
    with pytest.raises(FileExistsError):
        site.render(full=False)
    assert not outside.exists()


def test_a_failed_record_rename_leaves_the_pair_stale(
    site: Site, monkeypatch: pytest.MonkeyPatch
) -> None:
    site.render(full=True)
    clip, poster = site.pair("light")
    poster.unlink()
    clip.write_bytes(b"old clip")
    real_replace = os.replace

    def failing(source, destination, *args, **kwargs):
        if Path(destination) == site.record("light"):
            raise OSError("record rename")
        return real_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(clocks._site_media.os, "replace", failing)
    with pytest.raises(OSError, match="record rename"):
        site.render(full=False)
    monkeypatch.setattr(clocks._site_media.os, "replace", real_replace)

    assert clip.read_bytes() != b"old clip"
    assert probe(clip).frames == 92
    assert png_size(poster) == (1500, 900)
    assert not site.record("light").exists()
    assert site.render(full=False) == site.pair("light")


def test_an_archive_replaced_after_both_pairs_are_kept_is_rebuilt_next_run(
    site: Site,
) -> None:
    """The stated limit: a run that keeps every pair does not recheck the
    archive, and the next run's digest no longer matches the records."""
    site.render(full=True)

    def report(line: str) -> None:
        site.lines.append(line)
        if line == f"kept {DEMO}.light.mp4":
            _replace_archive(site)

    kept = render_site_media(
        site.archives,
        site.media,
        full=False,
        demos=(DEMO,),
        sources=site.sources,
        report=report,
    )
    assert kept == []
    assert site.render(full=False) == site.outputs()


def test_a_source_changed_during_a_render_certifies_nothing(
    site: Site, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pair drawn while a source changed is published without a record,
    and the next run rebuilds it."""
    site.render(full=True)
    site.pair("light")[1].unlink()

    def editing(replay, path, *, theme="light", encoder=None):
        site.source.write_text("x = 2\n")
        return render_clip(replay, path, theme=theme, encoder=encoder)

    monkeypatch.setattr(clocks._site_media, "render_clip", editing)
    with pytest.raises(clocks._site_media.SourcesChanged, match="changed"):
        site.render(full=False)
    monkeypatch.setattr(clocks._site_media, "render_clip", render_clip)

    assert not site.record("light").exists()
    assert site.render(full=False) == site.outputs()


def test_a_snapshot_given_is_what_the_records_name(site: Site) -> None:
    """A caller's snapshot, taken before anything ran, is what the digest is
    of: every pair is fresh for the tree, so only the snapshot can make one
    stale, and the tree then differs from it before the record is written."""
    site.render(full=True)
    with pytest.raises(clocks._site_media.SourcesChanged):
        render_site_media(
            site.archives,
            site.media,
            full=False,
            demos=(DEMO,),
            sources=site.sources,
            sources_snapshot=[("a.py", b"x = 0\n")],
            report=site.lines.append,
        )
    assert not site.record("light").exists()


@pytest.mark.parametrize("broken", ["missing", "unreadable_subdirectory"])
def test_an_unreadable_source_tree_fails_the_render(site: Site, broken: str) -> None:
    if broken == "missing":
        site.source.unlink()
        site.sources.rmdir()
        expected: type[OSError] = FileNotFoundError
    else:
        locked = site.sources / "sub"
        locked.mkdir()
        (locked / "b.py").write_text("y = 1\n")
        locked.chmod(0)
        expected = PermissionError
    try:
        with pytest.raises(expected):
            site.render(full=True)
    finally:
        if broken != "missing":
            (site.sources / "sub").chmod(0o755)
    assert [p for p in site.outputs() if p.exists()] == []


def test_a_source_that_cannot_be_inspected_fails_the_render(
    site: Site, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A listed file whose status cannot be read is an error, not left out."""
    real_stat = Path.stat

    def failing_stat(self, *args, **kwargs):
        if self == site.source:
            raise PermissionError(f"stat {self}")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", failing_stat)
    with pytest.raises(PermissionError, match="stat"):
        site.render(full=True)


def test_a_media_path_that_links_to_itself_is_refused(
    site: Site, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A link loop is a link: the documented ``ValueError``. On Python 3.12,
    which the site's workflows use, ``resolve`` raises ``RuntimeError`` on a
    loop (3.13 stopped); the stand-in does the same here."""
    site.media.symlink_to(site.media)
    real_resolve = Path.resolve

    def resolve_312(self, *args, **kwargs):
        if self == site.media:
            raise RuntimeError(f"Symlink loop from {self}")
        return real_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve_312)
    with pytest.raises(ValueError, match="symbolic link"):
        site.render(full=True)


def test_a_staged_poster_is_created_exclusively(
    site: Site, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A link planted at the staged poster's name while the clip encodes stops
    the poster, and nothing is written where it points."""
    site.render(full=True)
    site.pair("light")[1].unlink()
    outside = tmp_path / "outside.png"

    def planting(replay, path, *, theme="light", encoder=None):
        frames = render_clip(replay, path, theme=theme, encoder=encoder)
        site.path(f".staged-{DEMO}.{theme}.png").symlink_to(outside)
        return frames

    monkeypatch.setattr(clocks._site_media, "render_clip", planting)
    with pytest.raises(FileExistsError):
        site.render(full=False)
    assert not outside.exists()
    assert site.staged_names() == []


@pytest.mark.parametrize("name", ["old.mp4", "old.png", "demo_retired.light.sha256"])
def test_a_full_render_removes_a_dangling_link_of_an_owned_name(
    site: Site, tmp_path: Path, name: str
) -> None:
    site.media.mkdir()
    site.path(name).symlink_to(tmp_path / "nowhere")
    site.render(full=True)
    assert not site.path(name).is_symlink()
    assert f"removed {name}" in site.lines


@pytest.mark.parametrize("linked", ["clip", "poster", "record"])
def test_a_linked_output_is_stale(site: Site, tmp_path: Path, linked: str) -> None:
    """A clip, poster or record that is a link, even to a valid copy, is rebuilt."""
    site.render(full=True)
    clip, poster = site.pair("light")
    path = {"clip": clip, "poster": poster, "record": site.record("light")}[linked]
    copy = tmp_path / f"copy{path.suffix}"
    copy.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(copy)
    assert site.render(full=False) == site.pair("light")
    assert not path.is_symlink()


def test_staging_cleanup_never_follows_a_link(
    site: Site, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A staged link is removed without a lookup through it (on Python 3.12,
    ``is_dir`` follows the link and can raise for an inaccessible target)."""
    site.media.mkdir()
    link = site.path(".staged-link")
    link.symlink_to(tmp_path)
    real_is_dir = Path.is_dir

    def strict_is_dir(self, *args, **kwargs):
        if self.is_symlink():
            raise PermissionError(f"followed {self}")
        return real_is_dir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "is_dir", strict_is_dir)
    site.render(full=True)
    assert not link.is_symlink()


def test_a_media_path_ending_in_dotdot_through_a_link_is_refused(
    site: Site, tmp_path: Path
) -> None:
    """``linked/..`` names the parent of the link's target, not of the link:
    the guard compares the resolved path, which a symlink test alone misses."""
    elsewhere = tmp_path / "elsewhere" / "inner"
    elsewhere.mkdir(parents=True)
    (tmp_path / "linked").symlink_to(elsewhere, target_is_directory=True)
    media = tmp_path / "linked" / ".."
    assert not media.is_symlink()
    with pytest.raises(ValueError, match="resolves elsewhere"):
        render_site_media(
            site.archives,
            media,
            full=True,
            demos=(DEMO,),
            sources=site.sources,
            report=site.lines.append,
        )
    assert sorted(p.name for p in (tmp_path / "elsewhere").iterdir()) == ["inner"]


@dataclass
class _Package:
    """A copy of the package on ``PYTHONPATH``, and an archive and media beside it."""

    root: Path
    env: dict[str, str]
    archives: Path
    media: Path

    @property
    def style(self) -> Path:
        return self.root / "clocks" / "_style.py"

    def run(self, *args: str) -> str:
        result = subprocess.run(
            [sys.executable, *args],
            capture_output=True,
            text=True,
            env=self.env,
            cwd=self.root.parent,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout

    def prerender(self, mode: str, old: str, new: str) -> dict[str, object]:
        out = self.run(
            "-c",
            _PRERENDER_DRIVER,
            str(ROOT / "site" / "prerender.py"),
            str(self.archives),
            str(self.media),
            DEMO,
            mode,
            str(self.style),
            old,
            new,
        )
        return json.loads(out.splitlines()[-1])

    def light_background(self) -> str:
        return to_hex(_poster_pixels(self.media / f"{DEMO}.light.png")[0, 0, :3] / 255)

    def recorded_digest(self) -> str:
        return (self.media / f"{DEMO}.light.sha256").read_text()

    def tree_digest(self) -> str:
        archive = (self.archives / f"{DEMO}.npz").read_bytes()
        digest = clocks._site_media.inputs_digest(
            archive, read_sources(self.style.parent)
        )
        return f"{digest}\n"


@pytest.fixture
def package(tmp_path: Path) -> _Package:
    root = tmp_path / "pkg"
    shutil.copytree(
        ROOT / "src" / "clocks",
        root / "clocks",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    unset = {"PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX"}
    env = {k: v for k, v in os.environ.items() if k not in unset}
    env["PYTHONPATH"] = str(root)
    archives = tmp_path / "archives"
    archives.mkdir()
    save_replay(archives / f"{DEMO}.npz", _recorded())
    return _Package(root, env, archives, tmp_path / "media")


_OLD_BACKGROUND, _NEW_BACKGROUND = palette("light").background, "#123456"


def _recoloured(text: str) -> str:
    old, new = f'background="{_OLD_BACKGROUND}"', f'background="{_NEW_BACKGROUND}"'
    assert len(old) == len(new) and text.count(old) == 1
    return text.replace(old, new)


def _assert_nothing_left(state: dict[str, object]) -> None:
    """``main`` returned with no clocks module, path entry or font of the copy."""
    assert state == {"modules": [], "path_restored": True, "copy_fonts": 0}, state


def test_prerender_draws_with_the_source_it_records(package: _Package) -> None:
    """An imported renderer file is changed with its size and time kept, so a
    plain import reuses the old bytecode (checked first: the trap is armed).
    The pre-render step must draw with the new source, which its record names.
    """
    package.run("-c", "import clocks._site_media")
    assert list((package.root / "clocks" / "__pycache__").glob("_style.*.pyc"))
    text = package.style.read_text(encoding="utf-8")
    with _time_held(package.style):
        package.style.write_text(_recoloured(text), encoding="utf-8")
    shown = "from clocks._style import palette; print(palette('light').background)"
    assert package.run("-c", shown).strip() == _OLD_BACKGROUND

    state = package.prerender("plain", "", "")

    assert package.light_background() == _NEW_BACKGROUND
    assert package.recorded_digest() == package.tree_digest()
    _assert_nothing_left(state)


def test_prerender_draws_with_the_snapshot_whatever_the_tree_does(
    package: _Package,
) -> None:
    """The live file is edited after the snapshot, imported from in that state
    by any import of the live tree, and restored before the records are
    checked: the copy of the snapshot must be what draws."""
    text = package.style.read_text(encoding="utf-8")

    state = package.prerender("edit_import_restore", text, _recoloured(text))

    assert package.style.read_text(encoding="utf-8") == text
    assert package.light_background() == _OLD_BACKGROUND
    assert package.recorded_digest() == package.tree_digest()
    _assert_nothing_left(state)


def test_prerender_refuses_an_imported_clocks(tmp_path: Path) -> None:
    """In this process clocks is imported: its code, not a snapshot, would draw."""
    spec = importlib.util.spec_from_file_location(
        "prerender_under_test", ROOT / "site" / "prerender.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = list(sys.path)
    with pytest.raises(RuntimeError, match="already imported"):
        module.main(tmp_path / "archives", tmp_path / "media", full=True)
    assert sys.path == path
    assert list(tmp_path.iterdir()) == []


_PRERENDER_DRIVER = """\
import importlib.util
import json
import sys
from pathlib import Path

from matplotlib import font_manager

prerender, archives, media, demo, mode, style, old, new = sys.argv[1:]
spec = importlib.util.spec_from_file_location("prerender", prerender)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if mode == "edit_import_restore":
    snapshot, import_renderer = module._snapshot, module._import_renderer

    def edited_snapshot(package):
        taken = snapshot(package)
        Path(style).write_text(new, encoding="utf-8")
        return taken

    def restored_import(root):
        renderer = import_renderer(root)
        Path(style).write_text(old, encoding="utf-8")
        return renderer

    module._snapshot, module._import_renderer = edited_snapshot, restored_import
path = list(sys.path)
module.main(Path(archives), Path(media), full=True, demos=(demo,))
copy_fonts = [
    entry.fname
    for entry in font_manager.fontManager.ttflist
    if "clocks-snapshot-" in entry.fname
]
state = {
    "modules": sorted(n for n in sys.modules if n.split(".")[0] == "clocks"),
    "path_restored": sys.path == path,
    "copy_fonts": len(copy_fonts),
}
print(json.dumps(state))
"""


def _archive_with(site: Site, name: str, row: int) -> Path:
    """A copy of the fixture archive with 0.5 added to ``cloud[row]``."""
    base = load_replay(site.archive)
    cloud = np.array(base.arrays["cloud"])
    cloud[row] += 0.5
    directory = site.archive.parent.parent / name
    directory.mkdir()
    save_replay(directory / f"{DEMO}.npz", edited(base, cloud=cloud))
    return directory


def test_the_poster_is_the_last_row_of_its_archive(
    site: Site, monkeypatch: pytest.MonkeyPatch
) -> None:
    last_changed = _archive_with(site, "last-changed", -1)
    second_changed = _archive_with(site, "second-changed", 1)
    dots: list[np.ndarray] = []

    def reading(fig, file):
        (cloud,) = [ax for ax in fig.axes if ax.get_gid() == "map"]
        dots.append(np.array(one(cloud, "dots").get_xydata()))
        save_png(fig, file)

    monkeypatch.setattr(clocks._site_media, "save_png", reading)
    posters = []
    for archives in (site.archives, last_changed, second_changed):
        media = archives.parent / f"media-{archives.name}"
        render_site_media(
            archives,
            media,
            full=True,
            demos=(DEMO,),
            sources=site.sources,
            report=site.lines.append,
        )
        posters.append(matplotlib.image.imread(media / f"{DEMO}.light.png"))

    assert not np.array_equal(posters[1], posters[0])
    assert np.array_equal(posters[2], posters[0])
    expected = load_replay(site.archive).arrays["cloud"][-1][:, :2]
    assert len(dots) == 3 * len(THEMES)
    for found in dots[:2]:
        np.testing.assert_allclose(found, expected, rtol=1e-12, atol=1e-12)


def test_a_pair_is_drawn_from_one_loaded_replay(
    site: Site, monkeypatch: pytest.MonkeyPatch
) -> None:
    loads = _spy(monkeypatch, "load_replay", load_replay)
    clips = _spy(monkeypatch, "render_clip", render_clip)
    posters = _spy(monkeypatch, "final_frame", final_frame)
    site.render(full=True)

    assert [Path(call["path"]) for call in loads] == [site.archive]
    replay = loads[0]["returned"]
    assert len(clips) == len(posters) == len(THEMES)
    for theme, clip_call, poster_call in zip(THEMES, clips, posters, strict=True):
        assert clip_call["replay"] is replay
        assert poster_call["replay"] is replay
        assert clip_call["theme"] == theme
        assert poster_call["colors"] == palette(theme)
        assert Path(clip_call["path"]) == site.path(f".staged-{DEMO}.{theme}.mp4")


def test_the_report_names_the_encoder_and_each_file(site: Site) -> None:
    site.render(full=True)
    assert site.lines[0].startswith("encoder: ")
    assert "ffmpeg version" in site.lines[0]
    assert len(site.lines) == 1 + len(THEMES)
    for theme, line in zip(THEMES, site.lines[1:], strict=True):
        assert re.fullmatch(
            rf"rendered demo_2d\.{theme}\.mp4 \(92 frames, \d+\.\d s\)", line
        ), line


def test_site_media_covers_the_clip_demos() -> None:
    assert CLIP_DEMOS == tuple(name for name in DEMOS if name != "demo_density")
    assert set(CLIP_DEMOS) == set(CLIP_HEIGHTS)


def test_the_prerender_script_is_its_pinned_program() -> None:
    script = (ROOT / "site" / "prerender.py").read_text(encoding="utf-8")
    assert ast.dump(ast.parse(script)) == ast.dump(ast.parse(PINNED))


def test_site_media_is_git_ignored() -> None:
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "site/media/" in lines


def test_the_sources_module_imports_nothing_of_clocks() -> None:
    """``site/prerender.py`` runs it before ``clocks`` is imported."""
    tree = ast.parse((ROOT / "src" / "clocks" / "_sources.py").read_text("utf-8"))
    names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    names |= {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert names <= {"collections.abc", "pathlib", "stat"}, names
