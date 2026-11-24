"""The site's pre-render step: clips and posters staged, then published as pairs."""

import ast
import functools
import inspect
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.image
import numpy as np
import pytest
from _clip_replays import edited, first_pixel
from _figure_artists import one, png_size
from matplotlib.colors import to_rgba

import clocks._site_media
from clocks._clips import CLIP_HEIGHTS, final_frame, render_clip
from clocks._demos import demo_2d
from clocks._replay import DEMOS, Replay, load_replay, save_replay
from clocks._site_media import CLIP_DEMOS, render_site_media
from clocks._style import THEMES, palette, save_png
from clocks._video import EncoderError, EncoderNotFound, StreamInfo, probe

ROOT = Path(__file__).resolve().parents[1]
DEMO = "demo_2d"
OLD = ("old clip", "old poster")

PINNED = '''"""Quarto pre-render step: draw the site's clips and posters into media/."""

import os
from pathlib import Path

from clocks._site_media import render_site_media

SITE = Path(__file__).resolve().parent
render_site_media(
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

    def render(self, *, full: bool) -> list[Path]:
        return render_site_media(
            self.archives,
            self.media,
            full=full,
            demos=(DEMO,),
            sources=self.sources,
            report=self.lines.append,
        )

    def set_times(self, inputs: float = 1_000, outputs: float = 2_000) -> None:
        """The archive and the source at ``inputs``; the four outputs at ``outputs``."""
        for path in (self.archive, self.source):
            os.utime(path, (inputs, inputs))
        for path in self.outputs():
            if path.exists():
                os.utime(path, (outputs, outputs))

    def put_old_pair(self) -> None:
        self.media.mkdir(parents=True, exist_ok=True)
        clip, poster = self.pair("light")
        clip.write_bytes(OLD[0].encode())
        poster.write_bytes(OLD[1].encode())

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
    assert sorted(site.media.iterdir()) == sorted(site.outputs())
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
    site.set_times()
    calls = _spy(monkeypatch, "render_clip")
    assert site.render(full=True) == site.outputs()
    assert len(calls) == 2


def test_a_full_render_removes_other_media_and_nothing_else(site: Site) -> None:
    site.media.mkdir()
    for name in ("old.mp4", "old.png", "notes.txt", "other.tmp"):
        site.path(name).write_text(name)
    site.path("sub").mkdir()
    site.path("sub/keep.mp4").write_text("keep")
    site.render(full=True)
    left = {p.relative_to(site.media).as_posix() for p in site.media.rglob("*")}
    assert left == {
        *(p.name for p in site.outputs()),
        "notes.txt",
        "other.tmp",
        "sub",
        "sub/keep.mp4",
    }
    assert "removed old.mp4" in site.lines
    assert "removed old.png" in site.lines


@pytest.mark.parametrize("full", [True, False], ids=["full", "partial"])
def test_staging_leftovers_are_removed_on_any_render(site: Site, full: bool) -> None:
    site.render(full=True)
    site.set_times()
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
    site.set_times()
    calls = _spy(monkeypatch, "render_clip")
    site.lines.clear()
    assert site.render(full=False) == []
    assert "kept demo_2d.light.mp4" in site.lines
    assert "kept demo_2d.dark.mp4" in site.lines
    assert calls == []


def _archive_newer(site: Site) -> None:
    os.utime(site.archive, (3_000, 3_000))


def _source_newer(site: Site) -> None:
    os.utime(site.source, (3_000, 3_000))


def _tie(site: Site) -> None:
    os.utime(site.archive, (2_000, 2_000))


def _poster_missing(site: Site) -> None:
    site.pair("light")[1].unlink()


def _clip_older(site: Site) -> None:
    os.utime(site.pair("light")[0], (500, 500))


def _stray_file(site: Site) -> None:
    site.path("old.mp4").write_text("stray")


@pytest.mark.parametrize(
    ("change", "rebuilt"),
    [
        (_archive_newer, ("light", "dark")),
        (_source_newer, ("light", "dark")),
        (_tie, ("light", "dark")),
        (_poster_missing, ("light",)),
        (_clip_older, ("light",)),
        (_stray_file, ()),
    ],
    ids=[
        "archive_newer",
        "source_newer",
        "tie",
        "poster_missing",
        "clip_older",
        "stray_file_kept",
    ],
)
def test_a_partial_render_rebuilds_what_is_stale(
    site: Site, change: Callable[[Site], None], rebuilt: tuple[str, ...]
) -> None:
    site.render(full=True)
    site.set_times()
    change(site)
    written = site.render(full=False)
    assert written == [path for theme in rebuilt for path in site.pair(theme)]
    if change is _stray_file:
        assert site.path("old.mp4").read_text() == "stray"


def test_a_cached_bytecode_file_does_not_make_media_stale(site: Site) -> None:
    site.render(full=True)
    site.set_times()
    cached = site.sources / "__pycache__" / "a.pyc"
    cached.parent.mkdir()
    cached.write_bytes(b"cached")
    os.utime(cached, (3_000, 3_000))
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
    site.set_times()

    def failing(replay, path, *, theme="light", encoder=None):
        Path(path).write_bytes(b"partial")
        raise EncoderError("the encoder failed")

    monkeypatch.setattr(clocks._site_media, "render_clip", failing)
    with pytest.raises(EncoderError, match="the encoder failed"):
        site.render(full=True)
    assert site.old_pair() == OLD
    assert site.staged_names() == []


def test_a_failed_poster_leaves_the_old_pair_alone(
    site: Site, monkeypatch: pytest.MonkeyPatch
) -> None:
    site.put_old_pair()
    site.set_times()

    def failing(fig, path):
        Path(path).write_bytes(b"partial")
        raise OSError("disk")

    monkeypatch.setattr(clocks._site_media, "save_png", failing)
    with pytest.raises(OSError, match="disk"):
        site.render(full=True)
    assert site.old_pair() == OLD
    assert site.staged_names() == []


def test_an_interrupted_publication_leaves_a_file_missing(
    site: Site, monkeypatch: pytest.MonkeyPatch
) -> None:
    site.put_old_pair()
    site.set_times()
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

    def reading(fig, path):
        (cloud,) = [ax for ax in fig.axes if ax.get_gid() == "map"]
        dots.append(np.array(one(cloud, "dots").get_xydata()))
        save_png(fig, path)

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
    for line in site.lines[1:3]:
        assert re.fullmatch(
            r"rendered demo_2d\.(light|dark)\.mp4 \(92 frames, \d+\.\d s\)", line
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
