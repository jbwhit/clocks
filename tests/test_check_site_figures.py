"""Tests of the rendered-site figure check.

The limit of the clip checks, as decision D39 states it: the check refuses a
clip or a description whose own start tag, or an ancestor's, carries one of the
markers D39 names (a class in ``HIDING_CLASSES``, the ``hidden`` attribute,
``aria-hidden="true"``, an inline ``display:none`` or ``visibility:hidden``, a
``<template>``, a ``<noscript>`` or a closed ``<details>``), a description
whose only text lies inside such a marked element, a clip file anywhere but
``media/``, a poster that does not decode and a stream that does not decode
to the expected frames. It does not compute visibility: a stylesheet rule, a
script, an ancestor's opacity, an ancestor that clips it (``clip-path``, or
``overflow: hidden`` on a box too small to hold it) or any marker outside that
list can still hide a clip, and only the browser check of Task 13 finds those.
It establishes that a clip page links a ``replay.css`` holding exactly the
canonical rule, not that the rule takes effect: another stylesheet or a script
can override it, and the 390-pixel browser check shows a clip shrinking with
its column. And it establishes that a clip has its archive's frame count, not
that its frames show that archive; the full render ties the published files to
their inputs (decision D27).

Fixture clips are made with the real encoder through ``clip_media``, which only
clip tests request, so the plot-only cases run without FFmpeg; a test that
needs the encoder fails without it, never skips.
"""

import html
import importlib.util
import re
import shutil
import struct
import subprocess
import tempfile
import zlib
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest
from matplotlib import image as mpimg
from PIL import Image

from clocks._demos import demo_2d
from clocks._replay import load_replay, save_replay
from clocks._style import palette
from clocks._video import EncoderNotFound, find_encoder

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_site_figures.py"
_SPEC = importlib.util.spec_from_file_location("check_site_figures", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
check_site_figures = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_site_figures)

Fig = check_site_figures.Fig

PAGE = "story/demo.html"
CAPTION = "Rate of <code>x</code> versus distance."


def _wrapper(
    theme: str,
    images: list[tuple[str, int]],
    captions: list[str],
    height: int,
    tag: str = "div",
    extra_class: str = "",
    before_figure: str = "",
) -> str:
    imgs = "".join(
        f'<p><img src="{src}" width="{width}" height="{height}" '
        'class="figure-img"></p>\n'
        for src, width in images
    )
    caps = "".join(f"<figcaption>{c}</figcaption>\n" for c in captions)
    classes = f"cell-output cell-output-display {theme}-content{extra_class}"
    return (
        f'<{tag} class="{classes}">\n{before_figure}'
        '<div class="quarto-figure quarto-figure-center">\n'
        f'<figure class="figure">\n{imgs}{caps}</figure>\n</div>\n</{tag}>\n'
    )


def _build(
    tmp_path: Path,
    *,
    light: list[tuple[str, int]] | None = None,
    dark: list[tuple[str, int]] | None = None,
    light_captions: list[str] | None = None,
    dark_captions: list[str] | None = None,
    light_height: int = 420,
    dark_height: int = 420,
    dark_extra_class: str = "",
    tag: str = "div",
    before_figure: str = "",
    nest_dark: bool = False,
    extra: str = "",
) -> Path:
    for name in ("light.png", "dark.png", "plot.png"):
        (tmp_path / "story" / "demo_files").mkdir(parents=True, exist_ok=True)
        (tmp_path / "story" / "demo_files" / name).write_bytes(name.encode())
    light = light if light is not None else [("demo_files/light.png", 750)]
    dark = dark if dark is not None else [("demo_files/dark.png", 750)]
    lc = light_captions if light_captions is not None else [CAPTION]
    dc = dark_captions if dark_captions is not None else [CAPTION]
    light_html = _wrapper("light", light, lc, light_height, tag, "", before_figure)
    dark_html = _wrapper(
        "dark", dark, dc, dark_height, tag, dark_extra_class, before_figure
    )
    if nest_dark:
        closing = f"</{tag}>\n"
        light_html = light_html.removesuffix(closing) + dark_html + closing
        dark_html = ""
    body = (
        '<p><img src="demo_files/light.png" width="1" height="1"></p>\n'  # ignored
        + light_html
        + dark_html
        + extra
    )
    (tmp_path / "story" / "demo.html").write_text(
        f"<html><body>{body}</body></html>", encoding="utf-8"
    )
    return tmp_path


@pytest.fixture(autouse=True)
def one_expected_page(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        check_site_figures, "EXPECTED_FIGURES", {PAGE: (Fig("plot", 420),)}
    )


def _pair(n: int, height: int) -> tuple[str, str]:
    """The light and dark wrappers of figure ``n``."""
    return (
        _wrapper("light", [(f"demo_files/light{n}.png", 750)], [CAPTION], height),
        _wrapper("dark", [(f"demo_files/dark{n}.png", 750)], [CAPTION], height),
    )


def _page(
    tmp_path: Path, wrappers: Sequence[str], page: str = PAGE, head: str = ""
) -> Path:
    """Write the image files and a page holding ``wrappers`` in the order given."""
    files = tmp_path / Path(page).parent / "demo_files"
    files.mkdir(parents=True, exist_ok=True)
    for n in range(1, 4):
        for theme in ("light", "dark"):
            (files / f"{theme}{n}.png").write_bytes(f"{theme}{n}.png".encode())
    (tmp_path / page).write_text(
        f"<html><head>{head}</head><body>{''.join(wrappers)}</body></html>",
        encoding="utf-8",
    )
    return tmp_path


def _single_failure(tmp_path: Path, reason: str, page: str = PAGE) -> None:
    failures = check_site_figures.figure_failures(tmp_path)
    assert len(failures) == 1, failures
    assert page in failures[0]
    assert reason in failures[0]


def test_a_well_formed_page_passes(tmp_path: Path) -> None:
    assert check_site_figures.figure_failures(_build(tmp_path)) == []


def test_missing_dark_image(tmp_path: Path) -> None:
    _single_failure(
        _build(tmp_path, dark=[]), "dark-content: expected 1 image, found 0"
    )


def test_two_light_images(tmp_path: Path) -> None:
    light = [("demo_files/light.png", 750), ("demo_files/plot.png", 750)]
    _single_failure(
        _build(tmp_path, light=light), "light-content: expected 1 image, found 2"
    )


def test_wrong_width(tmp_path: Path) -> None:
    _single_failure(
        _build(tmp_path, light=[("demo_files/light.png", 597)]),
        "light-content: width 597, expected 750",
    )


def test_wrong_height(tmp_path: Path) -> None:
    _single_failure(
        _build(tmp_path, dark_height=540), "dark-content: height 540, expected 420"
    )


def test_same_file_in_both_wrappers(tmp_path: Path) -> None:
    _single_failure(
        _build(
            tmp_path,
            light=[("demo_files/plot.png", 750)],
            dark=[("./demo_files/plot.png", 750)],
        ),
        "same image file",
    )


def test_src_that_does_not_exist(tmp_path: Path) -> None:
    _single_failure(
        _build(tmp_path, dark=[("demo_files/gone.png", 750)]),
        "image 'demo_files/gone.png' does not exist",
    )


def test_no_figcaption_in_dark_wrapper(tmp_path: Path) -> None:
    _single_failure(
        _build(tmp_path, dark_captions=[]),
        "dark-content: expected 1 figcaption, found 0",
    )


def test_two_figcaptions_in_one_wrapper(tmp_path: Path) -> None:
    _single_failure(
        _build(tmp_path, light_captions=[CAPTION, CAPTION]),
        "light-content: expected 1 figcaption, found 2",
    )


def test_captions_differing_only_inside_nested_code(tmp_path: Path) -> None:
    _single_failure(
        _build(tmp_path, dark_captions=["Rate of <code>y</code> versus distance."]),
        "captions differ",
    )


def test_empty_caption_in_dark_wrapper(tmp_path: Path) -> None:
    _single_failure(_build(tmp_path, dark_captions=[""]), "dark-content: empty caption")


def test_a_well_formed_page_with_section_wrappers_passes(tmp_path: Path) -> None:
    # The committed image after the wrappers is ignored only if each wrapper's
    # end is found at its own </section>, not at the last </div> inside it.
    later = '<div><p><img src="demo_files/plot.png" width="1" height="1"></p></div>\n'
    site = _build(tmp_path, tag="section", extra=later)
    assert check_site_figures.figure_failures(site) == []


def test_a_dark_caption_in_a_second_dark_wrapper(tmp_path: Path) -> None:
    second = _wrapper("dark", [], [CAPTION], 420)
    _single_failure(
        _build(tmp_path, dark_captions=[], extra=second),
        "expected 1 figure(s), 2 theme wrappers; found 3",
    )


def test_a_wrapper_with_both_theme_classes(tmp_path: Path) -> None:
    _single_failure(
        _build(tmp_path, dark_extra_class=" light-content"),
        "wrapper has both theme classes",
    )


def test_an_absolute_image_path_is_outside_the_site(tmp_path: Path) -> None:
    site = tmp_path / "site"
    target = site / "story" / "demo_files" / "dark.png"
    _single_failure(
        _build(site, dark=[(str(target), 750)]), "image is outside the site"
    )
    assert target.is_file()


def test_a_percent_encoded_absolute_path_is_outside_the_site(tmp_path: Path) -> None:
    site = tmp_path / "site"
    target = site / "story" / "demo_files" / "dark.png"
    encoded = "%2F" + target.as_posix().removeprefix("/")
    _single_failure(_build(site, dark=[(encoded, 750)]), "image is outside the site")
    assert target.is_file()


def test_a_malformed_image_url_is_a_failure(tmp_path: Path) -> None:
    _single_failure(_build(tmp_path, dark=[("//[bad", 750)]), "image URL is malformed")


def test_an_image_path_escaping_the_site_is_outside_it(tmp_path: Path) -> None:
    site = tmp_path / "site"
    (tmp_path / "outside.png").write_bytes(b"outside")
    _single_failure(
        _build(site, dark=[("../../outside.png", 750)]), "image is outside the site"
    )


def test_a_well_formed_tall_page_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _build(tmp_path, light_height=540, dark_height=540)
    monkeypatch.setattr(
        check_site_figures, "EXPECTED_FIGURES", {PAGE: (Fig("plot", 540),)}
    )
    assert check_site_figures.figure_failures(site) == []
    monkeypatch.setattr(
        check_site_figures, "EXPECTED_FIGURES", {PAGE: (Fig("plot", 420),)}
    )
    failures = check_site_figures.figure_failures(site)
    assert len(failures) == 2, failures
    for theme, failure in zip(("light", "dark"), failures, strict=True):
        assert f"{PAGE}: figure 1 {theme}-content: height 540, expected 420" == failure


def test_section_wrappers_with_a_div_before_the_image_pass(tmp_path: Path) -> None:
    # A wrapper that closed at the first </div> inside it would lose its image.
    site = _build(
        tmp_path, tag="section", before_figure='<div class="cell-output"></div>\n'
    )
    assert check_site_figures.figure_failures(site) == []


def test_a_dark_wrapper_nested_inside_the_light_one(tmp_path: Path) -> None:
    _single_failure(
        _build(tmp_path, nest_dark=True), "theme wrapper nested inside another"
    )


def test_a_query_string_on_the_image_urls_passes(tmp_path: Path) -> None:
    site = _build(
        tmp_path,
        light=[("demo_files/light.png?v=2", 750)],
        dark=[("demo_files/dark.png?v=2", 750)],
    )
    assert check_site_figures.figure_failures(site) == []


def test_a_percent_escaped_image_url_names_its_file(tmp_path: Path) -> None:
    site = _build(tmp_path, dark=[("demo_files/dark%20plot.png", 750)])
    (site / "story" / "demo_files" / "dark plot.png").write_bytes(b"dark plot")
    assert check_site_figures.figure_failures(site) == []


def test_urls_differing_only_in_query_are_the_same_file(tmp_path: Path) -> None:
    _single_failure(
        _build(
            tmp_path,
            light=[("demo_files/plot.png?v=1", 750)],
            dark=[("demo_files/plot.png?v=2", 750)],
        ),
        "same image file",
    )


def test_a_missing_page_is_a_failure(tmp_path: Path) -> None:
    _single_failure(tmp_path, "page is missing")


TWO_PLOTS = (Fig("plot", 540), Fig("plot", 420))


def _set_manifest(monkeypatch: pytest.MonkeyPatch, figures: tuple[object, ...]) -> None:
    monkeypatch.setattr(check_site_figures, "EXPECTED_FIGURES", {PAGE: figures})


def test_a_page_with_two_plots_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_manifest(monkeypatch, TWO_PLOTS)
    site = _page(tmp_path, [*_pair(1, 540), *_pair(2, 420)])
    assert check_site_figures.figure_failures(site) == []


def test_two_plots_with_their_heights_swapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_manifest(monkeypatch, TWO_PLOTS)
    site = _page(tmp_path, [*_pair(1, 420), *_pair(2, 540)])
    assert check_site_figures.figure_failures(site) == [
        f"{PAGE}: figure 1 light-content: height 420, expected 540",
        f"{PAGE}: figure 1 dark-content: height 420, expected 540",
        f"{PAGE}: figure 2 light-content: height 540, expected 420",
        f"{PAGE}: figure 2 dark-content: height 540, expected 420",
    ]


def test_wrappers_out_of_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_manifest(monkeypatch, TWO_PLOTS)
    (light1, dark1), (light2, dark2) = _pair(1, 540), _pair(2, 420)
    site = _page(tmp_path, [light1, light2, dark1, dark2])
    assert check_site_figures.figure_failures(site) == [
        f"{PAGE}: figure 1: expected light-content then dark-content, "
        "found light-content then light-content",
        f"{PAGE}: figure 2: expected light-content then dark-content, "
        "found dark-content then dark-content",
    ]


def test_a_missing_second_figure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_manifest(monkeypatch, TWO_PLOTS)
    site = _page(tmp_path, _pair(1, 540))
    assert check_site_figures.figure_failures(site) == [
        f"{PAGE}: expected 2 figure(s), 4 theme wrappers; found 2"
    ]


def test_identical_light_and_dark_content(tmp_path: Path) -> None:
    site = _build(tmp_path)
    (site / "story/demo_files/dark.png").write_bytes(b"light.png")
    _single_failure(site, "figure 1: light and dark images have the same content")


def test_an_encoded_nul_is_a_failure(tmp_path: Path) -> None:
    _single_failure(
        _build(tmp_path, dark=[("demo_files/dark%00.png", 750)]),
        "image path contains a NUL",
    )


def test_a_wrapper_on_an_unlisted_page(tmp_path: Path) -> None:
    site = _build(tmp_path)
    _page(site, [_pair(1, 420)[0]], page="story/other.html")
    assert check_site_figures.figure_failures(site) == [
        "story/other.html: not in the figure manifest, but has 1 theme wrapper(s)"
    ]


def test_a_listed_page_with_no_figures_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_manifest(monkeypatch, ())
    site = _page(tmp_path, [])
    assert check_site_figures.figure_failures(site) == []


def test_a_listed_page_with_no_figures_must_have_no_wrappers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_manifest(monkeypatch, ())
    site = _page(tmp_path, _pair(1, 420))
    assert check_site_figures.figure_failures(site) == [
        f"{PAGE}: expected 0 figure(s), 0 theme wrappers; found 2"
    ]


# --- clips -------------------------------------------------------------------

CANONICAL = b"figure.replay video { max-width: 100%; height: auto; }\n"
CLIP = Fig("clip", 450, "demo_2d")
FRAME_WIDTH, FRAME_HEIGHT = 1500, 900
FRAMES = 92  # sum(dwell_frames(2)): 24 + 18 + 50
MEDIA_FILES = tuple(
    f"demo_2d.{theme}.{suffix}"
    for theme in ("light", "dark")
    for suffix in ("mp4", "png")
)
OK: list[str] = []


def _encode(
    path: Path,
    frames: Iterable[bytes],
    *,
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
    rate: int = 20,
) -> None:
    """Pipe raw RGBA frames to the encoder with Task 2's command, spelt out here."""
    command = [
        find_encoder(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{width}x{height}",
        "-r", str(rate), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "20", "-preset", "slow", "-movflags", "+faststart", "-f", "mp4",
        str(path),
    ]  # fmt: skip
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
        assert process.stdin is not None
        try:
            for frame in frames:
                process.stdin.write(frame)
        finally:
            process.stdin.close()
            status = process.wait()
        errors.seek(0)
        assert status == 0, errors.read().decode(errors="replace")


def _transcode(path: Path, *options: str) -> None:
    """Replace the clip at ``path`` with the encoder's output for ``options``."""
    temporary = path.parent.parent / f"transcoded{path.suffix}"
    subprocess.run(
        [find_encoder(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
        + ["-i", str(path), *options, str(temporary)],
        check=True,
        capture_output=True,
    )
    temporary.replace(path)


def _background(theme: str) -> tuple[int, int, int]:
    colour = palette(theme).background
    return int(colour[1:3], 16), int(colour[3:5], 16), int(colour[5:7], 16)


def _frames(
    theme: str, *, height: int = FRAME_HEIGHT, count: int = FRAMES
) -> Iterator[bytes]:
    """Solid frames in the theme's background with a block of fresh seeded noise
    that moves ten pixels a frame, so that no two frames are alike."""
    rng = np.random.default_rng(7)
    base = np.empty((height, FRAME_WIDTH, 4), dtype=np.uint8)
    base[...] = (*_background(theme), 255)
    for k in range(count):
        frame = base.copy()
        frame[100:164, 10 * k : 10 * k + 64, :3] = rng.integers(0, 256, (64, 64, 3))
        yield frame.tobytes()


def _write_poster(
    path: Path, theme: str, height: int = FRAME_HEIGHT, width: int = FRAME_WIDTH
) -> None:
    mpimg.imsave(path, np.full((height, width, 3), _background(theme), np.uint8))


@pytest.fixture(scope="module")
def clip_media(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An archive of two observations, and the clips and posters drawn for it."""
    root = tmp_path_factory.mktemp("clip_media")
    (root / "archives").mkdir()
    save_replay(
        root / "archives" / "demo_2d.npz",
        demo_2d.record(n_observations=2, n_particles=16),
    )
    for theme in ("light", "dark"):
        _encode(root / f"demo_2d.{theme}.mp4", _frames(theme))
        _write_poster(root / f"demo_2d.{theme}.png", theme)
    return root


def _video(theme: str, overrides: Mapping[str, object]) -> str:
    """One ``<video>`` as Quarto 1.11.9 writes the design's embed (decision D22).

    A value of None removes an attribute; ``flags`` replaces the boolean
    attributes written ``name=""``, and ``bare_flags`` writes names bare.
    """
    overrides = dict(overrides)
    flags = overrides.pop("flags", ("controls", "playsinline"))
    bare = overrides.pop("bare_flags", ())
    assert isinstance(flags, tuple) and isinstance(bare, tuple)
    values = {
        "class": f"{theme}-content",
        "aria-label": "Animation: the 2D search",
        "aria-describedby": "demo-2d-text",
        "src": f"../media/demo_2d.{theme}.mp4",
        "poster": f"../media/demo_2d.{theme}.png",
        "width": "750",
        "height": "450",
        "preload": "none",
        **overrides,
    }
    pending = [
        f'{name}=""' for name in flags if name not in values and name not in bare
    ]
    pending += list(bare)
    parts: list[str] = []
    for name, value in values.items():
        if name == "preload":
            parts += pending
            pending = []
        if value is not None:
            parts.append(f'{name}="{value}"')
    return f"<video {' '.join(parts + pending)}></video>"


def _clip_figure(
    light: Mapping[str, object],
    dark: Mapping[str, object],
    caption: str | None,
    figure_class: str,
) -> str:
    figcaption = "" if caption is None else f"<figcaption>{caption}</figcaption>\n"
    return (
        f'<figure class="{figure_class}">\n{_video("light", light)}\n'
        f"{_video('dark', dark)}\n{figcaption}</figure>\n"
    )


QUARTO_FIGURE = """\
<figure class="replay figure">
<video class="light-content" aria-label="Animation: the 2D search" \
aria-describedby="demo-2d-text" src="../media/demo_2d.light.mp4" \
poster="../media/demo_2d.light.png" width="750" height="450" controls="" \
playsinline="" preload="none"></video>
<video class="dark-content" aria-label="Animation: the 2D search" \
aria-describedby="demo-2d-text" src="../media/demo_2d.dark.mp4" \
poster="../media/demo_2d.dark.png" width="750" height="450" controls="" \
playsinline="" preload="none"></video>
<figcaption>A caption.</figcaption>
</figure>
"""


def _clip_page(
    site: Path,
    media_source: Path,
    *,
    light: Mapping[str, object] | None = None,
    dark: Mapping[str, object] | None = None,
    caption: str | None = "A caption.",
    description: str | None = "Words.",
    page: str = PAGE,
    around: tuple[str, str] = ("", ""),
    before: Sequence[str] = (),
    after: Sequence[str] = (),
    replay_css: bytes | None = CANONICAL,
    link: str | None = "../replay.css",
    link_rel: str = "stylesheet",
    media: bool = True,
    figure: bool = True,
    figure_class: str = "replay figure",
) -> Path:
    """Write a clip page, through ``_page``, and the site files it needs.

    ``before`` and ``after`` are plot wrappers placed around the clip figure and
    its description, whose image files ``_page`` writes.
    """
    if media:
        (site / "media").mkdir(exist_ok=True)
        for name in MEDIA_FILES:
            shutil.copyfile(media_source / name, site / "media" / name)
    if replay_css is not None:
        (site / "replay.css").write_bytes(replay_css)
    head = "" if link is None else f'<link rel="{link_rel}" href="{link}">'
    clip = _clip_figure(light or {}, dark or {}, caption, figure_class)
    text = (
        ""
        if description is None
        else f'<div id="demo-2d-text">\n<p>{description}</p>\n</div>\n'
    )
    parts = [*before, around[0], clip if figure else "", around[1], text, *after]
    return _page(site, parts, page=page, head=head)


@pytest.fixture
def clip_site(
    tmp_path: Path, clip_media: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[..., Path]:
    """A builder of clip pages in ``tmp_path``; the manifest lists one clip."""
    monkeypatch.setattr(check_site_figures, "ARCHIVES", clip_media / "archives")
    _set_manifest(monkeypatch, (CLIP,))

    def build(**options: object) -> Path:
        return _clip_page(tmp_path, clip_media, **options)  # type: ignore[arg-type]

    return build


def _edit(site: Path, old: str, new: str, page: str = PAGE) -> None:
    path = site / page
    text = path.read_text(encoding="utf-8")
    assert text.count(old) == 1, old
    path.write_text(text.replace(old, new), encoding="utf-8")


def _failures(site: Path, *reasons: str) -> None:
    """Assert one failure per reason, in order, each containing its reason."""
    failures = check_site_figures.figure_failures(site)
    assert len(failures) == len(reasons), failures
    for failure, reason in zip(failures, reasons, strict=True):
        assert reason in failure, failures


def test_the_fixture_writes_the_embed_as_quarto_does() -> None:
    assert _clip_figure({}, {}, "A caption.", "replay figure") == QUARTO_FIGURE
    bare = _video("light", {"bare_flags": ("controls", "playsinline")})
    assert bare.endswith(
        ' height="450" controls playsinline preload="none"></video>'
    ), bare


def test_a_well_formed_clip_passes(clip_site: Callable[..., Path]) -> None:
    assert check_site_figures.figure_failures(clip_site()) == OK


def test_a_clip_and_two_plots_in_order_pass(
    clip_site: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_manifest(monkeypatch, (Fig("plot", 540), CLIP, Fig("plot", 420)))
    site = clip_site(before=_pair(1, 540), after=_pair(2, 420))
    assert check_site_figures.figure_failures(site) == OK


def test_a_clip_in_a_plain_div_passes(clip_site: Callable[..., Path]) -> None:
    site = clip_site(around=('<div class="column-page">', "</div>"))
    assert check_site_figures.figure_failures(site) == OK


def test_a_missing_dark_video(clip_site: Callable[..., Path]) -> None:
    site = clip_site()
    _edit(site, _video("dark", {}) + "\n", "")
    _single_failure(site, "expected 1 figure(s), 2 theme wrappers; found 1")


def test_a_clip_missing_from_the_page(clip_site: Callable[..., Path]) -> None:
    _single_failure(
        clip_site(figure=False), "expected 1 figure(s), 2 theme wrappers; found 0"
    )


def test_an_embed_escaped_as_text(clip_site: Callable[..., Path]) -> None:
    site = clip_site()
    _edit(site, QUARTO_FIGURE, f"<pre><code>{html.escape(QUARTO_FIGURE)}</code></pre>")
    _single_failure(site, "expected 1 figure(s), 2 theme wrappers; found 0")


def test_a_plot_where_a_clip_is_listed(clip_site: Callable[..., Path]) -> None:
    site = clip_site(figure=False, before=_pair(1, 450))
    _single_failure(site, "figure 1: expected a clip, found a plot wrapper")


def test_a_clip_where_a_plot_is_listed(
    clip_site: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    site = clip_site(media=False)
    _set_manifest(monkeypatch, (Fig("plot", 420),))
    _single_failure(site, "figure 1: expected a plot, found a clip video")


def test_a_themed_video_outside_a_replay_figure(
    clip_site: Callable[..., Path],
) -> None:
    site = clip_site()
    _edit(site, '<figure class="replay figure">', "<div>")
    _edit(site, "</figure>", "</div>")
    _single_failure(site, "figure 1: expected a clip, found a plot wrapper")


def test_dark_video_before_light(clip_site: Callable[..., Path]) -> None:
    site = clip_site()
    light, dark = _video("light", {}), _video("dark", {})
    _edit(site, f"{light}\n{dark}", f"{dark}\n{light}")
    _single_failure(
        site,
        "figure 1: expected light-content then dark-content, "
        "found dark-content then light-content",
    )


def test_videos_in_two_replay_figures(clip_site: Callable[..., Path]) -> None:
    site = clip_site()
    dark = _video("dark", {})
    _edit(site, dark, f'</figure>\n<figure class="replay figure">\n{dark}')
    _single_failure(site, 'the two videos are not in one <figure class="replay">')


def test_a_clip_out_of_order(
    clip_site: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_manifest(monkeypatch, (CLIP, Fig("plot", 420)))
    site = clip_site(before=_pair(1, 420))
    assert check_site_figures.figure_failures(site) == [
        f"{PAGE}: figure 1: expected a clip, found a plot wrapper",
        f"{PAGE}: figure 2: expected a plot, found a clip video",
    ]


def test_a_clip_on_an_unlisted_page(clip_site: Callable[..., Path]) -> None:
    site = clip_site()
    clip_site(page="story/other.html")
    _single_failure(
        site,
        "not in the figure manifest, but has 2 theme wrapper(s)",
        page="story/other.html",
    )


HIDING_WRAPPERS = {
    "d_none": ('<div class="d-none">', "</div>"),
    "hidden_attribute": ("<div hidden>", "</div>"),
    "aria_hidden": ('<section aria-hidden="true">', "</section>"),
    "inline_style": ('<div style="display : none">', "</div>"),
    "visibility": ('<div style="VISIBILITY:hidden">', "</div>"),
    "template": ("<template>", "</template>"),
    "noscript": ("<noscript>", "</noscript>"),
    "closed_details": ("<details><summary>The clip</summary>", "</details>"),
    "after_a_closed_sibling": (
        '<div class="collapse"><div class="row"></div>',
        "</div>",
    ),
}


@pytest.mark.parametrize(
    "around", list(HIDING_WRAPPERS.values()), ids=list(HIDING_WRAPPERS)
)
def test_a_clip_in_a_hidden_element(
    clip_site: Callable[..., Path], around: tuple[str, str]
) -> None:
    _single_failure(
        clip_site(around=around), "figure 1: the clip is inside a hidden element: <"
    )


def test_a_hidden_video(clip_site: Callable[..., Path]) -> None:
    site = clip_site(light={"class": "light-content d-none"})
    _single_failure(site, "figure 1: the clip is inside a hidden element: <")


@pytest.mark.parametrize(
    "opening",
    ["<details open>", '<details open="false">'],
    ids=["bare_open", "open_false"],
)
def test_a_clip_in_an_open_disclosure_passes(
    clip_site: Callable[..., Path], opening: str
) -> None:
    site = clip_site(around=(f"{opening}<summary>The clip</summary>", "</details>"))
    assert check_site_figures.figure_failures(site) == OK


def test_a_hidden_description(clip_site: Callable[..., Path]) -> None:
    site = clip_site()
    _edit(
        site,
        '<div id="demo-2d-text">',
        '<div id="demo-2d-text" class="visually-hidden">',
    )
    reason = "the description 'demo-2d-text' is inside a hidden element"
    _failures(site, reason, reason)


NESTED_HIDDEN_DESCRIPTION = """\
<div id="demo-2d-text">
<div class="d-none">
<p>Words.</p>
</div>
</div>
"""


def test_a_description_whose_text_is_all_inside_a_hidden_element(
    clip_site: Callable[..., Path],
) -> None:
    """The visible id wrapper does not lend its hidden paragraph's text.

    ``NESTED_HIDDEN_DESCRIPTION`` is what Quarto's Pandoc writes for a nested
    ``::: {.d-none}`` inside the description div.
    """
    site = clip_site()
    _edit(site, '<div id="demo-2d-text">\n<p>Words.</p>\n</div>\n', "")
    _edit(site, "</figure>\n", "</figure>\n" + NESTED_HIDDEN_DESCRIPTION)
    reason = (
        "the description 'demo-2d-text' has text only inside a hidden element: "
        '<div class="d-none">'
    )
    _failures(site, reason, reason)


@pytest.mark.parametrize(
    "raw",
    ["<style>.x { color: red; }</style>", "<script>var x = 1;</script>"],
    ids=["style", "script"],
)
def test_style_and_script_contents_are_not_description_text(
    clip_site: Callable[..., Path], raw: str
) -> None:
    """All the prose hidden, a visible style or script beside it: still none."""
    site = clip_site()
    _edit(site, '<div id="demo-2d-text">\n<p>Words.</p>\n</div>\n', "")
    described = NESTED_HIDDEN_DESCRIPTION.replace(
        '<div id="demo-2d-text">\n', f'<div id="demo-2d-text">\n{raw}\n'
    )
    _edit(site, "</figure>\n", "</figure>\n" + described)
    reason = (
        "the description 'demo-2d-text' has text only inside a hidden element: "
        '<div class="d-none">'
    )
    _failures(site, reason, reason)


def test_a_description_with_visible_text_beside_hidden_text_passes(
    clip_site: Callable[..., Path],
) -> None:
    site = clip_site()
    _edit(site, "<p>Words.</p>", '<p>Words.</p>\n<p class="d-none">More.</p>')
    assert check_site_figures.figure_failures(site) == OK


def test_wrong_clip_height(clip_site: Callable[..., Path]) -> None:
    site = clip_site(light={"height": "500"}, dark={"height": "500"})
    _failures(
        site,
        f"{PAGE}: figure 1 light-content: height 500, expected 450",
        f"{PAGE}: figure 1 dark-content: height 500, expected 450",
    )


def test_wrong_clip_width(clip_site: Callable[..., Path]) -> None:
    _single_failure(
        clip_site(dark={"width": "700"}),
        "figure 1 dark-content: width 700, expected 750",
    )


@pytest.mark.parametrize(
    "light",
    [
        {"flags": ("controls", "playsinline", "autoplay")},
        {"bare_flags": ("autoplay",)},
        {"autoplay": "true"},
    ],
    ids=["empty_value", "bare", "valued"],
)
def test_autoplay(clip_site: Callable[..., Path], light: dict[str, object]) -> None:
    _single_failure(clip_site(light=light), "figure 1 light-content: autoplay is set")


@pytest.mark.parametrize(
    "light",
    [
        {"flags": ("controls", "playsinline", "loop")},
        {"bare_flags": ("loop",)},
        {"loop": "false"},
    ],
    ids=["empty_value", "bare", "valued"],
)
def test_loop(clip_site: Callable[..., Path], light: dict[str, object]) -> None:
    _single_failure(clip_site(light=light), "figure 1 light-content: loop is set")


def test_no_controls(clip_site: Callable[..., Path]) -> None:
    _single_failure(
        clip_site(light={"flags": ("playsinline",)}),
        "figure 1 light-content: no controls attribute",
    )


def test_no_playsinline(clip_site: Callable[..., Path]) -> None:
    _single_failure(
        clip_site(light={"flags": ("controls",)}),
        "figure 1 light-content: no playsinline attribute",
    )


def test_preload_auto(clip_site: Callable[..., Path]) -> None:
    _single_failure(
        clip_site(dark={"preload": "auto"}),
        "figure 1 dark-content: preload 'auto', expected 'none'",
    )


def test_missing_aria_label(clip_site: Callable[..., Path]) -> None:
    _single_failure(
        clip_site(light={"aria-label": None}),
        "figure 1 light-content: empty aria-label",
    )


def test_blank_aria_label(clip_site: Callable[..., Path]) -> None:
    _single_failure(
        clip_site(light={"aria-label": "  "}),
        "figure 1 light-content: empty aria-label",
    )


def test_no_aria_describedby(clip_site: Callable[..., Path]) -> None:
    _failures(
        clip_site(dark={"aria-describedby": None}),
        "figure 1 dark-content: no aria-describedby",
        "figure 1: aria-describedby differs between themes: 'demo-2d-text' vs None",
    )


def test_missing_description(clip_site: Callable[..., Path]) -> None:
    reason = "aria-describedby 'demo-2d-text' names no element on the page"
    _failures(clip_site(description=None), reason, reason)


def test_empty_description(clip_site: Callable[..., Path]) -> None:
    reason = "the description 'demo-2d-text' is empty"
    _failures(clip_site(description="  "), reason, reason)


def test_description_differs_between_themes(clip_site: Callable[..., Path]) -> None:
    site = clip_site(
        dark={"aria-describedby": "other-text"},
        around=("", '<p id="other-text">Other words.</p>\n'),
    )
    _single_failure(
        site, "aria-describedby differs between themes: 'demo-2d-text' vs 'other-text'"
    )


def test_two_elements_share_the_description_id(clip_site: Callable[..., Path]) -> None:
    site = clip_site(around=("", '<p id="demo-2d-text">More words.</p>\n'))
    _failures(site, "names 2 elements", "names 2 elements")


@pytest.mark.parametrize(
    ("figure_class", "expected"),
    [
        ("replay figure", OK),
        (
            "replay",
            [
                f"{PAGE}: figure 1: the replay figure's classes are ['replay'], "
                "expected ['figure', 'replay']"
            ],
        ),
        (
            "replay figure wide",
            [
                f"{PAGE}: figure 1: the replay figure's classes are "
                "['figure', 'replay', 'wide'], expected ['figure', 'replay']"
            ],
        ),
        ("figure replay", OK),
    ],
    ids=["as_quarto_writes_it", "as_in_the_source", "an_extra_class", "reversed_order"],
)
def test_the_replay_figure_classes(
    clip_site: Callable[..., Path], figure_class: str, expected: list[str]
) -> None:
    site = clip_site(figure_class=figure_class)
    assert check_site_figures.figure_failures(site) == expected


@pytest.mark.parametrize(
    "flags",
    [
        {},
        {"bare_flags": ("controls", "playsinline")},
        {"controls": "false", "playsinline": "false"},
    ],
    ids=["as_quarto_writes_them", "bare", "valued"],
)
def test_boolean_attributes_in_either_form(
    clip_site: Callable[..., Path], flags: dict[str, object]
) -> None:
    site = clip_site(light=flags, dark=flags)
    assert check_site_figures.figure_failures(site) == OK


def test_no_caption(clip_site: Callable[..., Path]) -> None:
    _single_failure(clip_site(caption=None), "figure 1: expected 1 figcaption, found 0")


def test_empty_clip_caption(clip_site: Callable[..., Path]) -> None:
    _single_failure(clip_site(caption=" "), "figure 1: empty caption")


def test_missing_poster(clip_site: Callable[..., Path]) -> None:
    site = clip_site()
    (site / "media" / "demo_2d.dark.png").unlink()
    _single_failure(
        site, "figure 1 dark-content: poster '../media/demo_2d.dark.png' does not exist"
    )


def test_wrong_clip_name(clip_site: Callable[..., Path]) -> None:
    light, dark = (
        {
            "src": f"../media/demo_1d.{theme}.mp4",
            "poster": f"../media/demo_1d.{theme}.png",
        }
        for theme in ("light", "dark")
    )
    site = clip_site(light=light, dark=dark)
    for name in MEDIA_FILES:
        shutil.copyfile(
            site / "media" / name, site / "media" / name.replace("demo_2d", "demo_1d")
        )
    _failures(
        site,
        "video is 'media/demo_1d.light.mp4', expected media/demo_2d.light.mp4",
        "poster is 'media/demo_1d.light.png', expected media/demo_2d.light.png",
        "video is 'media/demo_1d.dark.mp4', expected media/demo_2d.dark.mp4",
        "poster is 'media/demo_1d.dark.png', expected media/demo_2d.dark.png",
        "media/demo_1d.dark.mp4: not one of the site's clips and posters",
        "media/demo_1d.dark.png: not one of the site's clips and posters",
        "media/demo_1d.light.mp4: not one of the site's clips and posters",
        "media/demo_1d.light.png: not one of the site's clips and posters",
    )


def test_a_same_named_clip_outside_media(clip_site: Callable[..., Path]) -> None:
    site = clip_site(light={"src": "../story/demo_2d.light.mp4"})
    shutil.copyfile(
        site / "media" / "demo_2d.light.mp4", site / "story" / "demo_2d.light.mp4"
    )
    _single_failure(
        site, "video is 'story/demo_2d.light.mp4', expected media/demo_2d.light.mp4"
    )


def test_wrong_theme_file(clip_site: Callable[..., Path]) -> None:
    _failures(
        clip_site(light={"src": "../media/demo_2d.dark.mp4"}),
        "video is 'media/demo_2d.dark.mp4', expected media/demo_2d.light.mp4",
        "light and dark use the same video file",
    )


def test_identical_light_and_dark_videos(clip_site: Callable[..., Path]) -> None:
    site = clip_site()
    media = site / "media"
    shutil.copyfile(media / "demo_2d.light.mp4", media / "demo_2d.dark.mp4")
    _single_failure(site, "figure 1: light and dark videos have the same content")


def test_identical_light_and_dark_posters(clip_site: Callable[..., Path]) -> None:
    site = clip_site()
    media = site / "media"
    shutil.copyfile(media / "demo_2d.light.png", media / "demo_2d.dark.png")
    _single_failure(site, "figure 1: light and dark posters have the same content")


def test_poster_of_the_wrong_size(clip_site: Callable[..., Path]) -> None:
    site = clip_site()
    _write_poster(site / "media" / "demo_2d.light.png", "light", 450, 750)
    _single_failure(
        site, "light-content: poster is 750 by 450 pixels, expected 1500 by 900"
    )


def _jpeg(path: Path, size: tuple[int, int]) -> None:
    Image.new("RGB", size, (200, 100, 50)).save(path, format="JPEG")


@pytest.mark.parametrize("kind", ["text", "a_real_jpeg"])
def test_poster_that_is_not_a_png(clip_site: Callable[..., Path], kind: str) -> None:
    site = clip_site()
    poster = site / "media" / "demo_2d.light.png"
    if kind == "text":
        poster.write_bytes(b"JPEG")
    else:
        _jpeg(poster, (FRAME_WIDTH, FRAME_HEIGHT))
    _single_failure(site, "figure 1 light-content: poster cannot be opened as a PNG")


def _png(
    width: int, height: int, *, filter_byte: int = 0, data: bytes | None = None
) -> bytes:
    """An 8-bit RGB PNG, every chunk's checksum right, built with struct and zlib."""

    def chunk(kind: bytes, body: bytes) -> bytes:
        crc = struct.pack(">I", zlib.crc32(kind + body))
        return struct.pack(">I", len(body)) + kind + body + crc

    if data is None:
        row = bytes([filter_byte]) + bytes(3 * width)
        data = row * height
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(data))
        + chunk(b"IEND", b"")
    )


def _flip_idat_byte(data: bytes) -> bytes:
    """Change one byte of the first ``IDAT`` chunk's data, leaving its checksum."""
    changed = bytearray(data)
    changed[data.index(b"IDAT") + 4 + 2] ^= 0xFF
    return bytes(changed)


def test_a_poster_cut_after_its_header(clip_site: Callable[..., Path]) -> None:
    site = clip_site()
    poster = site / "media" / "demo_2d.light.png"
    poster.write_bytes(poster.read_bytes()[:33])
    _single_failure(site, "figure 1 light-content: poster cannot be opened as a PNG")


def _damage_light_poster(
    clip_site: Callable[..., Path], damage: Callable[[bytes], bytes]
) -> Path:
    site = clip_site()
    poster = site / "media" / "demo_2d.light.png"
    poster.write_bytes(damage(poster.read_bytes()))
    return site


DOES_NOT_DECODE = "figure 1 light-content: poster does not decode"


def test_a_poster_cut_in_its_image_data(clip_site: Callable[..., Path]) -> None:
    site = _damage_light_poster(clip_site, lambda data: data[: len(data) // 2])
    _single_failure(site, DOES_NOT_DECODE)


def test_a_poster_with_a_damaged_chunk(clip_site: Callable[..., Path]) -> None:
    _single_failure(_damage_light_poster(clip_site, _flip_idat_byte), DOES_NOT_DECODE)


def test_a_poster_with_invalid_filter_bytes(clip_site: Callable[..., Path]) -> None:
    """Only the pixel load catches this: verify() passes it (Q15)."""
    site = _damage_light_poster(
        clip_site, lambda _: _png(FRAME_WIDTH, FRAME_HEIGHT, filter_byte=9)
    )
    _single_failure(site, DOES_NOT_DECODE)


def test_a_poster_without_its_end_chunk(clip_site: Callable[..., Path]) -> None:
    """Only verify() catches this: the pixel load passes it (Q15)."""
    _single_failure(
        _damage_light_poster(clip_site, lambda data: data[:-12]), DOES_NOT_DECODE
    )


def test_a_truncated_video(clip_site: Callable[..., Path]) -> None:
    site = clip_site()
    video = site / "media" / "demo_2d.dark.mp4"
    video.write_bytes(video.read_bytes()[: video.stat().st_size // 2])
    failures = check_site_figures.figure_failures(site)
    assert len(failures) == 1, failures
    match = re.fullmatch(
        rf"{PAGE}: figure 1 dark-content: video "
        r"(cannot be decoded: .*|decodes to (\d+) frames, expected 92)",
        failures[0],
        flags=re.DOTALL,
    )
    assert match, failures
    if match.group(2) is not None:
        assert 1 <= int(match.group(2)) <= 91


def test_a_video_of_the_wrong_frame_count(
    clip_site: Callable[..., Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = clip_site()
    archives = tmp_path / "other_archives"
    archives.mkdir()
    one = demo_2d.record(n_observations=1, n_particles=16)
    save_replay(archives / "demo_2d.npz", one)
    monkeypatch.setattr(check_site_figures, "ARCHIVES", archives)
    reason = "video decodes to 92 frames, expected 74"
    _failures(site, f"light-content: {reason}", f"dark-content: {reason}")


def test_a_video_in_another_codec(clip_site: Callable[..., Path]) -> None:
    site = clip_site()
    video = site / "media" / "demo_2d.light.mp4"
    _transcode(video, "-c:v", "mpeg4", "-movflags", "+faststart")
    _single_failure(site, "light-content: video codec mpeg4, expected h264")


def test_a_video_in_another_pixel_format(clip_site: Callable[..., Path]) -> None:
    site = clip_site()
    video = site / "media" / "demo_2d.light.mp4"
    _transcode(
        video, "-c:v", "libx264", "-pix_fmt", "yuv444p", "-movflags", "+faststart"
    )
    _single_failure(site, "light-content: video pixel format yuv444p, expected yuv420p")


def test_a_video_at_another_rate(clip_site: Callable[..., Path]) -> None:
    site = clip_site()
    _encode(site / "media" / "demo_2d.light.mp4", _frames("light"), rate=25)
    _single_failure(
        site, "light-content: video runs at 25 frames per second, expected 20"
    )


def test_a_video_of_another_pixel_size(clip_site: Callable[..., Path]) -> None:
    site = clip_site()
    _encode(
        site / "media" / "demo_2d.light.mp4",
        _frames("light", height=840),
        height=840,
    )
    _single_failure(
        site, "light-content: video is 1500 by 840 pixels, expected 1500 by 900"
    )


def test_a_video_without_faststart(clip_site: Callable[..., Path]) -> None:
    site = clip_site()
    _transcode(site / "media" / "demo_2d.light.mp4", "-c", "copy")
    _single_failure(site, "light-content: video has mdat before moov")


def test_a_file_that_is_not_a_video(clip_site: Callable[..., Path]) -> None:
    site = clip_site()
    (site / "media" / "demo_2d.light.mp4").write_bytes(b"not a video")
    _single_failure(site, "figure 1 light-content: video cannot be decoded:")


def test_an_encoded_nul_in_a_video_path(clip_site: Callable[..., Path]) -> None:
    site = clip_site(dark={"src": "../media/demo_2d.dark%00.mp4"})
    _single_failure(site, "dark-content: video path contains a NUL")


def test_a_video_outside_the_site(clip_site: Callable[..., Path]) -> None:
    site = clip_site(light={"src": "/media/demo_2d.light.mp4"})
    _single_failure(site, "light-content: video is outside the site")


def test_no_archive_for_the_clip(
    clip_site: Callable[..., Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = clip_site()
    empty = tmp_path / "no_archives"
    empty.mkdir()
    monkeypatch.setattr(check_site_figures, "ARCHIVES", empty)
    _single_failure(site, "figure 1: no readable archive for clip 'demo_2d'")


def test_the_archive_is_read_once_per_clip(
    clip_site: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    site = clip_site()
    calls: list[Path] = []

    def spy(path: Path) -> object:
        calls.append(path)
        return load_replay(path)

    monkeypatch.setattr(check_site_figures, "load_replay", spy)
    assert check_site_figures.figure_failures(site) == OK
    assert calls == [check_site_figures.ARCHIVES / "demo_2d.npz"]


def test_no_encoder(
    clip_site: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    site = clip_site()

    def missing() -> str:
        raise EncoderNotFound("none")

    monkeypatch.setattr(check_site_figures, "find_encoder", missing)
    assert check_site_figures.figure_failures(site) == ["clips cannot be decoded: none"]


@pytest.mark.parametrize(
    "name",
    ["old.mp4", "demo_1d.light.png", "sub"],
    ids=["old_clip", "other_demo", "a_directory"],
)
def test_an_extra_file_in_media(clip_site: Callable[..., Path], name: str) -> None:
    site = clip_site()
    if name == "sub":
        (site / "media" / "sub").mkdir()
    else:
        (site / "media" / name).write_bytes(b"stale")
    assert check_site_figures.figure_failures(site) == [
        f"media/{name}: not one of the site's clips and posters"
    ]


def test_media_without_any_listed_clip(tmp_path: Path) -> None:
    site = _build(tmp_path)
    (site / "media").mkdir()
    (site / "media" / "demo_2d.light.mp4").write_bytes(b"stale")
    assert check_site_figures.figure_failures(site) == [
        "media/demo_2d.light.mp4: not one of the site's clips and posters"
    ]


def _root_page(build: Callable[..., Path], monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(check_site_figures, "EXPECTED_FIGURES", {"index.html": (CLIP,)})
    light, dark = (
        {"src": f"media/demo_2d.{theme}.mp4", "poster": f"media/demo_2d.{theme}.png"}
        for theme in ("light", "dark")
    )
    return build(page="index.html", link="replay.css", light=light, dark=dark)


def _copy_elsewhere(build: Callable[..., Path], _: pytest.MonkeyPatch) -> Path:
    site = build(link="replay.css")
    (site / "story" / "replay.css").write_bytes(CANONICAL)
    return site


def _second_page_unlinked(
    build: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> Path:
    monkeypatch.setattr(
        check_site_figures,
        "EXPECTED_FIGURES",
        {PAGE: (CLIP,), "story/other.html": (CLIP,)},
    )
    build()
    return build(page="story/other.html", link=None)


def _built(**options: object) -> Callable[..., Path]:
    return lambda build, _: build(**options)


UNLINKED = [f"{PAGE}: no stylesheet link to replay.css"]
NOT_CANONICAL = ["replay.css: not the canonical rule"]
STYLESHEET_CASES = {
    "canonical": (_built(), OK),
    "link_with_a_query": (_built(link="../replay.css?v=1"), OK),
    "root_page": (_root_page, OK),
    "capitalised_rel": (_built(link_rel="Stylesheet"), OK),
    "missing_file": (_built(replay_css=None), ["replay.css: missing from the site"]),
    "one_character_changed": (
        _built(replay_css=CANONICAL.replace(b"100%", b"99%")),
        NOT_CANONICAL,
    ),
    "a_rule_added": (
        _built(replay_css=CANONICAL + b"video { display: none; }\n"),
        NOT_CANONICAL,
    ),
    "no_final_newline": (_built(replay_css=CANONICAL[:-1]), NOT_CANONICAL),
    "commented_out": (_built(replay_css=b"/* " + CANONICAL + b" */"), NOT_CANONICAL),
    "no_link": (_built(link=None), UNLINKED),
    "link_to_another_file": (_built(link="../styles.css"), UNLINKED),
    "link_to_a_copy_elsewhere": (_copy_elsewhere, UNLINKED),
    "root_relative_link": (_built(link="/replay.css"), UNLINKED),
    "not_a_stylesheet_link": (_built(link_rel="preload"), UNLINKED),
    "malformed_link": (_built(link="//[bad"), UNLINKED),
    "nul_in_link": (_built(link="../replay%00.css"), UNLINKED),
    "second_clip_page_unlinked": (
        _second_page_unlinked,
        ["story/other.html: no stylesheet link to replay.css"],
    ),
}


@pytest.mark.parametrize(
    ("make", "expected"), list(STYLESHEET_CASES.values()), ids=list(STYLESHEET_CASES)
)
def test_the_replay_stylesheet(
    clip_site: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    make: Callable[[Callable[..., Path], pytest.MonkeyPatch], Path],
    expected: list[str],
) -> None:
    """There is no case about how the rule is spelt: nothing parses it, and the
    file is the canonical bytes or it is not (decision D45)."""
    site = make(clip_site, monkeypatch)
    assert check_site_figures.figure_failures(site) == expected


def _write_png(data: bytes) -> Callable[[Path], Path]:
    def write(directory: Path) -> Path:
        path = directory / "poster.png"
        path.write_bytes(data)
        return path

    return write


def _write_jpeg(directory: Path) -> Path:
    path = directory / "poster.png"
    _jpeg(path, (2, 2))
    return path


def _a_directory(directory: Path) -> Path:
    path = directory / "poster.png"
    path.mkdir()
    return path


WHOLE = _png(2, 2)
READ_POSTER_CASES = {
    "whole": (_write_png(WHOLE), (2, 2), None),
    "text": (_write_png(b"JPEG"), None, "cannot be opened as a PNG"),
    "jpeg": (_write_jpeg, None, "cannot be opened as a PNG"),
    "invalid_filter": (_write_png(_png(2, 2, filter_byte=9)), None, "does not decode"),
    "short_data": (_write_png(_png(2, 2, data=b"\0\0\0")), None, "does not decode"),
    "no_iend": (_write_png(WHOLE[:-12]), None, "does not decode"),
    "bad_checksum": (_write_png(_flip_idat_byte(WHOLE)), None, "does not decode"),
    "a_directory": (_a_directory, None, "cannot be read"),
}


@pytest.mark.parametrize(
    ("write", "size", "problem"),
    list(READ_POSTER_CASES.values()),
    ids=list(READ_POSTER_CASES),
)
def test_read_poster(
    tmp_path: Path,
    write: Callable[[Path], Path],
    size: tuple[int, int] | None,
    problem: str | None,
) -> None:
    found_size, found_problem = check_site_figures._read_poster(write(tmp_path))
    assert found_size == size
    if problem is None:
        assert found_problem is None
    else:
        assert found_problem is not None and found_problem.startswith(problem)


def test_plot_pages_need_no_stylesheet_or_encoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        check_site_figures, "find_encoder", lambda: pytest.fail("encoder looked up")
    )
    assert check_site_figures.figure_failures(_build(tmp_path)) == OK


def test_pages_under_site_libs_are_not_scanned(tmp_path: Path) -> None:
    site = _build(tmp_path)
    _page(site, [_pair(1, 420)[0]], page="site_libs/x/y.html")
    assert check_site_figures.figure_failures(site) == OK


@pytest.mark.parametrize("order", ["in_order", "dark_first"])
def test_an_unknown_kind_is_a_programming_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, order: str
) -> None:
    _set_manifest(monkeypatch, (Fig("map", 420),))
    light, dark = _pair(1, 420)
    site = _page(tmp_path, [light, dark] if order == "in_order" else [dark, light])
    with pytest.raises(ValueError, match="figure 1: no check for kind 'map'"):
        check_site_figures.figure_failures(site)
