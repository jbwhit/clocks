"""Check the figure contract for the rendered site.

Pages draw their plots twice each, once per colour theme. Quarto wraps the
two outputs in elements with the classes `light-content` and `dark-content`,
and the site shows one at a time. A clip is embedded as two themed `<video>`
elements in one `<figure class="replay">`, which Quarto writes as
`class="replay figure"`. `EXPECTED_FIGURES` lists, for each page, its figures
in document order, each a plot or a clip. This check parses the rendered HTML
and requires, for every listed page:

- exactly two themed elements per figure, in document order light then dark,
  none carrying both theme classes and no plot wrapper nested inside another;
  a themed `<video>` inside a replay figure and outside any plot wrapper is a
  clip, any other themed element a plot wrapper, and each figure must be of the
  kind its manifest entry names;
- for a plot: exactly one image in each wrapper, declared 750 pixels wide and
  the figure's expected height, whose URL path (query and fragment dropped)
  names a file inside the site, the two files existing, different, and
  differing in content; exactly one `figcaption` in each wrapper, with the same
  text in both;
- for a clip: both videos in one replay figure whose classes are exactly
  `replay` and `figure`, with one non-empty `figcaption`; each video declared
  750 by the clip's height, with `controls`, `playsinline` and
  `preload="none"`, no `autoplay` or `loop`, a non-empty `aria-label` and an
  `aria-describedby` naming exactly one non-empty element on the page, the
  same for both themes; `src` and `poster` naming `media/<clip>.<theme>.mp4`
  and `.png`, light differing from dark; a poster that Pillow opens as a PNG,
  verifies and loads, at twice the declared size; and a stream that decodes
  (with the encoder, `clocks._video.probe`) as H.264 `yuv420p` at twice the
  declared size, 20 frames per second, `moov` before `mdat`, to the frame count
  the dwell schedule gives the clip's archive under `assets/replays/`, read
  once per clip;
- no clip, and no element its `aria-describedby` names, inside an element
  marked hidden (decision D39): a class in `HIDING_CLASSES`, the attribute
  `hidden`, `aria-hidden="true"`, an inline `display:none` or
  `visibility:hidden`, a `<template>`, a `<noscript>` or a `<details>` without
  `open`; and a description's text is counted only outside such elements
  (and outside `<style>` and `<script>`, whose contents are code), so
  a visible description element whose every word sits inside a nested hidden
  one fails as having text only inside a hidden element.

A page not in the manifest must hold no themed element at all. Images outside
the wrappers are ignored. The output's `media/` must hold only
the clips and posters of the clips the manifest lists. When the manifest lists
a clip, the encoder is looked up once (its absence is one failure; the other
checks still run), the output's `replay.css` must be exactly `REPLAY_CSS`, and
each page with a clip must link it; a plot-only manifest needs neither.

The limit. The check refuses only the markers above; it does not compute
visibility: a stylesheet rule, a script, an ancestor's opacity, an ancestor
that clips it (`clip-path`, or `overflow: hidden` on a box too small to hold
it) or any other marker can still hide a clip, and only the browser check
(Task 13), which looks at the painted page, finds those. It establishes that a
clip page links a `replay.css` holding exactly the canonical rule, not that the
rule takes effect: another stylesheet or a script can override it, and the
390-pixel browser check is what shows a clip shrinking with its column. No
stylesheet is parsed (decision D45). It establishes that a clip has the frame
count of its archive's schedule, not that its frames show that archive; the
full render, which redraws every clip from `assets/replays/` and fails the
build when it cannot, ties the published files to their inputs (decision D27).

Run from the repo root, after rendering the site:

    cd site && uv run --frozen quarto render && cd ..
    uv run python scripts/check_site_figures.py
"""

from __future__ import annotations

import sys
import zlib
from collections.abc import Iterator, Sequence
from html.parser import HTMLParser
from pathlib import Path
from typing import NamedTuple
from urllib.parse import unquote, urlsplit

from PIL import Image, UnidentifiedImageError

from clocks._replay import load_replay
from clocks._video import FPS, EncoderNotFound, dwell_frames, find_encoder, probe

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "site" / "_output"
ARCHIVES = ROOT / "assets" / "replays"
MEDIA = "media"  # the output's directory of clips and posters
HIDING_CLASSES = frozenset(
    {"d-none", "hidden", "invisible", "visually-hidden", "collapse"}
)
REPLAY_CSS = b"figure.replay video { max-width: 100%; height: auto; }\n"

THEMES = ("light", "dark")
EXPECTED_WIDTH = 750
_KINDS = ("plot", "clip")
_POSTER_ERRORS = (OSError, SyntaxError, ValueError, zlib.error)

# Elements with no end tag; one of these carrying a theme class wraps nothing.
_VOID_TAGS = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta"}
    | {"source", "track", "wbr"}
)
# Elements whose contents are code, never counted as a description's text.
_RAW_TEXT_TAGS = frozenset({"style", "script"})


class Fig(NamedTuple):
    """One figure on a page: its kind, declared image height, and a clip name."""

    kind: str
    height: int
    clip: str | None = None


# Page (relative to the output directory) -> its figures, in document order.
EXPECTED_FIGURES: dict[str, tuple[Fig, ...]] = {
    "index.html": (Fig("clip", 450, "demo_2d"),),
    "story/clocks-as-gravimeters.html": (Fig("plot", 420),),
    "story/one-clock-is-not-enough.html": (Fig("plot", 420),),
    "story/the-search-in-one-dimension.html": (
        Fig("clip", 450, "demo_1d"),
        Fig("plot", 540),
    ),
    "story/into-the-plane.html": (Fig("clip", 450, "demo_2d"), Fig("plot", 540)),
    "story/two-hidden-masses.html": (
        Fig("clip", 500, "demo_multi_mass"),
        Fig("clip", 550, "demo_multi_mass_2d"),
        Fig("plot", 420),
    ),
    "story/how-many-masses.html": (
        Fig("clip", 420, "demo_model_comparison"),
        Fig("plot", 420),
    ),
    "story/beyond-point-masses.html": (Fig("plot", 540), Fig("plot", 420)),
    "story/gravitational-echolocation.html": (
        Fig("plot", 420),
        Fig("clip", 550, "demo_echolocation_3d"),
        Fig("plot", 540),
    ),
    "method/the-particle-filter.html": (Fig("plot", 540),),
}


class _Open:
    """An open element whose end is found by counting its own tag."""

    def __init__(self, tag: str, serial: int) -> None:
        self.tag = tag
        self.serial = serial  # the order in which tracked elements opened
        self.depth = 1


class _Wrapper(_Open):
    """One plot wrapper: an element carrying a theme class, and what it contains."""

    def __init__(self, tag: str, themes: list[str], serial: int) -> None:
        super().__init__(tag, serial)
        self.themes = themes
        self.nested = False
        self.images: list[dict[str, str | None]] = []
        self.captions: list[str] = []


class _Video:
    """One clip rendering: a themed ``<video>`` inside a replay figure."""

    tag = "video"

    def __init__(
        self,
        themes: list[str],
        attrs: dict[str, str | None],
        figure: int,
        hidden_by: str | None,
    ) -> None:
        self.themes = themes
        # A boolean attribute is present when its key is: bare (None), as
        # Quarto writes it (""), or with any other value, as HTML reads it.
        self.attrs = attrs
        self.figure = figure  # index into the parser's replay_* lists
        self.hidden_by = hidden_by


class _ReplayFigure(_Open):
    def __init__(self, index: int, serial: int) -> None:
        super().__init__("figure", serial)
        self.index = index


class _Hider(_Open):
    def __init__(self, tag: str, start_tag: str, serial: int) -> None:
        super().__init__(tag, serial)
        self.start_tag = start_tag


class _Text:
    """The text of one element carrying an ``id``, and what hides it.

    ``parts`` holds only the text outside every hiding element that opened
    inside this one, and outside ``<style>`` and ``<script>``;
    ``hidden_text_by`` is the start tag of the first such hiding element that
    held text, or ``None``.
    """

    def __init__(self, hidden_by: str | None) -> None:
        self.parts: list[str] = []
        self.hidden_by = hidden_by
        self.hidden_text_by: str | None = None

    @property
    def text(self) -> str:
        return " ".join("".join(self.parts).split())


class _TextElement(_Open):
    def __init__(self, tag: str, entry: _Text, serial: int) -> None:
        super().__init__(tag, serial)
        self.entry = entry


class _FigureParser(HTMLParser):
    """Collect each page's themed elements and what the checks need about them.

    Any element whose class list names `light-content` or `dark-content` is
    themed: a clip rendering if it is a `<video>` that opens inside a replay
    figure (a `<figure>` whose class list holds `replay`) and outside any plot
    wrapper, else a plot wrapper. A wrapper's end is found by counting its own
    tag, so a `section` wrapper with `div`s inside closes at its own
    `</section>`; replay figures, hiding elements and elements with an `id` are
    tracked the same way. Images belong to the innermost open wrapper; a
    `figcaption` to the innermost open wrapper or replay figure; a wrapper that
    opens while another is open is marked as nested.
    """

    def __init__(self) -> None:
        super().__init__()
        self.themed: list[_Wrapper | _Video] = []
        self.replay_captions: list[list[str]] = []
        self.replay_classes: list[list[str]] = []
        self.texts: dict[str, list[_Text]] = {}
        self.stylesheets: list[str] = []
        self._open: list[_Wrapper] = []
        self._figures: list[_ReplayFigure] = []
        self._hiders: list[_Hider] = []
        self._text_elements: list[_TextElement] = []
        self._caption: tuple[list[str], list[str]] | None = None
        self._serial = 0
        self._in_raw_text = 0  # inside a <style> or <script>, whose text is code

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _RAW_TEXT_TAGS:
            self._in_raw_text += 1
        for element in self._tracked():
            if element.tag == tag:
                element.depth += 1
        self._serial += 1
        attributes = dict(attrs)
        classes = (attributes.get("class") or "").split()
        start_tag = self.get_starttag_text() or f"<{tag}>"
        marked = _is_marked_hidden(tag, attributes, classes)
        if self._hiders:
            hidden_by: str | None = self._hiders[0].start_tag
        else:
            hidden_by = start_tag if marked else None
        void = tag in _VOID_TAGS
        if marked and not void:
            self._hiders.append(_Hider(tag, start_tag, self._serial))
        if (
            tag == "link"
            and "stylesheet" in (attributes.get("rel") or "").lower().split()
        ):
            self.stylesheets.append(attributes.get("href") or "")
        if attributes.get("id") is not None:
            self._start_text(tag, attributes["id"] or "", hidden_by, void)
        themes = [theme for theme in THEMES if f"{theme}-content" in classes]
        if themes:
            self._start_themed(tag, themes, attributes, hidden_by, void)
        elif self._open and tag == "img":
            self._open[-1].images.append(attributes)
        elif tag == "figcaption":
            self._start_caption()
        if tag == "figure" and "replay" in classes:
            index = len(self.replay_classes)
            self.replay_classes.append(classes)
            self.replay_captions.append([])
            self._figures.append(_ReplayFigure(index, self._serial))

    def handle_endtag(self, tag: str) -> None:
        if tag in _RAW_TEXT_TAGS and self._in_raw_text:
            self._in_raw_text -= 1
        if tag == "figcaption" and self._caption is not None:
            sink, parts = self._caption
            sink.append(" ".join("".join(parts).split()))
            self._caption = None
        for element in self._tracked():
            if element.tag == tag:
                element.depth -= 1
        self._open = [e for e in self._open if e.depth > 0]
        self._figures = [e for e in self._figures if e.depth > 0]
        self._hiders = [e for e in self._hiders if e.depth > 0]
        self._text_elements = [e for e in self._text_elements if e.depth > 0]

    def handle_data(self, data: str) -> None:
        if self._caption is not None:
            self._caption[1].append(data)
        if self._in_raw_text:
            return  # style and script contents are not description text
        for element in self._text_elements:
            inner = next((h for h in self._hiders if h.serial > element.serial), None)
            if inner is None:
                element.entry.parts.append(data)
            elif data.strip() and element.entry.hidden_text_by is None:
                element.entry.hidden_text_by = inner.start_tag

    def _tracked(self) -> Iterator[_Open]:
        yield from self._open
        yield from self._figures
        yield from self._hiders
        yield from self._text_elements

    def _start_text(
        self, tag: str, element_id: str, hidden_by: str | None, void: bool
    ) -> None:
        entry = _Text(hidden_by)
        self.texts.setdefault(element_id, []).append(entry)
        if not void:
            self._text_elements.append(_TextElement(tag, entry, self._serial))

    def _start_themed(
        self,
        tag: str,
        themes: list[str],
        attributes: dict[str, str | None],
        hidden_by: str | None,
        void: bool,
    ) -> None:
        if tag == "video" and self._figures and not self._open:
            figure = self._figures[-1].index
            self.themed.append(_Video(themes, attributes, figure, hidden_by))
            return
        wrapper = _Wrapper(tag, themes, self._serial)
        wrapper.nested = bool(self._open)
        self.themed.append(wrapper)
        if not void:
            self._open.append(wrapper)

    def _start_caption(self) -> None:
        wrapper = self._open[-1] if self._open else None
        figure = self._figures[-1] if self._figures else None
        if figure is not None and (wrapper is None or figure.serial > wrapper.serial):
            self._caption = (self.replay_captions[figure.index], [])
        elif wrapper is not None:
            self._caption = (wrapper.captions, [])


def _is_marked_hidden(
    tag: str, attributes: dict[str, str | None], classes: list[str]
) -> bool:
    """Whether the start tag carries one of decision D39's hiding markers."""
    if tag in ("template", "noscript"):
        return True
    if tag == "details" and "open" not in attributes:
        return True
    if HIDING_CLASSES.intersection(classes) or "hidden" in attributes:
        return True
    if attributes.get("aria-hidden") == "true":
        return True
    style = "".join((attributes.get("style") or "").lower().split())
    return "display:none" in style or "visibility:hidden" in style


def figure_failures(output_dir: Path) -> list[str]:
    """Return one message per violated rule across the manifest and the site."""
    failures: list[str] = []
    lists_a_clip = bool(_listed_clips())
    encoder: str | None = None
    if lists_a_clip:
        try:
            encoder = find_encoder()
        except EncoderNotFound as error:
            failures.append(f"clips cannot be decoded: {error}")
    for page, figures in EXPECTED_FIGURES.items():
        path = output_dir / page
        if not path.is_file():
            failures.append(f"{page}: page is missing from {output_dir}")
            continue
        failures.extend(_page_failures(page, path, figures, output_dir, encoder))
    failures.extend(_unlisted_page_failures(output_dir))
    failures.extend(_media_set_failures(output_dir))
    if lists_a_clip:
        failures.extend(_stylesheet_failures(output_dir))
    return failures


def _listed_clips() -> set[str]:
    return {
        str(figure.clip)
        for figures in EXPECTED_FIGURES.values()
        for figure in figures
        if figure.kind == "clip"
    }


def _parse(path: Path) -> _FigureParser:
    parser = _FigureParser()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser


def _unlisted_page_failures(output_dir: Path) -> list[str]:
    failures: list[str] = []
    for path in sorted(output_dir.rglob("*.html")):
        relative = path.relative_to(output_dir)
        if "site_libs" in relative.parts or relative.as_posix() in EXPECTED_FIGURES:
            continue
        parser = _parse(path)
        if parser.themed:
            failures.append(
                f"{relative.as_posix()}: not in the figure manifest, "
                f"but has {len(parser.themed)} theme wrapper(s)"
            )
    return failures


def _media_set_failures(output_dir: Path) -> list[str]:
    """Every entry directly inside the output's media/ must be a listed clip's."""
    directory = output_dir / MEDIA
    if not directory.is_dir():
        return []
    expected = {
        f"{clip}.{theme}.{suffix}"
        for clip in _listed_clips()
        for theme in THEMES
        for suffix in ("mp4", "png")
    }
    return [
        f"{MEDIA}/{entry.name}: not one of the site's clips and posters"
        for entry in sorted(directory.iterdir())
        if entry.name not in expected
    ]


def _stylesheet_failures(output_dir: Path) -> list[str]:
    path = output_dir / "replay.css"
    if not path.is_file():
        return ["replay.css: missing from the site"]
    if path.read_bytes() != REPLAY_CSS:
        return ["replay.css: not the canonical rule"]
    return []


def _page_failures(
    page: str,
    path: Path,
    figures: tuple[Fig, ...],
    output_dir: Path,
    encoder: str | None,
) -> list[str]:
    parser = _parse(path)
    failures = _figure_list_failures(page, parser, figures, path, output_dir, encoder)
    has_a_clip = any(figure.kind == "clip" for figure in figures)
    if has_a_clip and not _links_the_replay_stylesheet(
        parser.stylesheets, path, output_dir
    ):
        failures.append(f"{page}: no stylesheet link to replay.css")
    return failures


def _figure_list_failures(
    page: str,
    parser: _FigureParser,
    figures: tuple[Fig, ...],
    path: Path,
    output_dir: Path,
    encoder: str | None,
) -> list[str]:
    themed = parser.themed
    ambiguous = [e for e in themed if len(e.themes) > 1]
    if ambiguous:
        # Such an element counts for neither theme, so per-theme checks would
        # only repeat the same fault.
        return [f"{page}: <{e.tag}> wrapper has both theme classes" for e in ambiguous]
    # The outer wrapper is hidden in the other theme and takes the inner one
    # with it, so a nested wrapper never shows.
    failures = [
        f"{page}: <{e.tag}> {e.themes[0]}-content: theme wrapper nested inside another"
        for e in themed
        if isinstance(e, _Wrapper) and e.nested
    ]
    if len(themed) != 2 * len(figures):
        failures.append(
            f"{page}: expected {len(figures)} figure(s), "
            f"{2 * len(figures)} theme wrappers; found {len(themed)}"
        )
        return failures
    for index, figure in enumerate(figures, start=1):
        label = f"{page}: figure {index}"
        if figure.kind not in _KINDS:
            raise ValueError(f"{label}: no check for kind {figure.kind!r}")
        pair = themed[2 * index - 2 : 2 * index]
        problem = _kind_problem(figure.kind, pair) or _order_problem(pair)
        if problem:
            failures.append(f"{label}: {problem}")
        elif figure.kind == "plot":
            wrappers = [e for e in pair if isinstance(e, _Wrapper)]
            failures.extend(_plot_failures(label, wrappers, figure, path, output_dir))
        else:
            videos = [e for e in pair if isinstance(e, _Video)]
            failures.extend(
                _clip_failures(label, videos, figure, parser, path, output_dir, encoder)
            )
    return failures


def _kind_problem(kind: str, pair: list[_Wrapper | _Video]) -> str:
    if kind == "clip" and not all(isinstance(e, _Video) for e in pair):
        return "expected a clip, found a plot wrapper"
    if kind == "plot" and not all(isinstance(e, _Wrapper) for e in pair):
        return "expected a plot, found a clip video"
    return ""


def _order_problem(pair: list[_Wrapper | _Video]) -> str:
    if [e.themes[0] for e in pair] == list(THEMES):
        return ""
    found = " then ".join(f"{e.themes[0]}-content" for e in pair)
    return f"expected light-content then dark-content, found {found}"


def _plot_failures(
    label: str,
    pair: list[_Wrapper],
    figure: Fig,
    page_path: Path,
    output_dir: Path,
) -> list[str]:
    failures: list[str] = []
    files: dict[str, Path] = {}
    captions: dict[str, str] = {}
    for theme, wrapper in zip(THEMES, pair, strict=True):
        theme_label = f"{label} {theme}-content"
        failures.extend(_image_failures(theme_label, wrapper.images, figure.height))
        if len(wrapper.images) == 1:
            src = wrapper.images[0].get("src") or ""
            resolved, problem = _resolve_source(src, page_path, output_dir)
            if resolved is None:
                failures.append(f"{theme_label}: {problem}")
            else:
                files[theme] = resolved
        texts = wrapper.captions
        if len(texts) != 1:
            failures.append(f"{theme_label}: expected 1 figcaption, found {len(texts)}")
        elif not texts[0]:
            failures.append(f"{theme_label}: empty caption")
        else:
            captions[theme] = texts[0]
    failures.extend(_same_file_failures(label, "image", files))
    if len(captions) == 2 and captions["light"] != captions["dark"]:
        failures.append(
            f"{label}: captions differ: {captions['light']!r} vs {captions['dark']!r}"
        )
    return failures


def _same_file_failures(label: str, what: str, files: dict[str, Path]) -> list[str]:
    """The light and dark files must be two files with different content."""
    if len(files) != 2:
        return []
    if files["light"] == files["dark"]:
        return [f"{label}: light and dark use the same {what} file"]
    if files["light"].read_bytes() == files["dark"].read_bytes():
        return [f"{label}: light and dark {what}s have the same content"]
    return []


def _image_failures(
    label: str, images: list[dict[str, str | None]], height: int
) -> list[str]:
    if len(images) != 1:
        return [f"{label}: expected 1 image, found {len(images)}"]
    failures: list[str] = []
    for name, expected in (("width", EXPECTED_WIDTH), ("height", height)):
        if images[0].get(name) != str(expected):
            value = images[0].get(name)
            failures.append(f"{label}: {name} {value}, expected {expected}")
    return failures


def _clip_failures(
    label: str,
    pair: list[_Video],
    figure: Fig,
    parser: _FigureParser,
    page_path: Path,
    output_dir: Path,
    encoder: str | None,
) -> list[str]:
    light, dark = pair
    if light.figure != dark.figure:
        return [f'{label}: the two videos are not in one <figure class="replay">']
    failures = _replay_figure_failures(label, pair, parser)
    for theme, video in zip(THEMES, pair, strict=True):
        failures.extend(
            _video_attribute_failures(
                f"{label} {theme}-content", video.attrs, figure.height, parser.texts
            )
        )
    files: dict[str, dict[str, Path]] = {"video": {}, "poster": {}}
    intended: dict[str, Path] = {}
    for theme, video in zip(THEMES, pair, strict=True):
        failures.extend(
            _clip_file_failures(
                f"{label} {theme}-content",
                (theme, video.attrs, figure),
                (page_path, output_dir),
                files,
                intended,
            )
        )
    if encoder is not None:
        failures.extend(_decode_failures(label, figure, intended, encoder))
    described = [video.attrs.get("aria-describedby") for video in pair]
    if described[0] != described[1]:
        failures.append(
            f"{label}: aria-describedby differs between themes: "
            f"{described[0]!r} vs {described[1]!r}"
        )
    failures.extend(_same_file_failures(label, "video", files["video"]))
    failures.extend(_same_file_failures(label, "poster", files["poster"]))
    return failures


def _replay_figure_failures(
    label: str, pair: list[_Video], parser: _FigureParser
) -> list[str]:
    failures: list[str] = []
    index = pair[0].figure
    classes = parser.replay_classes[index]
    if sorted(classes) != ["figure", "replay"]:
        failures.append(
            f"{label}: the replay figure's classes are {sorted(classes)}, "
            "expected ['figure', 'replay']"
        )
    hidden_by = next((video.hidden_by for video in pair if video.hidden_by), None)
    if hidden_by:
        failures.append(f"{label}: the clip is inside a hidden element: {hidden_by}")
    captions = parser.replay_captions[index]
    if len(captions) != 1:
        failures.append(f"{label}: expected 1 figcaption, found {len(captions)}")
    elif not captions[0]:
        failures.append(f"{label}: empty caption")
    return failures


def _video_attribute_failures(
    label: str, attrs: dict[str, str | None], height: int, texts: dict[str, list[_Text]]
) -> list[str]:
    failures: list[str] = []
    for name, expected in (("width", EXPECTED_WIDTH), ("height", height)):
        if attrs.get(name) != str(expected):
            failures.append(f"{label}: {name} {attrs.get(name)}, expected {expected}")
    for name in ("controls", "playsinline"):
        if name not in attrs:
            failures.append(f"{label}: no {name} attribute")
    if attrs.get("preload") != "none":
        failures.append(f"{label}: preload {attrs.get('preload')!r}, expected 'none'")
    for name in ("autoplay", "loop"):
        if name in attrs:
            failures.append(f"{label}: {name} is set")
    if not (attrs.get("aria-label") or "").strip():
        failures.append(f"{label}: empty aria-label")
    problem = _description_problem(attrs.get("aria-describedby"), texts)
    if problem:
        failures.append(f"{label}: {problem}")
    return failures


def _description_problem(name: str | None, texts: dict[str, list[_Text]]) -> str:
    if not name or not name.strip():
        return "no aria-describedby"
    elements = texts.get(name, [])
    if not elements:
        return f"aria-describedby {name!r} names no element on the page"
    if len(elements) > 1:
        return f"aria-describedby {name!r} names {len(elements)} elements"
    if not elements[0].text and elements[0].hidden_text_by:
        return (
            f"the description {name!r} has text only inside a hidden element: "
            f"{elements[0].hidden_text_by}"
        )
    if not elements[0].text:
        return f"the description {name!r} is empty"
    if elements[0].hidden_by:
        return (
            f"the description {name!r} is inside a hidden element: "
            f"{elements[0].hidden_by}"
        )
    return ""


def _clip_file_failures(
    label: str,
    clip: tuple[str, dict[str, str | None], Fig],
    where: tuple[Path, Path],
    files: dict[str, dict[str, Path]],
    intended: dict[str, Path],
) -> list[str]:
    """Check one theme's video and poster files; record them in ``files``.

    ``intended`` receives the theme's video when it is the intended file.
    """
    theme, attrs, figure = clip
    page_path, output_dir = where
    failures: list[str] = []
    for what, attribute, suffix in (
        ("video", "src", "mp4"),
        ("poster", "poster", "png"),
    ):
        src = attrs.get(attribute) or ""
        path, problem = _resolve_source(src, page_path, output_dir, what)
        if path is None:
            failures.append(f"{label}: {problem}")
            continue
        files[what][theme] = path
        relative = path.relative_to(output_dir.resolve()).as_posix()
        expected = f"{MEDIA}/{figure.clip}.{theme}.{suffix}"
        if relative != expected:
            failures.append(f"{label}: {what} is {relative!r}, expected {expected}")
        elif what == "video":
            intended[theme] = path
        else:
            failures.extend(_poster_failures(label, path, figure.height))
    return failures


def _poster_failures(label: str, path: Path, height: int) -> list[str]:
    size, problem = _read_poster(path)
    if problem is not None or size is None:
        return [f"{label}: poster {problem}"]
    if size != (2 * EXPECTED_WIDTH, 2 * height):
        return [
            f"{label}: poster is {size[0]} by {size[1]} pixels, "
            f"expected {2 * EXPECTED_WIDTH} by {2 * height}"
        ]
    return []


def _decode_failures(
    label: str, figure: Fig, intended: dict[str, Path], encoder: str
) -> list[str]:
    """Read the clip's archive once, then decode each intended video."""
    failures: list[str] = []
    frames: int | None = None
    try:
        replay = load_replay(ARCHIVES / f"{figure.clip}.npz")
        frames = sum(dwell_frames(int(replay.meta["n_observations"])))
    except (ValueError, OSError) as error:
        failures.append(
            f"{label}: no readable archive for clip {figure.clip!r}: {error}"
        )
    for theme in THEMES:
        if theme in intended:
            failures.extend(
                _stream_failures(
                    f"{label} {theme}-content",
                    intended[theme],
                    encoder,
                    figure.height,
                    frames,
                )
            )
    return failures


def _stream_failures(
    label: str, path: Path, encoder: str, height: int, frames: int | None
) -> list[str]:
    try:
        info = probe(path, encoder)
    except ValueError as error:
        return [f"{label}: video cannot be decoded: {error}"]
    failures: list[str] = []
    if info.codec != "h264":
        failures.append(f"{label}: video codec {info.codec}, expected h264")
    if info.pix_fmt != "yuv420p":
        failures.append(f"{label}: video pixel format {info.pix_fmt}, expected yuv420p")
    if (info.width, info.height) != (2 * EXPECTED_WIDTH, 2 * height):
        failures.append(
            f"{label}: video is {info.width} by {info.height} pixels, "
            f"expected {2 * EXPECTED_WIDTH} by {2 * height}"
        )
    if info.fps != FPS:
        failures.append(
            f"{label}: video runs at {info.fps:g} frames per second, expected {FPS}"
        )
    if frames is not None and info.frames != frames:
        failures.append(
            f"{label}: video decodes to {info.frames} frames, expected {frames}"
        )
    if not info.moov_first:
        failures.append(f"{label}: video has mdat before moov")
    return failures


def _resolve_source(
    src: str, page_path: Path, output_dir: Path, what: str = "image"
) -> tuple[Path | None, str]:
    """Return the existing file ``src`` names, or None and what is wrong."""
    resolved, problem = _site_path(src, page_path, output_dir, what)
    if resolved is None:
        return None, problem
    if not resolved.is_file():
        return None, f"{what} {src!r} does not exist"
    return resolved, ""


def _site_path(
    src: str, page_path: Path, output_dir: Path, what: str = "image"
) -> tuple[Path | None, str]:
    """Return the path inside the site a URL names, or None and what is wrong.

    Only the URL's path names the file: a query or fragment is dropped and
    percent-escapes are decoded. The file must be referenced relative to the
    page and lie inside the site, so the deployed page can load it. An empty
    decoded path names no file.
    """
    try:
        url = urlsplit(src)
    except ValueError:
        return None, f"{what} URL is malformed: {src!r}"
    path = unquote(url.path)
    if "\x00" in path:
        return None, f"{what} path contains a NUL: {src!r}"
    # The decoded path is checked too: %2F decodes to a leading slash.
    if url.scheme or url.netloc or path.startswith("/"):
        return None, f"{what} is outside the site: {src!r}"
    if not path:
        return None, f"{what} {src!r} does not exist"
    resolved = (page_path.parent / path).resolve()
    if not resolved.is_relative_to(output_dir.resolve()):
        return None, f"{what} is outside the site: {src!r}"
    return resolved, ""


def _links_the_replay_stylesheet(
    hrefs: Sequence[str], page_path: Path, output_dir: Path
) -> bool:
    """Whether some ``href`` names the output's ``replay.css``.

    An ``href`` that ``_site_path`` refuses never counts. The file need not
    exist; its content is checked once per site.
    """
    target = (output_dir / "replay.css").resolve()
    return any(
        _site_path(href, page_path, output_dir, "stylesheet")[0] == target
        for href in hrefs
    )


def _read_poster(path: Path) -> tuple[tuple[int, int] | None, str | None]:
    """Return a poster's pixel size, or None and what is wrong with it.

    ``verify()`` and a pixel load are both needed: the first passes invalid
    filter bytes, the second a file with no ``IEND`` chunk (Q15).
    """
    try:
        image = Image.open(path)
    except UnidentifiedImageError:
        return None, "cannot be opened as a PNG"
    except OSError as error:
        return None, f"cannot be read: {type(error).__name__}"
    with image:
        if image.format != "PNG":
            return None, "cannot be opened as a PNG"
        try:
            image.verify()
        except _POSTER_ERRORS as error:
            return None, f"does not decode: {type(error).__name__}"
    try:
        with Image.open(path) as image:
            image.load()
            return image.size, None
    except _POSTER_ERRORS as error:
        return None, f"does not decode: {type(error).__name__}"


def main() -> int:
    failures = figure_failures(OUTPUT)
    if failures:
        print("Figure contract failures:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    figure_count = sum(len(figures) for figures in EXPECTED_FIGURES.values())
    print(
        f"Figure contract OK: {len(EXPECTED_FIGURES)} pages, "
        f"{figure_count} figures checked."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
