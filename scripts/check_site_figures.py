"""Check the figure contract for the rendered site.

Pages draw their plots twice each, once per colour theme. Quarto wraps the
two outputs in elements with the classes `light-content` and `dark-content`,
and the site shows one at a time. `EXPECTED_FIGURES` lists, for each page, its
figures in document order. This check parses the rendered HTML and requires,
for every listed page:

- exactly two wrappers per figure, in document order light then dark, none
  carrying both theme classes and none nested inside another;
- exactly one image in each wrapper, declared 750 pixels wide and the figure's
  expected height, whose URL path (query and fragment dropped) names a file
  inside the site, the two files existing, different, and differing in content;
- exactly one `figcaption` in each wrapper, with the same text in both.

A page not in the manifest must hold no theme wrapper at all. Images outside
the wrappers (committed GIFs and PNGs) are ignored.

Run from the repo root, after rendering the site:

    cd site && uv run --frozen quarto render && cd ..
    uv run python scripts/check_site_figures.py
"""

from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import NamedTuple
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "site" / "_output"

THEMES = ("light", "dark")
EXPECTED_WIDTH = 750

# Elements with no end tag; one of these carrying a theme class wraps nothing.
_VOID_TAGS = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta"}
    | {"source", "track", "wbr"}
)


class Fig(NamedTuple):
    """One figure on a page: its kind, declared image height, and a clip name."""

    kind: str
    height: int
    clip: str | None = None


# Page (relative to the output directory) -> its figures, in document order.
EXPECTED_FIGURES: dict[str, tuple[Fig, ...]] = {
    "story/clocks-as-gravimeters.html": (Fig("plot", 420),),
    "story/one-clock-is-not-enough.html": (Fig("plot", 420),),
    "story/the-search-in-one-dimension.html": (Fig("plot", 540),),
    "story/into-the-plane.html": (Fig("plot", 540),),
    "story/two-hidden-masses.html": (Fig("plot", 420),),
    "story/how-many-masses.html": (Fig("plot", 420),),
    "story/beyond-point-masses.html": (Fig("plot", 540), Fig("plot", 420)),
    "story/gravitational-echolocation.html": (Fig("plot", 420), Fig("plot", 540)),
    "method/the-particle-filter.html": (Fig("plot", 540),),
}


class _Wrapper:
    """One element carrying a theme class, and what it contains."""

    def __init__(self, tag: str, themes: list[str]) -> None:
        self.tag = tag
        self.themes = themes
        self.depth = 1
        self.nested = False
        self.images: list[dict[str, str | None]] = []
        self.captions: list[str] = []


class _FigureParser(HTMLParser):
    """Collect images and captions inside each light/dark wrapper.

    Any element whose class list names `light-content` or `dark-content` is a
    wrapper. Its end is found by counting that element's own tag, so a
    `section` wrapper with `div`s inside closes at its own `</section>`.
    Contents belong to the innermost open wrapper; a wrapper that opens while
    another is open is marked as nested.
    """

    def __init__(self) -> None:
        super().__init__()
        self.wrappers: list[_Wrapper] = []
        self._open: list[_Wrapper] = []
        self._caption: tuple[_Wrapper, list[str]] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for wrapper in self._open:
            if wrapper.tag == tag:
                wrapper.depth += 1
        classes = (dict(attrs).get("class") or "").split()
        themes = [theme for theme in THEMES if f"{theme}-content" in classes]
        if themes:
            wrapper = _Wrapper(tag, themes)
            wrapper.nested = bool(self._open)
            self.wrappers.append(wrapper)
            if tag not in _VOID_TAGS:
                self._open.append(wrapper)
        elif self._open and tag == "img":
            self._open[-1].images.append(dict(attrs))
        elif self._open and tag == "figcaption":
            self._caption = (self._open[-1], [])

    def handle_endtag(self, tag: str) -> None:
        if tag == "figcaption" and self._caption is not None:
            wrapper, parts = self._caption
            wrapper.captions.append(" ".join("".join(parts).split()))
            self._caption = None
        for wrapper in self._open:
            if wrapper.tag == tag:
                wrapper.depth -= 1
        self._open = [wrapper for wrapper in self._open if wrapper.depth > 0]

    def handle_data(self, data: str) -> None:
        if self._caption is not None:
            self._caption[1].append(data)


def figure_failures(output_dir: Path) -> list[str]:
    """Return one message per violated rule across the manifest and the site."""
    failures: list[str] = []
    for page, figures in EXPECTED_FIGURES.items():
        path = output_dir / page
        if not path.is_file():
            failures.append(f"{page}: page is missing from {output_dir}")
            continue
        failures.extend(_page_failures(page, path, figures, output_dir))
    failures.extend(_unlisted_page_failures(output_dir))
    return failures


def _unlisted_page_failures(output_dir: Path) -> list[str]:
    failures: list[str] = []
    for path in sorted(output_dir.rglob("*.html")):
        relative = path.relative_to(output_dir)
        if "site_libs" in relative.parts or relative.as_posix() in EXPECTED_FIGURES:
            continue
        parser = _FigureParser()
        parser.feed(path.read_text(encoding="utf-8"))
        if parser.wrappers:
            failures.append(
                f"{relative.as_posix()}: not in the figure manifest, "
                f"but has {len(parser.wrappers)} theme wrapper(s)"
            )
    return failures


def _page_failures(
    page: str, path: Path, figures: tuple[Fig, ...], output_dir: Path
) -> list[str]:
    parser = _FigureParser()
    parser.feed(path.read_text(encoding="utf-8"))
    ambiguous = [w for w in parser.wrappers if len(w.themes) > 1]
    if ambiguous:
        # Such an element counts for neither theme, so per-theme checks would
        # only repeat the same fault.
        return [f"{page}: <{w.tag}> wrapper has both theme classes" for w in ambiguous]
    # The outer wrapper is hidden in the other theme and takes the inner one
    # with it, so a nested wrapper never shows.
    failures = [
        f"{page}: <{w.tag}> {w.themes[0]}-content: theme wrapper nested inside another"
        for w in parser.wrappers
        if w.nested
    ]
    wrappers = parser.wrappers
    if len(wrappers) != 2 * len(figures):
        failures.append(
            f"{page}: expected {len(figures)} figure(s), "
            f"{2 * len(figures)} theme wrappers; found {len(wrappers)}"
        )
        return failures
    for index, figure in enumerate(figures, start=1):
        pair = wrappers[2 * index - 2 : 2 * index]
        if [w.themes[0] for w in pair] != list(THEMES):
            found = " then ".join(f"{w.themes[0]}-content" for w in pair)
            failures.append(
                f"{page}: figure {index}: expected light-content then "
                f"dark-content, found {found}"
            )
            continue
        if figure.kind != "plot":
            raise ValueError(
                f"{page}: figure {index}: no check for kind {figure.kind!r}"
            )
        failures.extend(
            _plot_failures(f"{page}: figure {index}", pair, figure, path, output_dir)
        )
    return failures


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
    if len(files) == 2:
        if files["light"] == files["dark"]:
            failures.append(f"{label}: light and dark use the same image file")
        elif files["light"].read_bytes() == files["dark"].read_bytes():
            failures.append(f"{label}: light and dark images have the same content")
    if len(captions) == 2 and captions["light"] != captions["dark"]:
        failures.append(
            f"{label}: captions differ: {captions['light']!r} vs {captions['dark']!r}"
        )
    return failures


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


def _resolve_source(
    src: str, page_path: Path, output_dir: Path
) -> tuple[Path | None, str]:
    """Return the file an image ``src`` names, or None and what is wrong.

    Only the URL's path names the file: a query or fragment is dropped and
    percent-escapes are decoded. The file must be referenced relative to the
    page and lie inside the site, so the deployed page can load it.
    """
    try:
        url = urlsplit(src)
    except ValueError:
        return None, f"image URL is malformed: {src!r}"
    path = unquote(url.path)
    if "\x00" in path:
        return None, f"image path contains a NUL: {src!r}"
    # The decoded path is checked too: %2F decodes to a leading slash.
    if url.scheme or url.netloc or path.startswith("/"):
        return None, f"image is outside the site: {src!r}"
    resolved = (page_path.parent / path).resolve()
    if not resolved.is_relative_to(output_dir.resolve()):
        return None, f"image is outside the site: {src!r}"
    if not path or not resolved.is_file():
        return None, f"image {src!r} does not exist"
    return resolved, ""


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
