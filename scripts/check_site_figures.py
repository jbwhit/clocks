"""Check the figure contract for the rendered site.

Each of the nine plotting pages draws its plot twice, once per colour theme.
Quarto wraps the two outputs in elements with the classes `light-content` and
`dark-content`, and the site shows one at a time. This check parses the
rendered HTML and requires, for every page:

- exactly one wrapper per theme, and no element carrying both theme classes;
- exactly one image in each wrapper, declared 750 pixels wide and the page's
  expected height, with the two referenced files existing inside the site and
  different;
- exactly one `figcaption` in each wrapper, with the same text in both.

Images outside the two wrappers (committed GIFs and PNGs) are ignored.

Run from the repo root, after rendering the site:

    cd site && uv run --frozen quarto render && cd ..
    uv run python scripts/check_site_figures.py
"""

from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "site" / "_output"

THEMES = ("light", "dark")
EXPECTED_WIDTH = 750

# Elements with no end tag; one of these carrying a theme class wraps nothing.
_VOID_TAGS = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta"}
    | {"source", "track", "wbr"}
)

# Page (relative to the output directory) -> declared image height in pixels.
EXPECTED_PAGES: dict[str, int] = {
    "story/clocks-as-gravimeters.html": 420,
    "story/one-clock-is-not-enough.html": 420,
    "story/the-search-in-one-dimension.html": 540,
    "story/into-the-plane.html": 540,
    "story/two-hidden-masses.html": 420,
    "story/how-many-masses.html": 420,
    "story/beyond-point-masses.html": 420,
    "story/gravitational-echolocation.html": 420,
    "method/the-particle-filter.html": 540,
}


class _Wrapper:
    """One element carrying a theme class, and what it contains."""

    def __init__(self, tag: str, themes: list[str]) -> None:
        self.tag = tag
        self.themes = themes
        self.depth = 1
        self.images: list[dict[str, str | None]] = []
        self.captions: list[str] = []


class _FigureParser(HTMLParser):
    """Collect images and captions inside each light/dark wrapper.

    Any element whose class list names `light-content` or `dark-content` is a
    wrapper. Its end is found by counting that element's own tag, so a
    `section` wrapper with `div`s inside closes at its own `</section>`.
    Contents belong to the innermost open wrapper.
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
    """Return one message per violated rule across the expected pages."""
    failures: list[str] = []
    for page, height in EXPECTED_PAGES.items():
        path = output_dir / page
        if not path.is_file():
            failures.append(f"{page}: page is missing from {output_dir}")
            continue
        failures.extend(_page_failures(page, path, height, output_dir))
    return failures


def _page_failures(page: str, path: Path, height: int, output_dir: Path) -> list[str]:
    parser = _FigureParser()
    parser.feed(path.read_text(encoding="utf-8"))
    ambiguous = [w for w in parser.wrappers if len(w.themes) > 1]
    if ambiguous:
        # Such an element counts for neither theme, so per-theme checks would
        # only repeat the same fault.
        return [f"{page}: <{w.tag}> wrapper has both theme classes" for w in ambiguous]
    failures: list[str] = []
    files: dict[str, Path] = {}
    captions: dict[str, str] = {}
    for theme in THEMES:
        label = f"{page}: {theme}-content"
        wrappers = [w for w in parser.wrappers if w.themes == [theme]]
        if len(wrappers) != 1:
            failures.append(f"{label}: expected 1 wrapper, found {len(wrappers)}")
            continue
        wrapper = wrappers[0]
        failures.extend(_image_failures(label, wrapper.images, height))
        if len(wrapper.images) == 1:
            src = wrapper.images[0].get("src") or ""
            problem = _source_problem(src, path, output_dir)
            if problem:
                failures.append(f"{label}: {problem}")
            else:
                files[theme] = (path.parent / src).resolve()
        texts = wrapper.captions
        if len(texts) != 1:
            failures.append(f"{label}: expected 1 figcaption, found {len(texts)}")
        elif not texts[0]:
            failures.append(f"{label}: empty caption")
        else:
            captions[theme] = texts[0]
    if len(files) == 2 and files["light"] == files["dark"]:
        failures.append(f"{page}: light and dark use the same image file")
    if len(captions) == 2 and captions["light"] != captions["dark"]:
        failures.append(
            f"{page}: captions differ: {captions['light']!r} vs {captions['dark']!r}"
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


def _source_problem(src: str, page_path: Path, output_dir: Path) -> str | None:
    """Say what is wrong with an image ``src``, or return None if it is fine.

    The file must be referenced relative to the page and lie inside the site,
    so the deployed page can load it.
    """
    if src.startswith("/") or urlsplit(src).scheme:
        return f"image is outside the site: {src!r}"
    resolved = (page_path.parent / src).resolve()
    if not resolved.is_relative_to(output_dir.resolve()):
        return f"image is outside the site: {src!r}"
    if not src or not resolved.is_file():
        return f"image {src!r} does not exist"
    return None


def main() -> int:
    failures = figure_failures(OUTPUT)
    if failures:
        print("Figure contract failures:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print(f"Figure contract OK: {len(EXPECTED_PAGES)} pages checked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
