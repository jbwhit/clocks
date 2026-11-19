"""Check the figure contract for the rendered site.

Each of the nine plotting pages draws its plot twice, once per colour theme.
Quarto wraps the two outputs in elements with the classes `light-content` and
`dark-content`, and the site shows one at a time. This check parses the
rendered HTML and requires, for every page:

- exactly one image in each wrapper, declared 750 pixels wide and the page's
  expected height, with the two referenced files existing and different;
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

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "site" / "_output"

THEMES = ("light", "dark")
EXPECTED_WIDTH = 750

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
    """One light-content or dark-content element and what it contains."""

    def __init__(self, theme: str) -> None:
        self.theme = theme
        self.depth = 1
        self.images: list[dict[str, str | None]] = []
        self.captions: list[str] = []


class _FigureParser(HTMLParser):
    """Collect images and captions inside each light/dark wrapper."""

    def __init__(self) -> None:
        super().__init__()
        self.wrappers: list[_Wrapper] = []
        self._open: _Wrapper | None = None
        self._caption_parts: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._open is not None and tag == "div":
            self._open.depth += 1
        elif self._open is None and tag == "div":
            classes = (dict(attrs).get("class") or "").split()
            for theme in THEMES:
                if f"{theme}-content" in classes:
                    self._open = _Wrapper(theme)
                    self.wrappers.append(self._open)
        elif self._open is not None and tag == "img":
            self._open.images.append(dict(attrs))
        elif self._open is not None and tag == "figcaption":
            self._caption_parts = []

    def handle_endtag(self, tag: str) -> None:
        if self._open is None:
            return
        if tag == "figcaption" and self._caption_parts is not None:
            self._open.captions.append(" ".join("".join(self._caption_parts).split()))
            self._caption_parts = None
        elif tag == "div":
            self._open.depth -= 1
            if self._open.depth == 0:
                self._open = None

    def handle_data(self, data: str) -> None:
        if self._caption_parts is not None:
            self._caption_parts.append(data)


def figure_failures(output_dir: Path) -> list[str]:
    """Return one message per violated rule across the expected pages."""
    failures: list[str] = []
    for page, height in EXPECTED_PAGES.items():
        path = output_dir / page
        if not path.is_file():
            failures.append(f"{page}: page is missing from {output_dir}")
            continue
        failures.extend(_page_failures(page, path, height))
    return failures


def _page_failures(page: str, path: Path, height: int) -> list[str]:
    parser = _FigureParser()
    parser.feed(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    files: dict[str, Path] = {}
    captions: dict[str, str] = {}
    for theme in THEMES:
        wrappers = [w for w in parser.wrappers if w.theme == theme]
        images = [img for w in wrappers for img in w.images]
        texts = [c for w in wrappers for c in w.captions]
        label = f"{page}: {theme}-content"
        if len(images) != 1:
            failures.append(f"{label}: expected 1 image, found {len(images)}")
        else:
            failures.extend(_image_failures(label, path, images[0], height, files))
        if len(texts) != 1:
            failures.append(f"{label}: expected 1 figcaption, found {len(texts)}")
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
    label: str,
    page_path: Path,
    img: dict[str, str | None],
    height: int,
    files: dict[str, Path],
) -> list[str]:
    failures: list[str] = []
    theme = label.rsplit(": ", 1)[1].removesuffix("-content")
    for name, expected in (("width", EXPECTED_WIDTH), ("height", height)):
        if img.get(name) != str(expected):
            failures.append(f"{label}: {name} {img.get(name)}, expected {expected}")
    src = img.get("src") or ""
    resolved = (page_path.parent / src).resolve()
    if not src or not resolved.is_file():
        failures.append(f"{label}: image {src!r} does not exist")
    else:
        files[theme] = resolved
    return failures


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
