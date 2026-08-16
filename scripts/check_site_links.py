"""Check the documentation link contract for the rendered site.

Verifies every internal link added by the Diataxis docs pass (spec:
docs/superpowers/specs/2026-08-10-diataxis-docs-pass-design.md).
For each contract row the SOURCE must contain a link resolving to the
target page (and fragment, if given), and the TARGET file must exist
(and contain the fragment id, if given). A missing source link is a
failure.

Rows are scoped: "content" rows only count links inside Quarto's
main#quarto-document-content element (the site-wide sidebar would
otherwise satisfy nearly every row trivially); "sidebar" rows only count
links inside nav#quarto-sidebar. README.md sources are parsed as
Markdown, fence-aware, with published-site URLs mapped onto _output
paths.

Run from the repo root, after rendering the site:

    cd site && uv run --frozen quarto render && cd ..
    uv run python scripts/check_site_links.py
"""

from __future__ import annotations

import re
import sys
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "site" / "_output"
README = ROOT / "README.md"
SITE_BASE = "https://jbwhit.github.io/clocks/"

# These strings describe interfaces or mathematical claims that no longer
# exist. Historical plans/specifications are intentionally records and are not
# part of this current-facing prose contract.
FORBIDDEN_CURRENT_CLAIMS: dict[str, str] = {
    "jit" + "ter_tau": "tempered SMC replaced annealed jitter",
    "jit" + "ter_std": "tempered SMC replaced annealed jitter",
    "jit" + "ter_mode": "tempered SMC replaced annealed jitter",
    "resampling with " + "jitter reduces": "resampling now has an MH correction",
    "perturbs the clones with " + "jitter": "use invariant MH rejuvenation",
    "jit" + "ter modes:": "use invariant MH rejuvenation",
    "annealed (the " + "default)": "adaptive tempered SMC is current",
    "annealed " + "jitter is now": "adaptive tempered SMC is current",
    "today's " + "jitter": "adaptive tempered SMC is current",
    "post-resampling " + "jitter, turning": "MH rejuvenation is implemented",
    "the demo here uses the annealed " + "default": "use current SMC controls",
    "covariance " + "jitter is the specialist tool": "use invariant MH moves",
    "at sampling and again after " + "every": (
        "MH proposals crossing order are rejected"
    ),
    "resample_" + "threshold": "ess_target controls adaptive tempering",
    "constraint_" + "fn": "the required prior density defines support",
    "support_" + "bounds": "the required prior density defines support",
    "mass_range shapes the initial " + "sample only": "mass range is prior support",
    "perfectly symmetric ring would leave mirror-" + "image": (
        "labeled channels generally distinguish reflected vectors"
    ),
    "deliberately deep in the relativistic " + "regime": "scenarios are weak-field",
    "_" + "cli": "demo entry points are packaged beneath clocks._demos",
}


def _current_prose_files(root: Path = ROOT) -> list[Path]:
    """Return current-facing source prose, excluding historical records."""
    files = [root / "README.md"]
    files.extend(
        path
        for path in (root / "docs").rglob("*.md")
        if not ({"plans", "superpowers"} & set(path.relative_to(root / "docs").parts))
    )
    files.extend((root / "site").rglob("*.qmd"))
    files.extend((root / "src" / "clocks").rglob("*.py"))
    files.extend((root / "scripts").rglob("*.py"))
    return sorted(path for path in files if path.is_file())


def _prose_claim_failures(root: Path = ROOT) -> list[str]:
    failures: list[str] = []
    for path in _current_prose_files(root):
        text = path.read_text(encoding="utf-8")
        for claim, correction in FORBIDDEN_CURRENT_CLAIMS.items():
            if claim.casefold() in text.casefold():
                relative = path.relative_to(root)
                failures.append(f"{relative}: {claim!r} ({correction})")
    return failures


# (scope, source, target, fragment) — source/target are html paths
# relative to site/_output; source may also be "README.md". scope is
# "content", "sidebar", or "readme:<heading>" (README rows are scoped to
# the ## section with that heading, so a pre-existing link elsewhere in
# the file cannot satisfy them). fragment=None means a page-only link:
# the source must contain a link to the page WITHOUT a fragment (exact
# match — a #fragment link does not satisfy a page-only row), and the
# target file must exist.
CONTRACT: list[tuple[str, str, str, str | None]] = [
    # Task 1 — glossary outbound links (one per destination page)
    (
        "content",
        "method/notation-and-glossary.html",
        "story/clocks-as-gravimeters.html",
        None,
    ),
    (
        "content",
        "method/notation-and-glossary.html",
        "story/one-clock-is-not-enough.html",
        None,
    ),
    (
        "content",
        "method/notation-and-glossary.html",
        "story/the-search-in-one-dimension.html",
        None,
    ),
    ("content", "method/notation-and-glossary.html", "story/into-the-plane.html", None),
    (
        "content",
        "method/notation-and-glossary.html",
        "story/two-hidden-masses.html",
        None,
    ),
    (
        "content",
        "method/notation-and-glossary.html",
        "story/how-many-masses.html",
        None,
    ),
    (
        "content",
        "method/notation-and-glossary.html",
        "story/beyond-point-masses.html",
        None,
    ),
    (
        "content",
        "method/notation-and-glossary.html",
        "story/gravitational-echolocation.html",
        None,
    ),
    (
        "content",
        "method/notation-and-glossary.html",
        "method/units-and-scales.html",
        None,
    ),
    (
        "content",
        "method/notation-and-glossary.html",
        "method/the-particle-filter.html",
        None,
    ),
    (
        "content",
        "method/notation-and-glossary.html",
        "method/the-particle-filter.html",
        "sec-resampling",
    ),
    (
        "content",
        "method/notation-and-glossary.html",
        "method/the-particle-filter.html",
        "sec-evidence",
    ),
    # Task 1 — sidebar entry
    ("sidebar", "index.html", "method/notation-and-glossary.html", None),
    # Task 2 — story/landing cross-links
    ("content", "index.html", "method/the-particle-filter.html", "sec-intro"),
    (
        "content",
        "index.html",
        "method/notation-and-glossary.html",
        "term-forward-model",
    ),
    (
        "content",
        "index.html",
        "method/notation-and-glossary.html",
        "term-inverse-problem",
    ),
    (
        "content",
        "story/clocks-as-gravimeters.html",
        "method/notation-and-glossary.html",
        "term-weak-field",
    ),
    (
        "content",
        "story/one-clock-is-not-enough.html",
        "method/notation-and-glossary.html",
        "term-sigma-obs",
    ),
    (
        "content",
        "story/into-the-plane.html",
        "method/notation-and-glossary.html",
        "term-prior",
    ),
    (
        "content",
        "story/into-the-plane.html",
        "method/notation-and-glossary.html",
        "term-index-conventions",
    ),
    (
        "content",
        "story/two-hidden-masses.html",
        "method/the-particle-filter.html",
        "sec-resampling",
    ),
    (
        "content",
        "story/two-hidden-masses.html",
        "method/notation-and-glossary.html",
        "term-posterior",
    ),
    (
        "content",
        "story/beyond-point-masses.html",
        "method/notation-and-glossary.html",
        "term-sigma-obs",
    ),
    (
        "content",
        "story/beyond-point-masses.html",
        "method/the-particle-filter.html",
        "sec-resampling",
    ),
    (
        "content",
        "story/gravitational-echolocation.html",
        "method/the-particle-filter.html",
        "sec-evidence",
    ),
    # Task 3 — architecture page sidebar entry
    ("sidebar", "index.html", "reproduce/architecture.html", None),
    # Task 4 — README → site links (section-scoped, so the pre-existing
    # site-root link at the top of the README cannot satisfy the
    # demo-catalog row) and the getting-started repoint
    ("readme:Use as a library", "README.md", "reproduce/getting-started.html", None),
    ("readme:Use as a library", "README.md", "method/the-particle-filter.html", None),
    ("readme:Run the demos", "README.md", "index.html", None),
    ("readme:Project structure", "README.md", "reproduce/architecture.html", None),
    (
        "content",
        "reproduce/getting-started.html",
        "method/the-particle-filter.html",
        None,
    ),
    # Final-review fix: prior link on how-many-masses
    (
        "content",
        "story/how-many-masses.html",
        "method/notation-and-glossary.html",
        "term-prior",
    ),
]

_VOID = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "source",
    "track",
    "wbr",
}


class _PageIndex(HTMLParser):
    """Collect element ids plus hrefs, scoped to content vs sidebar."""

    def __init__(self) -> None:
        super().__init__()
        self.content_hrefs: set[str] = set()
        self.sidebar_hrefs: set[str] = set()
        self.ids: set[str] = set()
        self._stack: list[str] = []
        self._content_depth: int | None = None
        self._sidebar_depth: int | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        el_id = attr_map.get("id")
        if el_id:
            self.ids.add(el_id)
        if tag not in _VOID:
            self._stack.append(tag)
            if el_id == "quarto-document-content" and self._content_depth is None:
                self._content_depth = len(self._stack)
            if el_id == "quarto-sidebar" and self._sidebar_depth is None:
                self._sidebar_depth = len(self._stack)
        href = attr_map.get("href")
        if href:
            depth = len(self._stack)
            if self._content_depth is not None and depth >= self._content_depth:
                self.content_hrefs.add(href)
            if self._sidebar_depth is not None and depth >= self._sidebar_depth:
                self.sidebar_hrefs.add(href)

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID:
            return
        while self._stack:
            top = self._stack.pop()
            if (
                self._content_depth is not None
                and len(self._stack) < self._content_depth
            ):
                self._content_depth = None
            if (
                self._sidebar_depth is not None
                and len(self._stack) < self._sidebar_depth
            ):
                self._sidebar_depth = None
            if top == tag:
                break


def _map_site_url(url: str) -> tuple[str, str | None] | None:
    """Map a published-site URL to an _output-relative path + fragment."""
    if not url.startswith(SITE_BASE):
        return None
    parsed = urlparse(url)
    rel = parsed.path[len(urlparse(SITE_BASE).path) :]
    if not rel or rel.endswith("/"):
        rel += "index.html"
    return rel, (parsed.fragment or None)


def _resolve_href(source: str, href: str) -> tuple[str, str | None] | None:
    """Resolve one href from `source` to an _output-relative target."""
    parsed = urlparse(href)
    if parsed.scheme in ("http", "https"):
        return _map_site_url(href)
    if parsed.scheme or not parsed.path:
        return None  # mailto:, same-page #fragment, etc.
    parts: list[str] = []
    for part in (PurePosixPath(source).parent / parsed.path).parts:
        if part == "..":
            if parts:
                parts.pop()
        elif part != ".":
            parts.append(part)
    return "/".join(parts), (parsed.fragment or None)


_FENCE_RE = re.compile(r"^(`{3,}|~{3,})")
# A run of backticks not adjacent to further backticks, its span, and the
# matching closing run of the same length — a CommonMark-ish code span.
_CODE_SPAN_RE = re.compile(r"(?<!`)(`+)(?!`).*?(?<!`)\1(?!`)")


def _readme_links() -> set[tuple[str, str, str | None]]:
    """(section, target, fragment) links in README.md.

    Fence-aware: skips ``` and ~~~ fenced blocks (closers must match the
    opener's character and be at least as long, so an outer ````-fence
    swallows inner ``` lines) and strips inline code spans of any
    delimiter length. Links are tagged with the ## section heading they
    appear under, so contract rows can require a link in a specific
    section.
    """
    links: set[tuple[str, str, str | None]] = set()
    fence: str | None = None
    section = ""
    for line in README.read_text(encoding="utf-8").splitlines():
        match = _FENCE_RE.match(line.lstrip())
        if fence is None:
            if match:
                fence = match.group(1)
                continue
        else:
            if (
                match
                and match.group(1)[0] == fence[0]
                and len(match.group(1)) >= len(fence)
            ):
                fence = None
            continue
        if line.startswith("## "):
            section = line[3:].strip()
            continue
        clean = _CODE_SPAN_RE.sub("", line)
        for url in re.findall(r"\]\(([^)\s]+)\)", clean):
            mapped = _map_site_url(url)
            if mapped is not None:
                links.add((section, *mapped))
    return links


_cache: dict[str, _PageIndex | None] = {}


def _load(rel: str) -> _PageIndex | None:
    if rel not in _cache:
        path = OUTPUT / rel
        if not path.exists():
            _cache[rel] = None
        else:
            parser = _PageIndex()
            parser.feed(path.read_text(encoding="utf-8"))
            _cache[rel] = parser
    return _cache[rel]


def _source_links(scope: str, source: str) -> set[tuple[str, str | None]] | None:
    if source == "README.md":
        section = scope.removeprefix("readme:")
        return {(t, f) for s, t, f in _readme_links() if s == section}
    page = _load(source)
    if page is None:
        return None
    hrefs = page.sidebar_hrefs if scope == "sidebar" else page.content_hrefs
    return {m for h in hrefs if (m := _resolve_href(source, h)) is not None}


def main() -> int:
    prose_failures = _prose_claim_failures()
    failures: list[str] = []
    for scope, source, target, fragment in CONTRACT:
        label = f"[{scope}] {source} -> {target}" + (f"#{fragment}" if fragment else "")
        target_page = _load(target)
        if target_page is None:
            failures.append(f"{label}: target file missing")
            continue
        if fragment is not None and fragment not in target_page.ids:
            failures.append(f"{label}: id '{fragment}' missing in target")
        links = _source_links(scope, source)
        if links is None:
            failures.append(f"{label}: source file missing")
            continue
        # Exact fragment match both ways: a page-only row (fragment=None)
        # is NOT satisfied by a #fragment link to the same page.
        if (target, fragment) not in links:
            failures.append(f"{label}: no source link found")
    if prose_failures:
        print("Current-prose contract failures:", file=sys.stderr)
        for failure in prose_failures:
            print(f"  {failure}", file=sys.stderr)
    if failures:
        print("Link contract failures:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
    if prose_failures or failures:
        return 1
    print(
        "Current-prose contract OK: "
        f"{len(_current_prose_files())} files, "
        f"{len(FORBIDDEN_CURRENT_CLAIMS)} stale claims checked."
    )
    print(f"Link contract OK: {len(CONTRACT)} rows checked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
