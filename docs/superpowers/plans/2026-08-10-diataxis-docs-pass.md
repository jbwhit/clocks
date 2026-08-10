# Diataxis Documentation Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a notation/glossary page, story→method cross-links, an architecture page, and a slimmed README to the "GPS in Reverse" Quarto site, per the approved spec `docs/superpowers/specs/2026-08-10-diataxis-docs-pass-design.md`.

**Architecture:** Four content changes to `site/` and `README.md`, landed in dependency order so every commit renders with no dangling link targets. A stdlib-only checker script (`scripts/check_site_links.py`) holds the link contract as data and verifies every added link bidirectionally against the rendered `_output/`; its contract grows with each task. The checker is location-aware: content links are only counted inside Quarto's `main#quarto-document-content`, sidebar rows only inside `nav#quarto-sidebar` — otherwise the site-wide sidebar would satisfy nearly every row trivially.

**Tech Stack:** Quarto (`.qmd`, Mermaid diagrams), Python 3.12+ via uv (checker script, stdlib only).

**Review trail:** Codex xhigh plan round 1 (2026-08-10) — NEEDS REVISION; all seven findings accepted and fixed in this revision (location-aware checker; fence-aware README parsing; contract completeness incl. a `#sec-intro` anchor so the landing-page particle-filter link carries a fragment per spec; notation inventory step and missing entries — evidence row, R, σ_y, particle-filter term; two link-matrix rows dropped as in-page definitions; six factual wording fixes; self-contained commands with branch creation, commit trailers, and a concrete Task 5).

## Global Constraints

- Branch: `claude-diataxis-docs-pass` (created from up-to-date `main` in Task 1 Step 0).
- Every commit message ends with a `Co-Authored-By:` trailer naming the model actually authoring it (the harness tells you which you are — never hardcode), plus the `Claude-Session:` trailer if the harness provides one. The commit commands below show a `<model trailer>` placeholder — substitute your actual identity.
- **Per-commit gate (all three, in order, before every commit):**
  1. Render: `cd site && uv run --frozen quarto render` — must complete with no errors.
  2. Link contract: `uv run python scripts/check_site_links.py` (from repo root) — must exit 0 (from Task 1 onward).
  3. Repo gate: `uv run ruff format --check .` AND `uv run ruff check .` AND `uv run pytest` (190 passing, 2 deselected as of plan date) — all green.
- After writing or editing `scripts/check_site_links.py`, run `uv run ruff format scripts/check_site_links.py` before the gate — long CONTRACT rows exceed the 88-char limit as typed.
- No changes to `src/clocks/` or `tests/` (spec: out of scope), **except** the two heading anchors and one intro span added to `site/method/the-particle-filter.qmd` (a site file, not library code). `scripts/check_site_links.py` is the only Python file added.
- Prose voice on story pages must not change: cross-links attach to existing wording; never add or reword sentences to create a linking opportunity. The only text edits outside link-wrapping are those written out verbatim in Tasks 1, 3, and 4.
- Published-site base URL for README links: `https://jbwhit.github.io/clocks/`.
- The rendered site output dir is `site/_output/` (gitignored — never commit it).
- The shell's cwd can reset between calls — begin every git step with an explicit `cd` to the repo root.

---

### Task 1: Glossary page, method-page anchors, and link checker

**Files:**
- Create: `site/method/notation-and-glossary.qmd`
- Create: `scripts/check_site_links.py`
- Modify: `site/method/the-particle-filter.qmd` (two heading anchors + one intro anchor)
- Modify: `site/_quarto.yml` (one sidebar entry)

**Interfaces:**
- Produces: glossary anchors `#term-weak-field`, `#term-time-dilation`, `#term-chronometric-leveling`, `#term-forward-model`, `#term-inverse-problem`, `#term-degeneracy`, `#term-particle-filter`, `#term-prior`, `#term-likelihood`, `#term-posterior`, `#term-evidence`, `#term-resampling`, `#term-jitter`, `#term-ess`, `#term-model-comparison`, `#term-label-switching`, `#term-sigma-obs`, `#term-index-conventions` (Task 2 links to a subset — names must match exactly).
- Produces: method anchors `#sec-intro`, `#sec-resampling`, `#sec-evidence` on `the-particle-filter.qmd`.
- Produces: `scripts/check_site_links.py` with a module-level `CONTRACT` list that Tasks 2–4 append rows to. Row shape: `(scope, source, target, fragment)` where scope is `"content"` or `"sidebar"`.

- [ ] **Step 0: Create the branch**

```bash
cd /Users/jonathan/projects/clocks
git checkout main && git pull --no-rebase --ff-only
git checkout -b claude-diataxis-docs-pass
```

- [ ] **Step 1: Verify the notation inventory**

The glossary tables below were drafted from a full read of every site
page (spec "implementation step 0"). Re-verify before writing the page:

```bash
grep -ohE '\$[^$]+\$' site/index.qmd site/story/*.qmd site/method/*.qmd site/reproduce/*.qmd | sort -u
```

Every symbol in the output must appear in the glossary tables below or be
a one-off expression built from symbols that do (e.g. $2\Phi/c^2$,
$10^{-16}/\sqrt{\tau}$, $\sqrt{1+2\Phi}$). If you find a standalone
symbol the tables miss, add a row for it following the same format — the
inventory is authoritative, per the spec.

- [ ] **Step 2: Add explicit anchors to the particle-filter page**

In `site/method/the-particle-filter.qmd`, make three edits:

Wrap the intro paragraph (starting "This page is for the reader…") in an
anchored div:

```markdown
::: {#sec-intro}
This page is for the reader who wants the machinery. The implementation is
`clocks.inference.ParticleFilter` — a few hundred lines, and everything below
maps onto it directly.
:::
```

Change `## Resampling` to:

```markdown
## Resampling {#sec-resampling}
```

Change `## Evidence` to:

```markdown
## Evidence {#sec-evidence}
```

- [ ] **Step 3: Create the glossary page**

Create `site/method/notation-and-glossary.qmd` with the content below
(drafted from the Step 1 inventory; if the Step 7 visual check finds an
entry exceeding two rendered lines, trim that entry's wording — never
drop entries):

````markdown
---
title: "Notation & Glossary"
---

A consolidated lookup for every symbol and term used across the site.
Symbols defined on their source pages are still listed here — this page
exists so you never have to hunt for where a definition first appeared.

## Physics notation {#sec-physics-notation}

| Symbol | Meaning | Where introduced |
|---|---|---|
| $\Phi$ | Newtonian gravitational potential (negative near a mass); $-M/r$ for a point mass | [Clocks as Gravimeters](../story/clocks-as-gravimeters.qmd) |
| $G$, $c$ | Newton's constant and the speed of light; the simulation sets $G = c = 1$ | [Units and Scales](units-and-scales.qmd) |
| $\tau$, $t$ | Proper time vs coordinate time; a clock's tick rate is $d\tau/dt$ | [Clocks as Gravimeters](../story/clocks-as-gravimeters.qmd) |
| $r$ | Bare $r$ is always a *distance* on this site | [Clocks as Gravimeters](../story/clocks-as-gravimeters.qmd) |
| $r_c$ | Tick rate of clock $c$ in the filter equations — same letter as distance, distinguished by the clock index | [The Particle Filter](the-particle-filter.qmd) |
| $r_{ij}$ | Distance from clock $i$ to mass $j$ | [Into the Plane](../story/into-the-plane.qmd) |
| $r_s$ | Schwarzschild radius, $r_s = 2M$ in simulation units | [Units and Scales](units-and-scales.qmd) |
| $R$ | Range from the clock lattice to an exterior mass; the differential signal falls as $1/R^2$, the curvature term as $1/R^3$ | [Gravitational Echolocation](../story/gravitational-echolocation.qmd) |
| $x$, $y$ | Mass position coordinates (subscripted $x_1, x_2$ with several masses) | [The Search in One Dimension](../story/the-search-in-one-dimension.qmd) |
| $M$, $M_j$ | Mass (of mass $j$) | [Clocks as Gravimeters](../story/clocks-as-gravimeters.qmd) |
| $\mu$, $\sigma_{\text{density}}$, $A$ | Center, width, and peak amplitude of a continuous Gaussian mass profile | [Beyond Point Masses](../story/beyond-point-masses.qmd) |

[**Index conventions**]{#term-index-conventions} — context-dependent: $i$
indexes clocks/evaluation points in the potential sum ($r_{ij}$) but
*particles* in the filter equations ($w_i$, $\theta_i$); $j$ indexes
masses; $c$ indexes clocks ($r_c$).

## Inference notation {#sec-inference-notation}

| Symbol | Meaning | Where introduced |
|---|---|---|
| $\sigma_{\text{obs}}$ | Observation noise standard deviation (0.005 in the demos); written plain $\sigma$ on the inference pages | [One Clock Is Not Enough](../story/one-clock-is-not-enough.qmd) |
| $N$ | Number of particles in the filter's cloud | [The Particle Filter](the-particle-filter.qmd) |
| $w_i$ | Weight of particle $i$ (weights sum to one) | [The Particle Filter](the-particle-filter.qmd) |
| $\theta_i$ | Particle $i$'s full parameter hypothesis, e.g. $(x, M)$ | [The Particle Filter](the-particle-filter.qmd) |
| $K$ | Number of masses a model assumes | [How Many Masses?](../story/how-many-masses.qmd) |
| ESS | Effective sample size, $1/\sum_i w_i^2$ | [The Particle Filter](the-particle-filter.qmd#sec-resampling) |
| evidence, log-evidence | Marginal likelihood of the observed data under a model, accumulated observation by observation | [How Many Masses?](../story/how-many-masses.qmd) |

::: {.callout-note title="Two overloaded symbols"}
**$\sigma$** means observation noise on the inference pages but Gaussian
profile *width* in [Beyond Point Masses](../story/beyond-point-masses.qmd)
(which flags the reuse itself); this page writes $\sigma_{\text{obs}}$ vs
$\sigma_{\text{density}}$ to keep them apart. **$\tau$** means proper time
in $d\tau/dt$, but in the
[echolocation coda](../story/gravitational-echolocation.qmd) it is a clock's
*averaging time*: there $\sigma_y(\tau) = 10^{-16}/\sqrt{\tau}$ is the
clock's fractional-frequency instability after averaging for time $\tau$.
:::

## Terms {#sec-terms}

- [**Weak field**]{#term-weak-field} — the regime where gravity is a small
  correction ($|2\Phi/c^2| \ll 1$) and the tick rate
  $\sqrt{1 + 2\Phi/c^2}$ applies; see
  [Clocks as Gravimeters](../story/clocks-as-gravimeters.qmd).
- [**Time dilation**]{#term-time-dilation} — clocks deeper in a
  gravitational well tick slower relative to distant ones; the measurable
  effect this whole site is built on
  ([Clocks as Gravimeters](../story/clocks-as-gravimeters.qmd)).
- [**Chronometric leveling**]{#term-chronometric-leveling} — surveying
  height differences by comparing clock rates, as in the Tokyo Skytree
  experiment ([Clocks as Gravimeters](../story/clocks-as-gravimeters.qmd)).
- [**Forward model**]{#term-forward-model} — the physics direction:
  from a mass configuration to the clock rates it produces
  ([Clocks as Gravimeters](../story/clocks-as-gravimeters.qmd)).
- [**Inverse problem**]{#term-inverse-problem} — the detection direction:
  from noisy clock rates back to the masses that caused them
  ([One Clock Is Not Enough](../story/one-clock-is-not-enough.qmd)).
- [**Mass–distance degeneracy**]{#term-degeneracy} — a small mass nearby
  and a heavy mass far away can produce identical readings; reduced or
  broken by sufficiently informative array geometry
  ([One Clock Is Not Enough](../story/one-clock-is-not-enough.qmd)).
- [**Particle filter**]{#term-particle-filter} — the site's inference
  engine: a cloud of weighted hypotheses, reweighted by each observation
  ([The Particle Filter](the-particle-filter.qmd)).
- [**Prior**]{#term-prior} — what the filter assumes before any data:
  uniform ranges for positions and masses
  ([The Particle Filter](the-particle-filter.qmd)).
- [**Likelihood**]{#term-likelihood} — how well a hypothesis predicts an
  observed set of clock rates; Gaussian in the observation noise
  ([The Particle Filter](the-particle-filter.qmd)).
- [**Posterior**]{#term-posterior} — the belief after data. The particle
  cloud approximates it; the cloud's spread is the filter's *claimed*
  uncertainty ([The Search in One Dimension](../story/the-search-in-one-dimension.qmd)).
- [**Evidence**]{#term-evidence} — the marginal likelihood a model
  assigned to the data it actually saw; what
  [model comparison](../story/how-many-masses.qmd) ranks
  ([The Particle Filter](the-particle-filter.qmd#sec-evidence)).
- [**Resampling**]{#term-resampling} — redrawing the cloud in proportion
  to weight when the ESS collapses; systematic, stratified, or residual
  ([The Particle Filter](the-particle-filter.qmd#sec-resampling)).
- [**Jitter**]{#term-jitter} — small perturbation of resampled clones so
  they don't sit on top of each other; fixed, covariance, or annealed (the
  default) ([The Particle Filter](the-particle-filter.qmd#sec-resampling)).
- [**Effective sample size**]{#term-ess} — $1/\sum_i w_i^2$; how many
  particles are meaningfully alive
  ([The Particle Filter](the-particle-filter.qmd#sec-resampling)).
- [**Model comparison**]{#term-model-comparison} — running one filter per
  candidate mass count $K$ and comparing evidence
  ([How Many Masses?](../story/how-many-masses.qmd)).
- [**Label switching**]{#term-label-switching} — with several masses the
  posterior is symmetric under relabeling; broken by sorting, at a known
  cost ([Two Hidden Masses](../story/two-hidden-masses.qmd)).
- [**Observation noise**]{#term-sigma-obs} — the Gaussian noise
  $\sigma_{\text{obs}}$ added to every clock reading; the floor that
  makes many observations necessary
  ([One Clock Is Not Enough](../story/one-clock-is-not-enough.qmd)).
````

- [ ] **Step 4: Add the sidebar entry**

In `site/_quarto.yml`, under `section: "Part 2 — Under the Hood"`, after
the Units and Scales entry, add:

```yaml
          - text: "Notation & Glossary"
            href: method/notation-and-glossary.qmd
```

- [ ] **Step 5: Write the link checker script**

Create `scripts/check_site_links.py` with exactly this content, then run
`uv run ruff format scripts/check_site_links.py`:

```python
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

# (scope, source, target, fragment) — source/target are html paths
# relative to site/_output; source may also be "README.md". scope is
# "content" or "sidebar". fragment=None means a page-only link: the
# source link and the target file must exist, but no id is required.
CONTRACT: list[tuple[str, str, str, str | None]] = [
    # Task 1 — glossary outbound links (one per destination page)
    ("content", "method/notation-and-glossary.html", "story/clocks-as-gravimeters.html", None),
    ("content", "method/notation-and-glossary.html", "story/one-clock-is-not-enough.html", None),
    ("content", "method/notation-and-glossary.html", "story/the-search-in-one-dimension.html", None),
    ("content", "method/notation-and-glossary.html", "story/into-the-plane.html", None),
    ("content", "method/notation-and-glossary.html", "story/two-hidden-masses.html", None),
    ("content", "method/notation-and-glossary.html", "story/how-many-masses.html", None),
    ("content", "method/notation-and-glossary.html", "story/beyond-point-masses.html", None),
    ("content", "method/notation-and-glossary.html", "story/gravitational-echolocation.html", None),
    ("content", "method/notation-and-glossary.html", "method/units-and-scales.html", None),
    ("content", "method/notation-and-glossary.html", "method/the-particle-filter.html", None),
    ("content", "method/notation-and-glossary.html", "method/the-particle-filter.html", "sec-resampling"),
    ("content", "method/notation-and-glossary.html", "method/the-particle-filter.html", "sec-evidence"),
    # Task 1 — sidebar entry
    ("sidebar", "index.html", "method/notation-and-glossary.html", None),
]

_VOID = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "source", "track", "wbr",
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

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
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
            if self._content_depth is not None and len(self._stack) < self._content_depth:
                self._content_depth = None
            if self._sidebar_depth is not None and len(self._stack) < self._sidebar_depth:
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


def _readme_links() -> set[tuple[str, str | None]]:
    """Site-internal links in README.md, skipping fenced/inline code."""
    links: set[tuple[str, str | None]] = set()
    in_fence = False
    for line in README.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        clean = re.sub(r"`[^`]*`", "", line)
        for url in re.findall(r"\]\(([^)\s]+)\)", clean):
            mapped = _map_site_url(url)
            if mapped is not None:
                links.add(mapped)
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
        return _readme_links()
    page = _load(source)
    if page is None:
        return None
    hrefs = page.sidebar_hrefs if scope == "sidebar" else page.content_hrefs
    return {m for h in hrefs if (m := _resolve_href(source, h)) is not None}


def main() -> int:
    failures: list[str] = []
    for scope, source, target, fragment in CONTRACT:
        label = f"[{scope}] {source} -> {target}" + (
            f"#{fragment}" if fragment else ""
        )
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
        found = any(
            t == target and (fragment is None or f == fragment)
            for t, f in links
        )
        if not found:
            failures.append(f"{label}: no source link found")
    if failures:
        print("Link contract failures:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print(f"Link contract OK: {len(CONTRACT)} rows checked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Render and run the checker — expect it to pass**

```bash
cd /Users/jonathan/projects/clocks/site && uv run --frozen quarto render
cd /Users/jonathan/projects/clocks && uv run python scripts/check_site_links.py
```
Expected: render completes; checker prints `Link contract OK: 13 rows checked.`
If any row fails, fix the glossary/anchors (not the checker) unless the
checker itself has a bug. Note the glossary's page-only particle-filter
row is satisfied by its $N$ / $w_i$ / prior / likelihood entry links.

- [ ] **Step 7: Visual check**

Open `site/_output/method/notation-and-glossary.html` in a browser: check
light and dark themes and a narrow (~400px) window. Every entry ≤ 2
rendered lines at desktop width (trim wording if not — never drop
entries); tables scroll or wrap at mobile width without horizontal page
scroll.

- [ ] **Step 8: Repo gate, commit, push**

```bash
cd /Users/jonathan/projects/clocks
uv run ruff format --check . && uv run ruff check . && uv run pytest
git add site/method/notation-and-glossary.qmd site/method/the-particle-filter.qmd site/_quarto.yml scripts/check_site_links.py
git commit -m "$(cat <<'EOF'
Add notation & glossary page with stable anchors and link checker

<model trailer>
EOF
)"
git push -u origin claude-diataxis-docs-pass
```

---

### Task 2: Story and landing-page cross-links

**Files:**
- Modify: `site/index.qmd`
- Modify: `site/story/clocks-as-gravimeters.qmd`
- Modify: `site/story/one-clock-is-not-enough.qmd`
- Modify: `site/story/into-the-plane.qmd`
- Modify: `site/story/two-hidden-masses.qmd`
- Modify: `site/story/beyond-point-masses.qmd`
- Modify: `site/story/gravitational-echolocation.qmd`
- Modify: `scripts/check_site_links.py` (append CONTRACT rows)

**Interfaces:**
- Consumes: glossary `#term-*` anchors and method `#sec-intro`, `#sec-resampling`, `#sec-evidence` anchors from Task 1 (exact names listed there).

Each edit wraps existing words in a link — no wording changes. The link
matrix (also the PR-description artifact), 12 rows:

| Source page | Linked text (existing words) | Target |
|---|---|---|
| index.qmd | "particle filter" (in "a particle filter for the inverse problem") | `method/the-particle-filter.qmd#sec-intro` |
| index.qmd | "forward model" (in "a forward model from general relativity") | `method/notation-and-glossary.qmd#term-forward-model` |
| index.qmd | "inverse problem" (same sentence) | `method/notation-and-glossary.qmd#term-inverse-problem` |
| clocks-as-gravimeters.qmd | "weak-field result" (first sentence) | `../method/notation-and-glossary.qmd#term-weak-field` |
| one-clock-is-not-enough.qmd | "observation noise $\sigma$" (noise-floor callout) | `../method/notation-and-glossary.qmd#term-sigma-obs` |
| into-the-plane.qmd | "prior" (in "only the prior gains a column") | `../method/notation-and-glossary.qmd#term-prior` |
| into-the-plane.qmd | "$-M_j/r_{ij}$" (in "The potential sums $-M_j/r_{ij}$") | `../method/notation-and-glossary.qmd#term-index-conventions` |
| two-hidden-masses.qmd | "resampling jitter" (in "after every resampling jitter") | `../method/the-particle-filter.qmd#sec-resampling` |
| two-hidden-masses.qmd | "posterior" (in "the true posterior is perfectly bimodal") | `../method/notation-and-glossary.qmd#term-posterior` |
| beyond-point-masses.qmd | "observation noise" (in "not the observation noise from earlier pages") | `../method/notation-and-glossary.qmd#term-sigma-obs` |
| beyond-point-masses.qmd | "covariance-shaped jitter mode" | `../method/the-particle-filter.qmd#sec-resampling` |
| gravitational-echolocation.qmd | "evidence normalization" (in "Only the absolute evidence normalization shifts") | `../method/the-particle-filter.qmd#sec-evidence` |

Deliberately **not** linked (record these in the PR description too):

- "particle filter" and "resampling" on the-search-in-one-dimension —
  the same paragraph already links The Particle Filter.
- "posterior" on the-search-in-one-dimension — "The cloud *is* the
  posterior" is itself that page's definition of the term, not an
  unexplained occurrence.
- "evidence" on how-many-masses — defined in the same sentence ("compare
  **evidence**: the probability each model assigned…"); that page is the
  evidence treatment.
- "simulation units" on clocks-as-gravimeters — same sentence already
  links Units and Scales.
- "mass–distance degeneracy" on one-clock-is-not-enough (that page is
  the treatment) and on gravitational-echolocation and
  beyond-point-masses (both already link one-clock-is-not-enough at that
  exact phrase).

- [ ] **Step 1: Apply the 12 link edits from the matrix**

Example of the edit pattern (index.qmd; all others follow the same shape):

```markdown
This site builds that detector: a
[forward model](method/notation-and-glossary.qmd#term-forward-model) from
general relativity, a
[particle filter](method/the-particle-filter.qmd#sec-intro) for the
[inverse problem](method/notation-and-glossary.qmd#term-inverse-problem),
and a series of increasingly hard detection puzzles.
```

Note: index.qmd sits at the site root, so its hrefs have no `../` prefix;
story-page hrefs do.

- [ ] **Step 2: Append the Task 2 rows to CONTRACT in `scripts/check_site_links.py`**

Append before the closing `]`, then re-run
`uv run ruff format scripts/check_site_links.py`:

```python
    # Task 2 — story/landing cross-links
    ("content", "index.html", "method/the-particle-filter.html", "sec-intro"),
    ("content", "index.html", "method/notation-and-glossary.html", "term-forward-model"),
    ("content", "index.html", "method/notation-and-glossary.html", "term-inverse-problem"),
    ("content", "story/clocks-as-gravimeters.html", "method/notation-and-glossary.html", "term-weak-field"),
    ("content", "story/one-clock-is-not-enough.html", "method/notation-and-glossary.html", "term-sigma-obs"),
    ("content", "story/into-the-plane.html", "method/notation-and-glossary.html", "term-prior"),
    ("content", "story/into-the-plane.html", "method/notation-and-glossary.html", "term-index-conventions"),
    ("content", "story/two-hidden-masses.html", "method/the-particle-filter.html", "sec-resampling"),
    ("content", "story/two-hidden-masses.html", "method/notation-and-glossary.html", "term-posterior"),
    ("content", "story/beyond-point-masses.html", "method/notation-and-glossary.html", "term-sigma-obs"),
    ("content", "story/beyond-point-masses.html", "method/the-particle-filter.html", "sec-resampling"),
    ("content", "story/gravitational-echolocation.html", "method/the-particle-filter.html", "sec-evidence"),
```

- [ ] **Step 3: Render and run the checker**

```bash
cd /Users/jonathan/projects/clocks/site && uv run --frozen quarto render
cd /Users/jonathan/projects/clocks && uv run python scripts/check_site_links.py
```
Expected: `Link contract OK: 25 rows checked.`

- [ ] **Step 4: Repo gate, commit, push**

```bash
cd /Users/jonathan/projects/clocks
uv run ruff format --check . && uv run ruff check . && uv run pytest
git add site/index.qmd site/story/*.qmd scripts/check_site_links.py
git commit -m "$(cat <<'EOF'
Cross-link story pages to method anchors and glossary terms

<model trailer>
EOF
)"
git push
```

---

### Task 3: Architecture page

**Files:**
- Create: `site/reproduce/architecture.qmd`
- Modify: `site/_quarto.yml` (one sidebar entry)
- Modify: `scripts/check_site_links.py` (append CONTRACT rows)

**Interfaces:**
- Produces: the page Task 4's README links target (`reproduce/architecture.html`).

- [ ] **Step 1: Create the architecture page**

Create `site/reproduce/architecture.qmd` with exactly this content:

````markdown
---
title: "Architecture"
---

A map of `src/clocks` for anyone reading or extending the code — a
C4-inspired **module map** (modules grouped into functional clusters),
not a formal C4 component diagram. Solid arrows mean "imports from";
the dashed arrow is a runtime relationship. Re-exports inside
`clocks/__init__.py` are deliberately omitted.

```{mermaid}
flowchart TB
    subgraph contracts["Data contracts"]
        types
        config
        results
    end
    subgraph physnoise["Physics & noise"]
        physics
        noise
    end
    subgraph infer_c["Inference"]
        inference
    end
    subgraph public["Public API"]
        api
    end
    subgraph vis["Visualization"]
        viz
        animate["_animate"]
        panels["_panels"]
        panels3d["_panels3d"]
    end
    subgraph scen["Scenario tooling"]
        scenarios["_scenarios"]
        echo["_echo_study"]
    end
    subgraph entry["Entry points"]
        cli["_cli"]
    end
    scripts_node["scripts/*.py"]

    config --> types
    results --> config
    results --> types
    physics --> types
    inference --> noise
    inference --> physics
    inference --> types
    api --> config
    api --> inference
    api --> noise
    api --> physics
    api --> results
    api --> types
    viz --> animate
    viz --> panels
    viz --> panels3d
    animate --> panels
    animate --> panels3d
    animate --> inference
    animate --> physics
    animate --> types
    panels --> types
    panels3d --> types
    scenarios --> api
    scenarios --> config
    scenarios --> inference
    scenarios --> physics
    scenarios --> results
    scenarios --> types
    echo --> scenarios
    echo --> physics
    cli -. "runs via runpy" .-> scripts_node
```

**In words** (text equivalent of the diagram): `types` is the foundation
nearly everything builds on — only `noise` and `_cli` stand apart.
`config` and `results` define the public configuration and result
dataclasses on top of it. `physics` computes clock rates from mass
configurations; `noise` (which imports nothing from the package) models
observation noise. `inference` — the particle filter — consumes physics,
noise, and types. `api` ties all of it into the `simulate` / `infer` /
`simulate_and_infer` / `build_particle_filter` entry points. `viz` is a
pure facade re-exporting the private `_animate`, `_panels`, and
`_panels3d` plotting modules. `_scenarios` builds shared demo/test
scenarios on the public API; `_echo_study` adds echolocation-study
reporting on top of it. `_cli` maps the `uv run demo-*` commands onto the
scripts in `scripts/`, executing them with `runpy` (with an `importlib`
fallback); `_cli.py` itself imports nothing from the library, but the
scripts it runs do.

## The public/private boundary

The package's promised surface is the set of names curated in
`clocks/__init__.py::__all__` — the API entry points, the config and
result dataclasses, core types, and selected physics, noise, inference,
and viz names. Entire modules are *not* the promised surface, and every
underscore-prefixed module (`_animate`, `_panels`, `_panels3d`,
`_scenarios`, `_echo_study`, `_cli`) is internal: import from `clocks`
directly, not from its internals.
````

- [ ] **Step 2: Add the sidebar entry**

In `site/_quarto.yml`, under `section: "Part 3 — Reproduce"`, after the
Reproducibility entry, add:

```yaml
          - text: "Architecture"
            href: reproduce/architecture.qmd
```

- [ ] **Step 3: Verify the diagram against the real import graph**

```bash
cd /Users/jonathan/projects/clocks && grep -n "^from clocks" src/clocks/*.py | grep -v __init__
```
Confirm every solid arrow in the diagram corresponds to an import and no
import is missing from the diagram. If they disagree, fix the diagram,
not the code.

- [ ] **Step 4: Append the Task 3 rows to CONTRACT**

```python
    # Task 3 — architecture page sidebar entry
    ("sidebar", "index.html", "reproduce/architecture.html", None),
```

- [ ] **Step 5: Render, check, visual pass**

```bash
cd /Users/jonathan/projects/clocks/site && uv run --frozen quarto render
cd /Users/jonathan/projects/clocks && uv run python scripts/check_site_links.py
```
Expected: `Link contract OK: 26 rows checked.`
Open `site/_output/reproduce/architecture.html`: diagram legible in light
and dark themes and at a narrow window (the Mermaid block may scroll
horizontally inside its own container — the page must not).

- [ ] **Step 6: Repo gate, commit, push**

```bash
cd /Users/jonathan/projects/clocks
uv run ruff format --check . && uv run ruff check . && uv run pytest
git add site/reproduce/architecture.qmd site/_quarto.yml scripts/check_site_links.py
git commit -m "$(cat <<'EOF'
Add architecture page: C4-inspired module map of src/clocks

<model trailer>
EOF
)"
git push
```

---

### Task 4: README slimming and getting-started consistency

**Files:**
- Modify: `README.md`
- Modify: `site/reproduce/getting-started.qmd`
- Modify: `scripts/check_site_links.py` (append CONTRACT rows)

**Interfaces:**
- Consumes: `reproduce/architecture.html` from Task 3.

- [ ] **Step 1: Replace README's "Use as a library" section**

Delete everything from `## Use as a library` up to (not including)
`## Run the demos` — both code blocks and the fixed-K/`build_particle_filter`
paragraphs — and replace with:

```markdown
## Use as a library

The package exposes stable end-to-end entry points — `simulate`, `infer`,
and `simulate_and_infer` — configured via public dataclasses
(`SimulationConfig`, `InferenceConfig`, ...), plus `build_particle_filter`
for driving the filter observation-by-observation. A complete
simulate-then-infer round trip runs live on the site's
[Getting Started](https://jbwhit.github.io/clocks/reproduce/getting-started.html)
page, and the filter machinery is documented in
[The Particle Filter](https://jbwhit.github.io/clocks/method/the-particle-filter.html).
```

- [ ] **Step 2: Compress README's demo catalog**

Replace the per-demo subsections (each command + embed + description,
from the first `**1D**` entry through the echolocation GIF embed,
keeping the `## Run the demos` heading) with the following — six GIF
demos plus one static figure, all seven commands retained:

```markdown
Six animated demos and one static figure:

​```bash
uv run demo-1d                  # → output/demo_1d.gif
uv run demo-2d                  # → output/demo_2d.gif
uv run demo-multi-mass          # → output/demo_multi_mass.gif
uv run demo-multi-mass-2d       # → output/demo_multi_mass_2d.gif
uv run demo-model-comparison    # → output/demo_model_comparison.gif
uv run demo-density             # → output/demo_density.png
uv run demo-echolocation-3d     # → output/demo_echolocation_3d.gif
​```

![2D inference demo](assets/demo_2d.gif)

The GIF demos animate the physical setup, the particle cloud converging,
and the estimates' uncertainty; `demo-density` produces a static
comparison figure. All seven, with commentary:
[jbwhit.github.io/clocks](https://jbwhit.github.io/clocks/). The
echolocation range study behind the site's final page:
`scripts/scan_echolocation_range.py`.
```

(The `​` marks above escape the nested code fence in this plan — write
plain ``` fences in the README.)

- [ ] **Step 3: Point README's project structure at the architecture page**

At the end of the `## Project structure` section (after the tests line),
add:

```markdown
Dependency structure and the public/private boundary:
[Architecture](https://jbwhit.github.io/clocks/reproduce/architecture.html).
```

- [ ] **Step 4: Fix the two getting-started.qmd claims**

Edit 1 — replace:
```markdown
The same example as the repository README, executed live on this page —
if you can read the output below, the install instructions above work:
```
with:
```markdown
A complete simulate-then-infer round trip, executed live on this page —
if you can read the output below, the install instructions above work:
```

Edit 2 — replace:
```markdown
For fixed-K inference, pass an integer to `n_masses`; to drive the filter
observation-by-observation (e.g. for animation), use
`build_particle_filter` — both are documented in the
[README](https://github.com/jbwhit/clocks#use-as-a-library).
```
with:
```markdown
For fixed-K inference, pass an integer to `n_masses`; to drive the filter
observation-by-observation (e.g. for animation), use
`build_particle_filter` — see
[The Particle Filter](../method/the-particle-filter.qmd).
```

- [ ] **Step 5: Append the Task 4 rows to CONTRACT**

```python
    # Task 4 — README → site links and getting-started repoint
    ("content", "README.md", "reproduce/getting-started.html", None),
    ("content", "README.md", "method/the-particle-filter.html", None),
    ("content", "README.md", "index.html", None),
    ("content", "README.md", "reproduce/architecture.html", None),
    ("content", "reproduce/getting-started.html", "method/the-particle-filter.html", None),
```

- [ ] **Step 6: Render, check**

```bash
cd /Users/jonathan/projects/clocks/site && uv run --frozen quarto render
cd /Users/jonathan/projects/clocks && uv run python scripts/check_site_links.py
```
Expected: `Link contract OK: 31 rows checked.`

- [ ] **Step 7: Repo gate, commit, push**

```bash
cd /Users/jonathan/projects/clocks
uv run ruff format --check . && uv run ruff check . && uv run pytest
git add README.md site/reproduce/getting-started.qmd scripts/check_site_links.py
git commit -m "$(cat <<'EOF'
Slim README to install-and-run; point details at the site

<model trailer>
EOF
)"
git push
```

---

### Task 5: PR, CI, and final review

- [ ] **Step 1: Open the PR**

Write the PR body to a scratch file first:

```bash
cd /Users/jonathan/projects/clocks
cat > /tmp/docs-pass-pr-body.md <<'EOF'
Implements the Diataxis documentation pass per the approved spec
`docs/superpowers/specs/2026-08-10-diataxis-docs-pass-design.md`
(Codex xhigh: SOUND ENOUGH TO IMPLEMENT, round 5).

Four commits in dependency order:
1. Notation & Glossary page + stable anchors + link-contract checker
2. Story/landing cross-links (matrix below)
3. Architecture page (C4-inspired module map)
4. README slimmed; getting-started claims fixed

## Link contract

<paste the Task 2 link matrix table AND the deliberately-not-linked list
from docs/superpowers/plans/2026-08-10-diataxis-docs-pass.md verbatim>

Verified bidirectionally by `scripts/check_site_links.py` (31 rows) on
the rendered site.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
gh pr create --title "Diataxis docs pass: glossary, cross-links, architecture page, slim README" --body-file /tmp/docs-pass-pr-body.md
```

- [ ] **Step 2: Verify CI green**

Poll `gh pr checks` until every check reports a **non-empty conclusion**;
an empty result is not evidence of green (it can mean pending, failing,
or no CI). Investigate any failure immediately.

- [ ] **Step 3: Codex xhigh review of the final diff**

```bash
cd /Users/jonathan/projects/clocks
codex exec --sandbox read-only -c model_reasoning_effort="xhigh" "Review the final diff of branch claude-diataxis-docs-pass against main (run: git diff main...claude-diataxis-docs-pass) for the Diataxis docs pass, against the approved spec docs/superpowers/specs/2026-08-10-diataxis-docs-pass-design.md. Check spec conformance, link correctness, physics wording, and the checker script. End with a hard verdict line: 'READY TO MERGE' or 'NEEDS REVISION' with blocking issues." < /dev/null > /tmp/codex-pr-review.txt 2>&1
```

The answer is after the last `codex` marker, before `tokens used`. Triage
findings with rigor; apply fixes, commit (full per-commit gate), push,
re-review — repeat until "READY TO MERGE". Post each round's findings,
responses, and the verdict on the PR via
`gh pr comment --body-file <round-file>`, attributed
("**Codex xhigh (round N):** …").

- [ ] **Step 4: Merge and confirm**

After the Codex verdict (counts as approval per the standing protocol):
merge with `gh pr merge --squash --delete-branch`, confirm CI stays green
on main afterward (`gh run list --branch main` — wait for non-empty
conclusions), fast-forward the local checkout
(`git checkout main && git pull --no-rebase --ff-only`), and confirm the
deployed site (`site.yml` runs on push to main) shows the two new pages.
