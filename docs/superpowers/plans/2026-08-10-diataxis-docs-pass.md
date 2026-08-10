# Diataxis Documentation Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a notation/glossary page, story→method cross-links, an architecture page, and a slimmed README to the "GPS in Reverse" Quarto site, per the approved spec `docs/superpowers/specs/2026-08-10-diataxis-docs-pass-design.md`.

**Architecture:** Four content changes to `site/` and `README.md`, landed in dependency order so every commit renders with no dangling link targets. A stdlib-only checker script (`scripts/check_site_links.py`) holds the link contract as data and verifies every added link bidirectionally against the rendered `_output/`; its contract grows with each task.

**Tech Stack:** Quarto (`.qmd`, Mermaid diagrams), Python 3.12+ via uv (checker script, stdlib only).

## Global Constraints

- Branch: `claude-diataxis-docs-pass` (from `main`).
- Every commit message ends with a `Co-Authored-By:` trailer naming the model actually authoring it, plus the `Claude-Session:` trailer if the harness provides one.
- **Per-commit gate (all three, in order, before every commit):**
  1. Render: `cd site && uv run --frozen quarto render` — must complete with no errors.
  2. Link contract: `uv run python scripts/check_site_links.py` (from repo root) — must exit 0 (from Task 1 onward).
  3. Repo gate: `uv run ruff format --check .` AND `uv run ruff check .` AND `uv run pytest` (190 passing, 2 deselected as of plan date) — all green.
- No changes to `src/clocks/` or `tests/` (spec: out of scope). `scripts/check_site_links.py` is the only Python file added.
- Prose voice on story pages must not change: cross-links attach to existing wording; never add or reword sentences to create a linking opportunity (exceptions listed explicitly in Task 2 are the *only* text changes).
- Published-site base URL for README links: `https://jbwhit.github.io/clocks/`.
- The rendered site output dir is `site/_output/` (gitignored — never commit it).

---

### Task 1: Glossary page, method-page anchors, and link checker

**Files:**
- Create: `site/method/notation-and-glossary.qmd`
- Create: `scripts/check_site_links.py`
- Modify: `site/method/the-particle-filter.qmd` (two heading anchors)
- Modify: `site/_quarto.yml` (one sidebar entry)

**Interfaces:**
- Produces: glossary anchors `#term-weak-field`, `#term-time-dilation`, `#term-forward-model`, `#term-inverse-problem`, `#term-degeneracy`, `#term-prior`, `#term-likelihood`, `#term-posterior`, `#term-evidence`, `#term-resampling`, `#term-jitter`, `#term-ess`, `#term-model-comparison`, `#term-label-switching`, `#term-chronometric-leveling`, `#term-sigma-obs`, `#term-index-conventions` (Task 2 links to these — names must match exactly).
- Produces: method anchors `#sec-resampling`, `#sec-evidence` on `the-particle-filter.qmd`.
- Produces: `scripts/check_site_links.py` with a module-level `CONTRACT` list that Tasks 2–4 append rows to.

- [ ] **Step 1: Add explicit anchors to the particle-filter page**

In `site/method/the-particle-filter.qmd`, change exactly two headings:

```markdown
## Resampling {#sec-resampling}
```
(replaces `## Resampling`)

```markdown
## Evidence {#sec-evidence}
```
(replaces `## Evidence`)

- [ ] **Step 2: Create the glossary page**

Create `site/method/notation-and-glossary.qmd` with exactly this content:

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

::: {.callout-note title="Two overloaded symbols"}
**$\sigma$** means observation noise on the inference pages but Gaussian
profile *width* in [Beyond Point Masses](../story/beyond-point-masses.qmd)
(which flags the reuse itself); this page writes $\sigma_{\text{obs}}$ vs
$\sigma_{\text{density}}$ to keep them apart. **$\tau$** means proper time
in $d\tau/dt$, but in the
[echolocation coda](../story/gravitational-echolocation.qmd) it is a clock's
*averaging time* in the stability model $\sigma_y(\tau) = 10^{-16}/\sqrt{\tau}$.
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
  and a heavy mass far away can produce identical readings; broken by
  array geometry ([One Clock Is Not Enough](../story/one-clock-is-not-enough.qmd)).
- [**Prior**]{#term-prior} — what the filter assumes before any data:
  uniform ranges for positions and masses
  ([The Particle Filter](the-particle-filter.qmd)).
- [**Likelihood**]{#term-likelihood} — how well a hypothesis predicts an
  observed set of clock rates; Gaussian in the observation noise
  ([The Particle Filter](the-particle-filter.qmd)).
- [**Posterior**]{#term-posterior} — the belief after data; the particle
  cloud *is* the posterior, its spread the calibrated uncertainty
  ([The Search in One Dimension](../story/the-search-in-one-dimension.qmd)).
- [**Evidence**]{#term-evidence} — the probability a model assigned to the
  data it actually saw; what
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

- [ ] **Step 3: Add the sidebar entry**

In `site/_quarto.yml`, under `section: "Part 2 — Under the Hood"`, after
the Units and Scales entry, add:

```yaml
          - text: "Notation & Glossary"
            href: method/notation-and-glossary.qmd
```

- [ ] **Step 4: Write the link checker script**

Create `scripts/check_site_links.py` with exactly this content:

```python
"""Check the documentation link contract for the rendered site.

Verifies every internal link added by the Diataxis docs pass (spec:
docs/superpowers/specs/2026-08-10-diataxis-docs-pass-design.md).
For each contract row the SOURCE must contain a link resolving to the
target page (and fragment, if given), and the TARGET file must exist
(and contain the fragment id, if given). A missing source link is a
failure.

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

# (source, target, fragment) — source/target are html paths relative to
# site/_output; source may also be "README.md" (parsed as Markdown, with
# published-site URLs mapped onto _output paths). fragment=None means a
# page-only link: the source link and the target file must exist, but no
# id is required.
CONTRACT: list[tuple[str, str, str | None]] = [
    # Task 1 — glossary outbound links (one per destination page)
    ("method/notation-and-glossary.html", "story/clocks-as-gravimeters.html", None),
    ("method/notation-and-glossary.html", "story/one-clock-is-not-enough.html", None),
    ("method/notation-and-glossary.html", "story/the-search-in-one-dimension.html", None),
    ("method/notation-and-glossary.html", "story/into-the-plane.html", None),
    ("method/notation-and-glossary.html", "story/two-hidden-masses.html", None),
    ("method/notation-and-glossary.html", "story/how-many-masses.html", None),
    ("method/notation-and-glossary.html", "story/beyond-point-masses.html", None),
    ("method/notation-and-glossary.html", "story/gravitational-echolocation.html", None),
    ("method/notation-and-glossary.html", "method/units-and-scales.html", None),
    ("method/notation-and-glossary.html", "method/the-particle-filter.html", "sec-resampling"),
    ("method/notation-and-glossary.html", "method/the-particle-filter.html", "sec-evidence"),
    # Task 1 — sidebar entry (checked on the landing page)
    ("index.html", "method/notation-and-glossary.html", None),
]


class _PageIndex(HTMLParser):
    """Collect all href values and element ids from one HTML page."""

    def __init__(self) -> None:
        super().__init__()
        self.hrefs: set[str] = set()
        self.ids: set[str] = set()

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        for name, value in attrs:
            if value is None:
                continue
            if name == "href":
                self.hrefs.add(value)
            elif name == "id":
                self.ids.add(value)


def _map_site_url(url: str) -> tuple[str, str | None] | None:
    """Map a published-site URL to an _output-relative path + fragment."""
    if not url.startswith(SITE_BASE):
        return None
    parsed = urlparse(url)
    rel = parsed.path[len(urlparse(SITE_BASE).path) :]
    if not rel or rel.endswith("/"):
        rel += "index.html"
    return rel, (parsed.fragment or None)


def _resolve_href(
    source: str, href: str
) -> tuple[str, str | None] | None:
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


def _source_links(source: str) -> set[tuple[str, str | None]] | None:
    """All resolved internal links in a source (HTML page or README)."""
    if source == "README.md":
        urls = re.findall(r"\]\(([^)\s]+)\)", README.read_text(encoding="utf-8"))
        return {m for u in urls if (m := _map_site_url(u)) is not None}
    page = _load(source)
    if page is None:
        return None
    return {m for h in page.hrefs if (m := _resolve_href(source, h)) is not None}


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


def main() -> int:
    failures: list[str] = []
    for source, target, fragment in CONTRACT:
        label = f"{source} -> {target}" + (f"#{fragment}" if fragment else "")
        target_page = _load(target)
        if target_page is None:
            failures.append(f"{label}: target file missing")
            continue
        if fragment is not None and fragment not in target_page.ids:
            failures.append(f"{label}: id '{fragment}' missing in target")
        links = _source_links(source)
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

After writing the script (and after every later CONTRACT edit), run
`uv run ruff format scripts/check_site_links.py` — the long CONTRACT rows
exceed the 88-char line limit as typed, and the format --check gate would
otherwise fail.

- [ ] **Step 5: Render and run the checker — expect it to pass**

```bash
cd site && uv run --frozen quarto render && cd ..
uv run python scripts/check_site_links.py
```
Expected: render completes; checker prints `Link contract OK: 12 rows checked.`
If any row fails, fix the glossary/anchors (not the checker) unless the
checker itself has a bug.

- [ ] **Step 6: Visual check**

Open `site/_output/method/notation-and-glossary.html` in a browser: check
light and dark themes and a narrow (~400px) window. Every entry ≤ 2
rendered lines at desktop width; tables scroll or wrap at mobile width
without horizontal page scroll.

- [ ] **Step 7: Repo gate, commit, push**

```bash
uv run ruff format --check . && uv run ruff check . && uv run pytest
git add site/method/notation-and-glossary.qmd site/method/the-particle-filter.qmd site/_quarto.yml scripts/check_site_links.py
git commit -m "Add notation & glossary page with stable anchors and link checker"
git push -u origin claude-diataxis-docs-pass
```

---

### Task 2: Story and landing-page cross-links

**Files:**
- Modify: `site/index.qmd`
- Modify: `site/story/clocks-as-gravimeters.qmd`
- Modify: `site/story/one-clock-is-not-enough.qmd`
- Modify: `site/story/the-search-in-one-dimension.qmd`
- Modify: `site/story/into-the-plane.qmd`
- Modify: `site/story/two-hidden-masses.qmd`
- Modify: `site/story/how-many-masses.qmd`
- Modify: `site/story/beyond-point-masses.qmd`
- Modify: `site/story/gravitational-echolocation.qmd`
- Modify: `scripts/check_site_links.py` (append CONTRACT rows)

**Interfaces:**
- Consumes: glossary `#term-*` anchors and method `#sec-*` anchors from Task 1 (exact names listed there).

Each edit below wraps existing words in a link — no wording changes. The
link matrix (also the PR-description artifact):

| Source page | Linked text (existing words) | Target |
|---|---|---|
| index.qmd | "particle filter" (in "a particle filter for the inverse problem") | `method/the-particle-filter.qmd` |
| index.qmd | "forward model" (in "a forward model from general relativity") | `method/notation-and-glossary.qmd#term-forward-model` |
| index.qmd | "inverse problem" (same sentence) | `method/notation-and-glossary.qmd#term-inverse-problem` |
| clocks-as-gravimeters.qmd | "weak-field result" (first sentence) | `../method/notation-and-glossary.qmd#term-weak-field` |
| one-clock-is-not-enough.qmd | "observation noise $\sigma$" (noise-floor callout) | `../method/notation-and-glossary.qmd#term-sigma-obs` |
| the-search-in-one-dimension.qmd | "posterior" (in "The cloud *is* the posterior") | `../method/notation-and-glossary.qmd#term-posterior` |
| into-the-plane.qmd | "prior" (in "only the prior gains a column") | `../method/notation-and-glossary.qmd#term-prior` |
| into-the-plane.qmd | "$-M_j/r_{ij}$" (in "The potential sums $-M_j/r_{ij}$") | `../method/notation-and-glossary.qmd#term-index-conventions` |
| two-hidden-masses.qmd | "resampling jitter" (in "after every resampling jitter") | `../method/the-particle-filter.qmd#sec-resampling` |
| two-hidden-masses.qmd | "posterior" (in "the true posterior is perfectly bimodal") | `../method/notation-and-glossary.qmd#term-posterior` |
| how-many-masses.qmd | "evidence" (the bolded "**evidence**" in the first paragraph — link text inside the bold) | `../method/the-particle-filter.qmd#sec-evidence` |
| beyond-point-masses.qmd | "observation noise" (in "not the observation noise from earlier pages") | `../method/notation-and-glossary.qmd#term-sigma-obs` |
| beyond-point-masses.qmd | "covariance-shaped jitter mode" | `../method/the-particle-filter.qmd#sec-resampling` |
| gravitational-echolocation.qmd | "evidence normalization" (in "Only the absolute evidence normalization shifts") | `../method/the-particle-filter.qmd#sec-evidence` |

Deliberately **not** linked (existing nearby links already serve the
reader — record these in the PR description too): "particle filter" and
"resampling" on the-search-in-one-dimension (same paragraph already links
The Particle Filter); "simulation units" on clocks-as-gravimeters (same
sentence already links Units and Scales); "mass–distance degeneracy" on
one-clock-is-not-enough (that page is the treatment) and on
gravitational-echolocation (already links one-clock-is-not-enough).

- [ ] **Step 1: Apply the 14 link edits from the matrix**

Example of the edit pattern (index.qmd; all others follow the same shape):

```markdown
This site builds that detector: a
[forward model](method/notation-and-glossary.qmd#term-forward-model) from
general relativity, a
[particle filter](method/the-particle-filter.qmd) for the
[inverse problem](method/notation-and-glossary.qmd#term-inverse-problem),
and a series of increasingly hard detection puzzles.
```

Note: index.qmd sits at the site root, so its hrefs have no `../` prefix;
story-page hrefs do.

- [ ] **Step 2: Append the Task 2 rows to CONTRACT in `scripts/check_site_links.py`**

```python
    # Task 2 — story/landing cross-links
    ("index.html", "method/the-particle-filter.html", None),
    ("index.html", "method/notation-and-glossary.html", "term-forward-model"),
    ("index.html", "method/notation-and-glossary.html", "term-inverse-problem"),
    ("story/clocks-as-gravimeters.html", "method/notation-and-glossary.html", "term-weak-field"),
    ("story/one-clock-is-not-enough.html", "method/notation-and-glossary.html", "term-sigma-obs"),
    ("story/the-search-in-one-dimension.html", "method/notation-and-glossary.html", "term-posterior"),
    ("story/into-the-plane.html", "method/notation-and-glossary.html", "term-prior"),
    ("story/into-the-plane.html", "method/notation-and-glossary.html", "term-index-conventions"),
    ("story/two-hidden-masses.html", "method/the-particle-filter.html", "sec-resampling"),
    ("story/two-hidden-masses.html", "method/notation-and-glossary.html", "term-posterior"),
    ("story/how-many-masses.html", "method/the-particle-filter.html", "sec-evidence"),
    ("story/beyond-point-masses.html", "method/notation-and-glossary.html", "term-sigma-obs"),
    ("story/beyond-point-masses.html", "method/the-particle-filter.html", "sec-resampling"),
    ("story/gravitational-echolocation.html", "method/the-particle-filter.html", "sec-evidence"),
```

- [ ] **Step 3: Render and run the checker**

```bash
cd site && uv run --frozen quarto render && cd ..
uv run python scripts/check_site_links.py
```
Expected: `Link contract OK: 26 rows checked.`

- [ ] **Step 4: Repo gate, commit, push**

```bash
uv run ruff format --check . && uv run ruff check . && uv run pytest
git add site/index.qmd site/story/*.qmd scripts/check_site_links.py
git commit -m "Cross-link story pages to method anchors and glossary terms"
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
every layer builds on. `config` and `results` define the public
configuration and result dataclasses on top of it. `physics` computes
clock rates from mass configurations; `noise` (which imports nothing from
the package) models observation noise. `inference` — the particle filter —
consumes physics, noise, and types. `api` ties all of it into the
`simulate` / `infer` / `build_particle_filter` entry points. `viz` is a
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
grep -n "^from clocks" src/clocks/*.py
```
Confirm every solid arrow in the diagram corresponds to an import and no
import is missing from the diagram (ignoring `__init__.py`). If they
disagree, fix the diagram, not the code.

- [ ] **Step 4: Append the Task 3 rows to CONTRACT**

```python
    # Task 3 — architecture page sidebar entry
    ("index.html", "reproduce/architecture.html", None),
```

- [ ] **Step 5: Render, check, visual pass**

```bash
cd site && uv run --frozen quarto render && cd ..
uv run python scripts/check_site_links.py
```
Expected: `Link contract OK: 27 rows checked.`
Open `site/_output/reproduce/architecture.html`: diagram legible in light
and dark themes and at a narrow window (the Mermaid block may scroll
horizontally inside its own container — the page must not).

- [ ] **Step 6: Repo gate, commit, push**

```bash
uv run ruff format --check . && uv run ruff check . && uv run pytest
git add site/reproduce/architecture.qmd site/_quarto.yml scripts/check_site_links.py
git commit -m "Add architecture page: C4-inspired module map of src/clocks"
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
and `build_particle_filter` — configured and returned via public
dataclasses (`SimulationConfig`, `InferenceConfig`, `SimulationResult`,
`InferenceResult`, ...). A complete simulate-then-infer round trip runs
live on the site's
[Getting Started](https://jbwhit.github.io/clocks/reproduce/getting-started.html)
page, and the filter machinery is documented in
[The Particle Filter](https://jbwhit.github.io/clocks/method/the-particle-filter.html).
```

- [ ] **Step 2: Compress README's demo catalog**

Replace the per-demo subsections (each command + GIF embed + description,
from the first `**1D**` entry through the echolocation GIF embed,
keeping the `## Run the demos` heading) with:

```markdown
All seven demo commands (rough laptop runtimes on the site):

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

Each demo animates the physical setup, the particle cloud converging, and
the estimates' uncertainty. All seven, with commentary:
[jbwhit.github.io/clocks](https://jbwhit.github.io/clocks/). The
echolocation range study behind the site's final page:
`scripts/scan_echolocation_range.py`.
```

(The `​` marks above are to escape the nested code fence in this plan —
write plain ``` fences in the README.)

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
    ("README.md", "reproduce/getting-started.html", None),
    ("README.md", "method/the-particle-filter.html", None),
    ("README.md", "index.html", None),
    ("README.md", "reproduce/architecture.html", None),
    ("reproduce/getting-started.html", "method/the-particle-filter.html", None),
]
```
(The closing `]` shown is the existing list terminator — the four rows go
before it.)

- [ ] **Step 6: Render, check**

```bash
cd site && uv run --frozen quarto render && cd ..
uv run python scripts/check_site_links.py
```
Expected: `Link contract OK: 32 rows checked.`

- [ ] **Step 7: Repo gate, commit, push**

```bash
uv run ruff format --check . && uv run ruff check . && uv run pytest
git add README.md site/reproduce/getting-started.qmd scripts/check_site_links.py
git commit -m "Slim README to install-and-run; point details at the site"
git push
```

---

### Task 5: PR, CI, and final review

- [ ] **Step 1: Open the PR**

```bash
gh pr create --title "Diataxis docs pass: glossary, cross-links, architecture page, slim README" --body "<summary of the four commits; paste the Task 2 link matrix (linked + deliberately-not-linked lists) as the link-contract artifact; link the spec docs/superpowers/specs/2026-08-10-diataxis-docs-pass-design.md>"
```

- [ ] **Step 2: Verify CI green**

Poll `gh pr checks` until every check reports a **non-empty conclusion**;
an empty result is not evidence of green. Investigate any failure
immediately.

- [ ] **Step 3: Codex xhigh review of the final diff**

Launch per the standing protocol; go rounds until "READY TO MERGE"; post
each round's findings and responses on the PR via `gh`, attributed.

- [ ] **Step 4: Merge and confirm**

After Codex verdict (counts as approval): `gh pr merge --squash` (or repo
convention), confirm CI stays green on main, fast-forward the local
checkout, and confirm the deployed site
(`site.yml` runs on push to main) shows the new pages.
