# Diataxis-Guided Documentation Pass — Design

**Date:** 2026-08-10
**Status:** Revised after Codex xhigh rounds 1–3 (all NEEDS REVISION → fixes applied); round 4 pending
**Review trail:**
Codex xhigh round 3 (2026-08-10) — verdict NEEDS REVISION. Three blockers,
all accepted and fixed: (1) index conventions were self-contradictory
(i = particles vs i = clock in r_ij) — glossary now declares index roles
context-dependent; (2) README is not rendered into `_output/`, so its links
are checked from Markdown source with URL→local mapping; (3) link contract
now distinguishes fragment links (require exact target `id`) from page-only
links (require target file), and adds the sidebar entries to scope. Stale
gate-frequency sentence removed (per-commit confirmed by Jonathan).
Codex xhigh round 2 (2026-08-10) — verdict NEEDS REVISION. Three blockers,
all accepted and fixed: (1) notation plan contradicted the site (bare r =
distance vs indexed r_c = tick rate; τ overloaded as proper time and as
averaging time in the echolocation stability model; r_ij/M_j/index roles
missing; inclusion criterion contradicted the tables — glossary is now
explicitly a consolidated lookup); (2) link verification was one-sided —
now bidirectional (source href + target id) over all links added by the
pass, via an HTML-parser checker; (3) gate frequency said "per push" while
the trail said "per commit" — now consistently per commit.
Codex xhigh round 1 (2026-08-10) — verdict NEEDS REVISION.
Blocking issues (all accepted and fixed below): incorrect/ambiguous notation
plan, incorrect dependency graph (noise→types edge was false; `_panels*`
edges missing), README slimming would falsify `getting-started.qmd` claims,
ineffective link verification, commit order incompatible with independently
renderable commits. Framing fixes also accepted: "Getting Started" is a
quickstart/how-to (not a Diataxis tutorial), quadrants are not completeness
boxes, "C4 Level 3" relabeled "C4-inspired module map", marginal reader made
concrete. One pushback: per-commit Python gate retained (suite is ~12 s;
standing repo policy) — see Verification.

## Goal

Improve the "GPS in Reverse" site for its marginal reader using Diataxis as
an audit lens (not a navigation scheme), plus one C4-inspired module map for
the secondary contributor audience.

**Marginal reader:** physics-curious site visitor — comfortable with algebra
and reading plots; no Bayesian/SMC vocabulary assumed; does not need to run
or read Python to follow the story.

## Background / audit findings

- **Part 1 — The Story** (8 pages) is understanding-oriented explanation and
  is the site's identity. It must not be restructured.
- **Part 2 — Under the Hood** is explanation with reference elements.
- **Part 3 — Reproduce** is quickstart/how-to material
  (`getting-started.qmd` is not a Diataxis tutorial — no managed learning
  arc — and does not need to become one).

The motivating gap is concrete, not quadrant-filling: mid-read, the visitor
has no lookup aid for symbols and terms of art. API reference (quartodoc)
and consumer how-to guides serve library users and are **out of scope**.

## Changes

Sections below are ordered by dependency, not priority; the commit plan
follows this order so every commit renders with no dangling link targets.

### 1. Notation & Glossary page (land first — link target for §2)

New file: `site/method/notation-and-glossary.qmd`, added to the Part 2
sidebar (after Units and Scales).

**Implementation step 0 — notation inventory.** Before writing the page,
inventory the notation actually displayed across all site pages (story,
method, index). The tables below are the expected shape; the inventory is
authoritative. The glossary is a **consolidated lookup**: a symbol defined
on its source page still belongs here — same-page definition does not
exclude it. Omit only notation that is purely local to a single method-page
derivation and never surfaces in story text.

Content:

- **Physics notation table:** Φ (potential), G, c (set to 1 in simulation
  units), τ vs t (proper vs coordinate time), tick rate (dτ/dt), bare r
  (distance) vs indexed r_c (tick rate of clock c in the particle-filter
  equations — same letter, distinguished by the index), r_ij (distance
  from clock i to mass j), M and indexed M_j (masses), index conventions
  — declared **context-dependent**: i indexes clocks/evaluation points in
  the potential sum (r_ij) but particles in the filter equations (w_i,
  θ_i); j indexes masses; c indexes clocks in r_c — r_s (Schwarzschild
  radius), x, y (positions), μ, A, σ_density (Gaussian density center,
  amplitude, width).
- **Inference notation table:** σ_obs (observation noise), N (particle
  count), w (particle weight), θ (parameter hypothesis), K (number of
  masses), ESS, evidence / log-evidence.
- **Overload disambiguation:** the site overloads two symbols and the
  glossary must surface both, noting each source page's shorthand:
  - σ — observation noise in the inference material vs Gaussian profile
    width in Beyond Point Masses (which itself flags the overload):
    glossary uses σ_obs vs σ_density.
  - τ — proper time in dτ/dt (Clocks as Gravimeters) vs clock averaging/
    integration time in the σ_y(τ) = 10⁻¹⁶/√τ stability model
    (Gravitational Echolocation coda): glossary lists both meanings
    explicitly.
- **Terms list:** weak field, time dilation, forward model, inverse
  problem, posterior, prior, likelihood, evidence, resampling
  (systematic/stratified/residual), jitter (incl. annealed), effective
  sample size, model comparison, degeneracy. One-line definitions, each
  linking to its fuller treatment.
- **Anchors:** every term gets a stable explicit ID (`#term-posterior`,
  `#term-jitter`, …). Story links target these fragments; auto-generated
  IDs are not relied on for cross-page links. Where §2 links target method
  pages, add explicit `{#sec-...}` IDs there too.

Size/format criterion (verifiable): two compact tables plus a terms list;
every entry ≤ 2 lines rendered at desktop width; page renders legibly at
mobile width; no prose sections.

### 2. Story → method/glossary cross-links

Scope: the eight story pages **and** `index.qmd` (the landing page
introduces *forward model*, *inverse problem*, and *particle filter* before
the story starts).

Link targets:

- Filter machinery (*particle filter*, *resampling*, *ESS*, *jitter*) →
  explicit anchors in `method/the-particle-filter.qmd`.
- Units/scales (*simulation units*, G = c = 1) →
  `method/units-and-scales.qmd`.
- Terms of art (*posterior*, *prior*, *likelihood*, *weak field*,
  *degeneracy*, *model comparison*, …) → `#term-*` anchors on the glossary.

Linking rule (replaces "first mention"):

- Link the first **unexplained** occurrence of a term on a page **only if
  no nearby existing link already serves the reader** — several story
  pages already link the particle-filter and units pages in the same
  passage where the terms appear; those existing links count.
- At most one link to the same page-and-fragment target per page.
- Never alter prose or introduce jargon to create a linking opportunity;
  links attach to existing wording only.

**Link contract (verification artifact):** implementation produces an
explicit matrix — source page × linked term × target fragment — checked
into the PR description. Verification checks every fragment in that matrix
exists in the rendered HTML (see Verification).

### 3. Architecture page — C4-inspired module map

New file: `site/reproduce/architecture.qmd`, added to the Part 3 sidebar
(after Reproducibility). Contributor-facing; lives in "Reproduce" — no new
Part 4.

This is a **module dependency map, C4-inspired** — not a C4 component
diagram (C4 warns that modules/packages are not normally components). One
Mermaid diagram, modules grouped into functional clusters:

- **Data contracts:** `types` (foundation), `config` → `types`,
  `results` → `config`, `types`.
- **Physics & noise:** `physics` → `types`; `noise` (imports NumPy only —
  no internal deps).
- **Inference:** `inference` → `noise`, `physics`, `types`.
- **Public API:** `api` → `config`, `inference`, `noise`, `physics`,
  `results`, `types`.
- **Visualization:** `viz` (pure facade) re-exports `_animate`, `_panels`,
  `_panels3d`; `_animate` → `_panels`, `_panels3d`, `inference`,
  `physics`, `types`; `_panels` → `types`; `_panels3d` → `types`.
- **Scenario tooling:** `_scenarios` → `api`, `config`, `inference`,
  `physics`, `results`, `types`; `_echo_study` → `_scenarios`, `physics`.
- **Entry points:** `_cli` runs `scripts/*.py` via `runpy` (with an
  `importlib` fallback); `_cli.py` itself has no library imports, but the
  scripts it runs do.

Diagram rules: import edges and the runtime "runs scripts" edge use
visually distinct styles with a legend; arrow direction = "depends on";
`__init__` re-export edges are deliberately omitted (stated in a caption).
The diagram gets a text summary (accessibility + non-rendering fallback).

Prose: a short walk of the public/private boundary — the public surface is
the set of names curated in `clocks/__init__.py::__all__` (modules
themselves are not the promised surface); underscore modules are internal.

Checks: renders in light and dark themes and at mobile width.

### 4. README slimming (+ getting-started.qmd consistency)

- Replace the "Use as a library" API walkthrough (both code blocks) with a
  2–3 sentence summary plus a link to the site.
- All **seven** demo commands remain listed; the **seven** embeds (six
  GIFs + the density PNG) compress to one representative asset with a link
  to the site for the rest.
- "Project structure": keep the top-level listing; point to the new
  architecture page for the dependency picture.
- Keep: intro, setup, run-tests sections unchanged.
- **Required consistency fixes in `site/reproduce/getting-started.qmd`:**
  it currently claims its example is "the same example as the repository
  README" and links the README for fixed-K inference and
  `build_particle_filter`. After the cut both claims are false. Reword the
  first (the page's live example stands on its own); repoint the second at
  the particle-filter method page (and, if the walkthrough content is
  worth keeping verbatim, this page — not the README — is its home).

## Out of scope

- Sidebar/nav restructure into Diataxis quadrant names.
- quartodoc / generated API reference; consumer how-to guides.
- C4 context/container/code diagrams or C4 notation ceremony.
- Any change to Python source or tests.

## Verification

- **Render gate (per commit):** `uv run --frozen quarto render` from
  `site/` — exactly matching the deploy workflow
  (`.github/workflows/site.yml`). Note: site rendering is not a PR CI
  gate (only a main-branch deploy step), so this local render **is** the
  gate and is mandatory per commit.
- **Link contract:** verification is bidirectional and covers **every
  internal link added or changed by this pass** — the §2 matrix rows,
  glossary outbound links, the getting-started replacement link, the two
  new sidebar entries, and the README architecture link. Two link
  classes: **fragment links** require (a) the rendered source contains an
  `href` resolving to the intended target page and fragment, and (b) the
  target HTML contains that exact `id`; **page-only links** (README →
  architecture, sidebar entries) require the resolved target HTML file to
  exist. Sources rendered by Quarto are checked in `_output/`; the README
  is **not** rendered there (Quarto renders only `site/*.qmd`), so its
  links are parsed from the Markdown source and their published site URLs
  mapped to local `_output/` paths. Implemented as a small
  HTML/Markdown-parser-based checker (e.g. a PEP 723 script using stdlib
  `html.parser`), not regex grep. Presence of sidebar URLs on unrelated
  pages proves nothing and is not used as evidence for §2 links.
- **Visual:** glossary and architecture pages checked in light and dark
  themes and at a narrow/mobile viewport; Mermaid diagram has a text
  summary.
- **Repo gate:** `uv run ruff format --check .`, `uv run ruff check .`,
  `uv run pytest` before **each commit**. (Codex round 1 suggested
  once-per-PR as proportionate; per-commit confirmed by Jonathan
  2026-08-10 — suite runs in ~12 s and this is standing repo policy.)

## Implementation shape

Single feature branch (`claude-diataxis-docs-pass`), one PR, four commits
in dependency order — each independently renderable with no dangling link
targets:

1. Glossary page + explicit method-page anchors (§1) — all link targets
   exist first.
2. Story/landing cross-links (§2).
3. Architecture page (§3).
4. README slimming + getting-started consistency fixes (§4).
