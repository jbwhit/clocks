# Diataxis-Guided Documentation Pass — Design

**Date:** 2026-08-10
**Status:** Draft — pending Codex xhigh review
**Review:** Codex xhigh verdict to be recorded here (no PR yet at spec stage)

## Goal

Improve the "GPS in Reverse" site for its marginal reader — the
physics-curious visitor — using Diataxis as an audit lens (not a navigation
scheme), plus one C4-Level-3-style component diagram for the secondary
contributor audience.

## Background / audit findings

The site is already implicitly Diataxis-shaped:

- **Part 1 — The Story** (8 pages) is understanding-oriented explanation and
  is the site's identity. It must not be restructured.
- **Part 2 — Under the Hood** is explanation with reference elements
  (`the-particle-filter.qmd` maps prose onto `clocks.inference.ParticleFilter`;
  `units-and-scales.qmd` is reference material).
- **Part 3 — Reproduce** covers the tutorial quadrant (`getting-started.qmd`)
  plus reproducibility notes.

For the physics-curious reader, the missing "reference" quadrant is not API
docs — it is the material a reader flips back to mid-read: notation, symbols,
and terms of art. API reference (quartodoc) and how-to guides serve library
users and are explicitly **out of scope** for this pass.

## Changes

### 1. Story → method cross-links

On **first mention in each story page**, link terms to the matching section
in Part 2:

- *particle filter*, *resampling*, *effective sample size / ESS*, *jitter*
  → anchors in `method/the-particle-filter.qmd`
- *simulation units*, G = c = 1, scales → `method/units-and-scales.qmd`
- Lighter terms of art (posterior, prior, weak field, likelihood, model
  comparison) → entries on the new notation/glossary page (see §2)

Rules:

- One link per term per page; no link farms.
- Quarto auto-generates heading IDs; add explicit `{#sec-...}` anchors only
  where a needed target has no heading.
- Do not alter the prose voice — links are added to existing wording; text
  edits only where a term must be introduced to be linkable.

### 2. Notation & Glossary page

New file: `site/method/notation-and-glossary.qmd`, added to the Part 2
sidebar (after Units and Scales).

Content (target: one screen, roughly 60–90 lines):

- **Symbols table:** Φ (potential), σ (observation noise), N (particles),
  ESS, w (weights), θ (parameter hypothesis), K (number of masses),
  M (mass), r (tick rate). Columns: symbol, meaning, where introduced
  (link).
- **Terms list:** weak field, time dilation, posterior, prior, likelihood,
  resampling (systematic/stratified/residual), jitter (incl. annealed),
  effective sample size, model comparison, degeneracy. One-line definitions,
  each linking to its fuller treatment in a story or method page.

This page is the glossary link target for story pages (§1).

### 3. Architecture page

New file: `site/reproduce/architecture.qmd`, added to the Part 3 sidebar
(after Reproducibility). Contributor-facing; lives in "Reproduce" — no new
Part 4.

Content:

- One Mermaid component diagram (C4 Level 3 in spirit; no C4 tooling or
  notation ceremony) of `src/clocks`, drawn from the **actual import
  graph** (verified 2026-08-10):
  - `types` is the foundation (imported by nearly everything).
  - `physics`, `noise` → `types` (noise does not import physics).
  - `inference` → `noise`, `physics`, `types`.
  - `config` → `types`; `results` → `config`, `types`.
  - `api` → `config`, `inference`, `noise`, `physics`, `results`, `types`.
  - `viz` is a pure facade re-exporting `_animate`, `_panels`, `_panels3d`;
    `_animate` → `_panels`, `_panels3d`, `inference`, `physics`, `types`.
  - `_scenarios` → `api`, `config`, `inference`, `physics`, `results`,
    `types` (shared by demos, scan harnesses, tests);
    `_echo_study` → `_scenarios`, `physics`.
  - `_cli` runs `scripts/*.py` via `runpy` (no library imports) — entry
    points for `uv run demo-*`.
- Short prose walk of the public/private boundary: public surface is
  `clocks.__init__` (api + config + results + types + viz + selected
  physics/noise/inference names); underscore modules are internal.
- Diagram must render in both light and dark site themes (Quarto handles
  Mermaid theming; verify visually in the rendered output).

### 4. README slimming

- Replace the "Use as a library" API walkthrough (both code blocks) with a
  2–3 sentence summary plus a link to the site.
- Compress the per-demo catalog (six GIF embeds + descriptions) to the
  command list with one representative GIF, linking to the site for the
  rest.
- "Project structure" section: keep the top-level listing but point to the
  new architecture page for the dependency picture.
- Keep: intro, setup, run-tests sections unchanged.

## Out of scope

- Sidebar/nav restructure into Diataxis quadrant names.
- quartodoc / generated API reference.
- How-to guides for library consumers.
- C4 Levels 1, 2, 4.
- Any change to Python source or tests.

## Verification

- `quarto render` (from `site/`) completes clean; index.qmd executes real
  Python, so the build is a genuine gate.
- Grep rendered `_output/` for links to the two new pages; confirm no
  broken relative links (spot-check anchors added in §1).
- Visual check of the Mermaid diagram in light and dark themes.
- Standard repo gate before each commit: `uv run ruff format --check .`,
  `uv run ruff check .`, `uv run pytest` (docs-only change; suite must
  stay green regardless).

## Implementation shape

Single feature branch (`claude-diataxis-docs-pass`), one PR, roughly four
commits mirroring §§1–4. Each commit independently renderable.
