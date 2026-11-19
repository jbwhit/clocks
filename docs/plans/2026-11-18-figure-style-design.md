# Figure Style for Build-Time Plots: Design

**Date:** 2026-11-18

**Status:** Draft, in review

**Source:** findings V2 and V4 of the 2026-11-17 audit
(`docs/audits/2026-11-17/README.md`), and the visual reviewer's proposed system
in `docs/audits/2026-11-17/reviewers/visual.md`, section 3. Jonathan approved
that direction on 2026-11-18.

## Purpose

Every plot the site draws is default matplotlib: DejaVu Sans, the tab10 colour
cycle and a black frame on a white canvas, sitting on a cream page in the light
theme and a near-black page in the dark one. Labels authored for a wide canvas
land at a few pixels once the image is scaled into the column. No colour means
the same thing from one page to the next: the truth is coral on one page and
dashed blue on another.

This work gives the nine plots that are drawn when the site is built one shared
style that belongs to the page, in both themes, with colours that carry fixed
meanings.

It is the first of three figure tracks. The second replaces the committed
GIFs and PNGs with replay data rendered at build time and restyles them. The
third changes what the model-comparison page shows. Both build on the module
this track adds.

## Scope

In scope: the plot drawn by an executed cell on each of these nine pages.

| Page | Plot |
|---|---|
| `story/clocks-as-gravimeters.qmd` | tick rate against distance for three masses |
| `story/one-clock-is-not-enough.qmd` | two configurations across five clocks |
| `story/the-search-in-one-dimension.qmd` | posterior mean and band for x and M |
| `story/into-the-plane.qmd` | particle cloud after 25 observations |
| `story/two-hidden-masses.qmd` | two masses against one combined mass |
| `story/how-many-masses.qmd` | posterior over K by observation |
| `story/beyond-point-masses.qmd` | point mass against two Gaussian widths |
| `story/gravitational-echolocation.qmd` | differential signal against the noise floor |
| `method/the-particle-filter.qmd` | final ESS and tempering stages |

Out of scope, and why:

- The six GIFs, `demo_density.png` and `echolocation_range_study.png`. Their
  bytes are pinned by `tests/test_generated_assets.py`; they change in the
  second track, where that pin is replaced.
- What any plot shows. Data, scenarios, seeds and the claims in the prose stay
  as they are, with one exception listed under "Per-plot changes".
- Page typography, callouts, code blocks and tables (audit finding V4's CSS
  items). Those are a separate, CSS-only change.

## Design

### A style module in the library

A new private module `src/clocks/_style.py`, re-exported from `clocks.viz`:

```python
THEMES: tuple[str, str] = ("light", "dark")

@dataclass(frozen=True)
class Palette:
    background: str
    ink: str          # instruments and observed data; primary text
    muted: str        # axis labels, tick labels, secondary text
    rule: str         # axis lines
    grid: str
    truth: str        # the true value, wherever one is drawn
    belief: str       # the filter's estimate, posterior or cloud
    belief_alt: str   # a second mass's estimate; a rival hypothesis
    threshold: str    # noise floors, targets, limits; always dashed
    ordered: tuple[str, str, str, str]   # light to dark, for ordered series

def palette(theme: str) -> Palette: ...

@contextmanager
def style(theme: str) -> Iterator[Palette]: ...
```

`style(theme)` applies the theme's rcParams with `matplotlib.rc_context` and
yields that theme's `Palette`, so the rcParams are restored on exit. An unknown
theme raises `ValueError` naming the theme and the allowed values.

The library is the right home, not the site, because the demo entry points
draw figures too and will adopt the same module in the second track.

### Colour roles

Values are the visual reviewer's, which they validated for contrast and for
separation under colour-vision deficiency against each page background.

| Role | Light (`#f9f5d7` page) | Dark (`#1d2021` page) |
|---|---|---|
| background | `#f9f5d7` | `#1d2021` |
| ink | `#282828` | `#fbf1c7` |
| muted | `#665c54` | `#bdae93` |
| rule | `#d5c4a1` | `#504945` |
| grid | `#ebe2bd` | `#2b2f30` |
| truth | `#bf4a0a` | `#e0701a` |
| belief | `#1a6fa3` | `#4595cf` |
| belief_alt | `#a0306e` | `#d05c9c` |
| threshold | `#665c54` | `#bdae93` |
| ordered | `#7fb2d9`, `#3f86bd`, `#1a6fa3`, `#0d456b` | `#255f8c`, `#2f74aa`, `#4595cf`, `#8fc1e8` |

Rules for using them:

- **Truth is always `truth`**, drawn as a solid line or a ring-and-crosshair
  marker. No other series uses that hue.
- **What the filter believes is always `belief`**: lines, bands (the same hue
  at low alpha) and particle clouds.
- **Two rival hypotheses, neither of them the truth**, use `belief` and
  `belief_alt`.
- **An ordered family** (three masses, K = 1, 2, 3, two widths) uses `ordered`,
  never unrelated hues.
- **Thresholds are the only dashed lines.**
- **Particle weights are not colour-mapped.** After resampling they are close
  to uniform, so a colour map encodes almost nothing; density shows through
  overlap of one translucent hue.
- No matplotlib named colours, `tab:` colours or built-in colour maps appear in
  a page's plotting cell.

### Type and size

- Figures use Source Sans 3, the face the site's body text is set in. Two
  static TrueType files, Regular and Semibold, are committed under
  `src/clocks/_fonts/` with the font's OFL licence, shipped in the wheel, and
  registered with matplotlib's font manager when `clocks._style` is imported.
  The files come from the `adobe-fonts/source-sans` release on GitHub; the
  commit that adds them records the release tag and each file's SHA-256.
- Figures are authored at the size they are shown. The text column is about
  750 CSS pixels wide, so the default figure is 7.5 by 4.2 inches at 100 dots
  per inch, with 10.5 pt labels and 9.5 pt ticks. Quarto's default retina
  output supplies the second pixel per CSS pixel.
- `figure.constrained_layout.use` is on, so the tight bounding box the inline
  backend applies removes almost nothing and the emitted image stays at the
  authored width.
- No figure has a matplotlib title; the caption carries it. Series are labelled
  directly at the line where there is room, and a legend is used only where
  direct labels would collide. Legends have no frame.
- Top and right spines are off. Axis lines and grid are hairlines in `rule` and
  `grid`.

The full rcParams block is the one in `reviewers/visual.md` section 3, with
`axes.prop_cycle` set to `belief`, `truth`, `belief_alt` so that an unstyled
call cannot fall back to tab10.

### Two renderings per plot

Each plotting cell carries `#| renderings: [light, dark]` and draws its figure
once per theme, in that order:

```python
for theme in THEMES:
    with style(theme) as colors:
        fig, ax = plt.subplots()
        ...
        plt.show()
```

Quarto wraps the two outputs in `light-content` and `dark-content` and shows
the one matching the active theme. This was checked against the installed
Quarto 1.9.32 with a two-figure test cell: the page contained one output of
each class and a single caption.

Any computation a cell does (simulation, inference) stays outside the loop, so
it runs once.

### Per-plot changes

Each plot is restyled to the rules above. Beyond that:

- **`clocks-as-gravimeters`**: the three masses use `ordered`; a faint rule
  marks a rate of 1.
- **`one-clock-is-not-enough`**: the two configurations are rival hypotheses,
  `belief` and `belief_alt`.
- **`the-search-in-one-dimension`**: both estimates are `belief`; both truth
  lines are solid `truth`.
- **`into-the-plane`**: the cloud is translucent `belief`; the truth is a
  `truth` ring and crosshair; clocks are `ink` squares.
- **`two-hidden-masses`**: the two true masses are `truth`; the single combined
  mass is `belief`.
- **`how-many-masses`**: K = 1, 2, 3 use `ordered`; the K = 2 line is labelled
  as the truth.
- **`beyond-point-masses`**: the point mass is `ink`; the two Gaussians use
  `ordered` by width.
- **`gravitational-echolocation`**: the signal is `belief`; the noise floor is
  a dashed `threshold` line with the region below it shaded.
- **`the-particle-filter`**: the one content change. The ESS axis currently
  starts near 320, which puts the target line on the frame; it runs from 0 to
  the particle count instead. The stage count is drawn as a step with integer
  ticks.

Captions change only where they name a colour or line style that no longer
applies.

## Testing

Unit tests for the module (`tests/test_style.py`):

- `palette(theme)` returns a `Palette` for each theme, every colour a valid
  six-digit hex string, and `ordered` has four entries.
- Contrast against the theme's background, by the WCAG relative-luminance
  formula: at least 3:1 for `truth`, `belief`, `belief_alt`, `muted` and every
  `ordered` entry; at least 7:1 for `ink`.
- `style(theme)` sets the figure face colour and font family inside the block
  and restores the previous rcParams after it, including when the block raises.
- An unknown theme raises `ValueError`.
- After import, matplotlib resolves "Source Sans 3" at weights 400 and 600 to
  the bundled files, with `fallback_to_default=False`.

A source contract for the pages (`tests/test_site_figure_style.py`), one case
per `.qmd` file under `site/` that calls `plt.subplots`:

- the plotting cell declares `renderings: [light, dark]` and uses `style(`;
- it contains none of: `tab:`, `cmap=`, or a matplotlib named colour from a
  short list of those in use today (`steelblue`, `lightcoral`, `red`, `black`).

These are string checks, as the repository's other page contracts are. Their
purpose is to stop a new or edited plot from bypassing the style, not to prove
a plot looks right.

Each new test is mutation-checked against the whole suite before the work is
called done: for example, changing one palette entry to fail contrast, removing
`renderings` from one cell, and reintroducing `tab:blue` in one cell must each
turn exactly one test red.

Checks on the rendered site, run locally because CI does not render it:

- every one of the nine pages contains one `light-content` and one
  `dark-content` image for its plot;
- at a 390-pixel viewport no page is wider than the viewport and no plot is
  wider than its container (the iframe harness from PR #21);
- a contact sheet of all nine plots in both themes is produced and looked at,
  and attached to the pull request.

## Risks

- **Font files in the wheel.** Two static files add roughly 0.6 MB. If that is
  unwelcome the fallback is to register the font only when the files are
  present and let matplotlib fall back otherwise, at the cost of the site's CI
  build needing them anyway. The spec chooses bundling.
- **`renderings` depends on the Quarto version.** The deploy workflow installs
  Quarto's current release, which is newer than the feature. An older local
  Quarto would show both figures stacked; the Getting Started page's Quarto
  requirement should state the minimum version.
- **Doubling the plots** adds drawing time only. Simulation and inference stay
  outside the theme loop.
- **Committed assets will not match** until the second track lands: for a
  while, build-time plots will be in the new style and the GIFs in the old.
  That is accepted; the alternative is one very large change.

## Not doing

- A general theming API. There are two themes and one palette.
- Colour-vision simulation in tests. The reviewer validated the palette once;
  the tests guard contrast, which is what a later edit is most likely to break.
- Restyling anything outside the nine cells.
