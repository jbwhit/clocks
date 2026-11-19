# Figure Style for Build-Time Plots: Design

**Date:** 2026-11-18

**Status:** Approved for implementation. Codex xhigh round 3, on `55af882`:
"SOUND ENOUGH TO IMPLEMENT", with four small findings that this revision folds
in.

**Source:** findings V2 and V4 of the 2026-11-17 audit
(`docs/audits/2026-11-17/README.md`), and the visual reviewer's proposed system
in `docs/audits/2026-11-17/reviewers/visual.md`, section 3. Jonathan approved
that direction on 2026-11-18. Everything an implementer needs is restated
here; the audit is background, not a second source of truth.

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

In scope: the matplotlib plot drawn by an executed cell on each of these nine
pages. These are all nine matplotlib cells on the site.

| Page | Plot |
|---|---|
| `story/clocks-as-gravimeters.qmd` | tick rate against distance for three masses |
| `story/one-clock-is-not-enough.qmd` | two configurations across five clocks |
| `story/the-search-in-one-dimension.qmd` | posterior mean and band for x and M |
| `story/into-the-plane.qmd` | particle cloud after 25 observations |
| `story/two-hidden-masses.qmd` | two masses against one combined mass |
| `story/how-many-masses.qmd` | posterior over K by observation |
| `story/beyond-point-masses.qmd` | point mass against two Gaussian widths |
| `story/gravitational-echolocation.qmd` | differential signal against the noise level |
| `method/the-particle-filter.qmd` | final ESS and tempering stages |

Out of scope, and why:

- The six GIFs, `demo_density.png` and `echolocation_range_study.png`. Their
  bytes are pinned by `tests/test_generated_assets.py`; they change in the
  second track, where that pin is replaced.
- Data, scenarios, seeds, printed output and prose. In particular the text a
  cell prints (the model-count cell prints its posterior and best model before
  its figure) and the sentences pinned by `tests/test_generated_assets.py` stay
  exactly as they are. What the plots show changes only as listed under
  "Declared changes to what a plot shows".
- Page typography, callouts, code blocks, tables, and switching the site's web
  font. Those are CSS-only and separate.

## Design

### A style module in the library

A new private module `src/clocks/_style.py`, re-exported from `clocks.viz`:

```python
THEMES: tuple[str, str] = ("light", "dark")

@dataclass(frozen=True)
class Palette:
    background: str
    ink: str          # instruments, observed data, neutral curves; primary text
    muted: str        # axis labels, tick labels, secondary text
    rule: str         # axis lines
    grid: str
    truth: str        # the true configuration, or what it predicts
    belief: str       # an estimate or a candidate: the filter's posterior, or
                      # a configuration proposed as an explanation of the data
    belief_alt: str   # the second of two candidates
    threshold: str    # noise levels, targets, limits; always dashed
    ordered: tuple[str, str, str]   # an ordered family, least to most contrast

def palette(theme: str) -> Palette: ...

@contextmanager
def style(theme: str) -> Iterator[Palette]: ...

def show(fig: Figure) -> None: ...
```

- `palette(theme)` returns the theme's colours. `style(theme)` applies the
  theme's rcParams with `matplotlib.rc_context` and yields the same `Palette`;
  the previous rcParams are restored on exit, including when the block raises.
- Both raise `ValueError` for an unknown theme, and the message names the
  value given and the two allowed values.
- The bundled font is registered with matplotlib's font manager the first time
  `style()` is entered, not at import, so importing `clocks.viz` does not
  change the process's font registry for existing callers. Registration is
  idempotent. Nothing in the module selects a matplotlib backend.
- `show(fig)` renders the figure to PNG at twice its logical dpi (the figure's
  dpi divided by the canvas's device pixel ratio, which is 1 except under
  interactive HiDPI backends), closes it, and displays it with
  `IPython.display.Image(..., retina=True)`. The save is made
  inside `rc_context({"savefig.bbox": None})` with `bbox_inches=None`,
  `transparent=False` and the figure's own face colour, so an inherited
  `savefig.bbox: tight` cannot crop it. IPython is imported inside the
  function, because it is present only where notebooks run. One call makes
  exactly one display output. `plt.show()` is not used in page cells, because
  it displays every open figure and crops to a tight bounding box.

The library is the right home, not the site, because the demo entry points
draw figures too and will adopt the same module in the second track.

### Colour roles

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
| ordered | `#3f86bd`, `#1a6fa3`, `#0d456b` | `#2f74aa`, `#4595cf`, `#8fc1e8` |

These are the visual reviewer's values, with one correction from review. Their
ramp had four entries and its faintest one measured 2.06:1 (light) and 2.42:1
(dark) against the page, below the 3:1 this spec requires of data strokes. No
plot needs more than three ordered series, so the faintest entry is dropped.
Measured WCAG contrast of each remaining swatch against its page:

| Role | Light | Dark |
|---|---:|---:|
| ink | 13.39 | 14.45 |
| muted, threshold | 5.92 | 7.53 |
| truth | 4.54 | 5.08 |
| belief | 4.96 | 5.04 |
| belief_alt | 6.07 | 4.44 |
| ordered | 3.56, 4.96, 9.17 | 3.28, 5.04, 8.56 |
| rule | 1.56 | 1.86 |
| grid | 1.18 | 1.21 |

`rule` and `grid` are furniture, not data, and are meant to be faint; the 3:1
requirement does not apply to them.

`ordered` runs from least to most contrast against the page in both themes, so
index 0 is always the quietest line. A three-member family uses indices 0, 1,
2 in the family's natural order (smallest mass, lowest K, narrowest width
first). A two-member family uses indices 0 and 2.

Rules:

- **Every data artist is given an explicit role colour.** The default colour
  cycle is the single colour `ink`, so a series that is left unstyled comes out
  neutral instead of borrowing a role.
- **`truth` is used only for the true configuration**: its value as a solid
  line, its position as a ring-and-crosshair marker, or the tick rates it
  predicts.
- **`belief` is an estimate or a candidate**: the filter's posterior lines,
  bands (the same hue at low alpha) and particle clouds, and equally a
  configuration put forward as a possible explanation of the data. Where two
  candidates are compared and neither is the truth, the second is
  `belief_alt`.
- **Filter diagnostics** (effective sample size, stage counts) and
  forward-model curves that are neither a candidate nor a truth are `ink`.
- **Thresholds are the only dashed lines**, in `threshold`.
- **Hue is never the only distinction.** `truth`, `belief` and `belief_alt`
  have similar luminance, so wherever two of them share an axes the series are
  also told apart by a direct text label or by different markers shown in a
  legend. The per-plot table says which. Members of an `ordered` family differ
  strongly in luminance, so a legend or end labels are enough for them.
- No matplotlib named colour, `tab:` colour or colour map appears in a page's
  plotting cell.

### Type and size

- Figures use Source Sans 3. The site's body text requests "Source Sans Pro",
  the earlier name of the same family, so the two match closely; aligning the
  web font is not part of this work. Bundling guarantees the family is
  available wherever the library runs, including the site's build runner. It
  does not guarantee which copy is used: on a machine that already has Source
  Sans 3 installed, matplotlib may pick the installed copy, which is the same
  design. Two static TrueType files, Regular
  (weight 400) and Semibold (weight 600), from the `3.052R` release of
  `adobe-fonts/source-sans`, are committed under `src/clocks/_fonts/` with the
  font's licence text beside them. The commit that adds them records each
  file's SHA-256.
- The fonts are under the SIL Open Font License 1.1 with a reserved font name;
  the code stays MIT. `pyproject.toml` changes `license` to
  `MIT AND OFL-1.1-RFN` and adds the font licence file to `license-files`. The
  built wheel is inspected to confirm it contains both font files and both
  licence files.
- Figures are authored at the size they are shown, and that size is a
  requirement, not a target. The text column is 750 CSS pixels wide. The
  default figure is 7.5 by 4.2 inches at 100 dots per inch; `show()` emits it
  as a 1500 by 840 pixel PNG declared as 750 by 420 CSS pixels. Labels are
  10.5 pt and ticks 9.5 pt, about 14 and 13 CSS pixels at full width. This was
  checked with Quarto 1.9.32: an image displayed this way is emitted with
  `width="750" height="420"`. The width is fixed for every plot. A plot that
  needs a different height (two stacked panels, the equal-aspect map) sets its
  figure height explicitly, and the rendered-site check records that page's
  expected height.
- On a 390-pixel phone the existing rule in `site/styles.css` scales the image
  to the column, which brings labels to about 7 CSS pixels. That is a known
  limit of one raster per theme; it is inspected, not fixed, here.
- Layout is `constrained`. No figure has a matplotlib title; the caption
  carries it. Top and right spines are off; axis lines and grid are hairlines.

The rcParams, complete:

```python
BASE = {
    "font.family": "Source Sans 3",
    "font.size": 10.5,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.linewidth": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.8,
    "lines.solid_capstyle": "round",
    "figure.figsize": (7.5, 4.2),
    "figure.dpi": 100,
    "figure.autolayout": False,  # it would take priority over constrained
    "figure.constrained_layout.use": True,
}
```

and per theme, from the `Palette`: `figure.facecolor`, `axes.facecolor` and
`savefig.facecolor` are `background`; `text.color` is `ink`;
`axes.labelcolor`, `xtick.color` and `ytick.color` are `muted`;
`axes.edgecolor` is `rule`; `grid.color` is `grid`; `axes.prop_cycle` is the
single colour `ink`.

### Two renderings per plot

Each plotting cell carries `#| renderings: [light, dark]`, draws its figure
once per theme in that order, and gives the caption once per rendering:

```python
#| renderings: [light, dark]
#| fig-cap:
#|   - "The caption."
#|   - "The caption."
for theme in THEMES:
    with style(theme) as colors:
        fig, ax = plt.subplots()
        ...
        show(fig)
```

Quarto wraps the two outputs in `light-content` and `dark-content` and shows
the one matching the active theme. The caption must
be a two-entry list with identical entries. Tested with Quarto 1.9.32: a
single `fig-cap` string is attached to the first output only, so it sits inside
`light-content` and disappears in the dark theme; a two-entry list puts one
caption inside each rendering. (A fenced figure div also works but numbers the
figure, which the site does not do.) Printed output stays outside both
wrappers and is unaffected.

Any computation a cell does (simulation, inference) and anything it prints
stay outside the loop, so they happen once.

`renderings` arrived in Quarto 1.7, but the caption-list behaviour this design
relies on was verified only on 1.9.32, so that is the stated minimum: the
Getting Started page's prerequisites gain a line saying the site is rendered
with Quarto 1.9.32 or later. The deploy workflow installs Quarto's current
release, which is newer; the rendered-site check below runs there on every
deployment and fails if a later Quarto changes the behaviour.

### Per-plot colour assignments

| Page | Series | Colour and mark | Told apart by |
|---|---|---|---|
| `clocks-as-gravimeters` | three masses, smallest first | `ordered[0..2]` lines | a label at each line's right end |
| `one-clock-is-not-enough` | near-light and far-heavy configurations, two candidates | `belief` line with circles, `belief_alt` line with squares, joined as now | legend showing the markers |
| `the-search-in-one-dimension` | posterior mean and band, both panels | `belief` | |
| | true value, both panels | solid `truth` line | a "truth" label at the line's right end in each panel |
| `into-the-plane` | particle cloud | translucent `belief` dots | legend showing the three markers |
| | true position | `truth` ring and crosshair | |
| | clocks | `ink` squares | |
| `two-hidden-masses` | tick rates predicted by the true pair: dense curve, circles at the clocks | `truth` | legend showing the markers |
| | tick rates predicted by the single combined mass, a candidate: dense curve, squares at the clocks | `belief` | |
| `how-many-masses` | K = 1, 2, 3 | `ordered[0..2]` lines | a label at each line's right end; K = 2 reads "K = 2 (true model)" |
| `beyond-point-masses` | point mass | `ink` line | a label at the point mass's minimum; legend for the two Gaussians |
| | Gaussians, narrow then wide | `ordered[0]`, `ordered[2]` lines | |
| `gravitational-echolocation` | differential signal, a forward-model curve | `ink` line | a label on each line |
| | noise of one observation | dashed `threshold` line | |
| `the-particle-filter` | final ESS | `ink` line with points | axis label |
| | ESS target | dashed `threshold` line | a label on the line |
| | tempering stages | `ink` step | axis label |

K = 2 is the model that generated the data; its curve is a posterior
probability, not a true value, so it takes its place in the ordered family and
is identified by its label, not by the `truth` colour.

### Declared changes to what a plot shows

This is a style change. These are the places it alters content, and each is
deliberate:

1. **`into-the-plane` stops colouring particles by weight.** Equal-opacity dots
   show where particles are, which is what the plot is for. The caption
   changes from "colored by weight" to describe the surviving hypotheses'
   positions.
2. **`clocks-as-gravimeters` gains a reference line at a rate of 1**, the
   far-away value every curve approaches. It is a solid line in `rule`, since
   dashes are reserved for thresholds. The upper y-limit is set to 1 plus 5% of
   the plotted range, where the range is the largest minus the smallest rate
   across the three curves, so the line sits visibly inside the frame; the lower
   limit stays where matplotlib puts it for the curves alone.
3. **`gravitational-echolocation` shades the region below the per-observation
   noise level**, in `threshold` at 12% opacity. The axis is logarithmic: the
   y-limits the plot has before the shading is added are recorded and restored
   after it, and the shading runs from the lower limit to the noise level. The
   caption describes it as below the noise of a single observation, not as a
   detection limit.
4. **`the-particle-filter` runs its ESS axis from 0 to the particle count.** It
   starts near 320 today, which puts the target line on the frame. The stage
   count keeps its existing step drawing and gains integer ticks.

Captions otherwise change only where they name a colour, line style or marker
that no longer applies. For example the two-mass caption's "dots" becomes "the
five marked clock positions", which holds for circles and squares alike.

## Testing

### Unit tests (`tests/test_style.py`)

- Each theme's palette: every colour is a six-digit hex string; `ordered` has
  three entries; the two themes differ in every role; within a theme `truth`,
  `belief` and `belief_alt` are three different colours.
- WCAG contrast of each opaque swatch against its theme's background: at least
  3:1 for `truth`, `belief`, `belief_alt`, `muted`, `threshold` and each
  `ordered` entry; at least 7:1 for `ink`; and `ordered` strictly increasing in
  contrast. This checks swatches, not rendered marks: alpha and line width
  change what is seen.
- `style(theme)`: starting from deliberately changed rcParams, inside the block
  `figure.facecolor`, `axes.edgecolor`, `text.color`, `font.family` and the
  colour cycle all hold the theme's values; after the block every one is back
  to its starting value, both when the block completes and when it raises.
- `palette()` and `style()` each raise `ValueError` for an unknown theme, and
  the message contains the bad value and both allowed values.
- After entering `style()` once, the font manager's list contains both bundled
  files, identified by their paths under the package's `_fonts` directory, as
  family "Source Sans 3" at weights 400 and 600; and a lookup of that family at
  each weight with `fallback_to_default=False` succeeds. The path check is what
  makes this fail if the files are not shipped or not registered, even on a
  machine that has the family installed. Entering `style()` a second time adds
  no further entries.
- `show(fig)` on a default figure, called with `savefig.bbox` set to `"tight"`
  in the surrounding rcParams, produces one display call whose PNG is exactly
  1500 by 840 pixels, opaque, with declared size 750 by 420, and closes the
  figure. IPython's `display` is replaced by a recorder for the test.

### Page source contract (`tests/test_site_figure_style.py`)

The nine pages are listed explicitly. The test first asserts that the set of
`.qmd` files under `site/` whose Python cells call `plt.subplots` or
`plt.figure` is exactly those nine, so the test cannot pass by collecting
nothing, and a new page that plots through either call is caught. A page that
reached matplotlib some other way, for example through an alias, would not be
caught by this test or by the rendered-site check, which visits the same nine
pages. Then, per page, looking only inside executable Python cells:

- the plotting cell has `renderings: [light, dark]`, a two-entry `fig-cap`
  whose entries are identical, a call to `style(` and a call to `show(`;
- it contains no `plt.show(`, no `cmap` keyword (whitespace-tolerant), and no
  quoted matplotlib colour name: any `tab:` colour, or `steelblue`,
  `lightcoral`, `red`, `black`, `gray` or `grey` as a quoted string.

These are string checks, as the repository's other page contracts are. They
stop a new or edited plot from bypassing the conventions; they do not prove a
plot looks right or that the style block encloses the drawing.

### Rendered-site check (`scripts/check_site_figures.py`)

A script in the manner of `scripts/check_site_links.py`, run on `site/_output`
after rendering. For each of the nine pages it parses the HTML and asserts:

- exactly one plot image inside a `light-content` wrapper and exactly one
  inside a `dark-content` wrapper, each declared 750 by 420 pixels unless the
  page's entry in the script names a different height, and the two image
  files differ;
- exactly one `<figcaption>` inside each wrapper, with the same text in both.

The deploy workflow runs this script and `scripts/check_site_links.py` after
`quarto render` and before uploading, so a page that loses a rendering or a
caption fails the deployment instead of reaching the site. Until now neither
check ran anywhere automatically.

### Mutation checks

Each mutation is applied in turn with bytecode caching off and the tree
confirmed clean between mutants, and the whole suite is run. The plan's
mutation task names, for each mutation, the exact test IDs expected to fail;
the result must be exactly that set, no more and no fewer. Several mutations
legitimately fail more than one test, because the same behaviour is checked
per theme or on both exit paths; the plan's table is the authority on the
expected sets.

- return the light palette for both themes;
- lighten `ordered[0]` in one theme to fail 3:1;
- make `style()` skip restoring rcParams;
- remove one allowed value from the unknown-theme message;
- make `show()` pass `bbox_inches="tight"`;
- remove `renderings` from one cell; make one cell's two captions differ;
  reintroduce `"tab:blue"` in one cell; add `cmap = "viridis"` in one cell.

### Checks by eye

- The phone harness from PR #21 (a 390-pixel iframe): no page wider than the
  viewport, no plot wider than its container.
- A contact sheet of all nine plots in both themes at full width and at 390
  pixels is produced, looked at against each cell's source for swapped series
  or wrong units, and attached to the pull request.

## Risks

- **Font files in the wheel** add about 0.9 MB (431 kB and 426 kB) and a
  second licence. The
  alternative, registering the font only if present, leaves the site's CI
  build needing the files anyway.
- **The two-rendering behaviour depends on the Quarto version.** An older local
  Quarto would show both figures stacked or drop a caption. The tested minimum
  is documented, and the rendered-site check fails on a build that gets it
  wrong.
- **Duplicated caption text** in each cell can drift. The source contract
  requires the two entries to be identical.
- **Committed assets will not match** until the second track lands: for a
  while, build-time plots will be in the new style and the GIFs in the old.
  That is accepted; the alternative is one very large change.

## Not doing

- A general theming API. There are two themes and one palette.
- Colour-vision simulation in tests. The reviewer validated hues once; the
  rule that hue is never the only distinction covers the residual risk, and
  the tests guard contrast, which a later edit is most likely to break.
- A phone-specific rendering of each plot.
- Restyling anything outside the nine cells.
