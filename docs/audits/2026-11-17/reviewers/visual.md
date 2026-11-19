# Visual review of "GPS in Reverse": figures, animations and page

I reviewed all 15 pages at 1440 px and 390 px in both themes, all six GIFs frame by frame, and both PNGs and every build-time plot. The repo is untouched (`git status` clean) and everything I made is in scratch. One apparatus caveat: 4 of about 40 of my headless page loads rendered unstyled; 12 fresh reloads did not reproduce it, so I believe it was my capture setup and have not counted it as a site problem.

## 1. Diagnosis

**The three changes that matter most**

1. **Rebuild the animated dashboards around what changes.** In every GIF, frame 0 is already the state after observation 1, so the prior is never shown and the collapse happens off-screen. In `demo_2d.gif` 6.3% of pixels change in the first transition and about 0.5% per frame after frame 10, so most of the 12.5 s is dead. The remaining motion is mostly noise (clock colours flickering) and apparatus (axes rescaling every frame).
2. **Give the figures one style that belongs to the page.** Every plot is default matplotlib (DejaVu Sans, tab10, black box) on a white canvas over a cream or near-black page. Text is authored for 12–15 in canvases and shown at 749 px, so 6–8 pt labels land at about 3.5–5 px on desktop and about 2 px on mobile. No colour has a fixed meaning: "truth" is red, coral, dashed, or blue depending on the page.
3. **Fix the page's broken and default parts.** Build-time plots are clipped on phones, two pages scroll sideways, the measure is 95–102 characters, and callouts, inline code, code blocks and dark-mode captions are stock Bootstrap colours.

**Keep**

- Fraunces headings with the hairline `h2` rule, and the gruvbox ground. This is the one part that reads as designed.
- The sidebar structure, the orange-ruled cell output, and the transparent-cluster Mermaid fix.
- The simple build-time line plots as *forms* (two-masses versus one, point versus Gaussian, signal versus noise floor). They only need the style.
- The range study's choice to show every run as a dot beside the median.

**Prototype of the direction** (real filter, committed `demo_2d` config, both themes):

- Contact sheet: `../prototype/sheet_proto.png`
- Video, light (14.8 s, 0.38 MB against the 1.5 MB GIF): `../prototype/proto_demo2d_light.mp4`
- Video, dark: `../prototype/proto_demo2d_dark.mp4`
- Script: `../prototype/proto_demo2d.py`

**For the technical reviewers.** Giving each parameter its own axis exposed something the current figure hides. In `demo_2d` the true mass (0.15) sits exactly on the prior's upper bound (`demo_2d.py:23,75`). After 50 observations M is 0.149 ± 0.001, pinned at the ceiling, and y is −1.15 ± 0.08 against a truth of −1.0, about 2 s.d. off. A sharper figure will show this to every visitor, so decide whether the demo should change.

## 2. Figure by figure

All paths are under ``. Contact sheets of each GIF are in `scratchpad/visual/frames/` as `<name>__sheet.png`.

**Shared by all four 2×2 dashboards** (`src/clocks/_animate.py:84-112`, `src/clocks/_panels.py:258-273`)

- **No prior frame.** `pf.history[0]` already holds the prior. Prepend it in `_precompute_filter_states` (`_animate.py:61-81`).
- **Uniform pacing, no end hold, infinite loop.** `repeat=False` (`_animate.py:111`) is ignored by the Pillow writer; the files have `loop=0` and every frame is 250 ms.
- **Convergence axis rescales every frame.** Frame 1 shows a degenerate 0.96–1.04 axis, and ticks read "1.25, 1.5" on an integer count. Fix the x-limits at 0..N.
- **Mixed units on one axis.** Position (±8) and mass (about 0.1) share a y-axis in `_plot_convergence` (`_panels.py:337-365`), so M is a flat line with no visible uncertainty. Use one small panel per parameter.
- **The legend sits where the trace ends** (`loc="upper right"`, `_panels.py:361`). Label directly.
- **Weight colouring is noise.** After resampling the weights are near-uniform, so viridis (`_panels.py:87,176,288,319`) encodes almost nothing, and its yellow end vanishes on white. Use one hue with alpha.
- **Three panels show the same x–y plane** in the 2D demos (setup, cloud, observed rates). Merge them into one map.

**`demo_1d.gif`** — The Search in One Dimension

- The mass axis is 0–2 while the prior tops out at 0.15 (`mlim` default, `_animate.py:171,193`), so the cloud is a sliver on the floor of its panel. Use the prior range.
- The rate bars run from zero (`_panels.py:129`), so 0.96 and 0.99 look identical. The true and observed value labels are printed on top of each other (`_panels.py:118-126`).
- Fix: plot the deficit (1 − rate) as dots with ±σ bars, the true curve, and a posterior-predictive band.
- "Physical Setup" spends a quarter of the canvas on three static squares. Put the clocks on the x-axis of the particle panel.
- Screenshot: `scratchpad/visual/pages/light/desktop/the-search-in-one-dimension__fig0.png`

**`demo_2d.gif`** — landing page and Into the Plane

- The most salient motion is the clock squares changing colour each frame. That is per-observation noise drawn on a diverging `coolwarm` scale with no meaningful midpoint (`_animate.py:115-160`).
- Fix: colour the clocks by the running-mean deficit on a single-hue ramp, so the pattern sharpens instead of flickering.
- The colourbar steals width, so the bottom-left panel is a different size from the top-left.
- By observation 10 the cloud is a 6 px speck. Add a fixed-window zoom inset and a 2 s.d. ellipse.
- The same asset is both the landing hero and the page-4 figure.
- Screenshot: `scratchpad/visual/pages/light/desktop/index__fig0.png`

**`demo_multi_mass.gif`** — Two Hidden Masses

- The x₁-versus-x₂ scatter (`_panels.py:276-303`) is an abstract parameter plane that needs explaining. Draw both clouds on the physical line in two hues over the clocks instead.
- The M₁ and M₂ traces coincide at about 0 on a ±8 axis. The legend is 7 pt in two columns, about 4 px as displayed.
- The bar-chart problems are the same as in `demo_1d`.

**`demo_multi_mass_2d.gif`** — Two Hidden Masses (4.9 MB)

- This is the one animation with real drama: slow convergence and a visible second mode around observations 37–57.
- Only mass 1's cloud is drawn (`_panels.py:306-334`), so the label-switching story is told by a six-line trace panel with overlapping bands and a 6 pt, three-column legend (`_animate.py:420`).
- Fix: one map with both clouds (blue and magenta) and both truths, plus six single-series small multiples. This one deserves a scrubber.

**`demo_model_comparison.gif`** — How Many Masses? (`_animate.py:436-507`)

- Bars are the wrong form for a time series. For 20 s, two bars jiggle between 0.3 and 0.7. The build-time line chart further down the same page is the better figure.
- Truth (K=2) is unmarked, and green on K=3 reads as "correct".
- The left panel is the same uninformative zero-based bars.
- Fix: replace it with a growing line chart, K=2 labelled "truth", with the final values as direct labels. Better still, retire the animation in favour of a restyled static figure.

**`demo_echolocation_3d.gif`** — Gravitational Echolocation (6.4 MB)

- **Layout jitter.** `constrained_layout=True` (`_panels3d.py:25`) re-solves every frame; the right column's left spine moves between x=949 and x=962 px across frames (measured). Use fixed `add_axes` rectangles.
- **Choppy orbit.** The camera turns 4.5° per frame at 4 fps (`_animate.py:582`). Render at 24–30 fps, with a slow orbit of about 60° and a hold.
- **The content is tiny.** The head and cloud fill about 15% of the 3D box (`lim`, `_panels3d.py:91`). The 250 pt star covers the collapsed cloud (`_panels3d.py:73`), the clocks are 15 pt, and the panes are grey.
- **Other axes jump too.** The mass histogram's axes rescale every frame (`_panels.py:194-217`), and the "best"-placed legend in `plot_centered_rates` (`_panels3d.py:128`) hops between corners.
- Fix: tighten the limits to the mass range, remove the panes, add drop-lines from the mass and cloud to the floor for depth, fix the histogram range to the prior, and replace the 27 bars with an observed-versus-predicted scatter.

**`demo_density.png`** — Beyond Point Masses (`src/clocks/_demos/demo_density.py:115-189`)

- A 15×4 in figure shown at 749 px gives about 7 px text on desktop and 3 px on mobile.
- Coral means "True" and steel blue means "Inferred" here, the reverse of other pages.
- μ, σ and A share one axis, so A is flat at zero.
- Fix: make it 7.5 in wide, with the density panel on top (posterior band from particle draws, not only the mean curve) and three small traces below.

**`echolocation_range_study.png`** (`src/clocks/_echo_study.py:120-184`)

- It is the only figure with a baked-in title, and both panels repeat the same legend.
- Run dots and the median line share one hue.
- Fix: grey the dots, label the two lines directly once, and draw the frozen pass tolerance as a rule so the 12, 12, 11, 8, 3, 0 pass counts are visible. Annotate those counts along the top.
- Re-render at the column width; the file is 1200 px for a 749 px slot.

**Build-time plots** (all default style, 597 px wide, centred in a 749 px column)

- **`story/clocks-as-gravimeters.qmd:50-74`**: three masses are an ordered quantity drawn in blue, orange and green. Use one hue light to dark, label each line at its right end, and add a faint rule at 1.
- **`story/one-clock-is-not-enough.qmd:40-61`**: good form. Add a ±σ noise band so the "noise floor" callout is visible in the figure, and label directly.
- **`story/the-search-in-one-dimension.qmd:33-74`**: x and M share an axis; split into two panels. The legend floats mid-plot.
- **`story/into-the-plane.qmd:33-75`**: limits are −4..5 against −8..8 in the GIF above it, the clocks are unlabelled, and it uses viridis. Use the shared map primitive.
- **`story/two-hidden-masses.qmd:45-76`**: the best static figure on the site. Here coral is truth and blue the alternative, the opposite of the page before it. Shade the gap between the curves at the clocks.
- **`story/how-many-masses.qmd:47-85`**: K is ordinal. Use a blue ramp or direct labels, and mark K=2 as truth.
- **`story/beyond-point-masses.qmd:22-47`**: fine. Order the colours by σ.
- **`story/gravitational-echolocation.qmd:35-66`**: strong idea. Shade the region below the noise floor, mark the crossing near R≈7, and replace "max orthonormal contrast" with a plain label.
- **`method/the-particle-filter.qmd:107-151`**: the ESS axis starts at 320, so the target line lies on the frame. Run the axis from 0 to N, and draw the stages as a stem or step with integer ticks.
- **Coda table** (`gravitational-echolocation.qmd:139-203`): the numbers are left-aligned in mixed notation (`1.1e-17` beside 10⁹). Right-align them with tabular figures in one notation. A log-scale dot plot against the floor would carry this better than a table.
- **Mermaid diagram** (`reproduce/architecture.qmd:10-79`): node labels are about 5 px at column width and low-contrast in dark. Break it out to the page width, or split it into two diagrams.
- Sheets: `scratchpad/visual/sheet_buildtime_light_A.png` and `scratchpad/visual/sheet_buildtime_light_B.png`

## 3. Proposed visual system

**Colour meaning, fixed on every page**

Raw gruvbox accents fail as data colours: `#076678` is below the chroma floor, and blue against purple collapses under deuteranopia (ΔE 4.0). The sets below pass the dataviz validator on all pairs against each page surface.

| Role | Light (`#f9f5d7`) | Dark (`#1d2021`) | Mark |
|---|---|---|---|
| Truth / the mass | `#bf4a0a` | `#e0701a` | ring + crosshair; solid 1.4 px rule |
| Belief / posterior / estimate | `#1a6fa3` | `#4595cf` | dots α≈0.28, line 1.8 px, ±1 s.d. band α 0.22 |
| Second mass's belief | `#a0306e` | `#d05c9c` | same as belief |
| Instruments / observed data | ink `#282828` | `#fbf1c7` | squares with ±σ bars |
| Thresholds, noise floor | `#665c54` | `#bdae93` | dashed (dashes used only for thresholds) |
| Axes / grid | `#d5c4a1` / `#ebe2bd` | `#504945` / `#2b2f30` | solid hairlines |

- **Ordered quantities** (M levels, K, σ) use the blue ramp: light `#7fb2d9, #3f86bd, #1a6fa3, #0d456b`; dark `#255f8c, #2f74aa, #4595cf, #8fc1e8`.
- **Rate deficit** uses the orange ramp: light `#e39a5f, #d4691f, #a83f08, #6b2603`.
- Drop viridis, coolwarm and tab10.

**Type and size**

- Use Source Sans 3 in figures, bundling the TTFs and registering them with `font_manager.fontManager.addfont`.
- Author at display size: 1 in = 100 CSS px, so a column figure is 7.5 in wide, text is 9.5–10.5 pt, and it is saved at 200 dpi.
- No matplotlib titles; the caption carries them. Keep one semibold status line in animations ("After 8 observations").
- Label directly instead of using legend boxes.

**Style block** (put it in a new `src/clocks/_style.py` with `apply(theme)`, called by `_panels.py`, `_panels3d.py`, `demo_density.py`, `_echo_study.py` and one setup cell per page):

```python
BASE = {
  "font.family": "Source Sans 3", "font.size": 10.5,
  "axes.labelsize": 10.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
  "axes.titlesize": 10.5, "axes.titleweight": "semibold", "axes.titlelocation": "left",
  "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
  "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
  "axes.grid": True, "grid.linewidth": 0.8, "lines.linewidth": 1.8,
  "lines.solid_capstyle": "round", "legend.frameon": False,
  "figure.figsize": (7.5, 4.2), "figure.dpi": 100, "savefig.dpi": 200,
}
LIGHT = {"figure.facecolor": "#f9f5d7", "axes.facecolor": "#f9f5d7", "savefig.facecolor": "#f9f5d7",
  "text.color": "#282828", "axes.labelcolor": "#665c54", "xtick.color": "#665c54", "ytick.color": "#665c54",
  "axes.edgecolor": "#d5c4a1", "grid.color": "#ebe2bd",
  "axes.prop_cycle": cycler(color=["#1a6fa3", "#bf4a0a", "#a0306e"])}
DARK = {"figure.facecolor": "#1d2021", "axes.facecolor": "#1d2021", "savefig.facecolor": "#1d2021",
  "text.color": "#fbf1c7", "axes.labelcolor": "#bdae93", "xtick.color": "#bdae93", "ytick.color": "#bdae93",
  "axes.edgecolor": "#504945", "grid.color": "#2b2f30",
  "axes.prop_cycle": cycler(color=["#4595cf", "#e0701a", "#d05c9c"])}
```

**Dark theme**

- Render every figure twice and swap. Quarto 1.9.32 is installed, so build-time cells can use `#| renderings: [light, dark]` and emit both.
- For the committed assets, write `name.light.*` and `name.dark.*` and wrap them in the `.figure-light` / `.figure-dark` classes. `site/styles.css:11-24` already defines these classes, and nothing uses them yet.

**Animation conventions**

- Use H.264 MP4 in `<video autoplay muted loop playsinline controls preload="metadata" poster=…>` with explicit `width` and `height`. Respect `prefers-reduced-motion` by showing the poster and controls.
- Start on the prior and hold it 1.2 s. Dwell 0.9 s on each of the first three updates, 0.5 s on the next five, then accelerate to 0.1 s. Hold the last frame 2.5 s. Total 12–15 s.
- Freeze every axis limit and every axes rectangle for the whole clip.
- Use 20–30 fps for camera moves, with the orbit eased.
- Target at most 0.5 MB for 2D clips and at most 1.5 MB for the 3D one. The poster is the final frame.
- Measured: transcoding the current GIFs unchanged takes 16.2 MB to 3.1 MB (in `scratchpad/visual/mp4_of_existing/`). The restyled prototype is 0.38 MB.

## 4. Page and CSS

**Broken now**

- **Clipped plots on phones.** Build-time `<img>` elements get `class="figure-img"` with a fixed `width="597"` and no `img-fluid`, inside an `overflow-x: auto` box, so a phone shows the left 57% of every plot. Fix: `.cell-output-display img, .quarto-figure img { max-width: 100%; height: auto; }`. Screenshot: `../screenshots/phone-clipping.jpg`
- **Sideways page scroll at 390 px.** `reproduce/reproducibility` is 569 px wide because of long inline code, and `method/the-particle-filter` is 443 px wide because of display math. Fix: `p code, li code { overflow-wrap: anywhere; }` and `mjx-container[display="true"] { overflow-x: auto; max-width: 100%; }`.
- **Dead rules.** `.quarto-title .quarto-title-subtitle` (`custom.scss:39-47`) matches nothing; the subtitle is `p.subtitle.lead` in plain sans. The `.navbar` rules (`custom.scss:56-73`, `custom-dark.scss:56-73`) match nothing either, because there is no navbar. Retarget the subtitle rule to `.quarto-title .subtitle` and delete the navbar block.
- **Layout shift.** GIF `<img>` elements have no dimensions and no lazy loading.

**Reads as default Quarto**

- **Callouts**: Bootstrap blue `#0d6efd` on a pale blue or navy header. Set `$callout-color-note: #1a6fa3` (the SCSS variable Quarto reads) and tint the header from the page background.
- **Inline code**: purple `#7d12ba` on a cool grey chip, and still a light chip in dark mode. Set `$code-color` to the body ink, with the background `#f2e5bc` in light and `#32302f` in dark.
- **Code blocks**: cool grey `rgba(233,236,239,.65)` on warm cream. Set `$code-block-bg` to match.
- **Dark-mode captions**: they come out mustard, `rgb(209,172,77)`. Set `figcaption, .figure-caption { color: #a89984; }`.
- **Measure**: 95–102 characters at 17 px across 749 px. Move the body to 18–19 px and cap the text with `main p, main li { max-width: 68ch; }`. Leave figures at the full column, or wider with `.column-page-right` for the dashboards and the Mermaid diagram.
- **Code fold**: the bare "▶ Code" is the template tell. Set `code-summary: "Show the code that drew this"` and style `details.code-fold > summary` as small caps-free sans in the secondary ink.
- **Figures**: give them `margin: 2rem 0` and style the caption at 0.9 rem in the secondary ink.
- **Tables**: `table { font-variant-numeric: tabular-nums; }` and right-align the numeric columns.
- **Site accent**: consider moving `$primary` and `$link-color` to `#1a6fa3` and `#4595cf`, so links and the "belief" colour are one colour.
- **Landing first screen**: two paragraphs come before any image, and the image is a four-panel diagnostic. See the next section.

## 5. Bigger moves and priority order

**Signature visual for the landing page.** A full-column field of small clock faces whose hands sweep at exaggerated, visibly different rates, slower near an unseen point, with a cloud of dots condensing onto that point. The reader understands "slow clocks point at the mass" before reading a word.

**Most valuable interactive.** One canvas component that replays precomputed Python output (particle states as JSON), with play, pause and scrub, a theme-aware palette read from CSS variables, and a stacked layout on phones. Replaying real output keeps the site's provenance rule intact, and it fixes resolution, dark mode, mobile legibility and file weight together. A second, cheaper widget is a draggable mass with live clock rates on "Clocks as Gravimeters"; it uses only the forward formula, so there is no inference to mis-state.

**Order I would do it in**

| # | Item | Effort | Payoff |
|---|---|---|---|
| 1 | CSS fixes: mobile clipping, overflow, measure, callouts, code, captions, dead rules | an afternoon | high |
| 2 | `_style.py`, palette and bundled font applied to all nine build-time plots, with light and dark renderings | an afternoon | high |
| 3 | Dashboard redesign: merged map, per-parameter panels, prior frame, fixed axes, pacing, MP4 with poster, both themes | a few days | highest |
| 4 | Echolocation scene: fixed layout, smooth orbit, tightened limits | a few days | high |
| 5 | Re-render `demo_density.png` and the range study in the system; replace the model-comparison animation with the line chart | an afternoon | medium |
| 6 | Landing hero and scrubbable replay component (OJS) | a larger build | highest, after 3 |
| 7 | Drag-the-mass widget; Mermaid breakout | a few days | medium |

Item 3 changes committed evidence assets, so it needs the same regeneration and certification note the site already carries.
