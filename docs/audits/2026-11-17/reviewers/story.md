# Editorial review: "GPS in Reverse"

Assessment only: no repo files were edited, and `git status` is clean. I ran a few read-only checks with the repo's own library, flagged below where they matter. All line references are to files under `/Users/jonathan/projects/clocks/site/`.

## 1. Diagnosis

**What holds the story back**

1. **The rigor pass rewrote the story in an auditor's voice.** Every story page now carries sentences addressed to a reviewer. "The committed GIF is the current packaged default output from the corrected implementation" appears in some form on six pages, plus a "Generated-evidence status" callout as the third thing on the landing page. `git diff 5609966 HEAD -- site/story` shows the trade line by line: "Models that need the data explained to them lose; models that predicted it win" became "The implementation accumulates pre-resampling normalizing-constant increments through each adaptive tempering stage."

2. **Several figures no longer show what the prose says they show.** The physics was rescaled (masses 0.8 to 0.10, offsets changed) and the claims were not re-checked against the new plots.
   - **Page 2 caption:** says the near, light mass gives the "sharper, more localized dip"; the plot shows the far, heavy one with by far the deepest dip.
   - **Page 3 "What to notice":** says position locks in first while M "keeps a wider band for longer"; the figure shows a visible x band and a flat M line. I ran the cell: fractional uncertainty is 11% for x and 10% for M after one reading, 2.2% and 1.7% after 25.
   - **Page 5 caption:** says the one-mass and two-mass curves "nearly agree"; the plot shows a W against a V, differing by up to three noise standard deviations at a clock. Page 6 then gives K=1 probability 0.00 from the first reading.
   - **Page 7:** asserts a "new degeneracy" over a figure in which inferred and true profiles coincide.

3. **The search is never on screen, and the story has no question.** Every single-mass animation opens at `n_obs=1` with the cloud already collapsed; in the page-3 run the spread in x goes from 4.6 to 0.28 on the first reading. The true mass is marked from frame one, so there is nothing to guess. The landing page promises "a series of increasingly hard detection puzzles", which describes a ladder of demos. The best question on the site, "could she feel the Sun?", sits at the bottom of page 8.

**What to protect**

- **The degeneracy thread.** Each strong page is about two different worlds that give the same clock readings, and what it takes to split them: near-light vs far-heavy, swapped labels, wide-weak vs narrow-dense, direction vs range. This is the spine, though the pages do not yet share it.
- **The echolocation premise and coda** (`story/gravitational-echolocation.qmd:7-33`, `205-245`). This is the best writing on the site and a real ending: "a presence, not a place."
- **The hand-off lines:** "Fixing the number in advance was quietly cheating. The next page stops cheating." "Complexity isn't punished by rule; it is punished by arithmetic." "The disagreement carries the information."

## 2. Page-by-page

### Landing: `/Users/jonathan/projects/clocks/site/index.qmd`

- **Works:** lines 6-16. The 45/7/38 microsecond arithmetic and "But flip the logic around" are a strong hook.
- **Fails:**
  - The promise sentence (17-22) turns into three glossary links: "a forward model from summed Newtonian potentials and a pedagogical weak-field rate map, a particle filter for the inverse problem".
  - The GIF (24) is a four-panel dashboard that starts already solved. Its caption used to tell the reader how to read the panels and now says "Current 2D inference demo showing…".
  - The callout (26-30) is written for an auditor.
  - The code cell prints `M = 0.10 ± 0.00` (57-58), a zero-uncertainty claim on a site about honest uncertainty.
  - The contents list (63-79) describes pages in abstract terms: "label switching and what honesty about degeneracies looks like".
- **After thirty seconds:** the reader knows GPS corrects for relativity and that the idea can be inverted. They have not been asked anything, so there is no reason to click.

### 1. `/Users/jonathan/projects/clocks/site/story/clocks-as-gravimeters.qmd`

- **Job:** convince the reader the premise is real, then introduce the toy.
- **Works:** the Skytree paragraph (29-35); "A sufficiently precise clock, then, is a gravimeter" (37-39); the one-pixel hand-off (76-81).
- **Fails:**
  - It opens with an equation and then lines 14-20 ("pedagogical weak-field surrogate", "Schwarzschild time-time factor", "relative to the chosen coordinate time"). The earlier text read: "We will not derive it here, just use it relentlessly… it is true of every clock, everywhere, all the time."
  - The order is backwards. "This is measured, not theoretical" should come first, the formula second.
  - The GPS bullet (26-28) repeats the landing page.
  - The best fact about the toy is hidden in `method/units-and-scales.qmd:52-55`: real dips are parts per billion, the demos use a few percent, and the 1D demo's clocks sit about 13 Schwarzschild radii from the mass. That belongs here as a reveal.
  - The "gravity camera" metaphor (37-39) is set up and never used again.

### 2. `/Users/jonathan/projects/clocks/site/story/one-clock-is-not-enough.qmd`

- **Job:** the keystone idea. It has the most consistent voice of any page.
- **Works:** "the clock simply cannot tell you which world it lives in" (10-11); "Two visibly different universes, one identical clock reading" (25); the noise-floor callout (67-74).
- **Fails:**
  - The reveal is two printed numbers (`0.979796` twice).
  - The fix changes the geometry between demonstrations. Prose says `track_offset=1.0` (37), code uses 2.0 (46), so the x=0 clock that just could not tell the worlds apart now can, and lines 63-64 have to explain that away.
  - The caption (42) contradicts the plot, as above.
- **Missing figure:** the degeneracy as a set of surviving worlds. I confirmed the library draws it today. With one clock, the particle cloud in (x, M) is a clean V; with clocks at −5 and 0 it is two separate arcs; with three clocks it is a knot. See `../screenshots/ridge_check.png`.

### 3. `/Users/jonathan/projects/clocks/site/story/the-search-in-one-dimension.qmd`

- **Job:** introduce the filter and show the first recovery.
- **Works:** the three-sentence setup (7-9).
- **Fails:**
  - The GIF comes before the reader knows what a particle is, and shows no search. Its mass axis runs to 2.0 against a prior of 0.005-0.15, and its rate bars start at zero, so all three look like 1.0.
  - "Meet the detective" (18-27) contains no detective. It contains three untaught terms: "Adaptive likelihood tempering… Metropolis-Hastings moves restore diversity while preserving each tempered target."
  - "its spread is the claimed uncertainty" hedges where the reader wants to know whether the spread is right.
  - "What to notice" (78-84) is not in the figure.
- **Needs:** a frame of the prior, the one-clock/two-clock/three-clock strip above, and the standard reweight / resample / move picture.

### 4. `/Users/jonathan/projects/clocks/site/story/into-the-plane.qmd`

- **Job:** currently unclear; this is the weakest page. It opens by saying what does not change (7-10).
- **Fails:**
  - The asymmetry bullet (12-17) answers an objection no reader raised: "a true degeneracy appears only if channel identity is discarded or the data vector itself respects the permutation." The true, simple statement is that clocks in a straight line cannot tell a mass from its mirror image across that line.
  - The scatter plot shows only the end state.
  - The callout promises the multi-mass pages "will feel" the curse of dimensionality "properly" (83-85). Nothing later pays that off, though the 2D two-mass GIF does visibly struggle for about 30 readings.
- **Give it a question:** where should the clocks go?

### 5. `/Users/jonathan/projects/clocks/site/story/two-hidden-masses.qmd`

- **Works:** "'Mass 1' and 'mass 2' are *our* labels; the physics doesn't care" (21); the admission that the mass "was moved to 4.5 to keep the demo crisp" (35-37); the closing callout (82-86).
- **Fails:**
  - Two four-panel GIFs are stacked at the top before any explanation and never referred to again. The 2D one is the most dramatic animation on the site and nobody tells the reader to watch it.
  - Lines 24-30 are implementation: "the normalized ordered version of the exchangeable prior".
  - The heading "an honest limitation" labels its own virtue.
  - The superposition caption (47) and the cliffhanger premise "If two masses can imitate one this well" (83) are contradicted by the plot and by page 6.
- **Missing figure:** label switching as a picture. Two modes in (x₁, x₂), the mean landing between them, then the fold along the diagonal.

### 6. `/Users/jonathan/projects/clocks/site/story/how-many-masses.qmd`

- **Works:** the opening (7-8) and lines 32-35.
- **Fails:**
  - The page is titled for Occam's razor and never shows it cutting. Its exhibits are 74/26 for the truth and 40/60 against it.
  - With a two-mass truth, K=1 loses by misfit, which is not the razor.
  - The caption announces the wrong answer; lines 18-22, 28-31, 35-38, the parenthetical (42-45) and the callout (87-97) are all audit prose.
- **A check I ran:** with a one-mass truth (x=1.5, M=0.05, five clocks, 25 readings, 400 particles, three seeds), the probabilities form a staircase. K=1 gets 0.67-0.78, K=2 gets 0.15-0.29, K=3 gets 0.04-0.07. That is the razor shown. It is a small check and needs proper runs.

### 7. `/Users/jonathan/projects/clocks/site/story/beyond-point-masses.qmd`

- **Works:** the first sentence (7-8), "ore bodies, voids, aquifers", is the only place the site says what this could be for. The spreading figure is the cleanest forward-model picture on the site.
- **Fails:**
  - The degeneracy section is asserted; the old "disagree visibly… yet rates match" beat was deleted when the figure stopped showing it.
  - Lines 56-62 are implementation.
  - The "Where this could go" callout (64-71) is a leftover ending from before page 8 existed.

### 8. `/Users/jonathan/projects/clocks/site/story/gravitational-echolocation.qmd`

- **Works:** the premise (7-21); "Direction is cheap; range is expensive" (31); the coda.
- **Fails:**
  - The Q-matrix derivation (68-80) belongs in Part 2.
  - The result (99-120) reads "pass counts from close to far were 12, 12, 11, 8, 3, 0… ratio was 66.524, above the frozen factor 20 gate".
  - The Verdict (124-129) declines to give one.
  - The last words of the story are certification jargon (247-257).
- **What the data says and the page does not.** From `docs/calibration/echolocation_range_certification.json`:
  - The truth was inside the filter's 3-sigma intervals in all 72 runs.
  - At eight radii the median direction error is about 3° while the distance error is about 12% of range. This is my own decomposition of the posterior means and should be re-derived in repo code.

### Method and Reproduce

- **Structure:** the three-part split is right.
- **`method/the-particle-filter.qmd`:** this is a reference page. Page 3 sends curious readers there for "mechanics", and by the second paragraph they hit "log density −∞". It needs a short "in pictures" section at the top.
- **`method/units-and-scales.qmd`:** should become the single home for "what this toy is and isn't".
- **`reproduce/getting-started.qmd`:** prints an unlabeled `2` and a dict containing `2.79e-205`. Line 45 leaks an internal name, "Task 11".
- **`reproduce/reproducibility.qmd`:** already holds all the certification language, which is why the story pages can drop it.

## 3. Proposed arc

**Spine:** what can a set of clocks know, and what can they never tell apart? Each page presents two worlds with the same readings and what splits them.

Keep eight pages, with one move: the blob page goes to position 5, so that two-masses, how-many and echolocation run uninterrupted to the end.

| # | Title | Question | Key beat | Figure |
|---|---|---|---|---|
| 1 | Clocks as Gravimeters | Is this real? | Skytree first, formula second, then "our toy turns gravity up tens of millions of times" | Existing curve, plus a real-vs-toy scale bar |
| 2 | One Clock Is Not Enough | Near and light, or far and heavy? | Reader fails to pin the mass with one clock, succeeds with three | **Interactive** and the V-shaped cloud |
| 3 | The Search | How does a machine do what you just did? | A thousand guesses; one-clock V, two arcs, knot | Cloud strip starting from the prior; reweight / resample / move |
| 4 | Where to Put the Clocks | Why not a neat row? | Mirror-image worlds; each pair of clocks pins the mass to a circle of constant distance ratio | Circles intersecting; 2D animation from the prior |
| 5 | Point or Blob? | Is it even a point? | Two visibly different profiles, same five readings | Twin profiles and the width-amplitude ridge |
| 6 | Two Hidden Masses | Which is which? | The average of two right answers is a wrong one | Two-mode cloud, then folded |
| 7 | How Many Masses? | Who told you there were two? | One-mass case: all three models fit, the simplest wins; two-mass case: the razor is dull, and says so | Staircase bars, then the existing run |
| 8 | Gravitational Echolocation | How far does the sense reach? Could she feel the Sun? | Direction survives, distance goes, the filter knows it; coda | Direction-vs-distance panel; stop after the coda |

- **Page 4 pays off the title.** GPS intersects spheres of known distance; this intersects spheres of known distance ratio (Apollonius circles in the plane). The geometry is exact for one point mass in this model.
- **Cut:** the Q-matrix block, the per-page provenance sentences, and the asymmetry bullet.
- **Move:** the roadmap callout to after the coda.

**The single interactive that matters most: "Match the clocks", on page 2.**
- Clocks show their observed rates.
- The reader moves a candidate mass with position and mass sliders; each clock shows a bar for predicted minus observed.
- With one clock active, the reader can zero the bar anywhere by trading mass for distance, and so discovers the degeneracy by hand.
- Switching on more clocks leaves only one place that works.
- Page 3 can then open with "the filter is a thousand of you doing this at once".
- The forward model is two lines of Observable JS; no server or inference code is needed.

## 4. Three rewrites

**Landing opening** (replaces `index.qmd:6-33`)

> Your phone can find you because engineers correct for Einstein. A GPS satellite's clock sits higher in Earth's gravity than yours, so it runs fast by about 45 microseconds a day; its orbital speed slows it by about 7. The net 38 microseconds sounds like nothing. Uncorrected, it would smear your position by kilometers within hours.
>
> So GPS treats gravity's effect on time as an error to subtract. This site runs the idea backwards. If gravity changes how fast a clock ticks, then a clock is a gravity sensor, and a handful of clocks that disagree are telling you where the mass is.
>
> Below, eight clocks surround a hidden mass, each colored by how slowly it ticks. Before you scroll, guess where the mass is.
>
> *[clocks only; then a cloud of guesses closing in; then the reveal]*
>
> Eight short pages follow. They begin with a single clock that cannot tell a small mass nearby from a large one far away, and end by asking whether a person with atomic clocks in her head could feel the Sun. The physics is a teaching model, with gravity turned up tens of millions of times so the effect is visible; [The Toy and the Real Thing] says what that leaves out.

**Page 7 opening** (replaces `how-many-masses.qmd:7-38`)

Numbers are from my three-seed check and are placeholders.

> Every page so far told the filter how many masses to look for. To stop cheating, run three filters side by side, one that believes in one mass, one in two, one in three, and ask each the same question: how probable did you say these readings were before you saw them? That number is the model's evidence.
>
> Start with the easy case: one mass. All three models can fit it, since a spare mass can always be parked somewhere faint. After 25 readings the one-mass model holds about 70% of the probability, two masses about a quarter, three about 5%. Nobody told the filter to prefer simple answers. A model with more freedom spreads its predictions over more possible data, so it assigned less to the data that arrived. Complexity isn't punished by rule; it is punished by arithmetic.
>
> Now two masses. The one-mass model is dead after the first reading, but that is misfit, not the razor. Between two and three the razor is dull: 74 to 26 for the truth after 25 readings, and 40 to 60 against it in the longer run above. *[One sentence of why, once the technical reviewers have established it.]* Read the bars as odds: at least two, possibly a faint third.

**Page 8 result and verdict** (replaces `gravitational-echolocation.qmd:99-129`)

> To find out, move the mass away in six steps, from two head-radii to eight, and at each distance run the whole experiment twelve times with fresh noise. A run passes if it places the mass within one unit, the head's own half-width, and gets the mass within 0.04.
>
> Close in, the sense is sharp. At two radii all twelve runs pass, typically within 1% of the distance. At three and a half, eleven of twelve. Then it goes: eight at 4.6 radii, three at 6.1, none at eight.
>
> What fails is not direction. At eight radii the head still points within about three degrees of the mass. It loses distance, and mass with it. This is page 2's problem returning from outside: the 1/R² gradient says which way, and only the fainter 1/R³ curvature says how far.
>
> The fading is honest. The dashed line is the uncertainty the filter claimed and the solid line is the error it made; they rise together, and in all 72 runs the truth lay inside the filter's own three-sigma range. The head stops knowing where the mass is, and knows that it has stopped.
>
> **Verdict.** In this toy (27 clocks, noise at one part in a thousand, 80 readings) the sense reaches three or four head-radii as a sense of place and persists beyond that as a sense of direction. The numbers belong to this setup and these 72 runs. The shape belongs to the physics.

## 5. Prioritized changes

| # | Change | Size | Impact |
|---|---|---|---|
| 1 | Move provenance and certification language off the story pages into one home (Units and Scales, Reproducibility); captions say only "one run, seed 42" | edit | high |
| 2 | Reconcile the four claim-vs-figure mismatches (`one-clock-is-not-enough.qmd:42`, `the-search-in-one-dimension.qmd:78-84`, `two-hidden-masses.qmd:47` and `:83`, `beyond-point-masses.qmd:51-55`) | edit, or new figure | high |
| 3 | Rewrite the landing opening and contents list as questions; trail the Sun question | edit | high |
| 4 | Animations start from the prior, hide the truth until the end, and show rates as deviations from 1; landing hero is a single panel | changed figures | high |
| 5 | One-clock / two-clock / three-clock cloud strip on pages 2-3 | new figure | high |
| 6 | "Match the clocks" | new interactive | high |
| 7 | Page 8: rewrite result and verdict, add direction-vs-distance panel, end on the coda | edit plus new figure | high |
| 8 | "How Many Masses?": lead with a one-mass razor case, reframe the 60/40 | restructure plus new figure | medium-high |
| 9 | Move the blob page to position 5 with a twin-profile figure; roadmap callout to the end; retitle pages as questions | restructure | medium |
| 10 | Page 4 becomes "Where to Put the Clocks" (mirror twins, distance-ratio circles) | edit plus new figure | medium |
| 11 | Page 1: evidence before formula; add the scale reveal | edit | medium |
| 12 | Label-switching picture; reweight / resample / move strip on page 3 and the method page | new figures | medium |
| 13 | Commit to or drop the "detective" and "camera" metaphors | edit | low |
| 14 | Label Getting Started output; remove "Task 11" | edit | low |

**For the visual reviewer:** in every mobile capture the body text is cut off at the right edge (for example "Finding hidden masses with an array of ti"). I cannot tell from the screenshots whether that is a real overflow or a capture artifact; it needs checking on a phone.
