# Repo and website audit, 2026-11-17

Audited commit: `0171af3` (`origin/main`, the commit deployed at
<https://jbwhit.github.io/clocks/>). Nothing in the library, tests or site was
changed by this audit.

## How it was run

Five independent reviewers, each briefed separately so none was anchored on
another's reading:

| Slice | Model | Full report |
|---|---|---|
| Physics, forward model, scientific claims | Opus | [reviewers/physics.md](reviewers/physics.md) |
| SMC sampler, weights, model comparison | Opus | [reviewers/smc.md](reviewers/smc.md) |
| Calibration, certification, evidence chain | Opus | [reviewers/evidence.md](reviewers/evidence.md) |
| Story and science communication | Fable | [reviewers/story.md](reviewers/story.md) |
| Plots, animations, page design | Fable | [reviewers/visual.md](reviewers/visual.md) |

The reviewers' probe scripts and raw outputs are in [probes/](probes/), and the
restyled-figure prototype is in [prototype/](prototype/). These are records of
one-off checks, not maintained code, and are excluded from ruff. Paths in the
reports that begin `scratchpad/` pointed at a session-temporary directory and
were not kept.

**What is measured and by whom.** Each report marks its own claims as measured
or believed. The coordinating session separately confirmed: the local gate
(format, lint, 946 tests, 2 slow replays, all passing); a clean site render and
link-contract check; the prose quoted in findings C1, C2, C4 and C7; the
seed-spread data behind C3; the phone clipping screenshot behind V3; and the
prior bound behind C6. Everything else rests on the reviewer who reported it.

## Summary

The library is sound. The forward model matches its equations, the sampler
agrees with brute-force integration for one, two and three masses, the factorial
prior normalization is right, and every number the site publishes about the
range study traces to a committed artifact that still reproduces at this commit.

The problems are in what the site says and shows:

1. **Several statements on the story pages are false or misleading**, mostly
   because the physics was rescaled during the rigor passes and the prose and
   captions were not re-read against the new figures.
2. **The model-comparison page reports seed noise as a result.**
3. **The guard tests protect less than their names and the README suggest.**
4. **The story is written partly for an auditor**, never poses a question, and
   its animations begin after the search is already over.
5. **The figures are unstyled default matplotlib** on a white canvas, authored
   at twice their display size, and several are clipped on phones.

## Correctness findings

Ordered by severity. "Two reviewers" means the finding was reached
independently.

### Wrong

**C1. "One Clock Is Not Enough" states a parameter the code does not use, and
two false things about its own figure** (two reviewers).
`site/story/one-clock-is-not-enough.qmd:37` says `track_offset=1.0`; the cell at
line 46 uses 2.0, and at 1.0 the configuration is outside the weak-field domain
and raises. The caption at line 42 says the near, light mass gives a "sharper,
more localized dip"; both masses sit at the same offset, so the two dips have
the same shape. Lines 63-65 say the other four clocks "disagree more strongly"
than the clock at x=0; two of the four disagree less.

**C2. "How Many Masses?" says the difference between its two runs "should not
be attributed solely to Monte Carlo noise". It is almost entirely Monte Carlo
noise.** With the data fixed and only the inference seed varied, P(K=2) has a
standard deviation of about 0.2 at both shipped budgets (range 0.11 to 0.84
over 30 seeds each). Reference runs at 20,000 particles give P(K=2) of
0.50 ± 0.03 at 25 observations and 0.54 ± 0.04 at 80. The page's two headline
numbers, 0.740 and 0.399, are draws from that spread; the data themselves leave
two versus three masses at about even odds. The GIF caption's "mildly favors
K=3" is therefore not a finding. Data: `probes/smc/k*.txt`.

### Misleading

**C3. The reported ESS reads as healthy while the three-mass cloud collapses.**
At observation 80 the K=3 filter reports ESS 1964/2000 with 378 distinct
particles and 1.8% Metropolis-Hastings acceptance. ESS computed from weights
resets at every resample and cannot see duplicates. This poor mixing is the
likely cause of C2's variance.

**C4. "Two Hidden Masses" overstates how well one mass imitates two** (two
reviewers). The caption says the curves "nearly agree" and the callout builds a
cliffhanger on it, but a single five-clock reading already favours two masses
by a likelihood ratio near e⁸ in the demo geometry, and the next page gives
K=1 zero probability from the first observation. The figure also uses
`track_offset=2.0` while the demo it describes uses 1.0.

**C5. "The Search in One Dimension" describes an effect its figure does not
show** (two reviewers). Lines 78-84 say position locks in first and mass "keeps
a wider band for longer". Both shrink as 1/√N and in relative terms M is the
tighter of the two. The stated mechanism, that noise cancels in ratios, is also
wrong: the noise is independent per clock.

**C6. The 2D demo's true mass sits exactly on the prior's upper bound** (two
reviewers, one via the prototype figure). `demo_2d.py:23,75` and
`into-the-plane.qmd:48,55` use M = 0.15 with `mass_range=(0.005, 0.15)`. The
posterior is pinned against the ceiling, its width is about 43% narrower than
the data support, and the estimate can only err low. This is also the landing
page's hero animation.

**C7. "Beyond Point Masses" never says the density is a line density.** The code
integrates mass per unit length along the track. A physicist will picture a 3D
Gaussian blob, whose exterior field is a point mass by the shell theorem, and
conclude the page's two main claims are wrong. They are right only for a
filament. The page also lists "voids" as an example; the model enforces
non-negative density.

**C8. The weak-field policy is called "conservative" and the rate map is said
to agree with Schwarzschild.** The surrogate is exact only to first order in Φ.
At the policy limit its second-order term differs from the post-Newtonian form
by up to Φ² ≈ 2.5×10⁻³, which the simulated echolocation data resolve at 15
times the effective noise at close range. The simulations are self-consistent,
not GR-accurate, and one sentence should say so.

**C9. "Run exactly once" is an attestation, not something the repo enforces.**
There is no run ledger, the study writer overwrites its output, and blocks 100,
200 and 300 were spent on earlier implementations without the site mentioning
it. Since PR #12, CI also re-executes 24 of the 72 protected cases on every
push to `main`.

**C10. Smaller items.** The landing page prints `M = 0.10 ± 0.00`. The
echolocation coda's "sitting time" is the time to a 1σ signal but is described
as unambiguous. The glossary gives σ_obs as 0.005 "in the demos" while
echolocation uses 0.001. `reproduce/getting-started.qmd:45` leaks the internal
label "Task 11" and prints a bare `2` whose correctness is a lucky seed (see
T3).

### Tests that guard less than they claim

**T1. The "deterministic regression replay" tests never compare against the
archived records.** They assert only the pass gates. A drop from 46/72 to 30/72
would stay green. Today the records do match (72 of 72, to 5.6e-14).

**T2. Eight of nine deliberate prose drifts passed the prose guards.** Changing
46/72 to 49/72, the gate factor from 20 to 10, or "run exactly once" to "run
three times" turns nothing red. Rendering those numbers from the JSON with
Quarto inline code would remove the need for a guard.

**T3. `test_infer_model_comparison_returns_model_probabilities` asserts
`best_model == 2` at one seed.** Fifteen of twenty other seeds pass. This is C2
again, wearing a test.

**T4. Metropolis-Hastings invariance rests on one test.** All 11 sampler mutants
were killed, but four were caught only by the conjugate-normal test, which uses
a Gaussian prior rather than the API's bounded, ordered one.
`test_mh_does_not_change_evidence` cannot fail.

**T5. No evidence JSON records the commit, library versions or platform that
produced it.**

### Proportionality

**P1. `src/clocks/_reliability.py` (1,255 lines, about 2,300 lines of tests) is
imported by nothing outside `tests/`.** It preregisters interval methods for
which no code exists.

**P2. The certified weight normalizer never changed a result** on the shipped
80-observation demo: no subnormal weight arose in 345 calls, and output was
bitwise identical to the fast path. It is correct (808 fuzz cases against a
400-digit reference); it is also the subject of the most recent PR (#18).

### Checked and sound

Forward-model code and its domain enforcement; density quadrature (within
2×10⁻⁹ of an independent reference); every recomputed real-world number (GPS
offsets, Skytree, the millimetre redshift result, the coda table); the
echolocation contrast algebra and its 1/R² and 1/R³ scalings; SMC against
quadrature for one mass and against brute-force Monte Carlo for one to three;
resampling schemes; rollback in model comparison; byte-identical regeneration
of five of the six GIFs and of the range-study figure.

## Story

Full report with page-by-page notes, a proposed arc and three rewritten
passages: [reviewers/story.md](reviewers/story.md).

**S1. The rigor pass rewrote the story in an auditor's voice.** A provenance
sentence appears on six story pages, and a "Generated-evidence status" callout
is the third thing on the landing page. `reproduce/reproducibility.qmd` already
holds this material, so the story pages can drop it.

**S2. The story has no question and the search is never on screen.** The
landing page promises "a series of increasingly hard detection puzzles". Every
animation opens after the first observation, with the cloud already collapsed
and the true mass marked. The best question on the site, whether a person with
clocks in her head could feel the Sun, is at the bottom of page 8.

**S3. The spine already exists and should be protected.** Each strong page is
about two different worlds that give the same readings and what it takes to
tell them apart: near-light versus far-heavy, swapped labels, wide-weak versus
narrow-dense, direction versus range. The echolocation premise and coda are the
best writing on the site.

**S4. Proposed arc.** Keep eight pages; move the continuous-density page to
position 5 so the last three run uninterrupted; retitle pages as questions;
make page 4 about where to put the clocks; give page 7 a one-mass case where
Occam's razor is visibly cutting; end page 8 on the coda.

**S5. The one interactive that matters most: "Match the clocks" on page 2.**
The reader drags a candidate mass and tries to zero each clock's residual. With
one clock it can be done anywhere, so the reader finds the degeneracy by hand.
The forward model is two lines of Observable JS.

**Caution for S4.** The story reviewer's page-7 rewrite quotes 74/26 and 40/60
as results. Finding C2 shows those are noise. The page needs runs averaged over
seeds before any numbers are written into it, and the proposed one-mass
staircase needs the same treatment.

## Visual craft

Full report with a figure-by-figure critique, palette, matplotlib style block
and CSS: [reviewers/visual.md](reviewers/visual.md). Prototype of the direction
(real filter, committed 2D configuration, both themes):
[prototype/sheet_proto.png](prototype/sheet_proto.png), with paced MP4s beside
it at 0.38 MB against the current 1.5 MB GIF.

**V1. Rebuild the animations around what changes.** No prior frame; uniform
250 ms pacing with no end hold; axes that rescale every frame; position and
mass on one y-axis, so mass is a flat line; three panels showing the same x-y
plane; weight colouring that encodes nothing after resampling. In the 2D demo
6.3% of pixels change in the first transition and about 0.5% per frame after
frame 10.

**V2. One figure style that belongs to the page.** Every plot is DejaVu Sans,
tab10 and a black box on white. Text authored for 12-15 inch canvases is shown
at 749 px, so labels land at 3.5-5 px on desktop. No colour has a fixed
meaning: truth is red, coral, dashed or blue depending on the page. The report
proposes fixed roles (truth, belief, second mass, instruments, thresholds) with
hex values validated for both themes.

**V3. Broken now.** Build-time plots show only their left 57% on phones
([screenshots/phone-clipping.jpg](screenshots/phone-clipping.jpg)). Two pages
scroll sideways at 390 px. The subtitle and navbar rules in `custom.scss` match
nothing. The `.figure-light` / `.figure-dark` classes in `styles.css` are
defined and unused, so every figure is a white rectangle in dark mode.

**V4. Reads as default Quarto.** Bootstrap-blue callouts, purple inline code on
a cool grey chip, a 95-102 character measure, the bare "Code" fold.

**V5. Container.** Transcoding the existing GIFs unchanged takes 16.2 MB to
3.1 MB. H.264 with a poster frame also gives pause and scrub.

**V6. Bigger moves.** A landing-page field of clock faces sweeping at visibly
different rates with a cloud condensing on the unseen mass; and one canvas
component that replays precomputed particle states with a scrubber, which fixes
resolution, dark mode, mobile legibility and weight together.

## Repo health

- Local gate at `0171af3`: `ruff format --check`, `ruff check`, 946 tests, 2
  slow replays, all passing. Site renders; link and prose contracts pass.
- `core.hooksPath` is unset in the main clone, so `.githooks/pre-commit` is not
  running there.
- `.github/workflows/ci.yml` has no draft gating, no `concurrency` group, no
  `ready_for_review` type and no `timeout-minutes`. The deploy workflow has no
  `timeout-minutes`.
- Local `main` was 35 commits behind `origin/main` when the audit began and was
  fast-forwarded.

## Recommended order

The order matters because every committed animation is regenerated once the
demos or the figure style change, and the prose should be written against the
final figures.

1. **Fix what is false on the live site** (C1, C2's callout, C4, C5, C7, C10)
   and the phone clipping and overflow (V3). An afternoon; no assets change.
2. **Settle the changes that alter data**: move the 2D truth off the prior
   bound (C6); make model comparison report a spread over seeds, and add a
   distinct-particle diagnostic (C2, C3); choose one `track_offset` per page
   pair (C1, C4); decide whether page 7 gains a one-mass case (S4).
3. **Build the figure system and rebuild the animations** (V1, V2, V5), then
   regenerate every asset once.
4. **Restructure and rewrite the story** against the new figures (S1-S4), with
   provenance moved to the Reproducibility page.
5. **Interactives**: "Match the clocks", then the replay component and the
   landing hero (S5, V6).
6. **In parallel, independent of the site**: T1-T5, the CI workflow, and a
   decision on P1.

## Decisions that are the owner's

- How much certification and provenance language stays on story pages (S1, C9).
- Whether to finish the population-reliability study or park
  `_reliability.py` (P1).
- Whether committed GIFs remain the evidence artifact, or are replaced by video
  or replayed particle states (V5, V6). `tests/test_generated_assets.py` pins
  the current bytes.
- Page order and titles (S4).
