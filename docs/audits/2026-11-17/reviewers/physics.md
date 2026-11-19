I found eight physics problems on the site: two are outright false (both on "One Clock Is Not Enough"), five mislead a careful reader, plus a handful of nits. The forward model code itself is correct, and every real-world number I recomputed holds up. Nothing here changes a result; every fix is a wording or parameter change. One finding (#7) touches inference, so flag it to that reviewer.

## Findings, most severe first

**1. Wrong — the prose gives a parameter the code doesn't use, and the stated value is outside the model's domain.**
- **Where:** `site/story/one-clock-is-not-enough.qmd:36-37` says `track_offset=1.0`; the cell at line 46 uses `track_offset=2.0`.
- **Evidence (measured):** at 1.0, the "far heavy" mass (x=4, M=0.08) sits √2 from the clock at x=3, so Φ = -0.057. `clock_rates` raises `PhysicsDomainError: weak-field policy requires |2*Phi| <= 0.1` (my `probe1_pages.py`, first run). The code was evidently moved to 2.0 to stay legal and the prose was not updated.
- **Reader impact:** anyone reproducing the page with the stated value gets an exception.
- **Fix:** change the prose to `track_offset=2.0`, and drop "a geometry the next page makes standard" (the next page uses 1.0).

**2. Wrong — two claims about the 5-clock figure on the same page are false.**
- **Caption (line 42):** "the nearby light mass produces a sharper, more localized dip". Both masses sit on the same track at the same offset h, so both profiles are −M/√((x−x₀)²+h²). That is the same shape and width, only scaled and shifted. This follows from the algebra.
- **Prose (lines 63-65):** "the other four clocks disagree more strongly" than the x=0 clock. Measured |rate difference| at x = −6, −3, 0, 3, 6 is [0.00301, 0.00359, **0.00381**, 0.01839, 0.01971]. Two of the four disagree *less* than x=0.
- **Reader impact:** the reader learns the wrong reason the array breaks the degeneracy. It works by locating where the dip is centred, not by how sharp it is.
- **Fix:** caption becomes "…the two dips have the same shape but sit in different places, with different depths". Prose becomes "the clocks near x=3–6 disagree far more strongly".

**3. Misleading — "Two Hidden Masses" overstates how well one mass can imitate two.**
- **Where:** `two-hidden-masses.qmd:47, 53, 78-84`. The caption says "the two demo masses… at the five clock positions… nearly agree", and the callout says "If two masses can imitate one this well".
- **Geometry mismatch:** the figure uses `track_offset=2.0`, but the demo it describes (`demo_multi_mass.py:26`) uses 1.0. At 1.0 the combined mass at x=0.6 is outside the domain (measured: it raises).
- **How well it imitates (measured):** grid-fitting the best single mass inside the prior box and weak-field domain gives Δχ² per single 5-clock reading of **16.0** at the demo's h=1 and **6.2** at the figure's h=2. Even the plotted centroid single mass has residuals up to 2.9σ on one clock in one reading.
- **Reader impact:** "no single reading need be decisive" is roughly backwards. One reading already favours two masses over the best one-mass explanation by a likelihood ratio of about e⁸ in the demo geometry. The genuinely hard comparison on the next page is K=2 against K=3, not K=1.
- **Fix:** label the figure as h=2 rather than the demo geometry, and soften the callout ("the curves are close by eye, but the data separate them quickly").

**4. Misleading — "The Search in One Dimension" (lines 78-83) says x and M "do not tighten at the same speed" and that M "keeps a wider band for longer".**
- **Evidence (measured, running the page's own cell):** both posterior standard deviations shrink as 1/√N. In relative terms M is actually tighter: 0.017 against 0.022 for x after 25 observations, with the Cramér–Rao bound giving 0.0018 and 0.053. The x–M correlation is 0.02, so they are essentially independent.
- On the page's shared y-axis (x≈2.5, M≈0.1), M's ±0.0017 band is invisible, so the figure cannot show the described effect.
- The mechanism "noise mostly cancels out of the ratios" is also wrong: the noise is independent on each clock.
- **Fix:** plot relative std, or std divided by prior width, on separate axes. Say x is tighter *relative to its prior range*, and drop the speed and noise-cancellation explanation.

**5. Misleading — "Beyond Point Masses" never says the profile is a one-dimensional line (filament) density.**
- **What the code does (read):** `physics.py:577-602` and `demo_density.py:1` integrate a mass-per-length λ(x) along the track, seen from clocks offset by h.
- **What the page says:** "Gaussian density", "continuous mass distribution", and a "mass density" axis.
- **Reader impact:** a physicist will picture a 3D Gaussian blob. By the shell theorem its exterior field is essentially a point mass, so both "spreading the mass flattens the dip" and the σ–A ridge would look wrong to them. Both are real only because the mass is a filament.
- **The degeneracy claim itself checks out (measured):** the Fisher-information correlation between σ and A is −0.96 for the demo truth.
- **"Voids" cannot be modelled:** the page lists voids as examples, but they are negative density contrasts. The code enforces A ≥ 0 and Φ ≤ 0 (`physics.py:305, 125`).
- **Fix:** say "a Gaussian line density (mass per unit length) along the hidden track", and drop "voids" or note that only positive excess mass is modelled.

**6. Misleading — the |2Φ| ≤ 0.1 policy is called "conservative", but it lets through second-order differences from GR that the simulated data resolve.**
- **Where:** `units-and-scales.qmd:47-48` ("conservative operational policy… keeps the model in its declared regime") and `clocks-as-gravimeters.qmd:15-17` ("agrees with the Schwarzschild time-time factor").
- **The physics:** √(1+2Φ) agrees with Schwarzschild only if r is the areal coordinate. In isotropic or post-Newtonian form, g_tt ≈ −(1+2Φ+2Φ²), giving a rate of 1+Φ+Φ²/2 against the surrogate's 1+Φ−Φ²/2. They differ by Φ², up to 2.5×10⁻³ under the policy, and r here is a flat Euclidean distance between grid points.
- **How visible the difference is (measured, `probe3_echo_fisher.py`):** in the echolocation contrasts at range 2, ‖Q(surrogate − post-Newtonian)‖ is **15×** the effective noise after 80 observations. It is 6× at 2.6 circumradii and 2.4× at 3.5.
- **For scale:** |Φ| = 0.05 is about 500× a white dwarf's surface potential; "weak" here means "not a neutron star".
- **What still holds (measured):** the surrogate's nonlinearity lets the "unobservable" common mode leak into the contrasts, but it supplies only 1–3% of the range and mass information (Cramér–Rao bounds for √(1+2Φ) and 1+Φ agree to 1–3% at every sweep range). So the echolocation page's "curvature separates mass from range" claim is sound.
- **Fix:** one sentence: "the surrogate is exact for nothing beyond first order in Φ; at the policy limit its second-order term differs from GR by up to Φ² ≈ 2.5×10⁻³, which the simulated data can resolve — the simulations are self-consistent, not GR-accurate." Drop "conservative", or say what it is conservative relative to.

**7. Misleading (touches inference) — on "Into the Plane", the true mass is exactly the prior's upper bound.**
- **Where:** `into-the-plane.qmd:48, 55` and `demo_2d.py:23, 75`: the truth is M=0.15 and `mass_range=(0.005, 0.15)`.
- **Evidence (measured, running the page cell):** 99.5% of the weight is at M > 0.145, the posterior mean is 0.1487, and the posterior std is 0.0010 against a Cramér–Rao bound of 0.0018.
- **Reader impact:** the "three parameters recovered" picture has M's error bar about 43% narrower than the data support, and the estimate can only fall short of the truth.
- **Fix:** widen the prior (e.g. to 0.2), checking it stays inside the weak-field domain, or lower the truth.

**8. Nits**
- **Echolocation coda (lines 167-176, 205-213):** "sitting time" is the time to reach signal/noise = 1, and it ignores the √2 from differencing two clocks. "A minute or two… permanently and unambiguously" therefore overstates a roughly 1σ result. Say "≈1σ" or use 3σ (9× the time).
- **Coda line 243:** baseline/range at 1 AU is 6.7×10⁻¹³, which the page rounds to "about 10⁻¹²". Fine as an order of magnitude.
- **Glossary line 34:** "σ_obs (0.005 in the demos)", but echolocation uses 0.001 (`_scenarios.py:193`).
- **`index.qmd:10-11`:** "smear your position by kilometers within hours" follows the standard source (Ashby). A pedantic physicist may object that a drift common to all satellites is largely absorbed into the receiver's clock solution. The numbers themselves are fine: I measure +45.7, −7.2 and +38.5 µs/day, and 38 µs/day is about 0.48 km per hour of range.
- **Fragile tests:** `tests/test_shipped_ground_truths.py:46-57` hard-codes the 2D demo's clock positions, and line 65 hard-codes seed 11, instead of reading `demo_2d.CLOCK_POSITIONS` and `demo_multi_mass_2d.SEED`. If the demo changes, the test keeps checking a stale copy.

## Checked and found sound
- **Forward model code:** the potential sum, distances with the track offset, the √(1+2Φ) map, the |Φ| ≤ 0.05 enforcement, and the batch and multi-mass paths all match the README and site equations.
- **Density quadrature (measured):** across 400 random cases with h from 10⁻³ to 3, the scalar `quad` path is within 2×10⁻⁹ of my independent split-interval reference and the batch path within 6×10⁻⁶. For the demo truth, scalar and batch agree to 2×10⁻¹⁰, against noise of 0.005.
- **Real-world numbers recomputed:**
  - Earth surface 2Φ/c² = −1.39×10⁻⁹.
  - 1−√0.9 = 5.13%, and every shipped truth is shallower.
  - 1 cm of height ↔ 1.09×10⁻¹⁸.
  - Every row of the coda table, the 0.081 AU horizon (Sun 12.3× beyond it), Jupiter 28,500 km above the cloud tops, and the lunar/solar tidal potentials of 3.9 and 1.8×10⁻¹⁷.
  - The free-fall cancellation of the 1/R² term and the tidal formula 2GM·R⊕·h/(r³c²) are physically right.
- **Checked against sources (web):**
  - **Skytree:** Takamoto et al., *Nature Photonics* 2020 — clocks at the base and the 450 m floor, redshift tested at the 10⁻⁵ level.
  - **Millimetre redshift:** Bothwell et al., *Nature* 602, 420 (2022); the page's "about 1.1×10⁻¹⁹" is my recomputation, consistent with the paper.
- **Only recomputed, not sourced:** the GPS figures (match the standard values) and the "anomalies need 10⁻¹⁸–10⁻¹⁷" range on the units page (plausible, not sourced).
- **Echolocation:**
  - The contrast noise algebra (QQᵀ = I) is right.
  - The signal falls off with a log slope approaching −2.0, as claimed (−2.43 near, −2.10 far for the max component; −2.01 for the norm).
  - The 1/R² and 1/R³ scalings are right.
  - The single-clock degeneracy is exact as stated.
- **Mutation tests:** scaling the density potential by 1.10 or 1.01 in both the scalar and batch paths turns 9 of the density tests red. So the loose `atol=1e-3` in `test_narrow_density_approximates_point_mass` (about 10⁴× slack against the true agreement of ~10⁻⁷) is not a coverage gap. The worktree has been removed and the main checkout is untouched.

## Not covered
- The particle-filter, getting-started and architecture pages, beyond grepping them for formulas.
- The rendered GIFs and PNGs, which I did not inspect.
- `_animate.py` and `_panels*.py`, so I have not checked whether the plotted "differential rates" match their labels.
- `scripts/scan_echolocation_range.py`.
- Whether the 10⁻¹⁶/√τ and 10⁻¹⁸ budget used in the coda is state of the art. It is plausible, and the page presents it as an assumption.

Probe scripts are in `../probes/physics/` (`probe1_pages.py` through `probe4_density.py`).

Sources:
- [Takamoto et al. 2020, Skytree (RIKEN release)](https://www.riken.jp/en/news_pubs/research_news/pr/2020/20200407_2)
- [Physics Today on the transportable clocks](https://physicstoday.aip.org/news/transportable-atomic-clocks-achieve-laboratory-precision)
- [Bothwell et al. 2022 (JILA)](https://jila.colorado.edu/bibcite/reference/12594)
- [arXiv 2109.12238](https://arxiv.org/abs/2109.12238v1)
