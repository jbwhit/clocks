I found no defect in the sampler's targeting or its evidence arithmetic: SMC matches brute-force integration for one, two and three masses. The real problems are in how its output is presented: one sentence on the site states something my measurements contradict, the reported ESS hides a collapsing three-mass cloud, and one test and the getting-started example rest on a seed that happened to win.

## Findings, most severe first

**1. Wrong (prose): the callout on `site/story/how-many-masses.qmd` says the two runs' difference "should not be attributed solely to Monte Carlo noise". It is almost entirely Monte Carlo noise.** Measured with `../probes/smc/kspread.py`, same data (sim seed 42), varying only the inference seed:
- 25 observations, 400 particles, 30 seeds: P(K=2) has mean 0.505, sd 0.205, range 0.11 to 0.82.
- 80 observations, 2,000 particles, 30 seeds: P(K=2) has mean 0.576, sd 0.175, range 0.18 to 0.84.
- Reference runs at 20,000 particles (4 seeds each): P(K=2) is 0.50 ± 0.03 at 25 observations and 0.54 ± 0.04 at 80.

The site's two numbers (0.740 and 0.399; I reproduced both exactly) are draws from those spreads. The effect of the longer data horizon is about 0.04, inside the error. At the shipped budget the K=2 vs K=3 verdict, and the GIF's "mildly favors K=3", is close to a coin flip. One driver: the three-mass log-evidence has sd 0.81 nats at 2,000 particles, against 0.21 for two masses.
- **Fix:** replace the callout with the measured spread. Better, report P(K) as a mean ± spread over several independent runs, or average the evidence estimates across runs.

**2. Misleading (diagnostic): the reported ESS says "healthy" while the three-mass cloud collapses.** In the 80-observation demo run (`../probes/smc/diversity.py`), at t=80 the three-mass filter reports ESS 1964/2000 but holds only 378 distinct particles, with MH acceptance of 1.8% at 2 steps per stage. At t=1 it had 932 distinct particles and 11% acceptance; the two-mass filter is down to 879 distinct at t=80. ESS from `1/Σw²` resets to N at every resample, so it never sees duplicates.
- The demo prints `ESS: …/2000` and the method page plots "final ESS", so both present it as a health figure. The 1D one-mass page case is healthy (378/400 distinct, 24–35% acceptance).
- The same poor mixing is a likely cause of finding 1's high three-mass variance. Since log of an unbiased estimate is biased down by roughly var/2, at a fixed budget the three-mass model is tilted by about −0.3 nats. That part is believed: N=2000 vs 20000 differ by −0.20 ± 0.22, which is not significant.
- **Fix:** report a duplicate-aware diversity measure (distinct-particle count, or ESS with duplicate weights merged), and adapt the number of rejuvenation steps to a target of accepted moves.

**3. Fragile (test and docs example): `tests/test_api.py::test_infer_model_comparison_returns_model_probabilities` asserts `best_model == 2` at seed 42.** Measured over 20 other inference seeds with the same configuration (40 observations, 1,500 particles): 15 of 20 pass, and P(K=2) ranges 0.16 to 0.85. Any change to how the code consumes random numbers fails it about one time in four. The getting-started page runs this exact configuration and prints `2`, which presents a lucky draw as recovery.
- **Fix:** assert only what the method reliably delivers, e.g. P(K=1) < 1e-3 and best model in {2, 3}, and caveat the docs example.

**4. Fragile (coverage): MH invariance rests on one test.** I ran 11 sampler mutants in a throwaway worktree against all 8 SMC/weight/API test files together, with bytecode cleared between mutants. All 11 were killed. But four of them were caught only by the two parametrizations of `test_conjugate_normal_posterior_and_evidence`:
- MH targeting β=1 at intermediate stages;
- MH not updating the stored target when a move is accepted;
- MH ignoring the fractional β;
- evidence ignoring the incoming weights.

That test uses a Gaussian prior. Nothing checks posterior or evidence against ground truth on the API's bounded, ordered, weak-field-conditioned prior.
- **Fix:** add a slow test built on `../probes/smc/quad_k1.py` and `../probes/smc/mc_kcomp.py`.

**5. Nit (a test that cannot fail): `test_mh_does_not_change_evidence` (`tests/test_smc_rigorous.py:803`).** `_metropolis_move` never touches `log_evidence`, so this cannot go red. Believed from reading; I did not mutate it.

**6. Nit/misleading: `ParticleFilter.converged()` (`inference.py:822`) is public, untested and undocumented.** It checks posterior narrowness (absolute std < 0.01 across parameters in mixed units) plus a stable mean. A stuck, collapsed cloud would report `converged=True`. Believed. Rename it or document what it measures.

**7. Nit (method-page prose):**
- "A final stage is also resampled if its ESS reaches the threshold": β=1 is only chosen when ESS ≥ target, so the `ess <= target + 1e-9` branch (`inference.py:781`) fires only inside a 1e-9 window. In practice a final stage is never resampled or moved.
- "Largest next exponent": bisection assumes ESS falls monotonically as the step grows, which nonuniform starting weights don't guarantee. Neither point affects validity.

**8. Nit (weight normalizer, the newest code): correct, but it never changes a result.**
- **Accuracy:** a fuzz of 808 cases against a 400-digit Decimal reference (offsets up to 1e18, subnormal-heavy, hand-picked edges) found no wrong weight, no lost weight and no invented weight.
- **Error inputs:** all −inf, NaN, +inf and empty arrays all raise `RuntimeError`. The message always says "All particles have zero weight", even for NaN or +inf. Inputs at ±1.7e308 leak a bare scipy overflow warning.
- **What the "certificate" certifies:** only that subnormal weights are correctly rounded relative to the binary64 log weights it is handed. It does not cover the evidence increment (plain `logsumexp`), the ESS trials, ordinary weights, or the upstream likelihood. The README sentence leaves out "relative to the supplied logs".
- **No effect on results:** on the shipped 80-observation demo there were 345 calls, no subnormal weight was ever produced, and every particle, weight and evidence value was bitwise identical to the fast path.
- **The cap failure is adversarial only:** the test's `WeightRoundingUndecided` population sits 7.7e-250 quanta from a rounding midpoint.

## Checked and sound
- **The algorithm, against a list written before reading the code:** incremental weight w·L^δ; evidence increment taken before resampling; MH target = prior × completed sufficient statistics × current likelihood^β; symmetric frozen proposal; out-of-support proposals rejected, never repaired; a −inf/−inf ratio gives NaN, which rejects. Systematic, stratified and residual resampling are unbiased by reading. The prior sampler sorts then rejects, giving the ordered density K!/V conditioned on validity. Rollback in ModelComparison is correct.
- **One mass vs quadrature** (12 seeds × 2,000 particles; log-evidence difference SMC − quadrature, ± standard error):

| Case | Difference ± se | Posterior means |
|---|---|---|
| Page case (25 obs, σ=0.005) | −0.005 ± 0.026 | x 2.4654 vs 2.4653; std matches to 4 digits |
| Medium data | −0.008 ± 0.012 | match |
| Weak data | −0.009 ± 0.028 | match |

- **Factorial normalization** (brute-force Monte Carlo over the unordered box, up to 8e7 draws):
  - Weak data (6 seeds): SMC − brute for K=1/2/3 = −0.02/+0.06/−0.02, with SE 0.04/0.07/0.04.
  - Moderate data (8 seeds): −0.02/−0.03/−0.08, with SE 0.04/0.05/0.12.
  - A mishandled K! would show offsets of 0.69 and 1.79 nats.
- **Method-page statements otherwise match the code:** proposal scaling `proposal_scale**2/D` on the weighted covariance before resampling, frozen per stage; the ESS formula; the support rules; "consistent, not exact" evidence.

## Not covered
- Posterior moments for two and three masses on sharp data (I checked their evidence only up to 4 observations).
- Whether the density-model and echolocation custom priors' samplers and log-densities are consistent with each other.
- 2D/3D inference paths.

The main checkout is untouched and the audit worktree has been removed. Probe scripts and raw outputs are in `../probes/smc/`. The mutation results are in `mutants.txt` there, and the seed-spread data in `k25_400.txt`, `k80_2000.txt`, `k25_20000.txt`, `k80_20000.txt` and `k40_1500.txt`.
