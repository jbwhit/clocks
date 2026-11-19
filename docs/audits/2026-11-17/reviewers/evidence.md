## Evidence-chain audit at HEAD 0171af3

Every quantitative claim the site makes about the range study and calibration traces to a committed artifact and recomputes exactly. I also re-ran the evidence at HEAD and the archived records came back unchanged. The weaknesses are in what guards these numbers, how strong the "one-shot" wording is, and a 1,255-line reliability module that nothing on the site uses. Nothing I found makes a published number wrong.

### Findings, most severe first

**1. Fragile: the "deterministic regression replay" tests never compare against the archived records.**
- **Where:** `tests/test_acceptance_echolocation_3d.py:67-104` and `tests/test_acceptance_multi_mass_2d.py:28-42`. They assert only the gates: at least 10/12 close-range passes, and a far/close ratio of at least 20.
- **Evidence (measured):** at HEAD, the full block 400–411 gives 46/72 passes, zero pass or coverage flips, zero evaluation-count mismatches, and at most 5.6e-14 relative difference in the means and standard deviations. The multi-mass block gives 12/12 with at most 2.2e-12 relative difference. The run took 39 s wall time (290 s CPU) with `../probes/evidence/replay_all.py` and `replay_multi.py`.
- **What a reader would wrongly conclude:** the README calls these outcomes "deterministic regression evidence", and the dev report says "Both deterministic slow replay tests subsequently passed". A reader assumes CI keeps HEAD matching 46/72 and "12, 12, 11, 8, 3, 0". It doesn't: a drop to 10/12 close passes, or to 30/72 overall, would stay green. Today it happens to match.
- **Smallest fix:** make each slow test compare every record with its archived JSON row, with pass, coverage and evaluation count exact and mean and std at roughly rtol 1e-9. The cost is about 40 s on this Mac. The reproducibility page's "compare floating results with tolerances while checking counts and controls exactly" is true only of the small replays in `tests/test_scenarios.py:1070-1162`.

**2. Fragile: the prose guards are literal-string spot checks, and 8 of 9 drifts pass.**
- **Evidence (measured):** I mutated prose in a throwaway worktree and ran `tests/test_generated_assets.py`, `tests/test_site_prose_contract.py` and `tests/test_calibration.py`. These are the only tests that read prose.

| Mutation | Result |
|---|---|
| Echolocation page: 46/72 changed to 49/72 | passes |
| Echolocation page: "close range passed 12/12" changed to 11/12 | passes |
| Echolocation page: gate "factor 20" changed to "factor 10" | passes |
| Echolocation page: "12/12 far-range" changed to 11/12 | caught (1 failed) |
| Dev report: per-range 12,12,12,8,4,1 changed to 12,12,12,9,4,1 | passes |
| Dev report: ratio 66.226 changed to 61.226 | passes |
| Dev report: multi-mass certification 12/12 changed to 11/12 | passes |
| Dev report: 49/72 changed to 50/72 | passes |
| Reproducibility page: "run exactly once" changed to "run three times" | passes |

- `test_site_prose_contract.py` is only a denylist of retired jitter-API phrases (`scripts/check_site_links.py:39-64`). It checks nothing about evidence.
- The `"12/12" in text` assertions are satisfied by any of several occurrences.
- The dev table lines are genuinely recomputed (`test_calibration.py:524`), but none of the sentences after those tables are.
- **Smallest fix:** compute the claims from the JSON inside a test and assert the exact sentences. Better, render them on the page with Quarto inline code reading the JSON.

**3. Misleading (mild): "the reserved 400–411 block was run exactly once" is an attestation the repo can't enforce or detect.**
- **What exists (believed, from reading):**
  - Protected blocks reject control and range overrides.
  - The archiver accepts only block 400.
  - The literal constant pins and the hash pin mean any post-hoc retune shows up in git history.
- **What doesn't exist:**
  - There is no run ledger.
  - `write_study` silently overwrites `output/..._seed_block_400.json`.
  - Blocks 500, 600, … are allowed and nothing records their use.
  - Nothing stops someone running block 400 against uncommitted code.
- **History (measured with git):**
  - At commit 02de41f (Aug 16, 14:49), runnable `@slow` tests targeted block 400 with the pre-freeze defaults (0.8, 2, 2.38). Commit 3fda465 replaced them nine minutes later.
  - CI did not run them: the slow step didn't exist yet, and `gh run list` shows a single PR run, at 26bdee3. Whether anyone ran them locally can't be determined.
  - Blocks 100, 200 and 300 were burned on earlier implementations; block 100 was used "for diagnostics". The site doesn't mention this.
  - Since #12, CI re-executes 24 of the 72 protected cases on every push to main.
- **Smallest fix:** word it as "by protocol (not mechanically enforced)", disclose the burned blocks, and record the commit SHA and a timestamp inside the certification JSON.

**4. Proportionality: `src/clocks/_reliability.py` (1,255 lines, plus about 2,300 lines of tests) protects nothing the site claims.**
- **Evidence (measured with `rg`):** only `tests/` imports it. No script, page or committed manifest uses it.
- The manifest preregisters `"wilson"` and `"stratified_bootstrap"` intervals, but no interval code exists anywhere in `src/` or `scripts/`. So there is no binomial math to check.
- The machinery is heavy for an artifact that has never been produced: 40-digit Decimal generation, directory fsync, rejection of negative zero and duplicate keys.
- `local_identifiability` is also unused. Its CRLB formula, sqrt(diag(V S⁻² Vᵀ)) with the n/σ² scaling, is correct by derivation.
- `docs/someday-maybe.md` still lists the population study as future work.
- **Recommendation:** finish the study or park the code on a branch. Until then, the site should not reference it.

**5. Fragile: the evidence JSON carries no code provenance.**
- None of the five JSON files records a commit SHA, library versions or platform. The certified code ("a1b016b") exists only in prose.
- HEAD has drifted from that code (26bdee3, plus the September SMC and density changes). It still reproduces the evidence (finding 1), but the artifacts can't show which code made them.
- **Fix:** record `git rev-parse HEAD`, the numpy and Python versions, and the platform in the JSON.

**6. Nit: `demo_density.png` no longer reproduces at HEAD.**
- **Evidence (measured):** 5 pixels differ by one intensity level. This is drift from the September density-quadrature changes and is invisible to the eye.
- The `index.qmd` callout says the committed PNGs come from the "corrected physics"; only the echolocation GIF carries a staleness caveat.
- `test_generated_assets.py` pins the committed bytes but never regenerates them, so it can't catch this.

**7. Nit: shared seed streams.**
- **Evidence (measured):** `run_echolocation_3d` (`_scenarios.py:782-783`) uses `PCG64(seed)` for both the observation noise and the inference stream. The multi-mass path does the same.
- The same seed is reused across all six ranges, so the 72 cases are 12 independent noise realisations, not 72 independent trials.
- The first-draw correlation over 200k seeds is -0.006, so there's no evidence of material bias.
- The population design spawns separate streams per case; the certified path doesn't. This matters only if anyone treats 46/72 as a binomial count.

**8. Nit: wording on the echolocation page.**
- Line 101 says the dev seeds "selected and froze the controls and tolerances by the declared ranking". The ranking selected only the controls.
- The 1.0 and 0.04 tolerances were carried over, and the factor 20 was set after seeing the 66.226 dev ratio. The dev report discloses this; the page doesn't.

**9. Nit: `index.qmd` prints "M = 0.10 ± 0.00".** The standard deviation rounds to zero, which reads as no uncertainty.

### Claims and the artifacts behind them
All of these were measured.

| Claim | Stated in | Source | Matches |
|---|---|---|---|
| 46/72 overall; per-range 12, 12, 11, 8, 3, 0 | echolocation page, dev report | `docs/calibration/echolocation_range_certification.json` | yes |
| Close range 12/12; far coverage 12/12 | echolocation page, dev report | same | yes |
| Far/close std ratio 66.524 (66.5244308246615) | same | same | yes |
| Close max errors 0.06890…, 0.003180…; medians 0.04193…, 2.7897… | dev report | same | yes |
| Gate factor 20; tolerances 1.0 / 0.04 | page, report | JSON `tolerances`, `_scenarios.py` | yes |
| Dev: 27-line tables (both studies) | dev report | `*_development.json` | yes (also tested) |
| Dev echo: winner (0.9, 1, 1.5) at 49/72, 0.333 vs 0.337 | dev report | same | yes |
| Dev echo: 12,12,12,8,4,1; ratio 66.226; close max 0.0633 / 0.00309 | dev report | same | yes |
| Dev multi: winner (0.7, 2, 3.0) at 11/12; sole failure seed 6 | dev report | same | yes |
| Multi certification: 12/12; median normalized error 0.24165…; median evaluations 833513 | dev report | `multi_mass_2d_certification.json` | yes |
| Source SHA a4c6b1… | report, test | `assets/echolocation_range_study.json` | yes |
| K=2: 0.740, K=3: 0.260 (live cell) | how-many-masses | rendered `_output` shows 0.74 / 0.26 | yes |
| K=2: 0.3990, K=3: 0.6010 | how-many-masses, dev report | `demo-model-comparison` re-run at HEAD | yes |
| Coda: 1.1e-17, 1.1e-19, about 7 yr, two orders, four further orders, a million times, 0.08 AU, 12 times, about 1e-12, 30,000 km | echolocation page (hard-coded prose) | recomputed from the cell's constants | yes (nothing guards them) |
| Asset hashes | dev report, test | `assets/` | yes; the density PNG and echo GIF no longer reproduce at HEAD |

### Checked and sound (all measured)
- **Archived records still reproduce at HEAD:**
  - Certification block: 72/72 records match.
  - The selected development cell: 72 records, 49 passes, at most 6.1e-14 relative difference.
  - The multi-mass certification block: 12/12 records match.
- **Range-study figure:** re-rendering `echolocation_range_study.png` from the JSON is byte-identical.
- **Demo GIFs:**
  - `demo_1d`, `demo_2d`, `demo_multi_mass`, `demo_multi_mass_2d` and `demo_model_comparison` are byte-identical at HEAD.
  - The echolocation GIF diff after frame 0 is confined to the rates panel, as the site discloses. Frame 0 differs throughout; I believe that is the GIF palette.
- **Asset copies:** `site/assets/` is byte-identical to `assets/`.
- **The two range-study JSON copies:** the shipped one equals the tracked one apart from the `source` block.
- **Pass and coverage flags:** they agree with the recorded tolerances.
- **Reference intervals, if the site ever wants them** (none exist in the repo):
  - 12/12: Clopper–Pearson lower bound 0.735 (closed form 0.025^(1/12)), Wilson 0.758.
  - 0/12: upper bound 0.265 / 0.243.
  - 46/72: Wilson (0.52, 0.74).
  - 1/1: Clopper–Pearson (0.025, 1), Wilson (0.21, 1).

### Not done
- I didn't run the slow tests on Linux to check cross-platform drift.
- I didn't verify the `_reliability` comment that macOS libm misrounds 2 of the 768 exp draws.
- I didn't check the release-manifest hash; your full suite covers it.
- I didn't do a numeric Fisher check beyond the derivation, or a link-contract run.

The probe scripts, regenerated media and the full HEAD replay are in `../probes/evidence/`; the HEAD replay is `head_cert_replay.json`. The audit worktree was clean before I removed it. The main checkout is untouched, and I left the other reviewer's `.worktrees/audit-smc` alone.
