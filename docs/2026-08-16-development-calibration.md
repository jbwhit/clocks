# Calibration, certification, and generated assets — 2026-08-16

This report records the development-only calibration used to freeze the two
shipped scenario configurations after the rigorous SMC remediation. It is a
finite fixed-seed recovery/regression study, not an estimate of reliability in
any declared population.

Both declared 27-cell grids were run on development seeds 0–11. Cells were
ranked by the predeclared lexicographic rule:

1. highest pass count;
2. lowest median normalized parameter error;
3. fewest median forward-model evaluations; and
4. lower rejuvenation-step count.

The protected seeds 400–411 were untouched while these controls, tolerances,
and gates were selected and frozen. The freeze was pushed as commit `a1b016b`;
the protected block was then executed exactly once for certification. Generated
evidence assets were regenerated only after that certification, as recorded
below.

## Raw development evidence

The tracked [multi-mass raw development artifact](calibration/multi_mass_2d_development.json)
and [echolocation raw development artifact](calibration/echolocation_range_development.json)
use the deterministic schema v1. Each declares `seed_block=0`, the exact
seeds, control grid, frozen tolerances, and all unrounded run records: 324 for
multi-mass and 1,944 for echolocation. This makes the rankings and gates below
reproducible without inferring values from rounded console summaries.

The multi-mass artifact was canonicalized from schema-v1 raw output whose
SHA-256 is
`f51c2f0733d3f558daf6a4d6c50efa16fdcea392156bcca6f51aa26002be81d8`.
The echolocation development scan predates schema v1; the archiver first
validated its exact grid and tuples, then wrapped the unchanged records with
the legacy source SHA-256
`eafc3ae9b74e33b278543bd76203979a7ab3e8d9b9f36cb02a150e63c158e7d0`.
Those provenance hashes identify the gitignored raw inputs without presenting
them as tracked certification artifacts.

## Two-hidden-mass recovery

The initial grid used the provisional tolerance only to characterize the error
distribution. That development evidence informed the simple utility gate of
2.5 for every position coordinate and 0.012 for every mass. After fixing that
gate uniformly, the complete 27-cell development grid was rerun and the final
controls were selected by the predeclared rule. These are all 27 final cell
summaries:

```text
ess=0.70 steps=1 scale=1.50: 9/12, median normalized error=0.295, median forward evaluations=738392
ess=0.70 steps=1 scale=2.38: 10/12, median normalized error=0.328, median forward evaluations=733646
ess=0.70 steps=1 scale=3.00: 9/12, median normalized error=0.233, median forward evaluations=722324
ess=0.70 steps=2 scale=1.50: 10/12, median normalized error=0.283, median forward evaluations=868170
ess=0.70 steps=2 scale=2.38: 10/12, median normalized error=0.289, median forward evaluations=844381
ess=0.70 steps=2 scale=3.00: 11/12, median normalized error=0.256, median forward evaluations=815920
ess=0.70 steps=4 scale=1.50: 10/12, median normalized error=0.290, median forward evaluations=1118416
ess=0.70 steps=4 scale=2.38: 10/12, median normalized error=0.266, median forward evaluations=1054872
ess=0.70 steps=4 scale=3.00: 10/12, median normalized error=0.292, median forward evaluations=1002633
ess=0.80 steps=1 scale=1.50: 10/12, median normalized error=0.283, median forward evaluations=942818
ess=0.80 steps=1 scale=2.38: 11/12, median normalized error=0.271, median forward evaluations=942038
ess=0.80 steps=1 scale=3.00: 9/12, median normalized error=0.274, median forward evaluations=900904
ess=0.80 steps=2 scale=1.50: 10/12, median normalized error=0.284, median forward evaluations=1111952
ess=0.80 steps=2 scale=2.38: 10/12, median normalized error=0.281, median forward evaluations=1075168
ess=0.80 steps=2 scale=3.00: 10/12, median normalized error=0.262, median forward evaluations=1027056
ess=0.80 steps=4 scale=1.50: 10/12, median normalized error=0.279, median forward evaluations=1441490
ess=0.80 steps=4 scale=2.38: 10/12, median normalized error=0.272, median forward evaluations=1377878
ess=0.80 steps=4 scale=3.00: 10/12, median normalized error=0.276, median forward evaluations=1296720
ess=0.90 steps=1 scale=1.50: 10/12, median normalized error=0.251, median forward evaluations=1334115
ess=0.90 steps=1 scale=2.38: 10/12, median normalized error=0.278, median forward evaluations=1318082
ess=0.90 steps=1 scale=3.00: 11/12, median normalized error=0.256, median forward evaluations=1294084
ess=0.90 steps=2 scale=1.50: 10/12, median normalized error=0.254, median forward evaluations=1632352
ess=0.90 steps=2 scale=2.38: 10/12, median normalized error=0.280, median forward evaluations=1576642
ess=0.90 steps=2 scale=3.00: 10/12, median normalized error=0.279, median forward evaluations=1528888
ess=0.90 steps=4 scale=1.50: 10/12, median normalized error=0.270, median forward evaluations=2218813
ess=0.90 steps=4 scale=2.38: 10/12, median normalized error=0.270, median forward evaluations=2071370
ess=0.90 steps=4 scale=3.00: 10/12, median normalized error=0.284, median forward evaluations=2002675
```

The selected cell is `ess_target=0.70`, `rejuvenation_steps=2`, and
`proposal_scale=3.00`. Three cells reach the best pass count, 11/12. The two
best printed median normalized errors both round to 0.256; ranking on the
unrounded values selected this cell. Its much lower median forward-evaluation
count than the other printed-0.256 cell would also favor it at the next
tie-break.

The frozen recovery gate uses the same absolute tolerance for all four
position coordinates (2.5) and the same absolute tolerance for both masses
(0.012). The position threshold is 15.625% of the 16-unit prior width, so it
allows material localization error without accepting arbitrary prior-scale
answers. The mass threshold is 24% of the 0.050 true mass and 40% of the 0.030
true mass. On the selected development cell this gate passes 11/12 seeds.
This is deliberately described as a recovery/regression gate rather than a
population success probability.

The selected controls, 11/12 gate result, and every rounded summary line above
recompute directly from the linked 324-record artifact. The sole failing seed
for the selected cell is seed 6.

## Echolocation range study

These are all 27 cell summaries derived from the development study JSON. Each
cell contains six ranges by twelve seeds, hence 72 runs.

```text
ess=0.70 steps=1 scale=1.50: 46/72, median normalized error=0.326, median forward evaluations=981911
ess=0.70 steps=1 scale=2.38: 48/72, median normalized error=0.338, median forward evaluations=975148
ess=0.70 steps=1 scale=3.00: 48/72, median normalized error=0.342, median forward evaluations=970445
ess=0.70 steps=2 scale=1.50: 46/72, median normalized error=0.339, median forward evaluations=1151226
ess=0.70 steps=2 scale=2.38: 48/72, median normalized error=0.333, median forward evaluations=1114244
ess=0.70 steps=2 scale=3.00: 46/72, median normalized error=0.337, median forward evaluations=1098948
ess=0.70 steps=4 scale=1.50: 48/72, median normalized error=0.337, median forward evaluations=1458246
ess=0.70 steps=4 scale=2.38: 48/72, median normalized error=0.330, median forward evaluations=1421379
ess=0.70 steps=4 scale=3.00: 48/72, median normalized error=0.336, median forward evaluations=1371573
ess=0.80 steps=1 scale=1.50: 45/72, median normalized error=0.335, median forward evaluations=1197796
ess=0.80 steps=1 scale=2.38: 47/72, median normalized error=0.359, median forward evaluations=1187180
ess=0.80 steps=1 scale=3.00: 48/72, median normalized error=0.308, median forward evaluations=1189963
ess=0.80 steps=2 scale=1.50: 48/72, median normalized error=0.338, median forward evaluations=1417957
ess=0.80 steps=2 scale=2.38: 47/72, median normalized error=0.331, median forward evaluations=1376401
ess=0.80 steps=2 scale=3.00: 48/72, median normalized error=0.336, median forward evaluations=1367350
ess=0.80 steps=4 scale=1.50: 48/72, median normalized error=0.334, median forward evaluations=1863888
ess=0.80 steps=4 scale=2.38: 47/72, median normalized error=0.334, median forward evaluations=1793731
ess=0.80 steps=4 scale=3.00: 47/72, median normalized error=0.334, median forward evaluations=1748669
ess=0.90 steps=1 scale=1.50: 49/72, median normalized error=0.333, median forward evaluations=1708242
ess=0.90 steps=1 scale=2.38: 48/72, median normalized error=0.332, median forward evaluations=1705052
ess=0.90 steps=1 scale=3.00: 48/72, median normalized error=0.322, median forward evaluations=1659200
ess=0.90 steps=2 scale=1.50: 48/72, median normalized error=0.332, median forward evaluations=2100686
ess=0.90 steps=2 scale=2.38: 47/72, median normalized error=0.330, median forward evaluations=2034848
ess=0.90 steps=2 scale=3.00: 48/72, median normalized error=0.331, median forward evaluations=2006070
ess=0.90 steps=4 scale=1.50: 49/72, median normalized error=0.337, median forward evaluations=2848832
ess=0.90 steps=4 scale=2.38: 47/72, median normalized error=0.335, median forward evaluations=2745403
ess=0.90 steps=4 scale=3.00: 48/72, median normalized error=0.336, median forward evaluations=2658560
```

The selected cell is `ess_target=0.90`, `rejuvenation_steps=1`, and
`proposal_scale=1.50`. It ties the best pass count at 49/72 and wins the next
rank criterion, median normalized error (0.333 versus 0.337 for the other
49-pass cell).

The position tolerance remains 1.0 and the mass tolerance remains 0.04. At
the close range, the selected cell passes 12/12; its maximum position and mass
errors are 0.0633 and 0.00309. Pass counts from close to far are
12, 12, 12, 8, 4, and 1. Every far-range truth is covered by its reported
three-standard-deviation interval (12/12), and the far/close median position
standard-deviation ratio is 66.226. The frozen honest-uncertainty factor is a
conservative 20.0, well below that observed development ratio.

The selected controls, range-by-range gates, coverage count, uncertainty
ratio, and every rounded summary line above recompute directly from the linked
1,944-record artifact.

## One-shot certification

After freeze commit `a1b016b`, the reserved seeds 400–411 were executed exactly
once with the frozen single cells and gates. The tracked
[multi-mass certification artifact](calibration/multi_mass_2d_certification.json)
and [echolocation certification artifact](calibration/echolocation_range_certification.json)
contain all unrounded schema-v1 records. Their source SHA-256 hashes are,
respectively,
`2fb47f532ac0429f83f71eaa49ba23763bcfb230949168042139257c9b200184`
and
`a4c6b1b7c3c2fce273aaa19f01289e6ca34de2ed8d633f827e798e1af4f47941`.
No controls, tolerances, or gates were retuned after viewing these results.

The multi-mass cell passed 12/12 certification cases. Its median normalized
error was 0.24165156854861075 and its median forward-model evaluation count
was 833513. The deterministic replay gate remains the predeclared threshold
of at least 10/12, rather than being tightened to the observed 12/12.

The echolocation cell passed 46/72 cases overall. From close to far, the pass
counts were 12, 12, 11, 8, 3, 0. At the close range, the median position
standard deviation was 0.041934482917603016; maximum position and mass errors
were 0.06890007743851188 and 0.003180807096232516. At the far range, the
median position standard deviation was 2.789667608020031, giving a far/close
ratio of 66.5244308246615. The deterministic replay gates remain at least
10/12 close passes and a ratio of at least the frozen factor 20.0. Far-range
three-standard-deviation coverage was 12/12 in certification and is reported
as a diagnostic, not promoted after inspection into a new gate.

These finite fixed-seed results are regression and calibration evidence for
the named simulated cases, not population reliability estimates. Both
deterministic slow replay tests subsequently passed without changing the
frozen gates.

## Corrected generated assets

All seven packaged default demos were run from the certified `a1b016b` freeze,
and the exact outputs were copied into both generated-asset trees. The
echolocation range figure was rendered from the already-certified block-400
JSON, without another inference run. These hashes identify the released bytes;
they are provenance pins, not a claim that plotting and animation encoders are
byte-deterministic across tool versions.

| Asset | SHA-256 |
|---|---|
| `demo_1d.gif` | `417ae3523e95e85e91feca7f67e2a8bc7347006883ea27b1101f0cc164477483` |
| `demo_2d.gif` | `eb581e327d950d7bedce87f28c383c35aa38e99792291d5ace2291ae090e1a3b` |
| `demo_multi_mass.gif` | `bb2a3a2a74270114a516133cf2b8ef9484eb1bbc2d40b542d42d54013e30f8ad` |
| `demo_multi_mass_2d.gif` | `ecce3ca3187010d73f5ea651826f43a1e2057ad24a45f8ad040d14979cdaa967` |
| `demo_model_comparison.gif` | `27af99630edf6578e635d60cfd2f085442f8429ca008ce25afc935b823c6c9f2` |
| `demo_density.png` | `a7a6e6e9628640ac08e89152b8471d37db981c264380ba567634032a0f59dc7a` |
| `demo_echolocation_3d.gif` | `5df30d7c052c3e366216866c9b0ecc991cc777467515dd1c7699de743413d34f` |
| `echolocation_range_study.json` | `a4c6b1b7c3c2fce273aaa19f01289e6ca34de2ed8d633f827e798e1af4f47941` |
| `echolocation_range_study.png` | `6d62a1c88c78837299bb434e66574d6930eef24c6cf9805e42ec66e141dd86bc` |

The default model-comparison asset is also a useful caution against narrating
one finite evidence estimate as a guaranteed recovery. With K=2 truth, its
80-observation, 2,000-particle, seed-42 run ended at K=2: 0.3990 and K=3:
0.6010. It mildly prefers K=3 in this realization. The site reports that
outcome directly and contrasts it with its smaller 25-observation,
400-particle executed example rather than treating either as a universal
model-selection verdict.
