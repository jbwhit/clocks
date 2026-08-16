# Development calibration freeze — 2026-08-16

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
and gates were selected and frozen. Certification and evidence-asset
regeneration remain pending.

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
