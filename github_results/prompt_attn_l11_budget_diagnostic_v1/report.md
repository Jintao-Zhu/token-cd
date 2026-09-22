# L11 Attention Position and Budget Diagnostic

This is an offline diagnostic over 60 fixed states (4 tasks × 5 episodes × early/middle/late). No newly computed policy action controlled an environment and no new success rate was produced.

## Aggregate facts

- Four corner tokens hold 0.54% of normalized visual attention versus 1.56% of positions.
- The outer ring holds 10.39% versus 23.44% of positions.
- Outer-ring tokens account for 9.27% of all Top-16 occurrences and 12.29% of Top-32 occurrences.
- At Top-p=.8, the outer ring consumes on average 8.43% of the cumulative threshold budget.
- Hard outer-ring suppression changes Top-p=.8 selection by adding 0.4 and dropping 6.6 tokens per state; mean mask Jaccard is 0.808.
- The most persistent edge token appears in Top-16 for 33.3% of states; the most persistent interior token appears for 43.3%.
- Early-to-late Top-16 Jaccard within an episode is 0.193.
- In 9 explicit same-image target switches, target-follow contrast is positive in 88.9% and the mean Top-m Jaccard before/after switching target is 0.636.

## Judgment

There is no broad outer-boundary attention sink: corners and the full outer ring are strongly under-represented by attention mass and by Top-16/Top-32 occurrence. The outer ring does not dominate the cumulative budget.
There are isolated task-specific top-edge hotspots: a top-edge token reaches Top-16 in 60% of close_drawer and pick_coke_can states. This is a local positional bias, not whole-edge enrichment; interior hotspots are more persistent overall.
The target-switch controls show that L11 is usually instruction-responsive rather than spatially fixed. Therefore these data do not support hard suppression of the entire outer ring as the next default. A later closed-loop test, if desired, should isolate a small recurrent-position penalty from broad edge removal.

## Interpretation boundary

Attention enrichment and budget occupancy do not establish that suppressing any position improves closed-loop success; that requires a separately paired intervention experiment.
