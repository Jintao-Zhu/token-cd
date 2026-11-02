# K=10 Matched at lambda 0.5 and 1.0: paired results against the grid reference

Generated 2026-10-16T15:44:57+0800. Every row is paired on identical (task, seed).

## lambda = 0.5

coverage per task: {'close_drawer': 100, 'open_drawer': 100, 'pick_coke_can': 100, 'move_near': 100} (expected 100 each, complete: **True**)

K=10 arm: 198/400 (49.5%)

| contrast | pairs | K=10 | reference | rescue | harm | diff (pp) | bootstrap 95% CI | McNemar p | Holm p |
|---|---|---|---|---|---|---|---|---|---|
| vs matched | 400 | 198 | 205 | 47 | 54 | -1.75 | [-8.25, +4.75] | 0.5507 | 0.5507 |
| vs fixed016 | 400 | 198 | 165 | 80 | 47 | +8.25 | [+1.75, +14.75] | 0.004331 | 0.008662 |

per task (versus matched at the same lambda):

| task | n | K=10 | matched | net | m_t K=10 | m_t matched |
|---|---|---|---|---|---|---|
| close_drawer | 100 | 77 | 71 | +6 | 24.9 | 33.9 |
| open_drawer | 100 | 44 | 50 | -6 | 27.0 | 36.8 |
| pick_coke_can | 100 | 32 | 32 | +0 | 21.3 | 28.8 |
| move_near | 100 | 45 | 52 | -7 | 26.9 | 36.9 |

## lambda = 1.0

coverage per task: {'close_drawer': 100, 'open_drawer': 100, 'pick_coke_can': 100, 'move_near': 100} (expected 100 each, complete: **True**)

K=10 arm: 186/400 (46.5%)

| contrast | pairs | K=10 | reference | rescue | harm | diff (pp) | bootstrap 95% CI | McNemar p | Holm p |
|---|---|---|---|---|---|---|---|---|---|
| vs matched | 400 | 186 | 201 | 53 | 68 | -3.75 | [-10.00, +2.50] | 0.2029 | 0.2029 |
| vs fixed016 | 400 | 186 | 206 | 56 | 76 | -5.00 | [-11.25, +1.50] | 0.09781 | 0.1956 |

per task (versus matched at the same lambda):

| task | n | K=10 | matched | net | m_t K=10 | m_t matched |
|---|---|---|---|---|---|---|
| close_drawer | 100 | 79 | 84 | -5 | 25.0 | 35.2 |
| open_drawer | 100 | 42 | 40 | +2 | 27.0 | 37.0 |
| pick_coke_can | 100 | 26 | 29 | -3 | 20.8 | 29.3 |
| move_near | 100 | 39 | 48 | -9 | 26.3 | 33.3 |

## Caveats

- K=10 is evaluated on the same seeds 200-299 as the reference, and was chosen after seeing the probe: this is an exploratory hyper-parameter check, not a pre-registered confirmatory test.
- K changes the matched union size, so this run changes the budget as well as the clustering; the measured m_t columns quantify how much budget was given away.
- Non-significance is not equivalence: every contrast is reported with its observed difference and a task-stratified paired bootstrap interval.
- The fixed016 contrast is a secondary, descriptive reference for the under-budget regime; the primary contrast is against matched K8 at the same lambda.

Figure: `docs/figures/openvla_k10_2lambda.png`.
