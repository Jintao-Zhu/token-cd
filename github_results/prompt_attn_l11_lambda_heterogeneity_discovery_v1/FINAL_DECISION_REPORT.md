# L11 Lambda Heterogeneity Final Decision

## Execution audit

- Discovery scope: 4 tasks x 100 seeds with lambda in {0, 0.25, 0.5}.
- New rollout summaries: 500/500.
- Technical audit: 500/500 passed.
- Canonical snapshot, initial state, and initial RGB pairing: passed for all 400 task-seed groups.
- Grouped OOF: all three lambda rows for each task-seed stayed in the same fold.
- Router decision: fallback to lambda=0.5 unless the best weaker arm exceeds q(lambda=0.5) by 0.05.
- Frozen confirmation and episode-start closed loop were not launched because the locked Discovery Go gate failed.

## Fixed-arm and oracle results

| Task | lambda=0 | lambda=0.25 | lambda=0.5 | Oracle | Oracle gap vs 0.5 |
|---|---:|---:|---:|---:|---:|
| open_drawer | 27/100 | 50/100 | 50/100 | 69/100 | +19 |
| close_drawer | 56/100 | 71/100 | 82/100 | 92/100 | +10 |
| pick_coke_can | 23/100 | 28/100 | 44/100 | 62/100 | +18 |
| move_near | 53/100 | 56/100 | 62/100 | 76/100 | +14 |
| **Overall** | **159/400** | **205/400** | **238/400** | **299/400** | **+61 (+15.3pp)** |

## Conservative grouped-OOF router

| Task | Router | Fixed 0.5 | Oracle | Rescue | Harm | Net | Exact p |
|---|---:|---:|---:|---:|---:|---:|---:|
| open_drawer | 52/100 | 50/100 | 69/100 | 9 | 7 | +2 | 0.8036 |
| close_drawer | 82/100 | 82/100 | 92/100 | 3 | 3 | 0 | 1.0000 |
| pick_coke_can | 42/100 | 44/100 | 62/100 | 0 | 2 | -2 | 0.5000 |
| move_near | 66/100 | 62/100 | 76/100 | 8 | 4 | +4 | 0.3877 |
| **Overall** | **242/400** | **238/400** | **299/400** | **20** | **16** | **+4** | **0.6177** |

- Row-level success AUC: 0.824.
- Balanced accuracy: 0.736.
- Oracle recovery: 4/61 = 6.6%.
- Switches away from lambda=0.5: 87/400 episodes.
- Locked Go requirement: oracle recovery greater than 25%.
- Decision: **STOP**.

## Interpretation

The experiment strongly confirms episode-level lambda heterogeneity: the three-arm oracle improves over fixed lambda=0.5 by 61/400 successes. However, high row-level success AUC does not translate into reliable counterfactual arm selection. The conservative router makes 87 switches but produces only four net successes, with 20 rescues and 16 harms. Initial task, prompt, and action-state features therefore do not recover enough of the available oracle gap to justify frozen confirmation or GPU closed-loop evaluation under the pre-registered gate.

The positive scientific result is the existence of substantial lambda heterogeneity. The negative method result is that the current episode-start initial-state predictor is not selective enough to exploit it safely.
