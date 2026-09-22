# LIBERO-90 Five-Task SIMPLER-Config Diagnosis

## Scope and data integrity

- 250 paired cases / 500 episodes; all completed; 0 runner errors.
- Overall: Vanilla 184/250, Matched 182/250, Net -2, exact McNemar p=0.902159.
- All matched episodes share code commit `72bd16f0...`, checkpoint revision `794ef81b...`, and config hash `4c2aa086...`.
- The frozen method used L11, full instruction query, all-head equal mean, K=8 seed=0 n_init=10, source+target union, lambda=0.5, harmonic beta=0, shared clean prefix, no sampling.

## Budget and mask audit

- Stepwise matched budget: mean=38.942, median=41.000, P10=11.000, P90=60.000, min=1, max=112.
- Episode-mean budget: mean=37.129, median=42.498, P10=21.391, P90=49.088. The earlier 37.13 was an episode-mean statistic.
- Empty budget steps: 0. No SIMPLER-config clip/floor/ceiling was applied; clip_triggered=not applicable.
- Steps where multiple entities matched the same deduplicated cluster: 15783; episodes affected: 188/250.

## Failure-stage classification (deterministic replay)

Failure classes are based on simulator target/object state, not on final-frame appearance alone.

| Task | Group | n | Approach/grasp fail | Wrong object | Target not lifted | Dropped/lost | Placement/subtask |
|---|---|---:|---:|---:|---:|---:|---:|
| KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it | harm | 2 | 1 | 0 | 1 | 0 | 0 |
| KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it | rescue | 7 | 5 | 0 | 1 | 1 | 0 |
| KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet | harm | 1 | 1 | 0 | 0 | 0 | 0 |
| LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket | harm | 12 | 9 | 1 | 1 | 1 | 0 |
| LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket | rescue | 9 | 8 | 0 | 1 | 0 | 0 |
| LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate | harm | 10 | 0 | 0 | 0 | 6 | 4 |
| LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate | rescue | 9 | 5 | 0 | 0 | 4 | 0 |
| STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy | harm | 9 | 0 | 0 | 0 | 6 | 3 |
| STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy | rescue | 7 | 0 | 0 | 0 | 2 | 5 |

### Harm: what new failures were introduced?

- `dropped_or_lost_target`: 13/34.
- `failed_to_approach_or_grasp_target`: 11/34.
- `placement_or_subtask_failure`: 7/34.
- `target_not_lifted_after_grasp`: 2/34.
- `wrong_object_grasp`: 1/34.

### Rescue: what did the intervention repair?

- `failed_to_approach_or_grasp_target`: 18/32.
- `dropped_or_lost_target`: 7/32.
- `placement_or_subtask_failure`: 5/32.
- `target_not_lifted_after_grasp`: 2/32.

## Action and intervention differences

The released compact traces do not store the original clean action numerically; they store the final guided action and `guided_changed_dims`. Therefore the complete table reports how many action dimensions were changed, but not signed clean-vs-guided differences for every case. The selected-state replay below provides exact signed comparisons for representative cases.

| Metric | Rescue median | Harm median | Mann-Whitney p (R vs H) |
|---|---:|---:|---:|
| mean_m | 44.67 | 46.35 | 0.0777 |
| max_budget_jump | 46 | 51 | 0.1542 |
| mean_perturbation | 0.3599 | 0.3652 | 0.6673 |
| max_perturbation | 0.4794 | 0.4929 | 0.09406 |
| guided_change_fraction | 0.897 | 0.94 | 3.297e-05 |
| mean_guided_changed_dims | 1.932 | 2.235 | 0.0001578 |
| mean_mask_components | 5.325 | 6.324 | 0.005693 |

Representative exact action comparisons (same matched state):

| Case | Group | Step | m | changed dims | translation L2 | rotation L2 | gripper delta |
|---|---|---:|---:|---:|---:|---:|---:|
| task03__init002 | rescue | 0 | 54 | 2 | 0.0232 | 0.0287 | 0.0000 |
| task03__init009 | harm | 1 | 6 | 1 | 0.0072 | 0.0000 | 0.0000 |
| task10__init014 | harm | 1 | 15 | 3 | 0.0507 | 0.0222 | 0.0000 |
| task49__init007 | harm | 0 | 33 | 1 | 0.1077 | 0.0000 | 0.0000 |
| task49__init015 | rescue | 0 | 33 | 3 | 0.0755 | 0.0225 | 0.0000 |
| task72__init002 | rescue | 0 | 39 | 2 | 0.1568 | 0.0024 | 0.0000 |
| task73__init008 | rescue | 0 | 51 | 2 | 0.0000 | 0.0467 | 0.0000 |
| task73__init033 | harm | 0 | 12 | 2 | 0.0860 | 0.0000 | 0.0000 |

## Mask and reconstruction observations

- Harm has a higher median number of changed action dimensions (2.235) than Rescue (1.932); the difference is statistically strong in this subset (p=0.000158).
- Harm also has more fragmented masks on average (6.324 components vs 5.325; p=0.00569).
- Mixed feature-perturbation magnitudes are similar (median 0.365 Harm vs 0.360 Rescue, p=0.667); perturbation magnitude alone does not separate the groups.

Representative panels are in `diagnostic_panels/`. Selected cases and video time points:

- `task03__init002` (rescue): first major divergence t≈0.03333333333333333s; panel `diagnostic_panels/task03__init002.png`; action/mask panel `diagnostic_panels/task03__init002__action_mask.png`.
- `task03__init009` (harm): first major divergence t≈0.36666666666666664s; panel `diagnostic_panels/task03__init009.png`; action/mask panel `diagnostic_panels/task03__init009__action_mask.png`.
- `task10__init014` (harm): first major divergence t≈0.03333333333333333s; panel `diagnostic_panels/task10__init014.png`; action/mask panel `diagnostic_panels/task10__init014__action_mask.png`.
- `task49__init007` (harm): first major divergence t≈0.0s; panel `diagnostic_panels/task49__init007.png`; action/mask panel `diagnostic_panels/task49__init007__action_mask.png`.
- `task49__init015` (rescue): first major divergence t≈0.0s; panel `diagnostic_panels/task49__init015.png`; action/mask panel `diagnostic_panels/task49__init015__action_mask.png`.
- `task72__init002` (rescue): first major divergence t≈0.0s; panel `diagnostic_panels/task72__init002.png`; action/mask panel `diagnostic_panels/task72__init002__action_mask.png`.
- `task73__init008` (rescue): first major divergence t≈0.03333333333333333s; panel `diagnostic_panels/task73__init008.png`; action/mask panel `diagnostic_panels/task73__init008__action_mask.png`.
- `task73__init033` (harm): first major divergence t≈0.0s; panel `diagnostic_panels/task73__init033.png`; action/mask panel `diagnostic_panels/task73__init033__action_mask.png`.

## Candidate causes (evidence-ranked)

### Candidate 1: Directionally over-strong intervention on some action dimensions

**支持证据**：Harm 的 guided-changed dimension count and fraction are higher than Rescue; 13/34 Harm end as dropped/lost target and 7/34 as placement/subtask failure after target engagement. Representative task49 and task73 Harm cases show guided translation changes at early steps.

**反证或限制**：Perturbation magnitude is not significantly higher; some Rescue and both-success cases also change multiple dimensions. The compact traces do not contain signed clean/guided actions for all steps.

**当前判断**：较强线索，但仍是相关性证据，不是已证明因果。

### Candidate 2: Mask/cardinality may mix target evidence with robot, destination, or distractor evidence

**支持证据**：Harm masks are more fragmented than Rescue masks; task49 Harm panel includes robot/base/basket regions; task03 Harm panel includes cabinet and non-target object regions; matched budget is moderately larger in Harm.

**反证或限制**：Target-focused masks also appear in Harm (e.g., task10); Rescue panels also cover robot/background regions. Mask overlap with semantics is not directly measured for every state.

**当前判断**：弱到中等线索，任务依赖明显。

### Candidate 3: The main new harm is post-grasp instability, not simply wrong-target selection

**支持证据**：Only 1/34 Harm is wrong-object grasp; 13/34 are dropped/lost target and 7/34 are placement/subtask failure. Rescue mostly repairs approach/grasp failures of Vanilla (18/32).

**反证或限制**：The replay classification uses simulator object/grasp state and does not identify the exact action dimension that caused the loss; downstream dynamics may amplify a small early change.

**当前判断**：最强的行为层结论，说明 Harm 主要发生在“抓住之后”和后续控制，而不是单纯看错目标。

## Recommended minimal follow-up (not executed)

The smallest discriminating test is to replay Harm states at the first post-grasp divergence and compare clean action, guided action, reconstructed-branch action, and simulator target/gripper state without changing the method. This would distinguish action-guidance drift from mask/reconstruction error using the same states already collected.

Analysis stopped here; no new rollout or parameter search was run.
