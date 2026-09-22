# VLA-Pruner reproduction — FORMAL 1200 closed-loop report

protocol: VLA_PRUNER_OPENVLA_REPRODUCTION_V1 / stage formal_1200 (frozen config in CONFIG_LOCK.json)
episodes: 1200/1200 technical_pass = True

## 1. Overall success rate
| arm | success |
|---|---|
| vanilla | 158/400 (39.5%) |
| vla_pruner_prune25 | 161/400 (40.2%) |
| vla_pruner_prune50 | 167/400 (41.8%) |

## 2. Per-task success
| task | vanilla | prune25 | prune50 |
|---|---|---|---|
| google_robot_open_drawer | 27/100 (27.0%) | 31/100 (31.0%) | 23/100 (23.0%) |
| google_robot_close_drawer | 56/100 (56.0%) | 47/100 (47.0%) | 59/100 (59.0%) |
| google_robot_pick_coke_can | 23/100 (23.0%) | 25/100 (25.0%) | 31/100 (31.0%) |
| google_robot_move_near | 52/100 (52.0%) | 58/100 (58.0%) | 54/100 (54.0%) |

## 3. Paired rescue / harm + McNemar (vs current-harness vanilla)
| arm | rescue | harm | unchanged | McNemar p |
|---|---|---|---|---|
| vla_pruner_prune25 | 54 | 51 | 295 | 0.8453 |
| vla_pruner_prune50 | 56 | 47 | 297 | 0.4305 |

### per-task discordant pairs
- google_robot_open_drawer: vla_pruner_prune25: rescue 13 / harm 9 / p=0.522; vla_pruner_prune50: rescue 15 / harm 19 / p=0.607
- google_robot_close_drawer: vla_pruner_prune25: rescue 13 / harm 22 / p=0.176; vla_pruner_prune50: rescue 9 / harm 6 / p=0.606
- google_robot_pick_coke_can: vla_pruner_prune25: rescue 13 / harm 11 / p=0.838; vla_pruner_prune50: rescue 17 / harm 9 / p=0.170
- google_robot_move_near: vla_pruner_prune25: rescue 15 / harm 9 / p=0.307; vla_pruner_prune50: rescue 15 / harm 13 / p=0.850

## 4. Latency (measured on GPU2/3, this harness)
| arm | own inference ms/env-step | own-inference speedup | env.step ms | projected end-to-end speedup (diagnostic ref excluded) | end-to-end with diagnostic ref (paired cost) |
|---|---|---|---|---|---|
| vanilla | 241.6 | 1.000 | 286.4 | 1.000 | 1.000 |
| vla_pruner_prune25 | 259.1 | 0.932 | 272.1 | 0.998 | 0.698 |
| vla_pruner_prune50 | 245.0 | 0.986 | 278.7 | 1.012 | 0.709 |

`own inference` = the arm's own 7-token generate per env step, incl. VLA-Pruner's attention/history bookkeeping; `env.step ms` = simulator; `projected end-to-end` = episode wall-clock after subtracting the diagnostic vanilla reference forward (which exists only to measure same-obs action deltas, not part of VLA-Pruner itself).

## 4b. Action-change / temporal behaviour
| arm | first divergence step (mean / median) | flip steps (mean) | mean raw L1 | mean overlap active |
|---|---|---|---|---|
| vla_pruner_prune25 | 5.28 / 3 | 43.2 | 0.141 | 165.9 |
| vla_pruner_prune50 | 4.76 / 3 | 55.1 | 0.209 | 102.9 |

## 5. Temporal / pruning sanity
| arm | mean prune activation steps | mean kept (active) | mean overlap active |
|---|---|---|---|
| vla_pruner_prune25 | 93.50 | 192.00 | 165.92 |
| vla_pruner_prune50 | 93.50 | 128.00 | 102.94 |

## 6. Conclusions (pre-registered interpretation)
- **Verdict: C: SR ≈ vanilla but Rescue/Harm are large and offsetting (pruning clearly changes the policy; no net gain, no measured speedup).**

## 7. Raw artifacts
- `episode_results.csv`, `paired_results.csv`, `task_summary.csv`, `latency_summary.csv`, `temporal_summary.csv`, `technical_audit.json`
