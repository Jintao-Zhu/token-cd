# VLA-Pruner P75: official-code L15 vs paper-v5 L16-31

Completed: 800/800 arm-episodes; hash mismatches=0; technical failures=0

| Task | N | Vanilla | Code-L15 P75 | Paper-L16-31 P75 |
|---|---:|---:|---:|---:|
| google_robot_open_drawer | 100 | 27/100 (27.0%) | 17/100 (17.0%) | 22/100 (22.0%) |
| google_robot_close_drawer | 100 | 56/100 (56.0%) | 43/100 (43.0%) | 44/100 (44.0%) |
| google_robot_pick_coke_can | 100 | 23/100 (23.0%) | 27/100 (27.0%) | 24/100 (24.0%) |
| google_robot_move_near | 100 | 52/100 (52.0%) | 59/100 (59.0%) | 56/100 (56.0%) |
| OVERALL | 400 | 158/400 (39.5%) | 146/400 (36.5%) | 146/400 (36.5%) |

## Paired comparisons

- google_robot_open_drawer: Paper vs Code rescue=12, harm=7, net=5, exact p=0.3593
- google_robot_close_drawer: Paper vs Code rescue=16, harm=15, net=1, exact p=1
- google_robot_pick_coke_can: Paper vs Code rescue=12, harm=15, net=-3, exact p=0.7011
- google_robot_move_near: Paper vs Code rescue=6, harm=9, net=-3, exact p=0.6072
- OVERALL: Paper vs Code rescue=46, harm=46, net=0, exact p=1
