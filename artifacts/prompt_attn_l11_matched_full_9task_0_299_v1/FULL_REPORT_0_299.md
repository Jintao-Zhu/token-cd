# L11-Matched full canonical evaluation: 9 tasks x 300 seeds

All 2,700 episodes use the same canonical snapshot and pass the canonical/state/RGB hash checks.

| Task | Vanilla | SHR | L11-Matched | L11 vs Vanilla net | L11 vs SHR net |
|---|---:|---:|---:|---:|---:|
| google_robot_open_drawer | 71/300 (23.7%) | 138/300 (46.0%) | 144/300 (48.0%) | +73 | +6 |
| google_robot_close_drawer | 154/300 (51.3%) | 209/300 (69.7%) | 230/300 (76.7%) | +76 | +21 |
| google_robot_pick_coke_can | 83/300 (27.7%) | 97/300 (32.3%) | 119/300 (39.7%) | +36 | +22 |
| google_robot_move_near | 184/300 (61.3%) | 197/300 (65.7%) | 182/300 (60.7%) | -2 | -15 |
| google_robot_place_apple_in_closed_top_drawer | 0/300 (0.0%) | 0/300 (0.0%) | 1/300 (0.3%) | +1 | +1 |
| widowx_carrot_on_plate | 14/300 (4.7%) | 4/300 (1.3%) | 0/300 (0.0%) | -14 | -4 |
| widowx_put_eggplant_in_basket | 1/300 (0.3%) | 0/300 (0.0%) | 12/300 (4.0%) | +11 | +12 |
| widowx_spoon_on_towel | 0/300 (0.0%) | 1/300 (0.3%) | 4/300 (1.3%) | +4 | +3 |
| widowx_stack_cube | 0/300 (0.0%) | 0/300 (0.0%) | 8/300 (2.7%) | +8 | +8 |
| overall | 507/2700 (18.8%) | 646/2700 (23.9%) | 700/2700 (25.9%) | +193 | +54 |

## Paired overall comparisons

| Comparison | Rescue | Harm | Net | exact p |
|---|---:|---:|---:|---:|
| L11 vs Vanilla | 302 | 109 | +193 | 4.63273e-22 |
| L11 vs SHR | 210 | 156 | +54 | 0.00552703 |
