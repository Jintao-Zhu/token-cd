# L11-Matched full run: live results

Updated: **2026-09-13T06:20:06+0800**
Completed: **1800/1800**
L11 success: **457**

| Task | n | Vanilla | SHR | L11 | Net vs Vanilla | Net vs SHR |
|---|---:|---:|---:|---:|---:|---:|
| google_robot_close_drawer | 200 | 98 | 131 | 148 | +50 | +17 |
| google_robot_move_near | 200 | 122 | 130 | 120 | -2 | -10 |
| google_robot_open_drawer | 200 | 44 | 92 | 94 | +50 | +2 |
| google_robot_pick_coke_can | 200 | 60 | 56 | 75 | +15 | +19 |
| google_robot_place_apple_in_closed_top_drawer | 200 | 0 | 0 | 0 | +0 | +0 |
| widowx_carrot_on_plate | 200 | 9 | 1 | 0 | -9 | -1 |
| widowx_put_eggplant_in_basket | 200 | 1 | 0 | 9 | +8 | +9 |
| widowx_spoon_on_towel | 200 | 0 | 1 | 3 | +3 | +2 |
| widowx_stack_cube | 200 | 0 | 0 | 8 | +8 | +8 |

## Paired same-seed comparisons

| Task | Baseline | Rescue | Harm | Net | exact p |
|---|---|---:|---:|---:|---:|
| google_robot_close_drawer | Vanilla | 59 | 9 | +50 | 3.91412e-10 |
| google_robot_close_drawer | SHR | 37 | 20 | +17 | 0.033144 |
| google_robot_move_near | Vanilla | 19 | 21 | -2 | 0.874629 |
| google_robot_move_near | SHR | 14 | 24 | -10 | 0.143307 |
| google_robot_open_drawer | Vanilla | 62 | 12 | +50 | 2.85555e-09 |
| google_robot_open_drawer | SHR | 35 | 33 | +2 | 0.903597 |
| google_robot_pick_coke_can | Vanilla | 43 | 28 | +15 | 0.0959236 |
| google_robot_pick_coke_can | SHR | 39 | 20 | +19 | 0.0183371 |
| google_robot_place_apple_in_closed_top_drawer | Vanilla | 0 | 0 | +0 | 1 |
| google_robot_place_apple_in_closed_top_drawer | SHR | 0 | 0 | +0 | 1 |
| widowx_carrot_on_plate | Vanilla | 0 | 9 | -9 | 0.00390625 |
| widowx_carrot_on_plate | SHR | 0 | 1 | -1 | 1 |
| widowx_put_eggplant_in_basket | Vanilla | 9 | 1 | +8 | 0.0214844 |
| widowx_put_eggplant_in_basket | SHR | 9 | 0 | +9 | 0.00390625 |
| widowx_spoon_on_towel | Vanilla | 3 | 0 | +3 | 0.25 |
| widowx_spoon_on_towel | SHR | 3 | 1 | +2 | 0.625 |
| widowx_stack_cube | Vanilla | 8 | 0 | +8 | 0.0078125 |
| widowx_stack_cube | SHR | 8 | 0 | +8 | 0.0078125 |

## Active jobs

