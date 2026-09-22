# LIBERO-90 Five-Task SIMPLER-Config Evaluation

This is a five-task subset result, not a complete LIBERO-90 score.

| Task | n | Vanilla | Matched | Rescue | Harm | Net | exact p |
|---|---:|---:|---:|---:|---:|---:|---:|
| KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it | 50 | 40/50 | 45/50 | 7 | 2 | 5 | 0.179688 |
| KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet | 50 | 49/50 | 48/50 | 0 | 1 | -1 | 1 |
| LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket | 50 | 34/50 | 31/50 | 9 | 12 | -3 | 0.663624 |
| LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate | 50 | 40/50 | 39/50 | 9 | 10 | -1 | 1 |
| STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy | 50 | 21/50 | 19/50 | 7 | 9 | -2 | 0.803619 |
| **Overall** | 250 | 184/250 | 182/250 | 32 | 34 | -2 | 0.902159 |

Overall paired rate difference (matched - vanilla): -0.0080
Overall bootstrap 95% CI for paired rate difference: [-0.0720, 0.0560]
Excluding previously observed init_state 0: n=245, vanilla=181/245, matched=178/245, net=-3, p=0.804317

## Matched budget distribution

- cases with budget: 250
- mean/std: 37.129 / 11.527
- P10/median/P90: 21.391 / 42.498 / 49.088
- min/max: 5.485 / 57.660

## Both-success completion steps

- n=150
- mean matched-minus-vanilla steps: -12.653333333333334
- mean vanilla/matched steps: 169.36 / 156.70666666666668

## Runtime and infrastructure

- available paired cases: 250 / 250
- error cases: 0
- errors are listed in FINAL_RESULTS.json and cases/error/
