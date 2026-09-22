# LIBERO-90 Migration Smoke Results

This table is an engineering smoke validation, not a method-performance comparison.

| Task | Init state | Arm | Normal end | Success | Steps | Mean m | Notes |
|---|---:|---|---|---|---:|---:|---|
| KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it | 0 | vanilla | True | True | 174 | - | paired initial state verified |
| KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it | 0 | matched | True | True | 169 | 21.22 | paired initial state verified; entities=['butter at the back', 'top drawer of the cabinet'] |
| KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet | 0 | vanilla | True | True | 205 | - | paired initial state verified |
| KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet | 0 | matched | True | True | 144 | 28.22 | paired initial state verified; entities=['black bowl', 'top of the cabinet'] |
| LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket | 0 | vanilla | True | True | 146 | - | paired initial state verified |
| LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket | 0 | matched | True | False | 400 | 56.05 | paired initial state verified; entities=['tomato sauce', 'basket'] |
| LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate | 0 | vanilla | True | False | 400 | - | paired initial state verified |
| LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate | 0 | matched | True | True | 131 | 45.78 | paired initial state verified; entities=['white mug', 'plate'] |
| STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy | 0 | vanilla | True | False | 400 | - | paired initial state verified |
| STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy | 0 | matched | True | True | 241 | 39.03 | paired initial state verified; entities=['book', 'front compartment of the caddy'] |
