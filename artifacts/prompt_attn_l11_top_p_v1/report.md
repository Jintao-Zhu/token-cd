# L11 Visual Top-p Adaptive Count — Final Report

All 1,200 new adaptive episodes completed; 400 paired Matched episodes were reused. Technical audits and initial snapshot/state/RGB hashes passed.

| Task | Matched | TopP75 | TopP80 | TopP85 |
|---|---:|---:|---:|---:|
| open_drawer | 44/100 (44.0%) | 26/100 (26.0%) | 32/100 (32.0%) | 38/100 (38.0%) |
| close_drawer | 76/100 (76.0%) | 69/100 (69.0%) | 68/100 (68.0%) | 73/100 (73.0%) |
| pick_coke_can | 33/100 (33.0%) | 34/100 (34.0%) | 34/100 (34.0%) | 35/100 (35.0%) |
| move_near | 60/100 (60.0%) | 46/100 (46.0%) | 60/100 (60.0%) | 60/100 (60.0%) |
| Overall | 213/400 (53.2%) | 175/400 (43.8%) | 194/400 (48.5%) | 206/400 (51.5%) |

| Arm | Rescue | Harm | Net | Exact p | Holm-adjusted p |
|---|---:|---:|---:|---:|---:|
| TopP75 | 44 | 82 | -38 | 0.000905277 | 0.00271583 |
| TopP80 | 50 | 69 | -19 | 0.0985236 | 0.197047 |
| TopP85 | 52 | 59 | -7 | 0.56922 | 0.56922 |
