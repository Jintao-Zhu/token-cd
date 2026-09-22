# L11 Visual Top-p 0.90 Extension — Final Report

TopP90 adds 400 new episodes on the same four tasks and seeds 100--199. Matched and TopP75/80/85 are reused; all paired snapshot/state/RGB hashes and technical audits passed.

| Task | Matched | TopP75 | TopP80 | TopP85 | TopP90 |
|---|---:|---:|---:|---:|---:|
| open_drawer | 44/100 (44.0%) | 26/100 (26.0%) | 32/100 (32.0%) | 38/100 (38.0%) | 38/100 (38.0%) |
| close_drawer | 76/100 (76.0%) | 69/100 (69.0%) | 68/100 (68.0%) | 73/100 (73.0%) | 63/100 (63.0%) |
| pick_coke_can | 33/100 (33.0%) | 34/100 (34.0%) | 34/100 (34.0%) | 35/100 (35.0%) | 34/100 (34.0%) |
| move_near | 60/100 (60.0%) | 46/100 (46.0%) | 60/100 (60.0%) | 60/100 (60.0%) | 56/100 (56.0%) |
| Overall | 213/400 (53.2%) | 175/400 (43.8%) | 194/400 (48.5%) | 206/400 (51.5%) | 191/400 (47.8%) |

| Candidate | Baseline | Rescue | Harm | Net | Exact p | Holm-adjusted p |
|---|---|---:|---:|---:|---:|---:|
| TopP75 | Matched | 44 | 82 | -38 | 0.000905277 | 0.00362111 |
| TopP80 | Matched | 50 | 69 | -19 | 0.0985236 | 0.197047 |
| TopP85 | Matched | 52 | 59 | -7 | 0.56922 | 0.56922 |
| TopP90 | Matched | 54 | 76 | -22 | 0.0650865 | 0.195259 |
| TopP90 | TopP85 | 52 | 67 | -15 | 0.199158 | — |

## TopP90 Per-task Paired Results

| Task | Baseline | Rescue | Harm | Net | Exact p |
|---|---|---:|---:|---:|---:|
| open_drawer | Matched | 13 | 19 | -6 | 0.377086 |
| open_drawer | TopP85 | 9 | 9 | +0 | 1 |
| close_drawer | Matched | 9 | 22 | -13 | 0.0294494 |
| close_drawer | TopP85 | 11 | 21 | -10 | 0.110184 |
| pick_coke_can | Matched | 17 | 16 | +1 | 1 |
| pick_coke_can | TopP85 | 17 | 18 | -1 | 1 |
| move_near | Matched | 15 | 19 | -4 | 0.607591 |
| move_near | TopP85 | 15 | 19 | -4 | 0.607591 |
