# L11 Token-Count Sweep — Final Report

All 2,400 arm-episodes completed on canonical seeds 100–199. All per-episode technical audits and paired initial snapshot/state/RGB hashes passed.

## Success rates

| Task | Matched | K16 | K24 | K32 | K48 | K64 |
|---|---:|---:|---:|---:|---:|---:|
| open_drawer | 44/100 (44.0%) | 21/100 (21.0%) | 26/100 (26.0%) | 40/100 (40.0%) | 37/100 (37.0%) | 42/100 (42.0%) |
| close_drawer | 76/100 (76.0%) | 60/100 (60.0%) | 62/100 (62.0%) | 67/100 (67.0%) | 77/100 (77.0%) | 67/100 (67.0%) |
| pick_coke_can | 33/100 (33.0%) | 31/100 (31.0%) | 27/100 (27.0%) | 35/100 (35.0%) | 33/100 (33.0%) | 32/100 (32.0%) |
| move_near | 60/100 (60.0%) | 39/100 (39.0%) | 58/100 (58.0%) | 54/100 (54.0%) | 60/100 (60.0%) | 57/100 (57.0%) |
| Overall | 213/400 (53.2%) | 151/400 (37.8%) | 173/400 (43.2%) | 196/400 (49.0%) | 207/400 (51.7%) | 198/400 (49.5%) |

## Overall paired comparison against Matched

| Arm | Rescue | Harm | Net | Exact p | Holm-adjusted p |
|---|---:|---:|---:|---:|---:|
| K16 | 25 | 87 | -62 | 3.22512e-09 | 1.61256e-08 |
| K24 | 36 | 76 | -40 | 0.000198237 | 0.000792948 |
| K32 | 45 | 62 | -17 | 0.121512 | 0.364537 |
| K48 | 54 | 60 | -6 | 0.639769 | 0.639769 |
| K64 | 46 | 61 | -15 | 0.17564 | 0.364537 |

## Main result

Matched achieved 213/400 (53.2%). The best fixed count was K48 at 207/400 (51.8%).
The fixed-count curve rises from K16 toward K48 and falls at K64. This supports an effective intervention range around 32–48 tokens, while no single fixed count dominates Matched across tasks.

Matched uses KMeans only for its own-state token budget; token identity remains the L11 attention ranking.
