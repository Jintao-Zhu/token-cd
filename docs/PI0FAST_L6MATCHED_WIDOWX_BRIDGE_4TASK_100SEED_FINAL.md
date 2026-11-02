# pi0-FAST × WidowX/Bridge — 4 tasks × 100 seeds × 7 arms (λ sweep), FINAL

*Run:* `runs/pi0fast_bridge_widowx_4task_100seed`  ·  *generated:* 2026-10-14 12:18:24
*Protocol:* `PI0FAST_L6MATCHED_WIDOWX_BRIDGE_RTFULL_4TASK_0_99_V1`

## 1. Integrity

| gate | value | pass |
|---|---|---|
| episodes complete / expected | 2800 / 2800 | PASS |
| rows with status=error | 0 | PASS |
| duplicate (task, seed, arm) | 0 | PASS |
| unreadable JSON | 0 | PASS |
| renderer variants | {(8, 32, True): 2800} | PASS |
| per-arm counts | {'l6_matched_050': 400, 'l6_matched_125': 400, 'l6_matched_025': 400, 'l6_matched_075': 400, 'l6_matched_150': 400, 'vanilla': 400, 'l6_matched_100': 400} | PASS |
| runtime mean / median (s) | 397.9 / 385.8 | PASS |

Guided fallbacks: **81 replans in 78 episodes** (those episodes are not 100 % guided; section 5 repeats the main comparison without them).

## 2. Success rate, same-seed matched (all episodes)

Matched (task, seed) cells where all seven arms are complete: **400** (carrot 100, eggplant 100, spoon 100, stack 100).

| task (n) | vanilla | λ0.25 | λ0.50 | λ0.75 | λ1.00 | λ1.25 | λ1.50 |
|---|---|---|---|---|---|---|---|
| **carrot_on_plate** (100) | 15/100 · 15% | 23/100 · 23% | 22/100 · 22% | 25/100 · 25% | 24/100 · 24% | 25/100 · 25% | 24/100 · 24% |
| **put_eggplant_in_basket** (100) | 41/100 · 41% | 38/100 · 38% | 34/100 · 34% | 36/100 · 36% | 46/100 · 46% | 44/100 · 44% | 50/100 · 50% |
| **spoon_on_towel** (100) | 7/100 · 7% | 32/100 · 32% | 27/100 · 27% | 26/100 · 26% | 31/100 · 31% | 19/100 · 19% | 24/100 · 24% |
| **stack_cube** (100) | 3/100 · 3% | 13/100 · 13% | 13/100 · 13% | 13/100 · 13% | 14/100 · 14% | 12/100 · 12% | 12/100 · 12% |
| **all tasks** | **66/400 · 16.5%** | **106/400 · 26.5%** | **96/400 · 24.0%** | **100/400 · 25.0%** | **115/400 · 28.8%** | **100/400 · 25.0%** | **110/400 · 27.5%** |

Paired vs vanilla (same seed cells; rescue = vanilla fails & arm succeeds):

| arm | net | rescue | harm | exact McNemar p |
|---|---|---|---|---|
| λ0.25 | +40 | 70 | 30 | 0.0001 |
| λ0.50 | +30 | 61 | 31 | 0.0023 |
| λ0.75 | +34 | 65 | 31 | 0.0007 |
| λ1.00 | +49 | 70 | 21 | 0.0000 |
| λ1.25 | +34 | 59 | 25 | 0.0003 |
| λ1.50 | +44 | 63 | 19 | 0.0000 |

## 3. λ vs λ (pairwise, same matched cells)

| pair | net (b wins) | exact McNemar p |
|---|---|---|
| λ0.25 vs λ0.50 | -10 (favours λ0.25) | 0.2288 |
| λ0.25 vs λ0.75 | -6 (favours λ0.25) | 0.5386 |
| λ0.25 vs λ1.00 | +9 (favours λ1.00) | 0.3211 |
| λ0.25 vs λ1.25 | -6 (favours λ0.25) | 0.5190 |
| λ0.25 vs λ1.50 | +4 (favours λ1.50) | 0.7122 |
| λ0.50 vs λ0.75 | +4 (favours λ0.75) | 0.6940 |
| λ0.50 vs λ1.00 | +19 (favours λ1.00) | 0.0370 |
| λ0.50 vs λ1.25 | +4 (favours λ1.25) | 0.7239 |
| λ0.50 vs λ1.50 | +14 (favours λ1.50) | 0.1034 |
| λ0.75 vs λ1.00 | +15 (favours λ1.00) | 0.0959 |
| λ0.75 vs λ1.25 | +0 (favours λ0.75) | 1.0000 |
| λ0.75 vs λ1.50 | +10 (favours λ1.50) | 0.2529 |
| λ1.00 vs λ1.25 | -15 (favours λ1.00) | 0.0534 |
| λ1.00 vs λ1.50 | -5 (favours λ1.00) | 0.5966 |
| λ1.25 vs λ1.50 | +10 (favours λ1.50) | 0.2529 |

## 4. Per-task best λ

| task | vanilla | best λ | best λ rate | runner-up | Δ vs vanilla (rescues/harms) |
|---|---|---|---|---|---|
| carrot_on_plate (100) | 15/100 · 15% | **λ1.25** | 25/100 · 25% | λ0.75 25/100 | +10 (16/6) |
| put_eggplant_in_basket (100) | 41/100 · 41% | **λ1.50** | 50/100 · 50% | λ1.00 46/100 | +9 (20/11) |
| spoon_on_towel (100) | 7/100 · 7% | **λ0.25** | 32/100 · 32% | λ1.00 31/100 | +25 (27/2) |
| stack_cube (100) | 3/100 · 3% | **λ1.00** | 14/100 · 14% | λ0.75 13/100 | +11 (11/0) |

## 5. Sensitivity: episodes with a fallback replan excluded

Matched (task, seed) cells where all seven arms are complete: **357** (carrot 94, eggplant 92, spoon 71, stack 100).

| task (n) | vanilla | λ0.25 | λ0.50 | λ0.75 | λ1.00 | λ1.25 | λ1.50 |
|---|---|---|---|---|---|---|---|
| **carrot_on_plate** (94) | 15/94 · 16% | 23/94 · 24% | 22/94 · 23% | 24/94 · 26% | 23/94 · 24% | 24/94 · 26% | 23/94 · 24% |
| **put_eggplant_in_basket** (92) | 39/92 · 42% | 34/92 · 37% | 30/92 · 33% | 32/92 · 35% | 43/92 · 47% | 42/92 · 46% | 46/92 · 50% |
| **spoon_on_towel** (71) | 7/71 · 10% | 22/71 · 31% | 24/71 · 34% | 25/71 · 35% | 21/71 · 30% | 13/71 · 18% | 18/71 · 25% |
| **stack_cube** (100) | 3/100 · 3% | 13/100 · 13% | 13/100 · 13% | 13/100 · 13% | 14/100 · 14% | 12/100 · 12% | 12/100 · 12% |
| **all tasks** | **64/357 · 17.9%** | **92/357 · 25.8%** | **89/357 · 24.9%** | **94/357 · 26.3%** | **101/357 · 28.3%** | **91/357 · 25.5%** | **99/357 · 27.7%** |

Paired vs vanilla (same seed cells; rescue = vanilla fails & arm succeeds):

| arm | net | rescue | harm | exact McNemar p |
|---|---|---|---|---|
| λ0.25 | +28 | 58 | 30 | 0.0037 |
| λ0.50 | +25 | 56 | 31 | 0.0097 |
| λ0.75 | +30 | 60 | 30 | 0.0021 |
| λ1.00 | +37 | 57 | 20 | 0.0000 |
| λ1.25 | +27 | 51 | 24 | 0.0024 |
| λ1.50 | +35 | 54 | 19 | 0.0001 |

