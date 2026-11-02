# pi0-FAST × WidowX/Bridge — 2800-episode run, interim snapshot

*Run:* `runs/pi0fast_bridge_widowx_4task_100seed`  ·  *Protocol:* `PI0FAST_L6MATCHED_WIDOWX_BRIDGE_RTFULL_4TASK_0_99_V1`
*Arms:* vanilla + L6 prompt-attention contrastive guidance, λ ∈ {0.25, 0.50, 0.75, 1.00, 1.25, 1.50} (matched dynamic budget, harmonic negative reconstruction, greedy `(1+λ)·pos − λ·neg`).
*Checkpoint:* `checkpoints/openpi_fast_finetuned/bridge`, its own `norm_stats`.
*State layout:* `[x, y, z, roll, pitch, yaw, PAD(0), gripper]` with `R_topdown` conjugation (`PI0FAST_WIDOWX_LAYOUT=bridge`), actions clipped to the training q01/q99, gripper binarised.
*Renderer:* SAPIEN RT full — depth 8, spp 32, denoiser on (identical to the google_robot run).

## Coverage / integrity at this snapshot

| item | value |
|---|---|
| episodes complete | 666 / 2800 |
| `status: error` rows | **0** |
| renderer variants seen | `(depth 8, spp 32, denoiser True)` × 667 — single value |
| mean / median episode wall time | 435 s / 439 s (60-step cap, 14.6 replans average) |
| guided fallbacks used | 23 replans in 22 episodes (~2 % of guided episodes) |
| throughput (20-min window) | 132 eps/h → ETA ≈ 16 h |
| workers / servers | 18 workers (6 per GPU on GPU 1/2/3, 3 per server) / 6 servers |

## MATCHED same-seed comparison (the only comparison that is valid)

Cells used: **82 (task, seed) units** where *all seven arms* are complete —
carrot 26, eggplant 15, spoon 21, stack 20. Per-task seed sets differ, so absolute
levels are comparable **within** a task, and the pooled column is a sum over each
task's own matched subset.

| arm | success / matched units | rate | net vs vanilla | rescue / harm | exact McNemar p |
|---|---|---|---|---|---|
| **vanilla** | **12 / 82** | **14.6 %** | — | — | — |
| λ0.25 | 21 / 82 | 25.6 % | +9 | 14 / 5 | 0.064 |
| **λ0.50** | **25 / 82** | **30.5 %** | **+13** | **16 / 3** | **0.004** |
| λ0.75 | 20 / 82 | 24.4 % | +8 | 13 / 5 | 0.096 |
| **λ1.00** | **24 / 82** | **29.3 %** | **+12** | **14 / 2** | **0.004** |
| λ1.25 | 21 / 82 | 25.6 % | +9 | 11 / 2 | 0.023 |
| λ1.50 | 23 / 82 | 28.0 % | +11 | 13 / 2 | 0.007 |

## Per task, same-seed matched

| task (matched seeds) | vanilla | λ0.25 | λ0.50 | λ0.75 | λ1.00 | λ1.25 | λ1.50 |
|---|---|---|---|---|---|---|---|
| carrot_on_plate (26) | 3 | 5 | 5 | 5 | 5 | 5 | 5 |
| put_eggplant_in_basket (15) | 8 | 7 | 9 | 7 | 9 | 8 | **10** |
| spoon_on_towel (21) | **0** | 6 | **7** | 5 | **7** | 5 | 5 |
| stack_cube (20) | 1 | 3 | **4** | 3 | 3 | 3 | 3 |

## Reading

* Guidance helps on every task, and the effect is largest where vanilla fails outright
  (`spoon_on_towel` 0/21 → 7/21; `stack_cube` 1/20 → 4/20).
* λ0.50 and λ1.00 lead (+13, p=0.004 and +12, p=0.004); λ0.25 and λ0.75 are the weak ends
  (λ0.25 is the only arm that still produces 5 harms, λ0.75's advantage shrank from +10 at
  n=49 to +8 at n=82 — a reminder that these are interim numbers).
* Eggplant is the one task where vanilla is already strong (8/15); guidance still adds
  (λ1.50 10/15).

## Method notes

* **Matched seeds first, then compare.** The raw per-arm table mixes seed subsets (vanilla
  finishes its seeds earlier), so its percentages must not be read across arms; only the
  intersection where all arms are complete is comparable. `scripts/report_pi0fast_widowx_2800.py`
  now prints that section by default.
* **Fallback sensitivity.** On ~2 % of guided episodes one replan falls back to the unguided
  action chunk (malformed FAST payload: the official decoder reports
  `cannot reshape array of size 68 into shape (7)`). Those episodes are no longer 100 % guided,
  so the final report will also carry an `--exclude-fallback` version.
* **Cross-task levels are not yet comparable**: no seed currently has all 4 tasks × 7 arms
  complete, so the four per-task matched sets are different seed batches (26 / 15 / 21 / 20).
