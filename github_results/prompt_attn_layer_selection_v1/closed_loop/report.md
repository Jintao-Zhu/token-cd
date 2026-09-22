# Prompt-Attn sparse-layer selection v1

Candidates were frozen using 20 exploration episodes before the 10-episode offline validation was read.
Prompt-Single uses layer `[11]`; Prompt-Sparse uses layers `[11, 14]`.

## Closed-loop success

| Task | Vanilla | Standard SHR | Prompt-v1 | Prompt-Single | Prompt-Sparse |
|---|---:|---:|---:|---:|---:|
| open_drawer | 27/100 (27.0%) | 47/100 (47.0%) | 21/100 (21.0%) | 50/100 (50.0%) | 36/100 (36.0%) |
| pick_coke_can | 23/100 (23.0%) | 41/100 (41.0%) | 27/100 (27.0%) | 44/100 (44.0%) | 38/100 (38.0%) |
| move_near | 62/100 (62.0%) | 60/100 (60.0%) | 61/100 (61.0%) | 62/100 (62.0%) | 62/100 (62.0%) |
| Overall | 112/300 (37.3%) | 148/300 (49.3%) | 109/300 (36.3%) | 156/300 (52.0%) | 136/300 (45.3%) |

## Paired changes

| Scope | Comparison | Rescue | Harm | Net |
|---|---|---:|---:|---:|
| open_drawer | prompt_single_vs_vanilla | 28 | 5 | 23 |
| open_drawer | prompt_single_vs_prompt_v1 | 35 | 6 | 29 |
| open_drawer | prompt_single_vs_standard_shr | 17 | 14 | 3 |
| open_drawer | prompt_sparse_vs_vanilla | 16 | 7 | 9 |
| open_drawer | prompt_sparse_vs_prompt_v1 | 26 | 11 | 15 |
| open_drawer | prompt_sparse_vs_standard_shr | 11 | 22 | -11 |
| pick_coke_can | prompt_single_vs_vanilla | 31 | 10 | 21 |
| pick_coke_can | prompt_single_vs_prompt_v1 | 26 | 9 | 17 |
| pick_coke_can | prompt_single_vs_standard_shr | 20 | 17 | 3 |
| pick_coke_can | prompt_sparse_vs_vanilla | 25 | 10 | 15 |
| pick_coke_can | prompt_sparse_vs_prompt_v1 | 21 | 10 | 11 |
| pick_coke_can | prompt_sparse_vs_standard_shr | 18 | 21 | -3 |
| move_near | prompt_single_vs_vanilla | 6 | 6 | 0 |
| move_near | prompt_single_vs_prompt_v1 | 15 | 14 | 1 |
| move_near | prompt_single_vs_standard_shr | 14 | 12 | 2 |
| move_near | prompt_sparse_vs_vanilla | 14 | 14 | 0 |
| move_near | prompt_sparse_vs_prompt_v1 | 14 | 13 | 1 |
| move_near | prompt_sparse_vs_standard_shr | 13 | 11 | 2 |
| Overall | prompt_single_vs_vanilla | 65 | 21 | 44 |
| Overall | prompt_single_vs_prompt_v1 | 76 | 29 | 47 |
| Overall | prompt_single_vs_standard_shr | 51 | 43 | 8 |
| Overall | prompt_sparse_vs_vanilla | 55 | 31 | 24 |
| Overall | prompt_sparse_vs_prompt_v1 | 61 | 34 | 27 |
| Overall | prompt_sparse_vs_standard_shr | 42 | 54 | -12 |

## Interpretation boundary

A better target-response heatmap is treated only as a mechanism diagnostic. The conclusion about layer averaging is determined by paired closed-loop success, not by residual magnitude or visual appearance alone.
