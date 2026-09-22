# LIBERO-Spatial Experiment Record

Date: 2026-09-22

## Locked protocol

For the official comparison:

```text
10 tasks
50 unique init states per task
env_seed = 0
10 no-op warmup steps
max policy steps = 220
matched and vanilla use identical states
```

The first custom experiment used 300 steps and repeated the 50 init states twice; those results are legacy and are not mixed with the official result.

## 1. Legacy custom 300-step experiment

| Arm | Success |
|---|---:|
| L11 + matched, all-entity | 761 / 1000 |
| Vanilla | 782 / 1000 |

Source/target-only diagnostic on T1/T2/T4, custom 300-step:

| Task | matched all-entity | source/target-only | vanilla |
|---|---:|---:|---:|
| T1 between plate / ramekin | 84 / 100 | 85 / 100 | 79 / 100 |
| T2 next to ramekin | 71 / 100 | 77 / 100 | 86 / 100 |
| T4 on cookie box | 89 / 100 | 91 / 100 | 92 / 100 |
| Total | 244 / 300 | 253 / 300 | 257 / 300 |

## 2. Official protocol: current L11 vs vanilla

| Arm | Success |
|---|---:|
| Vanilla | 423 / 500 |
| Current L11 matched | 410 / 500 |

Per task:

| Task | matched | vanilla | delta |
|---|---:|---:|---:|
| T1 between plate / ramekin | 45 | 47 | -2 |
| T2 next to ramekin | 36 | 48 | -12 |
| T3 from table center | 45 | 40 | +5 |
| T4 on cookie box | 43 | 49 | -6 |
| T5 in top drawer | 35 | 35 | 0 |
| T6 on ramekin | 46 | 43 | +3 |
| T7 next to cookie box | 48 | 45 | +3 |
| T8 on stove | 41 | 46 | -5 |
| T9 next to plate | 35 | 39 | -4 |
| T10 on wooden cabinet | 36 | 31 | +5 |
| Total | 410 | 423 | -13 |

Pairwise against vanilla:

```text
rescue = 57
harm = 44
net = +13 for vanilla
McNemar p = 0.2323
```

## 3. Dev closed-loop, states 0-9

100 paired episodes:

| Arm | Success |
|---|---:|
| Vanilla | 80 / 100 |
| Current L11 | 79 / 100 |
| Role L11 | 81 / 100 |
| Role deep 23-25 | 83 / 100 |

This set was used for development/selection, not final confirmation.

## 4. Held-out test, states 20-49

300 paired episodes:

| Arm | Success |
|---|---:|
| Vanilla | 260 / 300 |
| Current L11 | 250 / 300 |
| Role L11 | 251 / 300 |
| Role deep 23-25 | 248 / 300 |
| Full deep 23-25 | 241 / 300 |

Conclusion: old role and full-deep variants did not beat vanilla.

## 5. Ranking-repair matrix, states 20-49

All arms use matched count, official protocol, 300 episodes:

| Arm | Success |
|---|---:|
| Vanilla | 260 / 300 |
| Current L11 | 250 / 300 |
| endpoint_L11 | 250 / 300 |
| endpoint_L14 | 248 / 300 |
| selected3_instruction | 253 / 300 |
| selected3_endpoint | 243 / 300 |
| selected5_endpoint | 259 / 300 |
| selected3_full_endpoint | 250 / 300 |
| sr_d025_L11 | 252 / 300 |
| sr_d025_selected3 | 252 / 300 |

selected5_endpoint versus current:

```text
rescue = 30
harm = 21
net = +9
p = 0.2624
```

selected5_endpoint versus vanilla:

```text
rescue = 27
harm = 28
net = -1
p = 1.0000
```

selected5_endpoint per task:

| Task | Vanilla | current | selected5_endpoint |
|---|---:|---:|---:|
| T1 between plate / ramekin | 28 | 27 | 27 |
| T2 next to ramekin | 28 | 22 | 21 |
| T3 from table center | 26 | 27 | 28 |
| T4 on cookie box | 30 | 26 | 27 |
| T5 in top drawer | 23 | 19 | 27 |
| T6 on ramekin | 26 | 28 | 29 |
| T7 next to cookie box | 28 | 29 | 30 |
| T8 on stove | 29 | 28 | 25 |
| T9 next to plate | 24 | 20 | 23 |
| T10 on wooden cabinet | 18 | 24 | 22 |

## 6. Offline layer and role diagnostics

Dev set: 10 tasks x 20 states = 200 states.

Top-16 target discrimination:

| layer | full | role | SR | SR + plate 0.25 |
|---|---:|---:|---:|---:|
| L11 | 0.1265 | 0.1613 | 0.1548 | 0.1607 |
| mean 23-25 | 0.0031 | 0.0045 | 0.0062 | 0.0048 |

Relation-switch coverage response:

| layer | full | role | SR | SR + plate 0.25 |
|---|---:|---:|---:|---:|
| L11 | 0.0626 | 0.0190 | 0.0277 | 0.0451 |
| mean 23-25 | 0.1762 | 0.1736 | 0.1722 | 0.1684 |

Full layer scan 0-31:

- best top-16 target discrimination: L11 role 0.1613, L11 SR+plate0.25 0.1607
- best top-32 target discrimination: L14 role 0.1977, L14 SR+plate0.5 0.1943
- best relation-switch response: L23, L4, L30, L22, L24, L25
- no single layer is best for both target discrimination and relation-switch response

## 7. Selected attention heads

Selection set: states 0-9. Validation set: states 10-19.

Top-3 heads:

| layer | head | visual mass | target discrimination | switch response |
|---|---:|---:|---:|---:|
| 14 | 5 | 0.382 | 0.158 | 0.155 |
| 7 | 14 | 0.268 | 0.091 | 0.171 |
| 9 | 12 | 0.363 | 0.081 | 0.180 |

Top-5 adds:

| layer | head | visual mass | target discrimination | switch response |
|---|---:|---:|---:|---:|
| 19 | 11 | 0.229 | 0.093 | 0.151 |
| 21 | 2 | 0.242 | 0.074 | 0.203 |

Validation metrics stayed stable.

## 8. Rescue / harm analysis for selected5_endpoint vs vanilla

Paired 300 episodes:

| Group | n | First divergence step | First action L2 | guided change | mean m |
|---|---:|---:|---:|---:|---:|
| Rescue | 27 | 0.33 | 0.156 | 0.217 | 34.15 |
| Harm | 28 | 0.36 | 0.210 | 0.329 | 35.89 |
| Both success | 232 | 0.60 | 0.155 | 0.216 | 32.92 |
| Both fail | 13 | 0.38 | 0.161 | 0.332 | 35.44 |

Harm episodes had substantially larger guided change and larger immediate action changes.

## 9. Lambda sweep, partial snapshot at 09:59

Still running. Current lambda0 = 0.5.

| Arm | Completed | Success |
|---|---:|---:|
| current, scale 0.25 (lambda 0.125) | 113 / 300 | 99 |
| current, scale 0.50 (lambda 0.25) | 114 / 300 | 102 |
| selected5, scale 0.25 | 102 / 300 | 90 |
| selected5, scale 0.50 | 103 / 300 | 90 |

These are partial and must not be treated as final.

## Key source files

- Official comparison: `artifacts/libero_official_compare_500_v1/COMPARE_REPORT.md`
- Ranking repair matrix: `artifacts/libero_ranking_repair_matrix_v1/RANKING_REPAIR_REPORT.md`
- Full layer scan: `artifacts/libero_attention_full_layer_scan_dev_v1/GROUP_DIAGNOSTICS.csv`
- Head selection: `artifacts/libero_attention_head_selection_dev_v1/SELECTED_HEADS.json`
- Rescue/harm report: `artifacts/libero_rescue_harm_analysis_v1/RESCUE_HARM_REPORT.md`
- Lambda sweep: `artifacts/libero_lambda_sweep_v1/`
