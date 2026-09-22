# LIBERO lambda sweep final report

Paired states: 300

| Arm | Success | Rate |
|---|---:|---:|
| Vanilla | 260 / 300 | 0.867 |
| Current | 250 / 300 | 0.833 |
| Selected5_lam0.5 | 259 / 300 | 0.863 |
| current_lam025 | 261 / 300 | 0.870 |
| current_lam050 | 255 / 300 | 0.850 |
| selected5_lam025 | 252 / 300 | 0.840 |
| selected5_lam050 | 258 / 300 | 0.860 |

## Pairwise

| Arm | Reference | Rescue | Harm | Net | p |
|---|---|---:|---:|---:|---:|
| current_lam025 | Vanilla | 24 | 23 | +1 | 1.0000 |
| current_lam025 | Current | 34 | 23 | +11 | 0.1849 |
| current_lam025 | Selected5_lam0.5 | 28 | 26 | +2 | 0.8919 |
| current_lam050 | Vanilla | 24 | 29 | -5 | 0.5831 |
| current_lam050 | Current | 26 | 21 | +5 | 0.5601 |
| current_lam050 | Selected5_lam0.5 | 23 | 27 | -4 | 0.6718 |
| selected5_lam025 | Vanilla | 25 | 33 | -8 | 0.3581 |
| selected5_lam025 | Current | 30 | 28 | +2 | 0.8957 |
| selected5_lam025 | Selected5_lam0.5 | 25 | 32 | -7 | 0.4270 |
| selected5_lam050 | Vanilla | 27 | 29 | -2 | 0.8939 |
| selected5_lam050 | Current | 31 | 23 | +8 | 0.3409 |
| selected5_lam050 | Selected5_lam0.5 | 20 | 21 | -1 | 1.0000 |
