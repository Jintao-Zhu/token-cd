# L11 Lambda Counterfactual Advantage OOF

- Preference threshold: **0.65**
- Default arm: **lambda=0.5**

| Feature | Router | Rescue | Harm | Net | Oracle recovery | Switches |
|---|---:|---:|---:|---:|---:|---:|
| task_only | 232/400 | 8 | 14 | -6 | -9.8% | 60 |
| prompt_context | 239/400 | 33 | 32 | +1 | 1.6% | 162 |
| prompt_action | 244/400 | 37 | 31 | +6 | 9.8% | 181 |

## Primary task results

| Task | Router | Fixed .5 | Oracle | Rescue | Harm | Net |
|---|---:|---:|---:|---:|---:|---:|
| open_drawer | 60/100 | 50/100 | 69/100 | 16 | 6 | +10 |
| close_drawer | 83/100 | 82/100 | 92/100 | 5 | 4 | +1 |
| pick_coke_can | 34/100 | 44/100 | 62/100 | 5 | 15 | -10 |
| move_near | 67/100 | 62/100 | 76/100 | 11 | 6 | +5 |

- Decision: **STOP**
