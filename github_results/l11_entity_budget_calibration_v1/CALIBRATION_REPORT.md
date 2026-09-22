# Entity-Conditioned Budget Calibration

States: **800** (4 tasks × 50 seeds × 4 progress states)

Calibration target: mean matched m = **34.024**

| Estimator | tau | mean m | std | median | P10 | P90 | rho vs matched |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 0.7855 | 34.03 | 9.21 | 33.0 | 23 | 47 | 0.009 |
| entity | 0.7830 | 34.05 | 8.67 | 34.0 | 23 | 45 | -0.041 |
| generic | 0.7240 | 34.02 | 10.37 | 34.0 | 20 | 47 | -0.012 |

## Per-task Spearman

| Task | full | entity | generic |
|---|---:|---:|---:|
| open_drawer | -0.044 | 0.003 | 0.043 |
| close_drawer | -0.172 | -0.193 | -0.174 |
| pick_coke_can | 0.049 | 0.044 | -0.036 |
| move_near | -0.114 | -0.121 | -0.046 |

## Within-episode centered Spearman

| Estimator | rho | p |
|---|---:|---:|
| full | -0.090 | 0.0105 |
| entity | -0.094 | 0.0075 |
| generic | -0.080 | 0.0236 |
