# IC-SHR v1 report

All 1200 episodes reuse the released canonical snapshots.

## Success and paired result

| Task | Vanilla | Recon | SHR | IC-SHR | Delta pp | Rescue | Harm | Net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| close_drawer | 51.333% | 68.000% | 69.667% | 69.667% | +0.00 | 29 | 29 | +0 |
| open_drawer | 23.667% | 37.667% | 46.000% | 32.667% | -13.33 | 28 | 68 | -40 |
| pick_coke_can | 27.667% | 33.333% | 32.333% | 31.000% | -1.33 | 42 | 46 | -4 |
| move_near | 61.333% | 59.667% | 65.667% | 63.667% | -2.00 | 28 | 34 | -6 |

## Component diagnostics

| Task | Mean components | Mean selected | Mean filtered | Tokens before | Tokens after | Deleted | Filtered episodes |
|---|---:|---:|---:|---:|---:|---:|---:|
| close_drawer | 1.74 | 1.00 | 0.74 | 33.9 | 32.3 | 5.33% | 300/300 |
| open_drawer | 1.86 | 1.00 | 0.86 | 37.1 | 35.5 | 4.75% | 300/300 |
| pick_coke_can | 3.49 | 1.00 | 2.49 | 25.2 | 16.7 | 33.15% | 300/300 |
| move_near | 3.77 | 1.79 | 1.98 | 41.1 | 33.6 | 17.01% | 300/300 |
