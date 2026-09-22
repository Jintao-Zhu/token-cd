# Target-specific Attention Offline Gate

**STOP**

- 60/60 same-state cases complete.
- Historical Correct mask/action equivalence: PASS.
- Target-Diff selected-score positivity: FAIL.
- Primary/alternate generic wording mask Jaccard: 0.633.

| Task | mean m | Diff∩Correct Jaccard | Boost∩Correct | Reverse∩Correct | Generic wording Jaccard |
|---|---:|---:|---:|---:|---:|
| open_drawer | 34.1 | 0.311 | 0.738 | 0.270 | 0.745 |
| close_drawer | 30.5 | 0.255 | 0.779 | 0.329 | 0.562 |
| pick_coke_can | 31.3 | 0.323 | 0.808 | 0.284 | 0.597 |
| move_near | 38.9 | 0.424 | 0.790 | 0.210 | 0.630 |
