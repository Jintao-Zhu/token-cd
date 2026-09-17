# Target Positive-Boost Token Causal Audit

## Scale

- 400 episodes, 1200 early/middle/late same states, 727 budget-preserving atomic swaps.
- Every state compares Correct and Positive-Boost-0.5 on the same image and action prefix.

## Overall

| Group | States | Mean swaps | Zero-swap | Action-flip states | Interaction-only | Residual cosine | Action L2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rescue | 120 | 0.758 | 47.5% | 24.2% | 0.0% | 0.9678 | 0.0132 |
| harm | 183 | 0.497 | 57.9% | 8.2% | 0.0% | 0.9874 | 0.0044 |
| both_success | 531 | 0.557 | 53.9% | 17.1% | 1.3% | 0.9793 | 0.0071 |
| both_fail | 366 | 0.680 | 48.6% | 21.0% | 0.8% | 0.9661 | 0.0115 |

## Harm versus Rescue

- swap_count: Harm 0.49727, Rescue 0.75833, p=0.008795.
- residual_cosine: Harm 0.98738, Rescue 0.96782, p=0.02284.
- action_l2: Harm 0.00441, Rescue 0.01322, p=9.308e-05.
- residual_norm_delta: Harm 0.39648, Rescue 1.51863, p=0.09815.
- feature_norm_delta: Harm 0.21684, Rescue 0.62426, p=0.007866.

## Atomic swap confidence

- entered_p_rank: Harm 42.20879, Rescue 33.25275, p=7.565e-05.
- entered_difference: Harm 0.00293, Rescue 0.00618, p=5.006e-08.
- only_action_l2_vs_correct: Harm 0.01158, Rescue 0.02149, p=0.01152.
- only_residual_cosine_vs_correct: Harm 0.97433, Rescue 0.95743, p=7.688e-05.

The outcome-associated pattern is low-confidence rather than high-magnitude Harm: Harm swaps enter from deeper Correct-P ranks and carry smaller P-Q evidence. The exploratory gate `P rank <= 40 and P-Q >= 0.004` retains 52.7% of Rescue-associated swaps versus 17.6% of Harm-associated swaps. This gate is post-hoc and requires held-out validation.

## Reproducibility boundary

Mean recomputed-vs-historical Correct-mask Jaccard is 0.743; it decreases under long replay because historical executed actions were stored at finite precision. Correct-vs-Boost comparisons within every audited state remain exact same-state comparisons, but outcome-conditioned middle/late associations are diagnostic rather than definitive causal attribution.

## Files

- `state_rows.csv`: one row per same state.
- `atomic_swap_rows.csv`: one row per budget-preserving token swap.
- `GROUP_SUMMARIES.json` and `ATOMIC_SUMMARIES.json`: task/stage/outcome aggregates.
- Per-state masks, images and centered residuals are retained under `results/`.
