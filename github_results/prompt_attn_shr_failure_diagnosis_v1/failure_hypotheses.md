# Prompt-Attn-SHR v1 failure-cause analysis

## Boundary and closed-loop result

This is an outcome-stratified diagnostic set (30 episodes × 3 states = 90 states), not a new success-rate estimate.
The completed closed-loop screen was Standard 148/300, Prompt-v1 109/300, Random 125/300. Prompt-v1 vs Standard: Rescue 25, Harm 64, Net −39.

## Conclusion

Prompt-v1 的失败主要不是实现或覆盖量错误，而是 selector 在同一状态上选择了与 semantic group 大幅不同、空间结构和重建效应不同的集合；这些集合产生的 action residual 与 Standard SHR 方向一致性有限，在 drawer/coke 的闭环中 Harm 远多于 Rescue。完整指令后半层 attention 在本诊断中也没有表现出足够强的任务条件响应，且其效果未优于等覆盖 Random-SHR。

## Evidence chain

1. **Implementation extraction passed: True.** Nine observations verified query/key indices, layers, repeated aggregation and clean-action invariance. All 30 diagnostic Standard reruns matched the original closed-loop initial action/mask; all 90 selected cached RGB states reproduced the diagnostic driver's clean action/mask. Equal coverage, reconstruction scope, prefix, lambda and gripper paths were also verified. Therefore the failure cannot be explained by an obvious indexing, layer numbering, coverage, or downstream-configuration mismatch.
2. **Prompt selects substantially different evidence.** Mean Prompt/Standard overlap is 0.232 and Jaccard is 0.140; Random/Standard overlap is 0.128. Low overlap alone is not an error, but establishes that the intervention changed materially.
3. **Geometry differs.** Mean component counts: Standard 3.06, Prompt 12.98, Random 23.71, 10-mask random reference 24.00. Largest-component ratios: Standard 0.753, Prompt 0.283, Random 0.122. Outer-ring selection ratios: Standard 0.104, Prompt 0.226, Random 0.227; uniform expectation is 0.234. These measurements determine whether fragmentation/edge bias is present without declaring every edge token meaningless.
4. **Equal m does not imply equal intervention.** Mean feature perturbation totals: Standard 161.87, Prompt 133.89, Random 111.19. Mean selected-token relative perturbation: Standard 0.935, Prompt 0.820, Random 0.667.
5. **Action effect changes.** Mean centered residual norms: Standard 134.60, Prompt 145.99, Random 48.95. Prompt-vs-Standard residual cosine is 0.297; this tests direction rather than only strength. Mean winner flips per state: Standard 1.22, Prompt 1.52, Random 0.74 across six guided dimensions.
6. **Task response is limited in the audited counterfactuals.** For nine fixed images, a verb/entity/source-target change leaves mean Top-m Jaccard 0.909 and score cosine 0.987. This is evidence about sensitivity, not proof of semantic correctness.
7. **Closed-loop consequence agrees with the mechanism diagnosis.** Prompt-v1 is much worse on open_drawer (21% vs 47%) and pick_coke_can (27% vs 41%), while move_near is only tied (61% vs 60%) and also ties Random (61%). Thus the selected differences do not translate into useful task-grounded intervention overall.

## Per-task diagnostic summary

| Task | Prompt/Std overlap | Prompt components | Std components | Prompt residual norm | Std residual norm |
|---|---:|---:|---:|---:|---:|
| open_drawer | 0.307 | 13.40 | 1.40 | 173.01 | 165.68 |
| pick_coke_can | 0.115 | 12.13 | 4.23 | 124.68 | 65.50 |
| move_near | 0.274 | 13.40 | 3.53 | 140.26 | 172.62 |

## What is and is not established

Established: extraction is internally consistent; Prompt-v1 changes mask identity/geometry and the resulting reconstruction/residual; these changes correlate with an unfavorable closed-loop Rescue/Harm balance.

Not established: that every high-attention edge/corner token is an attention sink; that late layers are universally unsuitable; or that entity-only/mid-layer variants would succeed. Those require separate single-variable validation and are intentionally outside this run.

## Files

- `implementation_audit.md`, `query_index_audit.csv`, `patch_index_reference.png`
- `state_diagnostics.csv`, `AGGREGATE_DIAGNOSTICS.json`
- `case_gallery.html`, `layer_query_gallery.html`, `aggregate_diagnostics.pdf`
