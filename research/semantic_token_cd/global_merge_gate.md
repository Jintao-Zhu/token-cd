# Phase 1A — Global Spatial Merge CD (GSM-CD) Gate

**Protocol:** `ATTN_GLOBAL_MERGE_CD_V1`
**Hypothesis:** "clean fine-grained vision − coarse vision" (structured fine-to-coarse
visual degradation via 2×2 block-mean merge) yields a useful VLA contrastive direction,
replacing the failed "object-absence" framing.

## Operator (locked except η)

Split projector output V ∈ R^{16×16×d} (256 tokens, row-major) into 64 non-overlapping
2×2 blocks B_1..B_64. For each block, μ_k = (1/4)Σ_{i∈B_k} v_i, then

    ṽ_i = (1−η)·v_i + η·μ_k   for i ∈ B_k

Token count stays 256 (no pruning). Sweep η ∈ {0.25, 0.5, 1.0} ONLY. Everything else
locked: λ=0.5, block size 2×2, no layer selection, guided autoregressive prefix.

## CD formula (unchanged)

    z_q* = z_q+ + 0.5·(z_q+ − z_q−)      for action dims 0..5
    z_6* = z_6+                          (gripper keeps clean)

- z_q+ = F(V,  a_{<q}^clean) — greedy, self-consistent
- z_q− = F(Ṽ, a_{<q}^clean) — teacher-forced on the clean greedy prefix (both branches
  share the SAME autoregressive prefix; the negative branch is NOT independently greedy).
- Greedy argmax.

## Arms (5)

| arm              | negative branch                            |
|------------------|---------------------------------------------|
| `vanilla`        | — (clean argmax)                            |
| `semantic_attn_cd` | Oracle GT-mask Attention-CD (τ=0.5, L8–15, mask −1e4), guided prefix — baseline |
| `gsm_025`        | Global 2×2 merge, η=0.25, guided prefix     |
| `gsm_050`        | Global 2×2 merge, η=0.50, guided prefix     |
| `gsm_100`        | Global 2×2 merge, η=1.00, guided prefix     |

All four CD arms share identical seed/λ/guided-prefix; the only difference between
`semantic_attn_cd` and the `gsm_*` arms is *how the negative branch is built* (attention
blocking vs feature merge), so success/harm deltas isolate (a) merge strength η and
(b) merge-vs-attention-blocking.

## Tasks & seeds

3 tasks: `close_drawer`, `move_near`, `pick_coke_can`. 100 fresh paired seeds
200–299 (no prior artifact uses seeds ≥200).

## Sanity checks (pre-rollout, ~20 frozen states/task) — all must pass

- **A** block mean preserved ≈ 0
- **B** local variance ratio = (1−η)²  (0.25→56.25%, 0.5→25%, 1→0)
- **C** residual dose-response: ‖r_0.25‖ < ‖r_1.0‖ (endpoints) and min ‖r‖ > 0
- **D** coarse branch (η=1.0) stays a "weak but normal" policy (finite, ≥2 unique
  action tokens, bounded magnitude), not ~5% collapse

## Evaluation metrics

Per CD arm vs vanilla, per task:
- SR, paired Rescue R = #(V=0, arm=1), Harm H = #(V=1, arm=0), Net = R−H
- ΔSR_t = SR_arm − SR_vanilla, 95% CI (paired proportions), McNemar p
- ΔSR_macro = mean over 3 tasks of ΔSR_t
- Internal: mean ‖r_t‖, first-step cos(r^gsm, r^attn), within-arm temporal cos(r_t, r_{t−1}),
  action-token flip rate, first-step action divergence D_0.

## Gate (Phase 1A GO) — per η ∈ {0.25, 0.5, 1.0}

| # | condition                        | threshold            |
|---|----------------------------------|----------------------|
| G1 | #tasks with Net > 0             | ≥ 2 / 3              |
| G2 | ΔSR_macro                       | ≥ +3 pp              |
| G3 | min_t ΔSR_t                     | > −10 pp             |

**GO** iff G1 ∧ G2 ∧ G3 hold for at least one η. Pick η* = argmax ΔSR_macro
(tie-break: Net, then min_t ΔSR_t).

**NO-GO** if no η satisfies G1∧G2∧G3 → verdict "Fixed Global Spatial Merge NO-GO";
do **not** enter Phase 1B.

## Phase 1B (only if 1A GO)

Lock η*, compare: Global Spatial Merge vs Semantic Local Merge vs Random Local Merge vs
Semantic Attention-CD vs Vanilla.
