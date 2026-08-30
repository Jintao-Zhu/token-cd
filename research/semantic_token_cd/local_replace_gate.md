# Local Feature-Inpainting CD (LF-CD) — Pre-registered GO/NO-GO gate

**Protocol:** `ATTN_LOCAL_REPLACE_CD_V1`
**Artifact:** `artifacts/attn_local_replace_cd_v1`
**Question:** if instead of *blocking* Action→target attention, we *replace* the
target visual tokens' projector features with the surrounding background-ring
mean (feature inpainting), do we get a cleaner object-absent negative branch?

**Locked config:** λ=0.5, CD on action dims 0..5 (gripper dim 6 keeps clean),
Oracle GT mask τ=0.5 (per entity, source/target separated). Tasks:
`move_near`, `close_drawer`, `pick_coke_can`. Seeds 100..199. Three arms,
capture-once-reuse (all arms share the exact initial state/RGB per seed):
`vanilla`, `oracle_attn_cd` (Oracle Attention-CD, layers [8,16), mask −1e4),
`local_replace_cd` (feature inpainting at the projector output; ring
R(G)=Dilate(G,r)\G\all_target, r grows until |R|≥8, max 4; each entity's ring
excludes every other entity's target region).

The two CD arms share identical seed/mask/τ/λ/coverage, so any success/harm
delta isolates **how the negative branch is built** (attention blocking vs
feature inpainting).

This file is written **before** the full 100-seed results are read (only a
1-seed smoke test is observed to verify the pipeline). It pins the success
criterion so it cannot be moved after the fact.

## Per-task statistics (seeds 100..199)

For task t and CD arm a ∈ {oracle_attn_cd, local_replace_cd}:

- `SR_t(a)` = success rate
- `R_t(a)`  = #{seed : vanilla fails AND arm a succeeds}   (rescue)
- `H_t(a)`  = #{seed : vanilla succeeds AND arm a fails}   (harm)
- `Net_t(a)`= R_t(a) − H_t(a)

Delta (local_replace vs oracle_attn):
- `ΔSR_t`  = SR_t(local_replace) − SR_t(oracle_attn)
- `ΔNet_t` = Net_t(local_replace) − Net_t(oracle_attn)

## GO conditions — ALL three must hold

1. **LocalReplace ≥ Oracle-Attn on ≥2/3 tasks (by net):**
   `|{ t : ΔNet_t > 0 }| ≥ 2`

2. **Pooled success-rate gain ≥ +3pp:**
   `Σ_t ΔSR_t ≥ +3`

3. **Pick Coke strictly improves:**
   `ΔSR_{pick_coke_can} > 0`   (must be a real gain, not a tie)

## NO-GO conditions — EITHER suffices

- **LocalReplace ≈ Attn (no clean separation):**
  `|{ t : |ΔNet_t| ≤ 1 }| ≥ 2`, **or** `Σ_t ΔSR_t < +1`
- **Pick Coke fails to improve:**
  `ΔSR_{pick_coke_can} ≤ 0`   (flat or worse)

## Internal metrics recorded (round-1 diagnostics, not gating)

- `mean_residual_norm`  — ‖z⁺ − z⁻‖ (log-softmax residual, action dims 0..5)
- `mean_mu_norm_ratio`  — ‖μ_bg‖ / mean_j∈R ‖v_j‖ (feature norm ratio; recorded
  only, norm-matching is NOT applied in V1)
- `mean_semantic_sim_before` / `mean_semantic_sim_after` — mean cos(v_i, e_entity)
  over i∈G before / cos(μ_bg, e_entity) after
- residual cosine cos(r_replace, r_attn) — computed **offline** from the stored
  `positive/negative` logits arrays (needs both CD arms on the same seed/step).

## Interpretation

- **GO** — feature inpainting produces a strictly cleaner negative branch than
  attention blocking. Then the "object-absent" hypothesis is supported: proceed
  to V2 (Laplacian inpainting / multi-neighborhood means), keeping the same
  per-entity oracle masks.
- **NO-GO** — feature inpainting does **not** separate from attention blocking
  (the residual direction / magnitude is dominated by something else, e.g. the
  negative branch never truly suppresses the target's *language* semantics). If
  the projector-level intervention fails here, pivot to V2 = **pre-Vision-Transformer
  patch-embedding inpainting** to rule out vision-encoder internal leakage.
