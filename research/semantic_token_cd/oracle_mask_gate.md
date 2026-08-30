# Oracle Semantic-Mask Test — Pre-registered GO/NO-GO gate

**Protocol:** `ATTN_SEMANTIC_ORACLE_MASK_V1`
**Artifact:** `artifacts/attn_semantic_oracle_mask_v1`
**Locked config:** K=8, layers [8,16), λ=0.5, mask −1e4, Attention-CD on tokens 0..5
(gripper token 6 keeps clean). Tasks: `move_near`, `close_drawer`, `pick_coke_can`.
Seeds 100..199. Four arms, capture-once-reuse (all arms share the exact initial
state/RGB per seed): `vanilla`, `kmeans_cd` (= existing L8-15 semantic_hard),
`oracle_cd` (G_Oracle, τ=0.5), `oracle_budget_cd` (Top-B by overlap, B=|G_KMeans|/step).

This file is written **before** the full 100-seed results are read (only the 1-seed
smoke test was observed, and only to verify the pipeline). It pins the exact
success criterion so it cannot be moved after the fact.

## Per-task statistics (seeds 100..199)

For task t and arm a (a ∈ {kmeans_cd, oracle_cd, oracle_budget_cd}):

- `R_t(a)` = #{seed : vanilla fails AND arm a succeeds}   (rescue)
- `H_t(a)` = #{seed : vanilla succeeds AND arm a fails}   (harm)
- `Net_t(a)` = R_t(a) − H_t(a)

## GO conditions — ALL four must hold

1. **Oracle beats KMeans on ≥2/3 tasks (full mask):**
   `|{ t : Net_t(oracle_cd) > Net_t(kmeans_cd) }| ≥ 2`

2. **Pooled net effect favors rescue:**
   `Σ_t Net_t(oracle_cd) > 0`

3. **Instability improves on the two unstable tasks (move_near, pick_coke_can):**
   `Σ_{t∈{move_near, pick_coke_can}} Net_t(oracle_cd) > Σ_{t∈{move_near, pick_coke_can}} Net_t(kmeans_cd)`

4. **Improvement survives budget matching (region quality, not just mask size):**
   `|{ t : Net_t(oracle_budget_cd) > Net_t(kmeans_cd) }| ≥ 2`

## Interpretation

- **GO** — the Oracle GT mask beats the KMeans selector on the pre-registered
  criteria. Then the KMeans *region localization* is the confirmed bottleneck:
  a spatially-exact mask helps. Next step: replace G_KMeans with a better
  selector (GroundingDINO+SAM2), NOT GroupViT first.
- **NO-GO** — the Oracle GT mask does **not** reliably beat KMeans. Then region
  quality is NOT the bottleneck (even a perfect mask can't fix it); the residual
  direction and/or the blocking layer are. Per prior decision: **stop the
  selector line entirely**, pivot to Adaptive-Layer / Phase.

Note: a per-task success/harm asymmetry that flips sign between `oracle_cd` and
`oracle_budget_cd` (e.g. full mask rescues but budget-matched harms, as seen on
the single smoke seed for pick_coke_can) is itself diagnostic: it means the
oracle's advantage is mask *size*, which condition 4 is designed to catch.
