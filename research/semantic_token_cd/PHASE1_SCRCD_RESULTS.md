# Phase 1 — Semantic Token-CD (SCR-CD) Results

**Protocol:** `SCR_CD_SEMANTIC_RECON_K8_M10_V1`
**Decision:** **METHOD_GO** ✅ (all four binding criteria satisfied)
**Data:** `artifacts/semantic_recon_k8_m10_v1` (paired rollout, 9 tasks × 100 seeds × 5 arms)

## What was run

Full paired rollout over 9 tasks, seeds 300–399, 100 paired seeds/task, 5 arms:

| Arm | Degradation | Selector |
|-----|-------------|----------|
| `vanilla` | none (baseline) | — |
| `semantic_attn_k8_l8_15` | block Action Query → selected visual keys (L8–15, mask −1e4) | KMeans K=8 semantic |
| `semantic_merge_k8_eta100` | collapse each selected group to its prototype (η=1.0) | KMeans K=8 semantic |
| `semantic_recon_k8_m10` | **the method** — reconstruction `C = V\G` kept, `G` tokens → context `v̂_i` only | KMeans K=8 semantic |
| `random_recon_k8_m10` | same reconstruction, random-selector control | random K=8 |

Paired design: all 5 arms restore the **same** scene (`capture_snapshot`/`restore_snapshot`),
so every seed is a within-scene counterfactual — 500 episodes/task, 4500 total.

## Method (`semantic_recon`, locked)

- KMeans K=8 semantic selector; `G = G_source ∪ G_target`; reconstruction set `C = V\G`.
- `M = 10` basis tokens `B` via deterministic FPS (cosine distance, L2-normalized for FPS,
  RAW features for reconstruction).
- Ridge `α_i = (B_c^T B_c + ρI)^{-1} B_c^T (v_i − μ_B)`, `ρ = 1e-3·tr(B_c^T B_c)/M` (fixed M=10).
- Decomposition `v_i = v̂_i (context) + e_i (region-unique)`; negative branch keeps only `v̂_i` for `i∈G`.
- 256 tokens preserved; CD `z* = z+ + 0.5(z+ − z−)`, λ=0.5, first 6 action tokens,
  7th (gripper) positive-only, greedy, shared guided prefix, reconstruction once per control observation.

## Results (paired, ΔSR vs vanilla, n = 882 complete 5-arm pairs)

| Arm | ΔSR_macro | McNemar p | net | non-worse | worst task |
|-----|-----------|-----------|-----|-----------|------------|
| `semantic_attn_k8_l8_15` | −0.44 pp | 0.78 | −4 | 5/9 | −8 pp (carrot) |
| `semantic_merge_k8_eta100` | +0.89 pp | 0.54 | +8 | 7/9 | −8 pp (move_near) |
| `random_recon_k8_m10` | +0.44 pp | 0.78 | +4 | 7/9 | −11 pp (move_near) |
| **`semantic_recon_k8_m10`** | **+4.22 pp** | **0.0005** | **+38** | **8/9** | **−4 pp (carrot)** |

### Per-task ΔSR for `semantic_recon_k8_m10` (method)

| Task | SR_vanilla | SR_method | ΔSR | 95% CI |
|------|-----------|-----------|-----|--------|
| close_drawer | 0.50 | 0.65 | **+0.15** | [0.04, 0.26] |
| move_near | 0.71 | 0.72 | +0.01 | [−0.06, 0.08] |
| open_drawer | 0.28 | 0.41 | **+0.13** | [0.03, 0.23] |
| pick_coke_can | 0.20 | 0.30 | **+0.10** | [0.01, 0.20] |
| place_apple | 0.00 | 0.00 | 0.00 | — |
| carrot_on_plate | 0.08 | 0.04 | **−0.04** | [−0.10, 0.01] |
| put_eggplant | 0.01 | 0.01 | 0.00 | — |
| spoon_on_towel | 0.00 | 0.03 | +0.03 | [0.00, 0.07] |
| stack_cube | 0.00 | 0.00 | 0.00 | — |

## Gate decision (binding criteria)

| Criterion | Threshold | Observed | Pass |
|-----------|-----------|----------|------|
| ΔSR_macro | ≥ +3 pp | +4.22 pp | ✅ |
| non-worse tasks | ≥ 6/9 | 8/9 | ✅ |
| min ΔSR | > −10 pp | −4.0 pp | ✅ |
| SemanticRecon > RandomRecon | strict | +4.22 vs +0.44 | ✅ |

**Key dissociation:** reconstruction with a *semantic* selector (+4.22 pp, p=0.0005) far
outperforms reconstruction with a *random* selector (+0.44 pp, p=0.78). The signal is in
**which** tokens are preserved — i.e. the semantic selector picks the task-relevant
entities, and degrading exactly those tokens produces the useful contrastive direction.

The only worse task is `carrot_on_plate` (−4 pp, both arms already near-floor at 8%/4%
SR), consistent with the known EOS-early-termination + near-floor instability on the
widowx carrot task; the dominant positive signal comes from the google-robot
grasping/insertion tasks (close_drawer +15, open_drawer +13, pick_coke +10).

## Coverage note

Rollout reached 4488/4500 episode summaries at analysis time (882/900 complete 5-arm
pairs); the remaining 12 episodes are `place_apple` (0% SR across all arms) and cannot
change the decision. Final table will be re-emitted by `analyze_semantic_recon.py` once
all 45 `.done` markers land.
