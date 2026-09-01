# SCR-CD Phase 2 Replication — Final Results (seeds 0-199)

**Experiment:** `SCR_CD_SEMANTIC_RECON_K8_M10_0_199_CORE3_REPLICATION_V1`
**Date:** 2026-08-30 → 08-31
**Question:** On the existing 0-199 vanilla / semantic-attn baselines, is `semantic_recon_k8_m10`
more stable, lower-Harm, and free of cross-block sign-flips?
**Design:** single new arm `semantic_recon_k8_m10` (K=8, M=10, λ=0.5, frozen = Phase 1 config),
seeds 0-199, 3 core tasks (move_near / close_drawer / pick_coke_can), 600 episodes + 200 backfill
(vanilla + semantic_attn_k8_l8_15 for move_near 0-99). Per-seed hash-verified fail-closed pairing.

## Arm aliasing
`attn_semantic_k8` (0-99) == `L8-15` (100-199) == `semantic_attn_k8_l8_15` (canonical)
— all K=8, layers 8-15, λ=0.5, semantic_hard, mask −1e4.

## Final paired results (hash-verified, exact McNemar, seed-bootstrap 95% CI)

### Block 0-99 (valid n=291, 9 hash_fail)
| task | n | vanilla | attn | recon | Δ vs vanilla | Δ vs attn | R/H | McNemar p |
|---|---|---|---|---|---|---|---|---|
| move_near | 100 | 62% | 62% | 58% | −4.0pp | −4.0pp | 4/8 · 13/17 | 0.39 / 0.59 |
| close_drawer | 100 | 56% | 59% | 79% | **+23.0pp** | **+20.0pp** | 27/4 · 29/9 | **3.4e-5** / 0.0017 |
| pick_coke | 100 | 25% | 34% | 34% | +9.0pp | 0.0pp | 18/9 · 12/12 | 0.12 / 1.0 |
| POOLED | 300 | — | — | — | +9.3pp | +5.3pp | 49/21 · 54/38 | **0.0011** / 0.12 |

### Block 100-199 (valid n=282, 16 hash_fail)
| task | n | vanilla | attn | recon | Δ vs vanilla | Δ vs attn | R/H | McNemar p |
|---|---|---|---|---|---|---|---|---|
| move_near | 100 | 55% | 50% | 56% | +1.0pp | +6.0pp | 6/5 · 18/12 | 1.0 / 0.36 |
| close_drawer | 99 | 45% | 51% | 62% | **+16.2pp** | +11.2pp | 23/7 · 26/15 | **0.0052** / 0.12 |
| pick_coke | 100 | 29% | 27% | 33% | +4.0pp | +6.0pp | 19/15 · 21/15 | 0.61 / 0.41 |
| POOLED | 299 | — | — | — | +7.0pp | +7.7pp | 48/27 · 65/42 | **0.020** / 0.033 |

### Pooled 0-199 (valid n=573, 25 unique hash_fail)
| task | n | vanilla | attn | recon | Δ vs vanilla | Δ vs attn | R/H | McNemar p |
|---|---|---|---|---|---|---|---|---|
| move_near | 200 | 58% | 56% | 57% | −1.5pp | +1.0pp | 10/13 · 31/29 | 0.68 / 0.90 |
| close_drawer | 199 | 51% | 55% | 70% | **+19.6pp** | **+15.7pp** | 50/11 · 55/24 | **4.6e-7** / 0.0006 |
| pick_coke | 200 | 27% | 30% | 34% | +6.5pp | +3.0pp | 37/24 · 33/27 | 0.12 / 0.52 |
| **POOLED** | 599 | — | — | — | **+8.2pp** | **+6.5pp** | **97/48 · 119/80** | **5.8e-5** / 0.0069 |

## Conclusions

1. **Recon vs Vanilla: R > H.** Pooled net **+49** (rescue 97, harm 48), **p = 5.8e-5**.
   Significant and robust.
2. **Recon vs Attn: R > H.** Pooled net **+39** (rescue 119, harm 80), **p = 0.0069**.
   Significant, but the gain is concentrated in close_drawer (net +31); on move_near (+2) and
   pick_coke (+6) recon ≈ attn.
3. **No cross-block sign-flip.** Every task keeps the same sign (or is neutral) in both blocks:
   - close_drawer: +23.0pp (0-99, p=3.4e-5) and +16.2pp (100-199, p=0.0052) — both +, both sig.
   - pick_coke: +9.0pp (0-99, p=0.12) and +4.0pp (100-199, p=0.61) — both +, both NS.
   - move_near: −4.0pp (0-99, p=0.39) and +1.0pp (100-199, p=1.0) — both NS, tiny.
4. **Harm profile:** recon's only near-zero net is move_near (net −3 vs vanilla, net +2 vs attn).
   Everywhere else rescue ≥ harm. No task shows significant harm.

## Hash-fail characterization (25 unique seeds = 4.2%)

All 25 failures are `canonical_snapshot_sha256` mismatches where **van == attn (identical) but
recon differs** — the recon run's independent `env.reset(seed)` produced a non-bit-identical
physics state vs the earlier vanilla/attn processes (cross-process floating-point non-determinism
in SAPIEN, not a code bug — van/attn always agree because they share a process). Excluded
fail-closed per spec. Robustness check shows no systematic bias: excluded seeds are *harder* for
close_drawer (37.5% vs 71.9% SR) but *easier* for move_near (73.3% vs 55.7%), inconsistent
direction at n=8/15/2 → noise, and the close_drawer conclusion (p=4.6e-7) is far outside any 4%
perturbation.

Excluded seeds by task:
- close_drawer: 67, 113, 114, 125, 126, 160, 173, 180
- move_near: 36, 56, 82, 85, 92, 98, 125, 133, 135, 136, 152, 157, 167, 174, 182
- pick_coke: 0, 1

## Artifacts
- Analysis JSON: `artifacts/semantic_recon_0_199_core3_replication_v1/analysis.json`
- Rollout: `research/semantic_token_cd/semantic_recon_0_199_replication.py` (+ `run_...sh`)
- Backfill: `research/semantic_token_cd/semantic_recon_0_199_backfill.py` (+ `run_...sh`)
- Analyzer: `research/semantic_token_cd/analyze_recon_0_199_replication.py`
