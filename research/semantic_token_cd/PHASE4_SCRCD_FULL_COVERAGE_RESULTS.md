# SCR-CD Phase 4 — Full-Coverage Recon Analysis (seeds 0-399)

**Experiment:** `semantic_recon_k8_m10` (frozen K=8, M=10, λ=0.5) vs vanilla / semantic-attn baselines.
**Date:** 2026-08-31
**Artifact:** `artifacts/semantic_recon_0_399_full_v1/analysis.json`
**Analyzer:** `research/semantic_token_cd/analyze_recon_0_399_full.py`

This EXTENDS Phase 2 (which covered only the 3 core tasks on 0-199) to the full
0-399 seed range and adds the 6 remaining benchmark tasks. All comparisons are
hash-verified fail-closed (a seed enters only if every involved arm agrees on both
`initial_state_sha256` and `canonical_snapshot_sha256`). Statistics: exact McNemar
(`binomtest`), seed-bootstrap 95% CI (4000 resamples, seed 731902).

Arm aliasing: `attn_semantic_k8` (0-99) == `L8-15` (100-199) == `semantic_attn_k8_l8_15`.

---

## Part 1 — Core tasks, full 0-399 (n=1199, 28 hash-fail excluded)

| task | SR vanilla | SR attn | SR recon | Δ vs vanilla | Δ vs attn | R/H | p (v) | p (a) |
|---|---|---|---|---|---|---|---|---|
| move_near | 0.632 | 0.568 | 0.618 | −1.5pp | +5.0pp | 21/27 · 68/48 | 0.47 | 0.077 |
| close_drawer | 0.509 | 0.558 | **0.672** | **+16.3pp** | **+11.6pp** | 93/28 · 96/50 | **2.4e-9** | **1.8e-4** |
| pick_coke_can | 0.263 | 0.295 | 0.325 | **+6.2pp** | +3.0pp | 70/45 · 66/54 | **0.025** | 0.32 |
| **POOLED** | 0.468 | 0.473 | 0.538 | **+7.0pp** | **+6.5pp** | 184/100 · 230/152 | **7.1e-7** | **7.7e-5** |

### close_drawer: the sole robust, cross-block-stable win

recon − vanilla net by block (all four positive, no sign-flip):

| block | n | Δ vs vanilla | p |
|---|---|---|---|
| 0-99 | 100 | +23.0pp | 3.4e-5 |
| 100-199 | 99 | +16.2pp | 0.0052 |
| 200-299 | 100 | +11.0pp | 0.052 |
| 300-399 | 100 | +15.0pp | 0.0135 |
| **0-399** | 399 | **+16.3pp** | **2.4e-9** |

This is the single most robust result of the program: recon lifts close_drawer by
~+16pp vs vanilla (p≈1e-9) and ~+12pp vs attn (p≈2e-4), with the same sign in all
four seed blocks.

### pick_coke_can reaches significance at full coverage

Phase 2 (0-199) was NS (p=0.12); with 0-399 it is now **+6.2pp vs vanilla, p=0.025**
(70 rescued vs 45 harmed). vs attn remains NS (+3.0pp, p=0.32).

### move_near: consistently neutral

−1.5pp vs vanilla (p=0.47) and +5.0pp vs attn (p=0.077, marginal). No block is
significant. Recon neither helps nor hurts move_near.

---

## Part 2 — Other 6 tasks, seeds 300-399 (5-arm, n=600, 0 hash-fail)

`semantic_recon_k8_m10` vs the four Phase-1 baselines (all same-process → perfect
hash alignment).

| task | SR vanilla | SR attn | SR merge | SR random | SR recon | vs vanilla (p) | vs attn (p) |
|---|---|---|---|---|---|---|---|
| open_drawer | 0.280 | 0.220 | 0.310 | 0.360 | **0.410** | **+13pp (0.024)** | **+19pp (0.0013)** |
| place_apple | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 (1.0) | 0 (1.0) |
| carrot_on_plate | 0.080 | 0.000 | 0.080 | 0.030 | 0.040 | −4pp (0.29) | +4pp (0.13) |
| put_eggplant | 0.010 | 0.000 | 0.000 | 0.010 | 0.010 | 0 (1.0) | +1pp (1.0) |
| spoon_on_towel | 0.000 | 0.000 | 0.000 | 0.020 | 0.030 | +3pp (0.25) | +3pp (0.25) |
| stack_cube | 0.000 | 0.000 | 0.020 | 0.000 | 0.000 | 0 (1.0) | 0 (1.0) |
| **POOLED** | 0.062 | 0.037 | 0.068 | 0.070 | 0.082 | +2.0pp (0.088) | **+4.5pp (2.5e-5)** |

**Key finding — open_drawer is the only other task where recon wins:** +13pp vs
vanilla (p=0.024) and +19pp vs attn (p=0.0013). Together with close_drawer, this
means recon helps **both** drawer tasks. The attn baseline is *harmful* on open_drawer
(0.220 vs vanilla 0.280 — blocking the semantic tokens drops SR by 6pp), and recon
recovers that plus more.

**The other 5 tasks are near-floor** (SR ≤ 8% everywhere). With success rates this
low, McNemar has no power and no arm — including recon — meaningfully moves them.
These are hard tasks for OpenVLA-7b on this seed set, not a recon-specific failure.

---

## Part 3 — Other 6 tasks, single-arm recon SR (absolute)

| task | 0-299 (n=300) | 300-399 (n=100) | 0-399 (n=400) |
|---|---|---|---|
| open_drawer | 0.373 | 0.410 | 0.383 |
| carrot_on_plate | 0.067 | 0.040 | 0.060 |
| spoon_on_towel | 0.013 | 0.030 | 0.018 |
| put_eggplant | 0.003 | 0.010 | 0.005 |
| place_apple | 0.003 | 0.000 | 0.003 |
| stack_cube | 0.000 | 0.000 | 0.000 |

Only open_drawer has a usable absolute SR (~38%); the other five are below 7%.

---

## Conclusions

1. **recon > vanilla is now confirmed at p=7.1e-7 (pooled net +84, n=1199)** on the
   core tasks, and **recon > attn at p=7.7e-5 (net +78)**.
2. **The entire core-task effect is close_drawer + pick_coke.** close_drawer is the
   robust engine (net +65, p=2.4e-9, all four blocks same sign); pick_coke becomes
   significant at full coverage (+6.2pp, p=0.025); move_near is neutral.
3. **Both drawer tasks (close_drawer + open_drawer) favor recon** — recon helps
   localized drawer manipulation, and specifically beats attn (which *hurts*
   open_drawer by 6pp).
4. **The remaining 5 benchmark tasks are at the SR floor** (≤8%) and no arm moves
   them; recon's effect there is statistically undetectable, not negative.

## Hash-fail accounting (28 unique / 1200 = 2.3%)

- 0-99: 9, 100-199: 16 (both = the Phase-2 cross-process set, 25 total), 200-299: 3
  (new, Phase-3 recon vs Phase-1A/1B baselines), 300-399: 0 (single-process 5-arm).
- All are cross-process SAPIEN float non-determinism in `env.reset(seed)` — excluded
  fail-closed, consistent with the Phase-2 characterization.

## Next step (optional, needs new rollout)

The 6 non-core tasks have **no vanilla/attn baseline on 0-299**, so recon's 0-299
coverage there can only be reported as single-arm SR. To run the full recon-vs-vanilla
paired analysis on 0-299 for those 6 tasks, backfill `vanilla` + `semantic_attn_k8_l8_15`
over 6 tasks × 300 seeds = 3600 episodes (~10h on 5 GPUs). This is the only remaining
gap for a complete 9-task × 0-399 paired report.
