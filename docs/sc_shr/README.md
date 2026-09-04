# SC-SHR: spatial ablation and guidance-strength analysis

This branch adds the SC-SHR (Spatially Coherent SHR) ablation and the offline
lambda / RGB overlay diagnostics that followed the main SHR release.

## SC-SHR idea

Hypothesis under test: part of SHR's failure comes from spatially scattered
semantic clusters, not from wrong grounding. SC-SHR keeps the K=8 semantic
selector and harmonic reconstruction unchanged and only spatially cleans the
deleted region before inpainting:

1. build the semantic union mask (same as SHR);
2. run 4-neighbor connected-component analysis on the 16x16 grid;
3. keep the largest component for single-entity tasks and the top-2 components
   for source-target tasks;
4. harmonic-inpaint only the kept region.

## Local 100-seed result (seeds 300-399, 386 valid pairs)

| task | Vanilla | SHR | SC-SHR |
| --- | --: | --: | --: |
| close_drawer | 49.5% | 67.0% | 61.9% |
| open_drawer | 18.6% | 32.0% | 34.0% |
| pick_coke_can | 25.0% | 30.0% | 25.0% |
| move_near | 71.7% | 73.9% | 55.4% |

Conclusion of the simple top-k component rule: it does not help (net
SHR vs SC-SHR = rescue 46 / harm 71). The strongest regression is move_near.
Important caveat: several move_near "harms" have zero removed tokens (SC and SHR
share the same first-step mask), which indicates environment-reproduction noise
rather than the filter. RGB overlays in this repo's artifact notes are provided
for manual inspection of the removed satellite regions.

## Guidance-strength (lambda) sweep

`lambda_strength_rollout.py` runs strictly paired arms (lambda 0 / 0.25 / 0.5)
for the same (task, seed) snapshot inside one process, so rescue/harm tables are
clean. `analyze_offline_lambda.py` reports what can be computed offline from the
saved positive/negative logits (first-step action response) and the known
lambda 0 vs 0.5 outcome points.

## Files

- `research/semantic_token_cd/sc_shr_policy.py` - SC-SHR policy (component filter).
- `research/semantic_token_cd/sc_shr_local_rollout.py` - local 100-seed single-arm run.
- `research/semantic_token_cd/sc_shr_canonical_rollout.py` - snapshot-pickle based driver.
- `research/semantic_token_cd/run_sc_shr_local_4task.sh` - 4-task dispatcher.
- `research/semantic_token_cd/lambda_strength_rollout.py` - strictly paired lambda sweep.
- `research/semantic_token_cd/run_lambda_strength_4task.sh` - lambda dispatcher.
- `research/semantic_token_cd/analyze_offline_lambda.py` - offline lambda analysis.
- `research/semantic_token_cd/sc_shr_overlay_render.py` / `sc_shr_overlay_plot.py` -
  RGB overlay diagnostics for SHR vs SC-SHR removed regions.
