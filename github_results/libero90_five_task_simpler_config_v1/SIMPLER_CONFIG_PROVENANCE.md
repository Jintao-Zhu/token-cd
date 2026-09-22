# SIMPLER Matched 213/400 Configuration Provenance

## Source result

- Baseline: `Matched = 213/400 (53.2%)`
- Evaluation: 4 tasks x 100 paired seeds, seeds `100-199`
- Evidence:
  - `/home/leju-suzhou/zjt_ws/token-cd/docs/2026-09-17_matched_vs_topp_INVESTIGATION.md`
  - `/home/leju-suzhou/zjt_ws/token-cd/docs/2026-09-18_EXPERIMENT_LOG.md`
  - `/home/leju-suzhou/zjt_ws/token-cd/artifacts/prompt_attn_l11_token_count_v1/CONFIG_LOCK.json`
  - `/home/leju-suzhou/zjt_ws/token-cd/artifacts/prompt_attn_l11_token_count_v1/statistics/success_rates.json`
  - `/home/leju-suzhou/zjt_ws/token-cd/artifacts/prompt_attn_l11_token_count_v1/rollout_logs/`

The 4-task 213/400 result predates the LIBERO-Spatial `λ=0.125` repair experiments. It must not be replaced by the Spatial configuration.

## Frozen method configuration

| Component | Frozen value | Evidence |
|---|---|---|
| Attention layer | L11 (zero-based) | `prompt_attn_l11_token_count_v1/CONFIG_LOCK.json`, `locked_downstream.attention_layers=[11]` |
| Query | Full instruction, excluding special/template/padding tokens | Same `CONFIG_LOCK.json`, `query` |
| Heads | Equal arithmetic mean over all heads | Same `CONFIG_LOCK.json`, `head_query_aggregation` |
| KMeans features | projector features `h_i`, 256 visual tokens | `distractor_policy.py`, `_semantic_clusters` |
| KMeans | K=8, seed=0, n_init=10 | `prompt_attn_shr_rollout.py` constants and `distractor_policy.py` |
| Entity embedding | mean-pooled `embed_tokens` of source/target phrase | `rollout_policy.py`, `_embed_phrase`; `extract_entities` |
| Entity matching | entity set = source union target; per entity top-1 cosine matched KMeans group; group IDs deduplicated | `distractor_policy.py`, `_semantic_clusters`; `prompt_attn_shr_policy.py`, matched branch |
| Budget | `m = |union token set of matched clusters|`, every control step; no cap/floor | `prompt_attn_shr_policy.py`, matched branch |
| Mask | `stable_top_m` over L11 visual scores; `m` tokens; ascending token index tie-break | `prompt_attn_shr_policy.py`, `stable_top_m` |
| Harmonic | 16x16 four-neighbor Dirichlet, beta=0, gamma=1 | `st_shr_policy.py`, `harmonic_reconstruct`; CONFIG_LOCK |
| Guidance | `z*_(0:6) = (1+lambda) z+_(0:6) - lambda z-_(0:6)`; dimension 6 from clean positive branch | `st_shr_policy.py`, `_combine_action_scores`; CONFIG_LOCK |
| Lambda | `0.5` | `CONFIG_LOCK.json`, `locked_downstream.lambda=0.5` |
| Decoding | deterministic greedy, shared clean greedy prefix | `CONFIG_LOCK.json`, `sampling=false`, `prefix` |
| Scheduling | recompute entity selection, KMeans, budget, and mask at every control decision | `PrompAttentionSHRInference._forward_scores` call path |

## Conflict resolution

The LIBERO-Spatial results use a different `λ` and newer ranking repairs. They are not the source for this run. This five-task LIBERO-90 evaluation uses the `PromptAttentionSHRInference` matched configuration above.

## Environment adaptation (not method parameters)

- Checkpoint: `VQ-VLA/openvla-7b-finetuned-libero-90`
- Revision: `794ef81b7be928ea9270e81ca1ef5b60ffa9420f`
- Action statistics: `libero_90_no_noops`, explicitly injected from `dataset_statistics.json`
- Environment protocol: 10 no-op warmup, 400 maximum control steps, 50 official init states per selected task
- Image preprocessing: existing LIBERO runtime `prepare_agentview` path
- Entity parser: `source_target_libero90`, preserving source/destination roles and not adding the relation object to the budget
