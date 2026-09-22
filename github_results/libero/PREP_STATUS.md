# LIBERO preparation status

## Environment

- Env: `/home/leju-suzhou/zjt_ws/libero-openvla-env`
- Activation: `source /home/leju-suzhou/zjt_ws/token-cd/scripts/activate_libero_openvla.sh`
- LIBERO repo: `/home/leju-suzhou/zjt_ws/openpi/third_party/libero`
- LIBERO config: `/home/leju-suzhou/zjt_ws/tmp/libero_cfg/config.yaml`
- OpenVLA code dir: `/home/leju-suzhou/zjt_ws/token-cd/third_party/openvla/prismatic/extern/hf`
- `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`
- `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`

## Checkpoints

| suite | task suite | unnorm key | checkpoint |
|---|---|---|---|
| LIBERO-Spatial | libero_spatial | libero_spatial | `/home/leju-suzhou/zjt_ws/checkpoints/libero/openvla-7b-finetuned-libero-spatial` |
| LIBERO-Object | libero_object | libero_object | `/home/leju-suzhou/zjt_ws/checkpoints/libero/openvla-7b-finetuned-libero-object` |
| LIBERO-Goal | libero_goal | libero_goal | `/home/leju-suzhou/zjt_ws/checkpoints/libero/openvla-7b-finetuned-libero-goal` |
| LIBERO-Long | libero_10 | libero_10 | `/home/leju-suzhou/zjt_ws/checkpoints/libero/openvla-7b-finetuned-libero-10` |

All four checkpoint downloads report `DONE` and each contains 4 safetensors shards.

## Smoke checks

- LIBERO-Spatial `OffScreenRenderEnv` reset: pass
- Initial state load: pass
- OpenVLA-Spatial checkpoint decode: action shape `(7,)`, finite: pass
- OpenVLA-Object checkpoint decode: action shape `(7,)`, finite: pass
- OpenVLA-Goal checkpoint decode: action shape `(7,)`, finite: pass
- OpenVLA-Long checkpoint decode: action shape `(7,)`, finite: pass
- One real LIBERO-Spatial env step with predicted action: pass

## Not yet done

- Unified L11 + Matched policy adapter: done (see below)
- Fixed / Shuffle / Wrong-query / Random-cluster arms for LIBERO
- Result manifests and paired analysis


## Matched migration (2026-09-21)

- Runner: `research/semantic_token_cd/libero_matched_rollout.py`
- 100-episode orchestrator: `research/semantic_token_cd/run_libero_matched_spatial_100.sh`
- Protocol: LIBERO-Spatial, 10 tasks x 100 episodes, matched-only.
- Locked components:
  - KMeans K=8, seed=0 on projector features
  - per-entity top-1 cosine cluster matching, union of matched clusters -> m
  - L11 prompt-attention Top-m (ascending token-id tie break)
  - beta=0 16x16 four-neighbor harmonic reconstruction
  - lambda=0.5 guided-prefix contrastive decoding; seventh action dimension preserved
  - EOS suppressed during clean AR action generation
- Validation completed:
  - clean OpenVLA action parity: decoded clean action equals official `predict_action` exactly
  - 10-task x 3-step smoke: all actions finite
  - 300-step full smoke (`pick_up_the_black_bowl_between...`): success, 80 steps, mean m=36.35
