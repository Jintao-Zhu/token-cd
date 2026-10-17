# Transfer manifest

## Git source

- Repository: `Jintao-Zhu/token-cd`
- Transfer branch: `real-robot-transfer`
- Base implementation at bundle creation: `5fff4f061` (`main` before this branch's transfer documents)
- Clone the branch rather than copying an arbitrary working tree.
- The runnable, simulator-free inference entry is `deployment/real_robot_transfer/runner.py`.

## Method files

These files are already tracked by the repository and arrive with the branch:

- `research/semantic_token_cd/prompt_attn_shr_policy.py` — L11 prompt-attention selection, Matched budget handling, harmonic negative construction, and projector guidance implementation.
- `research/semantic_token_cd/prompt_attn_l11_budget_causal_rollout.py` — current policy construction and experiment configuration helpers; SIMPLER-specific rollout portions are not the physical-robot adapter.
- `research/semantic_token_cd/st_shr_policy.py` — harmonic reconstruction and shared SHR inference base.
- `research/semantic_token_cd/global_merge_policy.py` — model-forward/intervention hooks.
- `research/semantic_token_cd/distractor_policy.py` — OpenVLA action-logit and token decode helpers.
- `research/ar_token_counterfactual/intervention.py` — projector intervention primitives.
- `deployment/real_robot_transfer/openvla_runtime.py` — minimal OpenVLA wrapper without SIMPLER imports.
- `deployment/real_robot_transfer/policy.py` — canonical settings and fail-closed trace checks.
- `deployment/real_robot_transfer/runner.py`, `adapter.py`, `example_adapter.py` — offline and hardware entry points.
- `deployment/real_robot_transfer/METHOD.md` — step-by-step algorithm and equations.

Current method settings to preserve for the first physical pilot:

| Setting | Value |
|---|---|
| Model | OpenVLA-7B |
| Selector | prompt-to-visual attention, decoder layer L11 (zero-based index 11) |
| Budget | dynamic Matched budget |
| Negative branch | harmonic reconstruction of selected projector tokens |
| Intervention point | projector output |
| Guidance | logit contrast, `lambda=0.5` |
| Action dimensions guided | first six continuous dimensions; gripper left on clean branch |
| Decode | greedy action-token decoding |

Do not silently translate these settings into another robot's action schema. Any change to action dimensions, gripper handling, normalization, or camera input is a method adaptation and should be recorded separately.

## Model files outside Git

Local model directory: `/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source/pretrained/openvla-7b` (about 15 GB).

Transfer the full directory, including:

- `model-00001-of-00003.safetensors`
- `model-00002-of-00003.safetensors`
- `model-00003-of-00003.safetensors`
- `model.safetensors.index.json`
- `config.json`, `configuration_prismatic.py`, `modeling_prismatic.py`
- processor/tokenizer files: `processing_prismatic.py`, `preprocessor_config.json`, `processor_config.json`, `tokenizer.json`, `tokenizer.model`, and their config/token maps

The local model card declares MIT. Keep the model card and license file with the copied checkpoint. Do not put the weight shards in a normal Git commit; use a model registry or direct file transfer if a hosted copy is needed.

## Runtime snapshot on the source host

- Python 3.10.20
- PyTorch 2.7.1+cu128 (PyTorch CUDA build 12.8)
- Transformers 4.40.1
- Tokenizers 0.19.1
- timm 0.9.10
- NumPy 1.24.4
- SAPIEN 2.2.2 (simulation only; not needed for camera-to-action hardware inference)
- NVIDIA driver observed on source host: 570.133.20

See [SOURCE_RUNTIME.md](SOURCE_RUNTIME.md) for portability notes.
