# token-cd: OpenVLA L11-Matched contrastive decoding

This repository studies token-level counterfactual guidance for vision-language-action policies. The current **real-robot transfer package** uses the same OpenVLA L11-Matched method implementation as our simulation experiments. It supplies a camera-to-action inference entry point and a hardware adapter contract so a collaborator can evaluate the method on a physical robot.

**Start here:** [transfer guide](https://github.com/Jintao-Zhu/token-cd/blob/real-robot-transfer/deployment/real_robot_transfer/README.md) · [method design and equations](https://github.com/Jintao-Zhu/token-cd/blob/real-robot-transfer/deployment/real_robot_transfer/METHOD.md) · [hardware adapter contract](https://github.com/Jintao-Zhu/token-cd/blob/real-robot-transfer/deployment/real_robot_transfer/HARDWARE_ADAPTER.md) · [implementation branch](https://github.com/Jintao-Zhu/token-cd/tree/real-robot-transfer)

## Method in one control step

Given an RGB image and an instruction, OpenVLA-7B produces 256 visual projector tokens and predicts a seven-token action. L11-Matched changes inference in four stages:

1. **Set a dynamic budget.** Parse source and target entities from the instruction. KMeans (`K=8`) groups the projector tokens; each entity matches its closest group by embedding cosine similarity. The union size is the current step's budget `m_t`.
2. **Find task-sensitive positions.** Average the zero-based decoder layer 11, post-softmax attention from non-special instruction tokens to visual tokens. Select the top `m_t` positions. KMeans determines *how many* tokens to alter; L11 attention determines *which* tokens to alter.
3. **Construct a counterfactual.** Replace only selected projector features with their harmonic interpolation from unselected neighbors on the 16×16 visual grid. Unselected features stay bit-identical. This is a feature-space negative branch, not pixel inpainting or an object-removal claim.
4. **Decode against the same action prefix.** Teacher-force the negative branch on the clean branch's action-token prefix. For action dimensions 0–5, use `z* = 1.5 z_clean − 0.5 z_negative`; dimension 6 (gripper) remains clean. Greedy decoding and the checkpoint's action statistics produce the seven-dimensional model action.

The [method document](https://github.com/Jintao-Zhu/token-cd/blob/real-robot-transfer/deployment/real_robot_transfer/METHOD.md) specifies attention indices, reconstruction, prompts, action decoding, and step-level audit fields. The canonical code is [`prompt_attn_shr_policy.py`](https://github.com/Jintao-Zhu/token-cd/blob/real-robot-transfer/research/semantic_token_cd/prompt_attn_shr_policy.py); the transfer runner reuses that class instead of maintaining a second algorithm implementation.

## Reproduce the inference path

Use the `real-robot-transfer` branch. The OpenVLA checkpoint is **not** stored in Git. Prepare a complete local checkpoint directory and a CUDA environment matching the model; see the [checkpoint manifest](https://github.com/Jintao-Zhu/token-cd/blob/real-robot-transfer/deployment/real_robot_transfer/TRANSFER_MANIFEST.md) and [runtime snapshot](https://github.com/Jintao-Zhu/token-cd/blob/real-robot-transfer/deployment/real_robot_transfer/SOURCE_RUNTIME.md). The physical inference path does not import SIMPLER or SAPIEN.

From the repository root, first run one saved camera image. This only writes an inference record; it sends no robot command:

```bash
python -m deployment.real_robot_transfer.runner \
  --checkpoint /models/openvla-7b \
  --unnorm-key YOUR_ROBOT_STATS_KEY \
  --instruction "put the carrot on the plate" \
  --image /data/camera_frame.png \
  --log /data/l11_matched_offline.jsonl
```

`--unnorm-key` must correspond to the actual embodiment. The stock checkpoint contains keys such as `bridge_orig` and `fractal20220817_data`; those names do not make the output compatible with an unrelated arm. The JSONL record includes an image hash, clean and guided action tokens, selected visual positions, `m_t`, decoded action, and intervention checks.

For a camera and robot connection, implement the [adapter interface](https://github.com/Jintao-Zhu/token-cd/blob/real-robot-transfer/deployment/real_robot_transfer/adapter.py) and map the model action into the controller's frame, units, gripper convention, and limits. Running with `--adapter your_package.robot_adapter:RobotAdapter` remains a **dry-run** by default. Only an explicit `--execute` invokes its `send_command()` method.

## What is and is not reproduced

The branch fixes the OpenVLA L11-Matched algorithm, its configuration, preprocessing path, action decoding, and trace checks. All reported experiments so far were in simulation. A physical-robot study still needs an embodiment-matched checkpoint/statistics, camera and robot calibration, a device-specific adapter, control timing, safety limits, and recorded real-world outcomes. The arm, camera, and driver for the planned transfer are not yet known; no physical success rate is claimed here.

## Other research in this repository

`research/semantic_token_cd/` holds the main token-level CD experiments. `research/ar_*`, `research/coreact_*`, `research/token_pcd_*`, and `research/causal_failure_gate/` hold related action-token, activation, PCD, and failure-gating investigations. Earlier reports remain available in the [SHR methods and results report](docs/SHR_COMPREHENSIVE_REPORT_2026-09-05.md) and [compact follow-up data release](github_results/SHR_FOLLOWUP_RELEASE.md). Large checkpoints and raw rollout dumps are excluded from Git.
