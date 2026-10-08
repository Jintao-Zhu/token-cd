# Real-robot transfer bundle

This directory is a migration guide and manifest for the current OpenVLA + L11-Matched implementation. It is not a turnkey hardware controller: the current rollout code is SIMPLER-oriented, and a robot-specific observation/action adapter must be connected and validated on the destination machine.

## What is in Git

Clone this branch to obtain the method implementation and transfer notes:

```bash
git clone --branch real-robot-transfer git@github.com:Jintao-Zhu/token-cd.git
```

The method implementation is in the cloned repository. The important files are listed in [TRANSFER_MANIFEST.md](TRANSFER_MANIFEST.md). The OpenVLA checkpoint and Python environment are kept out of GitHub; use the separate copy and verification scripts in `tools/`.

## Model used on this machine

The local OpenVLA-7B checkpoint is approximately 15 GB and is MIT-licensed according to its model card. Its model directory contains three `safetensors` shards, the shard index, model configuration, tokenizer, and image processor configuration. Transfer the **whole** directory so the index and all referenced shards stay together.

Current source path:

```text
/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source/pretrained/openvla-7b
```

The model's `config.json` contains the OpenVLA normalization statistics, including `bridge_orig` and `fractal20220817_data`. For a real robot, choose statistics/action normalization that match that embodiment and control interface. The SIMPLER normalization key must not be assumed to match a physical robot.

## Copy the model outside Git

From the source machine, copy the checkpoint directory to a destination path using SSH/rsync. The helper does not delete destination files:

```bash
deployment/real_robot_transfer/tools/copy_checkpoint.sh \
  /home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source/pretrained/openvla-7b \
  user@robot-host:/models/openvla-7b
```

Generate a manifest on the source, copy that small manifest alongside the weights, then verify it on the destination:

```bash
deployment/real_robot_transfer/tools/checkpoint_manifest.sh create \
  /home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source/pretrained/openvla-7b
# On the destination, after copying the generated SHA256SUMS file:
deployment/real_robot_transfer/tools/checkpoint_manifest.sh verify \
  /models/openvla-7b /models/openvla-7b/SHA256SUMS
```

The source Python environment is about 7.6 GB and includes simulation-only packages. Recreate a destination environment from its own CUDA/robot-driver stack instead of copying this environment directory byte-for-byte. A version snapshot is in [SOURCE_RUNTIME.md](SOURCE_RUNTIME.md).

## Before connecting a physical robot

Implement the destination-specific adapter described in [HARDWARE_ADAPTER.md](HARDWARE_ADAPTER.md). It must define camera selection and RGB preprocessing, proprioception, prompt formatting, action unnormalization, coordinate frames, gripper mapping, control frequency, action limits, and emergency-stop behavior. Then validate the adapter in observation-only mode and with bounded low-speed motions before enabling autonomous rollouts.

The existing RT/SAPIEN settings are simulator settings. Real-camera inference does not need RT rendering, EGL, Vulkan, or SAPIEN; it does need the same camera view and image preprocessing assumed by the model.
