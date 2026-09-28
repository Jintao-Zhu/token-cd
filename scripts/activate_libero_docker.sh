#!/usr/bin/env bash
# Source inside dev_zjt_container before running token-cd LIBERO rollouts.
set -euo pipefail

export TOKEN_CD="${TOKEN_CD:-/root/code/token-cd}"
export LIBERO_ENV="${LIBERO_ENV:-/root/code/task1/.venvs/openvla-ar}"
export LIBERO_REPO="${LIBERO_REPO:-/root/code/LIBERO}"
export PCD_SOURCE="${PCD_SOURCE:-/root/code/official-reproductions/pcd_openvla_simpler_box_31b027e/source/PCD}"
export OPENVLA_HF_CODE_DIR="${OPENVLA_HF_CODE_DIR:-$TOKEN_CD/third_party/openvla/prismatic/extern/hf}"
export OPENVLA_CHECKPOINT="${OPENVLA_CHECKPOINT:-/root/code/checkpoints/openvla-7b-finetuned-libero-spatial/962318cec55ac10993ff0f5f43eda9a270b4c873}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/root/code/task1/.libero}"
export HF_HOME="${HF_HOME:-/root/code/task1/.hf-cache}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$TOKEN_CD:$PCD_SOURCE:$LIBERO_REPO:$TOKEN_CD/third_party/openvla${PYTHONPATH:+:$PYTHONPATH}"

export PATH="$LIBERO_ENV/bin:$PATH"
