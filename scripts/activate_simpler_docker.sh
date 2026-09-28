#!/usr/bin/env bash
# SIMPLER/OpenVLA runtime inside dev_zjt_container.
set -euo pipefail

export TOKEN_CD="${TOKEN_CD:-/root/code/token-cd}"
export PCD_ROOT="${PCD_ROOT:-/root/code/official-reproductions/pcd_openvla_simpler_box_31b027e}"
export PCD_SOURCE="${PCD_SOURCE:-$PCD_ROOT/source/PCD}"
export SIMPLER_ENV="${SIMPLER_ENV:-/root/code/task1/.venvs/openvla-ar}"
export OPENVLA_BASE_CHECKPOINT="${OPENVLA_BASE_CHECKPOINT:-$PCD_SOURCE/pretrained/openvla-7b}"

export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$TOKEN_CD:$PCD_SOURCE:$TOKEN_CD/third_party/openvla${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$SIMPLER_ENV/bin:$PATH"
