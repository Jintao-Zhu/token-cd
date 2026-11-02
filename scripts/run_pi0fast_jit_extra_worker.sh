#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 6 ]]; then
  echo "usage: $0 GPU PORT TASK FAMILY START END" >&2
  exit 2
fi
GPU="$1"
PORT="$2"
TASK="$3"
FAMILY="$4"
START="$5"
END="$6"
ROOT=/home/leju-suzhou/zjt_ws
OUT="$ROOT/token-cd/runs/pi0fast_jit_header_extra5_100seed_rtfull_v1"
mkdir -p "$OUT/logs"
exec >> "$OUT/logs/worker_${TASK}_${START}_${END}_gpu${GPU}.log" 2>&1

source /home/leju-suzhou/miniconda3/etc/profile.d/conda.sh
conda activate pi0
export CUDA_VISIBLE_DEVICES="$GPU"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.28
export JAX_PLATFORMS=cpu
export PI0FAST_WIDOWX_LAYOUT=bridge PI0FAST_WIDOWX_CLIP=1
export PYTHONPATH="/home/leju-suzhou/.cache/pi0fast_client_deps:$ROOT/openpi/packages/openpi-client/src:$ROOT/pcd_openvla_simpler_box_31b027e/source:$ROOT/SimplerEnv:$ROOT/SpatialVLA-SimplerEnv:$ROOT/SpatialVLA-SimplerEnv/ManiSkill2_real2sim:$ROOT/SpatialVLA/ManiSkill2_real2sim:$ROOT/token-cd:$ROOT/openpi/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$FAMILY" == google ]]; then
  CKPT="$ROOT/checkpoints/openpi_fast_finetuned/fractal"
else
  CKPT="$ROOT/checkpoints/openpi_fast_finetuned/bridge"
fi
SHARD="$OUT/manifests/${TASK}_${START}_${END}.json"
exec python -u "$ROOT/token-cd/research/semantic_token_cd/pi0_fast_simpler_l6_matched_rollout.py" \
  --gpu "$GPU" --shard "$SHARD" --host 127.0.0.1 --port "$PORT" \
  --out "$OUT" --checkpoint "$CKPT" --rt-depth 8
