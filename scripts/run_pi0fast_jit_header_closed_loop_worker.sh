#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "usage: $0 GPU TASK PORT" >&2
  exit 2
fi
GPU="$1"
TASK="$2"
PORT="$3"
ROOT=/home/leju-suzhou/zjt_ws
ARTIFACT="$ROOT/token-cd/artifacts/pi0_fast_jit_header_closed_loop_v1"
mkdir -p "$ARTIFACT/logs"
exec > >(tee -a "$ARTIFACT/logs/worker_${TASK}_gpu${GPU}.log") 2>&1

source "$ROOT/../miniconda3/etc/profile.d/conda.sh"
conda activate pi0
export CUDA_VISIBLE_DEVICES="$GPU"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.28
export JAX_PLATFORMS=cuda
export PYTHONPATH="/home/leju-suzhou/.cache/pi0fast_client_deps:$ROOT/openpi/packages/openpi-client/src:$ROOT/pcd_openvla_simpler_box_31b027e/source:$ROOT/SimplerEnv:$ROOT/SpatialVLA-SimplerEnv:$ROOT/SpatialVLA-SimplerEnv/ManiSkill2_real2sim:$ROOT/SpatialVLA/ManiSkill2_real2sim:$ROOT/token-cd:$ROOT/openpi/src${PYTHONPATH:+:$PYTHONPATH}"

exec python -u "$ROOT/token-cd/research/semantic_token_cd/pi0_fast_jit_header_closed_loop.py" \
  --task "$TASK" --gpu "$GPU" --port "$PORT" \
  --rt-depth 8 --seeds 0 1 --out "$ARTIFACT"
