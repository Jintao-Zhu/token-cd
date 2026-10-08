#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 GPU WORKER_ID OUT_DIR SHARD_JSON CHECKPOINT PORT" >&2
  exit 2
fi
GPU="$1"
WORKER_ID="$2"
OUT_DIR="$3"
SHARD_JSON="$4"
CHECKPOINT="$5"
PORT="$6"
ROOT="/home/leju-suzhou/zjt_ws"
CONDA_ROOT="/home/leju-suzhou/miniconda3"
PYTHON="$CONDA_ROOT/envs/pi0/bin/python"
ROLLOUT="$ROOT/token-cd/research/semantic_token_cd/pi0_fast_simpler_l6_matched_rollout.py"
LOG="$OUT_DIR/logs/worker_${WORKER_ID}_gpu${GPU}.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate pi0
export CUDA_VISIBLE_DEVICES="$GPU"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.28
export JAX_PLATFORMS=cuda
export PYTHONPATH="/home/leju-suzhou/.cache/pi0fast_client_deps:$ROOT/openpi/packages/openpi-client/src:$ROOT/pcd_openvla_simpler_box_31b027e/source:$ROOT/SimplerEnv:$ROOT/SpatialVLA-SimplerEnv:$ROOT/SpatialVLA-SimplerEnv/ManiSkill2_real2sim:$ROOT/SpatialVLA/ManiSkill2_real2sim:$ROOT/token-cd:$ROOT/openpi/src${PYTHONPATH:+:$PYTHONPATH}"

for attempt in 1 2 3; do
  echo "$(date '+%F %T') worker=$WORKER_ID gpu=$GPU attempt=$attempt starting"
  if "$PYTHON" -u "$ROLLOUT" \
      --gpu "$GPU" --worker-id "$WORKER_ID" --shard "$SHARD_JSON" \
      --out "$OUT_DIR" --checkpoint "$CHECKPOINT" --port "$PORT"; then
    echo "$(date '+%F %T') worker=$WORKER_ID gpu=$GPU complete"
    exit 0
  else
    rc=$?
  fi
  echo "$(date '+%F %T') worker=$WORKER_ID gpu=$GPU attempt=$attempt failed rc=$rc"
  if [[ "$attempt" -lt 3 ]]; then
    echo "$(date '+%F %T') retrying unfinished seed-paired cases after technical restart"
    sleep 30
  fi
done

echo "$(date '+%F %T') worker=$WORKER_ID exhausted retries; inspect $LOG" >&2
exit 1
