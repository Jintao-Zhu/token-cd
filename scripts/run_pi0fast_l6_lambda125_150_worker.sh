#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 6 ]]; then
  echo "usage: $0 GPU PORT FAMILY MODE OUT_DIR TASK:START-END [...]" >&2
  exit 2
fi
GPU="$1"; PORT="$2"; FAMILY="$3"; MODE="$4"; OUT="$5"
shift 5
case "$GPU" in 1|2|3) ;; *) echo "Refusing GPU $GPU; only GPUs 1-3 are authorized" >&2; exit 2 ;; esac
case "$FAMILY" in google|widowx) ;; *) echo "unknown family: $FAMILY" >&2; exit 2 ;; esac
case "$MODE" in preflight|full) ;; *) echo "unknown mode: $MODE" >&2; exit 2 ;; esac
ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD="$ROOT/token-cd"
TASK_TAG="$(printf '%s_' "$@" | tr ':/' '__' | tr -cd '[:alnum:]_-')"
mkdir -p "$OUT/logs"
LOG="$OUT/logs/${MODE}_gpu${GPU}_${FAMILY}_${TASK_TAG}.log"
exec >> "$LOG" 2>&1
source /home/leju-suzhou/miniconda3/etc/profile.d/conda.sh
conda activate pi0
export CUDA_VISIBLE_DEVICES="$GPU"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.20 JAX_PLATFORMS=cpu
export PI0FAST_WIDOWX_LAYOUT=bridge PI0FAST_WIDOWX_CLIP=1
export PYTHONPATH="/home/leju-suzhou/.cache/pi0fast_client_deps:$ROOT/openpi/packages/openpi-client/src:$ROOT/pcd_openvla_simpler_box_31b027e/source:$ROOT/SimplerEnv:$ROOT/SpatialVLA-SimplerEnv:$ROOT/SpatialVLA-SimplerEnv/ManiSkill2_real2sim:$ROOT/SpatialVLA/ManiSkill2_real2sim:$TOKEN_CD:$ROOT/openpi/src"
exec python -u "$TOKEN_CD/research/semantic_token_cd/pi0_fast_l6_lambda125_150_9task.py" \
  --gpu "$GPU" --port "$PORT" --family "$FAMILY" --out "$OUT" --mode "$MODE" --job "$@"
