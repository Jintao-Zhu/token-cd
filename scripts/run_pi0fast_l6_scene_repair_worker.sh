#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 5 ]]; then
  echo "usage: $0 GPU PORT TASK MODE SEEDS_CSV" >&2
  exit 2
fi
GPU="$1" PORT="$2" TASK="$3" MODE="$4" SEEDS="$5"
case "$GPU" in 6|7) ;; *) echo "Refusing render GPU $GPU; this repair uses only GPUs 6 and 7" >&2; exit 2 ;; esac
case "$TASK" in google_robot_move_near|widowx_put_eggplant_in_basket) ;; *) echo "unexpected task: $TASK" >&2; exit 2 ;; esac
case "$MODE" in preflight|full) ;; *) echo "unexpected mode: $MODE" >&2; exit 2 ;; esac
ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD="$ROOT/token-cd"
OUT="$TOKEN_CD/runs/pi0fast_l6_scene_repair_36seeds_rtfull_v1"
mkdir -p "$OUT/logs"
exec >> "$OUT/logs/${MODE}_gpu${GPU}_${TASK}.log" 2>&1
source /home/leju-suzhou/miniconda3/etc/profile.d/conda.sh
conda activate pi0
export CUDA_VISIBLE_DEVICES="$GPU"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.20 JAX_PLATFORMS=cpu
export PI0FAST_WIDOWX_LAYOUT=bridge PI0FAST_WIDOWX_CLIP=1
export PYTHONPATH="/home/leju-suzhou/.cache/pi0fast_client_deps:$ROOT/openpi/packages/openpi-client/src:$ROOT/pcd_openvla_simpler_box_31b027e/source:$ROOT/SimplerEnv:$ROOT/SpatialVLA-SimplerEnv:$ROOT/SpatialVLA-SimplerEnv/ManiSkill2_real2sim:$ROOT/SpatialVLA/ManiSkill2_real2sim:$TOKEN_CD:$ROOT/openpi/src"
exec python -u "$TOKEN_CD/research/semantic_token_cd/pi0_fast_l6_scene_repair_36.py" \
  --gpu "$GPU" --port "$PORT" --task "$TASK" --mode "$MODE" --seeds "$SEEDS"
