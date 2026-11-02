#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 4 ]]; then
  echo "usage: $0 GPU TASK PORT CHECKPOINT_FAMILY" >&2
  exit 2
fi
GPU="$1"
TASK="$2"
PORT="$3"
FAMILY="$4"
case "$GPU" in 1|2|3) ;; *) echo "Refusing GPU $GPU; only 1,2,3 are authorized" >&2; exit 2 ;; esac
case "$FAMILY" in google|widowx) ;; *) echo "unknown family: $FAMILY" >&2; exit 2 ;; esac

ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD="$ROOT/token-cd"
OUT="$TOKEN_CD/runs/pi0fast_jit_header_ab_4task_20seed_rtfull_v1"
mkdir -p "$OUT/logs"
exec >> "$OUT/logs/worker_${TASK}_gpu${GPU}.log" 2>&1

source /home/leju-suzhou/miniconda3/etc/profile.d/conda.sh
conda activate pi0
export CUDA_VISIBLE_DEVICES="$GPU"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.20
# Inference runs only on the model servers. Keep JAX on CPU in renderer clients so
# client processes do not reserve GPU memory needed by the RT renderer/model server.
export JAX_PLATFORMS=cpu
export PI0FAST_WIDOWX_LAYOUT=bridge PI0FAST_WIDOWX_CLIP=1
export PYTHONPATH="/home/leju-suzhou/.cache/pi0fast_client_deps:$ROOT/openpi/packages/openpi-client/src:$ROOT/pcd_openvla_simpler_box_31b027e/source:$ROOT/SimplerEnv:$ROOT/SpatialVLA-SimplerEnv:$ROOT/SpatialVLA-SimplerEnv/ManiSkill2_real2sim:$ROOT/SpatialVLA/ManiSkill2_real2sim:$TOKEN_CD:$ROOT/openpi/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$FAMILY" == google ]]; then
  CKPT="$ROOT/checkpoints/openpi_fast_finetuned/fractal"
else
  CKPT="$ROOT/checkpoints/openpi_fast_finetuned/bridge"
fi
SEEDS=($(seq 0 19))
exec python -u "$TOKEN_CD/research/semantic_token_cd/pi0_fast_jit_header_closed_loop.py" \
  --task "$TASK" --gpu "$GPU" --port "$PORT" --rt-depth 8 \
  --out "$OUT" --seeds "${SEEDS[@]}" --checkpoint "$CKPT"
