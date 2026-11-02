#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "usage: $0 GPU PORT" >&2
  exit 2
fi
GPU="$1" PORT="$2"
case "$GPU" in 1|2|3) ;; *) echo "Refusing GPU $GPU; this run is restricted to GPUs 1-3" >&2; exit 2 ;; esac

ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD="$ROOT/token-cd"
OUT="$TOKEN_CD/runs/pi0fast_9task_vanilla_vs_l6jit_lambda050_rtfull_100seed_v1"
mkdir -p "$OUT/logs"
exec >> "$OUT/logs/server_gpu${GPU}_port${PORT}.log" 2>&1

source /home/leju-suzhou/miniconda3/etc/profile.d/conda.sh
conda activate pi05
export CUDA_VISIBLE_DEVICES="$GPU"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.30
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OPENPI_DATA_HOME="$ROOT/checkpoints/openpi-assets" HF_HUB_OFFLINE=1
export PI0FAST_CD_KMEANS=sklearn PI0FAST_CD_TAP_MODE=separate PI0FAST_CD_DECODE=verified
export PI0FAST_CD_JIT_HEADER=1 PI0FAST_CD_CACHE_STATIC_VIEWS=1 PI0FAST_TIMING=1
export PYTHONPATH="$ROOT/pi05_sklearn_overlay:$ROOT/openpi/src:$ROOT/openpi/packages/openpi-client/src:$TOKEN_CD${PYTHONPATH:+:$PYTHONPATH}"

exec python -u "$TOKEN_CD/research/semantic_token_cd/pi0_fast_l6_simpler_server.py" \
  --gpu "$GPU" --port "$PORT" --checkpoint "$ROOT/checkpoints/openpi_fast_finetuned/fractal"
