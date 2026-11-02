#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 4 ]]; then
  echo "usage: $0 GPU PORT FAMILY OUT_DIR" >&2
  exit 2
fi
GPU="$1" PORT="$2" FAMILY="$3" OUT="$4"
case "$GPU" in 4|5|6|7) ;; *) echo "Refusing GPU $GPU; this run is restricted to GPUs 4-7" >&2; exit 2 ;; esac
case "$FAMILY" in google|widowx) ;; *) echo "unknown checkpoint family: $FAMILY" >&2; exit 2 ;; esac

ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD="$ROOT/token-cd"
mkdir -p "$OUT/logs"
exec >> "$OUT/logs/server_gpu${GPU}_port${PORT}.log" 2>&1

source /home/leju-suzhou/miniconda3/etc/profile.d/conda.sh
conda activate pi05
export CUDA_VISIBLE_DEVICES="$GPU"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.42
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OPENPI_DATA_HOME="$ROOT/checkpoints/openpi-assets" HF_HUB_OFFLINE=1
export PI0FAST_CD_KMEANS=sklearn PI0FAST_CD_TAP_MODE=separate PI0FAST_CD_DECODE=verified
export PI0FAST_CD_JIT_HEADER=1 PI0FAST_CD_CACHE_STATIC_VIEWS=1 PI0FAST_TIMING=1
export PYTHONPATH="$ROOT/pi05_sklearn_overlay:$ROOT/openpi/src:$ROOT/openpi/packages/openpi-client/src:$TOKEN_CD${PYTHONPATH:+:$PYTHONPATH}"

SERVER="$TOKEN_CD/research/semantic_token_cd/pi0_fast_l6_simpler_server.py"
if [[ "$FAMILY" == google ]]; then
  CKPT="$ROOT/checkpoints/openpi_fast_finetuned/fractal"
  exec python -u "$SERVER" --gpu "$GPU" --port "$PORT" --checkpoint "$CKPT"
else
  CKPT="$ROOT/checkpoints/openpi_fast_finetuned/bridge"
  exec python -u "$SERVER" --gpu "$GPU" --port "$PORT" --checkpoint "$CKPT" --norm-stats auto
fi
