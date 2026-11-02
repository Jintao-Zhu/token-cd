#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "usage: $0 GPU PORT FAMILY" >&2
  exit 2
fi
GPU="$1"
PORT="$2"
FAMILY="$3"
ROOT=/home/leju-suzhou/zjt_ws
OUT="$ROOT/token-cd/runs/pi0fast_jit_header_extra5_100seed_rtfull_v1"
mkdir -p "$OUT/logs"
exec >> "$OUT/logs/server_gpu${GPU}_p${PORT}.log" 2>&1

source /home/leju-suzhou/miniconda3/etc/profile.d/conda.sh
conda activate pi05
export CUDA_VISIBLE_DEVICES="$GPU"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.40
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OPENPI_DATA_HOME="$ROOT/checkpoints/openpi-assets" HF_HUB_OFFLINE=1
export PI0FAST_CD_KMEANS=sklearn PI0FAST_CD_TAP_MODE=separate PI0FAST_CD_DECODE=verified
export PI0FAST_CD_JIT_HEADER=1 PI0FAST_CD_CACHE_STATIC_VIEWS=1
export PYTHONPATH="$ROOT/pi05_sklearn_overlay:$ROOT/openpi/src:$ROOT/openpi/packages/openpi-client/src:$ROOT/token-cd${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$FAMILY" == google ]]; then
  CKPT="$ROOT/checkpoints/openpi_fast_finetuned/fractal"
  exec python -u "$ROOT/token-cd/research/semantic_token_cd/pi0_fast_l6_simpler_server.py" \
    --gpu "$GPU" --port "$PORT" --checkpoint "$CKPT"
elif [[ "$FAMILY" == widowx ]]; then
  CKPT="$ROOT/checkpoints/openpi_fast_finetuned/bridge"
  exec python -u "$ROOT/token-cd/research/semantic_token_cd/pi0_fast_l6_simpler_server.py" \
    --gpu "$GPU" --port "$PORT" --checkpoint "$CKPT" --norm-stats auto
else
  echo "unknown family: $FAMILY" >&2
  exit 2
fi
