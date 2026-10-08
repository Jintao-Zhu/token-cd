#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 GPU PORT [INSTANCE_TAG]" >&2
  exit 2
fi
GPU="$1"
PORT="$2"
INSTANCE_TAG="${3:-primary}"
ROOT="/home/leju-suzhou/zjt_ws"
CONDA_ROOT="/home/leju-suzhou/miniconda3"
OUT_DIR="${PI0FAST_ARTIFACT_DIR:-$ROOT/token-cd/artifacts/pi0_fast_l6_matched_simpler_google800_v1}"
LOG="$OUT_DIR/logs/server_gpu${GPU}_${INSTANCE_TAG}.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate pi05
export CUDA_VISIBLE_DEVICES="$GPU"
# Keep JAX's normal backend discovery (CUDA for model work, CPU available for
# debug.callback). This is the environment used by the successful adapter smoke.
unset JAX_PLATFORMS
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export TOKENIZERS_PARALLELISM=false
export PI0FAST_TIMING=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OPENPI_DATA_HOME="$ROOT/checkpoints/openpi-assets"
export PYTHONPATH="$ROOT/openpi/src:$ROOT/openpi/packages/openpi-client/src:/home/leju-suzhou/.cache/pi0fast_client_deps:$ROOT/token-cd${PYTHONPATH:+:$PYTHONPATH}"

exec python -u "$ROOT/token-cd/research/semantic_token_cd/pi0_fast_l6_simpler_server.py" \
  --gpu "$GPU" --port "$PORT" --checkpoint "$ROOT/checkpoints/openpi_fast_finetuned/fractal"
