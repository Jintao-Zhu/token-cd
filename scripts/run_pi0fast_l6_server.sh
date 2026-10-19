#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 GPU PORT [INSTANCE_TAG]" >&2
  exit 2
fi
GPU="$1"
PORT="$2"
INSTANCE_TAG="${3:-primary}"
ROOT="${PI0FAST_WORKSPACE_ROOT:-$HOME/zjt_ws}"
TOKEN_CD="${PI0FAST_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_ROOT="${PI0FAST_CONDA_ROOT:-$HOME/miniconda3}"
CONDA_ENV="${PI0FAST_SERVER_CONDA_ENV:-pi05}"
OPENPI_ROOT="${PI0FAST_OPENPI_ROOT:-$ROOT/openpi}"
CLIENT_DEPS="${PI0FAST_CLIENT_DEPS:-$HOME/.cache/pi0fast_client_deps}"
CHECKPOINT="${PI0FAST_CHECKPOINT:-$ROOT/checkpoints/openpi_fast_finetuned/fractal}"
OUT_DIR="${PI0FAST_ARTIFACT_DIR:-$TOKEN_CD/artifacts/pi0_fast_l6_matched_simpler_google800_v1}"
LOG="$OUT_DIR/logs/server_gpu${GPU}_${INSTANCE_TAG}.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export CUDA_VISIBLE_DEVICES="$GPU"
# Keep JAX's normal backend discovery (CUDA for model work, CPU available for
# debug.callback). This is the environment used by the successful adapter smoke.
unset JAX_PLATFORMS
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export TOKENIZERS_PARALLELISM=false
export PI0FAST_TIMING="${PI0FAST_TIMING:-1}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PI0FAST_WORKSPACE_ROOT="$ROOT" PI0FAST_REPO_ROOT="$TOKEN_CD" PI0FAST_OPENPI_ROOT="$OPENPI_ROOT"
export OPENPI_DATA_HOME="$ROOT/checkpoints/openpi-assets"
export PYTHONPATH="$ROOT/pi05_sklearn_overlay:$OPENPI_ROOT/src:$OPENPI_ROOT/packages/openpi-client/src:$CLIENT_DEPS:$TOKEN_CD${PYTHONPATH:+:$PYTHONPATH}"

exec python -u "$TOKEN_CD/research/semantic_token_cd/pi0_fast_l6_simpler_server.py" \
  --gpu "$GPU" --port "$PORT" --checkpoint "$CHECKPOINT"
