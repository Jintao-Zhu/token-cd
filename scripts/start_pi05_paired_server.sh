#!/usr/bin/env bash
# Serve pi0.5-LIBERO paired (vanilla / L6 matched harmonic) on one GPU.
# Usage: start_pi05_paired_server.sh <gpu> <port> [extra args...]
set -uo pipefail
GPU="$1"
PORT="$2"
shift 2
WS=/home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/miniconda3/envs/pi05/bin/python
LOG_DIR="$WS/runs/pi05_libero_long_paired_v1/logs"
mkdir -p "$LOG_DIR"
exec env -u PYTHONPATH \
  PI05_OPENPI_ROOT=/home/leju-suzhou/zjt_ws/openpi \
  CUDA_VISIBLE_DEVICES="$GPU" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_MEM_FRACTION:-0.22} \
  HF_HUB_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false \
  OMP_NUM_THREADS=8 \
  MKL_NUM_THREADS=8 \
  PYTHONPATH="$WS" \
  "$PY" -u research/semantic_token_cd/pi05_matched_flow_server.py \
  --gpu "$GPU" --port "$PORT" "$@" \
  >> "$LOG_DIR/server_gpu${GPU}_port${PORT}.log" 2>&1
