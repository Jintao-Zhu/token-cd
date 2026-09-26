#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?usage: run_libero_action_value_trace_worker.sh GPU WORKER_ID [RENDER_GPU]}"
WORKER="${2:?usage: run_libero_action_value_trace_worker.sh GPU WORKER_ID [RENDER_GPU]}"
RENDER_GPU="${3:-$GPU}"
case "$GPU" in 1|2|3|6|7) ;; *) echo "GPU $GPU is not authorized" >&2; exit 2 ;; esac
case "$RENDER_GPU" in 1|2|3|6|7) ;; *) echo "render GPU $RENDER_GPU is not authorized" >&2; exit 2 ;; esac
WS=/home/leju-suzhou/zjt_ws/token-cd
cd "$WS"
source scripts/activate_libero_openvla.sh
# CUDA and EGL enumerate devices independently. Keep model inference confined
# to its authorized physical GPU; the Python worker resolves EGL by PCI BDF.
export CUDA_VISIBLE_DEVICES="$GPU"
unset MUJOCO_EGL_DEVICE_ID
export TOKENIZERS_PARALLELISM=false
python research/semantic_token_cd/libero_action_value_trace_worker.py \
  --artifact artifacts/libero_action_value_trace_pilot_v2 \
  --gpu "$GPU" --render-gpu "$RENDER_GPU" --worker-id "$WORKER" \
  > "artifacts/libero_action_value_trace_pilot_v2/logs/worker_${WORKER}.log" 2>&1
