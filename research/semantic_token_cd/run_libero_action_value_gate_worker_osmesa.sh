#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?usage: run_libero_action_value_gate_worker_osmesa.sh GPU WORKER_ID ARTIFACT_DIR [MAX_CASES]}"
WORKER="${2:?usage: run_libero_action_value_gate_worker_osmesa.sh GPU WORKER_ID ARTIFACT_DIR [MAX_CASES]}"
ARTIFACT="${3:?usage: run_libero_action_value_gate_worker_osmesa.sh GPU WORKER_ID ARTIFACT_DIR [MAX_CASES]}"
MAX_CASES="${4:-0}"
case "$GPU" in 1|2|3|6|7) ;; *) echo "GPU $GPU is not authorized" >&2; exit 2 ;; esac
WS=/home/leju-suzhou/zjt_ws/token-cd
cd "$WS"
source scripts/activate_libero_openvla.sh
export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
unset MUJOCO_EGL_DEVICE_ID
OSMESA_LIB_DIR="$WS/artifacts/libero_osmesa_runtime/root/usr/lib/x86_64-linux-gnu"
if [ ! -f "$OSMESA_LIB_DIR/libOSMesa.so.8" ]; then
  echo "local OSMesa runtime is missing: $OSMESA_LIB_DIR/libOSMesa.so.8" >&2
  exit 3
fi
export LD_LIBRARY_PATH="$OSMESA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$ARTIFACT/logs"
python research/semantic_token_cd/libero_action_value_gate_worker.py \
  --artifact "$ARTIFACT" --gpu "$GPU" --renderer osmesa \
  --worker-id "$WORKER" --max-cases "$MAX_CASES"
