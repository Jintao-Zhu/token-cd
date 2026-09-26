#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?usage: run_libero_selector_placebo_osmesa.sh GPU WORKER_ID ARTIFACT [MAX_CASES]}"
WORKER="${2:?missing worker id}"
ARTIFACT="${3:?missing artifact directory}"
MAX_CASES="${4:-0}"
case "$GPU" in 1|2|3|4|5|6) ;; *) echo "GPU $GPU is not authorized" >&2; exit 2 ;; esac
WS=/home/leju-suzhou/zjt_ws/token-cd
cd "$WS"
source scripts/activate_libero_openvla.sh
export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
unset MUJOCO_EGL_DEVICE_ID
OSMESA_LIB_DIR="$WS/artifacts/libero_osmesa_runtime/root/usr/lib/x86_64-linux-gnu"
test -f "$OSMESA_LIB_DIR/libOSMesa.so.8"
export LD_LIBRARY_PATH="$OSMESA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
"/home/leju-suzhou/zjt_ws/libero-openvla-env/bin/python" research/semantic_token_cd/libero_action_value_trace_worker.py \
  --artifact "$ARTIFACT" --gpu "$GPU" --renderer osmesa --worker-id "$WORKER" \
  --selector-placebo --max-cases "$MAX_CASES"
