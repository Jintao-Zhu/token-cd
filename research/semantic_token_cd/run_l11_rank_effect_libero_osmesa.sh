#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?usage: run_l11_rank_effect_libero_osmesa.sh GPU CASES_FILE ARTIFACT [MAX_CASES] [STATES_PER_EPISODE] [strict]}"
CASES="${2:?missing cases file}"
ARTIFACT="${3:?missing artifact directory}"
MAX_CASES="${4:-0}"
STATES="${5:-10}"
STRICT="${6:-strict}"
WORKER_ID="${7:-osmesa-gpu${GPU}}"
case "$GPU" in 1|2|3|4|5|6) ;; *) echo "inference GPU $GPU is not allowed" >&2; exit 2 ;; esac
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
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "$ARTIFACT/logs"
STRICT_ARGS=()
if [ "$STRICT" = "strict" ]; then STRICT_ARGS+=(--strict-replay); fi
python3 research/semantic_token_cd/replay_libero_l11_rank_effects.py \
  --gpu "$GPU" --worker-id "$WORKER_ID" \
  --artifact "$ARTIFACT" --cases-file "$CASES" \
  --max-cases "$MAX_CASES" --states-per-episode "$STATES" \
  "${STRICT_ARGS[@]}"
