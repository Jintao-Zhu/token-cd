#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?usage: run_l11_rank_causal_libero_osmesa.sh GPU TASK_ID EPISODES ARTIFACT [SMOKE] [WORKER_ID]}"
TASK_ID="${2:?missing task id}"
EPISODES="${3:?missing episodes}"
ARTIFACT="${4:?missing artifact}"
MODE="${5:-full}"
WORKER_ID="${6:-libero-rank-gpu${GPU}}"
case "$GPU" in 1|2|3) ;; *) echo "only GPUs 1, 2, and 3 are authorized" >&2; exit 2 ;; esac
WS=/home/leju-suzhou/zjt_ws/token-cd
cd "$WS"
source scripts/activate_libero_openvla.sh
export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
unset MUJOCO_EGL_DEVICE_ID
OSMESA_LIB_DIR="$WS/artifacts/libero_osmesa_runtime/root/usr/lib/x86_64-linux-gnu"
[ -f "$OSMESA_LIB_DIR/libOSMesa.so.8" ] || { echo "missing OSMesa runtime: $OSMESA_LIB_DIR" >&2; exit 3; }
export LD_LIBRARY_PATH="$OSMESA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$ARTIFACT/logs"
ARGS=(--gpu "$GPU" --task-id "$TASK_ID" --episodes "$EPISODES" --artifact "$ARTIFACT")
[ "$MODE" = smoke ] && ARGS+=(--smoke)
python3 research/semantic_token_cd/run_l11_rank_causal_libero_closedloop.py "${ARGS[@]}"
