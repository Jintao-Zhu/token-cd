#!/usr/bin/env bash
# Paired vanilla vs L6-matched pi0.5 rollout on LIBERO-Plus.
#
#   PER_CELL=20 WORKERS=5 scripts/run_pi05_libero_plus_paired.sh
#
# Each worker owns one GPU (server + EGL rendering) and a strided shard of the
# deterministic (perturbation category, difficulty level) stratified task plan.
set -uo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
PY_ENV=/home/leju-suzhou/zjt_ws/libero-openvla-env/bin/python
PLUS=/home/leju-suzhou/zjt_ws/libero_plus_src
SUITE=${SUITE:-libero_10}
PER_CELL=${PER_CELL:-20}
CATEGORIES=${CATEGORIES:-}
LEVELS=${LEVELS:-}
WORKERS=${WORKERS:-5}
MAX_STEPS=${MAX_STEPS:-0}
ARTIFACT=${ARTIFACT:-$WS/artifacts/pi05_libero_plus_${SUITE}_v1}
GUIDANCE=${GUIDANCE:-}
LAYER=${LAYER:-}
ACTIVE_STEPS=${ACTIVE_STEPS:-}
NUM_STEPS=${NUM_STEPS:-}
NAME_FILTER=${NAME_FILTER:-}
GPUS=(${GPUS:-0 4 5 6 7})
PORTS=(${PORTS:-8125 8126 8127 8128 8129})

mkdir -p "$ARTIFACT/logs"
export LIBERO_CONFIG_PATH=/home/leju-suzhou/zjt_ws/tmp/libero_plus_cfg
export MAGICK_HOME=/home/leju-suzhou/zjt_ws/imagemagick_lib
export LD_LIBRARY_PATH=/home/leju-suzhou/zjt_ws/imagemagick_lib/lib
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH="$PLUS:/home/leju-suzhou/zjt_ws/openpi/packages/openpi-client/src:$WS"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1

EXTRA=()
[[ -n "$GUIDANCE" ]] && EXTRA+=(--guidance-scale "$GUIDANCE")
[[ -n "$LAYER" ]] && EXTRA+=(--layer-index "$LAYER")
[[ -n "$ACTIVE_STEPS" ]] && EXTRA+=(--active-steps "$ACTIVE_STEPS")
[[ -n "$NUM_STEPS" ]] && EXTRA+=(--num-steps "$NUM_STEPS")
[[ -n "$NAME_FILTER" ]] && EXTRA+=(--name-filter "$NAME_FILTER")

pids=()
for ((w = 0; w < WORKERS; w++)); do
  gpu=${GPUS[$((w % ${#GPUS[@]}))]}
  port=${PORTS[$((w % ${#PORTS[@]}))]}
  echo "[worker $w] gpu=$gpu port=$port shard=$w/$WORKERS" | tee "$ARTIFACT/logs/worker${w}_cmd.log"
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" \
    "$PY_ENV" -u "$WS/research/semantic_token_cd/pi05_libero_plus_paired_rollout.py" \
    --artifact "$ARTIFACT" --host 127.0.0.1 --port "$port" \
    --suite "$SUITE" --per-cell "$PER_CELL" --levels "$LEVELS" \
    --categories "$CATEGORIES" --shard "$w/$WORKERS" \
    --max-steps "$MAX_STEPS" ${EXTRA[@]+"${EXTRA[@]}"} \
    > "$ARTIFACT/logs/worker${w}.log" 2>&1 &
  pids+=("$!")
  sleep 3
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
echo "all workers finished (status=$status)" | tee -a "$ARTIFACT/logs/worker0.log"
exit "$status"
