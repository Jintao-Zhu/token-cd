#!/usr/bin/env bash
# Paired vanilla vs L6-matched pi0.5 rollout on LIBERO-Long (libero_10).
#
#   scripts/run_pi05_libero_long_paired.sh --init-states 0-4 --workers 5
#
# Each worker owns one GPU (server + EGL rendering) and a disjoint strided
# slice of the requested initial states, so the union covers every pair once.
set -uo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
PY_ENV=/home/leju-suzhou/zjt_ws/libero-openvla-env/bin/python
ARTIFACT=${ARTIFACT:-$WS/artifacts/pi05_libero_long_paired_vanilla_vs_l6lambda050_v1}
TASKS=${TASKS:-all}
INIT_STATES=${INIT_STATES:-0-49}
WORKERS=${WORKERS:-5}
MAX_STEPS=${MAX_STEPS:-520}
REPLAN_STEPS=${REPLAN_STEPS:-5}
GUIDANCE=${GUIDANCE:-0.5}
# GPU list must line up with the servers started by start_pi05_paired_server.sh
GPUS=(${GPUS:-0 4 5 6 7})
PORTS=(${PORTS:-8125 8126 8127 8128 8129})

while [[ $# -gt 0 ]]; do
  case "$1" in
    --init-states) INIT_STATES="$2"; shift 2 ;;
    --tasks) TASKS="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --artifact) ARTIFACT="$2"; shift 2 ;;
    --guidance-scale) GUIDANCE="$2"; shift 2 ;;
    --layer-index) LAYER_INDEX="$2"; shift 2 ;;
    --active-steps) ACTIVE_STEPS="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

EXTRA=()
if [[ -n "${GUIDANCE_OVERRIDE+set}" ]]; then
  EXTRA+=(--guidance-scale "$GUIDANCE_OVERRIDE")
fi
if [[ -n "${LAYER_INDEX+set}" ]]; then
  EXTRA+=(--layer-index "$LAYER_INDEX")
fi
if [[ -n "${ACTIVE_STEPS+set}" ]]; then
  EXTRA+=(--active-steps "$ACTIVE_STEPS")
fi
if [[ -n "${NUM_STEPS+set}" ]]; then
  EXTRA+=(--num-steps "$NUM_STEPS")
fi
if [[ -n "${RECORD_DIR+set}" ]]; then
  EXTRA+=(--record-dir "$RECORD_DIR")
fi

mkdir -p "$ARTIFACT/logs"
source "$WS/scripts/activate_libero_openvla.sh"
export PYTHONPATH="$PYTHONPATH:/home/leju-suzhou/zjt_ws/openpi/packages/openpi-client/src"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1

# Expand "a-b" into python with stride, so each worker gets a disjoint slice.
strided_ids() {
  "$PY_ENV" - "$1" "$2" "$3" <<'PY'
import sys
spec, stride, offset = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
values: list[int] = []
for part in spec.split(","):
    part = part.strip()
    if not part:
        continue
    if "-" in part:
        lo, hi = (int(v) for v in part.split("-", 1))
        values.extend(range(lo, hi + 1))
    else:
        values.append(int(part))
values = sorted(set(values))
print(",".join(str(v) for v in values[offset::stride]))
PY
}

pids=()
for ((w = 0; w < WORKERS; w++)); do
  ids=$(strided_ids "$INIT_STATES" "$WORKERS" "$w")
  if [[ -z "$ids" ]]; then
    continue
  fi
  gpu=${GPUS[$((w % ${#GPUS[@]}))]}
  port=${PORTS[$((w % ${#PORTS[@]}))]}
  echo "[worker $w] gpu=$gpu port=$port init_states=$ids" | tee "$ARTIFACT/logs/worker${w}_cmd.log"
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" MUJOCO_GL=egl \
    "$PY_ENV" -u research/semantic_token_cd/pi05_libero_long_paired_rollout.py \
    --artifact "$ARTIFACT" --host 127.0.0.1 --port "$port" \
    --tasks "$TASKS" --init-states "$ids" --max-steps "$MAX_STEPS" \
    --replan-steps "$REPLAN_STEPS" ${EXTRA[@]+"${EXTRA[@]}"} \
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
