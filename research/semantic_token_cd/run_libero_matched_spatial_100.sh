#!/usr/bin/env bash
set -uo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/zjt_ws/libero-openvla-env/bin/python
ROOT="$WS/artifacts/libero_matched_spatial_100_v1"
CKPT=/home/leju-suzhou/zjt_ws/checkpoints/libero/openvla-7b-finetuned-libero-spatial
source "$WS/scripts/activate_libero_openvla.sh"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs"
mapfile -t TASKS < <("$PY" - <<'PY'
import json
print('\n'.join(json.load(open('artifacts/libero/task_registry.json'))['spatial']))
PY
)
run_worker(){
  local cvd="$1" seeds="$2" worker="$3"
  export CUDA_VISIBLE_DEVICES="$cvd" MUJOCO_EGL_DEVICE_ID=0
  for task in "${TASKS[@]}"; do
    echo "[worker $worker] start task=$task seeds=$seeds $(date '+%F %T')"
    local attempt=1
    while true; do
      if "$PY" research/semantic_token_cd/libero_matched_rollout.py \
        --artifact "$ROOT" --checkpoint "$CKPT" --task "$task" --episodes "$seeds" \
        --gpu 1; then break; fi
      if [[ "$attempt" -ge 3 ]]; then return 1; fi
      echo "[worker $worker] retry $attempt/3 task=$task $(date '+%F %T')" >&2
      attempt=$((attempt+1)); sleep 60
    done
    echo "[worker $worker] done task=$task $(date '+%F %T')"
  done
}
pids=()
run_worker 0,2 0-24 gpu2_slot0 > "$ROOT/logs/gpu2_slot0.log" 2>&1 & pids+=("$!")
sleep 8
run_worker 0,2 25-49 gpu2_slot1 > "$ROOT/logs/gpu2_slot1.log" 2>&1 & pids+=("$!")
sleep 8
run_worker 0,3 50-74 gpu3_slot0 > "$ROOT/logs/gpu3_slot0.log" 2>&1 & pids+=("$!")
sleep 8
run_worker 0,3 75-99 gpu3_slot1 > "$ROOT/logs/gpu3_slot1.log" 2>&1 & pids+=("$!")
st=(); for p in "${pids[@]}"; do if wait "$p"; then st+=(0); else st+=("$?"); fi; done
if [[ "${st[*]}" != "0 0 0 0" ]]; then echo "worker statuses: ${st[*]}" >&2; exit 1; fi
"$PY" research/semantic_token_cd/analyze_libero_matched_spatial.py --artifact "$ROOT"
