#!/usr/bin/env bash
set -uo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/zjt_ws/libero-openvla-env/bin/python
ROOT="$WS/artifacts/libero_full_deep_test_v1"
CKPT=/home/leju-suzhou/zjt_ws/checkpoints/libero/openvla-7b-finetuned-libero-spatial
source "$WS/scripts/activate_libero_openvla.sh"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs"
mapfile -t TASKS < <("$PY" - <<'PY'
import json
print('\n'.join(json.load(open('artifacts/libero/task_registry.json'))['spatial']))
PY
)
run_arm() {
  local states="$1" cvd="$2" worker="$3"
  export CUDA_VISIBLE_DEVICES="$cvd" MUJOCO_EGL_DEVICE_ID=0
  for task in "${TASKS[@]}"; do
    echo "[worker $worker] start task=$task states=$states $(date '+%F %T')"
    local attempt=1
    while true; do
      if "$PY" research/semantic_token_cd/libero_matched_rollout.py \
        --artifact "$ROOT" --checkpoint "$CKPT" --task "$task" \
        --episodes "$states" --gpu 1 --entity-mode source_target \
        --query-mode instruction_only --attention-layers 23-25 \
        --env-seed 0 --settle-steps 10 --max-steps 220; then break; fi
      if [[ "$attempt" -ge 3 ]]; then return 1; fi
      echo "[worker $worker] retry $attempt/3 task=$task $(date '+%F %T')" >&2
      attempt=$((attempt+1)); sleep 60
    done
    echo "[worker $worker] done task=$task $(date '+%F %T')"
  done
}
pids=()
run_arm 20-29 0,1 fd_t1 > "$ROOT/logs/fd_t1.log" 2>&1 & pids+=("$!"); sleep 5
run_arm 30-39 0,2 fd_t2 > "$ROOT/logs/fd_t2.log" 2>&1 & pids+=("$!"); sleep 5
run_arm 40-49 0,3 fd_t3 > "$ROOT/logs/fd_t3.log" 2>&1 & pids+=("$!")
st=(); for p in "${pids[@]}"; do if wait "$p"; then st+=(0); else st+=("$?"); fi; done
if [[ "${st[*]}" != "0 0 0" ]]; then echo "worker statuses: ${st[*]}" >&2; exit 1; fi
echo "FULL_DEEP_DONE $(date '+%F %T')"
