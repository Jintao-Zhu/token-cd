#!/usr/bin/env bash
set -uo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/zjt_ws/libero-openvla-env/bin/python
CKPT=/home/leju-suzhou/zjt_ws/checkpoints/libero/openvla-7b-finetuned-libero-spatial
ROOT="$WS/artifacts/libero_lambda_sweep_v1"
source "$WS/scripts/activate_libero_openvla.sh"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs"
mapfile -t TASKS < <("$PY" - <<'PY'
import json
print('\n'.join(json.load(open('artifacts/libero/task_registry.json'))['spatial']))
PY
)
run_worker() {
  local arm="$1" query="$2" layers="$3" heads="$4" scale="$5" states="$6" cvd="$7" worker="$8"
  local arm_root="$ROOT/$arm"
  mkdir -p "$arm_root"
  export CUDA_VISIBLE_DEVICES="$cvd" MUJOCO_EGL_DEVICE_ID=0
  for task in "${TASKS[@]}"; do
    echo "[worker $worker] arm=$arm start task=$task states=$states $(date '+%F %T')"
    local attempt=1
    while true; do
      if "$PY" research/semantic_token_cd/libero_matched_rollout.py \
        --artifact "$arm_root" --checkpoint "$CKPT" --task "$task" \
        --episodes "$states" --gpu 1 --entity-mode source_target \
        --query-mode "$query" --attention-layers "$layers" \
        --attention-heads "$heads" --destination-weight 0.0 \
        --lambda-scale "$scale" --env-seed 0 --settle-steps 10 --max-steps 220; then break; fi
      if [[ "$attempt" -ge 3 ]]; then return 1; fi
      echo "[worker $worker] arm=$arm retry $attempt/3 task=$task $(date '+%F %T')" >&2
      attempt=$((attempt+1)); sleep 60
    done
    echo "[worker $worker] arm=$arm done task=$task $(date '+%F %T')"
  done
}
# arm|query|layers|heads|lambda_scale
ARMS=(
"current_lam025|instruction_only|11||0.25"
"current_lam050|instruction_only|11||0.5"
"selected5_lam025|target_relation_endpoint|11|14:5,7:14,9:12,19:11,21:2|0.25"
"selected5_lam050|target_relation_endpoint|11|14:5,7:14,9:12,19:11,21:2|0.5"
)
pids=()
idx=0
for spec in "${ARMS[@]}"; do
  IFS='|' read -r arm query layers heads scale <<< "$spec"
  run_worker "$arm" "$query" "$layers" "$heads" "$scale" 20-29 0,1 "${arm}_t1" > "$ROOT/logs/${arm}_t1.log" 2>&1 & pids+=("$!"); sleep 4
  run_worker "$arm" "$query" "$layers" "$heads" "$scale" 30-39 0,2 "${arm}_t2" > "$ROOT/logs/${arm}_t2.log" 2>&1 & pids+=("$!"); sleep 4
  run_worker "$arm" "$query" "$layers" "$heads" "$scale" 40-49 0,3 "${arm}_t3" > "$ROOT/logs/${arm}_t3.log" 2>&1 & pids+=("$!")
  idx=$((idx+1))
done
st=(); for p in "${pids[@]}"; do if wait "$p"; then st+=(0); else st+=("$?"); fi; done
for status in "${st[@]}"; do if [[ "$status" -ne 0 ]]; then echo "worker statuses: ${st[*]}" >&2; exit 1; fi; done
echo "LAMBDA_SWEEP_DONE $(date '+%F %T')"
