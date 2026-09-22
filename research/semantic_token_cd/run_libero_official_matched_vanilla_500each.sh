#!/usr/bin/env bash
set -uo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/zjt_ws/libero-openvla-env/bin/python
ROOT_M="$WS/artifacts/libero_official_matched_500_v1"
ROOT_V="$WS/artifacts/libero_official_vanilla_500_v1"
ROOT_CMP="$WS/artifacts/libero_official_compare_500_v1"
CKPT=/home/leju-suzhou/zjt_ws/checkpoints/libero/openvla-7b-finetuned-libero-spatial
source "$WS/scripts/activate_libero_openvla.sh"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT_M/logs" "$ROOT_V/logs" "$ROOT_CMP"

mapfile -t TASKS < <("$PY" - <<'PY'
import json
print('\n'.join(json.load(open('artifacts/libero/task_registry.json'))['spatial']))
PY
)

run_matched() {
  local cvd="$1" seeds="$2" worker="$3"
  export CUDA_VISIBLE_DEVICES="$cvd" MUJOCO_EGL_DEVICE_ID=0
  for task in "${TASKS[@]}"; do
    echo "[worker $worker] matched start task=$task seeds=$seeds $(date '+%F %T')"
    local attempt=1
    while true; do
      if "$PY" research/semantic_token_cd/libero_matched_rollout.py \
        --artifact "$ROOT_M" --checkpoint "$CKPT" --task "$task" \
        --episodes "$seeds" --gpu 1 --entity-mode source_target \
        --query-mode instruction_only --env-seed 0 --settle-steps 10 \
        --max-steps 220; then
        break
      fi
      if [[ "$attempt" -ge 3 ]]; then return 1; fi
      echo "[worker $worker] matched retry $attempt/3 task=$task $(date '+%F %T')" >&2
      attempt=$((attempt+1))
      sleep 60
    done
    echo "[worker $worker] matched done task=$task $(date '+%F %T')"
  done
}

run_vanilla() {
  local cvd="$1" seeds="$2" worker="$3"
  export CUDA_VISIBLE_DEVICES="$cvd" MUJOCO_EGL_DEVICE_ID=0
  for task in "${TASKS[@]}"; do
    echo "[worker $worker] vanilla start task=$task seeds=$seeds $(date '+%F %T')"
    local attempt=1
    while true; do
      if "$PY" research/semantic_token_cd/libero_vanilla_rollout.py \
        --artifact "$ROOT_V" --checkpoint "$CKPT" --task "$task" \
        --episodes "$seeds" --gpu 1 --env-seed 0 --settle-steps 10 \
        --max-steps 220; then
        break
      fi
      if [[ "$attempt" -ge 3 ]]; then return 1; fi
      echo "[worker $worker] vanilla retry $attempt/3 task=$task $(date '+%F %T')" >&2
      attempt=$((attempt+1))
      sleep 60
    done
    echo "[worker $worker] vanilla done task=$task $(date '+%F %T')"
  done
}

pids=()
# GPU1 logical card: two matched workers + one vanilla worker.
run_matched 0,1 0-8  gpu1_m0 > "$ROOT_M/logs/gpu1_m0.log" 2>&1 & pids+=("$!"); sleep 6
run_matched 0,1 9-17 gpu1_m1 > "$ROOT_M/logs/gpu1_m1.log" 2>&1 & pids+=("$!"); sleep 6
run_vanilla 0,1 0-16 gpu1_v0 > "$ROOT_V/logs/gpu1_v0.log" 2>&1 & pids+=("$!"); sleep 6

# GPU2 logical card: two matched workers + one vanilla worker.
run_matched 0,2 18-25 gpu2_m0 > "$ROOT_M/logs/gpu2_m0.log" 2>&1 & pids+=("$!"); sleep 6
run_matched 0,2 26-33 gpu2_m1 > "$ROOT_M/logs/gpu2_m1.log" 2>&1 & pids+=("$!"); sleep 6
run_vanilla 0,2 17-33 gpu2_v0 > "$ROOT_V/logs/gpu2_v0.log" 2>&1 & pids+=("$!"); sleep 6

# GPU3 logical card: two matched workers + one vanilla worker.
run_matched 0,3 34-41 gpu3_m0 > "$ROOT_M/logs/gpu3_m0.log" 2>&1 & pids+=("$!"); sleep 6
run_matched 0,3 42-49 gpu3_m1 > "$ROOT_M/logs/gpu3_m1.log" 2>&1 & pids+=("$!"); sleep 6
run_vanilla 0,3 34-49 gpu3_v0 > "$ROOT_V/logs/gpu3_v0.log" 2>&1 & pids+=("$!")

st=()
for p in "${pids[@]}"; do
  if wait "$p"; then st+=(0); else st+=("$?"); fi
done
if [[ "${st[*]}" != "0 0 0 0 0 0 0 0 0" ]]; then
  echo "worker statuses: ${st[*]}" >&2
  exit 1
fi
"$PY" research/semantic_token_cd/analyze_libero_matched_spatial.py --artifact "$ROOT_M"
"$PY" research/semantic_token_cd/analyze_libero_vanilla_spatial.py --artifact "$ROOT_V"
"$PY" research/semantic_token_cd/analyze_libero_official_compare.py \
  --matched-root "$ROOT_M" --vanilla-root "$ROOT_V" --out-root "$ROOT_CMP"
