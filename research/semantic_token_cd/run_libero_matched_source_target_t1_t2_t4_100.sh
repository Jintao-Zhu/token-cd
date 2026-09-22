#!/usr/bin/env bash
set -uo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/zjt_ws/libero-openvla-env/bin/python
ROOT="$WS/artifacts/libero_matched_source_target_t1_t2_t4_100_v1"
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

T1=pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate
T2=pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate
T4=pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate

run_worker() {
  local task="$1" worker="$2"
  export CUDA_VISIBLE_DEVICES=0,1 MUJOCO_EGL_DEVICE_ID=0
  echo "[worker $worker] start task=$task seeds=0-99 $(date '+%F %T')"
  local attempt=1
  while true; do
    if "$PY" research/semantic_token_cd/libero_matched_rollout.py \
      --artifact "$ROOT" --checkpoint "$CKPT" --task "$task" \
      --episodes 0-99 --gpu 1 --entity-mode source_target; then
      break
    fi
    if [[ "$attempt" -ge 3 ]]; then return 1; fi
    echo "[worker $worker] retry $attempt/3 task=$task $(date '+%F %T')" >&2
    attempt=$((attempt+1))
    sleep 60
  done
  echo "[worker $worker] done task=$task $(date '+%F %T')"
}

pids=()
run_worker "$T1" t1 > "$ROOT/logs/t1.log" 2>&1 & pids+=("$!")
sleep 8
run_worker "$T2" t2 > "$ROOT/logs/t2.log" 2>&1 & pids+=("$!")
sleep 8
run_worker "$T4" t4 > "$ROOT/logs/t4.log" 2>&1 & pids+=("$!")

st=()
for p in "${pids[@]}"; do
  if wait "$p"; then st+=(0); else st+=("$?"); fi
done
if [[ "${st[*]}" != "0 0 0" ]]; then
  echo "worker statuses: ${st[*]}" >&2
  exit 1
fi
"$PY" research/semantic_token_cd/analyze_libero_matched_spatial.py --artifact "$ROOT"
