#!/usr/bin/env bash
# SCR-CD Phase 1 — 9 tasks x 100 seeds (300-399) x 5 arms = 4500 episodes.
# 18 concurrent worker slots = 3 per GPU on the 6 free GPUs (1,4,5,6,7,0); GPU 0
# (post-Xid) is last in rotation. GPU 2/3 = RoboTwin (DO NOT TOUCH).
#
# Crash-resume: each (task, arm, seed) writes episode_*_summary.json + _arrays.npz
# and is skipped on re-run (skip-if-exists). A shard's .done marker is written only
# on a clean (exit-0) pass; the trailing RETRY pass re-dispatches any shard whose
# python crashed (technical-audit RuntimeError / OOM / Xid), up to RETRIES times.
set -u
cd ~/zjt_ws/token-cd

PY=~/zjt_ws/openvla-ar-h100/bin/python
SCD=research/semantic_token_cd
ART=artifacts/semantic_recon_k8_m10_v1
PP=~/zjt_ws/token-cd:~/zjt_ws/pcd_openvla_simpler_box_31b027e/source
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

GPUS=(1 4 5 6 7 0)
SLOTS_PER_GPU=3
NSLOTS=$((${#GPUS[@]} * SLOTS_PER_GPU))   # 18

TASKS=(
  google_robot_pick_coke_can
  google_robot_open_drawer
  google_robot_close_drawer
  google_robot_move_near
  google_robot_place_apple_in_closed_top_drawer
  widowx_carrot_on_plate
  widowx_put_eggplant_in_basket
  widowx_spoon_on_towel
  widowx_stack_cube
)
SHARDS=("300-319" "320-339" "340-359" "360-379" "380-399")

mkdir -p "$ART/logs" "$ART/episodes"

# Build job list (task-major): each job = "task|shard"; index -> slot (i % NSLOTS).
JOBS=()
for t in "${TASKS[@]}"; do
  for s in "${SHARDS[@]}"; do
    JOBS+=("$t|$s")
  done
done
NJOBS=${#JOBS[@]}   # 45

shortname() { echo "$1" | sed 's/google_robot_//'; }

run_job() {
  local idx=$1 gpu=$2
  IFS='|' read -r task shard <<< "${JOBS[$idx]}"
  local short; short=$(shortname "$task")
  local log="$ART/logs/rollout_${short}_s${idx}.log"
  if [ -f "$log.done" ]; then
    echo "[slot gpu=$gpu] skip done job=$idx $task $shard"
    return 0
  fi
  env CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$PP" "$PY" \
    "$SCD/semantic_recon_rollout.py" --artifact "$ART" --task "$task" --gpu "$gpu" \
    --seeds "$shard" > "$log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then touch "$log.done"; else
    echo "[slot gpu=$gpu] JOB $idx FAILED rc=$rc $task $shard (log: $log)"
  fi
  return $rc
}

# One slot runs its jobs (stride NSLOTS) strictly sequentially.
run_slot() {
  local slot=$1
  local gpu=${GPUS[$((slot / SLOTS_PER_GPU))]}
  local idx=$slot
  while [ $idx -lt $NJOBS ]; do
    run_job "$idx" "$gpu"
    idx=$((idx + NSLOTS))
  done
}

dispatch_slot() { run_slot "$1" > "$ART/logs/slot_$1.log" 2>&1 & }

echo "=== Phase 1 dispatch: $NJOBS jobs across $NSLOTS slots ==="
for slot in $(seq 0 $((NSLOTS - 1))); do
  dispatch_slot "$slot"
  echo "slot $slot -> GPU ${GPUS[$((slot / SLOTS_PER_GPU))]}"
done
echo "=== waiting on ${NSLOTS} slots ==="
wait

# RETRY pass: any job whose log exists but has no .done marker crashed — re-run.
for attempt in 1 2 3; do
  MISSING=()
  for idx in $(seq 0 $((NJOBS - 1))); do
    IFS='|' read -r task shard <<< "${JOBS[$idx]}"
    short=$(shortname "$task")
    log="$ART/logs/rollout_${short}_s${idx}.log"
    [ -f "$log.done" ] || MISSING+=("$idx")
  done
  if [ ${#MISSING[@]} -eq 0 ]; then
    echo "=== ALL $NJOBS JOBS COMPLETE (attempt $attempt) ==="
    break
  fi
  echo "=== retry attempt $attempt: ${#MISSING[@]} incomplete jobs ==="
  for idx in "${MISSING[@]}"; do
    gpu=${GPUS[$(((idx % NSLOTS) / SLOTS_PER_GPU))]}
    run_job "$idx" "$gpu"
  done
done

# Final tally
DONE=0; TODO=0
for idx in $(seq 0 $((NJOBS - 1))); do
  IFS='|' read -r task shard <<< "${JOBS[$idx]}"
  short=$(shortname "$task")
  log="$ART/logs/rollout_${short}_s${idx}.log"
  if [ -f "$log.done" ]; then DONE=$((DONE+1)); else TODO=$((TODO+1)); echo "INCOMPLETE: $task $shard"; fi
done
echo "=== Phase 1 tally: $DONE done, $TODO incomplete ==="