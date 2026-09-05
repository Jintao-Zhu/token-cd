#!/usr/bin/env bash
set -u

cd /home/leju-suzhou/zjt_ws/token-cd

PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
ART=artifacts/sp_shr_boundary_partial_3task_0_99_v1
SOURCE=artifacts/vanilla_recon_shr_canonical_0_299_v2
PP=/home/leju-suzhou/zjt_ws/token-cd:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source
TASKS=(google_robot_close_drawer google_robot_move_near google_robot_pick_coke_can)
GPUS=(1 4 5)

completed_in_shard() {
  local task=$1 lo=$2 hi=$3 seed arm count=0
  for arm in boundary_shr partial_shr50; do
    for seed in $(seq "$lo" "$hi"); do
      dir="$ART/episodes/$task/$arm"
      [ -f "$dir/episode_$(printf '%03d' "$seed")_summary.json" ] && \
        [ -f "$dir/episode_$(printf '%03d' "$seed")_arrays.npz" ] && \
        count=$((count + 1))
    done
  done
  printf '%s' "$count"
}

run_shard() {
  local task=$1 gpu=$2 worker=$3 lo=$4 hi=$5 log lock count status
  log="$ART/logs/${task}_${lo}_${hi}.log"
  lock="$ART/logs/${task}_${lo}_${hi}.lock"
  mkdir -p "$ART/logs"
  (
    flock -n 9 || return 0
    while :; do
      count=$(completed_in_shard "$task" "$lo" "$hi")
      if [ "$count" -eq 50 ]; then
        touch "$log.done"
        return 0
      fi
      printf '%s gpu=%s complete=%s/50\n' "$(date --iso-8601=seconds)" "$gpu" "$count" >> "$log"
      env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$PP" \
        "$PY" research/semantic_token_cd/sp_shr_rollout.py \
          --artifact "$ART" --snapshot-artifact "$SOURCE" \
          --task "$task" --seeds "$lo-$hi" --gpu "$gpu" --worker-id "$worker" \
          >> "$log" 2>&1
      status=$?
      count=$(completed_in_shard "$task" "$lo" "$hi")
      [ "$count" -eq 50 ] && continue
      printf '%s retry status=%s complete=%s/50\n' \
        "$(date --iso-8601=seconds)" "$status" "$count" >> "$log"
      sleep 3
    done
  ) 9> "$lock"
}

for task_index in 0 1 2; do
  task=${TASKS[$task_index]}
  gpu=${GPUS[$task_index]}
  for worker_index in 0 1 2 3; do
    lo=$((worker_index * 25))
    hi=$((lo + 24))
    run_shard "$task" "$gpu" "sp-shr-gpu${gpu}-worker${worker_index}" "$lo" "$hi" &
  done
done
wait

count=$(find "$ART/episodes" \( -path '*/boundary_shr/episode_*_summary.json' -o -path '*/partial_shr50/episode_*_summary.json' \) | wc -l)
if [ "$count" -eq 600 ]; then
  printf 'SP-SHR experiment complete: %s/600\n' "$count" > "$ART/logs/COMPLETE"
fi
