#!/usr/bin/env bash
set -u

cd /home/leju-suzhou/zjt_ws/token-cd

PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
ART=artifacts/vanilla_recon_shr_canonical_0_299_v2
PP=/home/leju-suzhou/zjt_ws/token-cd:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source
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
GPUS=(1 4 5)
SLOTS_PER_GPU=4
NSLOTS=12
NJOBS=135

completed_in_shard() {
  local task=$1 lo=$2 hi=$3 seed count=0 dir
  dir="$ART/episodes/$task/st_shr_beta100"
  for seed in $(seq "$lo" "$hi"); do
    [ -f "$dir/episode_$(printf '%03d' "$seed")_summary.json" ] && \
      [ -f "$dir/episode_$(printf '%03d' "$seed")_arrays.npz" ] && \
      count=$((count + 1))
  done
  printf '%s' "$count"
}

run_job() {
  local i=$1 gpu=$2 slot=$3 task lo hi log lock count status
  task=${TASKS[$((i / 15))]}
  lo=$(((i % 15) * 20))
  hi=$((lo + 19))
  log="$ART/logs/st_shr_beta100_job_${i}_${task}_${lo}_${hi}.log"
  lock="$ART/logs/st_shr_beta100_job_${i}.lock"

  (
    flock -n 9 || return 0
    while :; do
      count=$(completed_in_shard "$task" "$lo" "$hi")
      if [ "$count" -eq 20 ]; then
        touch "$log.done"
        return 0
      fi
      printf '%s slot=%s gpu=%s complete=%s/20\n' \
        "$(date --iso-8601=seconds)" "$slot" "$gpu" "$count" >> "$log"
      env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$PP" \
        "$PY" research/semantic_token_cd/st_shr_canonical_rollout.py \
          --artifact "$ART" --snapshot-artifact "$ART" \
          --task "$task" --beta 1.0 --seeds "$lo-$hi" --gpu "$gpu" \
          --worker-id "st-shr-12proc-slot${slot}" \
          >> "$log" 2>&1
      status=$?
      count=$(completed_in_shard "$task" "$lo" "$hi")
      [ "$count" -eq 20 ] && continue
      printf '%s retry status=%s complete=%s/20\n' \
        "$(date --iso-8601=seconds)" "$status" "$count" >> "$log"
      sleep 3
    done
  ) 9> "$lock"
}

slot() {
  local slot=$1 gpu i
  gpu=${GPUS[$((slot / SLOTS_PER_GPU))]}
  i=$slot
  while [ "$i" -lt "$NJOBS" ]; do
    run_job "$i" "$gpu" "$slot"
    i=$((i + NSLOTS))
  done
}

for slot_id in $(seq 0 $((NSLOTS - 1))); do
  slot "$slot_id" > "$ART/logs/st_shr_beta100_slot_${slot_id}.log" 2>&1 &
done
wait

count=$(find "$ART/episodes" -path '*/st_shr_beta100/episode_*_summary.json' | wc -l)
if [ "$count" -eq 2700 ]; then
  printf 'ST-SHR beta=1 canonical arm complete: %s/2700\n' "$count" \
    > "$ART/logs/ST_SHR_BETA100_COMPLETE"
fi
