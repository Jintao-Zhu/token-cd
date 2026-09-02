#!/usr/bin/env bash
set -u

cd /home/leju-suzhou/zjt_ws/token-cd

PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
ART=artifacts/vanilla_recon_shr_canonical_0_299_v2
PP=/home/leju-suzhou/zjt_ws/token-cd:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source

run_task() {
  local gpu=$1 task=$2 worker=$3 log
  log="$ART/logs/st_shr_beta100_${task}.log"
  while :; do
    env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$PP" \
      "$PY" research/semantic_token_cd/st_shr_canonical_rollout.py \
        --artifact "$ART" --snapshot-artifact "$ART" \
        --task "$task" --beta 1.0 --seeds 0-299 --gpu "$gpu" \
        --worker-id "$worker" --wait-for-snapshots \
        >> "$log" 2>&1
    status=$?
    count=$(find "$ART/episodes/$task/st_shr_beta100" -name 'episode_*_summary.json' 2>/dev/null | wc -l)
    [ "$status" -eq 0 ] && [ "$count" -eq 300 ] && return 0
    printf '%s retry task=%s status=%s complete=%s/300\n' \
      "$(date --iso-8601=seconds)" "$task" "$status" "$count" >> "$log"
    sleep 5
  done
}

worker() {
  local gpu=$1 worker_id=$2
  shift 2
  local task
  for task in "$@"; do
    run_task "$gpu" "$task" "$worker_id"
  done
}

# Full-snapshot tasks run first; incomplete WidowX tasks wait in the final position.
worker 1 st-shr-gpu1 \
  google_robot_pick_coke_can \
  google_robot_move_near \
  widowx_put_eggplant_in_basket &
p1=$!

worker 4 st-shr-gpu4 \
  google_robot_open_drawer \
  google_robot_place_apple_in_closed_top_drawer \
  widowx_stack_cube &
p2=$!

worker 5 st-shr-gpu5 \
  google_robot_close_drawer \
  widowx_spoon_on_towel \
  widowx_carrot_on_plate &
p3=$!

wait "$p1" "$p2" "$p3"
printf 'ST-SHR beta=1 canonical arm complete\n' > "$ART/logs/ST_SHR_BETA100_COMPLETE"
