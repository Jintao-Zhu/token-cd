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

mkdir -p "$ART/logs" "$ART/episodes"

run() {
  local i=$1 gpu task lo hi log
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}
  task=${TASKS[$((i / 15))]}
  lo=$(((i % 15) * 20))
  hi=$((lo + 19))
  log="$ART/logs/job_${i}_${task}_${lo}_${hi}.log"
  [ -f "$log.done" ] && return
  printf '\n--- 3-GPU resume on GPU %s ---\n' "$gpu" >> "$log"
  env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$PP" \
    "$PY" research/semantic_token_cd/semantic_recon_rollout.py \
      --artifact "$ART" --task "$task" --gpu "$gpu" --seeds "$lo-$hi" \
      >> "$log" 2>&1
  [ $? -eq 0 ] && touch "$log.done"
}

slot() {
  local i=$1
  while [ "$i" -lt "$NJOBS" ]; do
    run "$i"
    i=$((i + NSLOTS))
  done
}

for s in $(seq 0 $((NSLOTS - 1))); do
  slot "$s" > "$ART/logs/slot_3gpu_${s}.log" 2>&1 &
done
wait
echo "complete summaries=$(find "$ART/episodes" -name 'episode_*_summary.json' | wc -l)" \
  > "$ART/logs/COMPLETE_3GPU"
