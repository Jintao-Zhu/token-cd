#!/usr/bin/env bash
set -u
cd /home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
ART=artifacts/three_arm_canonical_0_299_v1
PP=/home/leju-suzhou/zjt_ws/token-cd:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source
TASKS=(google_robot_pick_coke_can google_robot_open_drawer google_robot_close_drawer google_robot_move_near google_robot_place_apple_in_closed_top_drawer widowx_carrot_on_plate widowx_put_eggplant_in_basket widowx_spoon_on_towel widowx_stack_cube)
GPUS=(1 4 5); SLOTS_PER_GPU=4; NSLOTS=12; mkdir -p "$ART/logs" "$ART/episodes"
run(){ local i=$1; local gpu=${GPUS[$((i%3))]}; local task=${TASKS[$((i/15))]}; local lo=$(( (i%15)*20 )); local hi=$((lo+19)); local log="$ART/logs/job_${i}_${task}_${lo}_${hi}.log"; [ -f "$log.done" ] && return; env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$PP" "$PY" research/semantic_token_cd/semantic_recon_rollout.py --artifact "$ART" --task "$task" --gpu "$gpu" --seeds "$lo-$hi" > "$log" 2>&1; [ $? -eq 0 ] && touch "$log.done"; }
slot(){ local slot=$1; local i=$slot; while [ $i -lt 135 ]; do run "$i"; i=$((i+NSLOTS)); done; }
for slot_id in $(seq 0 $((NSLOTS-1))); do slot "$slot_id" > "$ART/logs/slot_${slot_id}.log" 2>&1 & done
wait
echo "complete summaries=$(find "$ART/episodes" -name 'episode_*_summary.json' | wc -l)" > "$ART/logs/COMPLETE"
