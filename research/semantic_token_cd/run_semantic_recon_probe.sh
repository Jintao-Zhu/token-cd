#!/usr/bin/env bash
# SCR-CD Phase 0 integrity probe — 9 tasks x 10 frozen states (90 states), NO rollout.
# One process per task, fanned across the 6 free GPUs. GPU 0 (post-Xid, cautious)
# gets the least load. GPU 2/3 = RoboTwin (DO NOT TOUCH).
#
# Each process writes artifacts/semantic_recon_k8_m10_v1/probe/<short_task>.json .
# A task is PASS iff finite && G∩B=∅ && not identity-degenerate && not zero-degenerate.
# The Phase 1 rollout launches only after all 9 tasks pass (checked separately).
set -u
cd ~/zjt_ws/token-cd

PY=~/zjt_ws/openvla-ar-h100/bin/python
SCD=research/semantic_token_cd
ART=artifacts/semantic_recon_k8_m10_v1
PP=~/zjt_ws/token-cd:~/zjt_ws/pcd_openvla_simpler_box_31b027e/source
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

GPUS=(1 4 5 6 7 0)

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

mkdir -p "$ART/probe" "$ART/logs"
: > "$ART/probe/pids.txt"

for i in "${!TASKS[@]}"; do
  task="${TASKS[$i]}"
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}
  short=$(echo "$task" | sed 's/google_robot_//')
  log="$ART/logs/probe_${short}.log"
  nohup env CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$PP" "$PY" \
    "$SCD/semantic_recon_probe.py" --artifact "$ART" --task "$task" --gpu "$gpu" \
    --seeds 300-309 > "$log" 2>&1 &
  echo "$! $task $gpu" >> "$ART/probe/pids.txt"
  echo "probe launched: $short -> GPU $gpu (pid $!)"
done

echo "9 probe processes dispatched."