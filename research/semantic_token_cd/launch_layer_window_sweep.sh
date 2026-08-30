#!/usr/bin/env bash
# Launch the Attention-CD layer-window sweep: 3 tasks x 8 seed-shards = 24 procs
# over 6 GPUs (0,1,4,5,6,7), 4 procs per GPU.
#   move_near     -> GPU 0 (shards 0-3) + GPU 1 (shards 4-7)
#   close_drawer  -> GPU 4 (shards 0-3) + GPU 5 (shards 4-7)
#   pick_coke_can -> GPU 6 (shards 0-3) + GPU 7 (shards 4-7)
set -u
cd ~/zjt_ws/token-cd

ART=artifacts/attn_semantic_layer_window_sweep_v1
PY=~/zjt_ws/openvla-ar-h100/bin/python
PP=~/zjt_ws/token-cd:~/zjt_ws/pcd_openvla_simpler_box_31b027e/source
export HF_HUB_OFFLINE=1
mkdir -p "$ART/logs"

# 8 contiguous shards covering seeds 100..199 (100 seeds).
SHARDS=("100-112" "113-124" "125-136" "137-148" "149-160" "161-172" "173-184" "185-199")

launch() {  # task gpu seeds name
  local task=$1 gpu=$2 seeds=$3 name=$4
  nohup env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$PP" "$PY" \
    research/semantic_token_cd/attn_semantic_layer_window_sweep.py \
    --artifact "$ART" --task "$task" --gpu "$gpu" --seeds "$seeds" \
    > "$ART/logs/${name}.log" 2>&1 &
}

# move_near
for i in 0 1 2 3; do launch google_robot_move_near 0 "${SHARDS[$i]}" "move_near_${i}"; done
for i in 4 5 6 7; do launch google_robot_move_near 1 "${SHARDS[$i]}" "move_near_${i}"; done
# close_drawer
for i in 0 1 2 3; do launch google_robot_close_drawer 4 "${SHARDS[$i]}" "close_drawer_${i}"; done
for i in 4 5 6 7; do launch google_robot_close_drawer 5 "${SHARDS[$i]}" "close_drawer_${i}"; done
# pick_coke_can
for i in 0 1 2 3; do launch google_robot_pick_coke_can 6 "${SHARDS[$i]}" "pick_coke_can_${i}"; done
for i in 4 5 6 7; do launch google_robot_pick_coke_can 7 "${SHARDS[$i]}" "pick_coke_can_${i}"; done

echo "launched 24 shard processes (3 tasks x 8 shards over 6 GPUs)"
