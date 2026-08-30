#!/usr/bin/env bash
# Selection-geometry sweep: move_near + close_drawer, 16 shards over GPU 4,5,6,7.
set -u
cd ~/zjt_ws/token-cd

ART=artifacts/attn_semantic_selection_sweep_v1
PY=~/zjt_ws/openvla-ar-h100/bin/python
PP=~/zjt_ws/token-cd:~/zjt_ws/pcd_openvla_simpler_box_31b027e/source
export HF_HUB_OFFLINE=1
mkdir -p "$ART/logs"

SHARDS=("100-112" "113-124" "125-136" "137-148" "149-160" "161-172" "173-184" "185-199")

launch() {
  local task=$1 gpu=$2 seeds=$3 name=$4
  nohup env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$PP" "$PY" \
    research/semantic_token_cd/attn_semantic_selection_sweep.py \
    --artifact "$ART" --task "$task" --gpu "$gpu" --seeds "$seeds" \
    > "$ART/logs/${name}.log" 2>&1 &
}

# move_near -> GPU 4 (shards 0-3) + GPU 5 (shards 4-7)
for i in 0 1 2 3; do launch google_robot_move_near 4 "${SHARDS[$i]}" "sel_move_near_${i}"; done
for i in 4 5 6 7; do launch google_robot_move_near 5 "${SHARDS[$i]}" "sel_move_near_${i}"; done
# close_drawer -> GPU 6 (shards 0-3) + GPU 7 (shards 4-7)
for i in 0 1 2 3; do launch google_robot_close_drawer 6 "${SHARDS[$i]}" "sel_close_drawer_${i}"; done
for i in 4 5 6 7; do launch google_robot_close_drawer 7 "${SHARDS[$i]}" "sel_close_drawer_${i}"; done

echo "launched 16 selection-sweep shards (move_near+close_drawer over GPU 4,5,6,7)"
