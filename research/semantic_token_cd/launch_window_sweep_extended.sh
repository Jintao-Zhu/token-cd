#!/usr/bin/env bash
# Layer-window sweep extension: open_drawer + stack_cube, 16 shards over 6 GPUs.
set -u
cd ~/zjt_ws/token-cd

ART=artifacts/attn_semantic_layer_window_sweep_v1
PY=~/zjt_ws/openvla-ar-h100/bin/python
PP=~/zjt_ws/token-cd:~/zjt_ws/pcd_openvla_simpler_box_31b027e/source
export HF_HUB_OFFLINE=1
mkdir -p "$ART/logs"

SHARDS=("100-112" "113-124" "125-136" "137-148" "149-160" "161-172" "173-184" "185-199")

launch() {
  local task=$1 gpu=$2 seeds=$3 name=$4
  nohup env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$PP" "$PY" \
    research/semantic_token_cd/attn_semantic_layer_window_sweep.py \
    --artifact "$ART" --task "$task" --gpu "$gpu" --seeds "$seeds" \
    > "$ART/logs/${name}.log" 2>&1 &
}

# open_drawer: 8 shards over GPU 0 (3) + 1 (3) + 4 (2)
for i in 0 1 2; do launch google_robot_open_drawer 0 "${SHARDS[$i]}" "ext_open_drawer_${i}"; done
for i in 3 4 5; do launch google_robot_open_drawer 1 "${SHARDS[$i]}" "ext_open_drawer_${i}"; done
for i in 6 7;   do launch google_robot_open_drawer 4 "${SHARDS[$i]}" "ext_open_drawer_${i}"; done
# stack_cube: 8 shards over GPU 5 (3) + 6 (3) + 7 (2)
for i in 0 1 2; do launch widowx_stack_cube 5 "${SHARDS[$i]}" "ext_stack_cube_${i}"; done
for i in 3 4 5; do launch widowx_stack_cube 6 "${SHARDS[$i]}" "ext_stack_cube_${i}"; done
for i in 6 7;   do launch widowx_stack_cube 7 "${SHARDS[$i]}" "ext_stack_cube_${i}"; done

echo "launched 16 extension shards (open_drawer + stack_cube over 6 GPUs)"
