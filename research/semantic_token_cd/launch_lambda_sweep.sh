#!/usr/bin/env bash
# Lambda (intervention-strength) sweep: move_near, 8 shards over GPU 0,1.
set -u
cd ~/zjt_ws/token-cd

ART=artifacts/attn_semantic_lambda_sweep_v1
PY=~/zjt_ws/openvla-ar-h100/bin/python
PP=~/zjt_ws/token-cd:~/zjt_ws/pcd_openvla_simpler_box_31b027e/source
export HF_HUB_OFFLINE=1
mkdir -p "$ART/logs"

SHARDS=("100-112" "113-124" "125-136" "137-148" "149-160" "161-172" "173-184" "185-199")

launch() {
  local task=$1 gpu=$2 seeds=$3 name=$4
  nohup env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$PP" "$PY" \
    research/semantic_token_cd/attn_semantic_lambda_sweep.py \
    --artifact "$ART" --task "$task" --gpu "$gpu" --seeds "$seeds" \
    > "$ART/logs/${name}.log" 2>&1 &
}

for i in 0 1 2 3; do launch google_robot_move_near 0 "${SHARDS[$i]}" "lambda_move_near_${i}"; done
for i in 4 5 6 7; do launch google_robot_move_near 1 "${SHARDS[$i]}" "lambda_move_near_${i}"; done

echo "launched 8 lambda-sweep shards (move_near over GPU 0,1)"
