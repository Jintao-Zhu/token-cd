#!/usr/bin/env bash
# Full 100-seed Global Spatial Merge CD (GSM-CD, Phase 1A) run (ATTN_GLOBAL_MERGE_CD_V1).
# 3 tasks x 5 shards (20 seeds each) = 15 processes, 3 per GPU (one per task).
# Free GPUs only: 1,4,5,6,7.  GPU 0 is dead (Xid 45); GPU 2/3 = RoboTwin (DO NOT TOUCH).
# Seeds 200-299 (fresh; no prior artifact uses seeds >= 200).
set -u
cd ~/zjt_ws/token-cd

PY=~/zjt_ws/openvla-ar-h100/bin/python
SCD=research/semantic_token_cd
ART=artifacts/attn_global_merge_v1
PP=~/zjt_ws/token-cd:~/zjt_ws/pcd_openvla_simpler_box_31b027e/source
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PP"
export TOKENIZERS_PARALLELISM=false

GPUS=(1 4 5 6 7)
SHARDS=("200-219" "220-239" "240-259" "260-279" "280-299")
TASKS=(google_robot_move_near google_robot_close_drawer google_robot_pick_coke_can)

mkdir -p "$ART/logs"

for i in 0 1 2 3 4; do
  gpu=${GPUS[$i]}
  for task in "${TASKS[@]}"; do
    short=$(echo "$task" | sed 's/google_robot_//')
    nohup env CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$PP" "$PY" \
      "$SCD/global_merge_rollout.py" --artifact "$ART" --task "$task" --gpu "$gpu" \
      --seeds "${SHARDS[$i]}" \
      > "$ART/logs/gsm_${short}_${i}.log" 2>&1 &
    echo "launched $task shard $i -> GPU $gpu (seeds ${SHARDS[$i]})"
  done
done

echo "15 processes dispatched."
