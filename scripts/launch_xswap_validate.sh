#!/usr/bin/env bash
# XSWAP-V1 validation: GPU3 open_drawer 5-arm smoke; GPU2 4-task entry audit.
set -u
cd /home/leju-suzhou/zjt_ws/token-cd
export PYTHONPATH=/home/leju-suzhou/zjt_ws/token-cd:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
mkdir -p logs/xswap

# GPU3: open_drawer, 3 scenes, full 5-arm rollout smoke.
setsid $PY research/semantic_token_cd/xswap_rollout.py \
  --task google_robot_open_drawer --seeds 0,1,2 --gpu 3 \
  --arms vanilla,correct,paraphrase,swapped,random \
  --worker-id smoke_open_drawer_v2 \
  > logs/xswap/smoke_opendrawer_v3.log 2>&1 &
echo "smoke_gpu3_pid=$!"

# GPU2: entry audit, all four tasks, 10 manifest scenes each.
(
  for t in google_robot_open_drawer google_robot_close_drawer google_robot_pick_coke_can google_robot_move_near; do
    echo "ENTRY_START $t $(date +%H:%M:%S)"
    $PY research/semantic_token_cd/xswap_entry_check.py --task "$t" --gpu 2 \
      --worker-id entry_gpu2 2>&1
    echo "ENTRY_DONE $t rc=$?"
  done
) > logs/xswap/entry_full_gpu2.log 2>&1 &
echo "entry_gpu2_pid=$!"
