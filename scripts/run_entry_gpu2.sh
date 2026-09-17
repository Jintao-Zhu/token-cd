#!/usr/bin/env bash
set -u
cd /home/leju-suzhou/zjt_ws/token-cd
export PYTHONPATH=/home/leju-suzhou/zjt_ws/token-cd:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
for t in google_robot_open_drawer google_robot_close_drawer google_robot_pick_coke_can google_robot_move_near; do
  echo "ENTRY_START $t $(date +%H:%M:%S)"
  $PY research/semantic_token_cd/xswap_entry_check.py --task "$t" --gpu 2 2>&1
  echo "ENTRY_DONE $t rc=$?"
done
echo ALL_ENTRY_DONE
