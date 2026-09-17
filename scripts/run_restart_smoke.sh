#!/usr/bin/env bash
set -u
cd /home/leju-suzhou/zjt_ws/token-cd
export PYTHONPATH=/home/leju-suzhou/zjt_ws/token-cd:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
$PY research/semantic_token_cd/xswap_rollout.py \
  --task google_robot_open_drawer --seeds 0,1,2 --gpu 3 \
  --arms vanilla,correct,paraphrase,swapped,random \
  --worker-id smoke_open_drawer_v3 2>&1
echo SMOKE_V3_DONE rc=$?
