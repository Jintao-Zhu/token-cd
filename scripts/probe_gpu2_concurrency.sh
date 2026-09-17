#!/usr/bin/env bash
# Probe: three concurrent envs+models on physical GPU2 (one episode each).
set -u
cd /home/leju-suzhou/zjt_ws/token-cd
export PYTHONPATH=/home/leju-suzhou/zjt_ws/token-cd:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
mkdir -p logs/xswap
for spec in "google_robot_open_drawer 0 vanilla" "google_robot_open_drawer 1 correct" "google_robot_open_drawer 2 vanilla"; do
  set -- $spec
  t=$1; s=$2; a=$3
  setsid $PY research/semantic_token_cd/xswap_rollout.py --task "$t" --seeds "$s" \
    --gpu 2 --arms "$a" --worker-id "probe_${s}_${a}" \
    > "logs/xswap/probe_gpu2_seed${s}_${a}.log" 2>&1 &
  echo "probe started $s $a pid=$!"
done
echo PROBES_LAUNCHED
