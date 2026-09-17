#!/usr/bin/env bash
# Staged capacity probe on physical GPU2: one correct-arm episode (113 steps)
# at concurrency 1, then 2, then 3. Logs wall time per stage.
set -u
cd /home/leju-suzhou/zjt_ws/token-cd
export PYTHONPATH=/home/leju-suzhou/zjt_ws/token-cd:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
run_one() { # seed log
  $PY research/semantic_token_cd/xswap_rollout.py --task google_robot_open_drawer \
    --seeds "$1" --gpu 2 --arms correct --worker-id "cap_$1" > "$2" 2>&1
}
echo "STAGE1 start $(date +%T)"
run_one 7 logs/xswap/cap1_seed7.log
echo "STAGE1 done $(date +%T)"
echo "STAGE2 start $(date +%T)"
run_one 8 logs/xswap/cap2_seed8.log & p8=$!
run_one 9 logs/xswap/cap2_seed9.log & p9=$!
wait $p8 $p9
echo "STAGE2 done $(date +%T)"
echo "STAGE3 start $(date +%T)"
run_one 10 logs/xswap/cap3_seed10.log & q1=$!
run_one 11 logs/xswap/cap3_seed11.log & q2=$!
run_one 12 logs/xswap/cap3_seed12.log & q3=$!
wait $q1 $q2 $q3
echo "STAGE3 done $(date +%T)"
echo CAPACITY_TEST_ALL_DONE
