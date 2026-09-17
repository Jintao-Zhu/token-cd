#!/usr/bin/env bash
# Resume failed shards after vanilla-only pass. Single env per GPU per wave to
# avoid concurrent Vulkan renderer creation on the same physical device.
set -u
cd /home/leju-suzhou/zjt_ws/token-cd
export PYTHONPATH=/home/leju-suzhou/zjt_ws/token-cd:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source
export HF_HUB_OFFLINE=1
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
RUN=artifacts/xdrawer_topmid_v1/runs/full
SNAP=artifacts/xdrawer_topmid_v1
EMIT=$RUN/emitted_states
mkdir -p logs/xdrawer
launch() {
  setsid $PY research/semantic_token_cd/xdrawer_rollout.py \
    --artifact "$RUN" --snapshots "$SNAP" --drawer "$1" --seeds "$3" \
    --gpu "$4" --worker-id "$1-$2-$(echo "$3" | tr '-' '_')" \
    --arms "$2" --emit-states "$EMIT" \
    > "logs/xdrawer/resume_${1}_${2}_$(echo "$3" | tr '-' '_').log" 2>&1 &
  echo "launched $1 $2 $3 gpu$4"
}
# wave 1: six workers, one per GPU
launch top    vanilla    100-109 0
launch top    shr_target 110-119 1
launch top    shr_other  100-109 2
launch middle shr_target 110-119 3
launch middle shr_other  100-109 4
launch middle shr_other  110-119 5
wait
echo "WAVE1_DONE"
# wave 2: remaining three, one per GPU
launch top    shr_target 100-109 0
launch top    shr_other  110-119 1
launch middle shr_target 100-109 2
wait
echo "WAVE2_DONE"
echo "ALL_RESUME_WORKERS_DONE"
