#!/usr/bin/env bash
# Full 120-episode run for PROMPT_SIMREGION_OPEN_TOP_MID_V1.
# 12 shards = {top,middle} x {vanilla,shr_target,shr_other} x {seeds 100-109, 110-119}.
set -u
cd /home/leju-suzhou/zjt_ws/token-cd
export PYTHONPATH=/home/leju-suzhou/zjt_ws/token-cd:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source
export HF_HUB_OFFLINE=1
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
RUN=artifacts/xdrawer_topmid_v1/runs/full
SNAP=artifacts/xdrawer_topmid_v1
EMIT=$RUN/emitted_states
mkdir -p logs/xdrawer
launch() { # drawer arm seeds gpu emitflag
  local extra=()
  if [ "$5" = "emit" ]; then extra=(--emit-states "$EMIT"); fi
  setsid $PY research/semantic_token_cd/xdrawer_rollout.py \
    --artifact "$RUN" --snapshots "$SNAP" --drawer "$1" --seeds "$3" \
    --gpu "$4" --worker-id "$1-$2-$(echo "$3" | tr '-' '_')" \
    --arms "$2" "${extra[@]}" \
    > "logs/xdrawer/full_${1}_${2}_$(echo "$3" | tr '-' '_').log" 2>&1 &
  echo "launched $1 $2 $3 gpu$4"
}
launch top    vanilla    100-109 0 emit
launch top    vanilla    110-119 1 emit
launch top    shr_target 100-109 0 no
launch top    shr_target 110-119 1 no
launch top    shr_other  100-109 2 no
launch top    shr_other  110-119 2 no
launch middle vanilla    100-109 3 emit
launch middle vanilla    110-119 4 emit
launch middle shr_target 100-109 3 no
launch middle shr_target 110-119 4 no
launch middle shr_other  100-109 5 no
launch middle shr_other  110-119 5 no
wait
echo "ALL_FULL_WORKERS_DONE"
