#!/usr/bin/env bash
# Detached runner for the source-change equivalence check (scratch artifact only).
# Never touches the official artifact; exits nonzero on the first failed arm.
set -uo pipefail
ROOT=/home/leju-suzhou/zjt_ws/token-cd
cd "$ROOT"
source scripts/activate_simpler_local.sh
ART=artifacts/vtcg_source_change_check
LOG=/tmp/vtcg_equivalence.log
: >"$LOG"
for arm in "$@"; do
  echo "=== start $arm $(date --iso-8601=seconds)" >>"$LOG"
  CUDA_VISIBLE_DEVICES=1 "$SIMPLER_ENV/bin/python" -u research/semantic_token_cd/vtcg_factorial_episode.py \
    --artifact "$ART" --task google_robot_pick_coke_can --seed 503 --arm "$arm" --gpu 1 \
    --snapshot-mode run --phase formal --run-config-id VTCG_FACTORIAL_CONTROLS_V1-formal --video \
    >>"$LOG" 2>&1
  echo "=== $arm exit=$? $(date --iso-8601=seconds)" >>"$LOG"
done
echo "ALL_ARMS_DONE" >>"$LOG"
