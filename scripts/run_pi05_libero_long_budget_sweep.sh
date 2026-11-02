#!/usr/bin/env bash
# Headroom-regime lambda sweep on LIBERO-Long.
#
# π0.5 solves libero_10 at the official 520-step budget (vanilla ~98%), so a
# strict step budget is used to create headroom (~47% at 250 steps) where a
# guidance effect is measurable.  Every lambda re-runs both arms (vanilla +
# matched) with identical initial states and per-replan noise.
#
#   LAMBDAS="0.25 0.5 0.75" BUDGET=250 scripts/run_pi05_libero_long_budget_sweep.sh
set -uo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/miniconda3/envs/pi05/bin/python
BUDGET=${BUDGET:-250}
INIT=${INIT:-0-49}
WORKERS=${WORKERS:-5}
LAMBDAS=${LAMBDAS:-0.25 0.5 0.75}
ROOT=${ROOT:-$WS/artifacts/pi05_libero_long_budget${BUDGET}_lambda_sweep_v1}
mkdir -p "$ROOT"

for LAM in $LAMBDAS; do
  TAG=$(echo "$LAM" | tr -d '.')
  ART="$ROOT/lambda${TAG}"
  echo "=== lambda=$LAM budget=$BUDGET artifact=$ART $(date '+%F %T')" | tee -a "$ROOT/sweep.log"
  GUIDANCE_OVERRIDE="$LAM" ARTIFACT="$ART" \
    "$WS/scripts/run_pi05_libero_long_paired.sh" \
    --artifact "$ART" --init-states "$INIT" --max-steps "$BUDGET" --workers "$WORKERS" \
    >> "$ROOT/sweep.log" 2>&1
  echo "--- summary lambda=$LAM" | tee -a "$ROOT/sweep.log"
  "$PY" "$WS/research/semantic_token_cd/analyze_pi05_libero_long_paired.py" \
    --artifact "$ART" 2>&1 | tee -a "$ROOT/sweep.log"
done
echo "sweep complete $(date '+%F %T')" | tee -a "$ROOT/sweep.log"
