#!/usr/bin/env bash
# Mechanism probe: where in the flow trajectory does guidance hurt or help?
# Early steps (large noise) vs late steps (near-clean), plus a stronger lambda.
set -uo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
BUDGET=${BUDGET:-250}
INIT=${INIT:-0-11}
ROOT=${ROOT:-$WS/artifacts/pi05_libero_long_budget250_flowstep_probe_v1}
mkdir -p "$ROOT"

run_case() {
  local tag="$1" lam="$2" steps="$3"
  local art="$ROOT/$tag"
  echo "=== $tag lambda=$lam active_steps=$steps $(date '+%F %T')" | tee -a "$ROOT/probe.log"
  GUIDANCE_OVERRIDE="$lam" ACTIVE_STEPS="$steps" ARTIFACT="$art" \
    "$WS/scripts/run_pi05_libero_long_paired.sh" \
    --artifact "$art" --init-states "$INIT" --max-steps "$BUDGET" --workers 5 \
    >> "$ROOT/probe.log" 2>&1
  /home/leju-suzhou/miniconda3/envs/pi05/bin/python \
    "$WS/research/semantic_token_cd/analyze_pi05_libero_long_paired.py" \
    --artifact "$art" 2>&1 | tail -14 | tee -a "$ROOT/probe.log"
}

run_case early_lam050 0.5 "0,1,2"
run_case late_lam050 0.5 "5,6,7,8,9"
run_case early_lam150 1.5 "0,1,2"
echo "probe complete $(date '+%F %T')" | tee -a "$ROOT/probe.log"
