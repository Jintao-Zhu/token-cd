#!/usr/bin/env bash
# Cross-suite validation of the best selector layer on the "distractor added"
# (Objects-Layout `_add_`) instances of the three remaining LIBERO-Plus suites.
#
# libero_10's 173 instances were used to pick the layer, so they are NOT reused
# here: this run is the held-out half of the train/test split.
set -uo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
LAYER=${LAYER:-10}
LAMBDA=${LAMBDA:-0.5}
SUITES=${SUITES:-"libero_spatial libero_object libero_goal"}
ROOT=${ROOT:-$WS/artifacts/pi05_plus_add_L${LAYER}_crosssuite_v1}
mkdir -p "$ROOT"

for SUITE in $SUITES; do
  ART="$ROOT/$SUITE"
  echo "=== suite=$SUITE layer=$LAYER lambda=$LAMBDA $(date '+%F %T')" | tee -a "$ROOT/sweep.log"
  CATEGORIES="Objects Layout" NAME_FILTER="_add_" GUIDANCE="$LAMBDA" LAYER="$LAYER" \
    PER_CELL=200 WORKERS=3 GPUS="1 2 3" PORTS="8125 8126 8127" \
    SUITE="$SUITE" ARTIFACT="$ART" \
    "$WS/scripts/run_pi05_libero_plus_paired.sh" >> "$ROOT/sweep.log" 2>&1
  /home/leju-suzhou/miniconda3/envs/pi05/bin/python \
    "$WS/research/semantic_token_cd/analyze_pi05_libero_plus_paired.py" \
    --artifact "$ART" --suite "$SUITE" 2>&1 | head -12 | tee -a "$ROOT/sweep.log"
done
echo "cross-suite done $(date '+%F %T')" | tee -a "$ROOT/sweep.log"
