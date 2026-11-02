#!/usr/bin/env bash
# Online selector-layer sweep on the Objects-Layout "distractor added" instances.
#
# The offline screen (pi05_layer_selection_offline.py) ranks L10/L5/L8/L12 above
# the inherited L6, so validate those layers closed-loop at fixed lambda=0.5.
set -uo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
ROOT=${ROOT:-$WS/artifacts/pi05_plus_libero10_add_layer_sweep_v1}
LAYERS=${LAYERS:-10 5 12}
LAMBDA=${LAMBDA:-0.5}
mkdir -p "$ROOT"

for LAYER in $LAYERS; do
  ART="$ROOT/L${LAYER}"
  echo "=== layer=$LAYER lambda=$LAMBDA $(date '+%F %T')" | tee -a "$ROOT/sweep.log"
  CATEGORIES="Objects Layout" NAME_FILTER="_add_" GUIDANCE="$LAMBDA" LAYER="$LAYER" \
    PER_CELL=200 WORKERS=3 GPUS="1 2 3" PORTS="8125 8126 8127" \
    SUITE=libero_10 ARTIFACT="$ART" \
    "$WS/scripts/run_pi05_libero_plus_paired.sh" >> "$ROOT/sweep.log" 2>&1
  /home/leju-suzhou/miniconda3/envs/pi05/bin/python \
    "$WS/research/semantic_token_cd/analyze_pi05_libero_plus_paired.py" --artifact "$ART" \
    2>&1 | head -14 | tee -a "$ROOT/sweep.log"
done
echo "layer sweep complete $(date '+%F %T')" | tee -a "$ROOT/sweep.log"
