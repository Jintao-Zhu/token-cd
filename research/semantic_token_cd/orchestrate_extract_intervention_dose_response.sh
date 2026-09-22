#!/usr/bin/env bash
set -uo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
ROOT="$WS/artifacts/intervention_dose_response_v1"
CANON="$WS/artifacts/vanilla_recon_shr_canonical_0_299_v2"
CAL="$WS/artifacts/l11_entity_budget_calibration_v1"
export PYTHONPATH="$WS:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT/logs"; cd "$WS"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$ROOT/logs/ORCH.log"; }
run(){ "$PY" research/semantic_token_cd/extract_intervention_dose_response.py \
  --artifact "$ROOT" --canonical "$CANON" --calibration "$CAL" \
  --calibration-results "$CAL/CALIBRATION_RESULTS.json" --task "$1" --seeds 0-49 \
  --gpu "$2" --worker-id dose_$1; }
say "=== intervention dose response extraction start ==="
run google_robot_open_drawer 2 > "$ROOT/logs/google_robot_open_drawer.log" 2>&1 & P1=$!
sleep 10
run google_robot_close_drawer 2 > "$ROOT/logs/google_robot_close_drawer.log" 2>&1 & P2=$!
sleep 10
run google_robot_pick_coke_can 3 > "$ROOT/logs/google_robot_pick_coke_can.log" 2>&1 & P3=$!
sleep 10
run google_robot_move_near 3 > "$ROOT/logs/google_robot_move_near.log" 2>&1 & P4=$!
say "workers: [$P1 $P2 $P3 $P4]"
st=(); for p in $P1 $P2 $P3 $P4; do wait $p && st+=(0) || st+=($?); done
N=$(find "$ROOT/dose" -name "seed_*.json" 2>/dev/null | wc -l)
say "exit=${st[*]} seed files=$N (expected 200)"
