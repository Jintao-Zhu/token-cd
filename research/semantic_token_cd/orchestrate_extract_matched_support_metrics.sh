#!/usr/bin/env bash
set -uo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
ROOT="$WS/artifacts/matched_dynamic_budget_analysis_v1"
CAL="$WS/artifacts/l11_entity_budget_calibration_v1"
CANON="$WS/artifacts/vanilla_recon_shr_canonical_0_299_v2"
export PYTHONPATH="$WS:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT/logs"; cd "$WS"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$ROOT/logs/ORCH.log"; }
run(){ "$PY" research/semantic_token_cd/extract_matched_support_metrics.py \
  --calibration "$CAL" --artifact "$ROOT" --canonical "$CANON" --task "$1" --seeds 0-49 --gpu "$2" --worker-id support_$1; }
say "=== matched support metrics extraction start ==="
run google_robot_open_drawer 2 > "$ROOT/logs/google_robot_open_drawer.log" 2>&1 & P1=$!
sleep 10
run google_robot_close_drawer 2 > "$ROOT/logs/google_robot_close_drawer.log" 2>&1 & P2=$!
sleep 10
run google_robot_pick_coke_can 3 > "$ROOT/logs/google_robot_pick_coke_can.log" 2>&1 & P3=$!
sleep 10
run google_robot_move_near 3 > "$ROOT/logs/google_robot_move_near.log" 2>&1 & P4=$!
say "workers: [$P1 $P2 $P3 $P4]"
st=(); for p in $P1 $P2 $P3 $P4; do wait $p && st+=(0) || st+=($?); done
N=$(find "$ROOT/metrics" -name "seed_*_step_*.json" 2>/dev/null | wc -l)
say "exit=${st[*]} metric files=$N (expected 800)"
