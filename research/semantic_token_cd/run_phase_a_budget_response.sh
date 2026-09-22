#!/usr/bin/env bash
# Phase A: frozen-state m-sweep response curves. 4 tasks x seeds 0-49 x 4 progress states.
set -euo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
ROOT="$WS/artifacts/l11_budget_response_curve_v1"
CANON="$WS/artifacts/vanilla_recon_shr_canonical_0_299_v2"
export PYTHONPATH="$WS:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT/logs"; cd "$WS"
run_task() {
  local gpu="$1" task="$2" worker="$3"
  local a=1
  while true; do
    if "$PY" research/semantic_token_cd/phase_a_l11_budget_response_scan.py \
         --artifact "$ROOT" --canonical "$CANON" --task "$task" \
         --seeds 0-49 --gpu "$gpu" --worker-id "$worker"; then return 0; fi
    if [[ "$a" -ge 3 ]]; then echo "FAILED $worker $task" >&2; return 1; fi
    a=$((a+1)); sleep 60
  done
}
case "$1" in
  A) run_task 2 google_robot_open_drawer pA_g2a; run_task 2 google_robot_close_drawer pA_g2a ;;
  B) run_task 3 google_robot_pick_coke_can pA_g3b; run_task 3 google_robot_move_near pA_g3b ;;
esac
