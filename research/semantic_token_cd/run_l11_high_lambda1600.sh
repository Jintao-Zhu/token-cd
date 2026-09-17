#!/usr/bin/env bash
set -euo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
ROOT="$WS/artifacts/l11_matched_dtp_positive_lambda_060_075_v1"
ARMS=l11_matched_lambda_0p6,l11_matched_lambda_0p75,dtp_l11_lambda_0p6,dtp_l11_lambda_0p75
export PYTHONPATH="$WS:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT/logs"
cd "$WS"
run_shard() {
  local gpu="$1" seeds="$2" worker="$3"
  for task in google_robot_open_drawer google_robot_close_drawer google_robot_pick_coke_can google_robot_move_near; do
    "$PY" research/semantic_token_cd/l11_high_lambda_rollout.py \
      --task "$task" --seeds "$seeds" --arms "$ARMS" --gpu "$gpu" \
      --worker-id "$worker" --artifact "$ROOT"
  done
}
case "${1:-}" in
  g2a) run_shard 2 0-16 g2a;;
  g2b) run_shard 2 17-33 g2b;;
  g2c) run_shard 2 34-49 g2c;;
  g3a) run_shard 3 50-66 g3a;;
  g3b) run_shard 3 67-83 g3b;;
  g3c) run_shard 3 84-99 g3c;;
  finalize)
    while [ "$(find "$ROOT/closed_loop/episodes" -type f -name 'episode_*_summary.json' 2>/dev/null | wc -l)" -lt 1600 ]; do sleep 30; done
    "$PY" research/semantic_token_cd/analyze_l11_high_lambda.py > "$ROOT/logs/final_analysis.json"
    ;;
  *) echo "usage: $0 {g2a|g2b|g2c|g3a|g3b|g3c|finalize}" >&2; exit 2;;
esac

