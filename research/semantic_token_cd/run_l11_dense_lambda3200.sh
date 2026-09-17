#!/usr/bin/env bash
set -euo pipefail

WS=/home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
ROOT="$WS/artifacts/prompt_attn_l11_matched_dense_lambda_0_99_v1"
ARMS=l11_matched_lambda_010,l11_matched_lambda_015,l11_matched_lambda_020,l11_matched_lambda_030,l11_matched_lambda_035,l11_matched_lambda_040,l11_matched_lambda_045,l11_matched_lambda_055
export PYTHONPATH="$WS:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT/logs"
cd "$WS"

run_shard() {
  local gpu="$1" seeds="$2" worker="$3"
  for task in google_robot_open_drawer google_robot_close_drawer google_robot_pick_coke_can google_robot_move_near; do
    "$PY" research/semantic_token_cd/prompt_attn_l11_dense_lambda_rollout.py \
      --task "$task" --seeds "$seeds" --arms "$ARMS" --gpu "$gpu" \
      --worker-id "$worker" --artifact "$ROOT"
  done
}

case "${1:-}" in
  g2a) run_shard 2 0-24 g2a;;
  g2b) run_shard 2 25-49 g2b;;
  g3a) run_shard 3 50-74 g3a;;
  g3b) run_shard 3 75-99 g3b;;
  finalize)
    while [ "$(find "$ROOT/episodes" -type f -name 'episode_*_summary.json' 2>/dev/null | wc -l)" -lt 3200 ]; do
      sleep 30
    done
    "$PY" research/semantic_token_cd/analyze_l11_dense_lambda.py > "$ROOT/logs/final_analysis.json"
    ;;
  *) echo "usage: $0 {g2a|g2b|g3a|g3b|finalize}" >&2; exit 2;;
esac

