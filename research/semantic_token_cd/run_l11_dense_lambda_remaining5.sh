#!/usr/bin/env bash
set -euo pipefail

WS=/home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
ROOT="$WS/artifacts/prompt_attn_l11_matched_dense_lambda_remaining5_0_99_v1"
ARMS=l11_matched_lambda_010,l11_matched_lambda_015,l11_matched_lambda_020,l11_matched_lambda_025,l11_matched_lambda_030,l11_matched_lambda_035,l11_matched_lambda_040,l11_matched_lambda_045,l11_matched_lambda_055,l11_matched_lambda_060,l11_matched_lambda_075
TASKS="google_robot_place_apple_in_closed_top_drawer widowx_carrot_on_plate widowx_put_eggplant_in_basket widowx_spoon_on_towel widowx_stack_cube"
export PYTHONPATH="$WS:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT/logs"
cd "$WS"

run_shard() {
  local gpu="$1" seeds="$2" worker="$3"
  for task in $TASKS; do
    "$PY" research/semantic_token_cd/prompt_attn_l11_dense_lambda_remaining5_rollout.py \
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
    while [ "$(find "$ROOT/episodes" -type f -name 'episode_*_summary.json' 2>/dev/null | wc -l)" -lt 5500 ]; do sleep 60; done
    "$PY" research/semantic_token_cd/analyze_l11_dense_lambda_remaining5.py > "$ROOT/logs/final_analysis.json"
    ;;
  *) echo "usage: $0 {g2a|g2b|g2c|g3a|g3b|g3c|finalize}" >&2; exit 2;;
esac
