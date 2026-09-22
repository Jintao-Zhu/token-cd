#!/usr/bin/env bash
# Matched-budget FLOOR sweep: 3 floors x 4 tasks x seeds 100-199.
# Launched under tmux. Adds exactly TWO processes per GPU on GPUs 2 and 3.
set -euo pipefail

WS=/home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
ROOT="$WS/artifacts/l11_matched_budget_floor_v1"
CANONICAL="$WS/artifacts/vanilla_recon_shr_canonical_0_299_v2"
MATCHED="$WS/artifacts/prompt_attn_l11_token_count_v1"
ARMS=matched_floor24,matched_floor32,matched_floor40
TASKS=(google_robot_open_drawer google_robot_close_drawer google_robot_pick_coke_can google_robot_move_near)

export PYTHONPATH="$WS:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT/logs"
cd "$WS"

run_rollout() {
  local gpu="$1" task="$2" seeds="$3" worker="$4"
  "$PY" research/semantic_token_cd/prompt_attn_l11_budget_floor_rollout.py \
    --artifact "$ROOT" --canonical "$CANONICAL" --matched-artifact "$MATCHED" \
    --task "$task" --seeds "$seeds" --gpu "$gpu" --worker-id "$worker" --arms "$ARMS"
}

run_worker() {
  local gpu="$1" seeds="$2" worker="$3"
  for task in "${TASKS[@]}"; do
    local attempt=1
    while true; do
      if run_rollout "$gpu" "$task" "$seeds" "$worker"; then break; fi
      if [[ "$attempt" -ge 3 ]]; then
        echo "WORKER_FAILED gpu=$gpu worker=$worker task=$task" >&2
        return 1
      fi
      attempt=$((attempt + 1))
      echo "RETRY attempt=$attempt gpu=$gpu task=$task" >&2
      sleep 60
    done
    echo "TASK_DONE worker=$worker task=$task" >&2
  done
  echo "WORKER_DONE gpu=$gpu worker=$worker" >&2
}

case "${1:-}" in
  worker) run_worker "$2" "$3" "$4" ;;
  *) echo "usage: $0 worker <gpu> <seeds> <worker-id>" >&2; exit 2 ;;
esac
