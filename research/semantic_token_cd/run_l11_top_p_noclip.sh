#!/usr/bin/env bash
# Unclipped TopP85 closed-loop rollout (1 arm x 4 tasks x seeds 100-199).
# Launched under tmux so it survives the Codex session. Does NOT touch the
# provenance experiment on GPUs 2/3 (adds exactly one process per GPU).
set -euo pipefail

WS=/home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
ROOT="$WS/artifacts/l11_top_p85_unclipped_100_199_v1"
CANONICAL="$WS/artifacts/vanilla_recon_shr_canonical_0_299_v2"
MATCHED="$WS/artifacts/prompt_attn_l11_token_count_v1"
TASKS=(google_robot_open_drawer google_robot_close_drawer google_robot_pick_coke_can google_robot_move_near)

export PYTHONPATH="$WS:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT/logs"
cd "$WS"

run_rollout() {
  local gpu="$1" task="$2" seeds="$3" worker="$4"
  "$PY" research/semantic_token_cd/prompt_attn_l11_top_p_noclip_rollout.py \
    --artifact "$ROOT" --canonical "$CANONICAL" --matched-artifact "$MATCHED" \
    --task "$task" --seeds "$seeds" --gpu "$gpu" --worker-id "$worker"
}

run_worker() {
  local gpu="$1" seeds="$2" worker="$3"
  for task in "${TASKS[@]}"; do
    local attempt=1
    while true; do
      if run_rollout "$gpu" "$task" "$seeds" "$worker"; then
        break
      fi
      if [[ "$attempt" -ge 3 ]]; then
        echo "WORKER_FAILED gpu=$gpu worker=$worker task=$task" >&2
        return 1
      fi
      attempt=$((attempt + 1))
      echo "RETRY attempt=$attempt gpu=$gpu task=$task" >&2
      sleep 60
    done
  done
  echo "WORKER_DONE gpu=$gpu worker=$worker" >&2
}

case "${1:-}" in
  preflight)
    # single episode on GPU 2 to validate config lock + audit before fanning out
    run_rollout 2 google_robot_open_drawer 100 noclip_preflight
    ;;
  worker)
    run_worker "$2" "$3" "$4"
    ;;
  *)
    echo "usage: $0 preflight | $0 worker <gpu> <seeds> <worker-id>" >&2
    exit 2
    ;;
esac
