#!/usr/bin/env bash
# L14 single-layer closed loop: 4 google_robot tasks x seeds 0-99.
# Seeded to match the existing prompt_single (L11) / prompt_sparse (L11+L14) runs.
set -euo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
ROOT="$WS/artifacts/l14_single_layer_v1/closed_loop"
LAYER="$WS/artifacts/l14_single_layer_v1"
REF="$WS/artifacts/prompt_attn_layer_selection_v1/closed_loop"
SNAP="$WS/artifacts/vanilla_recon_shr_canonical_0_299_v2"
TASKS=google_robot_open_drawer,google_robot_close_drawer,google_robot_pick_coke_can,google_robot_move_near
export PYTHONPATH="$WS:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT/logs"
cd "$WS"
run_one() {
  local gpu="$1" task="$2" seeds="$3" worker="$4"
  "$PY" research/semantic_token_cd/prompt_attn_layer_rollout.py \
    --artifact "$ROOT" --layer-artifact "$LAYER" --closed-loop-v1 "$REF" \
    --snapshot-artifact "$SNAP" --task "$task" --seeds "$seeds" --gpu "$gpu" \
    --worker-id "$worker" --arms prompt_single --config-tasks "$TASKS"
}
run_worker() {
  local gpu="$1" seeds="$2" worker="$3"
  for task in ${TASKS//,/ }; do
    local attempt=1
    while true; do
      if run_one "$gpu" "$task" "$seeds" "$worker"; then break; fi
      if [[ "$attempt" -ge 3 ]]; then echo "FAILED worker=$worker task=$task" >&2; return 1; fi
      attempt=$((attempt+1)); sleep 60
    done
    echo "TASK_DONE worker=$worker task=$task" >&2
  done
  echo "WORKER_DONE worker=$worker" >&2
}
run_worker "$1" "$2" "$3"
