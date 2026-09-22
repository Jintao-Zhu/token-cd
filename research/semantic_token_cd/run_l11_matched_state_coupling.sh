#!/usr/bin/env bash
# State-coupling ablation: frozen permuted matched budget tape. 4 tasks x seeds 100-199.
set -euo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
ROOT="$WS/artifacts/l11_matched_state_coupling_v1"
TAPE="$ROOT/SHUFFLE_TAPES.json"
CANON="$WS/artifacts/vanilla_recon_shr_canonical_0_299_v2"
MATCHED="$WS/artifacts/prompt_attn_l11_token_count_v1"
TASKS=(google_robot_open_drawer google_robot_close_drawer google_robot_pick_coke_can google_robot_move_near)
export PYTHONPATH="$WS:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT/logs"; cd "$WS"
run_one() {
  "$PY" research/semantic_token_cd/prompt_attn_l11_matched_shuffle_rollout.py \
    --artifact "$ROOT" --tape-file "$TAPE" --canonical "$CANON" --matched-artifact "$MATCHED" \
    --task "$1" --seeds "$2" --gpu "$3" --worker-id "$4"
}
run_worker() {
  local gpu="$1" seeds="$2" worker="$3"
  for task in "${TASKS[@]}"; do
    local a=1
    while true; do
      if run_one "$task" "$seeds" "$gpu" "$worker"; then break; fi
      if [[ "$a" -ge 3 ]]; then echo "FAILED $worker $task" >&2; return 1; fi
      a=$((a+1)); sleep 60
    done
    echo "TASK_DONE $worker $task" >&2
  done
  echo "WORKER_DONE $worker" >&2
}
run_worker "$1" "$2" "$3"
