#!/usr/bin/env bash
# matched mask UNION a 16-token evenly strided grid. 9 tasks x seeds 100-199.
set -euo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
ROOT="$WS/artifacts/l11_matched_plus_uniform16_v1"
CANON="$WS/artifacts/vanilla_recon_shr_canonical_0_299_v2"
MATCHED="$WS/artifacts/prompt_attn_l11_token_count_v1"
TASKS=(google_robot_open_drawer google_robot_close_drawer google_robot_pick_coke_can google_robot_move_near
       google_robot_place_apple_in_closed_top_drawer widowx_carrot_on_plate widowx_put_eggplant_in_basket
       widowx_spoon_on_towel widowx_stack_cube)
export PYTHONPATH="$WS:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT/logs"; cd "$WS"
run_one() {
  "$PY" research/semantic_token_cd/prompt_attn_l11_matched_plus_uniform_rollout.py \
    --artifact "$ROOT" --canonical "$CANON" --matched-artifact "$MATCHED" \
    --task "$1" --seeds "$2" --gpu "$3" --worker-id "$4"
}
run_worker() {
  local gpu="$1" seeds="$2" worker="$3"
  for task in "${TASKS[@]}"; do
    local attempt=1
    while true; do
      if run_one "$task" "$seeds" "$gpu" "$worker"; then break; fi
      if [[ "$attempt" -ge 3 ]]; then echo "FAILED worker=$worker task=$task" >&2; return 1; fi
      attempt=$((attempt+1)); sleep 60
    done
    echo "TASK_DONE worker=$worker task=$task" >&2
  done
  echo "WORKER_DONE worker=$worker" >&2
}
run_worker "$1" "$2" "$3"
