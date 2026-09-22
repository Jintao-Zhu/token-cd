#!/usr/bin/env bash
set -euo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
ROOT="$WS/artifacts/l11_budget_response_curve_v1"
CANON="$WS/artifacts/vanilla_recon_shr_canonical_0_299_v2"
export PYTHONPATH="$WS:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$WS"
run_task() {
  local gpu="$1" task="$2" worker="$3"
  local a=1
  while true; do
    if "$PY" research/semantic_token_cd/phase_b_continuation_rollout.py \
         --artifact "$ROOT" --canonical "$CANON" --task "$task" \
         --gpu "$gpu" --worker-id "$worker"; then return 0; fi
    if [[ "$a" -ge 8 ]]; then echo "GIVEUP $worker $task" >&2; return 1; fi
    echo "RETRY $a $worker $task" >&2
    a=$((a+1)); sleep 30
  done
}
run_task "$2" "$3" "$4"
