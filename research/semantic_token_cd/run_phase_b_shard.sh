#!/usr/bin/env bash
set -euo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
ROOT="$WS/artifacts/l11_budget_response_curve_v1"
CANON="$WS/artifacts/vanilla_recon_shr_canonical_0_299_v2"
export PYTHONPATH="$WS:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$WS"
a=1
while true; do
  if "$PY" research/semantic_token_cd/phase_b_continuation_rollout.py \
       --artifact "$ROOT" --canonical "$CANON" --task "$2" \
       --gpu "$1" --worker-id "$5" --shard-index "$4" --shard-count "$3"; then exit 0; fi
  if [[ "$a" -ge 8 ]]; then echo "GIVEUP $5" >&2; exit 1; fi
  echo "RETRY $a $5" >&2; a=$((a+1)); sleep 30
done
