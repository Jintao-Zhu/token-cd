#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 6 || $(( ($# - 3) % 3 )) -ne 0 ]]; then
  echo "usage: $0 MODE GPU WORKER ARM TASK SEEDS [ARM TASK SEEDS ...]" >&2
  exit 2
fi
MODE=$1 GPU=$2 WORKER=$3
shift 3
ROOT=/home/leju-suzhou/zjt_ws/token-cd
BASE="$ROOT/artifacts/l11_matched_dose_response_fig4_v1"
SNAPSHOT="$ROOT/artifacts/vanilla_recon_shr_canonical_0_299_v2"
if [[ "$MODE" != preflight && "$MODE" != formal ]]; then
  echo "invalid mode: $MODE" >&2
  exit 2
fi
source "$ROOT/scripts/activate_simpler_local.sh"
cd "$ROOT"
while [[ $# -gt 0 ]]; do
  ARM=$1 TASK=$2 SEEDS=$3
  shift 3
  if [[ "$MODE" == preflight ]]; then
    ARTIFACT="$BASE/preflight/$ARM"
  else
    ARTIFACT="$BASE/formal/$ARM"
  fi
  echo "[$(date -Is)] fig4 mode=$MODE gpu=$GPU worker=$WORKER arm=$ARM task=$TASK seeds=$SEEDS"
  python research/semantic_token_cd/l11_table2_openvla_rollout.py \
    --card 6 --artifact "$ARTIFACT" --snapshot-artifact "$SNAPSHOT" \
    --task "$TASK" --seeds "$SEEDS" --gpu "$GPU" --worker-id "$WORKER" \
    --arms "$ARM"
done
echo "[$(date -Is)] fig4 worker complete mode=$MODE worker=$WORKER"
