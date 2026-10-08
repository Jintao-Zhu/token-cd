#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $(( $# % 2 )) -ne 0 ]]; then
  echo "usage: $0 GPU WORKER_ID TASK SEEDS [TASK SEEDS ...]" >&2
  exit 2
fi
GPU="$1"
WORKER_ID="$2"
shift 2
ROOT="/home/leju-suzhou/zjt_ws"
REPO="$ROOT/token-cd"
ARTIFACT="$REPO/artifacts/openvla_guidance_strength_scan_100_199_v1"
ARMS="lambda_025,lambda_075,lambda_100"

source "$REPO/scripts/activate_simpler_local.sh"
mkdir -p "$ARTIFACT/logs"
cd "$REPO"

while [[ $# -gt 0 ]]; do
  TASK="$1"
  SEEDS="$2"
  shift 2
  echo "[$(date -Is)] worker=$WORKER_ID gpu=$GPU task=$TASK seeds=$SEEDS arms=$ARMS"
  python "$REPO/research/semantic_token_cd/seven_card_openvla_rollout.py" \
    --card 6 \
    --artifact "$ARTIFACT" \
    --snapshot-artifact "$REPO/artifacts/vanilla_recon_shr_canonical_0_299_v2" \
    --task "$TASK" --seeds "$SEEDS" --gpu "$GPU" \
    --worker-id "$WORKER_ID" --arms "$ARMS"
done
