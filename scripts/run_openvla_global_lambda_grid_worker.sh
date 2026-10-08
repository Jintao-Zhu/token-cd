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
ARTIFACT="$REPO/artifacts/openvla_global_lambda_grid_0_199_v1"
ARMS="lambda_000,lambda_025,lambda_050,lambda_075,lambda_100,lambda_125,lambda_150,lambda_175,lambda_200"

source "$REPO/scripts/activate_simpler_local.sh"
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$ARTIFACT/logs"
cd "$REPO"

chunk=0
while [[ $# -gt 0 ]]; do
  TASK="$1"
  SEEDS="$2"
  shift 2
  chunk=$((chunk + 1))
  echo "[$(date -Is)] worker=$WORKER_ID gpu=$GPU chunk=$chunk task=$TASK seeds=$SEEDS arms=$ARMS"
  attempt=1
  while true; do
    chunk_log="$ARTIFACT/logs/${WORKER_ID}_chunk${chunk}_attempt${attempt}.log"
    set +e
    CUDA_VISIBLE_DEVICES="$GPU" python "$REPO/research/semantic_token_cd/openvla_global_lambda_grid_rollout.py" \
      --card 6 \
      --artifact "$ARTIFACT" \
      --snapshot-artifact "$REPO/artifacts/vanilla_recon_shr_canonical_0_299_v2" \
      --task "$TASK" --seeds "$SEEDS" --gpu "$GPU" \
      --worker-id "$WORKER_ID" --arms "$ARMS" 2>&1 | tee -a "$chunk_log"
    status=${PIPESTATUS[0]}
    set -e
    if [[ "$status" -eq 0 ]]; then
      break
    fi
    if [[ "$attempt" -lt 2 ]] && grep -Eqi 'VK_ERROR_DEVICE_LOST|ErrorDeviceLost|device lost' "$chunk_log"; then
      echo "[$(date -Is)] retrying technical Vulkan DeviceLost once; experiment config unchanged" >&2
      attempt=$((attempt + 1))
      sleep 5
      continue
    fi
    exit "$status"
  done
done

echo "[$(date -Is)] worker=$WORKER_ID completed all assigned chunks"
