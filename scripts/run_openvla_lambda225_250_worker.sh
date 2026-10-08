#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 5 || $(( $# % 2 )) -ne 1 ]]; then
  echo "usage: $0 GPU WORKER_ID MODE TASK SEEDS [TASK SEEDS ...]" >&2
  exit 2
fi
GPU=$1
WORKER=$2
MODE=$3
shift 3
ROOT=/home/leju-suzhou/zjt_ws/token-cd
ARTIFACT=$ROOT/artifacts/openvla_global_lambda_grid_225_250_0_199_v1
SNAPSHOT=$ROOT/artifacts/vanilla_recon_shr_canonical_0_299_v2
ARMS=lambda_225,lambda_250
source "$ROOT/scripts/activate_simpler_local.sh"
export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$ARTIFACT/logs"
cd "$ROOT"
chunk=0
while [[ $# -gt 0 ]]; do
  TASK=$1
  SEEDS=$2
  shift 2
  if [[ "$MODE" == "preflight" ]]; then SEEDS=0-4; fi
  chunk=$((chunk+1))
  attempt=1
  while true; do
    LOG=$ARTIFACT/logs/${WORKER}_${MODE}_chunk${chunk}_attempt${attempt}.log
    echo "[$(date -Is)] worker=$WORKER gpu=$GPU mode=$MODE task=$TASK seeds=$SEEDS arms=$ARMS"
    set +e
    python research/semantic_token_cd/openvla_global_lambda_grid_rollout.py \
      --card 6 --artifact "$ARTIFACT" --snapshot-artifact "$SNAPSHOT" \
      --task "$TASK" --seeds "$SEEDS" --gpu "$GPU" \
      --worker-id "$WORKER" --arms "$ARMS" 2>&1 | tee -a "$LOG"
    status=${PIPESTATUS[0]}
    set -e
    if [[ $status -eq 0 ]]; then break; fi
    if [[ $attempt -lt 2 ]] && grep -Eqi 'VK_ERROR_DEVICE_LOST|ErrorDeviceLost|device lost' "$LOG"; then
      echo "[$(date -Is)] retrying technical Vulkan DeviceLost once; lambda/config unchanged" >&2
      attempt=$((attempt+1)); sleep 5; continue
    fi
    exit "$status"
  done
done
echo "[$(date -Is)] worker=$WORKER mode=$MODE completed assigned chunks"
