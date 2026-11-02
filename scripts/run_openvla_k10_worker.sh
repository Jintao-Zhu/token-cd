#!/usr/bin/env bash
# One K=10 worker: runs TASK:ARMS:SEEDS chunks sequentially.
# Usage: bash run_openvla_k10_worker.sh GPU WORKER_ID TASK:ARMS:SEEDS [ ... ]
set -uo pipefail
GPU="$1"; WORKER_ID="$2"; shift 2
ROOT=/home/leju-suzhou/zjt_ws
REPO=$ROOT/token-cd
ARTIFACT="$REPO/artifacts/openvla_k10_2lambda_200_299_v1"
PROTOCOL=OPENVLA_K10_2LAMBDA_200_299_V1
RUNNER=research/semantic_token_cd/openvla_matched_k10_rollout.py

source "$REPO/scripts/activate_simpler_local.sh"
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$ARTIFACT/logs"
cd "$REPO"

idx=0
for spec in "$@"; do
  idx=$((idx + 1))
  TASK="${spec%%:*}"; rest="${spec#*:}"; ARMS="${rest%%:*}"; SEEDS="${rest#*:}"
  attempt=1
  while true; do
    log="$ARTIFACT/logs/${WORKER_ID}_c${idx}_a${attempt}.log"
    echo "[$(date -Is)] worker=$WORKER_ID gpu=$GPU chunk=$idx task=$TASK arms=$ARMS seeds=$SEEDS attempt=$attempt"
    set +e
    python "$RUNNER" --artifact "$ARTIFACT" --task "$TASK" --seeds "$SEEDS" --gpu "$GPU" \
      --worker-id "$WORKER_ID" --arms "$ARMS" --protocol-id "$PROTOCOL" --lock-seeds 200-299 >> "$log" 2>&1
    status=$?
    set -e
    if [[ "$status" -eq 0 ]]; then break; fi
    if [[ "$attempt" -lt 2 ]] && grep -Eqi 'VK_ERROR_DEVICE_LOST|ErrorDeviceLost|device lost' "$log"; then
      echo "[$(date -Is)] retrying after Vulkan DeviceLost; config unchanged" >&2
      attempt=$((attempt + 1)); sleep 5; continue
    fi
    echo "[$(date -Is)] chunk $idx FAILED status=$status (see $log)" >&2
    exit "$status"
  done
done
echo "[$(date -Is)] worker=$WORKER_ID completed all $idx chunks"
