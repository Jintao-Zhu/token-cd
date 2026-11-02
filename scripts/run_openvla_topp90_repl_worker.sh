#!/usr/bin/env bash
# One worker for the TopP90-vs-matched replication at lambda=1.0 on an INDEPENDENT
# seed range (100-199). Chunks are TASK:ARMS:SEEDS.
set -uo pipefail
GPU="$1"; WORKER_ID="$2"; shift 2
ROOT=/home/leju-suzhou/zjt_ws
REPO=$ROOT/token-cd
ARTIFACT="${ARTIFACT:-$REPO/artifacts/openvla_topp90_repl_100_199_v1}"
PROTOCOL="${PROTOCOL:-OPENVLA_TOPP90_REPLICATION_100_199_V1}"
LOCKSEEDS="${LOCKSEEDS:-100-199}"
ROLLOUT=research/semantic_token_cd/openvla_budget_rule_rollout.py
source "$REPO/scripts/activate_simpler_local.sh"
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$ARTIFACT/logs"
cd "$REPO"
idx=0
for spec in "$@"; do
  idx=$((idx+1))
  TASK="${spec%%:*}"; rest="${spec#*:}"; ARMS="${rest%%:*}"; SEEDS="${rest#*:}"
  attempt=1
  while true; do
    log="$ARTIFACT/logs/${WORKER_ID}_c${idx}_a${attempt}.log"
    echo "[$(date -Is)] worker=$WORKER_ID gpu=$GPU chunk=$idx task=$TASK arms=$ARMS seeds=$SEEDS"
    set +e
    python "$ROLLOUT" --artifact "$ARTIFACT" --task "$TASK" --seeds "$SEEDS" --gpu "$GPU" \
      --worker-id "$WORKER_ID" --arms "$ARMS" --protocol-id "$PROTOCOL" --lock-seeds "$LOCKSEEDS" >> "$log" 2>&1
    status=$?
    set -e
    [[ "$status" -eq 0 ]] && break
    if [[ "$attempt" -lt 2 ]] && grep -Eqi 'VK_ERROR_DEVICE_LOST|ErrorDeviceLost|device lost' "$log"; then
      echo "[$(date -Is)] retrying after Vulkan DeviceLost; config unchanged" >&2
      attempt=$((attempt+1)); sleep 5; continue
    fi
    echo "[$(date -Is)] chunk $idx FAILED status=$status (see $log)" >&2
    exit "$status"
  done
done
echo "[$(date -Is)] worker=$WORKER_ID completed all $idx chunks"
