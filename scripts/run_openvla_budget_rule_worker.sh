#!/usr/bin/env bash
# One OpenVLA budget-rule worker: run a list of chunks, each = TASK:ARMS:SEEDS.
#
# Usage: bash run_openvla_budget_rule_worker.sh GPU WORKER_ID TASK:ARMS:SEEDS [TASK:ARMS:SEEDS ...]
#   e.g. ... 1 openvla_br_g1_w1 google_robot_open_drawer:top_p085_l100:200-224 ...
#
# Resumes by re-running the same chunk: the rollout skips episodes whose summary+arrays exist.
# Retries once on a Vulkan DEVICE_LOST (technical failure only, config unchanged).
set -uo pipefail

GPU="$1"; WORKER_ID="$2"; shift 2
ROOT=/home/leju-suzhou/zjt_ws
REPO=$ROOT/token-cd
ARTIFACT="$REPO/artifacts/openvla_budget_rule_9task_200_299_v1"
PROTOCOL=OPENVLA_BUDGET_RULE_9TASK_200_299_V1
ROLLOUT=research/semantic_token_cd/openvla_budget_rule_rollout.py

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
    python "$ROLLOUT" --artifact "$ARTIFACT" --task "$TASK" --seeds "$SEEDS" --gpu "$GPU" \
      --worker-id "$WORKER_ID" --arms "$ARMS" --protocol-id "$PROTOCOL" --lock-seeds 200-299 \
      >> "$log" 2>&1
    status=$?
    set -e
    if [[ "$status" -eq 0 ]]; then break; fi
    if [[ "$attempt" -lt 2 ]] && grep -Eqi 'VK_ERROR_DEVICE_LOST|ErrorDeviceLost|device lost' "$log"; then
      echo "[$(date -Is)] retrying after Vulkan DeviceLost; experiment config unchanged" >&2
      attempt=$((attempt + 1)); sleep 5; continue
    fi
    echo "[$(date -Is)] chunk $idx FAILED status=$status (see $log)" >&2
    exit "$status"
  done
done

echo "[$(date -Is)] worker=$WORKER_ID completed all $idx chunks"
