#!/usr/bin/env bash
set -u
if [[ $# -ne 7 ]]; then
  echo "usage: $0 GPU ARTIFACT SNAPSHOT_ARTIFACT WORKER PHASE TASK SEEDS" >&2
  exit 2
fi
GPU=$1
ARTIFACT=$2
SNAPSHOTS=$3
WORKER=$4
PHASE=$5
TASK=$6
SEEDS=$7
ROOT=/home/leju-suzhou/zjt_ws/token-cd
source "$ROOT/scripts/activate_simpler_local.sh"
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$ROOT" || exit 2
echo "[$(date -Is)] worker=$WORKER gpu=$GPU task=$TASK seeds=$SEEDS phase=$PHASE"
for attempt in 1 2 3; do
  echo "[$(date -Is)] attempt=$attempt worker=$WORKER task=$TASK seeds=$SEEDS"
  "$SIMPLER_ENV/bin/python" -u research/semantic_token_cd/l11_reconstruction_fresh_rollout.py \
    --task "$TASK" --seeds "$SEEDS" --gpu "$GPU" --worker-id "$WORKER" \
    --artifact "$ARTIFACT" --snapshot-artifact "$SNAPSHOTS" --phase "$PHASE" \
    --protocol OPENVLA_L11_HARMONIC_ZERO_GLOBALMEAN_C_EXTENDED_300_499_V1
  rc=$?
  if [[ $rc -eq 0 ]]; then
    break
  fi
  echo "[$(date -Is)] attempt_failed worker=$WORKER exit=$rc; retrying the same seeds (infra errors are not counted as task failures)"
done
if [[ $rc -ne 0 ]]; then
  echo "[$(date -Is)] FAILED after retries worker=$WORKER task=$TASK seeds=$SEEDS exit=$rc"
  exit "$rc"
fi
echo "[$(date -Is)] COMPLETE worker=$WORKER"
