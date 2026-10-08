#!/usr/bin/env bash
set -u
if [[ $# -ne 6 ]]; then
  echo "usage: $0 GPU ARTIFACT SNAPSHOT_ARTIFACT WORKER PHASE 'task=lo-hi,task=lo-hi'" >&2
  exit 2
fi
GPU=$1
ARTIFACT=$2
SNAPSHOTS=$3
WORKER=$4
PHASE=$5
JOBS=$6
ROOT=/home/leju-suzhou/zjt_ws/token-cd
source "$ROOT/scripts/activate_simpler_local.sh"
set +e
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$ROOT" || exit 2
IFS=',' read -r -a JOB_LIST <<< "$JOBS"
for item in "${JOB_LIST[@]}"; do
  task=${item%%=*}
  seeds=${item#*=}
  echo "[$(date -Is)] worker=$WORKER gpu=$GPU task=$task seeds=$seeds phase=$PHASE"
  "$SIMPLER_ENV/bin/python" -u research/semantic_token_cd/l11_reconstruction_fresh_rollout.py \
    --task "$task" --seeds "$seeds" --gpu "$GPU" --worker-id "$WORKER" \
    --artifact "$ARTIFACT" --snapshot-artifact "$SNAPSHOTS" --phase "$PHASE"
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[$(date -Is)] FAILED worker=$WORKER task=$task seeds=$seeds exit=$rc"
    exit "$rc"
  fi
done
echo "[$(date -Is)] COMPLETE worker=$WORKER"
