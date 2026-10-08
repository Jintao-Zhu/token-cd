#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 GPU TASK [TASK ...]" >&2
  exit 2
fi
GPU="$1"
shift
TASKS=("$@")
ROOT="/home/leju-suzhou/zjt_ws"
REPO="$ROOT/token-cd"
SOURCE_ARTIFACT="$REPO/artifacts/univla_hidden_cd_lambda_010_015_rtdepth0_0_99_v1"
ARTIFACT="$REPO/artifacts/univla_hidden_cd_lambda_035_rtdepth0_0_99_v1"
PROTOCOL="UNIVLA_NINE_TASK_HIDDEN_CD_LAMBDA_035_RT_DEPTH0_V1"
export UNIVLA_ALLOW_SEED_ONLY_REFERENCE=1

mkdir -p "$ARTIFACT/logs" "$ARTIFACT/worker_manifests" "$ARTIFACT/locks" "$ARTIFACT/pairs" "$ARTIFACT/episodes"

run_google() {
  local task="$1" worker="$2"
  echo "[$(date -Is)] starting lambda-0.35 for $task; using the same task-seed, with hash equality retained as diagnostic only"
  env UNIVLA_ARTIFACT="$ARTIFACT" \
      UNIVLA_PROTOCOL_ID="$PROTOCOL" \
      UNIVLA_ARMS="hidden_cd_035" \
      UNIVLA_REFERENCE_PAIR_ARTIFACT="$SOURCE_ARTIFACT" \
      bash "$REPO/scripts/run_univla_ft30k_depth0_paired_fill_worker.sh" \
      "$GPU" "$task" "0-99" "$worker"
}

run_widowx() {
  local task="$1" worker="$2"
  echo "[$(date -Is)] starting lambda-0.35 for $task; using the same task-seed, with hash equality retained as diagnostic only"
  env UNIVLA_ARTIFACT="$ARTIFACT" \
      UNIVLA_PROTOCOL_ID="$PROTOCOL" \
      UNIVLA_ARMS="hidden_cd_035" \
      UNIVLA_REFERENCE_PAIR_ARTIFACT="$SOURCE_ARTIFACT" \
      UNIVLA_SNAPSHOT_ROOT="$SOURCE_ARTIFACT/snapshots" \
      bash "$REPO/scripts/run_univla_hidden_cd_depth0_three_arm_worker.sh" \
      "$GPU" "$task" "0-99" "hidden_cd_035" "$worker"
}

for task in "${TASKS[@]}"; do
  case "$task" in
    google_robot_*) run_google "$task" "univla_hcd035_g${GPU}_$(echo "$task" | sed 's/^google_robot_//')" ;;
    widowx_*) run_widowx "$task" "univla_hcd035_g${GPU}_$(echo "$task" | sed 's/^widowx_//')" ;;
    *) echo "unsupported task: $task" >&2; exit 2 ;;
  esac
done

echo "GPU $GPU lambda-0.35 queued task group complete: ${TASKS[*]}"
