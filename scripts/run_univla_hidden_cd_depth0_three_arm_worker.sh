#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 5 ]]; then
  echo "usage: $0 GPU TASK SEEDS ARMS WORKER_ID" >&2
  exit 2
fi
GPU="$1" TASK="$2" SEEDS="$3" ARMS="$4" WORKER_ID="$5"
ROOT="/home/leju-suzhou/zjt_ws"
REPO="$ROOT/token-cd"
ARTIFACT="${UNIVLA_ARTIFACT:-$REPO/artifacts/univla_nine_task_hidden_cd_rtdepth0_extension_v1}"
PROTOCOL="${UNIVLA_PROTOCOL_ID:-UNIVLA_BRIDGE_HIDDEN_CD_RT_DEPTH0_3ARM_300_V1}"
SOCKET="/tmp/${WORKER_ID}.sock"
mkdir -p "$ARTIFACT/logs" "$ARTIFACT/worker_manifests"
rm -f "$SOCKET"
export CUDA_VISIBLE_DEVICES="$GPU"
export UNIVLA_L14_SOCK="/tmp/${WORKER_ID}.unused_l14.sock"
export UNIVLA_HIDDEN_SOCK="$SOCKET"
export UNIVLA_PROTOCOL_ID="$PROTOCOL"
export UNIVLA_ARMS="$ARMS"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

cleanup() {
  if [[ -n "${MODEL_PID:-}" ]] && kill -0 "$MODEL_PID" 2>/dev/null; then
    kill "$MODEL_PID" 2>/dev/null || true
    wait "$MODEL_PID" 2>/dev/null || true
  fi
  rm -f "$SOCKET"
}
trap cleanup EXIT INT TERM

source "$ROOT/UniVLA/scripts/activate_local_simpler.sh"
python "$REPO/research/semantic_token_cd/univla_hidden_cd_depth0_three_arm_model_service.py" \
  >"$ARTIFACT/logs/${WORKER_ID}.model.log" 2>&1 &
MODEL_PID=$!
ready=0
for _ in $(seq 1 360); do
  if rg -q '"model_ready": true' "$ARTIFACT/logs/${WORKER_ID}.model.log"; then ready=1; break; fi
  if ! kill -0 "$MODEL_PID" 2>/dev/null; then cat "$ARTIFACT/logs/${WORKER_ID}.model.log" >&2; exit 1; fi
  sleep 1
done
if [[ "$ready" -ne 1 ]]; then
  echo "UniVLA Bridge service startup timeout: $WORKER_ID" >&2
  cat "$ARTIFACT/logs/${WORKER_ID}.model.log" >&2
  exit 1
fi

# Keep the UniVLA checkpoint/model in its supported inference environment, but
# run SIMPLER SAPIEN2 RT rendering in the same local runtime used by the
# successful Google workers. Switching this shell's Python environment does
# not affect the already-running model-service process.
source "$REPO/scripts/activate_simpler_local.sh"
export CUDA_VISIBLE_DEVICES="$GPU"
REFERENCE_PAIR_ARGS=()
if [[ -n "${UNIVLA_REFERENCE_PAIR_ARTIFACT:-}" ]]; then
  REFERENCE_PAIR_ARGS+=(--reference-pair-artifact "$UNIVLA_REFERENCE_PAIR_ARTIFACT")
fi
if [[ "${UNIVLA_ALLOW_SEED_ONLY_REFERENCE:-0}" == "1" ]]; then
  REFERENCE_PAIR_ARGS+=(--allow-seed-only-reference)
fi
python "$REPO/research/semantic_token_cd/univla_hidden_cd_depth0_three_arm_rollout.py" \
  --task "$TASK" --seeds "$SEEDS" --worker-id "$WORKER_ID" \
  --artifact "$ARTIFACT" --socket "$SOCKET" --arms "$ARMS" --no-historical-controls \
  "${REFERENCE_PAIR_ARGS[@]}" \
  2>&1 | tee "$ARTIFACT/logs/${WORKER_ID}.rollout.log"
