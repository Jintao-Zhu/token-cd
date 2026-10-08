#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 GPU TASK SEEDS WORKER_ID" >&2
  exit 2
fi
GPU="$1"
TASK="$2"
SEEDS="$3"
WORKER_ID="$4"
ROOT="/home/leju-suzhou/zjt_ws"
REPO="$ROOT/token-cd"
CHECKPOINT="$ROOT/checkpoints/univla-fractal-google-runs/univla-7b-224-sft-simpler-bridge+fractal20220817_data+b8+lr-0.00035+lora-r16+dropout-0.0--fractal_google_4gpu_3k=w-LowLevelDecoder-ws-10"
ARTIFACT="$REPO/artifacts/univla_fractal_google_ft_smoke_v1"
REFERENCE="$REPO/artifacts/univla_simpler_google_univla_bridge_l14_rtfull_100_zero_shot_v1"
SOCKET="/tmp/${WORKER_ID}.sock"

mkdir -p "$ARTIFACT/logs" "$ARTIFACT/worker_manifests"
rm -f "$SOCKET"
export CUDA_VISIBLE_DEVICES="$GPU"
export UNIVLA_L14_SOCK="$SOCKET"
export UNIVLA_POLICY_SETUP=google_robot
export UNIVLA_CHECKPOINT="$CHECKPOINT"
export UNIVLA_UNNORM_KEY=fractal20220817_data
export UNIVLA_CHECKPOINT_DOMAIN=fractal_google_ft_3k
export UNIVLA_PROTOCOL_ID=UNIVLA_FRACTAL_GOOGLE_FT_VANILLA_SMOKE_RT_FULL_V1
export UNIVLA_ARMS=vanilla
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cleanup() {
  if [[ -n "${MODEL_PID:-}" ]] && kill -0 "$MODEL_PID" 2>/dev/null; then
    kill "$MODEL_PID" 2>/dev/null || true
    wait "$MODEL_PID" 2>/dev/null || true
  fi
  rm -f "$SOCKET"
}
trap cleanup EXIT INT TERM

source "$ROOT/UniVLA/scripts/activate_local_simpler.sh"
python "$REPO/research/semantic_token_cd/univla_l14_google_model_service.py" \
  >"$ARTIFACT/logs/${WORKER_ID}.model.log" 2>&1 &
MODEL_PID=$!

ready=0
for _ in $(seq 1 600); do
  if rg -q '"model_ready": true' "$ARTIFACT/logs/${WORKER_ID}.model.log"; then
    ready=1
    break
  fi
  if ! kill -0 "$MODEL_PID" 2>/dev/null; then
    cat "$ARTIFACT/logs/${WORKER_ID}.model.log" >&2
    exit 1
  fi
  sleep 1
done
if [[ "$ready" -ne 1 ]]; then
  echo "model service timeout for $WORKER_ID" >&2
  cat "$ARTIFACT/logs/${WORKER_ID}.model.log" >&2
  exit 1
fi

source "$REPO/scripts/activate_simpler_local.sh"
export CUDA_VISIBLE_DEVICES="$GPU"
python "$REPO/research/semantic_token_cd/univla_l14_google_rollout.py" \
  --task "$TASK" \
  --seeds "$SEEDS" \
  --worker-id "$WORKER_ID" \
  --artifact "$ARTIFACT" \
  --socket "$SOCKET" \
  --reference-artifact "$REFERENCE" \
  2>&1 | tee "$ARTIFACT/logs/${WORKER_ID}.rollout.log"
