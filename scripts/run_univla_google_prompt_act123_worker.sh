#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 5 ]]; then
  echo "usage: $0 GPU TASK SEEDS WORKER_ID RENDER_DEPTH(0|8)" >&2
  exit 2
fi
GPU="$1" TASK="$2" SEEDS="$3" WORKER_ID="$4" DEPTH="$5"
[[ "$GPU" =~ ^[123]$ ]] || { echo "only GPUs 1,2,3 are allowed" >&2; exit 2; }
[[ "$DEPTH" == 0 || "$DEPTH" == 8 ]] || { echo "render depth must be 0 or 8" >&2; exit 2; }
ROOT="/home/leju-suzhou/zjt_ws"
REPO="$ROOT/token-cd"
RUN_DIR="$ROOT/checkpoints/univla-fractal-google-runs/univla-7b-224-sft-simpler-bridge+fractal20220817_data+b12+lr-3.5e-05+lora-r16+dropout-0.0--fractal_google_resume3k_to30k_20261002=w-LowLevelDecoder-ws-10"
CHECKPOINT="$RUN_DIR--27000_chkpt"
DECODER="$RUN_DIR/action_decoder-27000.pt"
if [[ "$DEPTH" == 8 ]]; then
  SUFFIX="rtfull"
  ARTIFACT="$REPO/artifacts/univla_fractal_google_ft30k_hidden_cd_prompt_act123_rtfull_0_99_v1"
  REFERENCE="$REPO/artifacts/univla_fractal_google_ft_30k_vanilla_vs_l14matched_rtfull_100_v1"
else
  SUFFIX="rtdepth0"
  ARTIFACT="$REPO/artifacts/univla_fractal_google_ft30k_hidden_cd_prompt_act123_rtdepth0_0_99_v1"
  REFERENCE="$REPO/artifacts/univla_fractal_google_ft30k_vanilla_vs_l14matched_rtdepth0_100_v1"
fi
SOCKET="/tmp/${WORKER_ID}.sock"

for required in "$CHECKPOINT/config.json" "$CHECKPOINT/dataset_statistics.json" "$DECODER"; do
  [[ -f "$required" ]] || { echo "Missing checkpoint component: $required" >&2; exit 1; }
done
mkdir -p "$ARTIFACT/logs" "$ARTIFACT/worker_manifests"
rm -f "$SOCKET"
export CUDA_VISIBLE_DEVICES="$GPU"
export UNIVLA_L14_SOCK="/tmp/${WORKER_ID}.unused_l14.sock"
export UNIVLA_HIDDEN_SOCK="$SOCKET"
export UNIVLA_POLICY_SETUP=google_robot
export UNIVLA_CHECKPOINT="$CHECKPOINT"
export UNIVLA_DECODER_PATH="$DECODER"
export UNIVLA_UNNORM_KEY=fractal20220817_data
export UNIVLA_CHECKPOINT_DOMAIN=fractal_google_ft_30k
export UNIVLA_GRIPPER_ADAPTER=absolute_binary_delta_v1
export UNIVLA_PROTOCOL_ID="UNIVLA_FT30K_GOOGLE_HIDDEN_PROMPT_ACT123_${SUFFIX^^}_0_99_V1"
export UNIVLA_ARMS=hidden_cd_025,hidden_cd_05
export UNIVLA_HIDDEN_ALIGNMENT=prompt_act123
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
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
python "$REPO/research/semantic_token_cd/univla_hidden_cd_google_model_service.py" \
  >"$ARTIFACT/logs/${WORKER_ID}.model.log" 2>&1 &
MODEL_PID=$!
ready=0
for _ in $(seq 1 600); do
  if rg -q '"model_ready": true' "$ARTIFACT/logs/${WORKER_ID}.model.log"; then ready=1; break; fi
  if ! kill -0 "$MODEL_PID" 2>/dev/null; then cat "$ARTIFACT/logs/${WORKER_ID}.model.log" >&2; exit 1; fi
  sleep 1
done
if [[ "$ready" -ne 1 ]]; then
  echo "UniVLA model startup timeout: $WORKER_ID" >&2
  cat "$ARTIFACT/logs/${WORKER_ID}.model.log" >&2
  exit 1
fi
rg -q '"hidden_alignment": "prompt_act123"' "$ARTIFACT/logs/${WORKER_ID}.model.log" || {
  echo "Model service did not acknowledge prompt_act123 alignment" >&2; exit 1;
}

source "$REPO/scripts/activate_simpler_local.sh"
export CUDA_VISIBLE_DEVICES="$GPU"
python "$REPO/research/semantic_token_cd/univla_hidden_cd_google_rollout.py" \
  --task "$TASK" --seeds "$SEEDS" --worker-id "$WORKER_ID" \
  --artifact "$ARTIFACT" --socket "$SOCKET" --reference-artifact "$REFERENCE" \
  --render-depth "$DEPTH" \
  2>&1 | tee "$ARTIFACT/logs/${WORKER_ID}.rollout.log"
