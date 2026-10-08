#!/usr/bin/env bash
set -u

ROOT=/home/leju-suzhou/zjt_ws
OPENPI="$ROOT/openpi"
TOKEN_CD="$ROOT/token-cd"
CHECKPOINT="$ROOT/checkpoints/openpi-assets/openpi-assets/checkpoints/pi0_fast_base"
LOG_DIR="$ROOT/logs"
SERVER_LOG="$LOG_DIR/pi0fast_server.log"
CLIENT_LOG="$LOG_DIR/pi0fast_simpler_smoke.log"
STATUS_LOG="$LOG_DIR/pi0fast_pipeline.log"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$STATUS_LOG") 2>&1
echo "[$(date -Is)] waiting for completed official pi0-FAST checkpoint at $CHECKPOINT/params"

while [[ ! -f "$CHECKPOINT/params/_CHECKPOINT_METADATA" ]]; do
  sleep 10
done

echo "[$(date -Is)] checkpoint complete; starting policy server on physical GPU 6 if needed"
if ! tmux has-session -t codex_pi0fast_server 2>/dev/null; then
  tmux new-session -d -s codex_pi0fast_server
  tmux send-keys -t codex_pi0fast_server "source /home/leju-suzhou/miniconda3/etc/profile.d/conda.sh && conda activate pi05 && cd '$OPENPI' && export CUDA_VISIBLE_DEVICES=6 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.50 HF_ENDPOINT=https://huggingface.co HTTPS_PROXY=http://127.0.0.1:7897 HTTP_PROXY=http://127.0.0.1:7897 ALL_PROXY=http://127.0.0.1:7897 OPENPI_DATA_HOME='$ROOT/checkpoints/openpi-assets' PYTHONPATH='$OPENPI/packages/openpi-client/src:$HOME/.cache/pi0fast_client_deps' && python -u '$TOKEN_CD/research/semantic_token_cd/pi0_fast_simpler_server.py' --port 8117 > '$SERVER_LOG' 2>&1" C-m
fi

server_ready=0
for _ in $(seq 1 360); do
  if grep -q 'FAST decode warmup passed' "$SERVER_LOG" 2>/dev/null; then
    server_ready=1
    break
  fi
  if ! tmux has-session -t codex_pi0fast_server 2>/dev/null; then
    echo "[$(date -Is)] ERROR: policy server exited before warmup; inspect $SERVER_LOG"
    exit 3
  fi
  sleep 10
done
if [[ "$server_ready" != 1 ]]; then
  echo "[$(date -Is)] ERROR: policy server warmup timed out; inspect $SERVER_LOG"
  exit 4
fi

echo "[$(date -Is)] policy warmup passed; starting one SIMPLER vanilla episode on GPU 6"
tmux new-session -d -s codex_pi0fast_smoke
tmux send-keys -t codex_pi0fast_smoke "source /home/leju-suzhou/miniconda3/etc/profile.d/conda.sh && conda activate pi0 && export CUDA_VISIBLE_DEVICES=6 VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json TOKENIZERS_PARALLELISM=false PYTHONPATH='$HOME/.cache/pi0fast_client_deps:$OPENPI/packages/openpi-client/src:$ROOT/pcd_openvla_simpler_box_31b027e/source:$ROOT/SimplerEnv:$ROOT/SpatialVLA-SimplerEnv:$ROOT/SpatialVLA/ManiSkill2_real2sim:$TOKEN_CD' && python -u '$TOKEN_CD/research/semantic_token_cd/pi0_fast_simpler_smoke.py' --task google_robot_pick_coke_can --seed 999 --gpu 6 --port 8117 > '$CLIENT_LOG' 2>&1" C-m

for _ in $(seq 1 720); do
  if [[ -f "$TOKEN_CD/artifacts/pi0_fast_simpler_vanilla_smoke_v1/google_robot_pick_coke_can_seed999_summary.json" ]]; then
    echo "[$(date -Is)] SIMPLER smoke completed; see $CLIENT_LOG and artifact summary"
    exit 0
  fi
  if ! tmux has-session -t codex_pi0fast_smoke 2>/dev/null; then
    echo "[$(date -Is)] ERROR: SIMPLER smoke exited without writing a summary; inspect $CLIENT_LOG"
    exit 5
  fi
  sleep 10
done

echo "[$(date -Is)] ERROR: SIMPLER smoke timed out; inspect $CLIENT_LOG"
exit 6
