#!/usr/bin/env bash
# Start a bridge-checkpoint server + run the interface check on one WidowX episode.
set -uo pipefail
ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD=$ROOT/token-cd
CONDA_ROOT=/home/leju-suzhou/miniconda3
OVERLAY=$ROOT/pi05_sklearn_overlay
GPU="${1:-1}"; PORT=8121; TASK="${2:-widowx_carrot_on_plate}"; SEED="${3:-0}"
OUT=$TOKEN_CD/runs/widowx_interface_check
mkdir -p "$OUT/logs"

( source "$CONDA_ROOT/etc/profile.d/conda.sh"; conda activate pi05
  export CUDA_VISIBLE_DEVICES="$GPU"
  export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.18
  export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
  export OPENPI_DATA_HOME="$ROOT/checkpoints/openpi-assets" HF_HUB_OFFLINE=1
  export PI0FAST_CD_KMEANS=sklearn PI0FAST_CD_TAP_MODE=separate PI0FAST_CD_DECODE=verified
  export PYTHONPATH="$OVERLAY:$ROOT/openpi/src:$ROOT/openpi/packages/openpi-client/src:$TOKEN_CD"
  exec python -u "$TOKEN_CD/research/semantic_token_cd/pi0_fast_l6_simpler_server.py" \
    --gpu "$GPU" --port "$PORT" --checkpoint "$ROOT/checkpoints/openpi_fast_finetuned/bridge"
) > "$OUT/logs/server_bridge.log" 2>&1 &
SRV=$!
for i in $(seq 1 120); do
  grep -q "warmups passed" "$OUT/logs/server_bridge.log" 2>/dev/null && break
  kill -0 $SRV 2>/dev/null || { echo "server died"; tail -5 "$OUT/logs/server_bridge.log"; exit 1; }
  sleep 5
done
echo "bridge server ready"

( source "$CONDA_ROOT/etc/profile.d/conda.sh"; conda activate pi0
  export CUDA_VISIBLE_DEVICES="$GPU"
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
  export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export JAX_PLATFORMS=cpu
  export WK_TASK="$TASK" WK_SEED="$SEED" WK_PORT="$PORT"
  export PYTHONPATH="/home/leju-suzhou/.cache/pi0fast_client_deps:$ROOT/openpi/packages/openpi-client/src:$ROOT/pcd_openvla_simpler_box_31b027e/source:$ROOT/SimplerEnv:$ROOT/SpatialVLA-SimplerEnv:$ROOT/SpatialVLA-SimplerEnv/ManiSkill2_real2sim:$ROOT/SpatialVLA/ManiSkill2_real2sim:$TOKEN_CD:$ROOT/openpi/src"
  exec python -u "$TOKEN_CD/scripts/check_widowx_interface.py"
) > "$OUT/logs/check.log" 2>&1
echo "check rc=$?"
kill $SRV 2>/dev/null
echo "=== result ==="; cat "$OUT/logs/check.log" | tail -30
