#!/usr/bin/env bash
# Add the two new guidance arms to the existing workers' queues.
#
# The user's intent is to extend the TASK QUEUE, not to add processes: the same 15 worker
# slots and the same 6 policy servers keep running, but each worker's job list now contains
# six arms instead of five. Workers read their shard once at startup, so they are restarted;
# the resume logic skips every (task, seed, arm) whose result file is already complete, so the
# restart costs nothing and the new arms are simply appended to the work each worker has left.
#
# Servers are left untouched (the launcher is NOT used here, since it would try to rebind the
# ports already in use).
set -uo pipefail
ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD=$ROOT/token-cd
CONDA_ROOT=/home/leju-suzhou/miniconda3
RUN_DIR=$TOKEN_CD/runs/pi0fast_l6_matched_9task_100seed
ROLLOUT=$TOKEN_CD/research/semantic_token_cd/pi0_fast_simpler_l6_matched_rollout.py

GPUS=(1 1 1 1 1 2 2 2 2 2 3 3 3 3 3)
PORTS=(8101 8101 8101 8101 8101 8102 8102 8102 8102 8102 8103 8103 8103 8103 8103)

echo "=== regenerating manifests with six arms ==="
PI0FAST_SEED_CHUNKS=3 PYTHONPATH="$ROOT/openpi/packages/openpi-client/src:$ROOT/openpi/src:$TOKEN_CD" \
  "$CONDA_ROOT/envs/pi05/bin/python" "$ROLLOUT" --gpu 0 --manifest-only \
  --manifest-subdir manifests --seed-start 0 --seed-end 99 --out "$RUN_DIR"

echo "=== stopping the current workers (servers untouched) ==="
pkill -f "[p]i0_fast_simpler_l6_matched_rollout" 2>/dev/null
sleep 6
echo "  workers now: $(pgrep -cf '[p]i0_fast_simpler_l6_matched_rollout')"
echo "  servers still up: $(pgrep -cf '[p]i0_fast_l6_simpler_server')"

echo "=== relaunching the same 15 slots with the extended queue ==="
for w in $(seq 0 14); do
  gpu="${GPUS[$w]}"; port="${PORTS[$w]}"
  shard="$RUN_DIR/manifests/worker_${w}.json"
  [ -f "$shard" ] || { echo "  missing $shard"; continue; }
  log="$RUN_DIR/logs/worker_${w}_gpu${gpu}.log"
  (
    source "$CONDA_ROOT/etc/profile.d/conda.sh"; conda activate pi0
    export CUDA_VISIBLE_DEVICES="$gpu"
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
    export TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu
    export PYTHONPATH="/home/leju-suzhou/.cache/pi0fast_client_deps:$ROOT/openpi/packages/openpi-client/src:$ROOT/pcd_openvla_simpler_box_31b027e/source:$ROOT/SimplerEnv:$ROOT/SpatialVLA-SimplerEnv:$ROOT/SpatialVLA-SimplerEnv/ManiSkill2_real2sim:$ROOT/SpatialVLA/ManiSkill2_real2sim:$TOKEN_CD:$ROOT/openpi/src"
    exec python -u "$ROLLOUT" \
      --gpu "$gpu" --worker-id "$w" --shard "$shard" \
      --host 127.0.0.1 --port "$port" \
      --out "$RUN_DIR" --checkpoint "$ROOT/checkpoints/openpi_fast_finetuned/fractal" --rt-depth 8
  ) > "$log" 2>&1 &
  sleep 1
done
echo "=== relaunched ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 1,2,3
