#!/usr/bin/env bash
# Single clean relaunch of the tail: kill everything of mine, start 6 servers, wait for real
# readiness, then start exactly one worker per tail shard.
#
# Earlier attempts left duplicate workers (two scripts each launched 12) and, separately, the
# launcher's cleanup killed the servers as soon as its worker wait loop drained. Both are
# avoided here: one supervisor, one worker per shard, servers started and left alone.
set -uo pipefail
ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD=$ROOT/token-cd
CONDA_ROOT=/home/leju-suzhou/miniconda3
OVERLAY=$ROOT/pi05_sklearn_overlay
RUN_DIR=$TOKEN_CD/runs/pi0fast_l6_matched_9task_100seed
SERVER=$TOKEN_CD/research/semantic_token_cd/pi0_fast_l6_simpler_server.py
ROLLOUT=$TOKEN_CD/research/semantic_token_cd/pi0_fast_simpler_l6_matched_rollout.py
SHARD_DIR=$RUN_DIR/manifests_tail
LOG=$RUN_DIR/logs/relaunch_tail.log
exec > "$LOG" 2>&1

echo "=== $(date '+%F %T') teardown ==="
pkill -f "pi0_fast_simpler_l6_matched_rollout" 2>/dev/null
pkill -f "pi0_fast_l6_simpler_server" 2>/dev/null
sleep 8
echo "  workers=$(pgrep -cf 'pi0_fast_simpler_l6_matched_rollout') servers=$(pgrep -cf 'pi0_fast_l6_simpler_server')"

echo "=== starting servers ==="
for gpu in 1 2 3; do
  for slot in 0 1; do
    if [ "$slot" -eq 0 ]; then port=$((8100 + gpu)); tag=a; else port=$((8110 + gpu)); tag=b; fi
    srvlog="$RUN_DIR/logs/server_gpu${gpu}_port${port}_google-${tag}.log"
    (
      source "$CONDA_ROOT/etc/profile.d/conda.sh"; conda activate pi05
      export CUDA_VISIBLE_DEVICES="$gpu"
      export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
      export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
      export OPENPI_DATA_HOME="$ROOT/checkpoints/openpi-assets" HF_HUB_OFFLINE=1
      export PI0FAST_CD_KMEANS=sklearn PI0FAST_CD_TAP_MODE=separate PI0FAST_CD_DECODE=verified
      export PI0FAST_CD_EMBED_JIT=0 PI0FAST_CD_BATCH=0
      export PYTHONPATH="$OVERLAY:$ROOT/openpi/src:$ROOT/openpi/packages/openpi-client/src:$TOKEN_CD"
      exec python -u "$SERVER" --gpu "$gpu" --port "$port" \
        --checkpoint "$ROOT/checkpoints/openpi_fast_finetuned/fractal"
    ) > "$srvlog" 2>&1 &
    sleep 2
  done
done

echo "=== waiting for all six to report listening ==="
for gpu in 1 2 3; do
  for slot in 0 1; do
    if [ "$slot" -eq 0 ]; then port=$((8100 + gpu)); tag=a; else port=$((8110 + gpu)); tag=b; fi
    srvlog="$RUN_DIR/logs/server_gpu${gpu}_port${port}_google-${tag}.log"
    for a in $(seq 1 180); do
      grep -q "server listening on" "$srvlog" 2>/dev/null && { echo "  port$port listening"; break; }
      sleep 5
    done
  done
done

echo "=== starting one worker per tail shard ==="
GPU_SEQ=(3 3 3 3 2 2 2 2 1 1 1 1)
i=0
for f in $(ls "$SHARD_DIR"/worker_*.json 2>/dev/null | sort); do
  wid=$(basename "$f" .json | sed 's/worker_//')
  gpu="${GPU_SEQ[$((i % 12))]}"
  port=$((8100 + gpu))
  wlog="$RUN_DIR/logs/tail_worker_${wid}_gpu${gpu}.log"
  (
    source "$CONDA_ROOT/etc/profile.d/conda.sh"; conda activate pi0
    export CUDA_VISIBLE_DEVICES="$gpu"
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
    export TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu
    export PYTHONPATH="/home/leju-suzhou/.cache/pi0fast_client_deps:$ROOT/openpi/packages/openpi-client/src:$ROOT/pcd_openvla_simpler_box_31b027e/source:$ROOT/SimplerEnv:$ROOT/SpatialVLA-SimplerEnv:$ROOT/SpatialVLA-SimplerEnv/ManiSkill2_real2sim:$ROOT/SpatialVLA/ManiSkill2_real2sim:$TOKEN_CD:$ROOT/openpi/src"
    exec python -u "$ROLLOUT" \
      --gpu "$gpu" --worker-id "$wid" --shard "$f" \
      --host 127.0.0.1 --port "$port" \
      --out "$RUN_DIR" --checkpoint "$ROOT/checkpoints/openpi_fast_finetuned/fractal" --rt-depth 8
  ) > "$wlog" 2>&1 &
  echo "  worker $wid -> gpu$gpu port$port"
  i=$((i+1))
  sleep 2
done
echo "=== $(date '+%F %T') launched $i workers ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 1,2,3
