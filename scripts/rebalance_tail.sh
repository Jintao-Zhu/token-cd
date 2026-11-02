#!/usr/bin/env bash
# Rebalance the tail of the run.
#
# place_apple is the critical path: 200-step episodes, ~358 still to do, and only three
# workers on it, while five other shards finished and their workers exited. The GPU memory
# headroom (33-40 GB per card free) allows several more rendering workers, so the remaining
# place_apple seeds are re-split across more workers. Resume logic makes already-completed
# episodes free, so over-splitting only wastes a little startup time.
set -uo pipefail
ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD=$ROOT/token-cd
CONDA_ROOT=/home/leju-suzhou/miniconda3
RUN_DIR=$TOKEN_CD/runs/pi0fast_l6_matched_9task_100seed
TASK=google_robot_place_apple_in_closed_top_drawer
NEW=${1:-6}          # number of workers to run place_apple with
GPUS=(1 2 3 1 2 3 1 2 3)

mkdir -p "$RUN_DIR/manifests_rebalance" "$RUN_DIR/logs"

# stop the three original place_apple workers (12, 13, 14)
for w in 12 13 14; do
  for p in $(pgrep -f "worker-id $w " 2>/dev/null); do kill "$p" 2>/dev/null; done
done
sleep 5
echo "stopped old place_apple workers"

# split the 100 seeds into $NEW shards
/home/leju-suzhou/miniconda3/envs/pi05/bin/python - "$RUN_DIR" "$TASK" "$NEW" <<'PY'
import json, sys
run, task, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
arms = ["vanilla", "l6_matched_025", "l6_matched_050", "l6_matched_075", "l6_matched_100"]
seeds = list(range(100))
per = (len(seeds) + n - 1) // n
for i in range(n):
    part = seeds[i*per:(i+1)*per]
    if not part:
        continue
    jobs = [{"task": task, "seed": s, "arms": arms} for s in part]
    json.dump({"worker_id": 100 + i, "jobs": jobs},
              open(f"{run}/manifests_rebalance/worker_{100+i}.json", "w"), indent=2)
    print(f"  shard {100+i}: seeds {part[0]}-{part[-1]} ({len(part)} seeds)")
PY

# launch them
i=0
for f in "$RUN_DIR"/manifests_rebalance/worker_*.json; do
  wid=$(basename "$f" .json | sed 's/worker_//')
  gpu="${GPUS[$((i % 3))]}"
  port=$([ "$gpu" = "1" ] && echo 8101 || ([ "$gpu" = "2" ] && echo 8102 || echo 8103))
  log="$RUN_DIR/logs/rebalance_worker_${wid}_gpu${gpu}.log"
  echo "starting rebalance worker $wid on gpu $gpu port $port -> $log"
  (
    source "$CONDA_ROOT/etc/profile.d/conda.sh"; conda activate pi0
    export CUDA_VISIBLE_DEVICES="$gpu"
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
    export TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu
    export PYTHONPATH="/home/leju-suzhou/.cache/pi0fast_client_deps:$ROOT/openpi/packages/openpi-client/src:$ROOT/pcd_openvla_simpler_box_31b027e/source:$ROOT/SimplerEnv:$ROOT/SpatialVLA-SimplerEnv:$ROOT/SpatialVLA-SimplerEnv/ManiSkill2_real2sim:$ROOT/SpatialVLA/ManiSkill2_real2sim:$TOKEN_CD:$ROOT/openpi/src"
    exec python -u "$TOKEN_CD/research/semantic_token_cd/pi0_fast_simpler_l6_matched_rollout.py" \
      --gpu "$gpu" --worker-id "$wid" --shard "$f" \
      --host 127.0.0.1 --port "$port" \
      --out "$RUN_DIR" --checkpoint "$ROOT/checkpoints/openpi_fast_finetuned/fractal" --rt-depth 8
  ) > "$log" 2>&1 &
  i=$((i+1))
  sleep 2
done
echo "launched $i rebalance workers"
