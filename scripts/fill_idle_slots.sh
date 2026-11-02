#!/usr/bin/env bash
# Fill the idle worker slots with the remaining work.
#
# The run is at 3096/3500 but only 5 of the 15 worker slots are still occupied: three tasks
# finished completely and their workers exited, leaving place_apple (375/700) and open_drawer
# (621/700) as the tail at ~23 episodes/hour. This does NOT add processes: the same worker
# slots are re-pointed at the unfinished tasks, and the resume logic skips every
# (task, seed, arm) whose result file is already complete, so nothing is recomputed.
#
# Memory is not a concern (35-45 GB per card of 81.6 GB).
set -uo pipefail
ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD=$ROOT/token-cd
CONDA_ROOT=/home/leju-suzhou/miniconda3
RUN_DIR=$TOKEN_CD/runs/pi0fast_l6_matched_9task_100seed
ROLLOUT=$TOKEN_CD/research/semantic_token_cd/pi0_fast_simpler_l6_matched_rollout.py
SHARD_DIR=$RUN_DIR/manifests_tail

mkdir -p "$SHARD_DIR" "$RUN_DIR/logs"
rm -f "$SHARD_DIR"/worker_*.json

# place_apple gets 9 slots (it is 4x slower per episode), open_drawer 3
/home/leju-suzhou/miniconda3/envs/pi05/bin/python - "$SHARD_DIR" <<'PY'
import json, sys
d = sys.argv[1]
arms = ["vanilla", "l6_matched_025", "l6_matched_050", "l6_matched_075",
        "l6_matched_100", "l6_matched_125", "l6_matched_150"]
plan = [("google_robot_place_apple_in_closed_top_drawer", 9),
        ("google_robot_open_drawer", 3)]
seeds = list(range(100))
i = 0
for task, k in plan:
    per = (len(seeds) + k - 1) // k
    for c in range(k):
        part = seeds[c*per:(c+1)*per]
        if not part:
            continue
        jobs = [{"task": task, "seed": s, "arms": arms} for s in part]
        json.dump({"worker_id": 400 + i, "jobs": jobs},
                  open(f"{d}/worker_{400+i}.json", "w"), indent=2)
        print(f"  shard {400+i}: {task.replace('google_robot_',''):34s} seeds {part[0]:2d}-{part[-1]:2d}")
        i += 1
print(f"  {i} shards")
PY

echo "=== stopping the 5 remaining workers (servers untouched) ==="
pkill -f "[p]i0_fast_simpler_l6_matched_rollout" 2>/dev/null
sleep 6
echo "  workers now: $(pgrep -cf '[p]i0_fast_simpler_l6_matched_rollout')  servers: $(pgrep -cf '[p]i0_fast_l6_simpler_server')"

echo "=== relaunching 12 slots on the tail ==="
GPU_SEQ=(3 3 3 3 2 2 2 2 1 1 1 1)
PORTS=(8101 8102 8103)
i=0
for f in $(ls "$SHARD_DIR"/worker_*.json | sort); do
  wid=$(basename "$f" .json | sed 's/worker_//')
  gpu="${GPU_SEQ[$i]}"
  port="${PORTS[$((gpu-1))]}"
  log="$RUN_DIR/logs/tail_worker_${wid}_gpu${gpu}.log"
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
  ) > "$log" 2>&1 &
  echo "  worker $wid -> gpu $gpu port $port"
  i=$((i+1))
  sleep 1
done
echo "=== launched $i workers ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 1,2,3
