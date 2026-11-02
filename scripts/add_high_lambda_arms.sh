#!/usr/bin/env bash
# Launch workers for the two new guidance arms (lambda 1.25, 1.50).
#
# 5 tasks x 100 seeds x 2 arms = 1000 episodes. Twelve shards, assigned to GPUs in
# proportion to free memory (3 / 4 / 5), because the third server per GPU is already running
# and each card has only 13-20 GB left.
#
# The policy servers (now 9, one per GPU per slot) are the throughput limit; the workers just
# need to be numerous enough to keep them fed.
set -uo pipefail
ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD=$ROOT/token-cd
CONDA_ROOT=/home/leju-suzhou/miniconda3
RUN_DIR=$TOKEN_CD/runs/pi0fast_l6_matched_9task_100seed
SHARD_DIR=$RUN_DIR/manifests_lambda_high
PORTS=(8121 8122 8123)      # third server per GPU

mkdir -p "$SHARD_DIR" "$RUN_DIR/logs"
rm -f "$SHARD_DIR"/worker_*.json

/home/leju-suzhou/miniconda3/envs/pi05/bin/python - "$SHARD_DIR" <<'PY'
import json, sys
d = sys.argv[1]
arms = ["l6_matched_125", "l6_matched_150"]
# (task, number of shards) chosen so the total is 12 and the long place_apple task gets 3
plan = [("google_robot_open_drawer", 3), ("google_robot_close_drawer", 2),
        ("google_robot_pick_coke_can", 2), ("google_robot_move_near", 2),
        ("google_robot_place_apple_in_closed_top_drawer", 3)]
seeds = list(range(100))
i = 0
tot = 0
for task, k in plan:
    per = (len(seeds) + k - 1) // k
    for c in range(k):
        part = seeds[c*per:(c+1)*per]
        if not part:
            continue
        jobs = [{"task": task, "seed": s, "arms": arms} for s in part]
        json.dump({"worker_id": 300 + i, "jobs": jobs},
                  open(f"{d}/worker_{300+i}.json", "w"), indent=2)
        print(f"  shard {300+i}: {task.replace('google_robot_',''):34s} seeds {part[0]:2d}-{part[-1]:2d} "
              f"-> {len(part)*len(arms)} episodes")
        tot += len(part)*len(arms)
        i += 1
print(f"  total {i} shards, {tot} episodes")
PY

# GPU assignment by free memory: GPU3 (most free) gets 5, GPU2 4, GPU1 3
GPU_SEQ=(3 3 3 3 3 2 2 2 2 1 1 1)
i=0
for f in $(ls "$SHARD_DIR"/worker_*.json | sort); do
  wid=$(basename "$f" .json | sed 's/worker_//')
  gpu="${GPU_SEQ[$i]}"
  port="${PORTS[$((gpu-1))]}"
  log="$RUN_DIR/logs/highlambda_worker_${wid}_gpu${gpu}.log"
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
  echo "worker $wid -> gpu $gpu port $port"
  i=$((i+1))
  sleep 2
done
echo "launched $i workers"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 1,2,3
