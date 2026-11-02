#!/usr/bin/env bash
# Re-split the place_apple tail across all 12 slots.
#
# place_apple is the only real work left (205 of 213 remaining episodes) and its per-episode
# cost is high: 200 steps = 50 replans, about 15 minutes each. Three of its nine shards have
# finished and their workers exited, so six workers carry the tail at ~22 episodes/hour
# (9.7 h). Re-splitting all 100 seeds across 12 slots brings that to ~4 h.
#
# The resume logic skips any (task, seed, arm) whose result file already exists, so
# over-splitting costs nothing but startup time.
#
# Order matters: tear down, VERIFY zero, then launch - the earlier duplicate-worker incident
# came from two launchers running concurrently.
set -uo pipefail
ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD=$ROOT/token-cd
CONDA_ROOT=/home/leju-suzhou/miniconda3
RUN_DIR=$TOKEN_CD/runs/pi0fast_l6_matched_9task_100seed
ROLLOUT=$TOKEN_CD/research/semantic_token_cd/pi0_fast_simpler_l6_matched_rollout.py
SHARD_DIR=$RUN_DIR/manifests_apple
LOG=$RUN_DIR/logs/resplit_apple.log
exec > "$LOG" 2>&1

echo "=== $(date '+%F %T') teardown workers (servers untouched) ==="
pkill -f "pi0_fast_simpler_l6_matched_rollout" 2>/dev/null
sleep 8
n=$(pgrep -cf "pi0_fast_simpler_l6_matched_rollout")
echo "  workers remaining: $n"
if [ "$n" -gt 0 ]; then
  pkill -9 -f "pi0_fast_simpler_l6_matched_rollout" 2>/dev/null
  sleep 4
  echo "  after SIGKILL: $(pgrep -cf 'pi0_fast_simpler_l6_matched_rollout')"
fi
echo "  servers still up: $(pgrep -cf 'pi0_fast_l6_simpler_server')"

mkdir -p "$SHARD_DIR"
rm -f "$SHARD_DIR"/worker_*.json

echo "=== generating 12 shards over all place_apple seeds ==="
"$CONDA_ROOT/envs/pi05/bin/python" - "$SHARD_DIR" <<'PY'
import json, sys
d = sys.argv[1]
arms = ["vanilla", "l6_matched_025", "l6_matched_050", "l6_matched_075",
        "l6_matched_100", "l6_matched_125", "l6_matched_150"]
task = "google_robot_place_apple_in_closed_top_drawer"
seeds = list(range(100))
n = 12
per = (len(seeds) + n - 1) // n
for i in range(n):
    part = seeds[i*per:(i+1)*per]
    if not part:
        continue
    jobs = [{"task": task, "seed": s, "arms": arms} for s in part]
    json.dump({"worker_id": 500 + i, "jobs": jobs},
              open(f"{d}/worker_{500+i}.json", "w"), indent=2)
    print(f"  shard {500+i}: seeds {part[0]:2d}-{part[-1]:2d} ({len(part)} seeds, {len(part)*len(arms)} eps)")
PY

echo "=== launching 12 workers ==="
GPU_SEQ=(3 3 3 3 2 2 2 2 1 1 1 1)
i=0
for f in $(ls "$SHARD_DIR"/worker_*.json | sort); do
  wid=$(basename "$f" .json | sed 's/worker_//')
  gpu="${GPU_SEQ[$i]}"
  port=$((8100 + gpu))
  wlog="$RUN_DIR/logs/apple_worker_${wid}_gpu${gpu}.log"
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
echo "=== $(date '+%F %T') launched $i workers; verifying no duplicates ==="
sleep 5
"$CONDA_ROOT/envs/pi05/bin/python" - <<'PY'
import subprocess, collections, os
out = subprocess.run(["ps","-eo","pid,cmd"], capture_output=True, text=True).stdout
c = collections.Counter()
for ln in out.splitlines():
    if "pi0_fast_simpler_l6_matched_rollout" not in ln or "grep" in ln:
        continue
    s = next((os.path.basename(p) for p in ln.split() if p.endswith(".json") and "worker_" in p), "?")
    c[s] += 1
print(f"  distinct shards: {len(c)}  duplicates: {[k for k,v in c.items() if v>1]}")
PY
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 1,2,3
