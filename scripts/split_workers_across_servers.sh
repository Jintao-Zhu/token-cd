#!/usr/bin/env bash
# Point half the workers at the second server per GPU.
#
# Bug being fixed: the previous launcher computed `port = 8100 + gpu`, so all 12 workers
# connected to the "a" server on each card. The three "b" servers (8111/8112/8113) were idle
# at 0.7% CPU while the "a" servers sat at 85-87%, halving throughput to ~15 episodes/hour.
# No processes are added: the same 12 workers are simply split across the two servers per GPU.
set -uo pipefail
ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD=$ROOT/token-cd
CONDA_ROOT=/home/leju-suzhou/miniconda3
RUN_DIR=$TOKEN_CD/runs/pi0fast_l6_matched_9task_100seed
ROLLOUT=$TOKEN_CD/research/semantic_token_cd/pi0_fast_simpler_l6_matched_rollout.py
SHARD_DIR=$RUN_DIR/manifests_apple
LOG=$RUN_DIR/logs/split_servers.log
exec > "$LOG" 2>&1

echo "=== $(date '+%F %T') teardown workers (servers untouched) ==="
pkill -f "pi0_fast_simpler_l6_matched_rollout" 2>/dev/null
sleep 8
n=$(pgrep -cf "pi0_fast_simpler_l6_matched_rollout")
echo "  workers remaining: $n"
if [ "$n" -gt 0 ]; then pkill -9 -f "pi0_fast_simpler_l6_matched_rollout" 2>/dev/null; sleep 4; fi
echo "  servers still up: $(pgrep -cf 'pi0_fast_l6_simpler_server')"

echo "=== relaunching, alternating between the two servers per GPU ==="
# (gpu, port) pairs: alternate a/b so each server gets roughly half the workers
PAIRS=("3 8103" "3 8113" "3 8103" "3 8113" "2 8102" "2 8112" "2 8102" "2 8112" \
       "1 8101" "1 8111" "1 8101" "1 8111")
i=0
for f in $(ls "$SHARD_DIR"/worker_*.json | sort); do
  wid=$(basename "$f" .json | sed 's/worker_//')
  set -- ${PAIRS[$i]}
  gpu=$1; port=$2
  wlog="$RUN_DIR/logs/apple_worker_${wid}_gpu${gpu}_p${port}.log"
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
echo "=== launched $i workers ==="
sleep 5
"$CONDA_ROOT/envs/pi05/bin/python" - <<'PY'
import subprocess, collections, os
out = subprocess.run(["ps","-eo","pid,cmd"], capture_output=True, text=True).stdout
sh = collections.Counter(); po = collections.Counter()
for ln in out.splitlines():
    if "pi0_fast_simpler_l6_matched_rollout" not in ln or "grep" in ln:
        continue
    parts = ln.split()
    s = next((os.path.basename(p) for p in parts if p.endswith(".json") and "worker_" in p), "?")
    p = next((parts[i+1] for i,x in enumerate(parts) if x == "--port"), "?")
    sh[s] += 1; po[p] += 1
print(f"  distinct shards: {len(sh)}  duplicates: {[k for k,v in sh.items() if v>1]}")
print(f"  workers per server: {dict(sorted(po.items()))}")
PY
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 1,2,3
