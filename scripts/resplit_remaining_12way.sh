#!/usr/bin/env bash
# Balanced re-split of ALL remaining episodes across 12 worker slots (2 per policy server).
#
# Why: the current 5 workers own contiguous seed ranges, so the slowest shard (worker 503,
# 37 episodes) is the critical path at ~4 eps/h -> ~9 h. Servers are idle (GPU util mostly 0%,
# load 26/256), so re-balancing the *remaining* work evenly and filling all six server slots
# with two workers each collapses the tail.
#
# Safety rules taken from the earlier duplicate-worker incident (see resplit_apple_tail.sh):
#   * teardown FIRST, verify zero workers, THEN launch;
#   * servers are never touched (they stay warm/compiled);
#   * exactly one supervisor script, one worker per generated shard.
# Resume logic in the rollout script skips any (task, seed, arm) whose result file already has
# status=complete, so killing mid-episode only costs the in-flight episodes (recomputed below).
set -uo pipefail
ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD=$ROOT/token-cd
CONDA_ROOT=/home/leju-suzhou/miniconda3
RUN_DIR=$TOKEN_CD/runs/pi0fast_l6_matched_9task_100seed
ROLLOUT=$TOKEN_CD/research/semantic_token_cd/pi0_fast_simpler_l6_matched_rollout.py
SHARD_DIR=$RUN_DIR/manifests_final
LOG=$RUN_DIR/logs/resplit_remaining.log
PY="$CONDA_ROOT/envs/pi05/bin/python"
mkdir -p "$SHARD_DIR"
exec > "$LOG" 2>&1
echo "=== $(date '+%F %T') teardown workers (servers untouched) ==="
pkill -f "pi0_fast_simpler_l6_matched_rollout" 2>/dev/null
sleep 8
n=$(pgrep -cf "pi0_fast_simpler_l6_matched_rollout" || true)
echo "  workers remaining: $n"
if [ "$n" -gt 0 ]; then
  pkill -9 -f "pi0_fast_simpler_l6_matched_rollout" 2>/dev/null
  sleep 4
  echo "  after SIGKILL: $(pgrep -cf 'pi0_fast_simpler_l6_matched_rollout' || true)"
fi
srv=$(pgrep -cf "pi0_fast_l6_simpler_server" || true)
echo "  servers still up: $srv"
if [ "$srv" -lt 6 ]; then
  echo "  ABORT: expected 6 servers, found $srv - not launching."
  exit 1
fi

echo "=== $(date '+%F %T') generating balanced shards from live remaining work ==="
rm -f "$SHARD_DIR"/worker_*.json
"$PY" - "$SHARD_DIR" <<'PY'
import glob, itertools, json, os, sys

shard_dir = sys.argv[1]
RUN = "/home/leju-suzhou/zjt_ws/token-cd/runs/pi0fast_l6_matched_9task_100seed"
ARMS = ["vanilla", "l6_matched_025", "l6_matched_050", "l6_matched_075",
        "l6_matched_100", "l6_matched_125", "l6_matched_150"]
TASKS = ["google_robot_place_apple_in_closed_top_drawer",
         "google_robot_open_drawer"]
# matched episode ~620 s, vanilla ~140 s -> weights for LPT balancing
W = {a: (0.25 if a == "vanilla" else 1.0) for a in ARMS}

done = set()
broken = []
for f in glob.glob(f"{RUN}/episodes/*/*.json"):
    try:
        d = json.load(open(f))
    except Exception:
        broken.append(f); continue
    if d.get("status") == "complete":
        done.add((d.get("task"), d.get("seed"), d.get("arm")))

items = []
for t in TASKS:
    for s in range(100):
        for a in ARMS:
            if (t, s, a) not in done:
                items.append((t, s, a))
if broken:
    print(f"  WARNING: {len(broken)} unreadable episode files: {broken[:5]}")

print(f"  remaining episodes: {len(items)}  "
      f"(matched={sum(1 for _,_,a in items if a!='vanilla')}, "
      f"vanilla={sum(1 for _,_,a in items if a=='vanilla')})")
if not items:
    print("  nothing to do")
    sys.exit(0)

N = 12
# LPT: heaviest first into the currently lightest bin
order = sorted(items, key=lambda x: (W[x[2]], x[1], x[0]), reverse=True)
bins = [[] for _ in range(N)]
load = [0.0] * N
for it in order:
    i = load.index(min(load))
    bins[i].append(it)
    load[i] += W[it[2]]

# bin 0..5 -> first pass over the six slots, bin 6..11 -> second pass (2 workers / slot)
for i, b in enumerate(bins):
    b.sort(key=lambda x: (x[0], x[1]))          # group same task/seed -> one env build
    jobs = []
    for (t, s), grp in itertools.groupby(b, key=lambda x: (x[0], x[1])):
        jobs.append({"task": t, "seed": s, "arms": [a for _, _, a in grp]})
    wid = 600 + i
    json.dump({"worker_id": wid, "jobs": jobs},
              open(f"{shard_dir}/worker_{wid}.json", "w"), indent=2)
    print(f"  shard {wid}: {len(b):3d} eps  weight={load[i]:5.2f}  "
          f"tasks={sorted({x[0].split('_')[-1] for x in b})}  seeds={len({x[1] for x in b})}")
print(f"  total weight={sum(load):.2f}  max bin={max(load):.2f}  min bin={min(load):.2f}")
PY
[ -f "$SHARD_DIR"/worker_600.json ] || { echo "  ABORT: shard generation failed"; exit 1; }

echo "=== $(date '+%F %T') launching 12 workers (2 per server slot) ==="
# slot order spreads heavy bins across GPUs: pass 1 covers all six servers, pass 2 doubles up
SLOTS=( "1:8101" "1:8111" "2:8102" "2:8112" "3:8103" "3:8113"
        "1:8101" "1:8111" "2:8102" "2:8112" "3:8103" "3:8113" )
i=0
for f in $(ls "$SHARD_DIR"/worker_*.json | sort); do
  wid=$(basename "$f" .json | sed 's/worker_//')
  gpu="${SLOTS[$i]%%:*}"; port="${SLOTS[$i]##*:}"
  wlog="$RUN_DIR/logs/final_worker_${wid}_gpu${gpu}_p${port}.log"
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
echo "=== $(date '+%F %T') launched $i workers; verifying ==="
sleep 10
"$PY" - <<'PY'
import collections, os, subprocess
out = subprocess.run(["ps", "-eo", "pid,cmd"], capture_output=True, text=True).stdout
c = collections.Counter()
for ln in out.splitlines():
    if "pi0_fast_simpler_l6_matched_rollout" not in ln or "grep" in ln:
        continue
    s = next((os.path.basename(p) for p in ln.split() if p.endswith(".json") and "worker_" in p), "?")
    c[s] += 1
print(f"  distinct shards: {len(c)}  total workers: {sum(c.values())}")
print(f"  duplicates: {[k for k, v in c.items() if v > 1]}")
PY
echo "  servers: $(pgrep -cf 'pi0_fast_l6_simpler_server' || true)"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i 1,2,3
echo "=== $(date '+%F %T') done ==="
