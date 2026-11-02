#!/usr/bin/env bash
# pi0-FAST WidowX full experiment: 4 tasks x 100 seeds x 7 arms = 2800 episodes.
#
# Layout (all on GPU 1/2/3 -- never GPU 0, never 4-7; see docs/AGENT_CONSTRAINTS.md):
#   6 policy servers (2 per GPU, bridge checkpoint, --norm-stats auto):
#       GPU1 -> 8101/8111, GPU2 -> 8102/8112, GPU3 -> 8103/8113
#   12 rollout workers (4 per GPU, 2 workers per server), disjoint seed shards covering 0-99
#
# Input convention: the checkpoint's TRAINED Bridge layout
#   state = [x, y, z, roll, pitch, yaw, PAD(0), gripper]   (PI0FAST_WIDOWX_LAYOUT=bridge)
#   actions clipped to the training q01/q99 (default when the bridge layout is active)
# Renderer: SAPIEN RT full, path depth 8, spp 32, denoiser on (same as the google_robot run).
#
# Usage: bash launch_pi0fast_widowx_2800.sh [servers|workers|all]
set -uo pipefail
MODE="${1:-all}"

ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD=$ROOT/token-cd
CONDA_ROOT=/home/leju-suzhou/miniconda3
OVERLAY=$ROOT/pi05_sklearn_overlay
RUN=$TOKEN_CD/runs/pi0fast_bridge_widowx_4task_100seed
CKPT=$ROOT/checkpoints/openpi_fast_finetuned/bridge
SERVER=$TOKEN_CD/research/semantic_token_cd/pi0_fast_l6_simpler_server.py
ROLLOUT=$TOKEN_CD/research/semantic_token_cd/pi0_fast_simpler_l6_matched_rollout.py
PYBIN=$CONDA_ROOT/envs/pi05/bin/python
PILOT=$TOKEN_CD/runs/widowx_bridge_layout2/episodes   # 16 validated episodes (bridge layout, depth8)
TASKS=(widowx_carrot_on_plate widowx_put_eggplant_in_basket widowx_spoon_on_towel widowx_stack_cube)
ARMS=(vanilla l6_matched_025 l6_matched_050 l6_matched_075 l6_matched_100 l6_matched_125 l6_matched_150)
NSHARD=12

mkdir -p "$RUN/logs" "$RUN/manifests" "$RUN/episodes"

# ---------------------------------------------------------------- manifests + protocol ---------
if [[ ! -f "$RUN/manifests/worker_0.json" ]]; then
  "$PYBIN" - "$RUN" "$NSHARD" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1]); n = int(sys.argv[2])
tasks = ["widowx_carrot_on_plate", "widowx_put_eggplant_in_basket", "widowx_spoon_on_towel", "widowx_stack_cube"]
arms = ["vanilla", "l6_matched_025", "l6_matched_050", "l6_matched_075", "l6_matched_100", "l6_matched_125", "l6_matched_150"]
seeds = list(range(100))
base, extra = divmod(len(seeds), n)
chunks, i = [], 0
for k in range(n):
    size = base + (1 if k < extra else 0)
    chunks.append(seeds[i:i + size]); i += size
tot = 0
for wid, part in enumerate(chunks):
    jobs = [{"task": t, "seed": s, "arms": list(arms)} for t in tasks for s in part]
    (run / "manifests" / f"worker_{wid}.json").write_text(json.dumps({"worker_id": wid, "jobs": jobs}, indent=2))
    tot += len(jobs) * len(arms)
protocol = {
    "experiment": "pi0fast_bridge_widowx_4task_100seed",
    "protocol_id": "PI0FAST_L6MATCHED_WIDOWX_BRIDGE_RTFULL_4TASK_0_99_V1",
    "checkpoint": "/home/leju-suzhou/zjt_ws/checkpoints/openpi_fast_finetuned/bridge",
    "norm_stats": "checkpoint's own assets/local/bridge_lerobot/norm_stats.json",
    "tasks": tasks, "seeds": [0, 99], "arms": arms,
    "episodes": tot, "shards": [len(c) for c in chunks],
    "widowx_state_layout": "[x,y,z,roll,pitch,yaw,PAD(0),gripper] with R_topdown conjugation (PI0FAST_WIDOWX_LAYOUT=bridge)",
    "action_postprocessing": "clip dims 0-5 to training q01/q99; gripper binarised to {-1,+1}",
    "renderer": {"shader_dir": "rt", "rt_max_path_depth": 8, "rt_samples_per_pixel": 32, "rt_use_denoiser": True},
    "servers": "6 (2 per GPU on GPU 1/2/3), ports 8101/8111, 8102/8112, 8103/8113",
    "workers": "12 (4 per GPU, 2 per server)",
}
(run / "protocol.json").write_text(json.dumps(protocol, indent=2))
print(f"manifests: {n} shards, {tot} episodes")
PY
fi

# ---------------------------------------------------------------- reuse the 16 pilot episodes --
if [[ -d "$PILOT" ]]; then
  reused=$("$PYBIN" - "$PILOT" "$RUN/episodes" <<'PY'
import os, shutil, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
n = 0
for f in src.glob('*/*.json'):
    tgt = dst / f.parent.name / f.name
    tgt.parent.mkdir(parents=True, exist_ok=True)
    if tgt.exists(): continue
    try: os.link(f, tgt)
    except OSError: shutil.copy2(f, tgt)
    n += 1
print(n)
PY
)
  echo "reused pilot episodes: $reused"
fi

# ---------------------------------------------------------------- servers -----------------------
start_servers() {
  for gpu in 1 2 3; do
    for slot in 0 1; do
      if [[ "$slot" == 0 ]]; then port=$((8100 + gpu)); tag=a; else port=$((8110 + gpu)); tag=b; fi
      sess="pf_wx_srv_g${gpu}${tag}"
      log="$RUN/logs/server_g${gpu}_p${port}.log"
      if tmux has-session -t "$sess" 2>/dev/null; then echo "server $sess already up"; continue; fi
      tmux new-session -d -s "$sess" "bash -lc 'source $CONDA_ROOT/etc/profile.d/conda.sh; conda activate pi05; export CUDA_VISIBLE_DEVICES=$gpu; export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.18; export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1; export OPENPI_DATA_HOME=$ROOT/checkpoints/openpi-assets HF_HUB_OFFLINE=1; export PI0FAST_CD_KMEANS=sklearn PI0FAST_CD_TAP_MODE=separate PI0FAST_CD_DECODE=verified; export PYTHONPATH=$OVERLAY:$ROOT/openpi/src:$ROOT/openpi/packages/openpi-client/src:$TOKEN_CD; python -u $SERVER --gpu $gpu --port $port --checkpoint $CKPT --norm-stats auto >> $log 2>&1'"
      echo "started $sess (gpu$gpu port$port)"
      sleep 2
    done
  done
  echo "waiting for 6 servers to finish warmup..."
  for gpu in 1 2 3; do
    for port in $((8100 + gpu)) $((8110 + gpu)); do
      log="$RUN/logs/server_g${gpu}_p${port}.log"
      ok=0
      for i in $(seq 1 90); do
        grep -q "warmups passed" "$log" 2>/dev/null && { ok=1; break; }
        sleep 5
      done
      if [[ "$ok" == 1 ]]; then
        echo "  port$port READY: $(grep -m1 'with norm stats' "$log" | tail -c 60)"
      else
        echo "  port$port FAILED to warm up"; tail -5 "$log"
      fi
    done
  done
}

# ---------------------------------------------------------------- workers -----------------------
start_workers() {
  for i in $(seq 0 $((NSHARD - 1))); do
    gpu=$(( i / 4 + 1 ))
    if (( (i % 4) % 2 == 0 )); then port=$((8100 + gpu)); else port=$((8110 + gpu)); fi
    sess="pf_wx_w$(printf '%02d' "$i")"
    log="$RUN/logs/worker_$(printf '%02d' "$i")_gpu${gpu}_p${port}.log"
    if tmux has-session -t "$sess" 2>/dev/null; then echo "worker $sess already up"; continue; fi
    tmux new-session -d -s "$sess" "bash -lc 'source $CONDA_ROOT/etc/profile.d/conda.sh; conda activate pi0; export CUDA_VISIBLE_DEVICES=$gpu; export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json; export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; export JAX_PLATFORMS=cpu; export PI0FAST_WIDOWX_LAYOUT=bridge; export PYTHONPATH=/home/leju-suzhou/.cache/pi0fast_client_deps:$ROOT/openpi/packages/openpi-client/src:$ROOT/pcd_openvla_simpler_box_31b027e/source:$ROOT/SimplerEnv:$ROOT/SpatialVLA-SimplerEnv:$ROOT/SpatialVLA-SimplerEnv/ManiSkill2_real2sim:$ROOT/SpatialVLA/ManiSkill2_real2sim:$TOKEN_CD:$ROOT/openpi/src; python -u $ROLLOUT --gpu $gpu --worker-id $i --shard $RUN/manifests/worker_$i.json --host 127.0.0.1 --port $port --out $RUN --checkpoint $CKPT --rt-depth 8 >> $log 2>&1'"
    echo "started $sess (gpu$gpu port$port shard$i)"
    sleep 2
  done
}

case "$MODE" in
  servers) start_servers ;;
  workers) start_workers ;;
  all) start_servers; start_workers ;;
  *) echo "usage: $0 [servers|workers|all]" >&2; exit 2 ;;
esac

echo "--- verification ---"
"$PYBIN" - "$RUN" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
n = len(list((run / 'episodes').glob('*/*.json')))
proto = json.loads((run / 'protocol.json').read_text())
print(f"episodes on disk: {n}/{proto['episodes']}")
PY
