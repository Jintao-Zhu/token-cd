#!/usr/bin/env bash
# pi0-FAST WidowX: (re)start the rollout side with NSHARD workers spread over GPU 1/2/3,
# NSHARD/3 workers per existing policy server.  The 6 servers (pf_wx_srv_*) are never touched.
#
# Shards come from manifests$NSHARD/ (scripts/resplit_pi0fast_widowx_18way.py --nshard N --apply),
# which cover ONLY the episodes still missing at resplit time; workers skip any episode whose
# JSON already says status == "complete", so nothing is lost or duplicated at any NSHARD.
#
# GPU policy: only GPU 1/2/3 (see docs/AGENT_CONSTRAINTS.md). Never GPU 0, never 4-7.
#
# Usage:
#   NSHARD=18 bash launch_pi0fast_widowx_w18.sh --kill-old        # default
#   NSHARD=12 bash launch_pi0fast_widowx_w18.sh --kill-old        # revert to 4 workers/GPU
#   KILL_PREFIXES="pf_wx_w pf_wx18_w" ...                         # sessions to stop (default below)
set -uo pipefail

NSHARD=${NSHARD:-18}
PER_GPU=$((NSHARD / 3))
PREFIX=${PREFIX:-pf_wx${NSHARD}_w}
SHARDDIR=${SHARDDIR:-manifests${NSHARD}}
LOGDIRNAME=${LOGDIRNAME:-w${NSHARD}}
KILL_OLD=0
[[ "${1:-}" == "--kill-old" ]] && KILL_OLD=1
KILL_PREFIXES=${KILL_PREFIXES:-"pf_wx_w pf_wx12_w pf_wx18_w"}

ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD=$ROOT/token-cd
CONDA_ROOT=/home/leju-suzhou/miniconda3
OVERLAY=$ROOT/pi05_sklearn_overlay
RUN=$TOKEN_CD/runs/pi0fast_bridge_widowx_4task_100seed
CKPT=$ROOT/checkpoints/openpi_fast_finetuned/bridge
ROLLOUT=$TOKEN_CD/research/semantic_token_cd/pi0_fast_simpler_l6_matched_rollout.py
PYBIN=$CONDA_ROOT/envs/pi05/bin/python
LOGDIR=$RUN/logs/$LOGDIRNAME

mkdir -p "$LOGDIR"

if [[ ! -f "$RUN/$SHARDDIR/worker_0.json" ]]; then
  echo "ERROR: $SHARDDIR/ missing -- run scripts/resplit_pi0fast_widowx_18way.py --nshard $NSHARD --apply first" >&2
  exit 2
fi

# ---------------------------------------------------------------- stop the previous workers ----
if [[ "$KILL_OLD" == 1 ]]; then
  for pat in $KILL_PREFIXES; do
    while read -r sess; do
      [[ -z "$sess" ]] && continue
      [[ "$sess" == "$PREFIX"* ]] && continue          # never kill the prefix we are about to use
      tmux kill-session -t "$sess" 2>/dev/null && echo "killed $sess"
    done < <(tmux ls 2>/dev/null | cut -d: -f1 | grep "^${pat}")
  done
  sleep 3
fi

# ---------------------------------------------------------------- start NSHARD workers --------
for i in $(seq 0 $((NSHARD - 1))); do
  gpu=$(( i / PER_GPU + 1 ))
  if (( (i % PER_GPU) % 2 == 0 )); then port=$((8100 + gpu)); else port=$((8110 + gpu)); fi
  sess="${PREFIX}$(printf '%02d' "$i")"
  log="$LOGDIR/worker_$(printf '%02d' "$i")_gpu${gpu}_p${port}.log"
  if tmux has-session -t "$sess" 2>/dev/null; then echo "worker $sess already up"; continue; fi
  tmux new-session -d -s "$sess" "bash -lc 'source $CONDA_ROOT/etc/profile.d/conda.sh; conda activate pi0; export CUDA_VISIBLE_DEVICES=$gpu; export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json; export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; export JAX_PLATFORMS=cpu; export PI0FAST_WIDOWX_LAYOUT=bridge; export PYTHONPATH=/home/leju-suzhou/.cache/pi0fast_client_deps:$ROOT/openpi/packages/openpi-client/src:$ROOT/pcd_openvla_simpler_box_31b027e/source:$ROOT/SimplerEnv:$ROOT/SpatialVLA-SimplerEnv:$ROOT/SpatialVLA-SimplerEnv/ManiSkill2_real2sim:$ROOT/SpatialVLA/ManiSkill2_real2sim:$TOKEN_CD:$ROOT/openpi/src; python -u $ROLLOUT --gpu $gpu --worker-id $i --shard $RUN/$SHARDDIR/worker_$i.json --host 127.0.0.1 --port $port --out $RUN --checkpoint $CKPT --rt-depth 8 >> $log 2>&1'"
  echo "started $sess (gpu$gpu port$port shard$i)"
  sleep 1
done

echo "--- waiting for $NSHARD workers to report environment_ready ---"
ready=0
for i in $(seq 1 60); do
  ready=$(grep -l '"event": "environment_ready"' "$LOGDIR"/worker_*.log 2>/dev/null | wc -l)
  [[ "$ready" -ge "$NSHARD" ]] && break
  sleep 10
done
echo "workers with environment_ready: $ready/$NSHARD"
"$PYBIN" - "$RUN" <<'PY'
import glob, json, sys
from pathlib import Path
run = Path(sys.argv[1])
files = glob.glob(str(run / 'episodes' / '*' / '*.json'))
comp = err = 0
for f in files:
    try:
        s = json.load(open(f)).get('status')
    except Exception:
        continue
    comp += s == 'complete'
    err += s == 'error'
proto = json.loads((run / 'protocol.json').read_text())
print(f"episodes on disk: {len(files)} (complete {comp}, error {err}) / {proto['episodes']}  "
      f"workers={proto.get('workers')}")
PY
