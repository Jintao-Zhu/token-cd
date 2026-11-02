#!/usr/bin/env bash
# Re-test the pi0-FAST Bridge checkpoint with ITS OWN norm stats.
#
# Why: every previous WidowX attempt served this checkpoint through
# pi0_fast_l6_simpler_server.py, which used the default FRACTAL norm stats
# (pi0_fast_matched_cd.FRACTAL_STATS) for a policy trained on bridge_orig_lerobot.
# That mis-normalisation is a concrete candidate cause of the 0-interaction results
# (moved_wrong_obj=False on 100% of 163 episodes).
#
# This run: server on GPU 1, port 8121, --norm-stats auto (auto-discovers
# <checkpoint>/assets/local/*/norm_stats.json -> bridge_lerobot), 2 workers, vanilla arm,
# 4 WidowX tasks x 8 seeds = 32 episodes, RT full depth8.
#
# Usage: bash run_widowx_bridge_normfix.sh [start|verify]
set -uo pipefail

ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD=$ROOT/token-cd
CONDA_ROOT=/home/leju-suzhou/miniconda3
OVERLAY=$ROOT/pi05_sklearn_overlay
CKPT=$ROOT/checkpoints/openpi_fast_finetuned/bridge
SERVER=$TOKEN_CD/research/semantic_token_cd/pi0_fast_l6_simpler_server.py
ROLLOUT=$TOKEN_CD/research/semantic_token_cd/pi0_fast_simpler_l6_matched_rollout.py
OUT=$TOKEN_CD/runs/widowx_bridge_normfix
GPU=1
PORT=8121
SESS_SRV=wx_normfix_server
MODE="${1:-start}"

mkdir -p "$OUT/logs"

if [[ "$MODE" == "verify" ]]; then
  echo "--- server log (norm stats line) ---"
  grep -iE "norm stats|Serving|warmups passed|server listening|Traceback|Error" "$OUT/logs/server_bridge_normfix.log" | tail -8
  echo "--- tmux ---"; tmux ls 2>/dev/null | grep -E "wx_normfix" || echo "(none)"
  echo "--- episodes ---"; find "$OUT/episodes" -name '*.json' 2>/dev/null | wc -l
  echo "--- results so far ---"
  /home/leju-suzhou/miniconda3/envs/pi05/bin/python - "$OUT" <<'PY'
import json, sys, glob
from pathlib import Path
out = Path(sys.argv[1])
rows = sorted(glob.glob(str(out/'episodes'/'*'/'*.json')))
print(f"{'task':30s} {'seed':>4} {'succ':>5} {'steps':>5} {'moved_wrong':>12} {'moved_correct':>13} {'grasped':>8}")
n_succ = n = 0
for f in rows:
    try: d = json.loads(Path(f).read_text())
    except Exception: continue
    if d.get('status') != 'complete': continue
    fi = d.get('final_info') or {}
    def b(x): return str(x).lower() == 'true'
    n += 1; n_succ += bool(d.get('success'))
    print(f"{d.get('task','?'):30s} {d.get('seed',-1):4d} {str(bool(d.get('success'))):>5} "
          f"{d.get('steps',-1):5d} {str(b(fi.get('moved_wrong_obj'))):>12} "
          f"{str(b(fi.get('moved_correct_obj'))):>13} {str(b(fi.get('is_src_obj_grasped'))):>8}")
print(f"\n{n_succ}/{n} success")
PY
  exit 0
fi

# --- guards ---------------------------------------------------------------------------------
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then echo "ABORT: port $PORT already in use" >&2; exit 1; fi
for s in "$SESS_SRV" wx_normfix_w0 wx_normfix_w1; do
  tmux has-session -t "$s" 2>/dev/null && { echo "ABORT: tmux session $s exists" >&2; exit 1; }
done
[[ -f "$CKPT/assets/local/bridge_lerobot/norm_stats.json" ]] || { echo "ABORT: bridge norm stats missing" >&2; exit 1; }

echo "=== $(date '+%F %T') starting bridge server on GPU$GPU:$PORT with its own norm stats ==="
tmux new-session -d -s "$SESS_SRV" "bash -lc 'cd $ROOT; source $CONDA_ROOT/etc/profile.d/conda.sh; conda activate pi05; export CUDA_VISIBLE_DEVICES=$GPU; export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.18; export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1; export OPENPI_DATA_HOME=$ROOT/checkpoints/openpi-assets HF_HUB_OFFLINE=1; export PI0FAST_CD_KMEANS=sklearn PI0FAST_CD_TAP_MODE=separate PI0FAST_CD_DECODE=verified; export PYTHONPATH=$OVERLAY:$ROOT/openpi/src:$ROOT/openpi/packages/openpi-client/src:$TOKEN_CD; python -u $SERVER --gpu $GPU --port $PORT --checkpoint $CKPT --norm-stats auto 2>&1 | tee $OUT/logs/server_bridge_normfix.log'"

for i in $(seq 1 120); do
  grep -q "warmups passed" "$OUT/logs/server_bridge_normfix.log" 2>/dev/null && break
  sleep 5
done
if ! grep -q "warmups passed" "$OUT/logs/server_bridge_normfix.log" 2>/dev/null; then
  echo "ABORT: server did not warm up"; tail -12 "$OUT/logs/server_bridge_normfix.log"; exit 1
fi
echo "server READY:"; grep -iE "norm stats|Serving" "$OUT/logs/server_bridge_normfix.log" | tail -2

echo "=== launching 2 workers (vanilla, 4 tasks x 4 seeds each) ==="
for w in 0 1; do
  tmux new-session -d -s "wx_normfix_w$w" "bash -lc 'cd $ROOT; source $CONDA_ROOT/etc/profile.d/conda.sh; conda activate pi0; export CUDA_VISIBLE_DEVICES=$GPU; export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json; export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; export JAX_PLATFORMS=cpu; export PYTHONPATH=/home/leju-suzhou/.cache/pi0fast_client_deps:$ROOT/openpi/packages/openpi-client/src:$ROOT/pcd_openvla_simpler_box_31b027e/source:$ROOT/SimplerEnv:$ROOT/SpatialVLA-SimplerEnv:$ROOT/SpatialVLA-SimplerEnv/ManiSkill2_real2sim:$ROOT/SpatialVLA/ManiSkill2_real2sim:$TOKEN_CD:$ROOT/openpi/src; python -u $ROLLOUT --gpu $GPU --worker-id $w --shard $OUT/manifests/worker_$w.json --host 127.0.0.1 --port $PORT --out $OUT --checkpoint $CKPT --rt-depth 8 2>&1 | tee $OUT/logs/worker_$w.log'"
  echo "  worker $w started"
  sleep 3
done

sleep 15
echo "--- verification ---"
tmux ls 2>/dev/null | grep wx_normfix
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i 0,1
echo "episodes so far: $(find "$OUT/episodes" -name '*.json' 2>/dev/null | wc -l)/32"
