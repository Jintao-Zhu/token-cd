#!/usr/bin/env bash
# Condition 2: pi0-FAST Bridge checkpoint fed with its *trained* Bridge input convention.
#
# Condition 1 (runs/widowx_bridge_normfix) = correct norm stats but the legacy Fractal/LIBERO
# state layout [pos, quat_xyzw, 1 - gripper] and no action clipping -> 0 interaction.
# Condition 2 (this) = same checkpoint, same server, but state = [pos, euler_rpy (R_topdown
# conjugated), PAD(0), gripper] and actions clipped to the training q01/q99, i.e. exactly what
# the checkpoint's own norm_stats say it was trained with (see the layout analysis in
# docs/AGENT_CONSTRAINTS.md and KNOWN_LIMITATIONS).
#
# Reuses the bridge policy server already listening on 127.0.0.1:8121 (GPU 1).
# GPU POLICY: only GPU 1/2/3 are authorised.
set -uo pipefail

ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD=$ROOT/token-cd
CONDA_ROOT=/home/leju-suzhou/miniconda3
CKPT=$ROOT/checkpoints/openpi_fast_finetuned/bridge
ROLLOUT=$TOKEN_CD/research/semantic_token_cd/pi0_fast_simpler_l6_matched_rollout.py
OUT=$TOKEN_CD/runs/widowx_bridge_layout2
GPU="${1:-1}"
PORT="${2:-8121}"
SEEDS="${3:-0-3}"

mkdir -p "$OUT/logs" "$OUT/manifests"
# guard: the bridge server must be alive
if ! (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then
  echo "ABORT: no server listening on 127.0.0.1:$PORT (start run_widowx_bridge_normfix.sh first)" >&2
  exit 1
fi
case "$GPU" in 1|2|3) ;; *) echo "ABORT: GPU $GPU not authorised (only 1/2/3)" >&2; exit 2;; esac

TASKS=(widowx_carrot_on_plate widowx_put_eggplant_in_basket widowx_spoon_on_towel widowx_stack_cube)
if [[ ! -f "$OUT/manifests/worker_0.json" ]]; then
  "$CONDA_ROOT/envs/pi05/bin/python" - "$OUT" "$SEEDS" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1]); spec = sys.argv[2]
seeds = []
for part in spec.split(','):
    if '-' in part:
        a, b = part.split('-'); seeds += list(range(int(a), int(b) + 1))
    elif part:
        seeds.append(int(part))
tasks = ["widowx_carrot_on_plate", "widowx_put_eggplant_in_basket", "widowx_spoon_on_towel", "widowx_stack_cube"]
half = (len(seeds) + 1) // 2
for wid, part in enumerate([seeds[:half], seeds[half:]]):
    jobs = [{"task": t, "seed": s, "arms": ["vanilla"]} for t in tasks for s in part]
    (out / f"manifests/worker_{wid}.json").write_text(json.dumps({"worker_id": wid, "jobs": jobs}, indent=2))
    print(f"worker_{wid}: {len(jobs)} episodes, seeds {part}")
PY
fi

for w in 0 1; do
  s="wx_layout2_w$w"
  tmux has-session -t "$s" 2>/dev/null && { echo "ABORT: tmux session $s exists" >&2; exit 1; }
done
for w in 0 1; do
  s="wx_layout2_w$w"
  tmux new-session -d -s "$s" "bash -lc 'source $CONDA_ROOT/etc/profile.d/conda.sh; conda activate pi0; export CUDA_VISIBLE_DEVICES=$GPU; export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json; export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; export JAX_PLATFORMS=cpu; export PI0FAST_WIDOWX_LAYOUT=bridge; export PYTHONPATH=/home/leju-suzhou/.cache/pi0fast_client_deps:$ROOT/openpi/packages/openpi-client/src:$ROOT/pcd_openvla_simpler_box_31b027e/source:$ROOT/SimplerEnv:$ROOT/SpatialVLA-SimplerEnv:$ROOT/SpatialVLA-SimplerEnv/ManiSkill2_real2sim:$ROOT/SpatialVLA/ManiSkill2_real2sim:$TOKEN_CD:$ROOT/openpi/src; python -u $ROLLOUT --gpu $GPU --worker-id $w --shard $OUT/manifests/worker_$w.json --host 127.0.0.1 --port $PORT --out $OUT --checkpoint $CKPT --rt-depth 8 2>&1 | tee $OUT/logs/worker_$w.log'"
  echo "started $s (gpu=$GPU port=$PORT layout=bridge)"
  sleep 3
done

sleep 12
echo "--- verification ---"
tmux ls 2>/dev/null | grep wx_layout2 || true
echo "episodes: $(find "$OUT/episodes" -name '*.json' 2>/dev/null | wc -l)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i $GPU
