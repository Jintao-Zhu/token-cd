#!/usr/bin/env bash
# 2x2 ablation for the WidowX pi0-FAST Bridge input convention.
#
# Factors:
#   A) state layout : legacy [pos, quat_xyzw, 1-gripper]  vs  bridge [pos, euler(R_topdown), PAD(0), gripper]
#   B) action clip  : off vs on (clip the 6 action dims to the training q01/q99)
#
# Already measured (conditions 1 and 2):
#   (legacy, off) -> runs/widowx_bridge_normfix : 0 interaction / 26 episodes
#   (bridge, on ) -> runs/widowx_bridge_layout2 : 4-5/8 interaction, 2/8 success
# This script runs the two remaining cells:
#   (bridge, off) and (legacy, on)
#
# Reuses the bridge policy server already listening on 127.0.0.1:8121 (GPU 1).
# GPU POLICY: only GPU 1/2/3 are authorised (docs/AGENT_CONSTRAINTS.md).
#
# Usage: bash run_widowx_bridge_ablation.sh <layout> <clip> <gpus> <seeds> <tag>
#   e.g. bash run_widowx_bridge_ablation.sh bridge 0 2 0-1 b-noclip
set -uo pipefail

LAYOUT="${1:?layout: legacy|bridge}"
CLIP="${2:?clip: 0|1}"
GPUS_CSV="${3:-2}"
SEEDS="${4:-0-1}"
TAG="${5:?tag, e.g. b-noclip}"
PORT="${6:-8121}"

ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD=$ROOT/token-cd
CONDA_ROOT=/home/leju-suzhou/miniconda3
CKPT=$ROOT/checkpoints/openpi_fast_finetuned/bridge
ROLLOUT=$TOKEN_CD/research/semantic_token_cd/pi0_fast_simpler_l6_matched_rollout.py
OUT=$TOKEN_CD/runs/widowx_bridge_ablation_$TAG

case "$LAYOUT" in legacy|bridge) ;; *) echo "ABORT: bad layout $LAYOUT" >&2; exit 2;; esac
case "$CLIP" in 0|1) ;; *) echo "ABORT: bad clip $CLIP" >&2; exit 2;; esac
IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
for g in "${GPUS[@]}"; do
  case "$g" in 1|2|3) ;; *) echo "ABORT: GPU $g not authorised (only 1/2/3)" >&2; exit 2;; esac
done
if ! (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then
  echo "ABORT: no bridge server on 127.0.0.1:$PORT" >&2; exit 1
fi

mkdir -p "$OUT/logs" "$OUT/manifests"
TASKS=(widowx_carrot_on_plate widowx_put_eggplant_in_basket widowx_spoon_on_towel widowx_stack_cube)

for w in 0 1; do
  s="wx_abl_${TAG}_w$w"
  tmux has-session -t "$s" 2>/dev/null && { echo "ABORT: tmux session $s exists" >&2; exit 1; }
done

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
  g="${GPUS[$(( w % ${#GPUS[@]} ))]}"
  s="wx_abl_${TAG}_w$w"
  tmux new-session -d -s "$s" "bash -lc 'source $CONDA_ROOT/etc/profile.d/conda.sh; conda activate pi0; export CUDA_VISIBLE_DEVICES=$g; export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json; export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; export JAX_PLATFORMS=cpu; export PI0FAST_WIDOWX_LAYOUT=$LAYOUT PI0FAST_WIDOWX_CLIP=$CLIP; export PYTHONPATH=/home/leju-suzhou/.cache/pi0fast_client_deps:$ROOT/openpi/packages/openpi-client/src:$ROOT/pcd_openvla_simpler_box_31b027e/source:$ROOT/SimplerEnv:$ROOT/SpatialVLA-SimplerEnv:$ROOT/SpatialVLA-SimplerEnv/ManiSkill2_real2sim:$ROOT/SpatialVLA/ManiSkill2_real2sim:$TOKEN_CD:$ROOT/openpi/src; python -u $ROLLOUT --gpu $g --worker-id $w --shard $OUT/manifests/worker_$w.json --host 127.0.0.1 --port $PORT --out $OUT --checkpoint $CKPT --rt-depth 8 2>&1 | tee $OUT/logs/worker_$w.log'"
  echo "started $s (layout=$LAYOUT clip=$CLIP gpu=$g)"
  sleep 3
done

sleep 10
echo "--- verification ---"
tmux ls 2>/dev/null | grep "wx_abl_${TAG}" || true
echo "episodes: $(find "$OUT/episodes" -name '*.json' 2>/dev/null | wc -l)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 1,2,3
