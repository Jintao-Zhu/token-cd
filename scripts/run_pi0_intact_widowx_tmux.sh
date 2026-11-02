#!/usr/bin/env bash
# INTACT Bridge pi0 (flow-matching) reference rollout on the 4 SIMPLER WidowX tasks.
#
# Discriminator: a Bridge-trained policy that the INTACT authors measured at 36-94% on these
# tasks.  If it also fails here, the problem is our environment/harness; if it works, the
# environment is fine and the pi0-FAST Bridge checkpoint (or its unknown input convention) is
# the problem.
#
# GPU POLICY: only GPU 1/2/3 are authorised (see docs/AGENT_CONSTRAINTS.md).
# Usage: bash run_pi0_intact_widowx_tmux.sh [seeds] [gpus]      e.g. ... "0-3" "2,3"
set -uo pipefail

ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD=$ROOT/token-cd
RUNTIME=$ROOT/pi0_intact_runtime/activate.sh
ROLLOUT=$TOKEN_CD/scripts/pi0_intact_widowx_rollout.py
OUT=$TOKEN_CD/runs/pi0_intact_widowx_rollout
SEEDS="${1:-0-3}"
GPUS_CSV="${2:-2,3}"
IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
# NOTE: keep PYTHONPATH minimal.  Adding $ROOT/SimplerEnv or $ROOT/SpatialVLA-SimplerEnv shadows
# mani_skill2_real2sim with another fork whose WidowX observations LACK "agent.eef_pos", which
# makes the reference bridge_state() fail with KeyError: 'eef_pos' (measured 2026-10-13).
ENVPATH="$TOKEN_CD"

mkdir -p "$OUT/logs"
[[ -f "$RUNTIME" ]] || { echo "missing INTACT runtime: $RUNTIME" >&2; exit 1; }
for g in "${GPUS[@]}"; do
  case "$g" in 1|2|3) ;; *) echo "ABORT: GPU $g not authorised (only 1/2/3)" >&2; exit 2;; esac
done

TASKS=(widowx_carrot_on_plate widowx_put_eggplant_in_basket widowx_spoon_on_towel widowx_stack_cube)
for T in "${TASKS[@]}"; do
  s="wx_intact_${T#widowx_}"
  if tmux has-session -t "$s" 2>/dev/null; then echo "ABORT: tmux session $s exists" >&2; exit 1; fi
done

i=0
for T in "${TASKS[@]}"; do
  g="${GPUS[$(( i % ${#GPUS[@]} ))]}"
  s="wx_intact_${T#widowx_}"
  tmux new-session -d -s "$s" "bash -lc 'source $RUNTIME; export CUDA_VISIBLE_DEVICES=$g; export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json; export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1; export PYTHONPATH=$ENVPATH:\$PYTHONPATH; cd $ROOT; python -u $ROLLOUT --gpu $g --task $T --seeds $SEEDS --out $OUT 2>&1 | tee $OUT/logs/$T.log'"
  echo "started $s (task=$T, seeds=$SEEDS, gpu=$g)"
  i=$(( i + 1 ))
  sleep 2
done

sleep 10
echo "--- verification ---"
tmux ls 2>/dev/null | grep wx_intact || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i 1,2,3
echo "episodes: $(find "$OUT" -name 'intact_seed*.json' 2>/dev/null | wc -l)"
