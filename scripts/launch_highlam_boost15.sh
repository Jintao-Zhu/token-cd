#!/usr/bin/env bash
# Boost the running high-lambda SpatialVLA WidowX scan from 9 to 15 workers (5 per GPU 1/2/3).
#
# The runner takes --tasks and --seeds only (no shard file), so the remaining work is split into
# 15 disjoint (task, seed-range) shards balanced by remaining episode count
# (runs/highlam_boost15_plan.json: 6x spoon, 6x stack, 3x eggplant = 1465 episodes).
# Resume logic skips any (task, seed, arm) whose result file already has status=complete, so a
# shard that overlaps finished work costs nothing.
#
# GPU POLICY: only 1/2/3 (docs/AGENT_CONSTRAINTS.md).
# Usage: bash launch_highlam_boost15.sh
set -uo pipefail

ROOT=/home/leju-suzhou/zjt_ws/token-cd
PY="$ROOT/research/semantic_token_cd/spatialvla_widowx_lambda_0_299_rollout.py"
ART="$ROOT/artifacts/spatialvla_widowx_lambda_high_0_299_v1"
SNAP="$ROOT/artifacts/vanilla_recon_shr_canonical_0_299_v2"
OVERRIDE="$ROOT/artifacts/final_closure_snapshot_overrides_20261003"
REF="$ROOT/artifacts/spatialvla_widowx_lambda_0_299_v1/reference_l10_matched"
PLAN="$ROOT/runs/highlam_boost15_plan.json"
LOG="$ART/logs"
PYBIN=/home/leju-suzhou/miniconda3/envs/pi05/bin/python

mkdir -p "$LOG"
[[ -f "$PLAN" ]] || { echo "missing plan: $PLAN" >&2; exit 1; }

"$PYBIN" - "$PLAN" > /tmp/highlam_shards.tsv <<'PY'
import json, sys
for i, p in enumerate(json.load(open(sys.argv[1]))):
    print(i, 1 + i % 3, p["task"], f"{p['lo']}-{p['hi']}")
PY

while read -r idx gpu task seeds; do
  name="svla_wxlamh2_g${gpu}_${idx}"
  if tmux has-session -t "$name" 2>/dev/null; then echo "skip existing $name"; continue; fi
  args=(python "$PY" --artifact "$ART" --snapshot-artifact "$SNAP"
        --snapshot-override-artifact "$OVERRIDE" --reference-artifact "$REF"
        --arms lambda_125,lambda_150 --gpu "$gpu" --worker-id "$name"
        --tasks "$task" --seeds "$seeds" --model-setup widowx_bridge)
  printf -v quoted '%q ' "${args[@]}"
  tmux new-session -d -s "$name" "bash -lc 'cd $(printf %q "$ROOT"); source scripts/activate_simpler_local.sh; export CUDA_VISIBLE_DEVICES=$gpu; export PYTHONUNBUFFERED=1; $quoted 2>&1 | tee -a $(printf %q "$LOG/$name.log")'"
  echo "started $name gpu=$gpu task=${task#widowx_} seeds=$seeds"
  sleep 2
done < /tmp/highlam_shards.tsv

echo "--- launched $(tmux ls 2>/dev/null | grep -c svla_wxlamh2) workers ---"
