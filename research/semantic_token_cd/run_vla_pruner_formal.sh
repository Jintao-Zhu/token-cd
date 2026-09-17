#!/usr/bin/env bash
# VLA-Pruner FORMAL 1200 -- 4 workers on GPUs 2/3, tmux, self-healing supervisor.
# Each worker owns a disjoint 25-seed band x 4 tasks (no duplicate claiming).
# The bash loop re-invokes the driver until --check reports the chunk complete,
# which transparently resumes after process crashes / technical failures.
set -u
WS=/home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
ROOT=$WS/artifacts/vla_pruner_openvla_reproduction/formal_1200
MAN=$ROOT/RUN_MANIFEST.json
mkdir -p "$ROOT/logs"
SUPERVISE() { # worker gpu
  local w=$1 gpu=$2
  local name="vp_formal_${w}"
  local log="$ROOT/logs/${w}.log"
  local body="WS=$WS; export PYTHONPATH=$WS:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source; export HF_HUB_OFFLINE=1; export TOKENIZERS_PARALLELISM=false; cd $WS; r=1; while [ \$r -le 60 ]; do echo \"[\$(date +%F_%T)] round \$r start\" >> '$log'; $PY research/semantic_token_cd/vla_pruner_formal_rollout.py --manifest '$MAN' --worker $w --gpu $gpu --round \$r >> '$log' 2>&1; if $PY research/semantic_token_cd/vla_pruner_formal_rollout.py --manifest '$MAN' --worker $w --gpu $gpu --check >> '$log' 2>&1; then echo \"[\$(date +%F_%T)] round \$r CHUNK COMPLETE\" >> '$log'; break; fi; sleep 20; r=\$((r+1)); done"
  tmux kill-session -t "$name" 2>/dev/null || true
  tmux new-session -d -s "$name" "$body"
  echo "launched $name (gpu=$gpu)"
}
SUPERVISE w1 2
SUPERVISE w2 2
SUPERVISE w3 3
SUPERVISE w4 3
