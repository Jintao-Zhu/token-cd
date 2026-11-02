#!/usr/bin/env bash
# Budget-rule smoke test: one task x 20 seeds x three arms, one worker per GPU.
#
#   GPU1  lambda_100      (reproduces the existing lambda=1.0 Ours matched episodes)
#   GPU2  top_p085_l100   (Attention-TopP p=0.85, UNCLIPPED: bounds [1, 256])
#   GPU3  fixed032_l100   (Fixed budget m=32)
#
# Everything else is the lambda=1.0 Ours protocol (OpenVLA fractal, L11 prompt attention,
# harmonic counterfactual, contrast execution, projector intervention, per-task RT depth).
# Writes into a SCRATCH artifact so the 900-episode Ours reference is never touched.
#
# GPU policy: GPU 1/2/3 only, never 0 and never 4-7.
set -uo pipefail

ROOT=/home/leju-suzhou/zjt_ws
REPO=$ROOT/token-cd
TASK=${TASK:-widowx_carrot_on_plate}
SEEDS=${SEEDS:-200-219}
ARTIFACT=${ARTIFACT:-$REPO/artifacts/openvla_budget_rule_smoke_v1}
PROTOCOL=OPENVLA_BUDGET_RULE_SMOKE_V1
ROLLOUT=research/semantic_token_cd/openvla_budget_rule_rollout.py

mkdir -p "$ARTIFACT/logs"
cd "$REPO"

start() {
  local gpu=$1 arms=$2 tag=$3
  local sess="budget_smoke_${tag}"
  local log="$ARTIFACT/logs/smoke_g${gpu}_${tag}.log"
  if tmux has-session -t "$sess" 2>/dev/null; then echo "$sess already up"; return; fi
  tmux new-session -d -s "$sess" "bash -lc '
    set -uo pipefail
    source $REPO/scripts/activate_simpler_local.sh
    export CUDA_VISIBLE_DEVICES=$gpu
    cd $REPO
    python $ROLLOUT --artifact $ARTIFACT --task $TASK --seeds $SEEDS --gpu $gpu \
      --worker-id smoke_g${gpu}_${tag} --arms $arms --protocol-id $PROTOCOL --lock-seeds 200-299 \
      >> $log 2>&1
    echo \"[exit \$?] $(date -Is)\" >> $log'"
  echo "started $sess (gpu$gpu arms=$arms)"
}

start 1 lambda_100      matched
start 2 top_p085_l100   topp
start 3 fixed032_l100   fixed

echo "--- logs ---"
for f in "$ARTIFACT"/logs/smoke_g*.log; do echo "$(basename "$f"): $(grep -c '^{' "$f" 2>/dev/null) episodes so far"; done
