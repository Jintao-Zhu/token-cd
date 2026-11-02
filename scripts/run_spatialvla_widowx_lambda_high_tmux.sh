#!/usr/bin/env bash
# High-lambda extension of the SpatialVLA WidowX scan: 4 tasks x 300 seeds x lambda {1.25, 1.50}
# = 2400 episodes. Same runner/protocol family as run_spatialvla_widowx_lambda_0_299_tmux.sh
# (arms lambda_125,lambda_150 were added to ARM_LAMBDAS on 2026-10-13), same canonical snapshots,
# same RT-full lambda=0.5 reference (reference_l10_matched in the low-scan artifact).
#
# Layout: tmux, 9 workers, 3 per GPU on 1/2/3, disjoint seed shards covering 0-299 once:
#   GPU1: 0-33, 102-135, 204-237 | GPU2: 34-67, 136-169, 238-271 | GPU3: 68-101, 170-203, 272-299
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "preflight" && "$MODE" != "formal" ]]; then
  echo "Usage: $0 preflight|formal" >&2
  exit 2
fi

ROOT="/home/leju-suzhou/zjt_ws/token-cd"
PY="$ROOT/research/semantic_token_cd/spatialvla_widowx_lambda_0_299_rollout.py"
SNAP="$ROOT/artifacts/vanilla_recon_shr_canonical_0_299_v2"
OVERRIDE="$ROOT/artifacts/final_closure_snapshot_overrides_20261003"
REF="$ROOT/artifacts/spatialvla_widowx_lambda_0_299_v1/reference_l10_matched"
ART_MAIN="$ROOT/artifacts/spatialvla_widowx_lambda_high_0_299_v1"
ART_PRE="$ROOT/artifacts/spatialvla_widowx_lambda_high_0_299_preflight"
WIDOWX="widowx_carrot_on_plate,widowx_put_eggplant_in_basket,widowx_spoon_on_towel,widowx_stack_cube"

if [[ "$MODE" == "preflight" ]]; then ART="$ART_PRE"; else ART="$ART_MAIN"; fi
LOG="$ART/logs"; mkdir -p "$LOG"

# guard: 6 sessions from the low scan must not collide by name
SESS=(svla_wxlamh_pre svla_wxlamh_g1a svla_wxlamh_g1b svla_wxlamh_g1c
      svla_wxlamh_g2a svla_wxlamh_g2b svla_wxlamh_g2c
      svla_wxlamh_g3a svla_wxlamh_g3b svla_wxlamh_g3c)
for s in "${SESS[@]}"; do
  if tmux has-session -t "$s" 2>/dev/null; then
    echo "Refusing to replace existing tmux session: $s" >&2
    exit 1
  fi
done
[[ -d "$REF/episodes" ]] || { echo "missing reference_l10_matched: $REF" >&2; exit 2; }

launch() {
  local name="$1" gpu="$2" seeds="$3" preflight="$4"
  local args=(python "$PY" --artifact "$ART" --snapshot-artifact "$SNAP"
    --snapshot-override-artifact "$OVERRIDE" --reference-artifact "$REF"
    --arms lambda_125,lambda_150 --gpu "$gpu" --worker-id "$name"
    --tasks "$WIDOWX" --seeds "$seeds" --model-setup widowx_bridge)
  if [[ "$preflight" == 1 ]]; then args+=(--preflight); fi
  local quoted=""
  printf -v quoted '%q ' "${args[@]}"
  tmux new-session -d -s "$name" "bash -lc 'cd $(printf %q "$ROOT"); source scripts/activate_simpler_local.sh; export CUDA_VISIBLE_DEVICES=$gpu; export PYTHONUNBUFFERED=1; $quoted 2>&1 | tee -a $(printf %q "$LOG/$name.log")'"
  echo "started $name gpu=$gpu seeds=$seeds preflight=$preflight"
}

if [[ "$MODE" == "preflight" ]]; then
  launch svla_wxlamh_pre 1 0-3 1
else
  launch svla_wxlamh_g1a 1 0-33 0
  launch svla_wxlamh_g1b 1 102-135 0
  launch svla_wxlamh_g1c 1 204-237 0
  launch svla_wxlamh_g2a 2 34-67 0
  launch svla_wxlamh_g2b 2 136-169 0
  launch svla_wxlamh_g2c 2 238-271 0
  launch svla_wxlamh_g3a 3 68-101 0
  launch svla_wxlamh_g3b 3 170-203 0
  launch svla_wxlamh_g3c 3 272-299 0
fi

sleep 10
echo "--- verification ---"
tmux ls 2>/dev/null | grep svla_wxlamh || true
# count real python workers only (pattern must not match the bash -lc wrapper)
echo "python workers: $(pgrep -cf "^python .*/spatialvla_widowx_lambda_0_299_rollout" || true)"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i 1,2,3
if [[ "$MODE" == "formal" ]]; then
  echo "high-lambda episodes: $(find "$ART_MAIN/episodes" -name 'episode_*_summary.json' 2>/dev/null | wc -l)/2400"
fi
