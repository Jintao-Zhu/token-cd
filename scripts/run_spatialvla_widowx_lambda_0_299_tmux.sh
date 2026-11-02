#!/usr/bin/env bash
# SpatialVLA WidowX lambda scan: 4 tasks x 300 seeds x lambda {0.25, 0.75, 1.00} = 3600 episodes.
#
# lambda=0.50 is the existing baseline (SPATIALVLA_RTfull9_300_SEEDS_V1 record), so this scan only
# needs the three missing strengths. Runner: spatialvla_widowx_lambda_0_299_rollout.py
# (protocol SPATIALVLA_SIMPLER_L10_MATCHED_LAMBDA_WIDOWX_RTFULL_4TASK_0_299_V1, seeds 0-299,
# SAPIEN RT full depth8/spp32/denoiser, paired on the canonical snapshot per task/seed).
#
# Layout: tmux, 9 workers, 3 per GPU on GPU 1/2/3 (GPU 0 unused, GPU 4-7 belong to another user).
# Disjoint seed shards cover 0-299 exactly once:
#   GPU1: 0-33, 102-135, 204-237
#   GPU2: 34-67, 136-169, 238-271
#   GPU3: 68-101, 170-203, 272-299
#
# 1620 episodes from the earlier rtfull9 lambda study (seeds 100-299, same renderer / snapshots /
# lambda definition, canonical-sha verified 1620/1620) are hard-linked into this artifact, so the
# resume logic skips them and only the remaining 1980 episodes actually run.
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
ART_MAIN="$ROOT/artifacts/spatialvla_widowx_lambda_0_299_v1"
ART_PRE="$ROOT/artifacts/spatialvla_widowx_lambda_0_299_preflight"
WIDOWX="widowx_carrot_on_plate,widowx_put_eggplant_in_basket,widowx_spoon_on_towel,widowx_stack_cube"

if [[ "$MODE" == "preflight" ]]; then ART="$ART_PRE"; else ART="$ART_MAIN"; fi
LOG="$ART/logs"
mkdir -p "$LOG"

# --- guard 1: the previous experiment must be fully gone before we take GPU 1/2/3 -------------
srv=$(pgrep -cf "pi0_fast_l6_simpler_server" || true)
wrk=$(pgrep -cf "pi0_fast_simpler_l6_matched_rollout" || true)
if [[ "$((srv + wrk))" -gt 0 ]]; then
  echo "Refusing to start: pi0-fast run still alive (servers=$srv workers=$wrk)." >&2
  echo "Wait for 3500/3500, run its self-check, then stop those processes first." >&2
  exit 3
fi

# --- guard 2: never take over existing tmux sessions ------------------------------------------
SESS=(svla_wxlam_pre svla_wxlam_g1a svla_wxlam_g1b svla_wxlam_g1c
      svla_wxlam_g2a svla_wxlam_g2b svla_wxlam_g2c
      svla_wxlam_g3a svla_wxlam_g3b svla_wxlam_g3c)
for s in "${SESS[@]}"; do
  if tmux has-session -t "$s" 2>/dev/null; then
    echo "Refusing to replace existing tmux session: $s" >&2
    exit 1
  fi
done

launch() {
  local name="$1" gpu="$2" seeds="$3" preflight="$4"
  local args=(python "$PY" --artifact "$ART" --snapshot-artifact "$SNAP"
    --snapshot-override-artifact "$OVERRIDE" --reference-artifact "$REF"
    --arms lambda_025,lambda_075,lambda_100 --gpu "$gpu" --worker-id "$name"
    --tasks "$WIDOWX" --seeds "$seeds" --model-setup widowx_bridge)
  if [[ "$preflight" == 1 ]]; then args+=(--preflight); fi
  local quoted=""
  printf -v quoted '%q ' "${args[@]}"
  tmux new-session -d -s "$name" "bash -lc 'cd $(printf %q "$ROOT"); source scripts/activate_simpler_local.sh; export CUDA_VISIBLE_DEVICES=$gpu; export PYTHONUNBUFFERED=1; $quoted 2>&1 | tee -a $(printf %q "$LOG/$name.log")'"
  echo "started $name gpu=$gpu seeds=$seeds preflight=$preflight"
}

if [[ "$MODE" == "preflight" ]]; then
  launch svla_wxlam_pre 1 0-3 1
else
  launch svla_wxlam_g1a 1 0-33 0
  launch svla_wxlam_g1b 1 102-135 0
  launch svla_wxlam_g1c 1 204-237 0
  launch svla_wxlam_g2a 2 34-67 0
  launch svla_wxlam_g2b 2 136-169 0
  launch svla_wxlam_g2c 2 238-271 0
  launch svla_wxlam_g3a 3 68-101 0
  launch svla_wxlam_g3b 3 170-203 0
  launch svla_wxlam_g3c 3 272-299 0
fi

sleep 10
echo "--- verification ---"
tmux ls 2>/dev/null | grep svla_wxlam || true
echo "workers running: $(pgrep -cf 'spatialvla_widowx_lambda_0_299_rollout' || true)"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i 1,2,3
echo "episodes already present (of 3600): $(find $ART_MAIN/episodes -name 'episode_*_summary.json' 2>/dev/null | wc -l)"
