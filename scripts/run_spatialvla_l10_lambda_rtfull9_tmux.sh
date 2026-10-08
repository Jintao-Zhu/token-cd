#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "preflight" && "$MODE" != "formal" ]]; then
  echo "Usage: $0 preflight|formal" >&2
  exit 2
fi

ROOT="/home/leju-suzhou/zjt_ws/token-cd"
PY="$ROOT/research/semantic_token_cd/spatialvla_l10_lambda_rtfull9_rollout.py"
SNAP="$ROOT/artifacts/vanilla_recon_shr_canonical_0_299_v2"
OVERRIDE="$ROOT/artifacts/final_closure_snapshot_overrides_20261003"
REF="$ROOT/artifacts/spatialvla_l10_lambda_rtfull9_100_299_v1/reference_l10_matched"
if [[ "$MODE" == "preflight" ]]; then
  ART="$ROOT/artifacts/spatialvla_l10_lambda_rtfull9_preflight_20261008"
else
  ART="$ROOT/artifacts/spatialvla_l10_lambda_rtfull9_100_299_v1"
fi
LOG="$ART/logs"
mkdir -p "$LOG"

launch() {
  local name="$1" gpu="$2" setup="$3" tasks="$4" seeds="$5" preflight="$6"
  local args=(python "$PY" --artifact "$ART" --snapshot-artifact "$SNAP"
    --snapshot-override-artifact "$OVERRIDE" --reference-artifact "$REF"
    --arms lambda_025,lambda_075,lambda_100 --gpu "$gpu" --worker-id "$name"
    --tasks "$tasks" --seeds "$seeds" --model-setup "$setup")
  if [[ "$preflight" == 1 ]]; then args+=(--preflight); fi
  if tmux has-session -t "$name" 2>/dev/null; then
    echo "Refusing to replace existing tmux session: $name" >&2
    exit 1
  fi
  local quoted=""
  printf -v quoted '%q ' "${args[@]}"
  tmux new-session -d -s "$name" "bash -lc 'cd $(printf %q "$ROOT"); source scripts/activate_simpler_local.sh; export CUDA_VISIBLE_DEVICES=$gpu; export PYTHONUNBUFFERED=1; $quoted 2>&1 | tee -a $(printf %q "$LOG/$name.log")'"
  echo "started $name gpu=$gpu setup=$setup tasks=$tasks seeds=$seeds preflight=$preflight"
}

if [[ "$MODE" == "preflight" ]]; then
  launch svla_l10lam_pre_g6 6 google_robot google_robot_open_drawer 100 1
  launch svla_l10lam_pre_g7 7 widowx_bridge widowx_carrot_on_plate 100 1
else
  # Each task/seed's three lambda arms are sequential in the same process and
  # on the same physical GPU. Shards are disjoint and cover seeds 100..299.
  GOOGLE="google_robot_open_drawer,google_robot_close_drawer,google_robot_pick_coke_can,google_robot_move_near,google_robot_place_apple_in_closed_top_drawer"
  WIDOWX="widowx_carrot_on_plate,widowx_put_eggplant_in_basket,widowx_spoon_on_towel,widowx_stack_cube"
  launch svla_l10lam_g6a 6 google_robot "$GOOGLE" 100-166 0
  launch svla_l10lam_g6b 6 google_robot "$GOOGLE" 167-233 0
  launch svla_l10lam_g7a 7 google_robot "$GOOGLE" 234-299 0
  launch svla_l10lam_g6c 6 widowx_bridge "$WIDOWX" 100-166 0
  launch svla_l10lam_g7b 7 widowx_bridge "$WIDOWX" 167-233 0
  launch svla_l10lam_g7c 7 widowx_bridge "$WIDOWX" 234-299 0
fi
