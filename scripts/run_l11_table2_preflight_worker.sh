#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 4 ]]; then echo "usage: $0 GPU CARD WORKER TASK [TASK...]" >&2; exit 2; fi
GPU=$1; CARD=$2; WORKER=$3; shift 3
ROOT=/home/leju-suzhou/zjt_ws/token-cd
ARTIFACT="$ROOT/artifacts/l11_table2_main_ablation_v1/preflight/card${CARD}"
SNAPSHOT="$ROOT/artifacts/vanilla_recon_shr_canonical_0_299_v2"
case "$CARD" in
  1) ARMS=zero,clean_global_mean,local_neighbor_mean ;;
  2) ARMS=hard_intervention_only,harmonic_negative_only ;;
  3) ARMS=layer5,layer11 ;;
  *) echo "invalid ablation group: $CARD" >&2; exit 2 ;;
esac
source "$ROOT/scripts/activate_simpler_local.sh"
cd "$ROOT"
for TASK in "$@"; do
  echo "[$(date -Is)] preflight worker=$WORKER gpu=$GPU group=$CARD task=$TASK seeds=100-104 arms=$ARMS"
  python research/semantic_token_cd/seven_card_openvla_rollout.py \
    --card "$CARD" --artifact "$ARTIFACT" --snapshot-artifact "$SNAPSHOT" \
    --task "$TASK" --seeds 100-104 --gpu "$GPU" --worker-id "$WORKER" --arms "$ARMS"
done
