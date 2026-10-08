#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 4 ]]; then
  echo "usage: $0 GPU ARM WORKER_ID TASK [TASK...]" >&2
  exit 2
fi
GPU=$1 ARM=$2 WORKER=$3
shift 3
ROOT=/home/leju-suzhou/zjt_ws/token-cd
BASE="$ROOT/artifacts/l11_table2_main_ablation_v1"
SNAPSHOT="$ROOT/artifacts/vanilla_recon_shr_canonical_0_299_v2"
source "$ROOT/scripts/activate_simpler_local.sh"
cd "$ROOT"
for TASK in "$@"; do
  echo "[$(date -Is)] formal shard worker=$WORKER gpu=$GPU arm=$ARM task=$TASK seeds=100-199"
  python research/semantic_token_cd/l11_table2_openvla_rollout.py \
    --card 6 --artifact "$BASE/formal/$ARM" --snapshot-artifact "$SNAPSHOT" \
    --task "$TASK" --seeds 100-199 --gpu "$GPU" --worker-id "$WORKER" --arms "$ARM"
done
