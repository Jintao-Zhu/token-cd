#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/leju-suzhou/zjt_ws/token-cd
ARTIFACT=$ROOT/artifacts/openvla_global_lambda_grid_225_250_0_199_v1
mkdir -p "$ARTIFACT/logs"
echo "[$(date -Is)] waiting for 90 preflight summaries and arrays"
deadline=$((SECONDS + 43200))
while (( SECONDS < deadline )); do
  summaries=$(find "$ARTIFACT/episodes" -name 'episode_*_summary.json' 2>/dev/null | wc -l)
  arrays=$(find "$ARTIFACT/episodes" -name 'episode_*_arrays.npz' 2>/dev/null | wc -l)
  if [[ "$summaries" -eq 90 && "$arrays" -eq 90 ]]; then
    break
  fi
  sleep 30
done
if [[ "${summaries:-0}" -ne 90 || "${arrays:-0}" -ne 90 ]]; then
  echo "[$(date -Is)] preflight timeout/failure: summaries=${summaries:-0}, arrays=${arrays:-0}" | tee "$ARTIFACT/PREFLIGHT_COORDINATOR_FAILED.txt"
  exit 1
fi
source "$ROOT/scripts/activate_simpler_local.sh"
cd "$ROOT"
python research/semantic_token_cd/validate_openvla_lambda225_250_preflight.py 2>&1 | tee "$ARTIFACT/logs/preflight_validation.log"
bash scripts/launch_openvla_lambda225_250_formal_g45_tmux.sh 2>&1 | tee "$ARTIFACT/logs/formal_launch.log"
echo "[$(date -Is)] formal 3,600 episode-arm evaluation launched after preflight PASS"
