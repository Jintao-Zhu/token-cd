#!/usr/bin/env bash
set -euo pipefail

repo=/home/leju-suzhou/zjt_ws/token-cd
python=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
pcd_source=/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source
artifact="$repo/adaptive_shr_experiment"
source_artifact="$repo/artifacts/vanilla_recon_shr_canonical_0_299_v2"
pi0_complete="$repo/artifacts/pi0_shr_5task_0_299_v1/logs/COMPLETE"
session=adaptive_shr_v1

mkdir -p "$artifact/rollout_logs" "$artifact/episode_json" "$artifact/paired_results" "$artifact/statistics"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "tmux session already exists: $session"
  exit 1
fi

wait_args="--wait-for '$pi0_complete'"
if [ "${1:-}" = "--now" ]; then
  wait_args=""
fi
command="cd '$repo' && PYTHONPATH='$repo:$pcd_source' HF_HUB_OFFLINE=1 '$python' research/semantic_token_cd/launch_adaptive_shr.py --artifact '$artifact' --snapshot-artifact '$source_artifact' $wait_args 2>&1 | tee -a '$artifact/rollout_logs/master.log'"
tmux new-session -d -s "$session" "$command"
echo "started detached tmux session: $session"
if [ -n "$wait_args" ]; then
  echo "it will wait for: $pi0_complete"
else
  echo "starting immediately"
fi
