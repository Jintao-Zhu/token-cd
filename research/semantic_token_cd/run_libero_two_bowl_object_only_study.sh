#!/usr/bin/env bash
set -euo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
ROOT="$WS/artifacts/libero_two_bowl_object_only_v1_20260926"
SESSION=libero_bowl_object_only_20260926_recovery01
WORKER="$WS/research/semantic_token_cd/run_libero_two_bowl_object_only_worker.sh"
T_CENTER=pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate
T_RAMEKIN=pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Refusing to overwrite existing tmux session: $SESSION" >&2
  exit 2
fi
existing=$(find "$ROOT/episodes" -type f -name 'episode_*.json' 2>/dev/null | wc -l)

# Twelve workers total: exactly three model processes on GPUs 1, 2, 3, and 6.
# The four center-task conditions are split into 25-seed shards; each GPU gets
# 100 episodes across its three workers. The four ramekin shards cover 0-49.
jobs=(
  "1 target_only $T_CENTER 0-24 recovery_target_c_001"
  "1 target_only $T_CENTER 25-49 recovery_target_c_002"
  "1 target_only $T_RAMEKIN 0-49 recovery_target_r_001"
  "2 distractor_only $T_CENTER 0-24 recovery_distractor_c_001"
  "2 distractor_only $T_CENTER 25-49 recovery_distractor_c_002"
  "2 distractor_only $T_RAMEKIN 0-49 recovery_distractor_r_001"
  "3 non_bowl_target_count $T_CENTER 0-24 recovery_nonbowl_target_c_001"
  "3 non_bowl_target_count $T_CENTER 25-49 recovery_nonbowl_target_c_002"
  "3 non_bowl_target_count $T_RAMEKIN 0-49 recovery_nonbowl_target_r_001"
  "6 non_bowl_distractor_count $T_CENTER 0-24 recovery_nonbowl_distractor_c_001"
  "6 non_bowl_distractor_count $T_CENTER 25-49 recovery_nonbowl_distractor_c_002"
  "6 non_bowl_distractor_count $T_RAMEKIN 0-49 recovery_nonbowl_distractor_r_001"
)

mkdir -p "$ROOT/logs" "$ROOT/episodes"
first_wid=$(awk '{print $5}' <<< "${jobs[0]}")
tmux new-session -d -s "$SESSION" -n "$first_wid" "bash '$WORKER' ${jobs[0]}"
for job in "${jobs[@]:1}"; do
  wid=$(awk '{print $5}' <<< "$job")
  tmux new-window -d -t "$SESSION" -n "$wid" "bash '$WORKER' $job"
done

echo "session=$SESSION workers=${#jobs[@]} seeds=0-49 tasks=2 arms=4 total_episodes=400 existing_to_skip=$existing render_gpu=7"
tmux list-windows -t "$SESSION"
