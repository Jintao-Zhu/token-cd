#!/usr/bin/env bash
set -euo pipefail

WS=/home/leju-suzhou/zjt_ws/token-cd
ROOT="$WS/artifacts/libero_two_bowl_attention_localization_v1_20260926"
SESSION=libero_bowl_anchor_causal_20260926
WORKER="$WS/research/semantic_token_cd/run_libero_two_bowl_anchor_worker.sh"

T_CENTER=pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate
T_RAMEKIN=pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Refusing to overwrite existing tmux session: $SESSION" >&2
  exit 2
fi
if find "$ROOT/episodes/target_anchor" "$ROOT/episodes/distractor_anchor" -type f -name 'episode_*.json' -print -quit 2>/dev/null | grep -q .; then
  echo "Refusing to overwrite existing anchor episodes under $ROOT/episodes" >&2
  exit 3
fi

jobs=(
  "1 target_anchor $T_CENTER 0-16 c_target_001"
  "1 target_anchor $T_RAMEKIN 0-16 r_target_001"
  "1 distractor_anchor $T_CENTER 0-16 c_distractor_001"
  "2 target_anchor $T_CENTER 17-33 c_target_002"
  "2 target_anchor $T_RAMEKIN 17-33 r_target_002"
  "2 distractor_anchor $T_CENTER 17-33 c_distractor_002"
  "3 target_anchor $T_CENTER 34-49 c_target_003"
  "3 target_anchor $T_RAMEKIN 34-49 r_target_003"
  "3 distractor_anchor $T_CENTER 34-49 c_distractor_003"
  "6 distractor_anchor $T_RAMEKIN 0-16 r_distractor_001"
  "6 distractor_anchor $T_RAMEKIN 17-33 r_distractor_002"
  "6 distractor_anchor $T_RAMEKIN 34-49 r_distractor_003"
)

mkdir -p "$ROOT/logs/anchor" "$ROOT/episodes/target_anchor" "$ROOT/episodes/distractor_anchor"
tmux new-session -d -s "$SESSION" -n "$(echo "${jobs[0]}" | awk '{print $5}')" \
  "bash '$WORKER' ${jobs[0]}"
for job in "${jobs[@]:1}"; do
  wid=$(awk '{print $5}' <<< "$job")
  tmux new-window -d -t "$SESSION" -n "$wid" "bash '$WORKER' $job"
done

echo "session=$SESSION workers=${#jobs[@]} episodes_per_arm_task=50 total_new_episodes=200 render_gpu=7"
tmux list-windows -t "$SESSION"
