#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/leju-suzhou/zjt_ws"
REPO="$ROOT/token-cd"
WORKER="$REPO/scripts/run_univla_depth0_hidden_cd_worker.sh"

launch() {
  local gpu="$1" task="$2" seeds="$3" name="$4"
  if tmux has-session -t "$name" 2>/dev/null; then
    echo "tmux session already exists: $name" >&2
    return 1
  fi
  tmux new-session -d -s "$name" "bash -lc '$WORKER $gpu $task $seeds $name'"
}

# The GPU 1 carrot worker is the seed-0 runtime preflight and continues through
# seeds 0-99. The remaining shards balance the three workers per GPU.
launch 1 widowx_put_eggplant_in_basket 0-49 univla_d0_g1_eggplant_000_049
launch 1 widowx_spoon_on_towel 0-49 univla_d0_g1_spoon_000_049
launch 6 widowx_put_eggplant_in_basket 50-99 univla_d0_g6_eggplant_050_099
launch 6 widowx_stack_cube 0-99 univla_d0_g6_stack_000_099
launch 6 widowx_spoon_on_towel 50-99 univla_d0_g6_spoon_050_099

tmux list-sessions -F '#{session_name}' | rg '^univla_d0_g(1|6)_'
