#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD="$ROOT/token-cd"
OUT="$TOKEN_CD/runs/pi0fast_9task_vanilla_vs_l6jit_lambda050_rtfull_100seed_v1"
SERVER_LAUNCHER="$TOKEN_CD/scripts/run_pi0fast_9task_jit_compare_server.sh"
WORKER_LAUNCHER="$TOKEN_CD/scripts/run_pi0fast_9task_jit_compare_worker.sh"

for gpu in 4 5 6 7; do
  used=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpu" | sed '/^$/d' | wc -l)
  if [[ "$used" -gt 0 ]]; then
    echo "Refusing to use GPU $gpu: it already has $used compute process(es)." >&2
    exit 2
  fi
done
for port in 18841 18842 18843 18844; do
  if ss -ltn | grep -q ":$port "; then
    echo "Refusing to use occupied port $port." >&2
    exit 2
  fi
done

mkdir -p "$OUT/logs" "$OUT/pairs" "$OUT/episodes"
cat > "$OUT/protocol.json" <<'JSON'
{
  "experiment": "pi0fast_9task_vanilla_vs_l6jit_lambda050_rtfull_100seed_v1",
  "requested_arm_evaluations": 1800,
  "tasks": [
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
    "google_robot_place_apple_in_closed_top_drawer",
    "widowx_carrot_on_plate",
    "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel",
    "widowx_stack_cube"
  ],
  "seeds": [0, 99],
  "arms": ["vanilla", "l6_jit_lambda050"],
  "episodes_per_task_arm": 100,
  "method": {
    "selector": "L6 prompt-to-visual attention",
    "budget": "dynamic Matched m_t",
    "counterfactual": "harmonic reconstruction",
    "guidance_lambda": 0.5,
    "optimized_decode": "compiled fixed Action: header (JIT)",
    "FAST_action_stream": "validated against official OpenPI decoder; on decoder disagreement, use and record Vanilla fallback for that replan"
  },
  "renderer": {
    "shader_dir": "rt",
    "rt_max_path_depth": 8,
    "rt_samples_per_pixel": 32,
    "rt_use_denoiser": true
  },
  "checkpoint_by_task_family": {
    "google_robot": "/home/leju-suzhou/zjt_ws/checkpoints/openpi_fast_finetuned/fractal",
    "widowx": "/home/leju-suzhou/zjt_ws/checkpoints/openpi_fast_finetuned/bridge"
  },
  "widowx_input_protocol": {
    "state_layout": "Bridge [xyz, rpy, zero_pad, gripper]",
    "action_clip_to_bridge_q01_q99": true,
    "note": "Bridge checkpoint is not a SIMPLER-WidowX fine-tune; this is the existing Bridge checkpoint and verified SIMPLER input/action adapter. Interpret WidowX scores as transfer performance."
  },
  "pairing": "Each arm independently resets the same task seed. Pair rows require exact initial agent-state SHA-256 and 56x56 RGB difference MAE<=0.05 with max absolute difference<=8; exact full-resolution RGB SHA-256 is recorded because RT sampling noise can change a few pixels.",
  "gpu_plan": {
    "allowed_gpus": [4, 5, 6, 7],
    "servers": {"18841": {"gpu": 4, "family": "google"}, "18842": {"gpu": 5, "family": "google"}, "18843": {"gpu": 6, "family": "widowx"}, "18844": {"gpu": 7, "family": "widowx"}}
  }
}
JSON

start_server() {
  local gpu="$1" port="$2" family="$3" session="pi0f9_v05_srv${1}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Refusing to replace existing tmux session $session." >&2
    exit 2
  fi
  tmux new-session -d -s "$session" "bash -lc '$SERVER_LAUNCHER $gpu $port $family $OUT'"
}

start_server 4 18841 google
start_server 5 18842 google
start_server 6 18843 widowx
start_server 7 18844 widowx

for attempt in $(seq 1 360); do
  ready=0
  for port in 18841 18842 18843 18844; do
    if ss -ltn | grep -q ":$port "; then ready=$((ready + 1)); fi
  done
  if [[ "$ready" -eq 4 ]]; then break; fi
  if [[ "$attempt" -eq 360 ]]; then
    echo "Model servers did not all become ready. Check $OUT/logs/server_gpu*.log" >&2
    exit 3
  fi
  sleep 10
done

start_worker() {
  local gpu="$1" port="$2" family="$3" tag="$4"
  shift 4
  local session="pi0f9_v05_pre_${tag}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Refusing to replace existing tmux session $session." >&2
    exit 2
  fi
  local args=""
  for job in "$@"; do args+=" $job"; done
  tmux new-session -d -s "$session" "bash -lc '$WORKER_LAUNCHER $gpu $port $family $OUT${args}'"
}

start_worker 4 18841 google g4 google_robot_open_drawer:0-1 google_robot_close_drawer:0-1 google_robot_pick_coke_can:0-1
start_worker 5 18842 google g5 google_robot_move_near:0-1 google_robot_place_apple_in_closed_top_drawer:0-1
start_worker 6 18843 widowx w6 widowx_carrot_on_plate:0-1 widowx_put_eggplant_in_basket:0-1
start_worker 7 18844 widowx w7 widowx_spoon_on_towel:0-1 widowx_stack_cube:0-1

echo "Preflight started in tmux. Output: $OUT"
echo "After all four preflight worker sessions exit with matching hashes and valid traces, launch the full 0-99 shards with the documented worker map."
