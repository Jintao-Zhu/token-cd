#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD="$ROOT/token-cd"
OUT="$TOKEN_CD/runs/pi0fast_jit_header_ab_4task_20seed_rtfull_lambda100_v1"
mkdir -p "$OUT/logs"

for gpu in 1 2 3; do
  if ! tmux has-session -t "pi0fh_ab_srv${gpu}" 2>/dev/null; then
    echo "ABORT: expected warmed model service on GPU$gpu is missing" >&2
    exit 1
  fi
done
if [[ -e "$OUT/protocol.json" ]]; then
  echo "ABORT: output already has a protocol lock: $OUT" >&2
  exit 1
fi

python3 - "$OUT" "$TOKEN_CD" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
out, repo = map(Path, sys.argv[1:])
source_files = [
    repo / "research/semantic_token_cd/pi0_fast_jit_header_closed_loop.py",
    repo / "research/semantic_token_cd/pi0_fast_simpler_l6_matched_rollout.py",
    repo / "research/semantic_token_cd/pi0_fast_matched_cd.py",
    repo / "research/semantic_token_cd/pi0_fast_l6_simpler_server.py",
    repo / "scripts/run_pi0fast_jit_ab_server.sh",
    repo / "scripts/run_pi0fast_jit_ab_lambda100_worker.sh",
]
protocol = {
    "experiment": "pi0fast_jit_header_ab_4task_20seed_rtfull_lambda100_v1",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "question": "Does compiling the shared fixed Action: header preserve best-global-lambda L6-Matched action behavior while reducing latency?",
    "tasks": ["google_robot_open_drawer", "google_robot_pick_coke_can", "widowx_carrot_on_plate", "widowx_stack_cube"],
    "seeds": list(range(20)),
    "arms": ["l6_eager_header", "l6_jit_header"],
    "guidance_lambda": 1.0,
    "new_episode_arm_evaluations": 4 * 20 * 2,
    "renderer": {"shader_dir": "rt", "rt_max_path_depth": 8, "rt_samples_per_pixel": 32, "rt_use_denoiser": True},
    "checkpoint_by_family": {
        "google": str(repo.parent / "checkpoints/openpi_fast_finetuned/fractal"),
        "widowx": str(repo.parent / "checkpoints/openpi_fast_finetuned/bridge"),
    },
    "widowx_inputs": {"state_layout": "bridge", "action_clip": True},
    "fixed_method": {
        "selector": "L6 prompt attention", "budget": "dynamic Matched m_t",
        "negative": "harmonic reconstruction", "guidance": "FAST action-code logits after shared fixed Action: header",
        "tap_mode": "separate", "decode_mode": "verified", "static_view_cache": True, "embed_jit": False,
    },
    "only_intended_change": "eager versus compiled fixed-header decode steps, selected per request with cd_jit_header=0/1",
    "pairing": "same task seed, each arm reset independently; initial RGB/state SHA-256 must match",
    "primary_diagnostics": ["first action chunk equality", "first selected-token equality", "mean inference seconds per replan"],
    "closed_loop_diagnostics": ["success counts", "paired Rescue/Harm", "steps", "full-rollout equality, with later states potentially diverged"],
    "gpu_allowlist": [1, 2, 3],
    "gpu_worker_map": {
        "google_robot_open_drawer": {"renderer_gpu": 1, "server_gpu": 1, "server_port": 18761},
        "google_robot_pick_coke_can": {"renderer_gpu": 1, "server_gpu": 1, "server_port": 18761},
        "widowx_carrot_on_plate": {"renderer_gpu": 2, "server_gpu": 2, "server_port": 18762},
        "widowx_stack_cube": {"renderer_gpu": 3, "server_gpu": 3, "server_port": 18763},
    },
    "source_sha256": {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
}
(out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
print(json.dumps({"protocol": str(out / "protocol.json"), "episode_arm_evaluations": protocol["new_episode_arm_evaluations"], "lambda": protocol["guidance_lambda"]}))
PY

tmux new-session -d -s pi0fh1_ab_worker1 "bash $TOKEN_CD/scripts/run_pi0fast_jit_ab_lambda100_worker.sh 1 google_robot_open_drawer 18761 google"
tmux new-session -d -s pi0fh1_ab_worker2 "bash $TOKEN_CD/scripts/run_pi0fast_jit_ab_lambda100_worker.sh 1 google_robot_pick_coke_can 18761 google"
tmux new-session -d -s pi0fh1_ab_worker3 "bash $TOKEN_CD/scripts/run_pi0fast_jit_ab_lambda100_worker.sh 2 widowx_carrot_on_plate 18762 widowx"
tmux new-session -d -s pi0fh1_ab_worker4 "bash $TOKEN_CD/scripts/run_pi0fast_jit_ab_lambda100_worker.sh 3 widowx_stack_cube 18763 widowx"

printf 'Started lambda=1.0 paired rollout workers on GPUs 1/2/3.\n'
tmux ls | grep '^pi0fh1_ab_' || true
printf 'Run directory: %s\n' "$OUT"
