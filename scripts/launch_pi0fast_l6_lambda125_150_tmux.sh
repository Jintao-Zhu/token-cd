#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-}"
if [[ "$MODE" != preflight && "$MODE" != full ]]; then
  echo "usage: $0 preflight|full" >&2
  exit 2
fi
ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD="$ROOT/token-cd"
OUT="$TOKEN_CD/runs/pi0fast_9task_l6jit_lambda125_150_rtfull_seed0_99_v1"
WORKER="$TOKEN_CD/scripts/run_pi0fast_l6_lambda125_150_worker.sh"
RUNNER="$TOKEN_CD/research/semantic_token_cd/pi0_fast_l6_lambda125_150_9task.py"
VANILLA="$TOKEN_CD/runs/pi0fast_9task_vanilla_vs_l6jit_lambda050_rtfull_100seed_v1"
LAMBDA100="$TOKEN_CD/runs/pi0fast_9task_l6jit_lambda100_rtfull_seed0_99_v1"

declare -a SESSIONS=(pi0f125150_g1 pi0f125150_w1 pi0f125150_g2 pi0f125150_w2 pi0f125150_g3 pi0f125150_w3)
declare -a SPECS=(
  "1 18770 google"
  "1 18768 widowx"
  "2 18771 google"
  "2 18765 widowx"
  "3 18772 google"
  "3 18763 widowx"
)
declare -a JOBS=(
  "google_robot_open_drawer:0-99 google_robot_place_apple_in_closed_top_drawer:0-99"
  "widowx_carrot_on_plate:0-99"
  "google_robot_close_drawer:0-99 google_robot_move_near:0-99"
  "widowx_put_eggplant_in_basket:0-99"
  "google_robot_pick_coke_can:0-99"
  "widowx_spoon_on_towel:0-99 widowx_stack_cube:0-99"
)

for i in "${!SPECS[@]}"; do
  read -r gpu port family <<< "${SPECS[$i]}"
  if ! ss -ltn 2>/dev/null | grep -q ":${port} "; then
    echo "ABORT: expected warmed $family service port $port is not listening" >&2
    exit 1
  fi
done
for gpu in 1 2 3; do
  used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if (( used >= 75000 )); then
    echo "ABORT: GPU$gpu already uses ${used} MiB; safety ceiling is 75000 MiB" >&2
    exit 1
  fi
done
for session in "${SESSIONS[@]}"; do
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "ABORT: session already exists: $session" >&2
    exit 1
  fi
done

if [[ "$MODE" == preflight ]]; then
  mkdir -p "$OUT"
  python3 - "$OUT" "$TOKEN_CD" "$VANILLA" "$LAMBDA100" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
out, repo, vanilla, l100 = map(Path, sys.argv[1:])
tasks = [
 "google_robot_open_drawer", "google_robot_close_drawer", "google_robot_pick_coke_can",
 "google_robot_move_near", "google_robot_place_apple_in_closed_top_drawer",
 "widowx_carrot_on_plate", "widowx_put_eggplant_in_basket", "widowx_spoon_on_towel", "widowx_stack_cube",
]
for root, arms in ((vanilla, ("vanilla", "l6_jit_lambda050")),
                   (l100, ("l6_matched_100",))):
    for task in tasks:
        for arm in arms:
            found = sum(1 for seed in range(100)
                        if (root / "episodes" / task / f"{arm}_seed{seed:03d}.json").is_file())
            if found != 100:
                raise SystemExit(f"reference incomplete: {root.name}/{task}/{arm}: {found}/100")
sources = [
 repo / "research/semantic_token_cd/pi0_fast_l6_lambda125_150_9task.py",
 repo / "research/semantic_token_cd/pi0_fast_simpler_l6_matched_rollout.py",
 repo / "research/semantic_token_cd/pi0_fast_l6_simpler_server.py",
 repo / "research/semantic_token_cd/pi0_fast_matched_cd.py",
 repo / "scripts/run_pi0fast_l6_lambda125_150_worker.sh",
 repo / "scripts/launch_pi0fast_l6_lambda125_150_tmux.sh",
]
protocol = {
 "experiment": "pi0fast_9task_l6jit_lambda125_150_rtfull_seed0_99_v1",
 "created_utc": datetime.now(timezone.utc).isoformat(),
 "question": "Compare L6-Matched lambda=1.25 and 1.50 on the same seeds/scenes as lambda=1.0.",
 "tasks": tasks,
 "seeds_inclusive": [0, 99],
 "arms": {"l6_matched_125": 1.25, "l6_matched_150": 1.50},
 "episodes_per_arm": 900,
 "new_episode_arm_evaluations": 1800,
 "fixed_method": {
   "checkpoint_by_family": {"google": str(repo.parent / "checkpoints/openpi_fast_finetuned/fractal"),
                             "widowx": str(repo.parent / "checkpoints/openpi_fast_finetuned/bridge")},
   "selector": "L6 prompt-attention",
   "budget": "dynamic Matched m_t",
   "counterfactual": "harmonic reconstruction",
   "intervention": "verified FAST action-code logits after fixed Action: header JIT",
   "widowx_state_layout": "Bridge",
   "widowx_action_clip": True,
 },
 "renderer": {"shader_dir": "rt", "rt_max_path_depth": 8,
              "rt_samples_per_pixel": 32, "rt_use_denoiser": True},
 "only_intended_change": "guidance lambda: 1.25 or 1.50",
 "references": {"vanilla_and_lambda050": str(vanilla), "lambda100": str(l100)},
 "pairing": "Same task/seed, exact initial agent-state SHA; 56x56 RGB MAE<=0.05 and max abs<=8. Mismatched pairs are reported and excluded.",
 "gpu_allowlist": [1, 2, 3],
 "worker_service_map": {
  "gpu1_google": {"port": 18770, "tasks": [tasks[0], tasks[4]]},
  "gpu1_widowx": {"port": 18768, "tasks": [tasks[5]]},
  "gpu2_google": {"port": 18771, "tasks": [tasks[1], tasks[3]]},
  "gpu2_widowx": {"port": 18765, "tasks": [tasks[6]]},
  "gpu3_google": {"port": 18772, "tasks": [tasks[2]]},
  "gpu3_widowx": {"port": 18763, "tasks": [tasks[7], tasks[8]]},
 },
 "source_sha256": {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
}
lock = out / "protocol.json"
if lock.exists():
    existing = json.loads(lock.read_text())
    if existing.get("experiment") != protocol["experiment"] or existing.get("new_episode_arm_evaluations") != 1800:
        raise SystemExit(f"unexpected existing protocol lock: {lock}")
else:
    lock.write_text(json.dumps(protocol, indent=2) + "\n")
print(json.dumps({"protocol": str(out / "protocol.json"), "new_episode_arm_evaluations": 1800,
                  "reference_checks": "900 Vanilla + 900 lambda=0.5 + 900 lambda=1.0 files present"}))
PY
  python3 - "$OUT/protocol.json" "$TOKEN_CD" <<'PY'
import hashlib, json, sys
from pathlib import Path
lock, repo = Path(sys.argv[1]), Path(sys.argv[2])
d = json.loads(lock.read_text())
sources = [
 repo / "research/semantic_token_cd/pi0_fast_l6_lambda125_150_9task.py",
 repo / "research/semantic_token_cd/pi0_fast_simpler_l6_matched_rollout.py",
 repo / "research/semantic_token_cd/pi0_fast_l6_simpler_server.py",
 repo / "research/semantic_token_cd/pi0_fast_matched_cd.py",
 repo / "scripts/run_pi0fast_l6_lambda125_150_worker.sh",
 repo / "scripts/launch_pi0fast_l6_lambda125_150_tmux.sh",
]
d["source_sha256"] = {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
lock.write_text(json.dumps(d, indent=2) + "\n")
PY
  WORK_MODE=preflight
else
  if [[ ! -f "$OUT/protocol.json" ]]; then
    echo "ABORT: missing protocol lock at $OUT" >&2
    exit 1
  fi
  python3 - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
tasks = json.loads((out / "protocol.json").read_text())["tasks"]
for task in tasks:
    p = out / "preflight" / task / "seed000.json"
    if not p.is_file():
        raise SystemExit(f"ABORT: missing preflight report {p}")
    d = json.loads(p.read_text())
    if d.get("status") != "passed" or not d.get("initial_agent_state_equal_to_vanilla") or not d.get("reset_scene_metadata_equal") or not d.get("lambda100_reference_matches_vanilla"):
        raise SystemExit(f"ABORT: failed preflight: {task}: {d}")
    for arm, lam in (("l6_matched_125", 1.25), ("l6_matched_150", 1.5)):
        a = d.get("arms", {}).get(arm, {})
        if a.get("lambda") != lam or a.get("selector_layer") != "L6" or not a.get("matched_budget_ok") or a.get("jit_header") is not True or not a.get("fast_action_stream_valid"):
            raise SystemExit(f"ABORT: failed {arm} preflight for {task}: {a}")
print("All 9 tasks passed both lambda preflights.")
PY
  WORK_MODE=full
fi

for i in "${!SPECS[@]}"; do
  read -r gpu port family <<< "${SPECS[$i]}"
  JOB_SPEC="${JOBS[$i]}"
  if [[ "$WORK_MODE" == preflight ]]; then
    JOB_SPEC="${JOB_SPEC//:0-99/:0-0}"
  fi
  tmux new-session -d -s "${SESSIONS[$i]}" \
    "bash '$WORKER' $gpu $port $family $WORK_MODE '$OUT' $JOB_SPEC"
done
printf 'Started %s workers for mode=%s. Run directory: %s\n' "${#SESSIONS[@]}" "$WORK_MODE" "$OUT"
tmux ls | grep -E '^pi0f125150_' || true
