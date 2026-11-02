#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-}"
case "$MODE" in services|preflight|full|status) ;; *)
  echo "usage: $0 services|preflight|full|status" >&2; exit 2;;
esac
ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD="$ROOT/token-cd"
OUT="$TOKEN_CD/runs/pi0fast_l6_scene_repair_36seeds_rtfull_v1"
RUNNER="$TOKEN_CD/research/semantic_token_cd/pi0_fast_l6_scene_repair_36.py"
WORKER="$TOKEN_CD/scripts/run_pi0fast_l6_scene_repair_worker.sh"
SERVER="$TOKEN_CD/scripts/run_pi0fast_9task_jit_compare_server.sh"
VANILLA="$TOKEN_CD/runs/pi0fast_9task_vanilla_vs_l6jit_lambda050_rtfull_100seed_v1"
LAMBDA100="$TOKEN_CD/runs/pi0fast_9task_l6jit_lambda100_rtfull_seed0_99_v1"
SOURCE="$TOKEN_CD/runs/pi0fast_9task_l6jit_lambda125_150_rtfull_seed0_99_v1"
mkdir -p "$OUT/logs"

if [[ ! -f "$OUT/protocol.json" ]]; then
  python3 - "$OUT" "$SOURCE" "$VANILLA" "$LAMBDA100" "$RUNNER" "$WORKER" "$SERVER" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
out, source, vanilla, l100, *sources = map(Path, sys.argv[1:])
tasks = ("google_robot_move_near", "widowx_put_eggplant_in_basket")
groups = {task: [] for task in tasks}
for task in tasks:
    for seed in range(100):
        pair = json.loads((source / "pairs" / task / f"seed{seed:03d}.json").read_text())
        arms = pair["arms"]
        old = bool(pair.get("lambda050_pair_available"))
        ref = bool(pair.get("lambda100_pair_available"))
        current = all(arms[name][field] for name, field in (
            ("l6_matched_125", "valid_vs_vanilla"),
            ("l6_matched_150", "valid_vs_vanilla"),
            ("l6_matched_125", "valid_vs_lambda100"),
            ("l6_matched_150", "valid_vs_lambda100"),
        ))
        if not (old and ref and current):
            groups[task].append(seed)
if len(groups[tasks[0]]) != 12 or len(groups[tasks[1]]) != 24:
    raise SystemExit(f"unexpected mismatch set: {groups}")
protocol = {
    "experiment": "pi0fast_l6_scene_repair_36seeds_rtfull_v1",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "reason": "Repair only the 36 task-seed rows excluded from the strict five-arm lambda table.",
    "tasks_and_seeds": groups,
    "arms": {"vanilla": 0.0, "l6_matched_050": 0.5, "l6_matched_100": 1.0,
             "l6_matched_125": 1.25, "l6_matched_150": 1.5},
    "new_closed_loop_episodes": sum(map(len, groups.values())) * 5,
    "checkpoint_by_task_family": {
        "google_robot_move_near": str(Path("/home/leju-suzhou/zjt_ws/checkpoints/openpi_fast_finetuned/fractal")),
        "widowx_put_eggplant_in_basket": str(Path("/home/leju-suzhou/zjt_ws/checkpoints/openpi_fast_finetuned/bridge")),
    },
    "renderer": {"shader_dir": "rt", "rt_max_path_depth": 8,
                 "rt_samples_per_pixel": 32, "rt_use_denoiser": True},
    "scene_pairing": "Capture env.get_state() once after reset for each task-seed, restore that exact full simulator state before every arm, and verify complete state hash plus 56x56 RGB tolerance.",
    "gpu_roles": {"inference_fractal": 4, "inference_bridge": 5,
                  "render_move_near": 6, "render_eggplant": 7},
    "ports": {"fractal": 18840, "bridge": 18841},
    "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
}
if protocol["new_closed_loop_episodes"] != 180:
    raise SystemExit(f"unexpected repair episode count: {protocol['new_closed_loop_episodes']}")
out.mkdir(parents=True, exist_ok=True)
(out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
print(json.dumps({"protocol": str(out / "protocol.json"), "seeds": groups,
                  "closed_loop_episodes": protocol["new_closed_loop_episodes"]}, indent=2))
PY
fi

if [[ "$MODE" == status ]]; then
  tmux ls 2>/dev/null | grep -E '^pi0frepair_' || true
  python3 - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out=Path(sys.argv[1]); protocol=json.loads((out/"protocol.json").read_text())
for task,seeds in protocol["tasks_and_seeds"].items():
    pre=sum((out/"preflight"/task/f"seed{s:03d}.json").is_file() for s in seeds)
    pairs=[out/"pairs"/task/f"seed{s:03d}.json" for s in seeds]
    done=sum(p.is_file() and json.loads(p.read_text()).get("status")=="paired" for p in pairs)
    episodes=sum((out/"episodes"/task/f"{arm}_seed{s:03d}.json").is_file() and
                 json.loads((out/"episodes"/task/f"{arm}_seed{s:03d}.json").read_text()).get("status")=="complete"
                 for s in seeds for arm in protocol["arms"])
    print(f"{task}: preflight {pre}/{len(seeds)}, paired seeds {done}/{len(seeds)}, episodes {episodes}/{len(seeds)*5}")
PY
  exit 0
fi

if [[ "$MODE" == services ]]; then
  for gpu in 4 5; do
    used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if (( used >= 75000 )); then echo "ABORT: GPU$gpu has ${used} MiB in use" >&2; exit 1; fi
  done
  if ss -ltn 2>/dev/null | grep -Eq ':(18840|18841) '; then
    echo "ABORT: repair inference port 18840 or 18841 is already occupied" >&2; exit 1
  fi
  for session in pi0frepair_srv4 pi0frepair_srv5; do
    if tmux has-session -t "$session" 2>/dev/null; then echo "ABORT: tmux session exists: $session" >&2; exit 1; fi
  done
  tmux new-session -d -s pi0frepair_srv4 "bash '$SERVER' 4 18840 google '$OUT'"
  tmux new-session -d -s pi0frepair_srv5 "bash '$SERVER' 5 18841 widowx '$OUT'"
  echo "Started Fractal inference on GPU4:18840 and Bridge inference on GPU5:18841."
  exit 0
fi

if ! ss -ltn 2>/dev/null | grep -q ':18840 ' || ! ss -ltn 2>/dev/null | grep -q ':18841 '; then
  echo "ABORT: both inference services on ports 18840/18841 must be listening" >&2
  exit 1
fi
for gpu in 6 7; do
  used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if (( used >= 75000 )); then echo "ABORT: GPU$gpu has ${used} MiB in use" >&2; exit 1; fi
done
for session in pi0frepair_pre_g6 pi0frepair_pre_w7 pi0frepair_full_g6 pi0frepair_full_w7; do
  if tmux has-session -t "$session" 2>/dev/null; then echo "ABORT: tmux session exists: $session" >&2; exit 1; fi
done

python3 - "$OUT/protocol.json" "$MODE" <<'PY'
import json, sys
from pathlib import Path
protocol=json.loads(Path(sys.argv[1]).read_text()); mode=sys.argv[2]
if mode == "preflight":
    print(",".join(map(str,protocol["tasks_and_seeds"]["google_robot_move_near"])))
    print(",".join(map(str,protocol["tasks_and_seeds"]["widowx_put_eggplant_in_basket"])))
else:
    for task,seeds in protocol["tasks_and_seeds"].items():
        for seed in seeds:
            p=Path(sys.argv[1]).parent/"preflight"/task/f"seed{seed:03d}.json"
            if not p.is_file() or json.loads(p.read_text()).get("status") != "passed":
                raise SystemExit(f"ABORT: failed/missing preflight for {task}/{seed}")
    print("preflight_ok")
PY

if [[ "$MODE" == preflight ]]; then
  mapfile -t SEED_LISTS < <(python3 - "$OUT/protocol.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))["tasks_and_seeds"]
for t in ("google_robot_move_near","widowx_put_eggplant_in_basket"):
 print(",".join(map(str,d[t])))
PY
)
  tmux new-session -d -s pi0frepair_pre_g6 "bash '$WORKER' 6 18840 google_robot_move_near preflight '${SEED_LISTS[0]}'"
  tmux new-session -d -s pi0frepair_pre_w7 "bash '$WORKER' 7 18841 widowx_put_eggplant_in_basket preflight '${SEED_LISTS[1]}'"
  echo "Started canonical-scene preflight over all 36 task-seed fixtures."
else
  mapfile -t SEED_LISTS < <(python3 - "$OUT/protocol.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))["tasks_and_seeds"]
for t in ("google_robot_move_near","widowx_put_eggplant_in_basket"):
 print(",".join(map(str,d[t])))
PY
)
  tmux new-session -d -s pi0frepair_full_g6 "bash '$WORKER' 6 18840 google_robot_move_near full '${SEED_LISTS[0]}'"
  tmux new-session -d -s pi0frepair_full_w7 "bash '$WORKER' 7 18841 widowx_put_eggplant_in_basket full '${SEED_LISTS[1]}'"
  echo "Started 180 canonical-scene repair episodes on GPUs6/7."
fi
