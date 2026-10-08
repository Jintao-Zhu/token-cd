from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("artifacts/openvla_global_lambda_grid_225_250_0_199_v1")
TASKS = (
    "google_robot_open_drawer", "google_robot_close_drawer",
    "google_robot_pick_coke_can", "google_robot_move_near",
    "google_robot_place_apple_in_closed_top_drawer", "widowx_carrot_on_plate",
    "widowx_put_eggplant_in_basket", "widowx_spoon_on_towel", "widowx_stack_cube",
)
ARMS = {"lambda_225": 2.25, "lambda_250": 2.50}
records = {}
errors = []
for task in TASKS:
    for seed in range(5):
        pair = []
        for arm, expected_lambda in ARMS.items():
            base = ROOT / "episodes" / task / arm
            summary = base / f"episode_{seed:03d}_summary.json"
            arrays = base / f"episode_{seed:03d}_arrays.npz"
            if not summary.is_file() or not arrays.is_file():
                errors.append(f"missing output: {summary}")
                continue
            data = json.loads(summary.read_text())
            if data.get("lambda") != expected_lambda:
                errors.append(f"wrong lambda in {summary}: {data.get('lambda')}")
            if data.get("technical_pass") is not True:
                errors.append(f"technical audit failed: {summary}")
            if data.get("attention_layers") != [11]:
                errors.append(f"wrong layer: {summary}")
            pair.append((data.get("initial_state_sha256"), data.get("initial_rgb_sha256")))
        if len(pair) == 2 and pair[0] != pair[1]:
            errors.append(f"paired state/RGB mismatch: {task}:{seed}")
        records[f"{task}:{seed}"] = pair
if errors:
    raise SystemExit("PREFLIGHT FAIL\n" + "\n".join(errors[:40]))
out = {
    "status": "PASS", "protocol_id": "OPENVLA_GLOBAL_LAMBDA_GRID_0_199_V1",
    "arms": ARMS, "tasks": list(TASKS), "seeds": list(range(5)),
    "episode_arm_evaluations": len(TASKS) * 5 * len(ARMS),
    "checks": ["lambda exact", "technical audit pass", "L11 layer exact", "paired state/RGB hashes", "summary and arrays present"],
}
(ROOT / "PREFLIGHT_PASS_225_250.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
