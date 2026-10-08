import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts/openvla_lambda1_extension_200_299_v1"
TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
    "google_robot_place_apple_in_closed_top_drawer",
    "widowx_carrot_on_plate",
    "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel",
    "widowx_stack_cube",
)
DEPTH = {task: (8 if task in {"google_robot_open_drawer", "google_robot_close_drawer"} else 0) for task in TASKS}

failures = []
for task in TASKS:
    folder = ARTIFACT / "episodes" / task / "lambda_100"
    summary = folder / "episode_200_summary.json"
    arrays = folder / "episode_200_arrays.npz"
    if not summary.is_file() or not arrays.is_file():
        failures.append(f"{task}: missing summary or arrays")
        continue
    item = json.loads(summary.read_text())
    checks = {
        "protocol": item.get("protocol_id") == "OPENVLA_LAMBDA1_EXTENSION_200_299_V1",
        "task_seed": item.get("task") == task and item.get("seed") == 200,
        "lambda": item.get("lambda") == 1.0 and item.get("lambda_exact") is True,
        "technical": item.get("technical_pass") is True,
        "layer": item.get("attention_layers") == [11] and item.get("layer_exact") is True,
        "renderer": item.get("rendering", {}).get("rt_max_path_depth") == DEPTH[task],
        "finite arrays": arrays.stat().st_size > 0,
    }
    failures.extend(f"{task}: {name} failed" for name, ok in checks.items() if not ok)

if failures:
    raise SystemExit("PREFLIGHT FAIL\n" + "\n".join(failures))
print(json.dumps({"status": "PASS", "preflight_episodes": len(TASKS), "seed": 200, "lambda": 1.0}, indent=2))
