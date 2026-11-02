"""Lever diagnostic for pi0.5 VTCG.

For a fixed set of (task, init_state) observations we call the policy server
three times with the SAME noise seed:

    clean    : matched arm, lambda = 0   (identical to the untouched policy)
    guided   : matched arm, lambda > 0, at selector layer L

and measure how far the returned action chunk actually moves:

    lever      = ||a_guided - a_clean|| over the guided dims 0..5
    rel_lever  = lever / ||a_clean||
    mask_area  = |selected| / 256
    fragments  = number of 4-neighbour components of the selected grid set
    largest_frac = largest component / |selected|

These per-task numbers are correlated with per-task success deltas by
analyze_pi05_lever.py. The point is to separate "the benchmark is too easy"
from "the intervention has no physical grip on this architecture".
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

GRID = 16
N_VISUAL = GRID * GRID
GUIDED_DIMS = 6


def parse_ids(value: str, *, upper_bound: int) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(upper_bound))
    values: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lower, upper = map(int, part.split("-", 1))
            values.extend(range(lower, upper + 1))
        else:
            values.append(int(part))
    result = sorted(set(values))
    if not result or result[0] < 0 or result[-1] >= upper_bound:
        raise ValueError(f"IDs must be nonempty and in [0, {upper_bound - 1}]")
    return result


def axis_angle(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64).copy()
    value[3] = np.clip(value[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - value[3] * value[3]))
    if math.isclose(denominator, 0.0):
        return np.zeros((3,), dtype=np.float64)
    return value[:3] * 2.0 * math.acos(value[3]) / denominator


def components(selected) -> list[int]:
    """Sizes of the 4-neighbour connected components of the selected grid cells."""
    cells = {int(index) for index in selected}
    seen: set[int] = set()
    sizes: list[int] = []
    for cell in cells:
        if cell in seen:
            continue
        stack = [cell]
        seen.add(cell)
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            row, col = divmod(current, GRID)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = row + dr, col + dc
                if 0 <= rr < GRID and 0 <= cc < GRID:
                    neighbour = rr * GRID + cc
                    if neighbour in cells and neighbour not in seen:
                        seen.add(neighbour)
                        stack.append(neighbour)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def done_keys(path: Path) -> set[tuple[int, int, int, float]]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        keys.add(
            (
                int(row["task_id"]),
                int(row["init_state_id"]),
                int(row["layer"]),
                round(float(row["lambda"]), 6),
            )
        )
    return keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8125)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--init-states", default="0-2")
    parser.add_argument("--env-seed", type=int, default=7)
    parser.add_argument("--noise-seed", type=int, default=20261020)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--layers", default="6,10")
    parser.add_argument("--lambdas", default="0.5")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy

    artifact = args.artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    result_path = artifact / "lever.jsonl"
    seen = done_keys(result_path)
    layers = [int(part) for part in args.layers.split(",") if part.strip()]
    lambdas = [float(part) for part in args.lambdas.split(",") if part.strip()]

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task_ids = parse_ids(args.tasks, upper_bound=suite.n_tasks)
    init_ids = parse_ids(args.init_states, upper_bound=50)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    for task_id in task_ids:
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
        for init_state_id in init_ids:
            pending = [
                (layer, lam)
                for layer in layers
                for lam in lambdas
                if (task_id, init_state_id, layer, round(lam, 6)) not in seen
            ]
            if not pending:
                continue
            env = OffScreenRenderEnv(
                bddl_file_name=bddl, camera_heights=256, camera_widths=256
            )
            env.seed(args.env_seed)
            try:
                env.reset()
                observation = env.set_init_state(initial_states[init_state_id])
                for _ in range(args.settle_steps):
                    observation, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                base_image = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
                wrist_image = np.ascontiguousarray(
                    observation["robot0_eye_in_hand_image"][::-1, ::-1]
                )
                base_image = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(base_image, 224, 224)
                )
                wrist_image = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_image, 224, 224)
                )
                state = np.concatenate(
                    (
                        observation["robot0_eef_pos"],
                        axis_angle(observation["robot0_eef_quat"]),
                        observation["robot0_gripper_qpos"],
                    )
                ).astype(np.float32)
                request_seed = args.noise_seed + task_id * 1_000_000 + init_state_id * 10_000

                def call(layer: int, lam: float) -> dict:
                    request = {
                        "observation/image": base_image,
                        "observation/wrist_image": wrist_image,
                        "observation/state": state,
                        "prompt": str(task.language),
                        "pi05_arm": "matched",
                        "pi05_noise_seed": request_seed,
                        "pi05_guidance_scale": float(lam),
                        "pi05_layer_index": int(layer),
                        "pi05_num_steps": args.num_steps,
                    }
                    return client.infer(request)

                clean_cache: dict[int, dict] = {}
                for layer, lam in pending:
                    if layer not in clean_cache:
                        clean_cache[layer] = call(layer, 0.0)
                    clean = clean_cache[layer]
                    guided = call(layer, lam)
                    clean_actions = np.asarray(clean["actions"], dtype=np.float32)
                    guided_actions = np.asarray(guided["actions"], dtype=np.float32)
                    delta = guided_actions[:, :GUIDED_DIMS] - clean_actions[:, :GUIDED_DIMS]
                    reference = clean_actions[:, :GUIDED_DIMS]
                    lever = float(np.linalg.norm(delta))
                    reference_norm = float(np.linalg.norm(reference))
                    trace = guided.get("pi05_paired_trace", {})
                    selected = list(trace.get("selected_token_ids", []))
                    sizes = components(selected)
                    record = {
                        "task_id": task_id,
                        "task_name": getattr(task, "name", None) or str(task_id),
                        "instruction": str(task.language),
                        "init_state_id": init_state_id,
                        "layer": layer,
                        "lambda": lam,
                        "lever": lever,
                        "lever_per_step": lever / max(delta.shape[0], 1),
                        "rel_lever": lever / reference_norm if reference_norm else float("nan"),
                        "clean_action_norm": reference_norm,
                        "delta_dim_norms": np.linalg.norm(delta, axis=0).round(6).tolist(),
                        "delta_step_norms": np.linalg.norm(delta, axis=1).round(6).tolist(),
                        "selected_count": int(trace.get("selected_token_count") or len(selected)),
                        "matched_m_t": int(trace.get("matched_m_t") or 0),
                        "mask_area": len(selected) / N_VISUAL,
                        "fragments": len(sizes),
                        "largest_fraction": (sizes[0] / len(selected)) if sizes and selected else float("nan"),
                        "entities": list(trace.get("entities", [])),
                        "attention_span": float(trace.get("attention_score_max", 0.0))
                        - float(trace.get("attention_score_min", 0.0)),
                        "num_steps": args.num_steps,
                    }
                    with result_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    print(json.dumps({k: record[k] for k in ("task_name", "init_state_id", "layer", "lambda", "lever", "rel_lever", "selected_count", "fragments")}), flush=True)
            finally:
                env.close()


if __name__ == "__main__":
    main()
