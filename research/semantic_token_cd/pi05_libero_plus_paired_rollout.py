"""Paired vanilla/matched pi0.5 rollout on LIBERO-Plus (robustness benchmark).

Same pairing discipline as the LIBERO-Long harness: identical initial simulator
state per pair (sha256 audited) and identical action noise at every replan, with
the vanilla arm being the untouched official OpenPI policy.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
import os
import re
from pathlib import Path
import time

import numpy as np


DEFAULT_ARTIFACT = (
    "/home/leju-suzhou/zjt_ws/token-cd/artifacts/pi05_libero_plus_paired_v1"
)
SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
PLUS_ROOT = Path("/home/leju-suzhou/zjt_ws/libero_plus_src")
CLASSIFICATION = PLUS_ROOT / "libero/libero/benchmark/task_classification.json"
SUPPORTED_ARMS = ("vanilla", "matched", "pcd", "uniform")
DEFAULT_ARMS = ("vanilla", "pcd")
PERTURBATION_SUFFIX = re.compile(
    r"(_view_.*|_noise_\d+|_table_\d+|_tb_\d+|_add_\d+|_light_\d+|_language_\d+)$"
)


def clean_instruction(task_name: str) -> str:
    """Recover the intended instruction from a LIBERO-Plus perturbed task name.

    Upstream derives the prompt from the BDDL filename, so perturbed variants
    carry suffixes such as ``_table_3`` or ``_view_0_0_100_2_6_initstate_0``
    inside the prompt text.  Stripping them restores the sentence the task was
    authored with (the perturbation itself is applied by the environment).
    """
    stripped = re.sub(r"_language_\d+", "", task_name)
    stripped = PERTURBATION_SUFFIX.sub("", stripped)
    fragments = stripped.split("_")
    if stripped[:1].isupper() and "SCENE" in stripped:
        scene = stripped.find("SCENE")
        after = stripped[scene + (8 if "SCENE10" in stripped else 7) :]
        fragments = after.split("_")
    return " ".join(fragment for fragment in fragments if fragment)


def axis_angle(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64).copy()
    value[3] = np.clip(value[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - value[3] * value[3]))
    if math.isclose(denominator, 0.0):
        return np.zeros((3,), dtype=np.float64)
    return value[:3] * 2.0 * math.acos(value[3]) / denominator


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def completed_keys(path: Path) -> set[tuple[str, int, str]]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("error"):
            continue
        keys.add((str(row["task_name"]), int(row["trial"]), str(row["arm"])))
    return keys


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def select_tasks(
    suite_name: str,
    suite,
    *,
    categories: list[str],
    levels: list[int],
    per_cell: int,
    task_ids: list[int] | None,
    seed: int,
) -> list[int]:
    """Pick benchmark indices, stratified by (perturbation category, difficulty)."""
    if task_ids:
        return [index for index in task_ids if 0 <= index < suite.n_tasks]
    table = json.loads(CLASSIFICATION.read_text())[suite_name]
    meta = {row["name"]: row for row in table}
    cells: dict[tuple[str, int], list[int]] = {}
    for index in range(suite.n_tasks):
        task = suite.get_task(index)
        row = meta.get(task.name)
        if row is None:
            continue
        category = str(row.get("category", "unknown"))
        level = int(row.get("difficulty_level") or 0)
        if categories and category not in categories:
            continue
        if levels and level not in levels:
            continue
        cells.setdefault((category, level), []).append(index)
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for key in sorted(cells):
        pool = sorted(cells[key])
        if per_cell < len(pool):
            picked = rng.choice(np.asarray(pool), size=per_cell, replace=False)
            chosen.extend(sorted(int(value) for value in picked))
        else:
            chosen.extend(pool)
    return sorted(set(chosen))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=Path(DEFAULT_ARTIFACT))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8125)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--categories", default="")
    parser.add_argument("--levels", default="")
    parser.add_argument("--per-cell", type=int, default=2)
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--task-order-index", type=int, default=0)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--name-filter", default="")
    parser.add_argument("--prompt-mode", choices=("clean", "raw"), default="clean")
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    parser.add_argument("--env-seed", type=int, default=7)
    parser.add_argument("--noise-seed", type=int, default=20261021)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--layer-index", type=int, default=None)
    parser.add_argument("--active-steps", default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--trace-dir", type=Path, default=None)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy

    artifact = args.artifact.resolve()
    source_artifact = Path(
        "/home/leju-suzhou/zjt_ws/token-cd/artifacts/pi05_libero_pcd_full_v1_20261024_01"
    ).resolve()
    trace_dir = args.trace_dir.resolve() if args.trace_dir is not None else None
    if trace_dir is not None:
        if artifact == source_artifact or source_artifact in artifact.parents:
            raise ValueError("diagnostic output cannot be inside the source artifact")
        if trace_dir == source_artifact or source_artifact in trace_dir.parents:
            raise ValueError("trace output cannot be inside the source artifact")
        trace_dir.mkdir(parents=True, exist_ok=False)
    artifact.mkdir(parents=True, exist_ok=True)
    result_path = artifact / "episodes.jsonl"
    done = completed_keys(result_path)
    suite = benchmark.get_benchmark_dict()[args.suite](args.task_order_index)
    max_steps = args.max_steps or SUITE_MAX_STEPS.get(args.suite, 520)
    categories = parse_csv(args.categories)
    levels = [int(part) for part in parse_csv(args.levels)]
    task_ids = [int(part) for part in parse_csv(args.task_ids)] if args.task_ids else None
    arms = tuple(parse_csv(args.arms))
    if not arms or any(arm not in SUPPORTED_ARMS for arm in arms):
        raise ValueError(f"arms must be drawn from {SUPPORTED_ARMS}")

    selected = select_tasks(
        args.suite,
        suite,
        categories=categories,
        levels=levels,
        per_cell=args.per_cell,
        task_ids=task_ids,
        seed=args.noise_seed,
    )
    if args.name_filter:
        selected = [i for i in selected if args.name_filter in suite.get_task(i).name]
        if not selected:
            raise SystemExit(f"name filter {args.name_filter!r} matched nothing")
    shard_index, shard_count = (int(part) for part in args.shard.split("/"))
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard must be W/N with 0 <= W < N")
    selected = selected[shard_index::shard_count]
    if not selected:
        raise SystemExit("no tasks selected")
    print(
        json.dumps(
            {
                "suite": args.suite,
                "n_tasks_in_suite": suite.n_tasks,
                "selected_tasks": len(selected),
                "max_steps": max_steps,
                "categories": categories or "all",
                "levels": levels or "all",
                "per_cell": args.per_cell,
                "arms": list(arms),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.plan_only:
        plan = {
            "suite": args.suite,
            "task_indices": selected,
            "task_names": [suite.get_task(index).name for index in selected],
        }
        (artifact / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        return

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    bddl_root = Path(get_libero_path("bddl_files"))

    for task_index in selected:
        task = suite.get_task(task_index)
        instruction = (
            clean_instruction(task.name) if args.prompt_mode == "clean" else str(task.language)
        )
        initial_states = suite.get_task_init_states(task_index)
        bddl = str(bddl_root / task.problem_folder / task.bddl_file)
        for arm in arms:
            key = (task.name, args.trial, arm)
            if key in done:
                continue
            env = OffScreenRenderEnv(
                bddl_file_name=bddl, camera_heights=256, camera_widths=256
            )
            env.seed(args.env_seed)
            actions: list[np.ndarray] = []
            selected_counts: list[int] = []
            error = None
            success = False
            initial_state_hash = None
            started = time.monotonic()
            try:
                env.reset()
                observation = env.set_init_state(initial_states[args.trial])
                for _ in range(args.settle_steps):
                    observation, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                initial_state = np.asarray(env.get_sim_state()).copy()
                initial_state_hash = hashlib.sha256(initial_state.tobytes()).hexdigest()
                action_queue: deque[np.ndarray] = deque()
                replan_index = 0
                policy_steps = 0
                while policy_steps < max_steps and not success:
                    if not action_queue:
                        base_image = np.ascontiguousarray(
                            observation["agentview_image"][::-1, ::-1]
                        )
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
                        request_seed = (
                            args.noise_seed
                            + task_index * 1_000_000
                            + args.trial * 10_000
                            + replan_index
                        )
                        request = {
                            "observation/image": base_image,
                            "observation/wrist_image": wrist_image,
                            "observation/state": state,
                            "prompt": instruction,
                            "pi05_arm": arm,
                            "pi05_noise_seed": request_seed,
                        }
                        if trace_dir is not None:
                            request["pi05_trace_candidates"] = True
                        if arm == "matched":
                            if args.guidance_scale is not None:
                                request["pi05_guidance_scale"] = args.guidance_scale
                            if args.layer_index is not None:
                                request["pi05_layer_index"] = args.layer_index
                            if args.active_steps is not None:
                                request["pi05_active_steps"] = args.active_steps
                        if args.num_steps is not None:
                            request["pi05_num_steps"] = args.num_steps
                        response = client.infer(request)
                        chunk = np.asarray(response["actions"], dtype=np.float32)
                        if chunk.shape != (10, 7) or not np.isfinite(chunk).all():
                            raise RuntimeError(f"invalid action chunk {chunk.shape}")
                        if trace_dir is not None:
                            replay = response.pop("pi05_vtcg_candidates", None)
                            if replay is None:
                                raise RuntimeError("trace mode response omitted VTCG replay data")
                            trace_path = trace_dir / (
                                f"task{task_index:04d}_trial{args.trial}_{arm}_"
                                f"replan{replan_index:04d}.npz"
                            )
                            if trace_path.exists():
                                raise FileExistsError(trace_path)
                            trace_payload = {
                                "suite": np.asarray(args.suite),
                                "task_index": np.asarray(task_index, dtype=np.int32),
                                "task_name": np.asarray(task.name),
                                "trial": np.asarray(args.trial, dtype=np.int32),
                                "arm": np.asarray(arm),
                                "replan_index": np.asarray(replan_index, dtype=np.int32),
                                "noise_seed": np.asarray(request_seed, dtype=np.int64),
                                "prompt": np.asarray(instruction),
                                "image": base_image,
                                "wrist_image": wrist_image,
                                "state": state,
                                "action_chunk": chunk,
                                "initial_state_sha256": np.asarray(initial_state_hash),
                                "image_sha256": np.asarray(
                                    hashlib.sha256(base_image.tobytes()).hexdigest()
                                ),
                                "wrist_image_sha256": np.asarray(
                                    hashlib.sha256(wrist_image.tobytes()).hexdigest()
                                ),
                                "state_sha256": np.asarray(
                                    hashlib.sha256(state.tobytes()).hexdigest()
                                ),
                                "prompt_sha256": np.asarray(
                                    hashlib.sha256(instruction.encode("utf-8")).hexdigest()
                                ),
                                "trace_json": np.asarray(
                                    json.dumps(
                                        response.get("pi05_paired_trace", {}),
                                        sort_keys=True,
                                    )
                                ),
                            }
                            trace_payload.update(
                                {f"vtcg_{key}": np.asarray(value) for key, value in replay.items()}
                            )
                            np.savez_compressed(trace_path, **trace_payload)
                        action_queue.extend(chunk[: args.replan_steps])
                        trace = response.get("pi05_paired_trace", {})
                        if "selected_token_count" in trace:
                            selected_counts.append(int(trace["selected_token_count"]))
                        replan_index += 1
                    action = np.asarray(action_queue.popleft(), dtype=np.float32)
                    observation, _, success, _ = env.step(action.tolist())
                    actions.append(action.copy())
                    policy_steps += 1
                success = bool(success or env.check_success())
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
            finally:
                env.close()

            action_array = np.asarray(actions, dtype=np.float32)
            record = {
                "protocol": "PI05_LIBERO_PLUS_PAIRED_V1",
                "suite": args.suite,
                "task_index": task_index,
                "task_name": task.name,
                "instruction": instruction,
                "instruction_raw": str(task.language),
                "prompt_mode": args.prompt_mode,
                "trial": args.trial,
                "arm": arm,
                "env_seed": args.env_seed,
                "base_noise_seed": args.noise_seed,
                "settle_steps": args.settle_steps,
                "replan_steps": args.replan_steps,
                "max_steps": max_steps,
                "guidance_scale_requested": args.guidance_scale,
                "layer_index_requested": args.layer_index,
                "active_steps_requested": args.active_steps,
                "num_steps_requested": args.num_steps,
                "success": bool(success),
                "policy_steps": len(actions),
                "replans": int(math.ceil(len(actions) / args.replan_steps)) if actions else 0,
                "runtime_seconds": time.monotonic() - started,
                "initial_state_sha256": initial_state_hash,
                "action_sha256": hashlib.sha256(action_array.tobytes()).hexdigest(),
                "actions_finite": bool(not action_array.size or np.isfinite(action_array).all()),
                "mean_selected_token_count": (
                    float(np.mean(selected_counts)) if selected_counts else None
                ),
                "error": error,
            }
            append_jsonl(result_path, record)
            print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
