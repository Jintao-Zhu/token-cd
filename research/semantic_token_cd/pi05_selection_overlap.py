"""What does the matched selector actually land on?  (target vs added-distractor tokens)

For each LIBERO-Plus _add_N instance we ask the pi0.5 server for the 18 all-layer
attention maps (pi05_arm=capture) and the matched budget m_t (pi05_arm=matched,
lambda=0), reproduce the selector exactly (stable_top_m over the chosen layer --
which is precisely what Pi05MatchedFlowCDPolicy._select_negative does), and then
measure which objects those m_t grid cells fall on, using the ground-truth
instance segmentation of the very same frame:

  target_frac      selected cells on objects the instruction names
  distractor_frac  selected cells on objects that only exist in the _add_ scene
  other_frac       everything else (fixtures, table, robot arm, empty space)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

GRID = 16
N_VISUAL = GRID * GRID


def axis_angle(quaternion):
    value = np.asarray(quaternion, dtype=np.float64).copy()
    value[3] = np.clip(value[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - value[3] * value[3]))
    if math.isclose(denominator, 0.0):
        return np.zeros((3,), dtype=np.float64)
    return value[:3] * 2.0 * math.acos(value[3]) / denominator


def object_phrase(instance):
    return " ".join(part for part in instance.split("_") if not part.isdigit())


def resize_mask(mask, grid=GRID):
    height, width = mask.shape
    block_h, block_w = height // grid, width // grid
    trimmed = mask[: block_h * grid, : block_w * grid]
    return trimmed.reshape(grid, block_h, grid, block_w).mean(axis=(1, 3)) > 0.35


def stable_top_m(scores, count):
    order = np.lexsort((np.arange(N_VISUAL), -scores))
    return [int(index) for index in order[: int(count)]]


def bddl_objects(path):
    text = path.read_text(errors="ignore")
    match = re.search(r"\(:objects(.*?)\n\s*\)", text, re.S)
    names = []
    for line in (match.group(1).splitlines() if match else []):
        line = line.strip()
        if not line:
            continue
        left, _, _ = line.partition(" - ")
        names.extend(left.split())
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8125)
    parser.add_argument("--instances-per-task", type=int, default=3)
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--tasks", default="")
    parser.add_argument("--env-seed", type=int, default=7)
    parser.add_argument("--settle-steps", type=int, default=10)
    args = parser.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import SegmentationRenderEnv
    from openpi_client import image_tools, websocket_client_policy

    args.artifact.mkdir(parents=True, exist_ok=True)
    out_path = args.artifact / "selection_overlap.jsonl"
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    suite = benchmark.get_benchmark_dict()["libero_10"](0)
    folder = Path(get_libero_path("bddl_files")) / "libero_10"

    by_base = {}
    for index in range(suite.n_tasks):
        task = suite.get_task(index)
        match = re.match(r"(.+)_add_(\d+)$", task.name)
        if not match:
            continue
        by_base.setdefault(match.group(1), []).append((index, task.name))
    bases = sorted(by_base)
    if args.tasks:
        wanted = {p.strip() for p in args.tasks.split(",") if p.strip()}
        bases = [b for b in bases if any(w.lower() in b.lower() for w in wanted)]
    print("base tasks with _add_ instances:", len(bases), flush=True)

    for base in bases:
        entries = sorted(by_base[base], key=lambda it: int(it[1].rsplit("_add_", 1)[1]))
        base_objects = set(bddl_objects(folder / (base + ".bddl")))
        for index, name in entries[: args.instances_per_task]:
            task = suite.get_task(index)
            added = [o for o in bddl_objects(folder / (name + ".bddl")) if o not in base_objects]
            init_states = suite.get_task_init_states(index)
            env = SegmentationRenderEnv(
                bddl_file_name=str(folder / (name + ".bddl")),
                camera_heights=256,
                camera_widths=256,
            )
            env.seed(args.env_seed)
            env.reset()
            try:
                observation = env.set_init_state(init_states[0])
                for _ in range(args.settle_steps):
                    observation, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                run_instance(env, observation, task, index, name, base, added, args, client, image_tools, out_path)
            finally:
                env.close()


def run_instance(env, observation, task, index, name, base, added, args, client, image_tools, out_path):
    segmentation = np.asarray(observation["agentview_segmentation_instance"])
    segmentation = segmentation[..., 0] if segmentation.ndim == 3 else segmentation
    segmentation = segmentation[::-1, ::-1]
    instances = sorted(env.instance_to_id)
    masks = {inst: resize_mask(segmentation == env.instance_to_id[inst]) for inst in instances}

    base_image = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
    wrist_image = np.ascontiguousarray(observation["robot0_eye_in_hand_image"][::-1, ::-1])
    base_image = image_tools.convert_to_uint8(image_tools.resize_with_pad(base_image, 224, 224))
    wrist_image = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_image, 224, 224))
    state = np.concatenate(
        (
            observation["robot0_eef_pos"],
            axis_angle(observation["robot0_eef_quat"]),
            observation["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    instruction = str(getattr(task, "language", "") or "")

    common = {
        "observation/image": base_image,
        "observation/wrist_image": wrist_image,
        "observation/state": state,
        "prompt": instruction,
    }
    capture = client.infer(dict(common, **{"pi05_arm": "capture"}))
    scores = np.asarray(capture["attention_scores"], dtype=np.float32)
    budget = client.infer(
        dict(
            common,
            **{
                "pi05_arm": "matched",
                "pi05_guidance_scale": 0.0,
                "pi05_layer_index": args.layer,
                "pi05_noise_seed": 0,
            },
        )
    )
    trace = budget.get("pi05_paired_trace", {})
    m_t = int(trace.get("selected_token_count") or 0)
    chosen = stable_top_m(scores[args.layer], m_t)

    text = instruction.lower()
    target_instances = [
        inst for inst in instances
        if object_phrase(inst).lower() and object_phrase(inst).lower() in text
    ]
    target_mask = np.zeros((GRID, GRID), dtype=bool)
    for inst in target_instances:
        target_mask |= masks[inst]
    dist_mask = np.zeros((GRID, GRID), dtype=bool)
    for inst in added:
        if inst in masks:
            dist_mask |= masks[inst]

    cells = np.zeros(N_VISUAL, dtype=bool)
    cells[chosen] = True
    flat_target = target_mask.reshape(-1)
    flat_dist = dist_mask.reshape(-1)
    on_target = int(cells[flat_target].sum())
    on_dist = int(cells[flat_dist].sum())
    record = {
        "base_task": base,
        "instance": name,
        "task_index": index,
        "instruction": instruction,
        "layer": args.layer,
        "m_t": m_t,
        "selected": chosen,
        "target_instances": target_instances,
        "added_instances": added,
        "on_target": on_target,
        "on_distractor": on_dist,
        "other": m_t - on_target - on_dist,
        "target_frac": on_target / max(m_t, 1),
        "distractor_frac": on_dist / max(m_t, 1),
        "other_frac": (m_t - on_target - on_dist) / max(m_t, 1),
        "target_mask_cells": int(flat_target.sum()),
        "distractor_mask_cells": int(flat_dist.sum()),
    }
    with out_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(
        "%-54s m_t=%3d target=%.2f distractor=%.2f added=%d"
        % (name[:54], m_t, record["target_frac"], record["distractor_frac"], len(added)),
        flush=True,
    )


if __name__ == "__main__":
    main()
