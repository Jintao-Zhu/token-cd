#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
from libero.libero.envs.utils import postprocess_model_xml

from research.coreact_expert_direction.direction_audit import action_chunk, rewrite_demo_xml
from research.coreact_region.segmented_runtime import make_segmented_env


K = 16
GROUPS = ("object_pose", "eef_pose", "gripper", "robot_state")


def canonical_quaternion(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64).copy()
    if value[0] < 0:
        value *= -1
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)))
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    anchors = [json.loads(line) for line in (artifact / "state_manifest.jsonl").read_text().splitlines()]
    os.chdir(workspace / "LIBERO")
    all_rows = []
    audit = {}
    for task_id in args.task_ids:
        task_anchors = [row for row in anchors if row["task_id"] == task_id]
        if not task_anchors:
            continue
        demo_path = Path(task_anchors[0]["demo_path"])
        env = make_segmented_env("libero_spatial", task_id)
        pool = []
        actions_by_demo = {}
        try:
            with h5py.File(demo_path, "r") as handle:
                demos = sorted(handle["data"], key=lambda value: int(value.split("_")[-1]))
                for demo_ordinal, demo_id in enumerate(demos):
                    episode = handle["data"][demo_id]
                    states = np.asarray(episode["states"])
                    actions = np.asarray(episode["actions"], dtype=np.float32)
                    obs = episode["obs"]
                    actions_by_demo[demo_id] = actions
                    xml = episode.attrs["model_file"]
                    xml = xml.decode() if isinstance(xml, bytes) else xml
                    env._env.reset()
                    env._env.reset_from_xml_string(postprocess_model_xml(rewrite_demo_xml(xml, workspace), {}, demo_generation=False))
                    inner = env._env.env
                    body_ids = [inner.sim.model.body_name2id(obj.root_body) for obj in inner.objects]
                    for frame, state in enumerate(states):
                        inner.sim.set_state_from_flattened(state)
                        inner.sim.forward()
                        objects = np.concatenate([np.concatenate([inner.sim.data.body_xpos[body].copy(), canonical_quaternion(inner.sim.data.body_xquat[body])]) for body in body_ids])
                        pool.append({
                            "demo_id": demo_id,
                            "demo_ordinal": demo_ordinal,
                            "frame": frame,
                            "remaining": len(actions) - frame,
                            "object_pose": objects,
                            "eef_pose": np.concatenate([np.asarray(obs["ee_pos"][frame]), np.asarray(obs["ee_ori"][frame])]).astype(np.float64),
                            "gripper": np.asarray(obs["gripper_states"][frame], dtype=np.float64),
                            "robot_state": np.asarray(obs["joint_states"][frame], dtype=np.float64),
                        })
                stats = {}
                for group in GROUPS:
                    values = np.stack([row[group] for row in pool])
                    stats[group] = {"mean": values.mean(0), "std": np.maximum(values.std(0), 1e-6)}
                for anchor in task_anchors:
                    anchor_pool = next(row for row in pool if row["demo_id"] == anchor["demo_id"] and row["frame"] == anchor["resolved_frame"])
                    valid_horizon = min(50, anchor["episode_length"] - anchor["resolved_frame"])
                    eligible = [row for row in pool if row["demo_id"] != anchor["demo_id"] and row["remaining"] >= valid_horizon]
                    for candidate in eligible:
                        candidate["distance"] = float(sum(np.mean(((candidate[group] - anchor_pool[group]) / stats[group]["std"]) ** 2) for group in GROUPS))
                    nearest_by_trajectory = {}
                    for row in sorted(eligible, key=lambda item: (item["distance"], item["demo_ordinal"], item["frame"])):
                        nearest_by_trajectory.setdefault(row["demo_id"], row)
                    selected = sorted(nearest_by_trajectory.values(), key=lambda row: (row["distance"], row["demo_ordinal"], row["frame"]))[:K]
                    if len(selected) != K:
                        raise RuntimeError(f"{anchor['state_id']}: only {len(selected)} eligible neighbors")
                    chunks = np.stack([action_chunk(actions_by_demo[row["demo_id"]], row["frame"])[0].numpy() for row in selected])
                    np.savez_compressed(artifact / "neighbors" / f"{anchor['state_id']}.npz", actions=chunks)
                    neighbor_rows = [{"neighbor_id": f"task{task_id:02d}__demo{row['demo_ordinal']:02d}__frame{row['frame']:04d}", "trajectory_id": row["demo_id"], "demo_ordinal": row["demo_ordinal"], "frame": row["frame"], "distance": row["distance"], "remaining": row["remaining"]} for row in selected]
                    all_rows.append({"state_id": anchor["state_id"], "task_id": task_id, "anchor_trajectory_id": anchor["demo_id"], "anchor_frame": anchor["resolved_frame"], "anchor_valid_horizon": valid_horizon, "neighbors": neighbor_rows})
                audit[str(task_id)] = {"pool_frames": len(pool), "demos": len(demos), "feature_dimensions": {group: int(len(pool[0][group])) for group in GROUPS}, "standardization": {group: {"mean": stats[group]["mean"].tolist(), "std": stats[group]["std"].tolist()} for group in GROUPS}}
                print(json.dumps({"task": task_id, "pool": len(pool), "anchors": len(task_anchors)}), flush=True)
        finally:
            env.close()
    existing = {row["state_id"]: row for row in [json.loads(line) for line in (artifact / "neighbor_manifest.jsonl").read_text().splitlines()]} if (artifact / "neighbor_manifest.jsonl").exists() else {}
    existing.update({row["state_id"]: row for row in all_rows})
    ordered = [existing[row["state_id"]] for row in anchors if row["state_id"] in existing]
    (artifact / "neighbor_manifest.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered))
    old_audit = json.loads((artifact / "retrieval_audit.json").read_text()) if (artifact / "retrieval_audit.json").exists() else {}
    old_audit.update(audit)
    (artifact / "retrieval_audit.json").write_text(json.dumps(old_audit, indent=2, sort_keys=True) + "\n")
    if len(ordered) == 250:
        (artifact / "status/current.json").write_text(json.dumps({"stage": "NEIGHBORS_READY", "states": 250}, indent=2) + "\n")


if __name__ == "__main__":
    main()
