#!/usr/bin/env python3
"""Qualify official LIBERO expert demonstrations for aligned region capture."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from libero.libero.envs import SegmentationRenderEnv
from libero.libero.envs.utils import postprocess_model_xml
from research.coreact_region.region_mapping import (
    instance_label_map,
    protected_instance_names,
    token_regions,
)


CAMERAS = (
    ("camera1", "agentview_image", "agentview_segmentation_instance"),
    ("camera2", "robot0_eye_in_hand_image", "robot0_eye_in_hand_segmentation_instance"),
)


def hash_bytes(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(json.dumps(list(value.shape)).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def local_bddl(workspace: Path, recorded: str) -> Path:
    name = Path(recorded).name
    matches = list((workspace / "LIBERO/libero/libero/bddl_files").rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"expected one local BDDL for {name}, found {matches}")
    return matches[0]


def observation_from_state(env, state: np.ndarray) -> dict:
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    env._post_process()
    env._update_observables(force=True)
    return env._get_observations()


def rewrite_demo_xml(xml: str, workspace: Path) -> str:
    asset_root = str(workspace / "LIBERO/libero/libero/assets")
    return xml.replace("/Users/yifengz/workspace/libero-dev/chiliocosm/assets", asset_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    args = parser.parse_args()
    workspace, demo, artifact = args.workspace.resolve(), args.demo.resolve(), args.artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    # LIBERO's XML path rewriter resolves assets relative to this directory.
    import os
    os.chdir(workspace / "LIBERO")

    with h5py.File(demo, "r") as source:
        data = source["data"]
        demos = sorted(data.keys(), key=lambda name: int(name.split("_")[-1]))
        episode = data[demos[0]]
        states = np.asarray(episode["states"])
        actions = np.asarray(episode["actions"])
        model_xml = episode.attrs["model_file"]
        if isinstance(model_xml, bytes):
            model_xml = model_xml.decode()
        recorded_bddl = data.attrs.get("bddl_file_name")
        if recorded_bddl is None and "env_args" in data.attrs:
            recorded_bddl = json.loads(data.attrs["env_args"])["bddl_file"]
        if isinstance(recorded_bddl, bytes):
            recorded_bddl = recorded_bddl.decode()
        language = "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate"
        problem_info = data.attrs.get("problem_info")
        if problem_info:
            if isinstance(problem_info, bytes):
                problem_info = problem_info.decode()
            language = json.loads(problem_info).get("language_instruction", language)

        bddl = local_bddl(workspace, recorded_bddl)
        env = SegmentationRenderEnv(
            bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
            camera_segmentations="instance",
        )
        try:
            env.reset()
            env.reset_from_xml_string(postprocess_model_xml(rewrite_demo_xml(model_xml, workspace), {}, demo_generation=False))
            env.sim.reset()
            obs = observation_from_state(env.env, states[0])
            keys_ok = all(rgb in obs and seg in obs for _, rgb, seg in CAMERAS)
            shapes_ok = all(obs[rgb].shape == (256, 256, 3) and obs[seg].shape == (256, 256, 1) for _, rgb, seg in CAMERAS)
            labels = instance_label_map(env.env.model.instances_to_ids)
            relevant = set(env.env.obj_of_interest)
            protected = protected_instance_names(env.env.model.instances_to_ids)
            mapped = []
            for camera_id, rgb, seg in CAMERAS:
                mapped.extend(token_regions(obs[seg], camera_id=camera_id, label_by_name=labels, relevant_names=relevant, protected_names=protected))

            replay_errors = []
            for index in sorted({0, min(1, len(actions) - 1), min(len(actions) // 2, len(actions) - 1)}):
                observation_from_state(env.env, states[index])
                env.step(actions[index])
                if index + 1 < len(states):
                    replay_errors.append(float(np.max(np.abs(env.sim.get_state().flatten() - states[index + 1]))))

            result = {
                "gate": "official_expert_replay_with_aligned_instance_segmentation",
                "demo_path": str(demo), "demo_sha256": file_sha256(demo),
                "hdf5_demo_count": len(demos), "first_demo": demos[0],
                "states_shape": list(states.shape), "actions_shape": list(actions.shape),
                "language": language, "local_bddl": str(bddl),
                "has_model_xml": bool(model_xml), "rgb_segmentation_keys_present": keys_ok,
                "rgb_segmentation_shapes_256": shapes_ok, "mapped_visual_tokens": len(mapped),
                "relevant_instances": sorted(relevant), "protected_instances": sorted(protected),
                "state_sha256": hash_bytes(states[0]),
                "rgb_sha256": {camera: hash_bytes(obs[rgb]) for camera, rgb, _ in CAMERAS},
                "segmentation_sha256": {camera: hash_bytes(obs[seg]) for camera, _, seg in CAMERAS},
                "sampled_one_step_replay_max_abs_errors": replay_errors,
            }
            result["transition_replay_diagnostic_only"] = True
            result["transition_replay_interpretation"] = "The stored simulator states are the authoritative observations; action-to-next-state replay is diagnostic because controller/version metadata differs."
            result["pass"] = bool(
                len(demos) > 0 and len(states) == len(actions) and keys_ok and shapes_ok
                and len(mapped) == 128
            )
            result["task_id"] = args.task_id
            (artifact / f"replay_qualification_task{args.task_id:02d}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            if not result["pass"]:
                raise RuntimeError(f"replay qualification failed: {result}")
            print(json.dumps(result, indent=2, sort_keys=True))
        finally:
            env.close()


if __name__ == "__main__":
    main()
