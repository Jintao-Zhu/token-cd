#!/usr/bin/env python3
"""Capture target-free value-norm-weighted attention on locked expert states."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation

from research.coreact_closed_loop.runtime import env_config, load_policy_and_processors
from research.coreact_exploration.instrumentation import attention_ranking_scores, predict_teacher_forced_velocity
from research.coreact_exploration.run_development_gates import CAMERA_IDS
from research.coreact_region.segmented_runtime import batched_observation, make_segmented_env
from research.coreact_revision.run_region_sign_capture_v3 import NOISE_SEEDS, TAUS, action_chunk, rewrite_demo_xml


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append(path: Path, row: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--workspace", type=Path, required=True); p.add_argument("--source-artifact", type=Path, required=True); p.add_argument("--output", type=Path, required=True); args = p.parse_args()
    workspace, source, output = args.workspace.resolve(), args.source_artifact.resolve(), args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_rows = read_jsonl(source / "state_manifest.jsonl")
    unique = {(r["task_id"], r["demo_id"], r["frame_id"]): r for r in manifest_rows}
    if len(unique) != 300:
        raise RuntimeError(f"expected 300 unique locked states, got {len(unique)}")
    completed = set()
    if output.exists():
        completed = {(r["task_id"], r["demo_id"], r["frame_id"], r["group_id"]) for r in read_jsonl(output)}
    task_names = {row["task_id"]: row["name"] for row in json.loads((source / "task_manifest.json").read_text())}
    config, policy, preprocessor, _ = load_policy_and_processors(workspace)
    dataset = workspace / "LIBERO/libero/datasets/libero_spatial"
    for task_id in sorted(task_names):
        task_rows = [r for key, r in sorted(unique.items()) if key[0] == task_id]
        env_preprocessor, _ = make_env_pre_post_processors(env_cfg=env_config("libero_spatial", task_id), policy_cfg=config)
        env = make_segmented_env("libero_spatial", task_id)
        demo_path = dataset / f"{task_names[task_id]}_demo.hdf5"
        try:
            with h5py.File(demo_path, "r") as source_h5:
                by_demo = defaultdict(list)
                for row in task_rows: by_demo[row["demo_id"]].append(row)
                for demo_id in sorted(by_demo, key=lambda name: int(name.split("_")[-1])):
                    episode = source_h5["data"][demo_id]; states = np.asarray(episode["states"]); actions = np.asarray(episode["actions"])
                    xml = episode.attrs["model_file"]; xml = xml.decode() if isinstance(xml, bytes) else xml
                    env._env.reset(); env._env.reset_from_xml_string(__import__("libero.libero.envs.utils", fromlist=["postprocess_model_xml"]).postprocess_model_xml(rewrite_demo_xml(xml, workspace), {}, demo_generation=False)); env._env.env.sim.reset()
                    for row in sorted(by_demo[demo_id], key=lambda item: item["frame_id"]):
                        frame = row["frame_id"]
                        if all((task_id, demo_id, frame, f"prefix-{index}") in completed for index in row["selected_groups"]): continue
                        raw = env._env.regenerate_obs_from_state(states[frame]); observation = env._format_raw_obs(raw)
                        actions_chunk, pad = action_chunk(actions, frame)
                        batch = preprocess_observation(batched_observation(observation)); batch["task"] = [row["language"]]
                        batch["action"] = actions_chunk.unsqueeze(0); batch["action_is_pad"] = pad.unsqueeze(0); batch = preprocessor(env_preprocessor(batch))
                        images, masks = policy.prepare_images(batch); state = policy.prepare_state(batch); prepared_actions = policy.prepare_action(batch)
                        scores = []
                        for tau_value in TAUS:
                            for noise_seed in NOISE_SEEDS:
                                generator = torch.Generator(device=prepared_actions.device).manual_seed(noise_seed)
                                noise = torch.randn(prepared_actions.shape, generator=generator, device=prepared_actions.device, dtype=prepared_actions.dtype)
                                tau = torch.tensor([tau_value], device=prepared_actions.device, dtype=prepared_actions.dtype)
                                with torch.inference_mode():
                                    base = predict_teacher_forced_velocity(policy.model, images, masks, batch["observation.language.tokens"], batch["observation.language.attention_mask"], state, prepared_actions, tau, noise, record_attention=True, camera_ids=CAMERA_IDS)
                                score = attention_ranking_scores(base["attention_trace"], len(base["prefix_span_map"][0]))
                                if "late_half_value_weighted_attention" not in score: raise RuntimeError("value-weighted trace is absent")
                                scores.append(score)
                        late = torch.stack([s["late_half_value_weighted_attention"] for s in scores])
                        last = torch.stack([s["raw_last_expert_layer_value_weighted_attention"] for s in scores])
                        for index in row["selected_groups"]:
                            key = (task_id, demo_id, frame, f"prefix-{index}")
                            if key in completed: continue
                            append(output, {"task_id": task_id, "demo_id": demo_id, "frame_id": frame, "group_id": key[-1], "late_half_value_weighted_attention": float(late[:, index].median()), "late_half_value_weighted_attention_std": float(late[:, index].std(unbiased=False)), "raw_last_value_weighted_attention": float(last[:, index].median()), "flow_repetitions": 6})
                        print(f"task={task_id} {demo_id} frame={frame} complete", flush=True)
        finally:
            env.close()
    print(f"captured {len(read_jsonl(output))} group scores", flush=True)


if __name__ == "__main__": main()
