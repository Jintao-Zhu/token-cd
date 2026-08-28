#!/usr/bin/env python3
"""Capture same-state expert-flow effects and simulator region features."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation

from research.coreact_closed_loop.runtime import env_config, load_policy_and_processors
from research.coreact_exploration.instrumentation import (
    attention_ranking_scores,
    grouped_intervention,
    predict_teacher_forced_velocity,
)
from research.coreact_exploration.metrics import intervention_metrics
from research.coreact_exploration.run_development_gates import CAMERA_IDS, replacements_for
from research.coreact_region.region_mapping import attach_prefix_indices, instance_label_map, protected_instance_names
from research.coreact_region.segmented_runtime import batched_observation, make_segmented_env, raw_observation


TAUS = (0.2, 0.5, 0.8)
NOISE_SEEDS = (1729, 9473)
SELECTION_SEED = 20260808
STATES_PER_DEMO = 10
DEMO_COUNT = 6
GROUPS_PER_STATE = 20


def hash_array(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode()); h.update(str(value.shape).encode()); h.update(value.tobytes())
    return h.hexdigest()


def append(path: Path, row: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def action_chunk(actions: np.ndarray, index: int, size: int = 50) -> tuple[torch.Tensor, torch.Tensor]:
    valid = min(size, len(actions) - index)
    if valid <= 0:
        raise ValueError("state has no demonstration action")
    values = list(actions[index : index + valid])
    values.extend([values[-1]] * (size - valid))
    pad = [False] * valid + [True] * (size - valid)
    return torch.tensor(np.asarray(values), dtype=torch.float32), torch.tensor(pad, dtype=torch.bool)


def rewrite_demo_xml(xml: str, workspace: Path) -> str:
    return xml.replace("/Users/yifengz/workspace/libero-dev/chiliocosm/assets", str(workspace / "LIBERO/libero/libero/assets"))


def region_rows(raw: dict, env, span_map: list[dict]) -> list[dict]:
    labels = instance_label_map(env._env.env.model.instances_to_ids)
    relevant = list(env._env.env.obj_of_interest)
    if len(relevant) < 2:
        raise RuntimeError(f"expected target and goal instances, got {relevant}")
    target_label, goal_label = labels[relevant[0]], labels[relevant[1]]
    protected_labels = {labels[name] for name in protected_instance_names(env._env.env.model.instances_to_ids)}
    regions = []
    for camera_id, key in (("camera1", "agentview_segmentation_instance"), ("camera2", "robot0_eye_in_hand_segmentation_instance")):
        seg = np.asarray(raw[key])[..., 0]
        for token in range(64):
            row, col = divmod(token, 8); cell = seg[row * 32 : (row + 1) * 32, col * 32 : (col + 1) * 32]
            target = float(np.mean(cell == target_label)); goal = float(np.mean(cell == goal_label))
            protected = float(np.isin(cell, list(protected_labels)).mean())
            background = float(np.mean(cell == 0)); ambiguous = float(max(0.0, 1 - target - goal - protected - background))
            regions.append({
                "camera_id": camera_id, "visual_token_index": token, "row": row, "column": col,
                "target_object_fraction": target, "goal_container_fraction": goal,
                "protected_fraction": protected, "background_fraction": background,
                "ambiguous_fraction": ambiguous,
            })
    class Region:
        def __init__(self, row): self.__dict__.update(row)
    # attach_prefix_indices expects dataclasses; do the strict equivalent here.
    lookup = {(r["camera_id"], r["visual_token_index"]): r for r in regions}
    output = []
    for token in span_map:
        if token["modality"] != "visual": continue
        row = dict(lookup[(token["camera_id"], token["visual_token_index"])])
        row["prefix_index"] = token["index"]; output.append(row)
    if len(output) != 128 or len({r["prefix_index"] for r in output}) != 128:
        raise RuntimeError("invalid 128-token region mapping")
    return output


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--workspace", type=Path, required=True); p.add_argument("--demo", type=Path, required=True); p.add_argument("--artifact", type=Path, required=True); p.add_argument("--task-id", type=int, required=True)
    args = p.parse_args(); workspace, demo, artifact = args.workspace.resolve(), args.demo.resolve(), args.artifact.resolve()
    import os
    os.chdir(workspace / "LIBERO")
    if not json.loads((artifact / f"replay_qualification_task{args.task_id:02d}.json").read_text())["pass"]:
        raise RuntimeError("replay qualification did not pass")
    task_id = args.task_id
    config, policy, preprocessor, _ = load_policy_and_processors(workspace)
    env_preprocessor, _ = make_env_pre_post_processors(env_cfg=env_config("libero_spatial", task_id), policy_cfg=config)
    means = torch.load(workspace / "artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt", weights_only=True, map_location="cpu")
    env = make_segmented_env("libero_spatial", task_id)
    effects_path, states_path = artifact / "raw_effects.jsonl", artifact / "state_manifest.jsonl"
    completed = set()
    if effects_path.exists():
        for row in map(json.loads, effects_path.read_text().splitlines()):
            completed.add((row["task_id"], row["demo_id"], row["frame_id"], row["tau"], row["noise_seed"], row["group_id"]))
    with h5py.File(demo, "r") as source:
        demos = sorted(source["data"].keys(), key=lambda name: int(name.split("_")[-1]))[:DEMO_COUNT]
        info = source["data"].attrs.get("problem_info")
        info = info.decode() if isinstance(info, bytes) else info
        language = json.loads(info)["language_instruction"]
        for demo_ordinal, demo_id in enumerate(demos):
            episode = source["data"][demo_id]; states = np.asarray(episode["states"]); actions = np.asarray(episode["actions"])
            model_xml = episode.attrs["model_file"]; model_xml = model_xml.decode() if isinstance(model_xml, bytes) else model_xml
            env._env.reset(); env._env.reset_from_xml_string(__import__("libero.libero.envs.utils", fromlist=["postprocess_model_xml"]).postprocess_model_xml(rewrite_demo_xml(model_xml, workspace), {}, demo_generation=False)); env._env.env.sim.reset()
            frame_ids = np.linspace(0, max(0, len(actions) - 2), STATES_PER_DEMO, dtype=int).tolist()
            for frame_id in frame_ids:
                raw = env._env.regenerate_obs_from_state(states[frame_id]); observation = env._format_raw_obs(raw)
                chunk, pad = action_chunk(actions, frame_id)
                batch = preprocess_observation(batched_observation(observation)); batch["task"] = [language]
                batch["action"] = chunk.unsqueeze(0); batch["action_is_pad"] = pad.unsqueeze(0)
                batch = preprocessor(env_preprocessor(batch))
                images, image_masks = policy.prepare_images(batch); state = policy.prepare_state(batch); prepared_actions = policy.prepare_action(batch)
                bases, scores = {}, []
                for tau_value in TAUS:
                    for noise_seed in NOISE_SEEDS:
                        generator = torch.Generator(device=prepared_actions.device).manual_seed(noise_seed)
                        noise = torch.randn(prepared_actions.shape, generator=generator, device=prepared_actions.device, dtype=prepared_actions.dtype)
                        tau = torch.tensor([tau_value], device=prepared_actions.device, dtype=prepared_actions.dtype)
                        with torch.inference_mode():
                            base = predict_teacher_forced_velocity(policy.model, images, image_masks, batch["observation.language.tokens"], batch["observation.language.attention_mask"], state, prepared_actions, tau, noise, record_attention=True, camera_ids=CAMERA_IDS)
                        bases[(tau_value, noise_seed)] = (base, noise, tau); scores.append(attention_ranking_scores(base["attention_trace"], len(base["prefix_span_map"][0])))
                span = bases[(TAUS[0], NOISE_SEEDS[0])][0]["prefix_span_map"][0]
                mapped = region_rows(raw, env, span); by_index = {r["prefix_index"]: r for r in mapped}
                aggregate = torch.stack([s["late_half_action_to_context_attention"] for s in scores]).median(0).values
                allowed = [r["prefix_index"] for r in mapped if r["protected_fraction"] == 0]
                top = sorted(allowed, key=lambda i: (-float(aggregate[i]), i))[:10]
                rng = np.random.default_rng(SELECTION_SEED + task_id * 100000 + demo_ordinal * 1000 + frame_id)
                remaining = [i for i in allowed if i not in top]; uniform = rng.choice(remaining, size=10, replace=False).tolist()
                selected = top + uniform
                identity = {
                    "suite": "libero_spatial", "task_id": task_id, "demo_id": demo_id, "frame_id": frame_id, "language": language,
                    "initial_sim_state_sha256": hash_array(states[frame_id]),
                    "rgb_camera1_sha256": hash_array(raw["agentview_image"]), "rgb_camera2_sha256": hash_array(raw["robot0_eye_in_hand_image"]),
                    "segmentation_camera1_sha256": hash_array(raw["agentview_segmentation_instance"]), "segmentation_camera2_sha256": hash_array(raw["robot0_eye_in_hand_segmentation_instance"]),
                    "selected_groups": selected, "top_groups": top, "uniform_groups": uniform,
                }
                append(states_path, identity)
                for tau_value in TAUS:
                    for noise_seed in NOISE_SEEDS:
                        base, noise, tau = bases[(tau_value, noise_seed)]; action_dim = chunk.shape[-1]
                        valid = (~batch["action_is_pad"]).unsqueeze(-1).expand(-1, -1, action_dim)
                        for index in selected:
                            key = (task_id, demo_id, frame_id, tau_value, noise_seed, f"prefix-{index}")
                            if key in completed: continue
                            def intervene(prefix, maps, index=index):
                                return grouped_intervention(prefix, maps[0], [[index]], replacements_for(maps[0], means, mode="position"))
                            with torch.inference_mode():
                                neg = predict_teacher_forced_velocity(policy.model, images, image_masks, batch["observation.language.tokens"], batch["observation.language.attention_mask"], state, prepared_actions, tau, noise, prefix_intervention=intervene, record_attention=False, camera_ids=CAMERA_IDS)
                            metrics = intervention_metrics(
                                base["v_pred"][:, :, :action_dim].detach().cpu(),
                                neg["v_pred"][:, :, :action_dim].detach().cpu(),
                                base["u_target"][:, :, :action_dim].detach().cpu(),
                                valid.detach().cpu(),
                            )
                            append(effects_path, {**identity, **by_index[index], **metrics, "tau": tau_value, "noise_seed": noise_seed, "group_id": f"prefix-{index}", "selection_set": "top" if index in top else "uniform", "late_half_action_to_context_attention": float(aggregate[index]), "replacement_type": "position_conditioned_modality_mean", "finite": all(np.isfinite(v) for v in metrics.values())})
                print(f"{demo_id} frame={frame_id} complete", flush=True)
    env.close()
    (artifact / "capture_complete.json").write_text(json.dumps({"complete": True, "demos": DEMO_COUNT, "states": DEMO_COUNT * STATES_PER_DEMO, "groups_per_state": GROUPS_PER_STATE, "flow_conditions": len(TAUS) * len(NOISE_SEEDS)}, indent=2) + "\n")


if __name__ == "__main__": main()
