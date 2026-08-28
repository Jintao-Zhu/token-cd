from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from research.token_pcd_stage_a.core import array_sha256, file_sha256, tensor_sha256


ARMS = ("vanilla", "pixel_pcd", "object_token_pcd", "random_token_pcd")


def clone(value):
    if isinstance(value, np.ndarray): return value.copy()
    if isinstance(value, dict): return type(value)((key, clone(item)) for key, item in value.items())
    if isinstance(value, list): return [clone(item) for item in value]
    if isinstance(value, tuple): return tuple(clone(item) for item in value)
    if hasattr(value, "p") and hasattr(value, "q"):
        return type(value)(np.asarray(value.p).copy(), np.asarray(value.q).copy())
    return copy.deepcopy(value)


def wrapped_observation(env):
    obs = env.unwrapped.get_obs(); wrappers, current = [], env
    while hasattr(current, "env"):
        if hasattr(current, "observation"): wrappers.append(current)
        current = current.env
    for wrapper in reversed(wrappers): obs = wrapper.observation(obs)
    return obs


def capture_snapshot(env, seed):
    env.reset(seed=seed); inner = env.unwrapped
    return {"sim_state": np.asarray(inner.get_state()).copy(), "agent_state": clone(inner.agent.get_state()),
            "rng_state": clone(inner._episode_rng.get_state()), "instruction": inner.get_language_instruction()}


def restore_snapshot(env, seed, snapshot):
    env.reset(seed=seed); inner = env.unwrapped
    inner.set_state(snapshot["sim_state"].copy()); inner.agent.set_state(clone(snapshot["agent_state"]))
    inner._episode_rng.set_state(clone(snapshot["rng_state"])); inner._elapsed_steps = 0
    obs = wrapped_observation(env)
    return obs, array_sha256(np.asarray(inner.get_state())), array_sha256(__import__("parallel_inference").get_image_from_maniskill2_obs_dict(env, obs))


def snapshot_sha(snapshot):
    digest = hashlib.sha256(); digest.update(array_sha256(snapshot["sim_state"]).encode())
    digest.update(repr(snapshot["agent_state"]).encode()); digest.update(repr(snapshot["rng_state"]).encode())
    digest.update(snapshot["instruction"].encode()); return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd-root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--protocol-id", default="TOKEN_PCD_BRIDGE_CLOSED_LOOP_PILOT_R1")
    parser.add_argument("--intervention-location", choices=("late", "early"), default="late")
    parser.add_argument("--sentinel", action="store_true")
    args = parser.parse_args()
    pcd_root, artifact = args.pcd_root.resolve(), args.artifact.resolve()
    source = pcd_root / "source/PCD"; sys.path.insert(0, str(source)); sys.path.insert(0, str(pcd_root / "runner")); os.chdir(source)
    from audited_runner import AuditedParallelRunner, flatten_action, jsonable
    from contrast_utils.contrast_image_generator import ContrastImageGenerator
    from parallel_inference import get_image_from_maniskill2_obs_dict
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    from utils import convert_numpy_or_torch_to_python, stat_final, stat_first, summarize, tile_images, write_video
    from research.token_pcd_bridge.token_policy import FrozenTokenPCDInference

    seeds = [int(x) for x in args.seeds.split(",") if x]
    task_root = artifact / ("sentinel" if args.sentinel else "episodes") / args.task
    task_root.mkdir(parents=True, exist_ok=False)
    setup_dir = task_root / "_setup"; setup_dir.mkdir()
    runner = AuditedParallelRunner(gpu_id=0, arm="pixel_pcd", num_gpus=1, policy="openvla",
        checkpoint=str(source / "pretrained/openvla-7b"), task=args.task, result_root=str(task_root),
        n_trajs=len(seeds), contrast=True, opts=["by", "box_tracking", "alpha", "0.8"])
    runner.result_dir = str(setup_dir); runner._build_logger(); runner._set_gpu(0)
    env = runner._build_environment(False)
    mask_generator = ContrastImageGenerator(env, by="box_tracking", inpaint_mode="lama")
    config = get_policy_config("openvla", str(source / "pretrained/openvla-7b"), args.task, {}, False)
    vanilla = OpenVLAInference(**config)
    original_generate = vanilla.vla.generate
    generate_audit = []
    def locked_generate(*generate_args, **kwargs):
        if kwargs.get("max_new_tokens") != 7:
            raise RuntimeError(f"Expected max_new_tokens=7, got {kwargs.get('max_new_tokens')}")
        if kwargs.get("min_new_tokens") is not None:
            raise RuntimeError("min_new_tokens is forbidden")
        kwargs["eos_token_id"] = None
        output = original_generate(*generate_args, **kwargs)
        if isinstance(output, dict) or hasattr(output, "scores"):
            scores = output["scores"]
            if len(scores) != 7 or not all(torch.isfinite(score).all() for score in scores):
                raise RuntimeError("Generate did not return seven finite action logits")
            generate_audit.append({"generated_token_count": len(scores), "finite": True})
        return output
    vanilla.vla.generate = locked_generate
    if args.intervention_location == "early":
        mean_payload = torch.load(artifact / "frozen/early_patch_means.pt", map_location="cpu", weights_only=True)
    else:
        mean_payload = torch.load(artifact / "frozen/position_conditioned_visual_mean.pt", map_location="cpu", weights_only=True)
    # Copy only lightweight wrapper state. All four arms share one frozen 7B model and processor.
    from research.token_pcd_bridge.early_token_policy import FrozenEarlyTokenPCDInference
    policy_class = FrozenEarlyTokenPCDInference if args.intervention_location == "early" else FrozenTokenPCDInference
    object_policy = copy.copy(vanilla); object_policy.__class__ = policy_class
    object_policy.alpha = 0.8
    if args.intervention_location == "early": object_policy.replacement_means = mean_payload["means"]
    else: object_policy.replacement_mean = mean_payload["mean"]
    object_policy.branch = "object_token_pcd"; object_policy._audit_context = None; object_policy._audit_calls = []
    random_policy = copy.copy(vanilla); random_policy.__class__ = policy_class
    random_policy.alpha = 0.8
    if args.intervention_location == "early": random_policy.replacement_means = mean_payload["means"]
    else: random_policy.replacement_mean = mean_payload["mean"]
    random_policy.branch = "random_token_pcd"; random_policy._audit_context = None; random_policy._audit_calls = []
    # Pixel wrapper also shares weights; no extra checkpoint load occurs.
    from contrast_policies.openvla_contrast import OpenVLAContrastInference
    pixel_policy = copy.copy(vanilla); pixel_policy.__class__ = OpenVLAContrastInference; pixel_policy.alpha = 0.8

    manifests = []
    for seed in seeds:
        snapshot = capture_snapshot(env, seed); canonical = snapshot_sha(snapshot); initial = {}
        for arm in ARMS:
            arm_dir = task_root / arm; arm_dir.mkdir(exist_ok=True)
            summary_path = arm_dir / f"episode_{seed:03d}_summary.json"
            if summary_path.exists(): raise FileExistsError(summary_path)
            obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot); initial[arm] = (state_sha, rgb_sha)
            instruction = env.unwrapped.get_language_instruction(); policy = {"vanilla": vanilla, "pixel_pcd": pixel_policy,
                "object_token_pcd": object_policy, "random_token_pcd": random_policy}[arm]
            policy.reset(instruction, seed=seed); mask_generator.reset(); policy._audit_generate_calls = []
            if arm in ("object_token_pcd", "random_token_pcd"): policy._audit_calls = []
            generate_audit.clear()
            frames, records, step_infos = [], [], []; predicted_terminated = False; truncated = False; timestep = 0
            image = get_image_from_maniskill2_obs_dict(env, obs)
            while not (predicted_terminated or truncated):
                rgb_hash = array_sha256(image)
                if arm == "vanilla": raw_action, actions = policy.step(image, instruction, proprio=obs["agent"]["eef_pos"]); frames.append(image)
                elif arm == "pixel_pcd":
                    contrast = mask_generator.generate(obs, instruction); raw_action, actions, aux = policy.step(image, contrast, instruction, proprio=obs["agent"]["eef_pos"]); frames.append(tile_images([image, contrast]))
                else:
                    if instruction != mask_generator.task_description:
                        mask_generator.reset_mask_and_keep_object_names(instruction); mask_generator.task_description = instruction
                        from contrast_utils.mask_predictors import build_predictor
                        mask_generator.predictor = build_predictor("box_tracking"); mask_generator._set_points_or_boxes(obs)
                    mask, _ = mask_generator.get_mask_by_predictor(obs)
                    policy.set_counterfactual(mask, args.task, seed, timestep, rgb_hash)
                    raw_action, actions, aux = policy.step(image, None, instruction, proprio=obs["agent"]["eef_pos"]); frames.append(image)
                if not isinstance(actions, list): actions = [actions]
                for action in actions:
                    executed = flatten_action(action)
                    if not np.isfinite(executed).all(): raise FloatingPointError("Non-finite action")
                    obs, _, _, truncated, info = env.step(executed); image = get_image_from_maniskill2_obs_dict(env, obs)
                    timestep += 1; converted = convert_numpy_or_torch_to_python(info); step_infos.append(converted)
                    record = {"timestep": timestep, "pre_step_rgb_sha256": rgb_hash, "action": executed.tolist(), "finite": True}
                    record["generate_calls"] = generate_audit[-2:] if arm != "vanilla" else generate_audit[-1:]
                    if arm in ("object_token_pcd", "random_token_pcd"): record["token_audit"] = policy._audit_calls[-1]
                    records.append(record); predicted_terminated = bool(action["terminate_episode"][0] > 0)
                    if predicted_terminated and not env.unwrapped.is_final_subtask(): predicted_terminated = False; env.advance_to_next_subtask()
                    new_instruction = env.unwrapped.get_language_instruction()
                    if new_instruction != instruction: instruction = new_instruction
            result = summarize(step_infos); result.update(stat_first(step_infos)); result.update(stat_final(step_infos))
            summary = {"protocol_id": args.protocol_id, "intervention_location": args.intervention_location, "task": args.task, "seed": seed,
                "arm": arm, "success": bool(result["success"]), "control_steps": timestep,
                "canonical_snapshot_sha256": canonical, "initial_state_sha256": state_sha, "initial_rgb_sha256": rgb_sha,
                "all_actions_finite": True, "all_generate_calls_seven_tokens": all(call["generated_token_count"] == 7 for call in generate_audit),
                "generate_call_count": len(generate_audit), "result": jsonable(result)}
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
            (arm_dir / f"episode_{seed:03d}_trace.jsonl").write_text("".join(json.dumps(jsonable(x), sort_keys=True)+"\n" for x in records))
            write_video(frames, str(arm_dir / f"episode_{seed:03d}_success_{summary['success']}.gif"))
        if len(set(initial.values())) != 1: raise RuntimeError(f"Four-arm pairing failure seed {seed}: {initial}")
        for arm in ("object_token_pcd", "random_token_pcd"):
            trace = [json.loads(line) for line in (task_root / arm / f"episode_{seed:03d}_trace.jsonl").read_text().splitlines()]
            for trace_row in trace:
                audit = trace_row["token_audit"]
                if len(audit["object_token_ids"]) != len(audit["random_token_ids"]):
                    raise RuntimeError("Matched Random count mismatch")
                if set(audit["object_token_ids"]) & set(audit["random_token_ids"]):
                    raise RuntimeError("Matched Random intersects object tokens")
                expected = audit["object_token_ids"] if arm == "object_token_pcd" else audit["random_token_ids"]
                if audit["changed_indices"] != expected:
                    raise RuntimeError("Token changed-index audit mismatch")
        manifests.append({"seed": seed, "canonical_snapshot_sha256": canonical,
                          "initial_state_sha256": initial["vanilla"][0], "initial_rgb_sha256": initial["vanilla"][1],
                          "four_arm_exact_pairing": True, "within_state_matched_random_audit": True})
        print(json.dumps({"task": args.task, "seed": seed, "four_arms_complete": True}), flush=True)
    manifest = {"task": args.task, "seeds": seeds, "pairs": manifests, "all_four_arm_exact_pairing": True}
    (task_root / "pairing_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    main()
