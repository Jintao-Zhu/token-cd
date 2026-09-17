"""Closed-loop Target-Diff / Target-Boost / Reverse workers."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time

import imageio.v2 as imageio
import numpy as np

from research.semantic_token_cd.distractor_rollout import jsonable, restore_snapshot, snapshot_sha
from research.semantic_token_cd.target_specific_common import build_policy as build_difference_policy
from research.semantic_token_cd.target_specific_protocol import (
    ARTIFACT as DIFFERENCE_ARTIFACT, CANONICAL, NEW_ARMS as DIFFERENCE_ARMS, PCD_SOURCE,
    PROTOCOL as DIFFERENCE_PROTOCOL, TASKS, atomic_json, selector_config as difference_config,
)
from research.semantic_token_cd.target_positive_boost_common import build_policy as build_boost_policy
from research.semantic_token_cd.target_positive_boost_protocol import (
    ARTIFACT as BOOST_ARTIFACT, NEW_ARMS as BOOST_ARMS, PROTOCOL as BOOST_PROTOCOL,
    selector_config as boost_config,
)
from research.semantic_token_cd.xswap_rollout import make_environment


def parse_seeds(spec: str) -> list[int]:
    output = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = map(int, part.split("-", 1)); output.extend(range(lo, hi + 1))
        elif part.strip(): output.append(int(part))
    return sorted(set(output))


def run_loop(env, policy, instruction, obs, video_path):
    from parallel_inference import get_image_from_maniskill2_obs_dict
    from research.semantic_token_cd.rollout_pilot import flatten_action
    from utils import convert_numpy_or_torch_to_python, summarize
    image = get_image_from_maniskill2_obs_dict(env, obs); infos, actions, frames = [], [], []
    predicted = truncated = False; control = 0
    while not (predicted or truncated) and control < 140:
        frames.append(np.asarray(image, dtype=np.uint8))
        _raw, batch, _meta = policy.step(image, None, instruction, proprio=obs["agent"]["eef_pos"])
        if not isinstance(batch, list): batch = [batch]
        for action in batch:
            executed = flatten_action(action)
            if executed.shape != (7,) or not np.isfinite(executed).all(): raise FloatingPointError("invalid action")
            actions.append(executed.copy()); obs, _reward, _success, truncated, info = env.step(executed)
            image = get_image_from_maniskill2_obs_dict(env, obs); control += 1
            infos.append(convert_numpy_or_torch_to_python(info)); predicted = bool(action["terminate_episode"][0] > 0)
            if predicted and not env.unwrapped.is_final_subtask():
                predicted = False; env.advance_to_next_subtask()
    frames.append(np.asarray(image, dtype=np.uint8)); video_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(video_path, frames, fps=10, codec="libx264", quality=7, macro_block_size=None)
    result = summarize(infos); result["failure_reason"] = None if result.get("success") else ("time_limit" if truncated else "policy_terminated")
    return result, np.asarray(actions, dtype=np.float32), infos


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seeds", required=True); parser.add_argument("--arm", choices=DIFFERENCE_ARMS + BOOST_ARMS, required=True)
    parser.add_argument("--gpu", type=int, required=True); parser.add_argument("--worker-id", default="manual")
    args = parser.parse_args(); os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    boost = args.arm in BOOST_ARMS
    artifact = BOOST_ARTIFACT if boost else DIFFERENCE_ARTIFACT
    protocol = BOOST_PROTOCOL if boost else DIFFERENCE_PROTOCOL
    env, env_id = make_environment(args.task, args.gpu); checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    base = OpenVLAInference(**get_policy_config("openvla", checkpoint, args.task, {}, False))
    for seed in parse_seeds(args.seeds):
        out = artifact / "closed_loop/episodes" / args.task / args.arm
        summary_path = out / f"episode_{seed:03d}_summary.json"; arrays_path = out / f"episode_{seed:03d}_arrays.npz"
        video_path = artifact / "closed_loop/videos" / args.task / args.arm / f"episode_{seed:03d}.mp4"
        if summary_path.exists() and arrays_path.exists() and video_path.exists(): continue
        with (CANONICAL / "snapshots" / args.task / f"seed_{seed:03d}.pkl").open("rb") as handle: snapshot = pickle.load(handle)
        obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot); instruction = env.unwrapped.get_language_instruction()
        policy = (build_boost_policy if boost else build_difference_policy)(base, args.task, args.arm)
        cfg = (boost_config if boost else difference_config)(args.task, args.arm)
        policy.reset(instruction, seed=seed); policy._episode_seed = seed; policy._selector_step = 0
        policy._episode_trace = []; policy._episode_logits = []
        started = time.monotonic(); result, actions, infos = run_loop(env, policy, instruction, obs, video_path)
        trace = policy._episode_trace; runtime = time.monotonic() - started
        checks = {
            "all_feature_equal": all(x.get("feature_equal") is True for x in trace),
            "all_guided_prefix": all(x.get("guided_prefix") is True for x in trace),
            "all_coverage_exact": all(x.get("coverage_exact") is True for x in trace),
            "all_non_target_equal": all(x.get("non_target_bit_identical") is True for x in trace),
            "all_layer_11": all(x.get("attention_layers") == [11] for x in trace),
            "all_real_instruction_decode": all(x.get("instruction") == instruction for x in trace),
            "all_generic_locked": all(x.get("selector_contrast_instruction") == cfg["contrast_instruction"] for x in trace),
            "all_formula_locked": all(x.get("selector_difference_formula") == cfg["formula"] for x in trace),
            "all_lambda_0p5": all(abs(float(x.get("lambda")) - .5) < 1e-12 for x in trace),
        }
        checks["technical_pass"] = bool(trace) and all(checks.values())
        if not checks["technical_pass"]: raise RuntimeError(f"audit failed: {checks}")
        payload = {"executed_actions": actions}
        for key in ("positive", "negative", "selected_mask", "reference_shr_mask", "correct_attention_probability",
                    "contrast_attention_probability", "attention_difference", "selector_score", "per_token_perturbation_norm"):
            if trace and all(key in x for x in policy._episode_logits): payload[key] = np.stack([x[key] for x in policy._episode_logits])
        arrays_path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(arrays_path, **payload)
        summary = {"protocol": protocol, "task": args.task, "seed": seed, "arm": args.arm,
            "instruction": instruction, "generic_instruction": cfg["contrast_instruction"], "formula": cfg["formula"],
            "success": bool(result.get("success", False)), "result": jsonable(result), "control_steps": len(infos),
            "runtime_seconds": runtime, "worker_id": args.worker_id, "environment_id": env_id,
            "initial_state_sha256": state_sha, "initial_rgb_sha256": rgb_sha,
            "canonical_snapshot_sha256": snapshot_sha(snapshot), "mean_m": float(np.mean([x["m_t"] for x in trace])),
            "mean_correct_retention": float(np.mean([x["correct_prompt_retention"] for x in trace])),
            "mean_feature_perturbation_norm": float(np.mean([x["feature_perturbation_norm"] for x in trace])),
            "mean_residual_norm": float(np.mean([x["centered_logit_residual_norm"] for x in trace])),
            "video_sha256": file_sha(video_path), **checks}
        atomic_json(summary_path, summary)
        print(json.dumps({"task": args.task, "seed": seed, "arm": args.arm,
                          "success": summary["success"], "seconds": round(runtime, 1)}), flush=True)
    env.close()


if __name__ == "__main__": main()
