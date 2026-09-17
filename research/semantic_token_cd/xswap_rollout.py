"""XSWAP-V1 closed-loop worker: 5 arms x canonical snapshots for the 4 tasks.

Arms:
  vanilla    - clean greedy OpenVLA (also emits offline same-state samples).
  correct    - L11 prompt-attention selection with the actual task instruction.
  paraphrase - same, attention extracted with a fixed a-priori paraphrase.
  swapped    - same, attention extracted with a fixed scene-determined wrong
               instruction (different drawer/object or source<->target swap).
  random     - matched-coverage random tokens (deterministic SeedSequence).

The actual task instruction used by positive/negative decoding is identical
across all arms.  Fail-closed audits raise on any technical violation.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from research.semantic_token_cd.distractor_rollout import (
    PCD_SOURCE, jsonable, restore_snapshot, snapshot_sha,
)
from research.semantic_token_cd.prompt_attn_shr_policy import (
    N_VISUAL, PromptAttentionSHRInference,
)
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common
from research.semantic_token_cd.xswap_protocol import (
    ARMS, ATTENTION_LAYERS, CANONICAL, EMIT_STEPS, LAMBDA, PROTOCOL, STATE_STEPS,
    TASKS,
    array_sha, atomic_json, resolve_present_phrases, swap_for_scene,
    instruction_set,
)

ACTUAL_ARM_SELECTOR = {"correct": "prompt_attention", "paraphrase": "prompt_attention",
                       "swapped": "prompt_attention", "random": "random_matched"}


def parse_seeds(spec: str) -> list[int]:
    seeds = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            seeds.extend(range(lo, hi + 1))
        elif part:
            seeds.append(int(part))
    result = sorted(set(seeds))
    if not result:
        raise ValueError("empty seeds")
    return result


def make_environment(task: str, gpu: int):
    """Canonical google-robot env with the Vulkan renderer pinned to this GPU."""
    import gymnasium as gym
    import simpler_env

    renderer_kwargs = {"device": "cuda:0", "offscreen_only": True}
    if task == "google_robot_pick_coke_can":
        return gym.make(
            "GraspSingleOpenedCokeCanDistractorInScene-v0",
            obs_mode="rgbd", prepackaged_config=True, distractor_config="less",
            renderer_kwargs=renderer_kwargs,
        ), "GraspSingleOpenedCokeCanDistractorInScene-v0"
    environment_id, base_kwargs = simpler_env.ENVIRONMENT_MAP[task]
    kwargs = dict(base_kwargs)
    kwargs.update({"prepackaged_config": True, "renderer_kwargs": renderer_kwargs})
    return gym.make(environment_id, obs_mode="rgbd", **kwargs), environment_id


def build_policy(base, task: str, arm: str):
    if arm == "vanilla":
        from research.semantic_token_cd.distractor_policy import AuditedVanillaInference
        policy = copy.copy(base)
        policy.__class__ = AuditedVanillaInference
        return policy
    policy = copy.copy(base)
    policy.__class__ = PromptAttentionSHRInference
    _init_common(policy, LAMBDA)
    policy.beta = 0.0
    policy.selector_mode = ACTUAL_ARM_SELECTOR[arm]
    policy.task_index = TASK_INDEX[task]
    policy.attention_layers = tuple(ATTENTION_LAYERS)
    policy.save_prompt_attention = True
    policy.selection_count = None
    policy.selection_top_p = None
    policy.region_token_ids = None
    return policy


def audit_trace(trace: list[dict], arm: str, expected_selector: str | None) -> dict:
    if not trace:
        raise RuntimeError(f"{arm} produced no per-step trace")
    checks = {
        "all_feature_equal": all(s.get("feature_equal") is True for s in trace),
        "all_guided_prefix": all(s.get("guided_prefix") is True for s in trace),
        "all_reconstruction_finite": all(s.get("reconstruction_finite") is True for s in trace),
        "all_lambda_locked": all(abs(float(s.get("lambda", -1.0)) - LAMBDA) < 1e-12 for s in trace),
        "all_beta_zero": all(abs(float(s.get("beta", 1.0))) < 1e-12 for s in trace),
        "all_coverage_exact": all(s.get("coverage_exact") is True for s in trace),
        "all_non_target_bit_identical": all(s.get("non_target_bit_identical") is True for s in trace),
        "all_guided_dimensions": all(len(s.get("centered_logit_residual_norm_per_dim", [])) == 6 for s in trace),
    }
    if arm in ("correct", "paraphrase", "swapped"):
        checks["all_layer_11"] = all(s.get("attention_layers") == list(ATTENTION_LAYERS) for s in trace)
        checks["all_query_indices_nonempty"] = all(s.get("prompt_multimodal_query_indices") for s in trace)
        checks["all_visual_keys_locked"] = all(s.get("visual_key_indices") == [1, 256] for s in trace)
        checks["all_post_softmax"] = all(s.get("attention_post_softmax") is True for s in trace)
        if expected_selector is not None:
            checks["all_selector_instruction_locked"] = all(
                s.get("selector_instruction") == expected_selector for s in trace
            )
    if arm == "correct":
        checks["all_selector_matches_task"] = all(s.get("selector_matches_task") is True for s in trace)
    if arm in ("paraphrase", "swapped"):
        checks["all_selector_differs_from_task"] = all(
            s.get("selector_matches_task") is False for s in trace
        )
    checks["technical_pass"] = all(checks.values())
    if not checks["technical_pass"]:
        raise RuntimeError(f"XSWAP audit failed for {arm}: {json.dumps(checks, indent=1)[:4000]}")
    return checks


def run_episode_loop(env, policy, arm: str, instruction: str, obs, emit_out: Path | None,
                     scene_key: str) -> tuple[dict, list[dict], np.ndarray, list[dict]]:
    from parallel_inference import get_image_from_maniskill2_obs_dict
    from research.semantic_token_cd.rollout_pilot import flatten_action
    from utils import convert_numpy_or_torch_to_python, summarize

    image = get_image_from_maniskill2_obs_dict(env, obs)
    predicted_terminated = truncated = False
    step_infos: list[dict] = []
    executed: list[np.ndarray] = []
    state_buffer: list[dict] = []
    control = 0
    while not (predicted_terminated or truncated) and control < 140:
        if arm == "vanilla" and emit_out is not None:
            state_buffer.append({
                "control": control, "image": np.asarray(image, dtype=np.uint8).copy(),
                "rgb_sha": array_sha(np.asarray(image, dtype=np.uint8)),
            })
        raw_action, actions, _aux = policy.step(
            image, None, instruction, proprio=obs["agent"]["eef_pos"]
        )
        if not isinstance(actions, list):
            actions = [actions]
        for action in actions:
            a7 = flatten_action(action)
            if a7.shape != (7,) or not np.isfinite(a7).all():
                raise FloatingPointError(f"invalid executed action ctrl={control}: {a7}")
            executed.append(a7.copy())
            obs, _reward, _success, truncated, info = env.step(a7)
            image = get_image_from_maniskill2_obs_dict(env, obs)
            control += 1
            step_infos.append(convert_numpy_or_torch_to_python(info))
            predicted_terminated = bool(action["terminate_episode"][0] > 0)
            if predicted_terminated and not env.unwrapped.is_final_subtask():
                predicted_terminated = False
                env.advance_to_next_subtask()
    if arm == "vanilla" and emit_out is not None and state_buffer:
        n = len(state_buffer)
        indices = sorted({int(round(f * (n - 1))) for f in EMIT_STEPS})
        for i in indices:
            rec = state_buffer[i]
            out = emit_out / scene_key
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / f"step_{rec['control']:04d}.npz", image=rec["image"])
            atomic_json(out / f"step_{rec['control']:04d}.json", {
                "protocol_id": PROTOCOL, "task": env.unwrapped.env_id if hasattr(env.unwrapped, "env_id") else None,
                "instruction": instruction, "seed": scene_seed(scene_key), "control_step": rec["control"],
                "rgb_sha256": rec["rgb_sha"], "buffer_index": i, "total_buffer": n,
            })
    trace = getattr(policy, "_episode_trace", None) or []
    result = summarize(step_infos)
    reason = None if result.get("success") else ("time_limit" if truncated else "policy_terminated")
    result["failure_reason"] = reason
    actions_array = np.asarray(executed, dtype=np.float32) if executed else np.zeros((0, 7), dtype=np.float32)
    return result, trace, actions_array, step_infos


def scene_seed(key: str) -> int:
    return int(key.split("_")[1])


def run_seed_arm(env, policy, arm, task, seed, snapshot, emit_root: Path | None):
    obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
    instruction = env.unwrapped.get_language_instruction()
    present = resolve_present_phrases(env)
    sel = instruction_set(instruction)
    if sel["kind"] in ("pick_object", "move_near"):
        sel = swap_for_scene(instruction, present)
    if sel["kind"] == "move_near" and sel.get("swapped_kind", "").startswith("rel_swap"):
        pass  # labelled swap; still a fixed deterministic relation swap
    expected_selector = None
    if arm == "correct":
        expected_selector = instruction
        policy.selector_instruction = None
    elif arm == "paraphrase":
        expected_selector = sel["paraphrase"]
        policy.selector_instruction = sel["paraphrase"]
    elif arm == "swapped":
        expected_selector = sel["swapped"]
        policy.selector_instruction = sel["swapped"]
    elif arm == "random":
        policy.selector_instruction = None
    policy._episode_trace = []
    policy._episode_logits = []
    policy._episode_seed = seed
    policy._selector_step = 0
    if hasattr(policy, "reset"):
        policy.reset(instruction, seed=seed)
    started = time.monotonic()
    result, trace, actions_array, step_infos = run_episode_loop(
        env, policy, arm, instruction, obs, emit_root, f"seed_{seed:03d}"
    )
    runtime = time.monotonic() - started
    checks = audit_trace(trace, arm, expected_selector) if arm != "vanilla" else {}
    n_steps = len(trace)
    summary = {
        "protocol_id": PROTOCOL, "task": task, "seed": seed, "arm": arm,
        "instruction": instruction, "success": bool(result.get("success", False)),
        "result": jsonable(result), "failure_reason": result.get("failure_reason"),
        "selector_instruction": expected_selector,
        "selector_plan_kind": sel.get("kind"),
        "swapped_kind": sel.get("swapped_kind"),
        "present_object_phrases": present,
        "control_steps": len(step_infos), "trace_len": n_steps,
        "runtime_seconds": runtime, "initial_state_sha256": state_sha,
        "initial_rgb_sha256": rgb_sha, "canonical_snapshot_sha256": snapshot_sha(snapshot),
        "lambda": LAMBDA, "beta": 0.0,
        "attention_layers": list(ATTENTION_LAYERS),
        "mean_m_t": float(np.mean([t.get("num_tokens", float("nan")) for t in trace])) if trace else None,
        "mean_feature_perturbation_norm": float(np.mean(
            [t.get("feature_perturbation_norm", float("nan")) for t in trace])) if trace else None,
        "mean_centered_logit_residual_norm": float(np.mean(
            [t.get("centered_logit_residual_norm", float("nan")) for t in trace])) if trace else None,
        "action_jitter_index": (
            float(np.linalg.norm(np.diff(actions_array, axis=0), axis=1).mean()) if len(actions_array) > 1 else 0.0
        ),
        "emitted_state_count": 0 if arm != "vanilla" else None,
        "arrays_n_steps": len(policy._episode_logits) if hasattr(policy, "_episode_logits") else 0,
        "selector_trace": jsonable(trace),
        **checks,
    }
    return summary, policy._episode_logits, actions_array


def write_arrays(path: Path, records: list[dict], actions: np.ndarray) -> None:
    payload = {"executed_actions": actions}
    keys = ("positive", "negative", "selected_mask", "reference_shr_mask",
            "prompt_attention", "per_token_perturbation_norm")
    if records:
        for key in keys:
            if all(key in record for record in records):
                try:
                    payload[key] = np.stack([record[key] for record in records])
                except Exception:
                    pass
    np.savez_compressed(path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--emit", action="store_true", help="emit offline same-state samples from vanilla")
    parser.add_argument("--worker-id", default="manual")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    from research.semantic_token_cd.xswap_protocol import ARTIFACT
    task = args.task
    arms = tuple(a for a in args.arms.split(",") if a.strip())
    if any(a not in ARMS for a in arms):
        raise ValueError(f"invalid arms {arms}")
    artifact = ARTIFACT / "runs"
    env, environment_id = make_environment(task, args.gpu)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, task, {}, False)
    base = OpenVLAInference(**config)
    policies = {a: build_policy(base, task, a) for a in arms}
    emit_root = artifact / "emitted_states" if args.emit else None
    if emit_root is not None:
        (emit_root / task).mkdir(parents=True, exist_ok=True)

    for seed in parse_seeds(args.seeds):
        snapshot_path = CANONICAL / "snapshots" / task / f"seed_{seed:03d}.pkl"
        if not snapshot_path.exists():
            raise FileNotFoundError(snapshot_path)
        with snapshot_path.open("rb") as handle:
            snapshot = pickle.load(handle)
        for arm in arms:
            out = artifact / "episodes" / task / arm
            summary_path = out / f"episode_{seed:03d}_summary.json"
            arrays_path = out / f"episode_{seed:03d}_arrays.npz"
            if summary_path.exists() and arrays_path.exists():
                print(json.dumps({"skip": True, "task": task, "seed": seed, "arm": arm}), flush=True)
                continue
            summary, records, actions_array = run_seed_arm(
                env, policies[arm], arm, task, seed, snapshot, emit_root and emit_root / task
            )
            summary["environment_id"] = environment_id
            summary["worker_id"] = args.worker_id
            out.mkdir(parents=True, exist_ok=True)
            atomic_json(summary_path, summary)
            write_arrays(arrays_path, records, actions_array)
            print(json.dumps({"task": task, "seed": seed, "arm": arm,
                              "success": summary["success"], "steps": summary["control_steps"],
                              "runtime_seconds": round(summary["runtime_seconds"], 1)}), flush=True)
    env.close()


if __name__ == "__main__":
    main()
