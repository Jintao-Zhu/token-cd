"""Training-free L11 lambda selection by same-state short-horizon lookahead."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import subprocess
import time
from pathlib import Path

import numpy as np

from research.semantic_token_cd.distractor_rollout import (
    PCD_SOURCE,
    clone,
    flatten_action,
    get_image_from_maniskill2_obs_dict,
    jsonable,
    restore_snapshot,
    snapshot_sha,
)
from research.semantic_token_cd.prompt_attn_l11_count_rollout import (
    build_policies,
    make_environment,
)
from research.semantic_token_cd.prompt_attn_shr_rollout import atomic_json, load_reference
from research.semantic_token_cd.run_l11_token_causal_audit import progress
from research.semantic_token_cd.rollout_pilot import wrapped_observation


PROTOCOL = "PROMPT_ATTN_L11_LOOKAHEAD_LAMBDA_PILOT_V1"
TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
)
LAMBDAS = (0.0, 0.25, 0.5)
DEFAULT_LAMBDA = 0.5
APPROACH_SWITCH_MARGIN = 0.025
WEAK_TIE_TOLERANCE = 0.01


def parse_seeds(specification: str) -> list[int]:
    seeds = []
    for part in specification.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, high = map(int, part.split("-", 1))
            seeds.extend(range(low, high + 1))
        else:
            seeds.append(int(part))
    result = sorted(set(seeds))
    if not result or any(seed < 0 or seed > 299 for seed in result):
        raise ValueError("seeds must be within 0..299")
    return result


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_mid(env) -> dict:
    inner = env.unwrapped
    return {
        "sim_state": np.asarray(inner.get_state()).copy(),
        "agent_state": clone(inner.agent.get_state()),
        "rng_state": clone(inner._episode_rng.get_state()),
        "elapsed_steps": int(inner._elapsed_steps),
        "instruction": inner.get_language_instruction(),
    }


def restore_mid(env, seed: int, snapshot: dict):
    env.reset(seed=seed)
    inner = env.unwrapped
    inner.set_state(np.asarray(snapshot["sim_state"]).copy())
    inner.agent.set_state(clone(snapshot["agent_state"]))
    inner._episode_rng.set_state(clone(snapshot["rng_state"]))
    elapsed = int(snapshot["elapsed_steps"])
    inner._elapsed_steps = elapsed
    current = env
    while hasattr(current, "env"):
        if hasattr(current, "_elapsed_steps"):
            current._elapsed_steps = elapsed
        current = current.env
    return wrapped_observation(env)


def reset_policy(policy, instruction: str, seed: int, decision_step: int) -> None:
    policy.reset(instruction, seed=seed)
    policy._selector_step = int(decision_step)
    policy._episode_trace = []
    policy._episode_logits = []


def transition_score(task: str, before: dict, after: dict) -> tuple[str, float, dict]:
    success_bonus = 1.0 if after.get("success", False) else 0.0
    if "drawer" in task:
        direction = 1.0 if "open" in task else -1.0
        drawer_gain = direction * (after["qpos"] - before["qpos"])
        distance_gain = before["tcp_handle_distance"] - after["tcp_handle_distance"]
        if before["tcp_handle_distance"] > 0.08:
            phase = "approach"
            raw_score = distance_gain + 2.0 * max(drawer_gain, 0.0)
        else:
            phase = "manipulate"
            raw_score = drawer_gain + 0.1 * distance_gain
        terms = {"drawer_gain": drawer_gain, "distance_gain": distance_gain}
    elif task.endswith("pick_coke_can"):
        distance_gain = before["tcp_object_distance"] - after["tcp_object_distance"]
        lift_gain = after["object_z"] - before["object_z"]
        grasp_gain = float(after["is_grasped"]) - float(before["is_grasped"])
        if before["is_grasped"]:
            phase = "lift"
            raw_score = lift_gain + 0.25 * distance_gain + 0.05 * grasp_gain
        else:
            phase = "approach_or_grasp"
            raw_score = distance_gain + 2.0 * max(lift_gain, 0.0) + 0.05 * max(grasp_gain, 0.0)
        terms = {"distance_gain": distance_gain, "lift_gain": lift_gain, "grasp_gain": grasp_gain}
    else:
        source_gain = before["tcp_source_distance"] - after["tcp_source_distance"]
        target_gain = before["source_target_xy_distance"] - after["source_target_xy_distance"]
        moved_gain = float(after["moved_correct_obj"]) - float(before["moved_correct_obj"])
        near_gain = float(after["near_target"]) - float(before["near_target"])
        if before["moved_correct_obj"] or after["moved_correct_obj"]:
            phase = "transport"
            raw_score = target_gain + 0.25 * source_gain + 0.03 * moved_gain + 0.1 * near_gain
        else:
            phase = "approach"
            raw_score = source_gain + 0.25 * target_gain + 0.03 * moved_gain
        terms = {
            "source_gain": source_gain,
            "target_gain": target_gain,
            "moved_gain": moved_gain,
            "near_gain": near_gain,
        }
    return phase, float(success_bonus + raw_score), {key: float(value) for key, value in terms.items()}


def run_branch(env, policy, instruction: str, seed: int, decision_step: int,
               snapshot: dict, horizon: int) -> dict:
    obs = restore_mid(env, seed, snapshot)
    reset_policy(policy, instruction, seed, decision_step)
    before = progress(env.unwrapped, task=policy._lookahead_task)
    actions = []
    terminate_flags = []
    truncated = False
    predicted_terminated = False
    model_calls = 0
    while len(actions) < horizon and not truncated and not predicted_terminated:
        image = np.asarray(get_image_from_maniskill2_obs_dict(env, obs), dtype=np.uint8)
        _raw, predicted, _meta = policy.step(
            image, None, instruction, proprio=obs["agent"]["eef_pos"]
        )
        model_calls += 1
        if not isinstance(predicted, list):
            predicted = [predicted]
        for action in predicted:
            executed = flatten_action(action)
            obs, _reward, _terminated, truncated, _info = env.step(executed)
            terminate = bool(action["terminate_episode"][0] > 0)
            if terminate and not env.unwrapped.is_final_subtask():
                terminate = False
                env.advance_to_next_subtask()
                instruction = env.unwrapped.get_language_instruction()
            actions.append(executed.tolist())
            terminate_flags.append(terminate)
            predicted_terminated = terminate
            if len(actions) >= horizon or truncated or predicted_terminated:
                break
    after = progress(env.unwrapped, task=policy._lookahead_task)
    phase, score, terms = transition_score(policy._lookahead_task, before, after)
    return {
        "before": before,
        "after": after,
        "phase": phase,
        "score": score,
        "score_terms": terms,
        "actions": actions,
        "terminate_flags": terminate_flags,
        "model_calls": model_calls,
        "truncated": bool(truncated),
        "end_snapshot": capture_mid(env),
        "trace": jsonable(policy._episode_trace),
    }


def choose_branch(branches: dict[float, dict], manipulation_margin: float) -> tuple[float, dict]:
    baseline = branches[DEFAULT_LAMBDA]["score"]
    raw_best_score = max(branches[value]["score"] for value in LAMBDAS)
    near_best = [
        value for value in LAMBDAS
        if branches[value]["score"] >= raw_best_score - WEAK_TIE_TOLERANCE
    ]
    best = max(near_best)
    phase = branches[DEFAULT_LAMBDA]["phase"]
    margin = APPROACH_SWITCH_MARGIN if phase in {"approach", "approach_or_grasp"} else manipulation_margin
    if best != DEFAULT_LAMBDA and branches[best]["score"] < baseline + margin:
        best = DEFAULT_LAMBDA
    return best, {
        "best_raw_lambda": max(LAMBDAS, key=lambda value: branches[value]["score"]),
        "near_best_lambdas": near_best,
        "selection_phase": phase,
        "default_score": baseline,
        "chosen_score": branches[best]["score"],
        "switch_margin": margin,
        "weak_tie_tolerance": WEAK_TIE_TOLERANCE,
        "switched_from_default": best != DEFAULT_LAMBDA,
    }


def commit_branch(env, seed: int, branch: dict) -> tuple[dict, bool, bool, float]:
    obs = restore_mid(env, seed, branch["end_snapshot"])
    state_error = float(np.max(np.abs(
        np.asarray(env.unwrapped.get_state()) - np.asarray(branch["end_snapshot"]["sim_state"])
    )))
    predicted_terminated = bool(branch["terminate_flags"] and branch["terminate_flags"][-1])
    return obs, bool(branch["truncated"]), predicted_terminated, state_error


def run_episode(env, policies: dict[float, object], task: str, instruction: str, seed: int,
                obs, horizon: int, commit: int, margin: float) -> dict:
    control_steps = 0
    decision_step = 0
    truncated = False
    predicted_terminated = False
    decisions = []
    while not truncated and not predicted_terminated:
        snapshot = capture_mid(env)
        branch_horizon = min(horizon, commit)
        branches = {
            value: run_branch(env, policies[value], instruction, seed, decision_step, snapshot, branch_horizon)
            for value in LAMBDAS
        }
        chosen, selection = choose_branch(branches, margin)
        chosen_branch = branches[chosen]
        obs, truncated, predicted_terminated, state_error = commit_branch(env, seed, chosen_branch)
        if state_error > 1e-6:
            raise RuntimeError(f"chosen branch replay mismatch: {state_error}")
        actions_executed = len(chosen_branch["actions"])
        control_steps += actions_executed
        decision_step += int(chosen_branch["model_calls"])
        decisions.append({
            "control_step_start": control_steps - actions_executed,
            "decision_step_start": decision_step - int(chosen_branch["model_calls"]),
            "chosen_lambda": chosen,
            "phase": chosen_branch["phase"],
            "actions_executed": actions_executed,
            "model_calls": int(chosen_branch["model_calls"]),
            "commit_state_max_abs_difference": state_error,
            "selection": selection,
            "branches": {
                str(value): {
                    key: branches[value][key]
                    for key in ("before", "after", "phase", "score", "score_terms", "model_calls", "truncated")
                }
                for value in LAMBDAS
            },
        })
        if actions_executed == 0:
            raise RuntimeError("lookahead branch emitted no actions")
        current_success = bool(env.unwrapped.evaluate().get("success", False))
        if current_success:
            break
        instruction = env.unwrapped.get_language_instruction()
    evaluation = dict(env.unwrapped.evaluate())
    return {
        "success": bool(evaluation.get("success", False)),
        "evaluation": jsonable(evaluation),
        "control_steps": control_steps,
        "decision_steps": decision_step,
        "truncated": bool(truncated),
        "predicted_terminated": bool(predicted_terminated),
        "decisions": decisions,
    }


def ensure_config(artifact: Path, canonical: Path, horizon: int, commit: int, margin: float) -> dict:
    repo = Path(__file__).resolve().parents[2]
    payload = {
        "protocol_id": PROTOCOL,
        "created_date": "2026-09-14",
        "purpose": "training-free same-state short-horizon lambda selection",
        "tasks": list(TASKS),
        "lambdas": list(LAMBDAS),
        "default_lambda": DEFAULT_LAMBDA,
        "horizon_actions": horizon,
        "commit_actions": commit,
        "switch_margin": margin,
        "approach_switch_margin": APPROACH_SWITCH_MARGIN,
        "weak_tie_tolerance": WEAK_TIE_TOLERANCE,
        "score": "task-native privileged physical progress plus terminal-success bonus",
        "commit_mode": "restore the exact complete simulator state of the selected branch",
        "canonical_snapshot_artifact": str(canonical),
        "selector": "L11-Matched prompt attention, beta=0, own-state matched coverage",
        "code": {
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            "rollout_sha256": file_sha256(Path(__file__)),
        },
    }
    path = artifact / "CONFIG_LOCK.json"
    artifact.mkdir(parents=True, exist_ok=True)
    if path.exists() and json.loads(path.read_text()) != payload:
        raise RuntimeError(f"config lock differs: {path}")
    atomic_json(path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--commit", type=int, default=10)
    parser.add_argument("--margin", type=float, default=0.002)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    artifact = args.artifact.resolve()
    canonical = args.canonical.resolve()
    config_lock = ensure_config(artifact, canonical, args.horizon, args.commit, args.margin)
    env, environment_id = make_environment(args.task)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    base = OpenVLAInference(**config)
    policies = {}
    for value in LAMBDAS:
        policy = build_policies(base, args.task, ("l11_matched",))["l11_matched"]
        policy.lambd = value
        policy.save_prompt_attention = False
        policy._lookahead_task = args.task
        policies[value] = policy
    try:
        for seed in parse_seeds(args.seeds):
            destination = artifact / "episodes" / args.task / f"episode_{seed:03d}_summary.json"
            if destination.exists():
                print(json.dumps({"skip": str(destination)}), flush=True)
                continue
            with (canonical / "snapshots" / args.task / f"seed_{seed:03d}.pkl").open("rb") as handle:
                initial = pickle.load(handle)
            reference = load_reference(canonical, args.task, seed)
            if snapshot_sha(initial) != reference["canonical_snapshot_sha256"]:
                raise RuntimeError("canonical snapshot mismatch")
            obs, state_sha, rgb_sha = restore_snapshot(env, seed, initial)
            if (state_sha, rgb_sha) != (reference["initial_state_sha256"], reference["initial_rgb_sha256"]):
                raise RuntimeError("initial state or RGB mismatch")
            instruction = env.unwrapped.get_language_instruction()
            if instruction != reference["instruction"]:
                raise RuntimeError("instruction mismatch")
            started = time.monotonic()
            result = run_episode(
                env, policies, args.task, instruction, seed, obs,
                args.horizon, args.commit, args.margin,
            )
            runtime = time.monotonic() - started
            summary = {
                "protocol_id": PROTOCOL,
                "task": args.task,
                "environment_id": environment_id,
                "seed": seed,
                "instruction": instruction,
                "success": result["success"],
                "result": result,
                "runtime_seconds": runtime,
                "gpu_id": args.gpu,
                "canonical_snapshot_sha256": reference["canonical_snapshot_sha256"],
                "initial_state_sha256": state_sha,
                "initial_rgb_sha256": rgb_sha,
                "config": config_lock,
                "technical_pass": all(
                    decision["commit_state_max_abs_difference"] <= 1e-6
                    for decision in result["decisions"]
                ),
            }
            atomic_json(destination, summary)
            print(json.dumps({
                "task": args.task,
                "seed": seed,
                "success": summary["success"],
                "decisions": len(result["decisions"]),
                "switches": sum(x["chosen_lambda"] != DEFAULT_LAMBDA for x in result["decisions"]),
                "runtime_seconds": round(runtime, 2),
            }), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
