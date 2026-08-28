"""Two-arm (vanilla + attn_semantic) exact-paired rollout for pick_coke_can.

Variant: KMeans K=8, attention blocking layers 8-31. 50 seeds.
Single-task cell to compare against the K=32 pick_coke_can run.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np

from research.semantic_token_cd.distractor_policy import AuditedVanillaInference
from research.semantic_token_cd.distractor_rollout import (
    PCD_SOURCE,
    capture_snapshot,
    jsonable,
    restore_snapshot,
    snapshot_sha,
)
from research.semantic_token_cd.spatial_grid_policy import (
    SpatialGridAttentionCDInference,
)
from research.semantic_token_cd.spatial_grid_rollout import (
    make_environment,
    run_episode,
    write_arrays,
)


PROTOCOL = "ATTN_SEMANTIC_K8_L8_PICK_COKE_CAN_50_V1"
ALL_TASKS = (
    "google_robot_pick_coke_can",
)
ARMS = ("vanilla", "attn_semantic")
ATTENTION_LAYER_START = 8
ATTENTION_LAYER_END = 32
KMEANS_K = 8
EXPECTED_HOOK_CALLS = 7 * (ATTENTION_LAYER_END - ATTENTION_LAYER_START)  # 7 x 24 = 168


def build_vanilla_policy(base):
    policy = copy.copy(base)
    policy.__class__ = AuditedVanillaInference
    policy._episode_trace = []
    policy._episode_logits = []
    return policy


def build_attention_policy(base):
    policy = copy.copy(base)
    policy.__class__ = SpatialGridAttentionCDInference
    policy.alpha = 0.5
    policy.lambd = 0.5
    policy.kmeans_K = KMEANS_K
    policy.kmeans_seed = 0
    policy.selection_mode = "semantic"
    policy.spatial_selection_mode = "semantic_hard"
    policy.attention_layer_start = ATTENTION_LAYER_START
    policy.attention_layer_end = ATTENTION_LAYER_END
    policy.attention_mask_value = -1e4
    policy._selector_instr = None
    policy._entities = []
    policy._entity_emb = []
    policy._emb_cache = {}
    policy._episode_trace = []
    policy._episode_logits = []
    policy._episode_seed = 0
    policy._selector_step = 0
    return policy


def build_policies(base):
    return {
        "vanilla": build_vanilla_policy(base),
        "attn_semantic": build_attention_policy(base),
    }


def locked_config() -> dict:
    return {
        "experiment": PROTOCOL,
        "benchmark": "SIMPLER pick_coke_can (google_robot)",
        "tasks": list(ALL_TASKS),
        "seeds": list(range(50)),
        "arms": list(ARMS),
        "execution": (
            "one task per process; capture one snapshot per seed and restore the "
            "same snapshot before each arm"
        ),
        "lambda": 0.5,
        "attention_layers": [ATTENTION_LAYER_START, ATTENTION_LAYER_END],
        "attention_mask_value_requested": -10000.0,
        "semantic_selector": f"entity_set KMeans K={KMEANS_K} seed=0 n_init=10",
        "expected_attention_hook_calls_per_branch_per_control_step": (
            EXPECTED_HOOK_CALLS
        ),
    }


def write_config(artifact: Path) -> None:
    expected = locked_config()
    path = artifact / "CONFIG_LOCK.json"
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise RuntimeError("CONFIG_LOCK.json differs from the locked protocol")
        return
    temporary = artifact / f".CONFIG_LOCK.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def audit_trace(arm: str, trace: list[dict]) -> dict:
    if not trace:
        raise RuntimeError(f"{arm} produced no trace")
    if arm == "vanilla":
        return {"technical_pass": True, "all_visual_features_bit_identical": True}
    feature_equal = all(step["feature_equal"] for step in trace)
    hook_pass = all(
        step.get("negative_truncated", False)
        or step["attention_mask"]["hook_calls"] == EXPECTED_HOOK_CALLS
        for step in trace
    )
    technical_pass = feature_equal and hook_pass
    audit = {
        "all_visual_features_bit_identical": feature_equal,
        "all_attention_hook_audits_pass": hook_pass,
        "technical_pass": technical_pass,
    }
    if not audit["technical_pass"]:
        raise RuntimeError(f"Technical audit failed for {arm}: {audit}")
    return audit


def refuse_unsafe_resume(task_root: Path) -> bool:
    manifest = task_root / "pairing_manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        if data.get("all_two_arm_exact_pairing") and len(data.get("pairs", [])) == 50:
            print(json.dumps({"task": task_root.name, "already_complete": True}), flush=True)
            return True
        raise RuntimeError(f"Invalid existing pairing manifest: {manifest}")
    return False


def run_task(artifact: Path, task: str, gpu: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    task_root = artifact / "episodes" / task
    if refuse_unsafe_resume(task_root):
        return
    task_root.mkdir(parents=True, exist_ok=True)

    env, environment_id = make_environment(task)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    policy_config = get_policy_config("openvla", checkpoint, task, {}, False)
    policies = build_policies(OpenVLAInference(**policy_config))
    pairs = []
    completed_seeds = set()
    for seed in range(50):
        summary_paths = {
            arm: task_root / arm / f"episode_{seed:03d}_summary.json" for arm in ARMS
        }
        if all(p.exists() for p in summary_paths.values()):
            ref = json.loads(summary_paths["vanilla"].read_text())
            pairs.append(
                {
                    "seed": seed,
                    "canonical_snapshot_sha256": ref["canonical_snapshot_sha256"],
                    "initial_state_sha256": ref["initial_state_sha256"],
                    "initial_rgb_sha256": ref["initial_rgb_sha256"],
                    "exact_two_arm_pairing": True,
                }
            )
            completed_seeds.add(seed)
    if completed_seeds:
        print(
            json.dumps(
                {"task": task, "resume_skipping_seeds": sorted(completed_seeds)}
            ),
            flush=True,
        )

    for seed in range(50):
        if seed in completed_seeds:
            continue
        snapshot = capture_snapshot(env, seed)
        canonical = snapshot_sha(snapshot)
        summaries = {}
        initial_hashes = {}
        instructions = {}

        for arm in ARMS:
            arm_dir = task_root / arm
            arm_dir.mkdir(exist_ok=True)
            summary_path = arm_dir / f"episode_{seed:03d}_summary.json"
            arrays_path = arm_dir / f"episode_{seed:03d}_arrays.npz"
            obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
            initial_hashes[arm] = (state_sha, rgb_sha)
            instruction = env.unwrapped.get_language_instruction()
            instructions[arm] = instruction
            policy = policies[arm]
            policy.reset(instruction, seed=seed)
            policy._episode_trace = []
            policy._episode_logits = []
            result, steps, reason, actions, jitter = run_episode(
                env, policy, instruction, obs
            )
            write_arrays(arrays_path, policy._episode_logits, actions)
            trace = jsonable(policy._episode_trace)
            audit = audit_trace(arm, trace)
            summary = {
                "protocol_id": PROTOCOL,
                "environment_id": environment_id,
                "task": task,
                "seed": seed,
                "episode_id": seed,
                "arm": arm,
                "instruction": instruction,
                "success": bool(result["success"]),
                "failure_reason": reason,
                "control_steps": steps,
                "action_jitter_index": jitter,
                "first_step_residual_norm": (
                    0.0 if arm == "vanilla" else float(trace[0]["residual_norm"])
                ),
                "lambda": 0.0 if arm == "vanilla" else 0.5,
                "canonical_snapshot_sha256": canonical,
                "initial_state_sha256": state_sha,
                "initial_rgb_sha256": rgb_sha,
                "arrays_file": arrays_path.name,
                "selector_trace": trace,
                "result": jsonable(result),
                **audit,
            }
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n"
            )
            summaries[arm] = summary
            print(
                json.dumps(
                    {
                        "task": task,
                        "seed": seed,
                        "arm": arm,
                        "success": summary["success"],
                        "steps": steps,
                        "technical_pass": audit["technical_pass"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        if len(set(initial_hashes.values())) != 1:
            raise RuntimeError(f"Two-arm initial state/RGB mismatch: {task} {seed}")
        if len(set(instructions.values())) != 1:
            raise RuntimeError(f"Two-arm instruction mismatch: {task} {seed}")
        if len({item["canonical_snapshot_sha256"] for item in summaries.values()}) != 1:
            raise RuntimeError(f"Two-arm canonical snapshot mismatch: {task} {seed}")
        pairs.append(
            {
                "seed": seed,
                "canonical_snapshot_sha256": canonical,
                "initial_state_sha256": next(iter(initial_hashes.values()))[0],
                "initial_rgb_sha256": next(iter(initial_hashes.values()))[1],
                "exact_two_arm_pairing": True,
            }
        )
        (task_root / "task_progress.json").write_text(
            json.dumps(
                {"task": task, "completed_seeds": seed + 1, "last_seed": seed},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    manifest = {
        "protocol_id": PROTOCOL,
        "task": task,
        "seeds": list(range(50)),
        "arms": list(ARMS),
        "pairs": pairs,
        "all_two_arm_exact_pairing": True,
    }
    (task_root / "pairing_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"task": task, "DONE": True, "n_seeds": 50}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--tasks", type=str, default=",".join(ALL_TASKS))
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    tasks = [t for t in args.tasks.split(",") if t]
    for task in tasks:
        if task not in ALL_TASKS:
            raise ValueError(f"Task is not preregistered: {task}")

    artifact = args.artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    write_config(artifact)

    for task in tasks:
        run_task(artifact, task, args.gpu)


if __name__ == "__main__":
    main()
