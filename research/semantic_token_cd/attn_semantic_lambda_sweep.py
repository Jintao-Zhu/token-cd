"""Attention-CD lambda (intervention strength) sweep: vanilla + 5 CD arms varying
ONLY lambda in {0.1, 0.25, 0.5, 1.0, 2.0} (K=8, L8-15, semantic_hard fixed).

100 seeds per arm (default 100-199), capture-once-reuse pairing, self-contained.
final = (1+lambda) * clean - lambda * negative.

Per-arm audit: feature_equal on every step + hook_calls == 56 (7 x 8 layers).
Any failure raises.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from research.semantic_token_cd.distractor_policy import AuditedVanillaInference
from research.semantic_token_cd.distractor_rollout import (
    PCD_SOURCE,
    capture_snapshot,
    jsonable,
    restore_snapshot,
    snapshot_sha,
)
from research.semantic_token_cd.spatial_grid_policy import SpatialGridAttentionCDInference
from research.semantic_token_cd.spatial_grid_rollout import (
    make_environment,
    run_episode,
    write_arrays,
)

PROTOCOL = "ATTN_SEMANTIC_LAMBDA_SWEEP_V1"
TASKS = ("google_robot_move_near", "google_robot_close_drawer", "google_robot_pick_coke_can")
ARM_LAMBDAS = {
    "vanilla": None,
    "lambda_010": 0.1,
    "lambda_025": 0.25,
    "lambda_050": 0.5,
    "lambda_100": 1.0,
    "lambda_200": 2.0,
}
ARMS = tuple(ARM_LAMBDAS.keys())
KMEANS_K = 8
LAYER_START = 8
LAYER_END = 16
EXPECTED_HOOK_CALLS = 7 * (LAYER_END - LAYER_START)  # 56
SEED_START = 100
N_SEEDS = 100


def parse_seeds(spec: str) -> list[int]:
    if not spec.strip():
        return list(range(SEED_START, SEED_START + N_SEEDS))
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def build_policies(base):
    policies = {}
    vanilla = copy.copy(base)
    vanilla.__class__ = AuditedVanillaInference
    vanilla._episode_trace = []
    vanilla._episode_logits = []
    policies["vanilla"] = vanilla
    for arm, lambd in ARM_LAMBDAS.items():
        if lambd is None:
            continue
        policy = copy.copy(base)
        policy.__class__ = SpatialGridAttentionCDInference
        policy.alpha = lambd
        policy.lambd = lambd
        policy.kmeans_K = KMEANS_K
        policy.kmeans_seed = 0
        policy.selection_mode = "semantic"
        policy.spatial_selection_mode = "semantic_hard"
        policy.attention_layer_start = LAYER_START
        policy.attention_layer_end = LAYER_END
        policy.attention_mask_value = -1e4
        policy._selector_instr = None
        policy._entities = []
        policy._entity_emb = []
        policy._emb_cache = {}
        policy._episode_trace = []
        policy._episode_logits = []
        policy._episode_seed = 0
        policy._selector_step = 0
        policies[arm] = policy
    return policies


def audit_trace(trace: list[dict], expected_calls: int) -> dict:
    if not trace:
        raise RuntimeError("Attention arm produced no trace")
    feature_equal = all(step["feature_equal"] for step in trace)
    hook_pass = all(
        step.get("negative_truncated", False)
        or step["attention_mask"]["hook_calls"] == expected_calls
        for step in trace
    )
    audit = {
        "all_visual_features_bit_identical": feature_equal,
        "all_attention_hook_audits_pass": hook_pass,
        "technical_pass": feature_equal and hook_pass,
    }
    if not audit["technical_pass"]:
        raise RuntimeError(f"Technical audit failed: {audit}")
    return audit


def config_lock(task_root: Path, task: str) -> None:
    lock = {
        "experiment": PROTOCOL,
        "benchmark": "SIMPLER Attention-CD lambda sweep, K8 L8-15 semantic_hard",
        "task": task,
        "seeds": list(range(SEED_START, SEED_START + N_SEEDS)),
        "arms": list(ARMS),
        "arm_lambdas": ARM_LAMBDAS,
        "kmeans_k": KMEANS_K,
        "attention_layers": [LAYER_START, LAYER_END],
        "expected_hook_calls": EXPECTED_HOOK_CALLS,
        "attention_mask_value_requested": -10000.0,
        "selection_mode": "semantic",
        "spatial_selection_mode": "semantic_hard",
        "pairing": "capture-once-reuse",
    }
    path = task_root / "CONFIG_LOCK.json"
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            existing = None
        if existing is not None and existing != lock:
            raise RuntimeError(f"CONFIG_LOCK differs from requested run: {path}")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seeds", type=str, default="", help="comma/range; empty = 100..199")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    seeds = parse_seeds(args.seeds)
    artifact = args.artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    task_root = artifact / "episodes" / args.task
    task_root.mkdir(parents=True, exist_ok=True)
    config_lock(task_root, args.task)

    env, environment_id = make_environment(args.task)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    policy_config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    policies = build_policies(OpenVLAInference(**policy_config))

    pairs = []
    for seed in seeds:
        snapshot = capture_snapshot(env, seed)
        canonical = snapshot_sha(snapshot)
        summaries: dict[str, dict] = {}
        initial_hashes: dict[str, tuple] = {}
        for arm in ARMS:
            arm_dir = task_root / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            summary_path = arm_dir / f"episode_{seed:03d}_summary.json"
            arrays_path = arm_dir / f"episode_{seed:03d}_arrays.npz"
            if summary_path.exists() and arrays_path.exists():
                summaries[arm] = json.loads(summary_path.read_text())
                print(json.dumps({"skip": args.task, "seed": seed, "arm": arm}), flush=True)
                continue
            obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
            initial_hashes[arm] = (state_sha, rgb_sha)
            instruction = env.unwrapped.get_language_instruction()
            policy = policies[arm]
            policy.reset(instruction, seed=seed)
            policy._episode_trace = []
            policy._episode_logits = []
            result, steps, reason, actions, jitter = run_episode(env, policy, instruction, obs)
            write_arrays(arrays_path, policy._episode_logits, actions)
            trace = jsonable(policy._episode_trace)
            lambd = ARM_LAMBDAS[arm]
            audit = None if lambd is None else audit_trace(trace, EXPECTED_HOOK_CALLS)
            summary = {
                "protocol_id": PROTOCOL,
                "environment_id": environment_id,
                "task": args.task,
                "seed": seed,
                "episode_id": seed,
                "arm": arm,
                "instruction": instruction,
                "success": bool(result["success"]),
                "failure_reason": reason,
                "control_steps": steps,
                "action_jitter_index": jitter,
                "lambda": 0.0 if lambd is None else lambd,
                "kmeans_k": KMEANS_K if lambd is not None else None,
                "attention_layers": [LAYER_START, LAYER_END] if lambd is not None else None,
                "expected_hook_calls": EXPECTED_HOOK_CALLS if lambd is not None else None,
                "canonical_snapshot_sha256": canonical,
                "initial_state_sha256": state_sha,
                "initial_rgb_sha256": rgb_sha,
                "arrays_file": arrays_path.name,
                "selector_trace": trace,
                "result": jsonable(result),
            }
            if audit is not None:
                summary.update(audit)
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            summaries[arm] = summary
            print(json.dumps({"task": args.task, "seed": seed, "arm": arm,
                              "success": summary["success"], "steps": steps,
                              "technical_pass": audit is None or audit["technical_pass"]},
                             sort_keys=True), flush=True)

        for arm in ARMS:
            if arm not in initial_hashes:
                initial_hashes[arm] = (
                    summaries[arm]["initial_state_sha256"],
                    summaries[arm]["initial_rgb_sha256"],
                )
        if len({v[0] for v in initial_hashes.values()}) != 1 or len({v[1] for v in initial_hashes.values()}) != 1:
            raise RuntimeError(f"Cross-arm initial-state mismatch for {args.task} seed {seed}")
        if len({s["canonical_snapshot_sha256"] for s in summaries.values()}) != 1:
            raise RuntimeError(f"Cross-arm snapshot mismatch for {args.task} seed {seed}")
        pairs.append({"seed": seed, "canonical_snapshot_sha256": canonical, "exact_pairing": True})

    manifest_path = task_root / f"pairing_manifest_{min(seeds):03d}_{max(seeds):03d}.json"
    manifest_path.write_text(json.dumps(
        {"task": args.task, "seeds": seeds, "arms": list(ARMS), "pairs": pairs,
         "all_arm_exact_pairing": True}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"task": args.task, "DONE": True, "n_seeds": len(seeds)}), flush=True)


if __name__ == "__main__":
    main()
