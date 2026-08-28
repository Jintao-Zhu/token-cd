"""Run the K=8 layer_start=0 cell (block action->visual attention from layer 0).

Single missing cell of the move_near K x attention-layer ablation, extending the
layer sweep to the full 0-31 range. 49 seeds, hash-paired against the frozen
reference, exactly like the existing factorial cells.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

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


TASK = "google_robot_move_near"
EXCLUDED_SEEDS = (36,)
SEEDS = tuple(seed for seed in range(50) if seed not in EXCLUDED_SEEDS)
REFERENCE_ROOT = Path(
    "artifacts/orthogonal_dual_four_arm_6task_50_v1/episodes/"
    "google_robot_move_near/attn_semantic"
)
ALLOWED_CELLS = {(8, 0)}


def build_policy(base, kmeans_k: int, layer_start: int):
    policy = copy.copy(base)
    policy.__class__ = SpatialGridAttentionCDInference
    policy.alpha = 0.5
    policy.lambd = 0.5
    policy.kmeans_K = kmeans_k
    policy.kmeans_seed = 0
    policy.selection_mode = "semantic"
    policy.spatial_selection_mode = "semantic_hard"
    policy.attention_layer_start = layer_start
    policy.attention_layer_end = 32
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


def expected_reference(seed: int) -> dict:
    path = REFERENCE_ROOT / f"episode_{seed:03d}_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen reference summary: {path}")
    return json.loads(path.read_text())


def locked_config(kmeans_k: int, layer_start: int) -> dict:
    return {
        "experiment": f"ATTN_SEMANTIC_FACTORIAL_K{kmeans_k}_L{layer_start}_MOVE_NEAR_49_V1",
        "benchmark": "SIMPLER move_near (Google Robot)",
        "task": TASK,
        "seeds": list(SEEDS),
        "excluded_seeds": list(EXCLUDED_SEEDS),
        "exclusion_reason": "seed 36 snapshot differed between the two existing factorial cells",
        "arm": "attn_semantic",
        "lambda": 0.5,
        "kmeans_k": kmeans_k,
        "attention_layers": [layer_start, 32],
        "attention_mask_value_requested": -10000.0,
        "selector": "entity_set; per-entity top-1 KMeans group; union",
        "reference_root": str(REFERENCE_ROOT.resolve()),
        "pairing": "captured snapshot, initial state, and RGB must match the frozen reference per seed",
    }


def write_config(artifact: Path, config: dict) -> None:
    path = artifact / "CONFIG_LOCK.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError(f"CONFIG_LOCK differs from requested cell: {path}")
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def audit_trace(trace: list[dict], expected_calls: int) -> dict:
    if not trace:
        raise RuntimeError("Semantic arm produced no trace")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--kmeans-k", type=int, required=True)
    parser.add_argument("--layer-start", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if (args.kmeans_k, args.layer_start) not in ALLOWED_CELLS:
        raise ValueError(f"Only missing factorial cells are allowed: {sorted(ALLOWED_CELLS)}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    artifact = args.artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    config = locked_config(args.kmeans_k, args.layer_start)
    write_config(artifact, config)
    task_root = artifact / "episodes" / TASK / "attn_semantic"
    task_root.mkdir(parents=True, exist_ok=True)

    env, environment_id = make_environment(TASK)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    policy_config = get_policy_config("openvla", checkpoint, TASK, {}, False)
    policy = build_policy(
        OpenVLAInference(**policy_config), args.kmeans_k, args.layer_start
    )
    expected_calls = 7 * (32 - args.layer_start)
    completed = []

    for seed in SEEDS:
        summary_path = task_root / f"episode_{seed:03d}_summary.json"
        arrays_path = task_root / f"episode_{seed:03d}_arrays.npz"
        if summary_path.exists() and arrays_path.exists():
            summary = json.loads(summary_path.read_text())
            if not summary.get("technical_pass"):
                raise RuntimeError(f"Existing seed failed audit: {summary_path}")
            completed.append(seed)
            print(json.dumps({"seed": seed, "skip_complete": True}), flush=True)
            continue

        reference = expected_reference(seed)
        snapshot = capture_snapshot(env, seed)
        canonical = snapshot_sha(snapshot)
        obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
        actual_hashes = (canonical, state_sha, rgb_sha)
        reference_hashes = (
            reference["canonical_snapshot_sha256"],
            reference["initial_state_sha256"],
            reference["initial_rgb_sha256"],
        )
        if actual_hashes != reference_hashes:
            raise RuntimeError(
                f"Frozen pairing mismatch at seed {seed}: "
                f"actual={actual_hashes}, reference={reference_hashes}"
            )

        instruction = env.unwrapped.get_language_instruction()
        if instruction != reference["instruction"]:
            raise RuntimeError(f"Instruction mismatch at seed {seed}")
        policy.reset(instruction, seed=seed)
        policy._episode_trace = []
        policy._episode_logits = []
        result, steps, reason, actions, jitter = run_episode(
            env, policy, instruction, obs
        )
        write_arrays(arrays_path, policy._episode_logits, actions)
        trace = jsonable(policy._episode_trace)
        audit = audit_trace(trace, expected_calls)
        summary = {
            "protocol_id": config["experiment"],
            "environment_id": environment_id,
            "task": TASK,
            "seed": seed,
            "episode_id": seed,
            "arm": "attn_semantic",
            "instruction": instruction,
            "success": bool(result["success"]),
            "failure_reason": reason,
            "control_steps": steps,
            "action_jitter_index": jitter,
            "first_step_residual_norm": float(trace[0]["residual_norm"]),
            "lambda": 0.5,
            "kmeans_k": args.kmeans_k,
            "attention_layers": [args.layer_start, 32],
            "canonical_snapshot_sha256": canonical,
            "initial_state_sha256": state_sha,
            "initial_rgb_sha256": rgb_sha,
            "reference_protocol_id": reference["protocol_id"],
            "arrays_file": arrays_path.name,
            "selector_trace": trace,
            "result": jsonable(result),
            **audit,
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        completed.append(seed)
        (artifact / "task_progress.json").write_text(
            json.dumps(
                {"completed_seeds": completed, "n_completed": len(completed)},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(
            json.dumps(
                {
                    "seed": seed,
                    "success": summary["success"],
                    "steps": steps,
                    "technical_pass": True,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    manifest = {
        "protocol_id": config["experiment"],
        "task": TASK,
        "seeds": list(SEEDS),
        "excluded_seeds": list(EXCLUDED_SEEDS),
        "n_seeds": len(SEEDS),
        "all_reference_hashes_match": True,
        "all_technical_audits_pass": True,
    }
    (artifact / "PAIRING_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"DONE": True, "n_seeds": len(SEEDS)}), flush=True)


if __name__ == "__main__":
    main()
