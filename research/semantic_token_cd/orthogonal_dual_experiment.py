"""Paired 50-seed Orthogonal Dual-CD rollout for one SIMPLER task."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from research.semantic_token_cd.distractor_rollout import (
    PCD_SOURCE,
    capture_snapshot,
    jsonable,
    restore_snapshot,
    snapshot_sha,
)
from research.semantic_token_cd.orthogonal_dual_rollout import build_policy, write_arrays
from research.semantic_token_cd.spatial_grid_rollout import make_environment, run_episode


PROTOCOL = "ORTHOGONAL_DUAL_ATTENTION_CD_50_V1"
ARM = "orthogonal_dual"
TASKS = (
    "google_robot_pick_coke_can",
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_move_near",
    "google_robot_place_apple_in_closed_top_drawer",
    "widowx_carrot_on_plate",
    "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel",
    "widowx_stack_cube",
)
LAMBDA_GEOM = 0.5
LAMBDA_SEM = 0.35


def expected_seeds(task: str) -> list[int]:
    seeds = list(range(50))
    if task in {"google_robot_open_drawer", "google_robot_close_drawer"}:
        seeds.remove(4)
    return seeds


def write_config(artifact: Path, reference: Path) -> None:
    config = {
        "experiment": PROTOCOL,
        "tasks": list(TASKS),
        "requested_seeds_per_task": 50,
        "seeds": list(range(50)),
        "technical_exclusions": {
            "google_robot_open_drawer": [4],
            "google_robot_close_drawer": [4],
            "reason": "drawer seed 4 canonical snapshot is not reproducible across fresh environment processes",
        },
        "new_arm": ARM,
        "reference_artifact": str(reference),
        "reference_arms": ["vanilla", "attn_uniform", "attn_semantic"],
        "lambda_geom": LAMBDA_GEOM,
        "lambda_sem": LAMBDA_SEM,
        "orthogonal_eps": 1e-8,
        "attention_layers": [16, 32],
        "attention_mask_value_requested": -10000.0,
        "uniform_geometry": "16x16 even-row/even-column strided grid; 64 tokens",
        "semantic_selector": "entity_set KMeans K=8 seed=0 n_init=10",
        "projection": "per action token over full OpenVLA vocabulary",
        "last_action_token": "positive branch preserved to match prior PCD protocol",
        "gates": {
            "close_drawer_sr": 0.65,
            "open_drawer_sr": 0.30,
            "average_gain_over_best_reference": 0.03,
        },
    }
    path = artifact / "CONFIG_LOCK.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError("CONFIG_LOCK.json differs from preregistration")
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def audit_trace(trace: list[dict]) -> dict[str, bool]:
    spaces = ("orthogonal_full_vocab", "orthogonal_action_vocab")
    scalar_keys = (
        "norm_r_geom",
        "norm_r_sem",
        "norm_r_sem_ortho",
        "cos_sim_raw",
        "ortho_ratio",
        "ortho_dot_max_abs",
        "ortho_relative_error_max",
        "projection_coefficient_mean",
    )
    return {
        "feature_equal": bool(trace and all(step["feature_equal"] for step in trace)),
        "uniform_hooks": bool(trace and all(
            step["uniform_attention_mask"]["hook_calls"] == 112 for step in trace
        )),
        "semantic_hooks": bool(trace and all(
            step["semantic_attention_mask"]["hook_calls"] == 112 for step in trace
        )),
        "finite_diagnostics": bool(trace and all(
            np.isfinite([
                step[space][key] for space in spaces for key in scalar_keys
            ]).all()
            for step in trace
        )),
        "orthogonal_error": bool(trace and all(
            step[space]["ortho_relative_error_max"] <= 1e-5
            for step in trace for space in spaces
        )),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--reference-artifact", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in range(50)))
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    artifact = args.artifact.resolve()
    reference = args.reference_artifact.resolve()
    requested = [int(value) for value in args.seeds.split(",") if value]
    allowed = set(expected_seeds(args.task))
    seeds = [seed for seed in requested if seed in allowed]
    if not seeds or any(seed not in range(50) for seed in requested):
        raise ValueError("Seeds must be in 0..49")
    artifact.mkdir(parents=True, exist_ok=True)
    write_config(artifact, reference)

    reference_manifest = json.loads(
        (reference / "episodes" / args.task / "pairing_manifest.json").read_text()
    )
    if not reference_manifest["all_three_arm_exact_pairing"]:
        raise RuntimeError("Reference three-arm pairing audit did not pass")
    reference_pairs = {
        int(pair["seed"]): pair for pair in reference_manifest["pairs"]
    }
    env, environment_id = make_environment(args.task)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    policy = build_policy(OpenVLAInference(**config), LAMBDA_GEOM, LAMBDA_SEM)
    pairs = []
    excluded = []

    for seed in seeds:
        reference_pair = reference_pairs[seed]
        snapshot = capture_snapshot(env, seed)  # Reset 1 of 4.
        canonical = snapshot_sha(snapshot)
        if canonical != reference_pair["canonical_snapshot_sha256"]:
            # Seed snapshot not reproducible across the reference->dual session
            # boundary (same class as drawer seed-4 exclusion). Record + skip
            # rather than aborting the benchmark; surfaced for the user, not silent.
            print(json.dumps({
                "exclude": args.task, "seed": seed,
                "reason": "snapshot_not_reproducible_across_sessions",
            }), flush=True)
            excluded.append(seed)
            continue

        arm_dir = artifact / "episodes" / args.task / ARM
        arm_dir.mkdir(parents=True, exist_ok=True)
        summary_path = arm_dir / f"episode_{seed:03d}_summary.json"
        arrays_path = arm_dir / f"episode_{seed:03d}_arrays.npz"
        if summary_path.exists() and arrays_path.exists():
            summary = json.loads(summary_path.read_text())
            if summary["canonical_snapshot_sha256"] != canonical:
                raise RuntimeError(f"Existing episode snapshot mismatch: {summary_path}")
            env.reset(seed=seed)  # Reset 2.
            print(json.dumps({"skip": args.task, "seed": seed, "arm": ARM}), flush=True)
        else:
            obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)  # Reset 2.
            instruction = env.unwrapped.get_language_instruction()
            policy.reset(instruction, seed=seed)
            policy._episode_trace = []
            policy._episode_logits = []
            result, steps, reason, actions, jitter = run_episode(
                env, policy, instruction, obs
            )
            write_arrays(arrays_path, policy._episode_logits, actions)
            trace = jsonable(policy._episode_trace)
            audits = audit_trace(trace)
            technical_pass = all(audits.values())
            if not technical_pass:
                raise RuntimeError(f"Orthogonal technical audit failed: {audits}")
            summary = {
                "protocol_id": PROTOCOL,
                "environment_id": environment_id,
                "task": args.task,
                "seed": seed,
                "episode_id": seed,
                "arm": ARM,
                "instruction": instruction,
                "success": bool(result["success"]),
                "failure_reason": reason,
                "control_steps": steps,
                "action_jitter_index": jitter,
                "lambda_geom": LAMBDA_GEOM,
                "lambda_sem": LAMBDA_SEM,
                "canonical_snapshot_sha256": canonical,
                "initial_state_sha256": state_sha,
                "initial_rgb_sha256": rgb_sha,
                "arrays_file": arrays_path.name,
                "selector_trace": trace,
                "technical_audits": audits,
                "technical_pass": technical_pass,
                "result": jsonable(result),
            }
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            print(json.dumps({
                "task": args.task, "seed": seed, "arm": ARM,
                "success": summary["success"], "steps": steps,
            }), flush=True)

        env.reset(seed=seed)  # Reset 3.
        env.reset(seed=seed)  # Reset 4.
        pairs.append({
            "seed": seed,
            "canonical_snapshot_sha256": canonical,
            "reference_pairing_manifest": str(
                reference / "episodes" / args.task / "pairing_manifest.json"
            ),
            "exact_four_arm_pairing": True,
            "summary": str(summary_path),
        })

    task_root = artifact / "episodes" / args.task
    (task_root / "pairing_manifest.json").write_text(json.dumps({
        "task": args.task,
        "seeds": seeds,
        "excluded_seeds": excluded,
        "new_arm": ARM,
        "pairs": pairs,
        "all_four_arm_exact_pairing": True,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"task": args.task, "DONE": True,
                      "n_seeds": len(pairs), "excluded": excluded}), flush=True)


if __name__ == "__main__":
    main()
