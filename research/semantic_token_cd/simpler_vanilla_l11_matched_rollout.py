"""Paired vanilla and canonical L11-Matched closed-loop SIMPLER evaluation."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import random
import subprocess
import time
from pathlib import Path


TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
    "google_robot_place_apple_in_closed_top_drawer",
    "widowx_carrot_on_plate",
    "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel",
    "widowx_stack_cube",
)
ARMS = ("vanilla", "l11_matched")
PROTOCOL = "SIMPLER_VANILLA_L11_MATCHED_PAIRED_V1"


def parse_seeds(spec: str) -> list[int]:
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(value) for value in part.split("-", 1))
            seeds.extend(range(lo, hi + 1))
        else:
            seeds.append(int(part))
    result = sorted(set(seeds))
    if not result or result[0] < 0:
        raise ValueError("seeds must be non-negative integers or inclusive ranges")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seeds", required=True, help="for example 100-109")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--worker-id", default="manual")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import numpy as np
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    from research.semantic_token_cd.distractor_policy import AuditedVanillaInference
    from research.semantic_token_cd.distractor_rollout import (
        PCD_SOURCE, array_sha256, capture_snapshot, jsonable, restore_snapshot, snapshot_sha,
    )
    from research.semantic_token_cd.prompt_attn_l11_count_rollout import (
        ARM_COUNTS, audit as audit_l11, build_policies as build_l11_policies,
        finite_mean, write_arrays,
    )
    from research.semantic_token_cd.prompt_attn_layer_rollout import make_environment
    from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX
    from research.semantic_token_cd.spatial_grid_rollout import run_episode

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; SIMPLER renderer/model require the GPU")
    if not (PCD_SOURCE / "pretrained/openvla-7b/config.json").is_file():
        raise FileNotFoundError(f"SIMPLER OpenVLA base checkpoint missing under {PCD_SOURCE}")

    task_index = TASK_INDEX[args.task]
    seeds = parse_seeds(args.seeds)
    artifact = args.artifact.resolve()
    task_root = artifact / "episodes" / args.task
    snapshot_root = artifact / "snapshots" / args.task
    task_root.mkdir(parents=True, exist_ok=True)
    snapshot_root.mkdir(parents=True, exist_ok=True)

    repo = Path(os.environ.get("TOKEN_CD", Path(__file__).resolve().parents[2]))
    code_files = (
        Path(__file__).resolve(),
        repo / "research/semantic_token_cd/prompt_attn_l11_count_rollout.py",
        repo / "research/semantic_token_cd/prompt_attn_shr_policy.py",
    )
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    except Exception:
        commit = "unknown"
    config = {
        "protocol_id": PROTOCOL,
        "benchmark": "SIMPLER OpenVLA-7B base checkpoint",
        "task": args.task,
        "seeds": seeds,
        "arms": list(ARMS),
        "checkpoint": str(PCD_SOURCE / "pretrained/openvla-7b"),
        "checkpoint_config_sha256": sha256_file(PCD_SOURCE / "pretrained/openvla-7b/config.json"),
        "code_commit": commit,
        "code_sha256": {str(path.relative_to(repo)): sha256_file(path) for path in code_files},
        "renderer": "SAPIEN Vulkan NVIDIA ICD; offscreen cuda:0",
        "pairing": "capture one seeded SIMPLER state; restore the identical simulator/agent/RNG snapshot for both arms",
        "vanilla": "clean greedy OpenVLA",
        "l11_matched": {
            "attention_layers_zero_based": [11], "selector": "instruction-conditioned post-softmax prompt attention",
            "coverage": "own-state Standard-SHR KMeans K=8 matched m_t",
            "lambda": 0.5, "beta": 0.0, "gripper_dimension": "clean positive branch",
            "sampling": False,
        },
    }
    config_path = artifact / "CONFIG_LOCK.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError(f"CONFIG_LOCK mismatch; use a new artifact directory: {config_path}")
    atomic_json(config_path, config)

    env, environment_id = make_environment(args.task, args.gpu)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    policy_config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    base = OpenVLAInference(**policy_config)
    vanilla = copy.copy(base)
    vanilla.__class__ = AuditedVanillaInference
    vanilla._episode_trace = []
    vanilla._episode_logits = []
    l11 = build_l11_policies(base, args.task, ("l11_matched",))["l11_matched"]
    policies = {"vanilla": vanilla, "l11_matched": l11}

    pairs = []
    try:
        for seed in seeds:
            snapshot_path = snapshot_root / f"seed_{seed:03d}.pkl"
            if snapshot_path.exists():
                with snapshot_path.open("rb") as stream:
                    snapshot = pickle.load(stream)
            else:
                snapshot = capture_snapshot(env, seed)
                tmp_snapshot = snapshot_path.with_suffix(f".pkl.{os.getpid()}.tmp")
                with tmp_snapshot.open("wb") as stream:
                    pickle.dump(snapshot, stream, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp_snapshot, snapshot_path)
            canonical_sha = snapshot_sha(snapshot)
            initial_hashes = {}
            summaries = {}

            for arm in ARMS:
                arm_root = task_root / arm
                summary_path = arm_root / f"episode_{seed:03d}_summary.json"
                arrays_path = arm_root / f"episode_{seed:03d}_arrays.npz"
                if summary_path.exists() and arrays_path.exists():
                    summaries[arm] = json.loads(summary_path.read_text())
                    initial_hashes[arm] = (
                        summaries[arm]["initial_state_sha256"], summaries[arm]["initial_rgb_sha256"]
                    )
                    continue

                obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
                initial_hashes[arm] = (state_sha, rgb_sha)
                instruction = env.unwrapped.get_language_instruction()
                if instruction != snapshot["instruction"]:
                    raise RuntimeError(f"instruction changed while restoring seed {seed}")
                policy = policies[arm]
                policy.reset(instruction, seed=seed)
                policy._episode_trace = []
                policy._episode_logits = []
                started = time.monotonic()
                result, steps, reason, actions, jitter = run_episode(env, policy, instruction, obs)
                runtime = time.monotonic() - started
                trace = jsonable(policy._episode_trace)
                if arm == "l11_matched":
                    checks = audit_l11(trace, ARM_COUNTS["l11_matched"])
                else:
                    checks = {"technical_pass": bool(policy._episode_logits)}
                    if not checks["technical_pass"]:
                        raise RuntimeError("vanilla produced no policy logits")

                arm_root.mkdir(parents=True, exist_ok=True)
                write_arrays(arrays_path, policy._episode_logits, actions)
                summary = {
                    "protocol_id": PROTOCOL, "environment_id": environment_id,
                    "task": args.task, "seed": seed, "evaluation_seed": seed,
                    "episode_id": seed, "arm": arm, "instruction": instruction,
                    "success": bool(result["success"]), "result": jsonable(result),
                    "failure_reason": reason, "control_steps": steps,
                    "runtime_seconds": runtime, "action_jitter_index": jitter,
                    "canonical_snapshot_sha256": canonical_sha,
                    "initial_state_sha256": state_sha, "initial_rgb_sha256": rgb_sha,
                    "arrays_file": arrays_path.name, "selector_trace": trace,
                    "technical_audit": checks,
                }
                if arm == "l11_matched":
                    summary.update({
                        "attention_layers": [11], "lambda": 0.5, "beta": 0.0,
                        "mean_selected_token_count": finite_mean(x.get("actual_selected_count") for x in trace),
                    })
                atomic_json(summary_path, summary)
                summaries[arm] = summary
                print(json.dumps({
                    "task": args.task, "seed": seed, "arm": arm,
                    "success": summary["success"], "control_steps": steps,
                    "runtime_seconds": round(runtime, 2), "technical_pass": checks["technical_pass"],
                }), flush=True)

            if len(set(initial_hashes.values())) != 1:
                raise RuntimeError(f"Vanilla/L11 initial-state or RGB mismatch for seed {seed}: {initial_hashes}")
            if {x["canonical_snapshot_sha256"] for x in summaries.values()} != {canonical_sha}:
                raise RuntimeError(f"canonical snapshot hash mismatch for seed {seed}")
            pairs.append({
                "seed": seed, "canonical_snapshot_sha256": canonical_sha,
                "initial_state_sha256": initial_hashes["vanilla"][0],
                "initial_rgb_sha256": initial_hashes["vanilla"][1],
                "vanilla_success": summaries["vanilla"]["success"],
                "l11_matched_success": summaries["l11_matched"]["success"],
                "exact_pairing": True,
            })
            atomic_json(task_root / "pairing_manifest.json", {
                "protocol_id": PROTOCOL, "task": args.task,
                "seeds_completed": [row["seed"] for row in pairs],
                "arms": list(ARMS), "pairs": pairs, "all_arm_exact_pairing": True,
            })
    finally:
        env.close()

    print(json.dumps({
        "task": args.task, "artifact": str(artifact),
        "completed_pairs": len(pairs), "all_arm_exact_pairing": True,
    }), flush=True)


if __name__ == "__main__":
    main()
