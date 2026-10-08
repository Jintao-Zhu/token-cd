#!/usr/bin/env python3
"""Paired OpenVLA harmonic/zero/global-mean reconstruction study, fresh seeds."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
TASKS = (
    "google_robot_open_drawer", "google_robot_close_drawer",
    "google_robot_pick_coke_can", "google_robot_move_near",
)
ARMS = ("full", "zero", "clean_global_mean")
PROTOCOL = "OPENVLA_L11_HARMONIC_ZERO_GLOBALMEAN_FRESH_300_499_V1"


def parse_seeds(text: str) -> list[int]:
    out = set()
    for part in text.split(","):
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--snapshot-artifact", type=Path, required=True)
    ap.add_argument("--phase", choices=("preflight", "formal"), required=True)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from research.semantic_token_cd import l11_table2_openvla_rollout as table2
    from research.semantic_token_cd import prompt_attn_l11_budget_causal_rollout as shared
    from research.semantic_token_cd.distractor_rollout import restore_snapshot, snapshot_sha
    from research.semantic_token_cd.rollout_pilot import capture_snapshot
    from research.semantic_token_cd.spatial_grid_rollout import run_episode
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    from research.semantic_token_cd.distractor_rollout import PCD_SOURCE

    artifact = args.artifact.resolve()
    snaproot = args.snapshot_artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    lock = {
        "protocol_id": PROTOCOL, "phase": "paired fresh-seed reconstruction ablation",
        "model": "OpenVLA-7B Fractal SIMPLER checkpoint",
        "task_renderer": {t: {"shader_dir": "rt", "depth": table2.RT_DEPTH[t],
                              "samples_per_pixel": 32, "denoiser": True} for t in TASKS},
        "seeds": list(range(300, 500)), "arms": list(ARMS), "lambda": 0.5,
        "attention_layer": 11, "budget": "dynamic Matched", "ranking": "L11 prompt attention",
        "intervention": "projector output", "guidance": "z+ + 0.5*(z+ - z-)",
        "positive_negative_prefix": "same clean teacher-forced action prefix",
        "gripper": "clean/unmodified", "counterfactuals": {
            "full": "four-neighbor 16x16 harmonic extension",
            "zero": "zero projector feature on selected tokens",
            "clean_global_mean": "same-state mean of clean projector visual tokens",
        },
        "new_closed_loop_episodes": 2400,
    }
    lockpath = artifact / "CONFIG_LOCK.json"
    if lockpath.exists() and json.loads(lockpath.read_text()) != lock:
        raise RuntimeError(f"CONFIG_LOCK mismatch: {lockpath}")
    lockpath.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")

    env, env_id = shared.make_environment(args.task)
    cfg = get_policy_config("openvla", str(PCD_SOURCE / "pretrained/openvla-7b"), args.task, {}, False)
    base = OpenVLAInference(**cfg)
    policies = {}
    policies.update(table2.build_policies(base, args.task, "6", ("full",)))
    policies.update(table2.build_policies(base, args.task, "1", ("zero", "clean_global_mean")))
    for seed in parse_seeds(args.seeds):
        snap_dir = snaproot / "snapshots" / args.task
        ref_dir = snaproot / "references" / args.task
        snap_dir.mkdir(parents=True, exist_ok=True)
        ref_dir.mkdir(parents=True, exist_ok=True)
        snap_path = snap_dir / f"seed_{seed:03d}.pkl"
        ref_path = ref_dir / f"seed_{seed:03d}.json"
        if not snap_path.exists() or not ref_path.exists():
            snap = capture_snapshot(env, seed)
            obs, state_hash, rgb_hash = restore_snapshot(env, seed, snap)
            reference = {
                "task": args.task, "seed": seed, "environment_id": env_id,
                "instruction": str(env.unwrapped.get_language_instruction()),
                "canonical_snapshot_sha256": snapshot_sha(snap),
                "initial_state_sha256": state_hash, "initial_rgb_sha256": rgb_hash,
            }
            tmp = snap_path.with_suffix(f".tmp.{os.getpid()}")
            tmp.write_bytes(pickle.dumps(snap, protocol=pickle.HIGHEST_PROTOCOL))
            os.replace(tmp, snap_path)
            tmpj = ref_path.with_suffix(f".tmp.{os.getpid()}")
            tmpj.write_text(json.dumps(reference, indent=2, sort_keys=True) + "\n")
            os.replace(tmpj, ref_path)
        snap = pickle.loads(snap_path.read_bytes())
        reference = json.loads(ref_path.read_text())
        if snapshot_sha(snap) != reference["canonical_snapshot_sha256"]:
            raise RuntimeError(f"snapshot checksum mismatch {args.task}:{seed}")
        hashes = []
        for arm in ARMS:
            out = artifact / "episodes" / args.task / arm
            out.mkdir(parents=True, exist_ok=True)
            summary_path = out / f"episode_{seed:03d}_summary.json"
            arrays_path = out / f"episode_{seed:03d}_arrays.npz"
            if summary_path.exists() and arrays_path.exists():
                old = json.loads(summary_path.read_text())
                if old.get("protocol_id") != PROTOCOL:
                    raise RuntimeError(f"incompatible existing result {summary_path}")
                hashes.append((old["initial_state_sha256"], old["initial_rgb_sha256"]))
                continue
            obs, state_hash, rgb_hash = restore_snapshot(env, seed, snap)
            expected = (reference["initial_state_sha256"], reference["initial_rgb_sha256"])
            if (state_hash, rgb_hash) != expected:
                raise RuntimeError(f"paired initial state/RGB mismatch {args.task}:{seed}:{arm}")
            instruction = str(env.unwrapped.get_language_instruction())
            if instruction != reference["instruction"]:
                raise RuntimeError(f"instruction mismatch {args.task}:{seed}:{arm}")
            policy = policies[arm]
            policy.reset(instruction, seed=seed)
            policy._episode_trace, policy._episode_logits = [], []
            started = time.monotonic()
            result, steps, reason, actions, jitter = run_episode(env, policy, instruction, obs)
            trace = table2.jsonable(policy._episode_trace)
            card = "6" if arm == "full" else "1"
            checks = table2.audit(trace, arm, card)
            table2.write_arrays(arrays_path, policy._episode_logits, actions)
            payload = {
                "protocol_id": PROTOCOL, "task": args.task, "seed": seed, "episode_id": seed,
                "phase": args.phase, "arm": arm, "success": bool(result["success"]),
                "result": table2.jsonable(result), "failure_reason": reason, "steps": steps,
                "runtime_seconds": time.monotonic() - started, "worker_id": args.worker_id,
                "gpu_id": args.gpu, "environment_id": env_id,
                "canonical_snapshot_sha256": reference["canonical_snapshot_sha256"],
                "initial_state_sha256": state_hash, "initial_rgb_sha256": rgb_hash,
                "instruction": instruction, "rendering": lock["task_renderer"][args.task],
                "lambda": 0.5, "attention_layers": [11], "arm_configuration": lock["counterfactuals"][arm],
                "arrays_file": arrays_path.name, "action_jitter_index": jitter,
                "mean_selected_token_count": table2.finite_mean(x.get("actual_selected_count") for x in trace),
                "selector_trace": trace, **checks,
            }
            table2.atomic_json(summary_path, payload)
            hashes.append((state_hash, rgb_hash))
            print(json.dumps({"event": "episode_complete", "task": args.task, "seed": seed,
                              "arm": arm, "success": payload["success"], "steps": steps,
                              "runtime_seconds": round(payload["runtime_seconds"], 2)}, sort_keys=True), flush=True)
        if len(set(hashes)) != 1:
            raise RuntimeError(f"paired-arm hash mismatch {args.task}:{seed}: {hashes}")
    env.close()


if __name__ == "__main__":
    main()
