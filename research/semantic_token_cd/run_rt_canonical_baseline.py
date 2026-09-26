#!/usr/bin/env python3
"""Regenerate paired Vanilla / historical L11-Matched baselines under RT."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path("/home/leju-suzhou/zjt_ws/token-cd")
PCD_SOURCE = Path("/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source")
SNAPSHOTS = REPO / "artifacts/vanilla_recon_shr_canonical_0_299_v2"
TASKS = ("google_robot_pick_coke_can", "google_robot_move_near")
ARMS = ("vanilla", "l11_matched")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else str(x))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, choices=(1, 2, 3), required=True)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--worker-index", type=int, required=True)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--shader-dir", choices=("rt",), default="rt")
    a = ap.parse_args()
    if not 0 <= a.worker_index < a.workers:
        raise ValueError("worker-index must be in [0, workers)")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    for p in (REPO / "task1/shim_site", REPO, PCD_SOURCE):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    from properties import get_policy_config
    import torch
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    from research.semantic_token_cd.distractor_rollout import PCD_SOURCE as ROLLOUT_PCD, jsonable, restore_snapshot, snapshot_sha
    from research.semantic_token_cd.distractor_policy import AuditedVanillaInference
    from research.semantic_token_cd.spatial_grid_rollout import make_environment, run_episode
    from research.semantic_token_cd.l11_causal_group_preflight import build_historical_policy
    from research.semantic_token_cd.prompt_attn_shr_rollout import write_arrays

    jobs = [(task, seed) for task in TASKS for seed in range(100, 200)]
    jobs = [(i, task, seed) for i, (task, seed) in enumerate(jobs) if i % a.workers == a.worker_index]
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = None
    current = None
    policies = None
    base = vanilla = matched = policy = None
    for job_id, task, seed in jobs:
        try:
            if current != task:
                if env is not None:
                    env.close()
                    env = None
                # The next task uses the same checkpoint but new task-specific
                # policy objects. Drop every shallow copy of the old model
                # before loading another copy, otherwise each worker retains
                # one 7B model across the task boundary and runs out of VRAM.
                if policies is not None:
                    policies.clear()
                policies = None
                policy = base = vanilla = matched = None
                gc.collect()
                torch.cuda.empty_cache()
                env, env_id = make_environment(task, shader_dir=a.shader_dir)
                cfg = get_policy_config("openvla", str(ROLLOUT_PCD / "pretrained/openvla-7b"), task, {}, False)
                base = OpenVLAInference(**cfg)
                vanilla = copy.copy(base)
                vanilla.__class__ = AuditedVanillaInference
                vanilla._episode_trace, vanilla._episode_logits = [], []
                matched = build_historical_policy(base, task)
                policies = {"vanilla": vanilla, "l11_matched": matched}
                current = task

            for arm, policy in policies.items():
                arm_dir = out / "episodes" / task / arm
                summary_path = arm_dir / f"episode_{seed:03d}_summary.json"
                arrays_path = arm_dir / f"episode_{seed:03d}_arrays.npz"
                if summary_path.exists() and arrays_path.exists():
                    continue
                snapshot_path = SNAPSHOTS / "snapshots" / task / f"seed_{seed:03d}.pkl"
                with snapshot_path.open("rb") as f:
                    snapshot = pickle.load(f)
                obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
                instruction = env.unwrapped.get_language_instruction()
                policy.reset(instruction, seed=seed)
                policy._episode_trace = []
                policy._episode_logits = []
                started = time.monotonic()
                result, steps, reason, actions, jitter = run_episode(env, policy, instruction, obs)
                runtime = time.monotonic() - started
                trace = jsonable(policy._episode_trace)
                records = policy._episode_logits
                arm_dir.mkdir(parents=True, exist_ok=True)
                write_arrays(arrays_path, records, actions)
                summary = {
                    "protocol_id": "L11_CAUSAL_GROUP_RT_CANONICAL_BASELINE_V1",
                    "renderer": "rt", "task": task, "seed": seed, "episode_id": seed,
                    "arm": arm, "environment_id": env_id, "instruction": instruction,
                    "success": bool(result["success"]), "result": jsonable(result),
                    "failure_reason": reason, "control_steps": steps, "runtime_seconds": runtime,
                    "gpu_id": a.gpu, "worker_id": a.worker_id,
                    "canonical_snapshot_sha256": snapshot_sha(snapshot),
                    "initial_state_sha256": state_sha, "initial_rgb_sha256": rgb_sha,
                    "selector_trace": trace, "arrays_file": arrays_path.name,
                    "action_jitter_index": jitter,
                }
                atomic_json(summary_path, summary)
                print(json.dumps({"worker": a.worker_id, "task": task, "seed": seed, "arm": arm,
                                  "success": summary["success"], "steps": steps, "runtime_s": round(runtime, 2),
                                  "state": state_sha, "rgb": rgb_sha}), flush=True)
        except Exception as e:
            fail = out / "logs" / f"failure_{a.worker_id}_{task}_{seed}_{int(time.time())}.json"
            atomic_json(fail, {"task": task, "seed": seed, "error": repr(e), "traceback": traceback.format_exc()})
            print(json.dumps({"worker": a.worker_id, "task": task, "seed": seed, "status": "FAILED", "error": repr(e)}), flush=True)
            raise
    if env is not None:
        env.close()


if __name__ == "__main__":
    main()
