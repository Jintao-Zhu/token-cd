#!/usr/bin/env python3
"""Run canonical historical L11-Matched on SIMPLER RT depth-0, seeds 100-129."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import pickle
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
PCD = Path("/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source")
CANON = ROOT / "artifacts/vanilla_recon_shr_canonical_0_299_v2"
RANK = ROOT / "artifacts/l11_rank_causal_effect_study_v1_20260926/closedloop_simpler"
PICK_REUSE = ROOT / "artifacts/rt_appearance_factor_pilot/closed_loop/rt_depth0/google_robot_pick_coke_can/l11_matched"
TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    shutil.copy2(source, temp)
    os.replace(temp, destination)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--seeds", required=True, help="Inclusive range, e.g. 100-114")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-", 1))
    if lo < 100 or hi > 129 or lo > hi:
        raise ValueError("this run is restricted to seeds 100-129")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    for path in (ROOT / "task1/shim_site", ROOT, PCD):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import torch
    torch.set_num_threads(1)
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    from research.semantic_token_cd.distractor_rollout import jsonable, restore_snapshot, snapshot_sha
    from research.semantic_token_cd.prompt_attn_shr_rollout import write_arrays
    from research.semantic_token_cd.spatial_grid_rollout import run_episode
    from research.semantic_token_cd.l11_causal_group_preflight import build_historical_policy
    from research.semantic_token_cd.replay_simpler_l11_rank_effects import make_rt_depth0_environment

    out = a.output.resolve()
    seeds = range(lo, hi + 1)
    current_task = None
    env = base = policy = None
    failure_seed = lo
    for task in TASKS:
        # The existing 30-seed Pick-Coke RT-depth0 Matched arm is reused only
        # after exact snapshot, restored-state, and initial-RGB checks.
        if task == "google_robot_pick_coke_can":
            for seed in seeds:
                failure_seed = seed
                src = PICK_REUSE / f"episode_{seed:03d}_summary.json"
                src_arrays = PICK_REUSE / f"episode_{seed:03d}_arrays.npz"
                ref = RANK / task / "vanilla" / f"seed_{seed:03d}.json"
                if not src.exists() or not src_arrays.exists() or not ref.exists():
                    raise FileNotFoundError(f"missing Pick-Coke reuse/reference for seed {seed}")
                old, rank = json.loads(src.read_text()), json.loads(ref.read_text())
                if not (
                    old.get("variant") == "rt_depth0"
                    and old.get("arm") == "l11_matched"
                    and old.get("canonical_snapshot_sha256") == rank.get("canonical_snapshot_sha256")
                    and old.get("initial_state_sha256") == rank.get("initial_state_sha256")
                    and old.get("initial_rgb_sha256") == rank.get("initial_rgb_sha256_rt0")
                ):
                    raise RuntimeError(f"Pick-Coke RT-depth0 reuse hash/config mismatch at seed {seed}")
                dest = out / task / "l11_matched"
                old["protocol_id"] = "CANONICAL_L11_MATCHED_RT_DEPTH0_30SEED_V1"
                old["renderer"] = "SAPIEN RT depth0, spp=32, denoiser=true"
                old["arrays_file"] = f"seed_{seed:03d}_arrays.npz"
                old["reused_from"] = str(src)
                atomic_json(dest / f"seed_{seed:03d}_summary.json", old)
                atomic_copy(src_arrays, dest / f"seed_{seed:03d}_arrays.npz")
            continue

        try:
            if current_task != task:
                if env is not None:
                    env.close()
                policy = base = None
                gc.collect()
                torch.cuda.empty_cache()
                env = make_rt_depth0_environment(task)
                cfg = get_policy_config("openvla", str(PCD / "pretrained/openvla-7b"), task, {}, False)
                base = OpenVLAInference(**cfg)
                policy = build_historical_policy(base, task)
                current_task = task

            for seed in seeds:
                task_out = out / task / "l11_matched"
                summary_path = task_out / f"seed_{seed:03d}_summary.json"
                arrays_path = task_out / f"seed_{seed:03d}_arrays.npz"
                if summary_path.exists() and arrays_path.exists():
                    continue
                snapshot_path = CANON / "snapshots" / task / f"seed_{seed:03d}.pkl"
                with snapshot_path.open("rb") as f:
                    snapshot = pickle.load(f)
                canonical_sha = snapshot_sha(snapshot)
                ref_path = CANON / "episodes" / task / "vanilla" / f"episode_{seed:03d}_summary.json"
                ref = json.loads(ref_path.read_text())
                if canonical_sha != ref["canonical_snapshot_sha256"]:
                    raise RuntimeError(f"canonical snapshot mismatch {task}:{seed}")

                obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
                if state_sha != ref["initial_state_sha256"]:
                    raise RuntimeError(f"restored state mismatch {task}:{seed}")
                rank_ref = RANK / task / "vanilla" / f"seed_{seed:03d}.json"
                if rank_ref.exists():
                    rank = json.loads(rank_ref.read_text())
                    if rank.get("canonical_snapshot_sha256") != canonical_sha or rank.get("initial_state_sha256") != state_sha:
                        raise RuntimeError(f"rank-study snapshot/state mismatch {task}:{seed}")
                    if rank.get("initial_rgb_sha256_rt0") != rgb_sha:
                        raise RuntimeError(f"RT-depth0 RGB mismatch against rank study {task}:{seed}")

                instruction = env.unwrapped.get_language_instruction()
                if instruction != ref["instruction"]:
                    raise RuntimeError(f"instruction mismatch {task}:{seed}")
                policy.reset(instruction, seed=seed)
                policy._episode_trace = []
                policy._episode_logits = []
                started = time.monotonic()
                result, steps, reason, actions, jitter = run_episode(env, policy, instruction, obs)
                elapsed = time.monotonic() - started
                trace = jsonable(policy._episode_trace)
                records = policy._episode_logits
                if not trace or not all(row.get("coverage_exact") is True for row in trace):
                    raise RuntimeError(f"canonical Matched coverage audit failed {task}:{seed}")
                if not all(abs(float(row.get("lambda", -1)) - 0.5) < 1e-12 for row in trace):
                    raise RuntimeError(f"canonical lambda audit failed {task}:{seed}")

                task_out.mkdir(parents=True, exist_ok=True)
                write_arrays(arrays_path, records, actions)
                payload = {
                    "protocol_id": "CANONICAL_L11_MATCHED_RT_DEPTH0_30SEED_V1",
                    "benchmark": "SIMPLER",
                    "task": task,
                    "seed": seed,
                    "arm": "l11_matched",
                    "selector": "historical PromptAttentionSHRInference matched-budget path",
                    "selected_k": None,
                    "success": bool(result["success"]),
                    "steps": int(steps),
                    "failure_reason": reason,
                    "runtime_seconds": elapsed,
                    "action_jitter_index": jitter,
                    "canonical_snapshot_sha256": canonical_sha,
                    "initial_state_sha256": state_sha,
                    "initial_rgb_sha256_rt0": rgb_sha,
                    "renderer": "SAPIEN RT depth0, spp=32, denoiser=true",
                    "renderer_gpu": a.gpu,
                    "inference_gpu": a.gpu,
                    "lambda": 0.5,
                    "mean_m_t": float(np.mean([row["m_t"] for row in trace])),
                    "sampled_states": len(trace),
                    "trace": trace,
                }
                atomic_json(summary_path, payload)
                print(json.dumps({"worker": a.worker_id, "task": task, "seed": seed,
                                  "success": payload["success"], "steps": steps,
                                  "runtime_s": round(elapsed, 2), "mean_m_t": round(payload["mean_m_t"], 2)}, sort_keys=True), flush=True)
        except Exception as e:
            fail = out / "logs" / f"failure_{a.worker_id}_{task}_{failure_seed}_{int(time.time())}.json"
            atomic_json(fail, {"task": task, "seed": failure_seed, "error": repr(e), "traceback": traceback.format_exc()})
            raise
    if env is not None:
        env.close()


if __name__ == "__main__":
    main()
