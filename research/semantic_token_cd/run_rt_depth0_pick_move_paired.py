#!/usr/bin/env python3
"""Paired Vanilla vs canonical complete historical L11-Matched on SIMPLER RT depth 0."""
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

ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
PCD = Path("/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source")
CANON = ROOT / "artifacts/vanilla_recon_shr_canonical_0_299_v2"
PICK_RT0_REF = ROOT / "artifacts/l11_rank_causal_effect_study_v1_20260926/closedloop_simpler/google_robot_pick_coke_can/vanilla"
TASKS = {"google_robot_pick_coke_can", "google_robot_move_near"}
PROTOCOL = "L11_RT_DEPTH0_PICK_MOVE_FULL_MATCHED_100_129_V1"
RENDERER = "SAPIEN RT depth0, spp=32, denoiser=true"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            out.extend(range(lo, hi + 1))
        elif part:
            out.append(int(part))
    seeds = sorted(set(out))
    if not seeds or any(s < 100 or s > 129 for s in seeds):
        raise ValueError("This run is locked to seeds 100-129")
    return seeds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, choices=(1, 2, 3), required=True)
    ap.add_argument("--task", choices=sorted(TASKS), required=True)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    seeds = parse_seeds(a.seeds)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    for p in (ROOT / "task1/shim_site", ROOT, PCD):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    import torch
    torch.set_num_threads(1)
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    from research.semantic_token_cd.distractor_rollout import jsonable, restore_snapshot, snapshot_sha
    from research.semantic_token_cd.distractor_policy import AuditedVanillaInference
    from research.semantic_token_cd.l11_causal_group_preflight import build_historical_policy
    from research.semantic_token_cd.prompt_attn_shr_rollout import write_arrays
    from research.semantic_token_cd.replay_simpler_l11_rank_effects import make_rt_depth0_environment
    from research.semantic_token_cd.spatial_grid_rollout import run_episode

    out = a.output.resolve()
    env = base = vanilla = matched = None
    try:
        env = make_rt_depth0_environment(a.task)
        cfg = get_policy_config("openvla", str(PCD / "pretrained/openvla-7b"), a.task, {}, False)
        base = OpenVLAInference(**cfg)
        vanilla = copy.copy(base)
        vanilla.__class__ = AuditedVanillaInference
        vanilla._episode_trace, vanilla._episode_logits = [], []
        matched = build_historical_policy(base, a.task)
        policies = (("vanilla", vanilla), ("l11_matched", matched))

        for seed in seeds:
            snap_path = CANON / "snapshots" / a.task / f"seed_{seed:03d}.pkl"
            ref_path = CANON / "episodes" / a.task / "vanilla" / f"episode_{seed:03d}_summary.json"
            with snap_path.open("rb") as f:
                snapshot = pickle.load(f)
            ref = json.loads(ref_path.read_text())
            canonical_sha = snapshot_sha(snapshot)
            if canonical_sha != ref.get("canonical_snapshot_sha256"):
                raise RuntimeError(f"canonical snapshot mismatch {a.task}:{seed}")

            pair_rgb = None
            for arm, policy in policies:
                arm_dir = out / "episodes" / a.task / arm
                summary_path = arm_dir / f"episode_{seed:03d}_summary.json"
                arrays_path = arm_dir / f"episode_{seed:03d}_arrays.npz"
                if summary_path.exists() and arrays_path.exists():
                    prior = json.loads(summary_path.read_text())
                    if prior.get("protocol_id") != PROTOCOL or prior.get("renderer") != RENDERER:
                        raise RuntimeError(f"existing output has incompatible protocol: {summary_path}")
                    pair_rgb = pair_rgb or prior.get("initial_rgb_sha256")
                    continue

                obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
                if state_sha != ref.get("initial_state_sha256"):
                    raise RuntimeError(f"restored state mismatch {a.task}:{seed}")
                if a.task == "google_robot_pick_coke_can":
                    pick_ref = json.loads((PICK_RT0_REF / f"seed_{seed:03d}.json").read_text())
                    if (pick_ref.get("canonical_snapshot_sha256") != canonical_sha or
                            pick_ref.get("initial_state_sha256") != state_sha or
                            pick_ref.get("initial_rgb_sha256_rt0") != rgb_sha):
                        raise RuntimeError(f"Pick-Coke RT0 historical RGB/state mismatch at seed {seed}")
                if pair_rgb is not None and rgb_sha != pair_rgb:
                    raise RuntimeError(f"Vanilla/L11 RT0 initial RGB mismatch {a.task}:{seed}")
                pair_rgb = rgb_sha
                instruction = env.unwrapped.get_language_instruction()
                if instruction != ref.get("instruction"):
                    raise RuntimeError(f"instruction mismatch {a.task}:{seed}")
                arm_dir.mkdir(parents=True, exist_ok=True)
                policy.reset(instruction, seed=seed)
                policy._episode_trace, policy._episode_logits = [], []
                started = time.monotonic()
                result, steps, reason, actions, jitter = run_episode(env, policy, instruction, obs)
                elapsed = time.monotonic() - started
                trace = jsonable(policy._episode_trace)
                if arm == "l11_matched":
                    if not trace or not all(row.get("coverage_exact") is True for row in trace):
                        raise RuntimeError(f"historical Matched coverage audit failed {a.task}:{seed}")
                    if not all(abs(float(row.get("lambda", -1)) - 0.5) < 1e-12 for row in trace):
                        raise RuntimeError(f"historical lambda audit failed {a.task}:{seed}")
                write_arrays(arrays_path, policy._episode_logits, actions)
                payload = {
                    "protocol_id": PROTOCOL, "benchmark": "SIMPLER", "task": a.task,
                    "seed": seed, "arm": arm,
                    "selector": "historical PromptAttentionSHRInference matched-budget path" if arm == "l11_matched" else "Vanilla",
                    "success": bool(result["success"]), "steps": int(steps), "failure_reason": reason,
                    "runtime_seconds": elapsed, "action_jitter_index": jitter,
                    "canonical_snapshot_sha256": canonical_sha, "initial_state_sha256": state_sha,
                    "initial_rgb_sha256": rgb_sha, "renderer": RENDERER,
                    "renderer_gpu": a.gpu, "inference_gpu": a.gpu,
                    "lambda": 0.5 if arm == "l11_matched" else 0.0,
                    "mean_m_t": float(np.mean([row["m_t"] for row in trace])) if arm == "l11_matched" and trace else 0.0,
                    "sampled_states": len(trace), "selector_trace": trace,
                    "arrays_file": arrays_path.name, "worker_id": a.worker_id,
                    "historical_snapshot_reference": str(snap_path),
                }
                atomic_json(summary_path, payload)
                print(json.dumps({"worker": a.worker_id, "gpu": a.gpu, "task": a.task,
                                  "seed": seed, "arm": arm, "success": payload["success"],
                                  "steps": steps, "seconds": round(elapsed, 1),
                                  "mean_m_t": round(payload["mean_m_t"], 2),
                                  "rgb_sha256": rgb_sha}, sort_keys=True), flush=True)
    except Exception as exc:
        atomic_json(out / "logs" / f"failure_{a.worker_id}_{int(time.time())}.json",
                    {"task": a.task, "seeds": a.seeds, "gpu": a.gpu, "error": repr(exc),
                     "traceback": traceback.format_exc()})
        raise
    finally:
        if env is not None:
            env.close()
        del matched, vanilla, base, env
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
