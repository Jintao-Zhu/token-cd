"""DTP OpenVLA calibration v1 closed-loop rollout worker.

Each (task, seed, arm) restores the identical canonical snapshot, runs one
episode under the DTP autoregressive adapter, and records summary/arrays/video.
Control = adapter with key masking disabled; l11_k64_t05/l7_k64_t05 are the two
pruning candidates selected by the offline stage.

Run from repository root, e.g.:
  PYTHONPATH=... python research/semantic_token_cd/dtp_closed_loop_rollout.py \
    --task google_robot_open_drawer --seeds 118 --arms control,l11_k64_t05,l7_k64_t05 \
    --gpu 1 --worker-id smoke --artifact-dir artifacts/dtp_openvla_calibration_v1/closed_loop
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from research.semantic_token_cd.distractor_rollout import (  # noqa: E402
    PCD_SOURCE,
    jsonable,
    restore_snapshot,
    snapshot_sha,
)
from research.semantic_token_cd.dtp_closed_loop_policy import (  # noqa: E402
    ACTION_DIM,
    build_policy,
)
from research.semantic_token_cd.dtp_closed_loop_protocol import (  # noqa: E402
    ARM_CONFIG,
    ARMS,
    ARTIFACT,
    CANONICAL_SNAPSHOTS,
    PROTOCOL,
    SPATIAL,
    TASKS,
)
from research.semantic_token_cd.xswap_rollout import make_environment  # noqa: E402


def parse_seeds(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            out.extend(range(lo, hi + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def run_loop(env, policy, instruction, obs, record_frames: bool):
    """Execute one episode. Returns result, actions, step count, reason, frames."""
    from parallel_inference import get_image_from_maniskill2_obs_dict
    from research.semantic_token_cd.rollout_pilot import flatten_action
    from utils import convert_numpy_or_torch_to_python, summarize

    image = get_image_from_maniskill2_obs_dict(env, obs)
    infos = []
    actions = []
    frames = [] if record_frames else None
    predicted = False
    truncated = False
    control = 0
    while not (predicted or truncated) and control < 240:
        if record_frames:
            frames.append(np.asarray(image, dtype=np.uint8))
        _raw, acts, _meta = policy.step(image, None, instruction, proprio=obs["agent"]["eef_pos"])
        if not isinstance(acts, list):
            acts = [acts]
        for act in acts:
            executed = flatten_action(act)
            if executed.shape != (ACTION_DIM,) or not np.isfinite(executed).all():
                raise FloatingPointError(f"invalid executed action at step {control}: {executed}")
            actions.append(executed.copy())
            obs, _reward, _success, truncated, info = env.step(executed)
            image = get_image_from_maniskill2_obs_dict(env, obs)
            control += 1
            infos.append(convert_numpy_or_torch_to_python(info))
            predicted = bool(act["terminate_episode"][0] > 0)
            if predicted and not env.unwrapped.is_final_subtask():
                predicted = False
                env.advance_to_next_subtask()
    if record_frames:
        frames.append(np.asarray(image, dtype=np.uint8))
    result = summarize(infos)
    if result.get("success"):
        reason = None
    elif truncated:
        reason = "environment_time_limit"
    else:
        reason = "policy_terminated_without_success"
    return result, np.asarray(actions, dtype=np.float32), infos, reason, frames


def audit(trace: list[dict], arm: str) -> dict:
    cfg = ARM_CONFIG[arm]
    checks = {
        "trace_nonempty": bool(trace),
        "all_finite": all(float(x.get("total_pruned", -1)) >= 0 for x in trace),
        "all_config_locked": all(
            int(x["layer"]) == int(cfg["layer"])
            and int(x["k"]) == int(cfg["k"])
            and abs(float(x["tau"]) - float(cfg["tau"])) < 1e-12
            and bool(x["enabled"]) is bool(cfg["enabled"])
            and str(x.get("mode", "dynamic")) == str(cfg.get("mode", "dynamic"))
            for x in trace
        ),
    }
    if not cfg["enabled"]:
        checks["control_never_prunes"] = all(not bool(x["any_prune"]) for x in trace)
    else:
        checks["control_never_prunes"] = True
    checks["technical_pass"] = bool(trace) and all(checks.values())
    if not checks["technical_pass"]:
        raise RuntimeError(f"DTP audit failed for {arm}: {checks}")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--arms", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--worker-id", default="manual")
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    arms = [arm for arm in args.arms.split(",") if arm]
    for arm in arms:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm}")
    artifact = args.artifact_dir.resolve()
    task_root = artifact / "episodes" / args.task
    env, environment_id = make_environment(args.task, args.gpu)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    base = OpenVLAInference(**get_policy_config("openvla", checkpoint, args.task, {}, False))

    for seed in parse_seeds(args.seeds):
        snapshot_path = CANONICAL_SNAPSHOTS / args.task / f"seed_{seed:03d}.pkl"
        if not snapshot_path.exists():
            raise FileNotFoundError(f"missing canonical snapshot: {snapshot_path}")
        with snapshot_path.open("rb") as handle:
            snapshot = pickle.load(handle)
        canonical = snapshot_sha(snapshot)
        seed_hashes = []
        for arm in arms:
            arm_root = task_root / arm
            summary_path = arm_root / f"episode_{seed:03d}_summary.json"
            arrays_path = arm_root / f"episode_{seed:03d}_arrays.npz"
            if summary_path.exists() and arrays_path.exists():
                print(json.dumps({"skip": True, "task": args.task, "seed": seed, "arm": arm}), flush=True)
                continue
            obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
            instruction = env.unwrapped.get_language_instruction()
            policy = build_policy(base, args.task, arm)
            policy.reset(instruction, seed=seed)
            policy._episode_trace = []
            policy._episode_step = 0
            record = bool(args.save_video)
            started = time.monotonic()
            result, actions, infos, reason, frames = run_loop(
                env, policy, instruction, obs, record_frames=record
            )
            runtime = time.monotonic() - started
            trace = jsonable(policy._episode_trace)
            checks = audit(policy._episode_trace, arm)
            step_counts = np.asarray(
                [x["pruned_per_dim"] for x in policy._episode_trace], dtype=np.int64
            ) if policy._episode_trace else np.zeros((0, ACTION_DIM), dtype=np.int64)
            flip_counts = np.asarray(
                [x["token_flips_per_dim"] for x in policy._episode_trace], dtype=np.int64
            ) if policy._episode_trace else np.zeros((0, ACTION_DIM), dtype=np.int64)

            video_rel = None
            video_sha = None
            if record:
                video_path = task_root / "videos" / arm / f"episode_{seed:03d}.mp4"
                video_path.parent.mkdir(parents=True, exist_ok=True)
                imageio.mimwrite(
                    video_path, frames, fps=10, codec="libx264", quality=7, macro_block_size=None
                )
                video_rel = str(video_path.relative_to(artifact))
                video_sha = sha256_file(video_path)
            arrays_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                arrays_path,
                executed_actions=np.asarray(actions, dtype=np.float32),
                pruned_per_dim=step_counts,
                token_flips_per_dim=flip_counts,
            )
            summary = {
                "protocol_id": PROTOCOL,
                "task": args.task,
                "environment_id": environment_id,
                "seed": seed,
                "episode_id": seed,
                "arm": arm,
                "layer": ARM_CONFIG[arm]["layer"],
                "k": ARM_CONFIG[arm]["k"],
                "tau": ARM_CONFIG[arm]["tau"],
                "enabled": ARM_CONFIG[arm]["enabled"],
                "mode": ARM_CONFIG[arm].get("mode", "dynamic"),
                "spatial": SPATIAL,
                "instruction": instruction,
                "success": bool(result.get("success", False)),
                "result": jsonable(result),
                "failure_reason": reason,
                "control_steps": len(infos),
                "runtime_seconds": runtime,
                "gpu_id": args.gpu,
                "worker_id": args.worker_id,
                "canonical_snapshot_sha256": canonical,
                "initial_state_sha256": state_sha,
                "initial_rgb_sha256": rgb_sha,
                "arrays_file": arrays_path.name,
                "video_file": video_rel,
                "video_sha256": video_sha,
                "prune_activation_steps": int((step_counts.sum(axis=1) > 0).sum()) if len(step_counts) else 0,
                "total_pruned_tokens": int(step_counts.sum()),
                "total_token_flips": int(flip_counts.sum()),
                "mean_pruned_per_dim": float(step_counts.mean()) if len(step_counts) else 0.0,
                "trace": trace,
                **checks,
            }
            atomic_json(summary_path, summary)
            seed_hashes.append((canonical, state_sha, rgb_sha))
            print(json.dumps({
                "task": args.task,
                "seed": seed,
                "arm": arm,
                "success": summary["success"],
                "steps": len(infos),
                "runtime_seconds": round(runtime, 2),
                "prune_activation_steps": summary["prune_activation_steps"],
                "technical_pass": True,
            }), flush=True)
        if len(set(seed_hashes)) not in (0, 1):
            raise RuntimeError(f"arm pairing mismatch on {args.task} seed={seed}")
    env.close()


if __name__ == "__main__":
    main()
