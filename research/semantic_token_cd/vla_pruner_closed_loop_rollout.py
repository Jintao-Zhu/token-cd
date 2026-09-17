"""VLA-Pruner reproduction -- 22-scene closed-loop calibration worker.

Protocol: VLA_PRUNER_OPENVLA_REPRODUCTION_V1 (closed-loop stage).
Arms per calibration scene (identical canonical snapshot per arm):
  - vanilla                : current-harness vanilla (reference for Rescue/Harm)
  - vla_pruner_prune25     : paper-main prefill + temporal, fastv_r=0.25
  - vla_pruner_prune50     : fastv_r=0.50

For every prune-arm env step we additionally run one vanilla reference forward
on the *same* observation so the per-step report can separate the local effect
of pruning (action-token flips, raw-action L1/L2) from trajectory drift.

Frozen VLA-Pruner config (CONFIG_LOCK.json):
  fastv_k=3, temporal window=3, gamma=0.8, prefill selection,
  layer-15 action->vision history, image tokens 1..256.

Run from repository root, e.g.:
  PYTHONPATH=... python research/semantic_token_cd/vla_pruner_closed_loop_rollout.py \
    --task google_robot_pick_coke_can --seeds 100-109 \
    --arms vanilla,vla_pruner_prune25,vla_pruner_prune50 \
    --gpu 2 --worker-id smoke --artifact-dir artifacts/vla_pruner_openvla_reproduction/closed_loop
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
PCD_ROOT = Path("/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e")
PCD_SOURCE = PCD_ROOT / "source"
CANONICAL = REPO_ROOT / "artifacts/vanilla_recon_shr_canonical_0_299_v2"
CANONICAL_SNAPSHOTS = CANONICAL / "snapshots"
DEFAULT_ARTIFACT = REPO_ROOT / "artifacts/vla_pruner_openvla_reproduction/closed_loop"

PROTOCOL = "VLA_PRUNER_OPENVLA_REPRODUCTION_V1"
ACTION_DIM = 7
ARMS = ("vanilla", "vla_pruner_prune25", "vla_pruner_prune50")
ARM_TARGET_R = {"vanilla": None, "vla_pruner_prune25": 0.25, "vla_pruner_prune50": 0.50}
ARM_IMAGE_KEEP = {"vanilla": None, "vla_pruner_prune25": 192, "vla_pruner_prune50": 128}
TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
)

for _p in (str(REPO_ROOT), str(PCD_SOURCE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


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
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def build_arms(base, task: str, arm_names) -> dict:
    from research.semantic_token_cd.vla_pruner_policy import build_vla_pruner_policy

    return {name: build_vla_pruner_policy(base, name, task) for name in arm_names}


def run_episode(env, policy, ref_policy, instruction, obs, record_frames: bool,
                need_ref: bool):
    """One closed-loop episode with optional per-step vanilla reference.

    Mirrors dtp_closed_loop_rollout.run_loop execution semantics (execute every
    returned action, advance subtasks, cap 240 env steps).  When need_ref is
    True a fresh vanilla forward runs on the exact observation before the arm's
    forward, so token flips / raw-action L1/L2 measure the local prune effect.
    """
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
    ref_tokens = []
    flips = []
    raw_l1 = []
    raw_l2 = []
    step_ms = []
    while not (predicted or truncated) and control < 240:
        if record_frames:
            frames.append(np.asarray(image, dtype=np.uint8))
        proprio = obs["agent"]["eef_pos"]
        t0 = time.monotonic()
        if need_ref:
            _r0, _ra0, _ = ref_policy.step(image, None, instruction, proprio=proprio)
            ref_trace = ref_policy._episode_trace[-1]
            r_tok = list(ref_trace["token_ids"])
            r_raw = np.asarray(ref_trace["raw_action"], dtype=np.float64)
        else:
            r_tok, r_raw = None, None
        _raw, acts, _meta = policy.step(image, None, instruction, proprio=proprio)
        trace = policy._episode_trace[-1]
        p_tok = list(trace["token_ids"])
        p_raw = np.asarray(trace["raw_action"], dtype=np.float64)
        if need_ref:
            if len(p_tok) != len(r_tok):
                raise RuntimeError("prune/ref token length mismatch in episode step")
            flips.append([int(a != b) for a, b in zip(p_tok, r_tok)])
            ref_tokens.append(r_tok)
            raw_l1.append(float(np.abs(p_raw - r_raw).sum()))
            raw_l2.append(float(np.sqrt(((p_raw - r_raw) ** 2).sum())))
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
        step_ms.append(1000.0 * (time.monotonic() - t0))
    if record_frames:
        frames.append(np.asarray(image, dtype=np.uint8))
    result = summarize(infos)
    if result.get("success"):
        reason = None
    elif truncated:
        reason = "environment_time_limit"
    else:
        reason = "policy_terminated_without_success"
    return {
        "result": result,
        "actions": np.asarray(actions, dtype=np.float32),
        "infos": infos,
        "reason": reason,
        "frames": frames,
        "ref_tokens": (np.asarray(ref_tokens, dtype=np.int64)
                       if ref_tokens else np.zeros((0, ACTION_DIM), dtype=np.int64)),
        "flips": (np.asarray(flips, dtype=np.int64)
                  if flips else np.zeros((0, ACTION_DIM), dtype=np.int64)),
        "raw_l1": np.asarray(raw_l1, dtype=np.float64),
        "raw_l2": np.asarray(raw_l2, dtype=np.float64),
        "step_ms": np.asarray(step_ms, dtype=np.float64),
    }


def audit(trace: list[dict], arm: str, need_ref: bool, flips_first: np.ndarray,
          n_ref_steps: int) -> dict:
    target_r = ARM_TARGET_R[arm]
    checks = {"trace_nonempty": bool(trace)}
    if arm == "vanilla":
        checks["never_prunes"] = all("pruned_count" not in x for x in trace)
        checks["all_finite_raw"] = all(
            np.isfinite(x["raw_action"]).all() for x in trace
        )
        checks["technical_pass"] = bool(trace) and all(checks.values())
        return checks
    # Prune arms: every trace row comes from the fastv branch (r=0 during the
    # three-step history warm-up, target r afterwards).
    cfg_locked = []
    warmup_ok = []
    prune_rows = []
    for x in trace:
        hist = int(x.get("history_len", -1))
        r_eff = float(x.get("fastv_r_effective", -1.0))
        prune_count = int(x.get("pruned_count", -1))
        kept_img = x.get("kept_image_count", None)
        layer = x.get("pruning_layer", None)
        if hist < 3:
            warmup_ok.append(
                r_eff == 0.0 and prune_count == 0 and kept_img == 256
                and layer == 3
            )
        else:
            prune_rows.append(x)
            cfg_locked.append(
                abs(r_eff - float(target_r)) < 1e-9
                and prune_count > 0
                and kept_img == ARM_IMAGE_KEEP[arm]
                and layer == 3
            )
    checks["first_trace_history_zero"] = int(trace[0].get("history_len", -1)) == 0
    checks["warmup_steps_never_prune"] = bool(warmup_ok) and all(warmup_ok)
    checks["pruning_layer_locked_3"] = all(int(x.get("pruning_layer", -1)) == 3 for x in trace)
    checks["prune_rows_cfg_locked"] = (not prune_rows) or all(cfg_locked)
    checks["kept_image_count_violations"] = not any(
        x.get("kept_image_count") not in (256, ARM_IMAGE_KEEP[arm])
        for x in trace
    )
    checks["original_seq_constant"] = len({int(x["original_seq_length"]) for x in trace}) <= 1
    checks["all_finite_raw"] = all(np.isfinite(x["raw_action"]).all() for x in trace)
    # Per-step vanilla-reference invariants (need_ref arms only).  flips_first
    # holds only the warm-up rows (history_len < 3), which must be vanilla-exact.
    checks["warmup_exact_vs_ref"] = True
    checks["ref_step_count_matches"] = True
    if need_ref:
        warmup_rows = int(sum(int(x.get("history_len", -1)) < 3 for x in trace))
        checks["warmup_exact_vs_ref"] = (
            warmup_rows == int(flips_first.shape[0])
            and bool(np.count_nonzero(flips_first) == 0)
        )
        checks["ref_step_count_matches"] = n_ref_steps == len(trace)
    checks["technical_pass"] = all(bool(v) for k, v in checks.items() if k != "technical_pass")
    return checks


def summarize_trace(trace: list[dict], flips: np.ndarray, raw_l1: np.ndarray,
                    raw_l2: np.ndarray, need_ref: bool) -> dict:
    n = len(trace)
    out = {"trace_steps": n}
    if not n:
        return out
    hist = np.asarray([int(x.get("history_len", -1)) for x in trace], dtype=np.int64)
    prune_counts = np.asarray(
        [int(x.get("pruned_count", 0) or 0) for x in trace], dtype=np.int64
    )
    kept_img = np.asarray(
        [int(x.get("kept_image_count", 256)) for x in trace], dtype=np.int64
    )
    overlap = np.asarray(
        [x.get("guide_topk_overlap") if x.get("guide_topk_overlap") is not None else -1
         for x in trace], dtype=np.int64
    )
    active = hist >= 3
    out["prune_activation_steps"] = int((active & (prune_counts > 0)).sum())
    out["total_pruned_tokens"] = int(prune_counts.sum())
    out["mean_kept_image_count_active"] = float(kept_img[active].mean()) if active.any() else None
    out["mean_kept_image_count_all"] = float(kept_img.mean())
    # early / middle / late buckets over trace step index.
    bucket = np.zeros(n, dtype=np.int64)
    third = max(1, n // 3)
    for i in range(n):
        bucket[i] = min(2, i // third) if third else 0
    for label, b in (("early", 0), ("middle", 1), ("late", 2)):
        sel = (bucket == b) & active
        out[f"phase_{label}_prune_steps"] = int((sel & (prune_counts > 0)).sum())
        out[f"phase_{label}_pruned_tokens"] = int(prune_counts[sel].sum())
    out["guide_topk_overlap"] = (
        float(overlap[overlap >= 0].mean()) if (overlap >= 0).any() else None
    )
    out["overlap_used_redundancy_steps"] = int(
        sum(1 for x in trace if x.get("used_redundancy"))
    )
    if need_ref:
        out["total_token_flips"] = int(flips.sum())
        out["flip_steps"] = int((flips.sum(axis=1) > 0).sum())
        out["mean_raw_l1"] = float(raw_l1.mean()) if len(raw_l1) else None
        out["mean_raw_l2"] = float(raw_l2.mean()) if len(raw_l2) else None
        out["total_raw_l1"] = float(raw_l1.sum()) if len(raw_l1) else None
        out["total_raw_l2"] = float(raw_l2.sum()) if len(raw_l2) else None
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--arms", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--worker-id", default="manual")
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import imageio.v2 as imageio

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    from research.semantic_token_cd.dtp_closed_loop_rollout import (  # noqa: E402
        make_environment,
    )
    from research.semantic_token_cd.rollout_pilot import (  # noqa: E402
        jsonable,
        restore_snapshot,
        snapshot_sha,
    )

    arms = [arm for arm in args.arms.split(",") if arm]
    for arm in arms:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm}")
    prune_arms = [arm for arm in arms if arm != "vanilla"]
    if prune_arms and "vanilla" not in arms:
        # Reference forward for per-step flips / L1 / L2 needs a vanilla policy
        # even when the vanilla arm is not requested.
        arms.append("vanilla")
    artifact = args.artifact_dir.resolve()
    task_root = artifact / "episodes" / args.task
    env, environment_id = make_environment(args.task, args.gpu)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    base = OpenVLAInference(**get_policy_config("openvla", checkpoint, args.task, {}, False))
    policies = build_arms(base, args.task, arms)

    for seed in parse_seeds(args.seeds):
        snapshot_path = CANONICAL_SNAPSHOTS / args.task / f"seed_{seed:03d}.pkl"
        if not snapshot_path.exists():
            raise FileNotFoundError(f"missing canonical snapshot: {snapshot_path}")
        with snapshot_path.open("rb") as handle:
            snapshot = pickle.load(handle)
        canonical = snapshot_sha(snapshot)
        for arm in arms:
            arm_root = task_root / arm
            summary_path = arm_root / f"episode_{seed:03d}_summary.json"
            arrays_path = arm_root / f"episode_{seed:03d}_arrays.npz"
            if summary_path.exists() and arrays_path.exists():
                print(json.dumps({"skip": True, "task": args.task, "seed": seed, "arm": arm}),
                      flush=True)
                continue
            obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
            instruction = env.unwrapped.get_language_instruction()
            policy = policies[arm]
            policy.reset(instruction, seed=seed)
            policy._episode_trace = []
            policy._episode_step = 0
            need_ref = arm in prune_arms
            ref_policy = policies["vanilla"] if need_ref else None
            if need_ref:
                ref_policy.reset(instruction, seed=seed)
                ref_policy._episode_trace = []
                ref_policy._episode_step = 0
            record = bool(args.save_video)
            started = time.monotonic()
            ep = run_episode(env, policy, ref_policy, instruction, obs,
                             record_frames=record, need_ref=need_ref)
            runtime = time.monotonic() - started
            trace = jsonable(policy._episode_trace)
            trace_typed = list(policy._episode_trace)
            flips = ep["flips"] if need_ref else np.zeros((0, ACTION_DIM), dtype=np.int64)
            n_ref = int(ep["ref_tokens"].shape[0]) if need_ref else 0
            n_warmup = int(sum(int(x.get("history_len", -1)) < 3 for x in trace_typed))
            warmup_flips = flips[:n_warmup]
            checks = audit(trace_typed, arm, need_ref, warmup_flips, n_ref)
            if not checks["technical_pass"]:
                raise RuntimeError(f"VLA-Pruner audit failed for {arm} seed={seed}: {checks}")
            stats = summarize_trace(trace_typed, flips, ep["raw_l1"], ep["raw_l2"], need_ref)

            video_rel = None
            video_sha = None
            if record and ep["frames"]:
                video_path = task_root / "videos" / arm / f"episode_{seed:03d}.mp4"
                video_path.parent.mkdir(parents=True, exist_ok=True)
                imageio.mimwrite(video_path, ep["frames"], fps=10, codec="libx264",
                                 quality=7, macro_block_size=None)
                video_rel = str(video_path.relative_to(artifact))
                video_sha = sha256_file(video_path)

            arrays_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                arrays_path,
                executed_actions=np.asarray(ep["actions"], dtype=np.float32),
                ref_token_ids=ep["ref_tokens"],
                token_flips=flips,
                raw_l1=ep["raw_l1"],
                raw_l2=ep["raw_l2"],
                step_ms=ep["step_ms"],
            )
            summary = {
                "protocol_id": PROTOCOL,
                "task": args.task,
                "environment_id": environment_id,
                "seed": seed,
                "episode_id": seed,
                "arm": arm,
                "fastv_r": ARM_TARGET_R[arm],
                "fastv_k": 3,
                "temporal_w": 3,
                "temporal_gamma": 0.8,
                "selection": "prefill",
                "history_layer": 15,
                "instruction": instruction,
                "success": bool(ep["result"].get("success", False)),
                "result": jsonable(ep["result"]),
                "failure_reason": ep["reason"],
                "control_steps": len(ep["infos"]),
                "runtime_seconds": runtime,
                "gpu_id": args.gpu,
                "worker_id": args.worker_id,
                "canonical_snapshot_sha256": canonical,
                "initial_state_sha256": state_sha,
                "initial_rgb_sha256": rgb_sha,
                "arrays_file": arrays_path.name,
                "video_file": video_rel,
                "video_sha256": video_sha,
                "prune_stats": stats,
                "trace": trace,
                **checks,
            }
            atomic_json(summary_path, summary)
            print(json.dumps({
                "task": args.task, "seed": seed, "arm": arm,
                "success": summary["success"], "steps": len(ep["infos"]),
                "runtime_seconds": round(runtime, 2),
                "prune_activation_steps": stats.get("prune_activation_steps"),
                "mean_kept_image_count_active": stats.get("mean_kept_image_count_active"),
                "total_token_flips": stats.get("total_token_flips"),
                "technical_pass": True,
            }), flush=True)
    env.close()


if __name__ == "__main__":
    main()
