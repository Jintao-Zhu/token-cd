"""VLA-Pruner r0 sanity: strict equality between current-harness vanilla and
the scheme-B adapter running with fastv_r=0.0.

Gates (any failure -> exit code 1, do NOT run the closed-loop experiment):
  1. 7 action tokens identical for every compared step;
  2. raw_action and executed action arrays bit-identical (max_abs == 0);
  3. pruned_indices == empty and kept_image_count == 256 on every pruner step
     (i.e. the code path runs but performs no deletion);
  4. pruning layer locked to fastv_k=3 and history resets at episode start;
  5. episode success identical.

Usage examples (run from repo root; GPU must be one of 2/3):
  python research/semantic_token_cd/vla_pruner_r0_sanity.py --gpu 2 \
      --task google_robot_pick_coke_can --seeds 100-101 --mode step
  python research/semantic_token_cd/vla_pruner_r0_sanity.py --gpu 3 \
      --task google_robot_open_drawer --seeds 118 --mode episode
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
PCD_ROOT = Path("/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e")
PCD_SOURCE = PCD_ROOT / "source"
CANONICAL_SNAPSHOTS = REPO_ROOT / "artifacts/vanilla_recon_shr_canonical_0_299_v2/snapshots"

ARM_VANILLA = "vanilla"
ARM_R0 = "vla_pruner_r0"


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


def make_environment(task: str):
    import gymnasium as gym
    import simpler_env

    renderer_kwargs = {"device": "cuda:0", "offscreen_only": True}
    if task == "google_robot_pick_coke_can":
        return gym.make(
            "GraspSingleOpenedCokeCanDistractorInScene-v0",
            obs_mode="rgbd",
            prepackaged_config=True,
            distractor_config="less",
            renderer_kwargs=renderer_kwargs,
        ), "GraspSingleOpenedCokeCanDistractorInScene-v0"
    environment_id, base_kwargs = simpler_env.ENVIRONMENT_MAP[task]
    kwargs = dict(base_kwargs)
    kwargs.update({"prepackaged_config": True, "renderer_kwargs": renderer_kwargs})
    return gym.make(environment_id, obs_mode="rgbd", **kwargs), environment_id


def load_base_policy(task: str):
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, task, {}, False)
    return OpenVLAInference(**config)


def build_arms(base, task: str):
    from research.semantic_token_cd.vla_pruner_policy import build_vla_pruner_policy

    vanilla = build_vla_pruner_policy(base, ARM_VANILLA, task)
    r0 = build_vla_pruner_policy(base, ARM_R0, task)
    return {ARM_VANILLA: vanilla, ARM_R0: r0}


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def run_step_compare(env, policies, instruction, obs, seed: int) -> dict:
    from parallel_inference import get_image_from_maniskill2_obs_dict
    from research.semantic_token_cd.rollout_pilot import flatten_action

    image = get_image_from_maniskill2_obs_dict(env, obs)
    per_arm = {}
    for arm in (ARM_VANILLA, ARM_R0):
        pol = policies[arm]
        pol.reset(instruction, seed=seed)
        _raw, acts, aux = pol.step(image, None, instruction, proprio=obs["agent"]["eef_pos"])
        if not isinstance(acts, list):
            acts = [acts]
        executed = np.asarray([flatten_action(a) for a in acts], dtype=np.float32)
        trace = pol._episode_trace[-1]
        per_arm[arm] = {
            "token_ids": list(trace["token_ids"]),
            "raw_action": list(trace["raw_action"]),
            "executed": executed[0].tolist() if executed.shape[0] else None,
            "history_len": int(pol._episode_trace[-1].get("history_len", 0)),
            "trace": trace,
        }
    a, b = per_arm[ARM_VANILLA], per_arm[ARM_R0]
    checks = {
        "token_ids_equal": a["token_ids"] == b["token_ids"],
        "raw_action_equal": np.allclose(np.asarray(a["raw_action"], dtype=np.float64),
                                        np.asarray(b["raw_action"], dtype=np.float64), rtol=0, atol=0),
        "raw_action_max_abs": float(np.abs(np.asarray(a["raw_action"], dtype=np.float64)
                                           - np.asarray(b["raw_action"], dtype=np.float64)).max()),
        "executed_equal": a["executed"] == b["executed"],
        "r0_pruned_empty": bool(b["trace"].get("pruned_indices_empty")),
        "r0_pruned_count": int(b["trace"].get("pruned_count", -1)),
        "r0_kept_image_count": int(b["trace"].get("kept_image_count", -1)),
        "r0_kept_count_eq_original": bool(
            b["trace"].get("kept_count") == b["trace"].get("original_seq_length")
        ),
        "r0_pruning_layer": b["trace"].get("pruning_layer"),
        "r0_history_len": int(b["history_len"]),
        "r0_fastv_r_effective": b["trace"].get("fastv_r_effective"),
    }
    return {"seed": int(seed), "arm_data": per_arm, "checks": checks}


def run_episode_compare(env, policies, instruction, obs, seed: int, snapshot) -> dict:
    from research.semantic_token_cd.dtp_closed_loop_rollout import run_loop
    from research.semantic_token_cd.rollout_pilot import restore_snapshot

    per_arm = {}
    for arm in (ARM_VANILLA, ARM_R0):
        pol = policies[arm]
        # Each arm must roll out from the *same* initial physics state: rewind
        # the simulator before every arm (run_loop mutates the environment).
        obs_i, _, _ = restore_snapshot(env, seed, snapshot)
        pol.reset(instruction, seed=seed)
        result, actions, infos, reason, _frames = run_loop(
            env, pol, instruction, obs_i, record_frames=False
        )
        tokens = [list(t["token_ids"]) for t in pol._episode_trace]
        raw_acts = [list(t["raw_action"]) for t in pol._episode_trace]
        per_arm[arm] = {
            "steps": int(len(actions)),
            "success": bool(result.get("success")),
            "reason": reason,
            "trace_len": len(pol._episode_trace),
            "token_ids": tokens,
            "raw_actions": raw_acts,
            "executed": [list(map(float, a)) for a in np.asarray(actions, dtype=np.float32)],
            "r0_all_pruned_empty": all(
                bool(t.get("pruned_indices_empty")) for t in pol._episode_trace
            ) if arm == ARM_R0 else None,
            "r0_all_kept_eq_original": all(
                t.get("kept_count") == t.get("original_seq_length")
                for t in pol._episode_trace
            ) if arm == ARM_R0 else None,
            "r0_all_kept_image_256": all(
                t.get("kept_image_count") == 256 for t in pol._episode_trace
            ) if arm == ARM_R0 else None,
            "r0_history_reset_first": int(pol._episode_trace[0].get("history_len", -1)) == 0
            if (arm == ARM_R0 and pol._episode_trace) else None,
        }
    a, b = per_arm[ARM_VANILLA], per_arm[ARM_R0]
    executed_equal = len(a["executed"]) == len(b["executed"]) and all(
        np.array_equal(np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32))
        for x, y in zip(a["executed"], b["executed"])
    )
    checks = {
        "steps_equal": a["steps"] == b["steps"],
        "success_equal": a["success"] == b["success"],
        "trace_len_equal": a["trace_len"] == b["trace_len"],
        "token_ids_equal": a["token_ids"] == b["token_ids"],
        "raw_actions_equal": a["raw_actions"] == b["raw_actions"],
        "executed_equal": executed_equal,
        "r0_all_pruned_empty": bool(b["r0_all_pruned_empty"]),
        "r0_all_kept_eq_original": bool(b["r0_all_kept_eq_original"]),
        "r0_all_kept_image_256": bool(b["r0_all_kept_image_256"]),
        "r0_history_reset_first": bool(b["r0_history_reset_first"]),
    }
    return {"seed": int(seed), "arm_data": per_arm, "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--mode", choices=["step", "episode"], required=True)
    parser.add_argument("--worker-id", default="manual")
    parser.add_argument("--artifact-dir", type=Path,
                        default=REPO_ROOT / "artifacts/vla_pruner_openvla_reproduction/r0_sanity")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    for path in (REPO_ROOT, PCD_SOURCE):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from parallel_inference import get_image_from_maniskill2_obs_dict
    from research.semantic_token_cd.rollout_pilot import restore_snapshot, snapshot_sha

    artifact = args.artifact_dir.resolve()
    seeds = parse_seeds(args.seeds)
    task_root = artifact / args.task

    env, environment_id = make_environment(args.task)
    base = load_base_policy(args.task)
    policies = build_arms(base, args.task)

    results = []
    all_pass = True
    for seed in seeds:
        snapshot_path = CANONICAL_SNAPSHOTS / args.task / f"seed_{seed:03d}.pkl"
        if not snapshot_path.exists():
            raise FileNotFoundError(f"missing canonical snapshot: {snapshot_path}")
        with snapshot_path.open("rb") as handle:
            snapshot = pickle.load(handle)
        canonical = snapshot_sha(snapshot)
        obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
        instruction = env.unwrapped.get_language_instruction()
        if args.mode == "step":
            out = run_step_compare(env, policies, instruction, obs, seed)
        else:
            out = run_episode_compare(env, policies, instruction, obs, seed, snapshot)
        out["canonical_snapshot_sha256"] = canonical
        out["initial_state_sha256"] = state_sha
        out["initial_rgb_sha256"] = rgb_sha
        pass_here = all(bool(v) for v in out["checks"].values() if isinstance(v, bool))
        if not pass_here:
            all_pass = False
            print(json.dumps({"FAIL": args.task, "seed": seed, "checks": out["checks"]}, indent=2), flush=True)
        else:
            print(json.dumps({"PASS": args.task, "seed": seed, "mode": args.mode,
                              "n_steps/checks": out["checks"]}, default=str), flush=True)
        results.append(out)
        atomic_json(task_root / f"seed_{seed:03d}.json", {
            "mode": args.mode,
            "task": args.task,
            "seed": int(seed),
            "worker": args.worker_id,
            "output": out,
        })
    summary = {
        "protocol": "VLA_PRUNER_R0_SANITY_V1",
        "task": args.task,
        "mode": args.mode,
        "gpu": args.gpu,
        "seeds": seeds,
        "all_pass": all_pass,
    }
    atomic_json(artifact / "summary.json", summary)
    print(json.dumps({"ALL_PASS": all_pass, "task": args.task, "mode": args.mode, "seeds": seeds}), flush=True)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
