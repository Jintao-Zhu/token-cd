#!/usr/bin/env python3
"""L11 Ranking Value Study: paired fixed-K selector interventions on SIMPLER."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
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
RT0 = ROOT / "artifacts/simpler_rt_depth0_7task_4200_v1"
PROTOCOL = "L11_RANKING_VALUE_SIMPLER_6TASK_5ARM_SEEDS0_99_V1"
TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
    "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel",
)
ARMS = ("top32", "mid32", "bottom32", "random32", "top96_random32")
DEPTH = {
    "google_robot_open_drawer": 8,
    "google_robot_close_drawer": 8,
    "google_robot_pick_coke_can": 0,
    "google_robot_move_near": 0,
    "widowx_put_eggplant_in_basket": 0,
    "widowx_spoon_on_towel": 0,
}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def make_environment(task: str):
    import gymnasium as gym
    import simpler_env

    render = {
        "rt_samples_per_pixel": 32,
        "rt_max_path_depth": DEPTH[task],
        "rt_use_denoiser": True,
    }
    renderer = {"device": "cuda:0", "offscreen_only": True}
    if task == "google_robot_pick_coke_can":
        return gym.make(
            "GraspSingleOpenedCokeCanDistractorInScene-v0", obs_mode="rgbd",
            prepackaged_config=True, distractor_config="less", shader_dir="rt",
            render_config=render, renderer_kwargs=renderer,
        )
    env_id, base = simpler_env.ENVIRONMENT_MAP[task]
    kwargs = dict(base)
    kwargs.update({
        "prepackaged_config": True,
        "shader_dir": "rt",
        "render_config": render,
        "renderer_kwargs": renderer,
    })
    return gym.make(env_id, obs_mode="rgbd", **kwargs)


def parse_seeds(spec: str) -> list[int]:
    result: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(v) for v in part.split("-", 1))
            result.extend(range(lo, hi + 1))
        else:
            result.append(int(part))
    values = sorted(set(result))
    if not values or any(seed < 0 or seed > 99 for seed in values):
        raise ValueError("L11 Ranking Value Study is locked to seeds 0-99")
    return values


def save_rank_arrays(path: Path, records: list[dict], actions: np.ndarray) -> None:
    if not records:
        raise RuntimeError("ranking-value episode has no policy steps")
    keys = (
        "positive", "negative", "guided", "selected_mask", "ranking_attention",
        "full_ranking_token_ids", "rank_by_token", "selected_token_ids",
        "selected_mask_grid", "selected_source_features", "selected_harmonic_features",
        "per_token_reconstruction_dose", "step_index",
    )
    payload = {"executed_actions": np.asarray(actions, dtype=np.float32)}
    for key in keys:
        if all(key in record for record in records):
            payload[key] = np.stack([record[key] for record in records])
    if "ranking_attention" not in payload or "guided" not in payload:
        raise RuntimeError("required per-step ranking/logit audit arrays are missing")
    payload["total_reconstruction_dose"] = payload["per_token_reconstruction_dose"].sum(axis=1)
    np.savez_compressed(path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--seeds", default="0-99")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--preflight-one-step", action="store_true")
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)
    arms = tuple(value.strip() for value in args.arms.split(",") if value.strip())
    if not arms or any(arm not in ARMS for arm in arms):
        raise ValueError(f"unsupported arms: {arms}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
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
    from research.semantic_token_cd.prompt_attn_shr_policy import PromptAttentionSHRInference
    from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common
    from research.semantic_token_cd.spatial_grid_rollout import run_episode, get_image_from_maniskill2_obs_dict

    output = args.output.resolve()
    env = base = policy = None
    try:
        env = make_environment(args.task)
        checkpoint = str(PCD / "pretrained/openvla-7b")
        config = get_policy_config("openvla", checkpoint, args.task, {}, False)
        base = OpenVLAInference(**config)
        policy = copy.copy(base)
        policy.__class__ = PromptAttentionSHRInference
        _init_common(policy, 0.5)
        policy.task_index = TASK_INDEX[args.task]
        policy.selector_mode = "prompt_attention"
        policy.attention_layers = (11,)
        policy.selection_count = 32
        policy.selection_top_p = None
        policy.selection_budget_schedule = None
        policy.selection_budget_scale = 1.0
        policy.selection_rank_bin = None
        policy.save_prompt_attention = True

        for seed in seeds:
            snapshot_path = CANON / "snapshots" / args.task / f"seed_{seed:03d}.pkl"
            baseline_root = CANON if DEPTH[args.task] == 8 else RT0
            baseline_path = baseline_root / "episodes" / args.task / "vanilla" / f"episode_{seed:03d}_summary.json"
            baseline_arrays_path = baseline_path.with_name(f"episode_{seed:03d}_arrays.npz")
            if not snapshot_path.is_file() or not baseline_path.is_file() or not baseline_arrays_path.is_file():
                raise FileNotFoundError(f"missing snapshot or paired Vanilla: {args.task}:{seed}")
            snapshot = pickle.loads(snapshot_path.read_bytes())
            baseline = json.loads(baseline_path.read_text())
            canonical_hash = snapshot_sha(snapshot)
            if canonical_hash != baseline.get("canonical_snapshot_sha256"):
                raise RuntimeError(f"canonical snapshot mismatch: {args.task}:{seed}")
            expected_state = baseline.get("initial_state_sha256")
            expected_rgb = baseline.get("initial_rgb_sha256")
            instruction_expected = baseline.get("instruction")

            for arm in arms:
                arm_dir = output / "episodes" / args.task / arm
                summary_path = arm_dir / f"episode_{seed:03d}_summary.json"
                arrays_path = arm_dir / f"episode_{seed:03d}_arrays.npz"
                if not args.preflight_one_step and summary_path.exists() and arrays_path.exists():
                    old = json.loads(summary_path.read_text())
                    if old.get("protocol_id") != PROTOCOL or old.get("arm") != arm:
                        raise RuntimeError(f"incompatible existing result: {summary_path}")
                    continue

                obs, state_hash, rgb_hash = restore_snapshot(env, seed, snapshot)
                instruction = env.unwrapped.get_language_instruction()
                checks = {
                    "snapshot": canonical_hash == baseline.get("canonical_snapshot_sha256"),
                    "state": state_hash == expected_state,
                    "rgb": rgb_hash == expected_rgb,
                    # The historical RT-depth0 worker omitted the instruction
                    # field from its Vanilla summaries.  In that case the
                    # canonical snapshot + state + RGB hashes remain the
                    # paired identity checks; the live instruction is recorded
                    # in this study's own output and is shared by all arms.
                    "instruction": instruction_expected is None or instruction == instruction_expected,
                }
                if not all(checks.values()):
                    raise RuntimeError(
                        f"canonical identity mismatch {args.task}:{seed}: arm={arm}, "
                        f"checks={checks}, live_rgb={rgb_hash}, expected_rgb={expected_rgb}"
                    )

                policy.ranking_value_arm = arm
                policy.ranking_value_task_key = args.task
                policy.reset(instruction, seed=seed)
                policy._episode_trace, policy._episode_logits = [], []
                started = time.monotonic()
                if args.preflight_one_step:
                    image = get_image_from_maniskill2_obs_dict(env, obs)
                    _raw, _actions, _meta = policy.step(
                        image, None, instruction, proprio=obs["agent"]["eef_pos"]
                    )
                    elapsed = time.monotonic() - started
                    row = policy._episode_trace[-1]
                    rec = policy._episode_logits[-1]
                    picked = [int(x) for x in row["selected_token_ids"]]
                    selected_ranks = [int(x) for x in row["selected_attention_ranks"]]
                    expected_n = 32
                    check = {
                        "selected_exactly_32": len(picked) == expected_n and len(set(picked)) == expected_n,
                        "ranking_is_256": len(rec.get("full_ranking_token_ids", [])) == 256,
                        "attention_is_256": len(rec.get("ranking_attention", [])) == 256,
                        "fixed_budget": int(row.get("m_t", -1)) == 32,
                        "reconstruction_dose_finite": bool(np.isfinite(rec.get("per_token_reconstruction_dose", [])).all()),
                        "guided_logits_shape": np.asarray(rec.get("guided", [])).shape == (7, 256),
                        "selector_rank_definition": (
                            min(selected_ranks) == 1 and max(selected_ranks) == 32 if arm == "top32" else
                            min(selected_ranks) == 97 and max(selected_ranks) == 128 if arm == "mid32" else
                            min(selected_ranks) == 225 and max(selected_ranks) == 256 if arm == "bottom32" else
                            max(selected_ranks) <= 96 if arm == "top96_random32" else
                            len(selected_ranks) == 32
                        ),
                    }
                    if not all(check.values()):
                        raise RuntimeError(f"selector preflight failed {args.task}:{seed}:{arm}: {check}")
                    out = output / "analysis" / f"preflight_{args.task}_seed{seed:03d}_{arm}.json"
                    atomic_json(out, {
                        "protocol_id": PROTOCOL, "preflight": True, "task": args.task,
                        "seed": seed, "arm": arm, "gpu": args.gpu, "renderer_depth": DEPTH[args.task],
                        "identity": {"snapshot": canonical_hash, "state": state_hash, "rgb": rgb_hash},
                        "instruction": instruction,
                        "baseline_instruction_available": instruction_expected is not None,
                        "checks": check, "selected_token_ids": picked,
                        "selected_attention_ranks": selected_ranks,
                        "selected_attention_mean": float(np.asarray(rec["ranking_attention"])[picked].mean()),
                        "dose": float(np.asarray(rec["per_token_reconstruction_dose"]).sum()),
                        "seconds": elapsed,
                    })
                    print(json.dumps({"preflight": True, "task": args.task, "seed": seed,
                                      "arm": arm, "gpu": args.gpu, "pass": all(check.values()),
                                      "seconds": round(elapsed, 1)}, sort_keys=True), flush=True)
                    continue

                result, steps, reason, actions, jitter = run_episode(env, policy, instruction, obs)
                elapsed = time.monotonic() - started
                trace = jsonable(policy._episode_trace)
                records = policy._episode_logits
                if not trace or len(records) != len(trace):
                    raise RuntimeError(f"step audit count mismatch {args.task}:{seed}:{arm}")
                with np.load(baseline_arrays_path) as baseline_arrays:
                    baseline_logit_key = (
                        "positive_logits" if "positive_logits" in baseline_arrays.files else "positive"
                    )
                    if baseline_logit_key not in baseline_arrays.files or len(baseline_arrays[baseline_logit_key]) < 1:
                        raise RuntimeError(f"historical Vanilla clean logits missing: {baseline_arrays_path}")
                    initial_clean_logit_max_abs_diff = float(np.max(np.abs(
                        np.asarray(records[0]["positive"], dtype=np.float32)
                        - np.asarray(baseline_arrays[baseline_logit_key][0], dtype=np.float32)
                    )))
                if initial_clean_logit_max_abs_diff > 1e-5:
                    raise RuntimeError(
                        f"initial clean logits differ from paired Vanilla: {args.task}:{seed}:{arm} "
                        f"max_abs={initial_clean_logit_max_abs_diff}"
                    )
                if not all(row.get("coverage_exact") is True and row.get("m_t") == 32 for row in trace):
                    raise RuntimeError(f"fixed K=32 audit failed {args.task}:{seed}:{arm}")
                if not all(abs(float(row.get("lambda", -1)) - 0.5) < 1e-12 for row in trace):
                    raise RuntimeError(f"lambda audit failed {args.task}:{seed}:{arm}")
                arm_dir.mkdir(parents=True, exist_ok=True)
                save_rank_arrays(arrays_path, records, actions)
                doses = [float(np.asarray(record["per_token_reconstruction_dose"]).sum()) for record in records]
                payload = {
                    "protocol_id": PROTOCOL,
                    "benchmark": "SIMPLER",
                    "task": args.task,
                    "seed": seed,
                    "arm": arm,
                    "selector": "historical PromptAttentionSHRInference; L11; fixed K=32",
                    "success": bool(result["success"]),
                    "steps": int(steps),
                    "failure_reason": reason,
                    "runtime_seconds": elapsed,
                    "action_jitter_index": jitter,
                    "canonical_snapshot_sha256": canonical_hash,
                    "initial_state_sha256": state_hash,
                    "initial_rgb_sha256": rgb_hash,
                    "initial_clean_logits_max_abs_diff_vs_vanilla": initial_clean_logit_max_abs_diff,
                    "instruction": instruction,
                    "renderer": f"SAPIEN RT depth{DEPTH[args.task]}, spp=32, denoiser=true",
                    "renderer_gpu": args.gpu,
                    "inference_gpu": args.gpu,
                    "checkpoint": checkpoint,
                    "lambda": 0.5,
                    "K": 32,
                    "beta": 0.0,
                    "mean_reconstruction_dose": float(np.mean(doses)),
                    "mean_per_token_dose": float(np.mean(doses) / 32),
                    "sampled_states": len(trace),
                    "selector_trace": trace,
                    "arrays_file": arrays_path.name,
                    "worker_id": args.worker_id,
                }
                atomic_json(summary_path, payload)
                print(json.dumps({"worker": args.worker_id, "gpu": args.gpu,
                                  "task": args.task, "seed": seed, "arm": arm,
                                  "success": payload["success"], "steps": steps,
                                  "seconds": round(elapsed, 1),
                                  "dose": round(payload["mean_reconstruction_dose"], 4),
                                  "rgb": rgb_hash}, sort_keys=True), flush=True)
    except Exception as exc:
        fail_path = output / "logs" / f"failure_{args.worker_id}_{int(time.time())}.json"
        atomic_json(fail_path, {
            "task": args.task, "seeds": args.seeds, "arms": arms, "gpu": args.gpu,
            "error": repr(exc), "traceback": traceback.format_exc(),
        })
        raise
    finally:
        if env is not None:
            env.close()
        del policy, base, env
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
