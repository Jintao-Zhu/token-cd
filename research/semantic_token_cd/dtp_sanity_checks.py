"""DTP v2 pre-flight sanity checks (GPU worker + offline compare).

1. harness-vanilla: run the repo's canonical vanilla policy (AuditedVanillaInference)
   under the exact DTP rollout harness/env/loop for 1-3 (task, seed) and save
   executed actions, then compare against (a) the DTP control arm and (b) the
   released canonical vanilla episodes on the same snapshot.
2. hook audit: compare the attention_read hooks (output[1] last-row visual
   slice) against model-provided outputs.attentions per layer, elementwise,
   on a few offline states, both clean and with a blocked key set.

Run: python research/semantic_token_cd/dtp_sanity_checks.py --gpu 2 \
        --vanilla-task google_robot_move_near --vanilla-seeds 101 --hook-only
Then: python research/semantic_token_cd/dtp_sanity_checks.py --compare
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

WS = Path("/home/leju-suzhou/zjt_ws/token-cd")
ARTIFACT = WS / "artifacts/dtp_openvla_calibration_v1"
CLOSED_LOOP = ARTIFACT / "closed_loop/episodes"
SANITY = ARTIFACT / "sanity"
CANONICAL_EPISODES = (
    WS / "artifacts/vanilla_recon_shr_canonical_0_299_v2/episodes"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def make_env_and_policy(task: str, gpu: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    from research.semantic_token_cd.xswap_rollout import make_environment
    from research.semantic_token_cd.distractor_policy import AuditedVanillaInference
    from research.semantic_token_cd.distractor_rollout import PCD_SOURCE

    env, environment_id = make_environment(task, gpu)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    base = OpenVLAInference(**get_policy_config("openvla", checkpoint, task, {}, False))
    policy = copy.copy(base)
    policy.__class__ = AuditedVanillaInference
    policy._episode_trace = []
    policy._episode_logits = []
    return env, environment_id, policy


def run_vanilla_episode(env, policy, instruction, obs, seed: int):
    from parallel_inference import get_image_from_maniskill2_obs_dict
    from research.semantic_token_cd.rollout_pilot import flatten_action
    from utils import convert_numpy_or_torch_to_python, summarize

    policy.reset(instruction, seed=seed)
    image = get_image_from_maniskill2_obs_dict(env, obs)
    infos = []
    actions = []
    predicted = truncated = False
    control = 0
    while not (predicted or truncated) and control < 240:
        _raw, acts, _meta = policy.step(
            image, None, instruction, proprio=obs["agent"]["eef_pos"]
        )
        if not isinstance(acts, list):
            acts = [acts]
        for act in acts:
            executed = flatten_action(act)
            if executed.shape != (7,) or not np.isfinite(executed).all():
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
    result = summarize(infos)
    return (
        np.asarray(actions, dtype=np.float32),
        bool(result.get("success", False)),
        "environment_time_limit" if truncated else "policy_terminated_without_success",
    )


def _run_one(task: str, seed: int, gpu: int):
    from research.semantic_token_cd.distractor_rollout import (
        restore_snapshot,
        snapshot_sha,
    )

    root = SANITY / "vanilla_harness" / task
    out_arrays = root / f"episode_{seed:03d}_arrays.npz"
    out_json = root / f"episode_{seed:03d}_summary.json"
    if out_arrays.exists() and out_json.exists():
        print(json.dumps({"skip": True, "task": task, "seed": seed}), flush=True)
        return
    env, environment_id, policy = make_env_and_policy(task, gpu)
    try:
        snap_dir = WS / "artifacts/vanilla_recon_shr_canonical_0_299_v2/snapshots" / task
        with (snap_dir / f"seed_{seed:03d}.pkl").open("rb") as handle:
            snapshot = pickle.load(handle)
        canonical = snapshot_sha(snapshot)
        obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
        instruction = env.unwrapped.get_language_instruction()
        started = time.monotonic()
        actions, success, reason = run_vanilla_episode(env, policy, instruction, obs, seed)
        runtime = time.monotonic() - started
        out_arrays.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_arrays, executed_actions=actions)
        atomic_json(
            out_json,
            {
                "task": task,
                "environment_id": environment_id,
                "seed": seed,
                "arm": "vanilla_harness",
                "success": success,
                "failure_reason": reason,
                "steps": len(actions),
                "runtime_seconds": runtime,
                "canonical_snapshot_sha256": canonical,
                "initial_state_sha256": state_sha,
                "initial_rgb_sha256": rgb_sha,
                "instruction": instruction,
            },
        )
        print(json.dumps({"task": task, "seed": seed, "success": success,
                          "steps": len(actions), "runtime_seconds": round(runtime, 1)}),
              flush=True)
    finally:
        del env, policy


def cmd_vanilla(args) -> None:
    for task, seeds in args.vanilla_work:
        for seed in seeds:
            _run_one(task, seed, args.gpu)


def compare_two(a: np.ndarray, b: np.ndarray):
    n = min(len(a), len(b))
    if n == 0:
        return {"match": False, "reason": "empty"}
    diffs = []
    for i in range(n):
        if not np.array_equal(a[i], b[i]):
            diffs.append(i)
            if len(diffs) >= 12:
                break
    return {
        "length_a": int(len(a)),
        "length_b": int(len(b)),
        "exact_row_match": len(diffs) == 0,
        "first_diff_step": int(diffs[0]) if diffs else None,
        "first_diffs": diffs,
    }


def cmd_compare(_args) -> None:
    pairs = [
        ("google_robot_move_near", [101, 112]),
        ("google_robot_pick_coke_can", [100]),
        ("google_robot_open_drawer", [118]),
    ]
    results = {}
    for task, seeds in pairs:
        for seed in seeds:
            key = f"{task}__{seed:03d}"
            ctrl = np.load(CLOSED_LOOP / task / "control" / f"episode_{seed:03d}_arrays.npz")[
                "executed_actions"
            ]
            hv = np.load(SANITY / "vanilla_harness" / task / f"episode_{seed:03d}_arrays.npz")[
                "executed_actions"
            ]
            van = np.load(CANONICAL_EPISODES / task / "vanilla" / f"episode_{seed:03d}_arrays.npz")[
                "executed_actions"
            ]
            results[key] = {
                "control_vs_harness_vanilla": compare_two(ctrl, hv),
                "harness_vanilla_vs_canonical": compare_two(hv, van),
                "control_vs_canonical": compare_two(ctrl, van),
            }
    atomic_json(SANITY / "COMPARE_RESULTS.json", results)
    print(json.dumps(results, indent=2))


def hook_audit_states(args):
    samples = json.loads((ARTIFACT / "SAMPLES.json").read_text())
    seen = set()
    rows = []
    for row in samples:
        if row["task"] not in seen:
            seen.add(row["task"])
            rows.append(row)
        if len(rows) == 3:
            break
    return rows


def cmd_hook(args) -> None:
    import torch
    from PIL import Image
    from contrast_policies.openvla_contrast import OpenVLAContrastInference
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    from research.semantic_token_cd.distractor_rollout import PCD_SOURCE
    from research.semantic_token_cd.dtp_openvla_policy import attention_access
    from research.ar_token_counterfactual.intervention import ensure_empty_action_token

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["HF_HUB_OFFLINE"] = "1"
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    policy = OpenVLAContrastInference(
        **get_policy_config("openvla", checkpoint, "google_robot_open_drawer", {}, False)
    )
    model = policy.vla
    report = []
    for row in hook_audit_states(args):
        with np.load(row["source"]) as a:
            image = a["image"]
        assert image.dtype == np.uint8 and image.ndim == 3
        instruction = row["instruction"]
        inputs = policy.process_inputs(image, task_description=instruction)
        ids, mask = ensure_empty_action_token(
            inputs["input_ids"], inputs["attention_mask"]
        )
        blocked = sorted(set(np.random.RandomState(int(row["id"].split("__")[1])).choice(
            256, size=12, replace=False).tolist()))
        for label, blk in (("clean", []), ("blocked", blocked)):
            with attention_access(model, blk) as rows:
                out = model(
                    input_ids=ids,
                    attention_mask=mask,
                    pixel_values=inputs["pixel_values"],
                    use_cache=False,
                    output_attentions=True,
                    return_dict=True,
                )
            per_layer = []
            for layer_index, attn in enumerate(out.attentions):
                authoritative = attn[0, :, -1, 1:257].detach().float().mean(0).cpu().numpy()
                hooked = rows[layer_index]
                diff = float(np.abs(authoritative - hooked).max())
                per_layer.append(
                    {"layer": layer_index, "max_abs_diff": diff,
                     "hook_finite": bool(np.isfinite(hooked).all())}
                )
            report.append(
                {"state": row["id"], "case": label, "blocked": blk,
                 "max_over_layers": max(x["max_abs_diff"] for x in per_layer),
                 "per_layer": per_layer}
            )
            del out
    atomic_json(SANITY / "HOOK_AUDIT.json", report)
    bad = [x for x in report if x["max_over_layers"] > 1e-5]
    print(json.dumps({"states": len(hook_audit_states(args)) * 2, "failed_layers": len(bad)},
                     indent=2))
    if bad:
        raise SystemExit("hook audit mismatch vs outputs.attentions")
    print("hook audit OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--vanilla-task", default="google_robot_move_near")
    ap.add_argument("--vanilla-seeds", default="101,112")
    ap.add_argument("--sanity-all", action="store_true")
    ap.add_argument("--hook-only", action="store_true")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    if args.compare:
        cmd_compare(args)
        return
    if args.sanity_all:
        args.vanilla_work = [
            ("google_robot_move_near", [101, 112]),
            ("google_robot_pick_coke_can", [100]),
            ("google_robot_open_drawer", [118]),
        ]
    else:
        args.vanilla_work = [(args.vanilla_task, [int(x) for x in args.vanilla_seeds.split(",")])]
    if not args.hook_only:
        cmd_vanilla(args)
    cmd_hook(args)
    if args.sanity_all:
        cmd_compare(args)
    # gate: control must equal current-harness vanilla on every checked scene
    report = json.loads((SANITY / "COMPARE_RESULTS.json").read_text())
    failed = {k: v["control_vs_harness_vanilla"] for k, v in report.items()
              if not v["control_vs_harness_vanilla"]["exact_row_match"]}
    if failed:
        raise SystemExit(f"sanity gate failed control!=harness-vanilla: {failed}")


if __name__ == "__main__":
    main()
