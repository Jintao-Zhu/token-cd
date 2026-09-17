"""Run phase-matched drawer strength windows and one-decision token ablations."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np

from research.semantic_token_cd.distractor_rollout import flatten_action, get_image_from_maniskill2_obs_dict
from research.semantic_token_cd.prompt_attn_l11_count_rollout import build_policies, make_environment
from research.semantic_token_cd.prompt_attn_shr_rollout import atomic_json
from research.semantic_token_cd.run_l11_token_causal_audit import restore_mid, spatial_groups
from research.semantic_token_cd.xdrawer_protocol import drawer_geometry


def make_l11(base, task: str, lambd: float):
    policy = build_policies(base, task, ("l11_matched",))["l11_matched"]
    policy.lambd = float(lambd); policy.save_prompt_attention = False
    return policy


def make_region(base, task: str, lambd: float):
    policy = make_l11(base, task, lambd); policy.selector_mode = "sim_region"; policy.selection_count = None
    return policy


def physical(inner, task: str) -> dict:
    geometry = drawer_geometry(inner, "top"); tcp = np.asarray(inner.tcp.pose.p, dtype=np.float64)
    points = geometry.get("handle_points") or [geometry["handle_p"]]
    return {"qpos": float(np.asarray(inner.art_obj.get_qpos())[inner.joint_idx]),
            "tcp_handle_distance": min(float(np.linalg.norm(tcp - np.asarray(point))) for point in points),
            "success": bool(inner.evaluate().get("success", False))}


def gain(task: str, phase: str, before: dict, after: dict) -> float:
    if phase == "approach": return float(before["tcp_handle_distance"] - after["tcp_handle_distance"])
    sign = 1.0 if "open" in task else -1.0
    return float(sign * (after["qpos"] - before["qpos"]))


def reset_policy(policy, instruction: str, seed: int, step: int) -> None:
    policy.reset(instruction, seed=seed); policy._selector_step = step
    policy._episode_trace = []; policy._episode_logits = []


def execute_window(env, snapshot: dict, task: str, seed: int, step: int, phase: str,
                   first_policy, continuation_policy, horizon: int) -> dict:
    obs = restore_mid(env, seed, snapshot); before = physical(env.unwrapped, task)
    reset_policy(first_policy, snapshot["instruction"], seed, step)
    if continuation_policy is not first_policy:
        reset_policy(continuation_policy, snapshot["instruction"], seed, step + 1)
    actions, metas, truncated = [], [], False
    while len(actions) < horizon and not truncated:
        policy = first_policy if not actions else continuation_policy
        image = np.asarray(get_image_from_maniskill2_obs_dict(env, obs), dtype=np.uint8)
        _raw, predicted, meta = policy.step(image, None, snapshot["instruction"], proprio=obs["agent"]["eef_pos"])
        metas.append(meta)
        if not isinstance(predicted, list): predicted = [predicted]
        for action in predicted:
            executed = flatten_action(action); obs, _reward, _success, truncated, _info = env.step(executed)
            actions.append(executed.tolist())
            if len(actions) >= horizon or truncated: break
    after = physical(env.unwrapped, task)
    return {"before": before, "after": after, "primary_progress_gain": gain(task, phase, before, after),
            "actions": actions, "truncated": bool(truncated), "metas": metas,
            "mean_guided_change_ratio": float(np.mean([x.get("guided_change_ratio", 0.0) for x in metas])),
            "mean_guided_clean_action_l2": float(np.mean([x.get("guided_clean_action_l2", 0.0) for x in metas])),
            "mean_residual_norm": float(np.mean([x.get("centered_logit_residual_norm", 0.0) for x in metas]))}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--task", required=True); parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard", type=int, default=0); parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=10); args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu); os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    from research.semantic_token_cd.distractor_rollout import PCD_SOURCE

    root = args.artifact.resolve(); states = json.loads((root / "MANIFEST.json").read_text())["states"]
    states = [x for x in states if x["task"] == args.task][args.shard::args.num_shards]
    env, _ = make_environment(args.task); checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, args.task, {}, False); base = OpenVLAInference(**config)
    dynamic = {value: make_l11(base, args.task, value) for value in (.0, .25, .5)}
    region = make_region(base, args.task, .5)
    try:
        for state in states:
            seed, step, phase = int(state["seed"]), int(state["control_step"]), state["phase"]
            destination = root / "results" / args.task / f"seed_{seed:03d}_{state['category']}_{phase}_step_{step:03d}.json"
            if destination.exists(): print(json.dumps({"skip": str(destination)}), flush=True); continue
            with Path(state["snapshot_file"]).open("rb") as handle: snapshot = pickle.load(handle)
            obs = restore_mid(env, seed, snapshot); image = np.asarray(get_image_from_maniskill2_obs_dict(env, obs), dtype=np.uint8)
            source = np.load(state["arrays_file"], allow_pickle=False); difference = np.abs(image.astype(np.int16)-source["image"].astype(np.int16))
            state_error = float(np.max(np.abs(np.asarray(env.unwrapped.get_state())-np.asarray(snapshot["sim_state"]))))
            if state_error > 1e-6 or float(difference.mean()) > .1 or float(np.mean(difference > 16)) > 1e-3:
                raise RuntimeError(f"restore gate failed {args.task}/{seed}/{phase}")
            selected = [int(x) for x in state["selected_token_ids"]]; groups = spatial_groups(selected)
            results = {}
            for name, value in (("dynamic_l11_lambda_0p5", .5), ("dynamic_l11_lambda_0p25", .25), ("clean_lambda_0", .0)):
                results[name] = execute_window(env, snapshot, args.task, seed, step, phase,
                                               dynamic[value], dynamic[value], args.horizon)
            region.region_token_ids = selected; region.region_label = "locked_full"
            results["locked_full_then_l11_0p5"] = execute_window(
                env, snapshot, args.task, seed, step, phase, region, dynamic[.5], args.horizon)
            for group_name, removed in groups.items():
                kept = sorted(set(selected)-set(removed)); region.region_token_ids = kept; region.region_label = f"without_{group_name}"
                result = execute_window(env, snapshot, args.task, seed, step, phase, region, dynamic[.5], args.horizon)
                result["removed_token_ids"] = removed; result["kept_token_ids"] = kept
                results[f"without_{group_name}"] = result
            strength_baseline = results["dynamic_l11_lambda_0p5"]["primary_progress_gain"]
            token_baseline = results["locked_full_then_l11_0p5"]["primary_progress_gain"]
            for name, result in results.items():
                result["delta_vs_dynamic_l11_lambda_0p5"] = result["primary_progress_gain"]-strength_baseline
                result["delta_vs_locked_full"] = result["primary_progress_gain"]-token_baseline
            atomic_json(destination, {"protocol_id": "DRAWER_PHASE_STRENGTH_TOKEN_CAUSAL_V1",
                "task": args.task, "seed": seed, "category": state["category"], "phase": phase,
                "phase_rule": state["phase_rule"], "control_step": step, "horizon": args.horizon,
                "groups": groups, "source_selected_token_ids": selected,
                "restore_audit": {"state_max_abs_difference": state_error, "rgb_mean_abs_difference": float(difference.mean()),
                                  "rgb_gt16_fraction": float(np.mean(difference > 16))}, "results": results})
            print(json.dumps({"complete": f"{args.task}/{seed}/{state['category']}/{phase}", "branches": len(results)}), flush=True)
    finally: env.close()


if __name__ == "__main__": main()
