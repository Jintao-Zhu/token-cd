"""Phase B: does m-dagger have closed-loop meaning?

For each selected frozen state: restore it exactly, run ONE control step with a
tested budget m, then hand control back to the standard L11-Matched policy for
the remainder of the episode.  Only the tested step differs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from research.semantic_token_cd.distractor_rollout import (
    clone, flatten_action, get_image_from_maniskill2_obs_dict, jsonable, restore_snapshot, snapshot_sha,
)
from research.semantic_token_cd.prompt_attn_layer_rollout import make_environment
from research.semantic_token_cd.prompt_attn_shr_policy_floor import PromptAttentionSHRInference
from research.semantic_token_cd.prompt_attn_shr_rollout import LAMBDA
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common
from research.semantic_token_cd.prompt_attn_shr_rollout import atomic_json, load_reference
from research.semantic_token_cd.rollout_pilot import wrapped_observation

TASKS = ("google_robot_open_drawer", "google_robot_close_drawer",
         "google_robot_pick_coke_can", "google_robot_move_near")
TRAJ = {
    "google_robot_open_drawer": "artifacts/prompt_attn_layer_selection_v1/closed_loop/episodes/google_robot_open_drawer/prompt_single",
    "google_robot_pick_coke_can": "artifacts/prompt_attn_layer_selection_v1/closed_loop/episodes/google_robot_pick_coke_can/prompt_single",
    "google_robot_move_near": "artifacts/prompt_attn_layer_selection_v1/closed_loop/episodes/google_robot_move_near/prompt_single",
    "google_robot_close_drawer": "artifacts/prompt_attn_layer_selection_v1/closed_loop_remaining6/episodes/google_robot_close_drawer/prompt_single",
}
PER_STAGE = 10          # per (task, progress stage)
CLIP_LO, CLIP_HI = 4, 96


def build_policy(base, task: str, selection_count):
    """L11 prompt-attention policy on the PATCHED class (any m in [1,256] allowed)."""
    import copy
    p = copy.copy(base)
    p.__class__ = PromptAttentionSHRInference
    _init_common(p, LAMBDA)
    p.beta = 0.0
    p.selector_mode = "prompt_attention"
    p.task_index = TASK_INDEX[task]
    p.attention_layers = (11,)
    p.selection_count = selection_count
    p.selection_top_p = None
    p.selection_budget_schedule = None
    p.selection_budget_scale = 1.0
    p.selection_budget_source = "matched"
    p.selection_budget_entities = None
    p.selection_budget_min_count = None
    p.selection_budget_max_count = None
    p.selection_budget_small_threshold = None
    p.selection_budget_small_scale = 1.0
    p.selection_extra_uniform_count = None
    p.selection_spatial_mode = None
    p.selection_spatial_zero_ids = None
    p.log_current_matched_m = False
    p.save_prompt_attention = False
    return p


def current_snapshot(env) -> dict:
    inner = env.unwrapped
    return {"sim_state": np.asarray(inner.get_state()).copy(),
            "agent_state": clone(inner.agent.get_state()),
            "rng_state": clone(inner._episode_rng.get_state()),
            "elapsed_steps": int(inner._elapsed_steps),
            "instruction": inner.get_language_instruction()}


def restore_mid(env, seed: int, snapshot: dict):
    env.reset(seed=seed)
    inner = env.unwrapped
    inner.set_state(snapshot["sim_state"].copy())
    inner.agent.set_state(clone(snapshot["agent_state"]))
    inner._episode_rng.set_state(clone(snapshot["rng_state"]))
    elapsed = int(snapshot["elapsed_steps"])
    inner._elapsed_steps = elapsed
    current = env
    while hasattr(current, "env"):
        if hasattr(current, "_elapsed_steps"):
            current._elapsed_steps = elapsed
        current = current.env
    return wrapped_observation(env)


def run_branch(env, policy_first, policy_matched, instruction, seed, step, obs):
    policy_first.reset(instruction, seed=seed)
    policy_first._selector_step = step
    policy_first._episode_trace = []; policy_first._episode_logits = []
    policy_matched.reset(instruction, seed=seed)
    policy_matched._selector_step = step + 1
    policy_matched._episode_trace = []; policy_matched._episode_logits = []
    predicted = truncated = False
    infos = []; control = 0; first_meta = None
    while not (predicted or truncated):
        image = np.asarray(get_image_from_maniskill2_obs_dict(env, obs), dtype=np.uint8)
        policy = policy_first if control == 0 else policy_matched
        _raw, actions, meta = policy.step(image, None, instruction, proprio=obs["agent"]["eef_pos"])
        if control == 0:
            first_meta = meta
        if not isinstance(actions, list):
            actions = [actions]
        for action in actions:
            obs, _r, _s, truncated, info = env.step(flatten_action(action))
            infos.append(info)
            predicted = bool(action["terminate_episode"][0] > 0)
            if predicted and not env.unwrapped.is_final_subtask():
                predicted = False; env.advance_to_next_subtask()
            instruction = env.unwrapped.get_language_instruction()
        control += 1
    return {"success": bool(any(bool(i.get("success", False)) for i in infos)),
            "continuation_control_steps": control,
            "first_selected_count": int(first_meta["actual_selected_count"]) if first_meta else None}


def select_states(root: Path):
    """Fixed rule, no cherry-picking: for each task, walk seeds in order; a seed
    contributes a state only if ALL FOUR progress states of that seed have a
    valid knee.  Take the first PER_STAGE such seeds."""
    pa = json.loads((root / "PHASE_A_RESULTS.json").read_text())
    by_seed = {}
    for r in pa["per_state"]:
        by_seed.setdefault((r["task"], r["seed"]), []).append(r)
    chosen = []
    for task in TASKS:
        taken = 0
        for seed in sorted(s for (t, s) in by_seed if t == task):
            rows = sorted(by_seed[(task, seed)], key=lambda r: r["step"])
            if len(rows) != 4 or not all(r["valid_knee"] for r in rows):
                continue
            chosen.extend(rows)
            taken += 1
            if taken >= PER_STAGE:
                break
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--canonical", type=Path, required=True)
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    repo = Path(__file__).resolve().parents[2]
    root = args.artifact.resolve()
    canonical = args.canonical.resolve()
    task = args.task
    out_dir = root / "phase_b" / task
    out_dir.mkdir(parents=True, exist_ok=True)

    states = [r for r in select_states(root) if r["task"] == task]
    if args.shard_count > 1:
        states = [r for i, r in enumerate(states) if i % args.shard_count == args.shard_index]
    if not states:
        print(json.dumps({"no_states": task}), flush=True); return

    env, environment_id = make_environment(task, args.gpu)
    from research.semantic_token_cd.distractor_rollout import PCD_SOURCE
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, task, {}, False)
    base = OpenVLAInference(**config)
    matched_first = build_policy(base, task, None)
    matched_cont = build_policy(base, task, None)
    fixed_first = build_policy(base, task, None)

    traj_dir = repo / TRAJ[task]
    actions_all = {}

    for st in states:
        seed = int(st["seed"]); step = int(st["step"])
        dst = out_dir / f"seed_{seed:03d}_step_{step:03d}.json"
        if dst.exists():
            print(json.dumps({"skip": True, "task": task, "seed": seed, "step": step}), flush=True)
            continue
        if seed not in actions_all:
            actions_all[seed] = np.load(traj_dir / f"episode_{seed:03d}_arrays.npz")["executed_actions"]
        actions = actions_all[seed]
        ref = load_reference(canonical, task, seed)
        with (canonical / "snapshots" / task / f"seed_{seed:03d}.pkl").open("rb") as fh:
            snap0 = pickle.load(fh)
        if snapshot_sha(snap0) != ref["canonical_snapshot_sha256"]:
            raise RuntimeError("canonical snapshot mismatch")
        obs, state_sha, rgb_sha = restore_snapshot(env, seed, snap0)
        if (state_sha, rgb_sha) != (ref["initial_state_sha256"], ref["initial_rgb_sha256"]):
            raise RuntimeError("restored snapshot mismatch")
        for t in range(step):
            obs = env.step(np.asarray(actions[t]))[0]
        image = np.asarray(get_image_from_maniskill2_obs_dict(env, obs), dtype=np.uint8)
        got = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
        # The canonical snapshot hash is verified above; this is a belt-and-braces
        # check that the replayed render matches Phase A.  Under the full pipeline
        # the renderer can differ slightly, so record rather than abort.
        image_hash_match = bool(got == st.get("image_sha256"))
        if not image_hash_match:
            print(json.dumps({"warn": "image_hash_mismatch", "task": task,
                              "seed": seed, "step": step}), flush=True)
        mid = current_snapshot(env)

        knee = int(st["knee_m"]); matched_m = int(st["matched_m"])
        topp80 = int(st.get("topp80_m", -1))
        cand = [("knee-16", knee - 16), ("knee-8", knee - 8), ("knee", knee),
                ("knee+8", knee + 8), ("knee+16", knee + 16),
                ("matched", matched_m), ("topp80", topp80), ("fixed32", 32)]
        arms = {}
        for name, m in cand:
            if m is None or m < 0:
                continue
            mc = int(np.clip(m, CLIP_LO, CLIP_HI))
            arms.setdefault(f"{name}|m={mc}", mc)
        results = {}
        started = time.monotonic()
        for name, mc in arms.items():
            obs2 = restore_mid(env, seed, mid)
            if name.startswith("matched"):
                fixed_first.selection_count = None
            else:
                fixed_first.selection_count = int(mc)
            results[name] = run_branch(env, fixed_first, matched_cont, mid["instruction"], seed, step, obs2)
        atomic_json(dst, {
            "protocol_id": "PROMPT_ATTN_L11_BUDGET_PHASE_B_V1",
            "task": task, "seed": seed, "step": step, "progress": float(st["progress"]),
            "knee_m": knee, "matched_m": matched_m, "topp80_m": topp80,
            "image_sha256": got, "image_hash_match": image_hash_match,
            "canonical_snapshot_sha256": ref["canonical_snapshot_sha256"],
            "environment_id": environment_id,
            "arms": {k: int(v) for k, v in arms.items()},
            "results": results,
            "runtime_seconds": time.monotonic() - started,
        })
        print(json.dumps({"task": task, "seed": seed, "step": step, "knee": knee,
                          "n_arms": len(arms),
                          "success": {k: v["success"] for k, v in results.items()},
                          "sec": round(time.monotonic() - started, 1)}), flush=True)


if __name__ == "__main__":
    main()
