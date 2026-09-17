"""Replay recorded actions for the two decisive Prompt/Action comparisons.

No policy inference is performed.  Each arm starts from the canonical snapshot,
then its saved executed actions are replayed to recover physical phase traces.
Outputs are one JSON per task/seed and are resumable after simulator crashes.
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

ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
SOURCE = Path("/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source")
for path in (ROOT, SOURCE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

NEW = ROOT / "artifacts/prompt_action_complement_v1/closed_loop/episodes"
OLD = ROOT / "artifacts/prompt_attn_layer_selection_v1"
SNAP = ROOT / "artifacts/vanilla_recon_shr_canonical_0_299_v2/snapshots"
OUT = ROOT / "artifacts/prompt_action_complement_v1/mechanism_analysis/trajectory_replay"


def clone(v):
    if isinstance(v, np.ndarray):
        return v.copy()
    if isinstance(v, dict):
        return type(v)((k, clone(x)) for k, x in v.items())
    if isinstance(v, list):
        return [clone(x) for x in v]
    if isinstance(v, tuple):
        return tuple(clone(x) for x in v)
    return copy.deepcopy(v)


def restore(env, seed: int, snap: dict):
    env.reset(seed=seed)
    u = env.unwrapped
    u.set_state(np.asarray(snap["sim_state"]).copy())
    u.agent.set_state(clone(snap["agent_state"]))
    u._episode_rng.set_state(clone(snap["rng_state"]))
    u._elapsed_steps = 0
    return u


def result(task: str, arm: str, seed: int) -> bool:
    if arm == "original":
        sub = "closed_loop_remaining6" if task == "google_robot_close_drawer" else "closed_loop"
        p = OLD / sub / "episodes" / task / "prompt_single" / f"episode_{seed:03d}_summary.json"
    else:
        p = NEW / task / arm / f"episode_{seed:03d}_summary.json"
    return bool(json.loads(p.read_text())["success"])


def actions(task: str, arm: str, seed: int) -> np.ndarray:
    if arm == "original":
        sub = "closed_loop_remaining6" if task == "google_robot_close_drawer" else "closed_loop"
        p = OLD / sub / "episodes" / task / "prompt_single" / f"episode_{seed:03d}_arrays.npz"
    else:
        p = NEW / task / arm / f"episode_{seed:03d}_arrays.npz"
    return np.load(p, allow_pickle=False)["executed_actions"]


def category(task: str, candidate: str, seed: int) -> str:
    b, c = result(task, "original" if task.endswith("move_near") else "prompt_high_action_high", seed), result(task, candidate, seed)
    if c and not b:
        return "rescue"
    if b and not c:
        return "harm"
    return "both_success" if b else "both_failure"


def first_true(rows, key):
    return next((r["t"] for r in rows if r.get(key)), None)


def replay_move(env, u, act):
    rows = []
    src0 = np.asarray(u.episode_source_obj.pose.p, dtype=np.float64).copy()
    for t, a in enumerate(act):
        src = np.asarray(u.episode_source_obj.pose.p, dtype=np.float64)
        tgt = np.asarray(u.episode_target_obj.pose.p, dtype=np.float64)
        tcp = np.asarray(u.tcp.pose.p, dtype=np.float64)
        ev = dict(u.evaluate())
        rows.append({
            "t": t,
            "src_xy_move": float(np.linalg.norm(src[:2] - src0[:2])),
            "src_tgt_xy": float(np.linalg.norm(src[:2] - tgt[:2])),
            "tcp_src": float(np.linalg.norm(tcp - src)),
            "moved_correct_obj": bool(ev.get("moved_correct_obj", False)),
            "moved_wrong_obj": bool(ev.get("moved_wrong_obj", False)),
            "near_tgt_obj": bool(ev.get("near_tgt_obj", False)),
            "is_closest_to_tgt": bool(ev.get("is_closest_to_tgt", False)),
        })
        env.step(np.nan_to_num(np.asarray(a, dtype=np.float64)))
    ev = dict(u.evaluate())
    return {
        "n_steps": len(rows),
        "first_moved": first_true(rows, "moved_correct_obj"),
        "first_near": first_true(rows, "near_tgt_obj"),
        "min_src_tgt_xy": min(r["src_tgt_xy"] for r in rows),
        "min_tcp_src": min(r["tcp_src"] for r in rows),
        "final_src_xy_move": rows[-1]["src_xy_move"],
        "final_src_tgt_xy": rows[-1]["src_tgt_xy"],
        "final_eval": {k: bool(v) if isinstance(v, (bool, np.bool_)) else v for k, v in ev.items()},
        "trace": rows,
    }


def replay_close(env, u, act):
    rows = []
    joint = u.joint_idx
    q0 = float(np.asarray(u.art_obj.get_qpos())[joint])
    for t, a in enumerate(act):
        q = float(np.asarray(u.art_obj.get_qpos())[joint])
        rows.append({"t": t, "drawer_qpos": q, "progress": q0 - q,
                     "tcp_p": np.asarray(u.tcp.pose.p, dtype=np.float64).tolist(),
                     "gripper": float(a[6])})
        env.step(np.nan_to_num(np.asarray(a, dtype=np.float64)))
    thresholds = (0.01, 0.05, 0.10)
    return {
        "n_steps": len(rows), "qpos_initial": q0,
        "qpos_final": rows[-1]["drawer_qpos"],
        "qpos_min": min(r["drawer_qpos"] for r in rows),
        "first_progress": {str(x): next((r["t"] for r in rows if r["progress"] >= x), None) for x in thresholds},
        "trace": rows,
    }


def build(task):
    import gymnasium as gym
    import simpler_env  # noqa: F401
    env_id = "MoveNearGoogleBakedTexInScene-v1" if task.endswith("move_near") else "CloseDrawerCustomInScene-v0"
    return gym.make(env_id, obs_mode="state_dict", prepackaged_config=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("move", "close"), required=True)
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("DISPLAY", "")

    if args.task == "move":
        task, candidate = "google_robot_move_near", "prompt_high_action_high"
        baseline = "original"
        arms = (baseline, candidate)
    else:
        task, candidate = "google_robot_close_drawer", "prompt_low_action_high"
        baseline = "prompt_high_action_high"
        arms = (baseline, candidate)

    out_dir = OUT / task
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [s for s in range(100) if category(task, candidate, s) in ("rescue", "harm")]
    seeds = seeds[args.shard::args.num_shards]
    for seed in seeds:
        dst = out_dir / f"seed_{seed:03d}.json"
        if dst.exists():
            continue
        snap = pickle.loads((SNAP / task / f"seed_{seed:03d}.pkl").read_bytes())
        env = build(task)
        try:
            payload = {"task": task, "seed": seed, "category": category(task, candidate, seed), "arms": {}}
            for arm in arms:
                u = restore(env, seed, snap)
                a = actions(task, arm, seed)
                rec = replay_move(env, u, a) if args.task == "move" else replay_close(env, u, a)
                rec["success"] = result(task, arm, seed)
                payload["arms"][arm] = rec
            tmp = dst.with_suffix(".tmp")
            def _json(v):
                if isinstance(v, np.generic):
                    return v.item()
                if isinstance(v, np.ndarray):
                    return v.tolist()
                raise TypeError(type(v).__name__)
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json) + "\n")
            tmp.replace(dst)
            print(json.dumps({"task": task, "seed": seed, "category": payload["category"]}), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
