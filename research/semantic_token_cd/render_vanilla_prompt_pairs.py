"""Render fixed-order Vanilla vs Prompt-L11-Matched paired outcome cases.

This script does no model inference. It restores each canonical snapshot and
replays the exact actions saved by the completed experiments. It writes raw arm
videos, synchronized side-by-side videos, slow key-divergence clips, and an
event/geometry JSON used for human video review.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
SOURCE = Path("/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source")
for p in (ROOT, SOURCE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CANON = ROOT / "artifacts/vanilla_recon_shr_canonical_0_299_v2"
LAYER = ROOT / "artifacts/prompt_attn_layer_selection_v1"
OUT = ROOT / "artifacts/vanilla_vs_prompt_l11_matched_video_analysis"

TASK_SUBDIR = {
    "google_robot_open_drawer": "closed_loop",
    "google_robot_close_drawer": "closed_loop_remaining6",
    "google_robot_pick_coke_can": "closed_loop",
}
TASK_SHORT = {k: k.removeprefix("google_robot_") for k in TASK_SUBDIR}


def clone(v):
    if isinstance(v, np.ndarray): return v.copy()
    if isinstance(v, dict): return type(v)((k, clone(x)) for k, x in v.items())
    if isinstance(v, list): return [clone(x) for x in v]
    if isinstance(v, tuple): return tuple(clone(x) for x in v)
    return copy.deepcopy(v)


def restore(env, seed, snap):
    env.reset(seed=seed)
    u = env.unwrapped
    u.set_state(np.asarray(snap["sim_state"]).copy())
    u.agent.set_state(clone(snap["agent_state"]))
    u._episode_rng.set_state(clone(snap["rng_state"]))
    u._elapsed_steps = 0
    from research.semantic_token_cd.rollout_pilot import wrapped_observation
    return wrapped_observation(env)


def make_env(task):
    import gymnasium as gym
    import simpler_env
    kw = {"renderer_kwargs": {"device": "cuda:0", "offscreen_only": True}}
    if task == "google_robot_pick_coke_can":
        return gym.make("GraspSingleOpenedCokeCanDistractorInScene-v0", obs_mode="rgbd",
                        prepackaged_config=True, distractor_config="less", **kw)
    env_id, base = simpler_env.ENVIRONMENT_MAP[task]
    args = dict(base); args.update(prepackaged_config=True, **kw)
    return gym.make(env_id, obs_mode="rgbd", **args)


def paths(task, seed, arm):
    if arm == "vanilla":
        d = CANON / "episodes" / task / arm
    else:
        d = LAYER / TASK_SUBDIR[task] / "episodes" / task / "prompt_single"
    return d / f"episode_{seed:03d}_summary.json", d / f"episode_{seed:03d}_arrays.npz"


def font(size=19):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"):
        if Path(p).exists(): return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def status(task, u, info):
    if "drawer" in task:
        q = float(np.asarray(u.art_obj.get_qpos())[u.joint_idx])
        ok = q >= .15 if "open" in task else q <= .05
        return {"qpos": q, "success_now": ok}
    obj = np.asarray(u.obj_pose.p, dtype=float)
    tcp = np.asarray(u.tcp.pose.p, dtype=float)
    return {
        "tcp_obj_m": float(np.linalg.norm(tcp - obj)),
        "obj_z": float(obj[2]),
        "near_obj": bool(info.get("near_obj", False)),
        "is_grasped": bool(info.get("is_grasped", info.get("grasped", False))),
        "success_now": bool(info.get("success", False)),
    }


def label_frame(frame, arm, seed, task, t, st, saved_success):
    im = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    bar = Image.new("RGB", (im.width, 64), (18, 18, 18)); bar.paste(im, (0, 64))
    canvas = Image.new("RGB", (im.width, im.height + 64), (18, 18, 18)); canvas.paste(im, (0, 64))
    d = ImageDraw.Draw(canvas); f = font(18)
    d.text((10, 5), f"{arm}  |  {task}  seed {seed:03d}  |  step {t:03d}", fill="white", font=f)
    if "qpos" in st:
        detail = f"drawer qpos={st['qpos']:.3f}   now={'SUCCESS' if st['success_now'] else 'not done'}"
    else:
        detail = f"tcp-object={100*st['tcp_obj_m']:.1f}cm   grasp={int(st['is_grasped'])}   now={'SUCCESS' if st['success_now'] else 'not done'}"
    d.text((10, 34), detail + f"   saved final={int(saved_success)}", fill=(255, 220, 90), font=font(16))
    return np.asarray(canvas)


def replay(env, task, seed, arm, snap, actions, saved_success):
    from parallel_inference import get_image_from_maniskill2_obs_dict
    obs = restore(env, seed, snap); u = env.unwrapped
    frames, trace = [], []
    info = {}
    for t in range(len(actions) + 1):
        st = status(task, u, info); trace.append({"t": t, **st})
        image = get_image_from_maniskill2_obs_dict(env, obs)
        frames.append(label_frame(image, "Vanilla" if arm == "vanilla" else "Prompt-L11-Matched",
                                  seed, TASK_SHORT[task], t, st, saved_success))
        if t == len(actions): break
        obs, _r, _term, _trunc, info = env.step(np.nan_to_num(actions[t].astype(np.float64)))
    reproduced = any(x["success_now"] for x in trace)
    return frames, trace, reproduced


def event_steps(task, trace):
    if "drawer" in task:
        q0 = trace[0]["qpos"]
        direction = 1 if "open" in task else -1
        def first(delta):
            return next((x["t"] for x in trace if direction * (x["qpos"] - q0) >= delta), None)
        return {"first_motion_1cm": first(.01), "progress_5cm": first(.05),
                "progress_10cm": first(.10), "task_complete": next((x["t"] for x in trace if x["success_now"]), None)}
    return {
        "first_near": next((x["t"] for x in trace if x["near_obj"]), None),
        "first_grasp": next((x["t"] for x in trace if x["is_grasped"]), None),
        "task_complete": next((x["t"] for x in trace if x["success_now"]), None),
    }


def key_step(task, left, right):
    n = min(len(left), len(right))
    if "drawer" in task:
        for i in range(n):
            if abs(left[i]["qpos"] - right[i]["qpos"]) >= .025: return i
    else:
        for i in range(n):
            if (left[i]["near_obj"], left[i]["is_grasped"], left[i]["success_now"]) != (right[i]["near_obj"], right[i]["is_grasped"], right[i]["success_now"]): return i
        for i in range(n):
            if abs(left[i]["tcp_obj_m"] - right[i]["tcp_obj_m"]) >= .03: return i
    return min(n - 1, n // 2)


def pad(frames, n):
    return frames + [frames[-1]] * (n - len(frames))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASK_SUBDIR, required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    seeds = [int(x) for x in args.seeds.split(",")]
    task_out = OUT / TASK_SHORT[args.task]; task_out.mkdir(parents=True, exist_ok=True)
    env = make_env(args.task)
    try:
        for seed in seeds:
            dst = task_out / f"seed_{seed:03d}_analysis.json"
            if dst.exists():
                print(json.dumps({"skip": str(dst)}), flush=True); continue
            sp = CANON / "snapshots" / args.task / f"seed_{seed:03d}.pkl"
            snap = pickle.loads(sp.read_bytes())
            data, frames, traces = {}, {}, {}
            hashes = []
            for arm in ("vanilla", "prompt"):
                sm, ar = paths(args.task, seed, arm)
                summary = json.loads(sm.read_text()); arrays = np.load(ar, allow_pickle=False)
                hashes.append((summary["canonical_snapshot_sha256"], summary["initial_state_sha256"], summary["initial_rgb_sha256"]))
                fs, tr, reproduced = replay(env, args.task, seed, arm, snap, arrays["executed_actions"], bool(summary["success"]))
                data[arm] = {"saved_success": bool(summary["success"]), "reproduced_ever_success": reproduced,
                             "n_actions": len(arrays["executed_actions"]), "events": event_steps(args.task, tr)}
                frames[arm], traces[arm] = fs, tr
                imageio.mimwrite(task_out / f"seed_{seed:03d}_{arm}.mp4", fs, fps=10, codec="libx264", quality=7, macro_block_size=None)
            if len(set(hashes)) != 1: raise RuntimeError(f"hash mismatch {args.task} seed {seed}")
            category = "rescue" if (not data["vanilla"]["saved_success"] and data["prompt"]["saved_success"]) else "harm"
            n = max(map(len, frames.values())); lf, rf = pad(frames["vanilla"], n), pad(frames["prompt"], n)
            pair = [np.concatenate([lf[i], rf[i]], axis=1) for i in range(n)]
            imageio.mimwrite(task_out / f"{category}_seed_{seed:03d}_paired.mp4", pair, fps=10, codec="libx264", quality=7, macro_block_size=None)
            key = key_step(args.task, traces["vanilla"], traces["prompt"]); lo=max(0,key-12); hi=min(n,key+21)
            imageio.mimwrite(task_out / f"{category}_seed_{seed:03d}_key_slow.mp4", pair[lo:hi], fps=3, codec="libx264", quality=7, macro_block_size=None)
            payload = {"task": args.task, "seed": seed, "category": category, "paired_hash": hashes[0],
                       "key_divergence_step": key, "slow_clip_steps": [lo, hi-1], "arms": data,
                       "trace": traces}
            dst.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            print(json.dumps({"task": args.task, "seed": seed, "category": category, "key": key}), flush=True)
    finally:
        env.close()


if __name__ == "__main__": main()
