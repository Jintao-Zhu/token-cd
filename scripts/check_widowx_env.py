"""Is the widowx environment itself drivable, or is the policy the problem?

The bridge checkpoint also produced zero object interaction (moved_wrong_obj=False), so the
checkpoint is not the cause. This isolates the environment: step the same widowx task with
scripted actions and watch whether the robot TCP moves and whether the object state changes.

If random/scripted actions DO move the robot and disturb objects, the environment is fine and
the failure is in how the policy's actions are interpreted. If nothing moves even then, the
environment or the action interface is broken.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path("/home/leju-suzhou/zjt_ws")
sys.path.insert(0, str(ROOT / "token-cd"))
sys.path.insert(0, str(ROOT / "openpi/packages/openpi-client/src"))
sys.path.insert(0, str(ROOT / "openpi/src"))

import numpy as np  # noqa: E402


def obj_positions(env):
    """Positions of the task's source/target objects, by name."""
    out = {}
    try:
        for a in env.unwrapped.scene.get_all_actors():
            n = getattr(a, "name", "?")
            if any(k in n.lower() for k in ("carrot", "plate", "spoon", "towel", "cube", "block",
                                            "eggplant", "basket", "bowl")):
                try:
                    p = np.asarray(a.get_pose().p, dtype=np.float32).reshape(-1)
                    if p.size >= 3:
                        out[n] = p
                except Exception:
                    continue
    except Exception as exc:  # noqa: BLE001
        print(f"[env] obj_positions error: {type(exc).__name__}: {exc}", flush=True)
    return out


def main() -> None:
    task = os.environ.get("DET_TASK", "widowx_carrot_on_plate")
    seed = int(os.environ.get("DET_SEED", "0"))
    from research.semantic_token_cd.pi0_fast_simpler_l6_matched_rollout import _environment

    env = _environment(task, 8)
    obs, info = env.reset(seed=seed)
    cam = "3rd_view_camera"
    agent = env.unwrapped.agent
    robot = agent.robot

    tcp0 = np.asarray(env.unwrapped.tcp.get_pose().p, dtype=np.float32)
    o0 = obj_positions(env)
    print(f"[env] task={task} seed={seed}", flush=True)
    print(f"[env] action_space = {env.action_space}", flush=True)
    print(f"[env] tcp0 = {np.round(tcp0, 4).tolist()}", flush=True)
    for k, v in list(o0.items())[:6]:
        print(f"[env]   obj {k}: {np.round(v, 4).tolist()}", flush=True)

    rng = np.random.default_rng(0)
    for mode in ("zeros", "random", "open_gripper_forward"):
        obs, info = env.reset(seed=seed)
        tcp_start = np.asarray(env.unwrapped.tcp.get_pose().p, dtype=np.float32).copy()
        o_start = obj_positions(env)
        tcp_max = 0.0
        for step in range(30):
            if mode == "zeros":
                a = np.zeros(7, dtype=np.float32)
            elif mode == "random":
                a = rng.uniform(-0.2, 0.2, size=7).astype(np.float32)
            else:
                a = np.zeros(7, dtype=np.float32)
                a[0] = 0.05   # push forward in x
                a[6] = 1.0    # open gripper command
            obs, r, term, trunc, info = env.step(a)
            tcp_now = np.asarray(env.unwrapped.tcp.get_pose().p, dtype=np.float32)
            tcp_max = max(tcp_max, float(np.linalg.norm(tcp_now - tcp_start)))
        o_end = obj_positions(env)
        moved = {}
        for k in o_start:
            if k in o_end:
                a, b = o_start[k], o_end[k]
                if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape:
                    moved[k] = float(np.linalg.norm(b - a))
        print(f"[env] mode={mode:24s} tcp_displacement_max={tcp_max:.4f} m  "
              f"obj_moves={ {k: round(v,4) for k,v in moved.items()} }  "
              f"success={info.get('success')} moved_wrong={info.get('moved_wrong_obj')}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
