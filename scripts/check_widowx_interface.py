"""Does the gripper fix make the WidowX robot actually interact?

Runs one episode on a WidowX task with the bridge checkpoint and logs, per step, the raw
policy output, the processed action, and the robot TCP displacement. The previous behaviour
was moved_wrong_obj=False on 100% of episodes while the arm still moved, which is what a
stuck-at-zero gripper command looks like.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
ROOT = Path("/home/leju-suzhou/zjt_ws"); TOKEN_CD = ROOT / "token-cd"
sys.path.insert(0, str(TOKEN_CD))
sys.path.insert(0, str(ROOT / "openpi/packages/openpi-client/src"))
sys.path.insert(0, str(ROOT / "openpi/src"))
import numpy as np

def main():
    task = os.environ.get("WK_TASK", "widowx_carrot_on_plate")
    seed = int(os.environ.get("WK_SEED", "0"))
    port = int(os.environ.get("WK_PORT", "8121"))
    from research.semantic_token_cd.pi0_fast_simpler_l6_matched_rollout import (
        ActionChunk, RolloutPolicy, _environment, _raw_policy_input)
    policy = RolloutPolicy("127.0.0.1", port)
    env = _environment(task, 8)
    obs, _ = env.reset(seed=seed)
    instruction = str(env.unwrapped.get_language_instruction())
    camera = "3rd_view_camera"
    chunk = ActionChunk("widowx_bridge")
    tcp0 = np.asarray(env.unwrapped.tcp.get_pose().p, dtype=np.float32).copy()
    gripper_cmds, raws, tcps = [], [], []
    steps = 0; terminated = truncated = False; info = {}
    while not (terminated or truncated) and steps < 80:
        image = obs["image"][camera]["rgb"]
        raw = _raw_policy_input(image, instruction, obs["agent"], env)
        if steps % chunk.exec_horizon == 0:
            act, _ = policy.chunk("vanilla", raw, seed * 1009 + steps // chunk.exec_horizon)
            chunk.plan.extend(act[: chunk.exec_horizon])
        ra = chunk.plan.popleft()
        a = chunk.next(ra)
        raws.append(float(ra[6])); gripper_cmds.append(float(a[6]))
        obs, r, terminated, truncated, info = env.step(a)
        tcps.append(float(np.linalg.norm(np.asarray(env.unwrapped.tcp.get_pose().p, dtype=np.float32) - tcp0)))
        steps += 1
    es = (info or {}).get("episode_stats") or {}
    print(json.dumps({
        "task": task, "seed": seed, "steps": steps, "success": bool((info or {}).get("success")),
        "tcp_displacement_max_m": max(tcps) if tcps else 0.0,
        "gripper_cmd_unique": sorted(set(round(g, 3) for g in gripper_cmds)),
        "gripper_cmd_n_zero": sum(1 for g in gripper_cmds if abs(g) < 1e-6),
        "raw_openness_range": [round(min(raws), 3), round(max(raws), 3)] if raws else None,
        "episode_stats": es,
    }, indent=2, default=str))
    env.close()

if __name__ == "__main__":
    main()
