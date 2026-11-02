"""Vanilla pi0 baseline on gym_aloha sim tasks (TransferCube / Insertion).

Uses the official openpi observation convention:
  state  = obs["agent_pos"]            (14,)
  images = {"cam_high": top image}     (3,224,224) after resize+transpose
and executes each predicted 50-step chunk with an action horizon of 10.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np


def convert_obs(gym_obs, image_tools):
    img = gym_obs["pixels"]["top"]
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
    img = np.transpose(img, (2, 0, 1))
    return {"state": np.asarray(gym_obs["agent_pos"], dtype=np.float32),
            "images": {"cam_high": img}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--task", default="gym_aloha/AlohaTransferCube-v0")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8126)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    args = parser.parse_args()

    import gymnasium
    import gym_aloha  # noqa: F401
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy

    args.artifact.mkdir(parents=True, exist_ok=True)
    out = args.artifact / "episodes.jsonl"
    seen = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                seen.add(int(json.loads(line)["episode"]))
    env = gymnasium.make(args.task, obs_type="pixels_agent_pos", render_mode="rgb_array")
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    for episode in range(args.seed0, args.seed0 + args.episodes):
        if episode in seen:
            continue
        started = time.monotonic()
        gym_obs, _ = env.reset(seed=episode)
        obs = convert_obs(gym_obs, image_tools)
        queue = []
        steps = 0
        best_reward = 0.0
        infers = 0
        done = False
        while steps < args.max_steps and not done:
            if not queue:
                response = client.infer({"state": obs["state"], "images": obs["images"], "prompt": ""})
                actions = np.asarray(response["actions"], dtype=np.float32)
                queue.extend(actions[: args.action_horizon])
                infers += 1
            action = queue.pop(0)
            gym_obs, reward, terminated, truncated, info = env.step(action)
            best_reward = max(best_reward, float(reward))
            obs = convert_obs(gym_obs, image_tools)
            steps += 1
            done = bool(terminated or truncated)
        record = {"episode": episode, "task": args.task, "success": bool(best_reward >= 4.0),
                  "best_reward": best_reward, "steps": steps, "infers": infers,
                  "seconds": round(time.monotonic() - started, 1)}
        with out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps(record, sort_keys=True), flush=True)
    env.close()


if __name__ == "__main__":
    main()
