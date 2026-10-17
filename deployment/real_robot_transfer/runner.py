"""Offline image probe and gated real-robot inference loop.

Run from the repository root with ``python -m deployment.real_robot_transfer.runner``.
No hardware command is sent unless ``--execute`` is explicitly present.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .adapter import Observation
from .policy import infer, load_policy


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def load_adapter(spec: str):
    if ":" not in spec:
        raise ValueError("--adapter expects module:ClassName")
    module_name, class_name = spec.split(":", 1)
    return getattr(importlib.import_module(module_name), class_name)()


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenVLA VTCG robot transfer")
    parser.add_argument("--checkpoint", required=True, help="Complete local OpenVLA checkpoint directory")
    parser.add_argument("--unnorm-key", required=True, help="Embodiment-matched action statistics key")
    parser.add_argument("--instruction", required=True, help="Exact task prompt")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="One RGB image for offline inference")
    source.add_argument("--adapter", help="Installed adapter as module:ClassName")
    parser.add_argument("--log", type=Path, required=True, help="Append-only JSONL inference record")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--interval-s", type=float, default=0.0)
    parser.add_argument("--execute", action="store_true", help="Permit adapter.send_command")
    args = parser.parse_args()
    if args.execute and not args.adapter:
        parser.error("--execute requires a physical adapter")
    if args.steps < 1 or args.interval_s < 0:
        parser.error("--steps must be positive and --interval-s nonnegative")
    if args.image and args.steps != 1:
        parser.error("--image performs exactly one inference")

    policy = load_policy(args.checkpoint, args.unnorm_key, args.instruction,
                         seed=args.seed, device=args.device)
    adapter = load_adapter(args.adapter) if args.adapter else None
    args.log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.log.open("a", encoding="utf-8") as output:
            for step in range(args.steps):
                if adapter is None:
                    rgb = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.uint8)
                    observation = Observation(rgb, time.time(), {})
                else:
                    observation = adapter.observe()
                rgb = observation.rgb
                if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
                    raise ValueError("adapter.observe() must return RGB uint8 [H,W,3]")
                raw_action, trace = infer(policy, rgb, args.instruction)
                command = adapter.map_action(raw_action, observation) if adapter else None
                record = {
                    "schema": "token-cd-real-robot-v1", "step": step,
                    "timestamp_s": observation.timestamp_s,
                    "image_sha256": hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest(),
                    "image_shape": list(rgb.shape), "instruction": args.instruction,
                    "unnorm_key": args.unnorm_key, "robot_state": observation.robot_state,
                    "raw_action": raw_action, "mapped_command": command,
                    "method_trace": trace, "command_sent": False,
                }
                # Write the proposal before an enabled physical command.
                output.write(json.dumps(jsonable(record), ensure_ascii=False) + "\n")
                output.flush()
                if args.execute:
                    adapter.send_command(command)
                    output.write(json.dumps({"schema": "token-cd-command-event-v1",
                                             "step": step, "timestamp_s": time.time(),
                                             "command_sent": True}) + "\n")
                    output.flush()
                print(json.dumps({"step": step, "raw_action": jsonable(raw_action),
                                  "mapped_command": jsonable(command),
                                  "command_sent": bool(args.execute)}, ensure_ascii=False))
                if step + 1 < args.steps and args.interval_s:
                    time.sleep(args.interval_s)
    finally:
        if adapter is not None:
            adapter.close()


if __name__ == "__main__":
    main()
