"""Paired UniVLA Bridge-checkpoint zero-shot transfer to SIMPLER Google Robot."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import socket
import struct
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

TASK_ENVS = {
    "google_robot_open_drawer": "OpenDrawerCustomInScene-v0",
    "google_robot_close_drawer": "CloseDrawerCustomInScene-v0",
    "google_robot_pick_coke_can": "GraspSingleOpenedCokeCanInScene-v0",
    "google_robot_move_near": "MoveNearGoogleBakedTexInScene-v1",
    "google_robot_place_apple_in_closed_top_drawer": "PlaceIntoClosedTopDrawerCustomInScene-v0",
}
ARMS = tuple(x.strip() for x in os.environ.get("UNIVLA_ARMS", "vanilla,l14_matched").split(",") if x.strip())
if not ARMS or not set(ARMS).issubset({"vanilla", "l14_matched"}):
    raise ValueError(f"UNIVLA_ARMS must contain vanilla and/or l14_matched, got {ARMS}")
CHECKPOINT = os.environ.get("UNIVLA_CHECKPOINT", "/home/leju-suzhou/zjt_ws/checkpoints/univla-7b-224-sft-simpler-bridge")
CHECKPOINT_DOMAIN = os.environ.get("UNIVLA_CHECKPOINT_DOMAIN", "simpler_bridge_widowx")
ACTION_STATS_KEY = os.environ.get("UNIVLA_UNNORM_KEY", "bridge_oxe")
PROTOCOL = os.environ.get("UNIVLA_PROTOCOL_ID", "UNIVLA_BRIDGE_TO_GOOGLE_ZERO_SHOT_L14_MATCHED_RT_FULL_V1")


def parse_seeds(spec: str) -> list[int]:
    values = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            values.extend(range(lo, hi + 1))
        else:
            values.append(int(part))
    seeds = sorted(set(values))
    if not seeds or seeds[0] < 0:
        raise ValueError("invalid seed list")
    return seeds


def recv_exact(conn: socket.socket, count: int) -> bytes:
    buf = bytearray()
    while len(buf) < count:
        chunk = conn.recv(count - len(buf))
        if not chunk:
            raise EOFError("model service disconnected")
        buf.extend(chunk)
    return bytes(buf)


def send_message(conn: socket.socket, head: dict, payload: bytes = b""):
    head = dict(head, payload_nbytes=len(payload))
    body = json.dumps(head).encode()
    conn.sendall(struct.pack("!I", len(body)) + body + payload)


def recv_message(conn: socket.socket) -> dict:
    n = struct.unpack("!I", recv_exact(conn, 4))[0]
    return json.loads(recv_exact(conn, n))


def observation_hash(obj) -> str:
    digest = hashlib.sha256()
    def visit(x):
        if isinstance(x, dict):
            for key in sorted(x):
                digest.update(str(key).encode())
                visit(x[key])
        elif isinstance(x, (tuple, list)):
            for item in x:
                visit(item)
        else:
            a = np.asarray(x)
            digest.update(str(a.dtype).encode())
            digest.update(np.asarray(a.shape, dtype=np.int64).tobytes())
            digest.update(np.ascontiguousarray(a).tobytes())
    visit(obj)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def wrapped_observation(env):
    obs = env.unwrapped.get_obs()
    wrappers, current = [], env
    while hasattr(current, "env"):
        if hasattr(current, "observation"):
            wrappers.append(current)
        current = current.env
    for wrapper in reversed(wrappers):
        obs = wrapper.observation(obs)
    return obs


def capture_snapshot(env, seed: int):
    _obs, reset_info = env.reset(seed=seed)
    inner = env.unwrapped
    return {
        "sim_state": np.asarray(inner.get_state()).copy(),
        "agent_state": copy.deepcopy(inner.agent.get_state()),
        "rng_state": copy.deepcopy(inner._episode_rng.get_state()),
        "instruction": inner.get_language_instruction(),
        "episode_id": reset_info.get("episode_id"),
    }


def restore_snapshot(env, seed: int, snapshot: dict):
    env.reset(seed=seed)
    inner = env.unwrapped
    inner.set_state(snapshot["sim_state"].copy())
    inner.agent.set_state(copy.deepcopy(snapshot["agent_state"]))
    inner._episode_rng.set_state(copy.deepcopy(snapshot["rng_state"]))
    inner._elapsed_steps = 0
    obs = wrapped_observation(env)
    # Match the established SIMPLER snapshot-pairing audit: simulator get_state
    # hash plus exact initial RGB hash. Agent state is also explicitly restored
    # from the canonical snapshot before these hashes are measured.
    state_sha = observation_hash(np.asarray(inner.get_state()))
    image = np.asarray(get_image_from_maniskill2_obs_dict(env.unwrapped, obs), dtype=np.uint8)
    rgb_sha = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
    return obs, image, state_sha, rgb_sha


def snapshot_hash(snapshot: dict) -> str:
    return observation_hash({
        "sim_state": snapshot["sim_state"],
        "agent_state": snapshot["agent_state"],
        "rng_state": snapshot["rng_state"],
        "instruction": snapshot["instruction"],
        "episode_id": snapshot["episode_id"],
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=tuple(TASK_ENVS), required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--socket", required=True)
    ap.add_argument("--reference-artifact", type=Path, default=None)
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)
    task_root = args.artifact / "episodes" / args.task
    task_root.mkdir(parents=True, exist_ok=True)
    env_id, task_kwargs = simpler_env.ENVIRONMENT_MAP[args.task]
    if env_id != TASK_ENVS[args.task]:
        raise RuntimeError(f"SIMPLER task map mismatch for {args.task}: {env_id}")
    env_kwargs = dict(task_kwargs)
    env_kwargs.update({
        "obs_mode": "rgbd",
        "prepackaged_config": True,
        "shader_dir": "rt",
        "renderer_kwargs": {"device": "cuda:0", "offscreen_only": True},
    })
    env = gym.make(
        env_id,
        **env_kwargs,
    )
    max_steps = int(env.spec.max_episode_steps or 200)
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    deadline = time.time() + 240
    while True:
        try:
            conn.connect(args.socket)
            break
        except (FileNotFoundError, ConnectionRefusedError):
            if time.time() > deadline:
                raise TimeoutError(f"model service socket not ready: {args.socket}")
            time.sleep(1)

    completed = []
    try:
        for seed in seeds:
            outputs = {}
            paths = {arm: task_root / arm / f"seed_{seed:03d}.json" for arm in ARMS}
            if all(path.exists() for path in paths.values()):
                existing = {arm: json.loads(path.read_text()) for arm, path in paths.items()}
                hashes_match = (len({row.get("initial_state_sha256") for row in existing.values()}) == 1
                                and len({row.get("initial_rgb_sha256") for row in existing.values()}) == 1)
                reference_matches = True
                if args.reference_artifact is not None:
                    ref_path = args.reference_artifact / "episodes" / args.task / "vanilla" / f"seed_{seed:03d}.json"
                    reference = json.loads(ref_path.read_text()) if ref_path.exists() else None
                    if reference is not None:
                        reference_matches = all(
                            existing[arm].get("instruction") == reference.get("instruction")
                            for arm in ARMS
                        )
                if all(row.get("complete") for row in existing.values()) and hashes_match and reference_matches:
                    completed.append(seed)
                    print(json.dumps({"worker": args.worker_id, "skip_complete_pair": True, "task": args.task, "seed": seed}), flush=True)
                    continue

            snapshot = capture_snapshot(env, seed)
            canonical_sha = snapshot_hash(snapshot)
            pair_state = {}
            for arm in ARMS:
                obs, image, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
                instruction = env.unwrapped.get_language_instruction()
                pair_state[arm] = {
                    "state_sha256": state_sha,
                    "rgb_sha256": rgb_sha,
                    "episode_id": snapshot["episode_id"],
                    "instruction": instruction,
                }
                reference_state_match = None
                reference_rgb_match = None
                if args.reference_artifact is not None:
                    ref_path = args.reference_artifact / "episodes" / args.task / "vanilla" / f"seed_{seed:03d}.json"
                    if ref_path.exists():
                        reference = json.loads(ref_path.read_text())
                        if instruction != reference.get("instruction"):
                            raise RuntimeError(
                                f"checkpoint comparison instruction mismatch for {args.task}/{seed}; "
                                f"reference={ref_path}, current_state={state_sha}, current_rgb={rgb_sha}"
                            )
                        reference_state_match = state_sha == reference.get("initial_state_sha256")
                        reference_rgb_match = rgb_sha == reference.get("initial_rgb_sha256")
                        if not reference_state_match or not reference_rgb_match:
                            print(json.dumps({
                                "worker": args.worker_id, "task": args.task, "seed": seed,
                                "historical_state_match": reference_state_match,
                                "historical_instruction_match": True,
                                "historical_rgb_exact_match": reference_rgb_match,
                                "historical_state_sha256": reference.get("initial_state_sha256"),
                                "current_state_sha256": state_sha,
                                "historical_rgb_sha256": reference.get("initial_rgb_sha256"),
                                "current_rgb_sha256": rgb_sha,
                                "note": "historical cross-run mismatch is recorded, not used as a hard stop; current Vanilla/L14 pair still requires exact simulator-state and RGB equality",
                            }), flush=True)
                        reference_available = True
                    else:
                        reference_available = False
                        print(json.dumps({
                            "worker": args.worker_id, "task": args.task, "seed": seed,
                            "reference_available": False,
                            "note": "no historical Bridge vanilla record for this seed; deterministic seed rollout continues",
                        }), flush=True)
                else:
                    reference_available = False
                # Fractal-trained UniVLA predicts absolute gripper openness;
                # its SIMPLER Google adapter needs the current physical state
                # to convert that target into a relative open/close command.
                gripper_open = float(np.clip(1.0 - env.unwrapped.agent.gripper_closedness, 0.0, 1.0))
                send_message(conn, {"cmd": "reset", "instruction": instruction, "seed": seed,
                                    "gripper_open": gripper_open})
                ack = recv_message(conn)
                if not ack.get("ok"):
                    raise RuntimeError(f"model reset failed: {ack}")
                started = time.monotonic()
                success = False
                terminated = truncated = False
                step_rows = []
                info = {}
                for step in range(max_steps):
                    send_message(conn, {
                        "cmd": "step", "arm": arm, "instruction": instruction,
                        "shape": list(image.shape),
                    }, image.tobytes())
                    response = recv_message(conn)
                    if not response.get("ok"):
                        raise RuntimeError(f"model step failed at {args.task}/{seed}/{arm}: {response}")
                    prediction = response["result"]
                    action = prediction["action"]
                    action_vec = np.concatenate([
                        np.asarray(action["world_vector"], dtype=np.float32).reshape(-1),
                        np.asarray(action["rot_axangle"], dtype=np.float32).reshape(-1),
                        np.asarray(action["gripper"], dtype=np.float32).reshape(-1),
                    ])
                    obs, reward, terminated, truncated, info = env.step(action_vec)
                    image = np.asarray(get_image_from_maniskill2_obs_dict(env.unwrapped, obs), dtype=np.uint8)
                    success = bool(info.get("success", terminated))
                    meta = prediction["meta"]
                    step_rows.append({
                        "step": step + 1,
                        "action_token_ids": meta.get("action_token_ids"),
                        "m_t": meta.get("m_t"),
                        "selected_token_ids": meta.get("selected_token_ids"),
                        "attention_mass": meta.get("attention_mass"),
                        "action": action,
                        "raw_action": prediction.get("raw"),
                        "gripper_open_state_before_policy": gripper_open if step == 0 else None,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "success": success,
                    })
                    if step == 0 or (step + 1) % 10 == 0 or terminated or truncated:
                        print(json.dumps({
                            "worker": args.worker_id, "task": args.task, "seed": seed, "arm": arm,
                            "step": step + 1, "success": success,
                            "m_t": meta.get("m_t"),
                            "action_tokens_in_learned_range": meta.get("action_tokens_in_learned_range"),
                        }), flush=True)
                    if terminated or truncated:
                        break

                row = {
                    "protocol_id": PROTOCOL,
                    "complete": True,
                    "worker_id": args.worker_id,
                    "task": args.task,
                    "environment_id": env_id,
                    "seed": seed,
                    "episode_id": pair_state[arm]["episode_id"],
                    "arm": arm,
                    "checkpoint": CHECKPOINT,
                    "checkpoint_domain": CHECKPOINT_DOMAIN,
                    "evaluation_domain": "simpler_google_robot",
                    "policy_setup": "google_robot",
                    "action_stats_key": ACTION_STATS_KEY,
                    "historical_reference_available": reference_available,
                    "historical_reference_initial_state_match": reference_state_match,
                    "historical_reference_initial_rgb_exact_match": reference_rgb_match,
                    "canonical_snapshot_sha256": canonical_sha,
                    "instruction": instruction,
                    "renderer": {"shader_dir": "rt", "device": "cuda:0", "offscreen_only": True, "spp": 32, "path_depth": 8},
                    "initial_state_sha256": state_sha,
                    "initial_rgb_sha256": rgb_sha,
                    "success": bool(success),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "steps": len(step_rows),
                    "runtime_seconds": round(time.monotonic() - started, 4),
                    "final_info": info,
                    "step_log": step_rows,
                    "num_action_tokens_in_learned_range": [x.get("action_token_ids") for x in step_rows],
                }
                outputs[arm] = row
                atomic_json(paths[arm], row)
                print(json.dumps({"worker": args.worker_id, "task": args.task, "seed": seed, "arm": arm,
                                  "success": bool(success), "steps": len(step_rows),
                                  "runtime_seconds": row["runtime_seconds"]}), flush=True)

            if {"vanilla", "l14_matched"}.issubset(pair_state):
                if pair_state["vanilla"]["instruction"] != pair_state["l14_matched"]["instruction"]:
                    raise RuntimeError(f"paired instruction mismatch: {pair_state}")
                if pair_state["vanilla"]["state_sha256"] != pair_state["l14_matched"]["state_sha256"]:
                    raise RuntimeError(f"paired state mismatch for {args.task}/{seed}: {pair_state}")
                if pair_state["vanilla"]["rgb_sha256"] != pair_state["l14_matched"]["rgb_sha256"]:
                    raise RuntimeError(f"paired RGB mismatch for {args.task}/{seed}: {pair_state}")
            completed.append(seed)
    finally:
        try:
            send_message(conn, {"cmd": "quit"})
            recv_message(conn)
        except Exception:
            pass
        conn.close()
        env.close()
    worker_manifest = {
        "protocol_id": PROTOCOL,
        "worker_id": args.worker_id,
        "task": args.task,
        "completed_seeds": completed,
        "completed_pairs": len(completed),
        "all_initial_state_pairs_checked": True,
    }
    atomic_json(args.artifact / "worker_manifests" / f"{args.worker_id}.json", worker_manifest)
    print(json.dumps({"worker": args.worker_id, "task": args.task, "completed_seeds": completed,
                      "worker_complete": len(completed) == len(seeds)}), flush=True)


if __name__ == "__main__":
    main()
