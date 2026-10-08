"""Paired UniVLA RT-depth0 WidowX rollout (Vanilla vs Hidden-CD 0.5)."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import socket
import struct
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
TASK_ENVS = {
    "widowx_carrot_on_plate": "PutCarrotOnPlateInScene-v0",
    "widowx_put_eggplant_in_basket": "PutEggplantInBasketScene-v0",
    "widowx_spoon_on_towel": "PutSpoonOnTableClothInScene-v0",
    "widowx_stack_cube": "StackGreenCubeOnYellowCubeBakedTexInScene-v0",
}
ARMS = ("vanilla", "hidden_cd_05")
MAX_STEPS = 80
FULL_PILOT = ROOT / "artifacts/univla_hidden_cd_rtfull_4arm_pilot_v1"
FULL_REFERENCE = FULL_PILOT / "episodes"


def parse_seeds(spec: str) -> list[int]:
    values = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            values.extend(range(lo, hi + 1))
        elif part:
            values.append(int(part))
    result = sorted(set(values))
    if not result or result[0] < 0 or result[-1] > 99:
        raise ValueError("this paired depth0 comparison is locked to seeds 0-99")
    return result


def sha_array(value) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def atomic_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def recv_exact(conn: socket.socket, count: int) -> bytes:
    data = bytearray()
    while len(data) < count:
        part = conn.recv(count - len(data))
        if not part:
            raise EOFError("model service disconnected")
        data.extend(part)
    return bytes(data)


def send_message(conn: socket.socket, head: dict, payload: bytes = b""):
    head = dict(head, payload_nbytes=len(payload))
    body = json.dumps(head).encode()
    conn.sendall(struct.pack("!I", len(body)) + body + payload)


def recv_message(conn: socket.socket) -> dict:
    size = struct.unpack("!I", recv_exact(conn, 4))[0]
    return json.loads(recv_exact(conn, size))


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


def observation_hash(obj) -> str:
    digest = hashlib.sha256()
    def visit(x):
        if isinstance(x, dict):
            for key in sorted(x):
                digest.update(str(key).encode()); visit(x[key])
        elif isinstance(x, (tuple, list)):
            for item in x: visit(item)
        else:
            a = np.asarray(x)
            digest.update(str(a.dtype).encode())
            digest.update(np.asarray(a.shape, dtype=np.int64).tobytes())
            digest.update(np.ascontiguousarray(a).tobytes())
    visit(obj)
    return digest.hexdigest()


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
    state_sha = observation_hash(np.asarray(inner.get_state()))
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    image = np.asarray(get_image_from_maniskill2_obs_dict(env.unwrapped, obs), dtype=np.uint8)
    return obs, image, state_sha, sha_array(image)


def snapshot_hash(snapshot: dict) -> str:
    return observation_hash({k: snapshot[k] for k in
                             ("sim_state", "agent_state", "rng_state", "instruction", "episode_id")})


def make_rt0_env(task: str):
    import gymnasium as gym
    import simpler_env
    kwargs = dict(simpler_env.ENVIRONMENT_MAP[task][1])
    kwargs.update({"obs_mode": "rgbd", "prepackaged_config": True, "shader_dir": "rt",
                   "render_config": {"rt_samples_per_pixel": 32, "rt_max_path_depth": 0,
                                     "rt_use_denoiser": True},
                   "renderer_kwargs": {"device": "cuda:0", "offscreen_only": True}})
    return gym.make(simpler_env.ENVIRONMENT_MAP[task][0], **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=tuple(TASK_ENVS), required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--socket", required=True)
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)
    artifact = args.artifact.resolve()
    env = make_rt0_env(args.task)
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    deadline = time.time() + 300
    while True:
        try:
            conn.connect(args.socket); break
        except (FileNotFoundError, ConnectionRefusedError):
            if time.time() > deadline: raise TimeoutError(f"model service not ready: {args.socket}")
            time.sleep(1)

    completed = []
    try:
        for seed in seeds:
            task_dir = artifact / "episodes" / args.task
            paths = {arm: task_dir / arm / f"seed_{seed:03d}.json" for arm in ARMS}
            pair_path = artifact / "pairs" / args.task / f"seed_{seed:03d}.json"
            if all(p.exists() for p in paths.values()) and pair_path.exists():
                old = {a: json.loads(p.read_text()) for a, p in paths.items()}
                pair = json.loads(pair_path.read_text())
                if (all(x.get("complete") for x in old.values()) and pair.get("exact_initial_pair")
                        and all(x.get("renderer", {}).get("path_depth") == 0 for x in old.values())):
                    completed.append(seed); continue

            ref_path = FULL_REFERENCE / args.task / "vanilla" / f"seed_{seed:03d}.json"
            ref = json.loads(ref_path.read_text())
            if not ref.get("complete") or ref.get("renderer", {}).get("path_depth") != 8:
                raise RuntimeError(f"missing/incompatible RT-full seed reference: {ref_path}")
            source_pair_path = FULL_PILOT / "pairs" / args.task / f"seed_{seed:03d}.json"
            source_pair = json.loads(source_pair_path.read_text())
            if not source_pair.get("exact_initial_pair"):
                raise RuntimeError(f"source RT-full pilot pair is not exact: {source_pair_path}")
            snapshot_path = FULL_PILOT / "snapshots" / args.task / f"seed_{seed:03d}.pkl"
            if not snapshot_path.is_file():
                raise FileNotFoundError(f"missing RT-full canonical snapshot: {snapshot_path}")
            snapshot = pickle.loads(snapshot_path.read_bytes())
            canonical_sha = source_pair["canonical_snapshot_sha256"]
            if source_pair.get("initial_state_sha256") != ref.get("initial_state_sha256"):
                raise RuntimeError(f"source snapshot reference state mismatch: {source_pair_path}")
            pair_rows = {}
            first_rgb = None
            for arm in ARMS:
                obs, image, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
                if state_sha != ref.get("initial_state_sha256"):
                    raise RuntimeError(f"physical initial state differs from RT-full {args.task}:{seed}")
                instruction = env.unwrapped.get_language_instruction()
                if instruction != ref.get("instruction") or instruction != snapshot["instruction"]:
                    raise RuntimeError(f"instruction mismatch at {args.task}:{seed}")
                if first_rgb is not None and rgb_sha != first_rgb:
                    raise RuntimeError(f"depth0 paired RGB mismatch at {args.task}:{seed}")
                first_rgb = rgb_sha

                send_message(conn, {"cmd": "reset", "instruction": instruction, "seed": seed})
                ack = recv_message(conn)
                if not ack.get("ok"): raise RuntimeError(f"model reset failed: {ack}")
                start = time.monotonic(); success = False; terminated = truncated = False
                info = {}; trace = []
                for step in range(MAX_STEPS):
                    send_message(conn, {"cmd": "step", "arm": arm, "instruction": instruction,
                                        "shape": list(image.shape)}, image.tobytes())
                    response = recv_message(conn)
                    if not response.get("ok"): raise RuntimeError(f"inference failed: {response}")
                    prediction = response["result"]; action = prediction["action"]
                    action_vec = np.concatenate([
                        np.asarray(action["world_vector"], dtype=np.float32).reshape(-1),
                        np.asarray(action["rot_axangle"], dtype=np.float32).reshape(-1),
                        np.asarray(action["gripper"], dtype=np.float32).reshape(-1)])
                    obs, _reward, terminated, truncated, info = env.step(action_vec)
                    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
                    image = np.asarray(get_image_from_maniskill2_obs_dict(env.unwrapped, obs), dtype=np.uint8)
                    success = bool(info.get("success", terminated))
                    meta = prediction.get("meta", {})
                    trace.append({"step": step + 1, "normalized_phase": step / max(1, MAX_STEPS - 1),
                                  "action": action, "success": success,
                                  "guidance_trace": {k: v for k, v in meta.items()
                                      if k.startswith(("clean_", "guided_", "preagg_", "postagg_",
                                                       "grip_threshold", "action_hidden_", "lambda_h"))}})
                    if step == 0 or (step + 1) % 10 == 0 or terminated or truncated:
                        print(json.dumps({"worker": args.worker_id, "task": args.task, "seed": seed,
                                          "arm": arm, "step": step + 1, "success": success}), flush=True)
                    if terminated or truncated: break
                row = {
                    "protocol_id": "UNIVLA_SAPIEN2_RT_DEPTH0_HIDDEN_CD_PAIRED_V1", "complete": True,
                    "worker_id": args.worker_id, "task": args.task, "environment_id": TASK_ENVS[args.task],
                    "seed": seed, "episode_id": snapshot["episode_id"], "arm": arm,
                    "checkpoint": str(Path("/home/leju-suzhou/zjt_ws/checkpoints/univla-7b-224-sft-simpler-bridge")),
                    "canonical_snapshot_sha256": canonical_sha, "instruction": instruction,
                    "renderer": {"shader_dir": "rt", "device": "cuda:0", "spp": 32,
                                 "path_depth": 0, "denoiser": True},
                    "initial_state_sha256": state_sha, "initial_rgb_sha256": rgb_sha,
                    "rtfull_initial_state_match": True,
                    "rtfull_initial_rgb_match": rgb_sha == ref.get("initial_rgb_sha256"),
                    "source_rtfull_artifact": str(FULL_PILOT),
                    "success": bool(success), "terminated": bool(terminated), "truncated": bool(truncated),
                    "steps": len(trace), "runtime_seconds": round(time.monotonic() - start, 4),
                    "final_info": info, "step_log": trace,
                }
                atomic_json(paths[arm], row)
                pair_rows[arm] = row
                print(json.dumps({"worker": args.worker_id, "task": args.task, "seed": seed,
                                  "arm": arm, "success": row["success"], "steps": row["steps"],
                                  "seconds": row["runtime_seconds"]}), flush=True)

            if pair_rows[ARMS[0]]["initial_state_sha256"] != pair_rows[ARMS[1]]["initial_state_sha256"] or \
                    pair_rows[ARMS[0]]["initial_rgb_sha256"] != pair_rows[ARMS[1]]["initial_rgb_sha256"]:
                raise RuntimeError(f"Vanilla/Hidden-CD depth0 pair is not exact at {args.task}:{seed}")
            atomic_json(pair_path, {
                "task": args.task, "seed": seed, "arms": list(ARMS), "exact_initial_pair": True,
                "rtfull_initial_state_match": True, "rtfull_rgb_expected_to_differ": True,
                "source_rtfull_artifact": str(FULL_PILOT),
                "vanilla_success": pair_rows["vanilla"]["success"],
                "hidden_cd_05_success": pair_rows["hidden_cd_05"]["success"],
                "initial_state_sha256": state_sha, "initial_rgb_sha256": first_rgb,
                "canonical_snapshot_sha256": canonical_sha,
            })
            completed.append(seed)
    finally:
        try:
            send_message(conn, {"cmd": "quit"}); recv_message(conn)
        except Exception: pass
        conn.close(); env.close()
    atomic_json(artifact / "worker_manifests" / f"{args.worker_id}.json", {
        "worker_id": args.worker_id, "task": args.task, "completed_seeds": completed,
        "completed_pairs": len(completed), "renderer_depth": 0})
    print(json.dumps({"worker": args.worker_id, "worker_complete": len(completed) == len(seeds),
                      "completed_pairs": len(completed)}), flush=True)


if __name__ == "__main__":
    main()
