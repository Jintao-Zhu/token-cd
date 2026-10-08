"""Three-arm paired SAPIEN2 RT-depth0 rollout for the 300-seed UniVLA extension."""
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

import gymnasium as gym
import numpy as np
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

TASK_ENVS = {
    "widowx_spoon_on_towel": "PutSpoonOnTableClothInScene-v0",
    "widowx_carrot_on_plate": "PutCarrotOnPlateInScene-v0",
    "widowx_stack_cube": "StackGreenCubeOnYellowCubeBakedTexInScene-v0",
    "widowx_put_eggplant_in_basket": "PutEggplantInBasketScene-v0",
}
ARMS = ("vanilla", "l14_matched", "hidden_cd_010", "hidden_cd_015", "hidden_cd_025", "hidden_cd_035", "hidden_cd_05")
MAX_STEPS = 80
HISTORICAL_ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd/artifacts/univla_simpler_l14_matched_rtfull_100_v1/episodes")
PROTOCOL_ID = os.environ.get("UNIVLA_PROTOCOL_ID", "UNIVLA_BRIDGE_HIDDEN_CD_RT_DEPTH0_3ARM_300_V1")


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


def atomic_pickle(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
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
    ap.add_argument("--arms", default=",".join(ARMS),
                    help="Comma-separated arms to run; defaults to the original four-arm pilot.")
    ap.add_argument("--reference-pair-artifact", type=Path, default=None,
                    help="Wait for an existing result for each seed and record its initial-state/RGB comparison.")
    ap.add_argument("--allow-seed-only-reference", action="store_true",
                    help="Allow initial-state/RGB differences from the reference; task and seed remain aligned.")
    ap.add_argument("--no-historical-controls", action="store_true",
                    help="For new seed ranges, pair requested arms against each other using the same saved snapshot.")
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)
    arms = tuple(x.strip() for x in args.arms.split(",") if x.strip())
    if not arms or len(set(arms)) != len(arms) or any(x not in ARMS for x in arms):
        raise ValueError(f"invalid --arms={args.arms!r}; valid arms are {ARMS}")
    args.artifact.mkdir(parents=True, exist_ok=True)
    config_lock = {
        "protocol_id": PROTOCOL_ID,
        "checkpoint": "/home/leju-suzhou/zjt_ws/checkpoints/univla-7b-224-sft-simpler-bridge",
        "task": args.task,
        "seeds": seeds,
        "arms": list(arms),
        "renderer": {"shader_dir": "rt", "spp": 32, "path_depth": 0, "denoiser": True},
        "selector": {"attention_layer_zero_based": 14, "budget": "dynamic L14-Matched", "negative": "4-neighbor harmonic"},
        "guidance": {
            "l14_matched": "1.5*positive_logits - 0.5*negative_logits",
            "hidden_cd_010": 0.10,
            "hidden_cd_015": 0.15,
            "hidden_cd_025": 0.25,
            "hidden_cd_035": 0.35,
            "hidden_cd_05": 0.5,
        },
        "pairing": "all requested arms restore the same persisted per-seed simulator snapshot; seed IDs align with existing task records; historical baseline hashes are diagnostic only",
    }
    lock_path = args.artifact / "locks" / f"{args.task}_{min(seeds):03d}_{max(seeds):03d}.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != config_lock:
        raise RuntimeError(f"configuration lock mismatch: {lock_path}")
    atomic_json(lock_path, config_lock)
    task_root = args.artifact / "episodes" / args.task
    task_root.mkdir(parents=True, exist_ok=True)
    snapshot_parent = Path(os.environ.get("UNIVLA_SNAPSHOT_ROOT", args.artifact / "snapshots"))
    snapshot_root = snapshot_parent / args.task
    snapshot_root.mkdir(parents=True, exist_ok=True)
    # Use the task's current SimpleEnv registry entry. Hard-coded v0 IDs are
    # deprecated and can silently bypass task-specific scene kwargs.
    env_id, task_kwargs = simpler_env.ENVIRONMENT_MAP[args.task]
    # Some checked-in SimpleEnv maps still name v0 while only v1 is registered.
    # Resolve that stale alias before gym.make rather than invoking Gym's
    # deprecated-version failure path.
    if env_id not in gym.envs.registry and env_id.endswith("-v0"):
        candidate = env_id[:-3] + "-v1"
        if candidate in gym.envs.registry:
            env_id = candidate
    env_kwargs = dict(task_kwargs)
    env_kwargs.update({
        "obs_mode": "rgbd",
        "prepackaged_config": True,
        "shader_dir": "rt",
        "render_config": {"rt_samples_per_pixel": 32, "rt_max_path_depth": 0,
                          "rt_use_denoiser": True},
        "renderer_kwargs": {"device": "cuda:0", "offscreen_only": True},
    })
    env = gym.make(env_id, **env_kwargs)
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
            paths = {arm: task_root / arm / f"seed_{seed:03d}.json" for arm in arms}
            baseline_paths = {
                arm: HISTORICAL_ROOT / args.task / arm / f"seed_{seed:03d}.json"
                for arm in ("vanilla", "l14_matched")
            }
            baseline = None
            if not args.no_historical_controls:
                if not all(path.exists() for path in baseline_paths.values()):
                    raise FileNotFoundError(f"historical paired controls missing for {args.task}/{seed}: {baseline_paths}")
                baseline = {arm: json.loads(path.read_text()) for arm, path in baseline_paths.items()}
                if not all(row.get("complete") and row.get("renderer", {}).get("shader_dir") == "rt"
                           for row in baseline.values()):
                    raise RuntimeError(f"historical controls are incomplete or not RT for {args.task}/{seed}")
                if (baseline["vanilla"].get("initial_state_sha256") != baseline["l14_matched"].get("initial_state_sha256") or
                        baseline["vanilla"].get("initial_rgb_sha256") != baseline["l14_matched"].get("initial_rgb_sha256")):
                    raise RuntimeError(f"historical controls are not internally paired for {args.task}/{seed}")
            if all(path.exists() for path in paths.values()):
                existing = {arm: json.loads(path.read_text()) for arm, path in paths.items()}
                state_hashes = {row.get("initial_state_sha256") for row in existing.values()}
                rgb_hashes = {row.get("initial_rgb_sha256") for row in existing.values()}
                if (all(row.get("complete") and row.get("protocol_id") == PROTOCOL_ID for row in existing.values())
                    and len(state_hashes) == 1 and len(rgb_hashes) == 1):
                    completed.append(seed)
                    print(json.dumps({"worker": args.worker_id, "skip_complete_paired_arms": True, "task": args.task, "seed": seed}), flush=True)
                    continue

            expected_pair = None
            if args.reference_pair_artifact is not None:
                pair_path = args.reference_pair_artifact / "pairs" / args.task / f"seed_{seed:03d}.json"
                pair_deadline = time.monotonic() + 12 * 60 * 60
                while not pair_path.exists():
                    if time.monotonic() >= pair_deadline:
                        raise TimeoutError(f"reference pair not ready after 12h for {args.task}/{seed}: {pair_path}")
                    print(json.dumps({"worker": args.worker_id, "waiting_for_reference_pair": True,
                                      "task": args.task, "seed": seed}), flush=True)
                    time.sleep(10)
                expected_pair = json.loads(pair_path.read_text())
                if expected_pair.get("exact_initial_pair") is not True and not args.allow_seed_only_reference:
                    raise RuntimeError(f"reference pair is not exact for {args.task}/{seed}: {pair_path}")

            snapshot_path = snapshot_root / f"seed_{seed:03d}.pkl"
            if snapshot_path.exists():
                with snapshot_path.open("rb") as stream:
                    snapshot = pickle.load(stream)
            else:
                snapshot = capture_snapshot(env, seed)
                atomic_pickle(snapshot_path, snapshot)
            canonical_sha = snapshot_hash(snapshot)
            pair_state = {}
            reference_state_sha = None
            reference_rgb_sha = None
            for arm in arms:
                obs, image, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
                if reference_state_sha is None:
                    reference_state_sha, reference_rgb_sha = state_sha, rgb_sha
                elif state_sha != reference_state_sha or rgb_sha != reference_rgb_sha:
                    raise RuntimeError(
                        f"same-snapshot initial observation changed across arms for {args.task}/{seed}: "
                        f"state_match={state_sha == reference_state_sha}, RGB_match={rgb_sha == reference_rgb_sha}"
                    )
                instruction = env.unwrapped.get_language_instruction()
                if expected_pair is not None and not args.allow_seed_only_reference and (
                    state_sha != expected_pair.get("initial_state_sha256")
                    or rgb_sha != expected_pair.get("initial_rgb_sha256")
                    or instruction != expected_pair.get("instruction")
                ):
                    raise RuntimeError(
                        f"new arm initial observation does not match the existing sweep for {args.task}/{seed}; "
                        f"state={state_sha == expected_pair.get('initial_state_sha256')}, "
                        f"rgb={rgb_sha == expected_pair.get('initial_rgb_sha256')}, "
                        f"instruction={instruction == expected_pair.get('instruction')}"
                    )
                pair_state[arm] = {
                    "state_sha256": state_sha,
                    "rgb_sha256": rgb_sha,
                    "episode_id": snapshot["episode_id"],
                    "instruction": instruction,
                }
                send_message(conn, {"cmd": "reset", "instruction": instruction, "seed": seed})
                ack = recv_message(conn)
                if not ack.get("ok"):
                    raise RuntimeError(f"model reset failed: {ack}")
                started = time.monotonic()
                success = False
                terminated = truncated = False
                step_rows = []
                info = {}
                for step in range(MAX_STEPS):
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
                        "normalized_phase": float(step / max(1, MAX_STEPS - 1)),
                        "action_token_ids": meta.get("action_token_ids"),
                        "m_t": meta.get("m_t"),
                        "selected_token_ids": meta.get("selected_token_ids"),
                        "attention_mass": meta.get("attention_mass"),
                        "action": action,
                        "guidance_trace": {k: v for k, v in meta.items()
                                           if k.startswith(("clean_", "guided_", "preagg_", "postagg_",
                                                            "grip_threshold", "action_hidden_", "lambda_h"))},
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
                    "protocol_id": PROTOCOL_ID,
                    "complete": True,
                    "worker_id": args.worker_id,
                    "task": args.task,
                    "environment_id": env_id,
                    "seed": seed,
                    "episode_id": pair_state[arm]["episode_id"],
                    "arm": arm,
                    "canonical_snapshot_sha256": canonical_sha,
                    "same_seed_as_reference": expected_pair is not None,
                    "reference_initial_hashes_match": (
                        None if expected_pair is None else (
                            state_sha == expected_pair.get("initial_state_sha256")
                            and rgb_sha == expected_pair.get("initial_rgb_sha256")
                            and instruction == expected_pair.get("instruction")
                        )
                    ),
                    "instruction": instruction,
                    "renderer": {"shader_dir": "rt", "device": "cuda:0", "spp": 32, "path_depth": 0, "denoiser": True, "offscreen_only": True},
                    "initial_state_sha256": state_sha,
                    "initial_rgb_sha256": rgb_sha,
                    "historical_baseline_hash_match": ({
                        "vanilla": state_sha == baseline["vanilla"].get("initial_state_sha256") and rgb_sha == baseline["vanilla"].get("initial_rgb_sha256"),
                        "l14_matched": state_sha == baseline["l14_matched"].get("initial_state_sha256") and rgb_sha == baseline["l14_matched"].get("initial_rgb_sha256"),
                    } if baseline is not None else None),
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

            instructions = {value["instruction"] for value in pair_state.values()}
            state_hashes = {value["state_sha256"] for value in pair_state.values()}
            rgb_hashes = {value["rgb_sha256"] for value in pair_state.values()}
            if len(instructions) != 1 or len(state_hashes) != 1 or len(rgb_hashes) != 1:
                raise RuntimeError(f"four-arm pairing failed for {args.task}/{seed}: {pair_state}")
            pair = {
                "task": args.task, "seed": seed,
                **{f"{arm}_success": outputs[arm]["success"] for arm in arms},
                "exact_initial_pair": len(arms) > 1,
                "same_seed_as_reference": expected_pair is not None,
                "matches_reference_pair": (None if expected_pair is None else (
                    next(iter(state_hashes)) == expected_pair.get("initial_state_sha256")
                    and next(iter(rgb_hashes)) == expected_pair.get("initial_rgb_sha256")
                    and next(iter(instructions)) == expected_pair.get("instruction")
                )),
                "initial_state_sha256": next(iter(state_hashes)),
                "initial_rgb_sha256": next(iter(rgb_hashes)),
                "episode_id": next(iter({value["episode_id"] for value in pair_state.values()})),
                "canonical_snapshot_sha256": canonical_sha,
                "arms": list(arms),
                "historical_baseline_hash_match": ({
                    "vanilla": outputs["vanilla"]["historical_baseline_hash_match"]["vanilla"],
                    "l14_matched": outputs["l14_matched"]["historical_baseline_hash_match"]["l14_matched"],
                } if baseline is not None else None),
            }
            atomic_json(args.artifact / "pairs" / args.task / f"seed_{seed:03d}.json", pair)
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
        "protocol_id": PROTOCOL_ID,
        "worker_id": args.worker_id,
        "task": args.task,
        "completed_seeds": completed,
        "completed_pairs": len(completed),
        "arms": list(arms),
        "all_requested_arm_initial_states_checked": True,
    }
    atomic_json(args.artifact / "worker_manifests" / f"{args.worker_id}.json", worker_manifest)
    print(json.dumps({"worker": args.worker_id, "task": args.task, "completed_seeds": completed,
                      "worker_complete": len(completed) == len(seeds), "arms": list(arms)}), flush=True)


if __name__ == "__main__":
    main()
