"""Paired UniVLA Fractal-30k Hidden-CD rollout on SIMPLER Google Robot."""
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
ARMS = tuple(x.strip() for x in os.environ.get("UNIVLA_ARMS", "hidden_cd_025,hidden_cd_05").split(",") if x.strip())
if not ARMS or not set(ARMS).issubset({"vanilla", "hidden_cd_010", "hidden_cd_015", "hidden_cd_025", "hidden_cd_035", "hidden_cd_05"}):
    raise ValueError(f"UNIVLA_ARMS must contain vanilla and/or Hidden-CD arms, got {ARMS}")
CHECKPOINT = os.environ.get("UNIVLA_CHECKPOINT", "/home/leju-suzhou/zjt_ws/checkpoints/univla-fractal-google-runs/univla-7b-224-sft-simpler-bridge+fractal20220817_data+b12+lr-3.5e-05+lora-r16+dropout-0.0--fractal_google_resume3k_to30k_20261002=w-LowLevelDecoder-ws-10--27000_chkpt")
CHECKPOINT_DOMAIN = os.environ.get("UNIVLA_CHECKPOINT_DOMAIN", "fractal_google_ft_30k")
ACTION_STATS_KEY = os.environ.get("UNIVLA_UNNORM_KEY", "fractal20220817_data")
PROTOCOL = os.environ.get("UNIVLA_PROTOCOL_ID", "UNIVLA_FRACTAL_GOOGLE_FT30K_HIDDEN_CD_RT_FULL_0_99_V1")


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
    ap.add_argument("--reference-pair-artifact", type=Path, default=None,
                    help="Wait for an existing result for each seed and record its initial-state/RGB comparison.")
    ap.add_argument("--allow-seed-only-reference", action="store_true",
                    help="Allow initial-state/RGB differences from the reference; task and seed remain aligned.")
    ap.add_argument("--render-depth", type=int, choices=(0, 8), default=8)
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)
    renderer_lock = {"shader_dir": "rt", "offscreen_only": True, "spp": 32, "path_depth": args.render_depth}
    instruction_schedule = {
        "google_robot_place_apple_in_closed_top_drawer": {
            "initial": "open top drawer",
            "after_half_time_limit": "place apple into top drawer",
            "terminate_token_supported_by_adapter": False,
        }
    }.get(args.task, {"initial": "environment_language_instruction", "after_half_time_limit": None})
    pairing_lock = (
        "All arms restore the same per-seed initial snapshot and follow the same SIMPLER language schedule; "
        "the Apple drawer task switches from open-top-drawer to place-apple instruction after half the time limit"
    )
    if args.render_depth == 0:
        renderer_lock["denoiser"] = True
        pairing_lock = (
            "All arms restore the same per-seed snapshot and share the same initial state, depth0 RGB, "
            "and task instruction schedule"
        )
    config_lock = {
        "protocol_id": PROTOCOL,
        "checkpoint": CHECKPOINT,
        "checkpoint_domain": CHECKPOINT_DOMAIN,
        "action_stats_key": ACTION_STATS_KEY,
        "policy_setup": "google_robot",
        "gripper_adapter": "absolute_binary_delta_v1",
        "task": args.task,
        "seeds": seeds,
        "arms": list(ARMS),
        "instruction_schedule": instruction_schedule,
        "selector": {"attention_layer_zero_based": 14, "budget": "dynamic L14-Matched", "negative": "4-neighbor harmonic"},
        "guidance": {
            "space": "last_hidden_action_states",
            "hidden_alignment": os.environ.get("UNIVLA_HIDDEN_ALIGNMENT", "act1234"),
            "hidden_cd_010": 0.10,
            "hidden_cd_015": 0.15,
            "hidden_cd_025": 0.25,
            "hidden_cd_035": 0.35,
            "hidden_cd_05": 0.5,
        },
        "renderer": renderer_lock,
        "pairing": pairing_lock,
    }
    lock_path = args.artifact / "locks" / f"{args.task}_{min(seeds):03d}_{max(seeds):03d}.json"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists() and json.loads(lock_path.read_text()) != config_lock:
        old_lock = json.loads(lock_path.read_text())
        changed = {key: {"locked": old_lock.get(key), "requested": config_lock.get(key)}
                   for key in sorted(set(old_lock) | set(config_lock))
                   if old_lock.get(key) != config_lock.get(key)}
        raise RuntimeError(f"configuration lock mismatch: {lock_path}; changed={json.dumps(changed, sort_keys=True)}")
    atomic_json(lock_path, config_lock)
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
    # Preserve the historical RT-full initialization path exactly: its
    # prepackaged config already selects depth 8. Only override the renderer
    # when requesting the depth0 arm.
    if args.render_depth == 0:
        env_kwargs["render_config"] = {
            "rt_samples_per_pixel": 32,
            "rt_max_path_depth": 0,
            "rt_use_denoiser": True,
        }
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
                initial_instruction = env.unwrapped.get_language_instruction()
                instruction = initial_instruction
                pair_state[arm] = {
                    "state_sha256": state_sha,
                    "rgb_sha256": rgb_sha,
                    "episode_id": snapshot["episode_id"],
                    "instruction": initial_instruction,
                }
                if expected_pair is not None and not args.allow_seed_only_reference and (
                    state_sha != expected_pair.get("initial_state_sha256")
                    or rgb_sha != expected_pair.get("initial_rgb_sha256")
                    or initial_instruction != expected_pair.get("instruction")
                ):
                    raise RuntimeError(
                        f"new arm initial observation does not match the existing sweep for {args.task}/{seed}; "
                        f"state={state_sha == expected_pair.get('initial_state_sha256')}, "
                        f"rgb={rgb_sha == expected_pair.get('initial_rgb_sha256')}, "
                        f"instruction={initial_instruction == expected_pair.get('instruction')}"
                    )
                reference_state_match = None
                reference_rgb_match = None
                reference_instruction_match = None
                if args.reference_artifact is not None:
                    ref_path = args.reference_artifact / "episodes" / args.task / "vanilla" / f"seed_{seed:03d}.json"
                    if ref_path.exists():
                        reference = json.loads(ref_path.read_text())
                        reference_instruction_match = initial_instruction == reference.get("instruction")
                        reference_state_match = state_sha == reference.get("initial_state_sha256")
                        reference_rgb_match = rgb_sha == reference.get("initial_rgb_sha256")
                        reference_available = True
                        if (not reference_state_match or not reference_instruction_match
                                or (args.render_depth == 8 and not reference_rgb_match)):
                            print(json.dumps({
                                "worker": args.worker_id, "task": args.task, "seed": seed,
                                "historical_reference_available": True,
                                "historical_state_match": reference_state_match,
                                "historical_instruction_match": reference_instruction_match,
                                "historical_rgb_match": reference_rgb_match,
                                "note": "retain the internally paired Hidden-CD result; exclude this row from historical Vanilla/cross-renderer paired comparisons unless the required hashes and instruction match",
                            }), flush=True)
                    else:
                        reference_available = False
                        print(json.dumps({
                            "worker": args.worker_id, "task": args.task, "seed": seed,
                            "reference_available": False,
                            "note": "no existing Vanilla reference for this seed yet; current Hidden-CD arms remain paired to their shared saved snapshot",
                        }), flush=True)
                else:
                    reference_available = False
                # Fractal-trained UniVLA predicts absolute gripper openness;
                # its SIMPLER Google adapter needs the current physical state
                # to convert that target into a relative open/close command.
                gripper_open = float(np.clip(1.0 - env.unwrapped.agent.gripper_closedness, 0.0, 1.0))
                send_message(conn, {"cmd": "reset", "instruction": initial_instruction, "seed": seed,
                                    "gripper_open": gripper_open})
                ack = recv_message(conn)
                if not ack.get("ok"):
                    raise RuntimeError(f"model reset failed: {ack}")
                started = time.monotonic()
                success = False
                terminated = truncated = False
                step_rows = []
                instruction_transitions = []
                info = {}
                for step in range(max_steps):
                    # SIMPLER's long-horizon Apple drawer task is explicitly two-stage:
                    # ask the policy to open the drawer first, then ask it to place
                    # the apple after a terminate action or after half the time limit.
                    if (args.task == "google_robot_place_apple_in_closed_top_drawer"
                            and step >= max_steps // 2 and instruction != "place apple into top drawer"):
                        instruction = "place apple into top drawer"
                        instruction_transitions.append({"step": step + 1, "instruction": instruction,
                                                        "reason": "half_time_limit"})
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
                        "instruction": instruction,
                        "hidden_alignment": meta.get("hidden_alignment"),
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
                            "step": step + 1, "success": success, "instruction": instruction,
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
                    "hidden_alignment": os.environ.get("UNIVLA_HIDDEN_ALIGNMENT", "act1234"),
                    "checkpoint": CHECKPOINT,
                    "checkpoint_domain": CHECKPOINT_DOMAIN,
                    "evaluation_domain": "simpler_google_robot",
                    "policy_setup": "google_robot",
                    "action_stats_key": ACTION_STATS_KEY,
                    "historical_reference_available": reference_available,
                    "historical_reference_initial_state_match": reference_state_match,
                    "historical_reference_initial_rgb_exact_match": reference_rgb_match,
                    "historical_reference_instruction_match": reference_instruction_match,
                    "same_seed_as_reference": expected_pair is not None,
                    "reference_initial_hashes_match": (
                        None if expected_pair is None else (
                            state_sha == expected_pair.get("initial_state_sha256")
                            and rgb_sha == expected_pair.get("initial_rgb_sha256")
                            and initial_instruction == expected_pair.get("instruction")
                        )
                    ),
                    "canonical_snapshot_sha256": canonical_sha,
                    "instruction": initial_instruction,
                    "instruction_schedule": instruction_schedule,
                    "instruction_transitions": instruction_transitions,
                    "renderer": (renderer_lock | {"device": "cuda:0"}),
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

            if (len({x["instruction"] for x in pair_state.values()}) != 1
                    or len({x["state_sha256"] for x in pair_state.values()}) != 1
                    or len({x["rgb_sha256"] for x in pair_state.values()}) != 1):
                raise RuntimeError(f"Hidden-CD arms are not exactly paired for {args.task}/{seed}: {pair_state}")
            reference_hash_match = None if expected_pair is None else (
                next(iter({x["state_sha256"] for x in pair_state.values()})) == expected_pair.get("initial_state_sha256")
                and next(iter({x["rgb_sha256"] for x in pair_state.values()})) == expected_pair.get("initial_rgb_sha256")
                and next(iter({x["instruction"] for x in pair_state.values()})) == expected_pair.get("instruction")
            )
            atomic_json(args.artifact / "pairs" / args.task / f"seed_{seed:03d}.json", {
                "task": args.task, "seed": seed, "arms": list(ARMS), "exact_initial_pair": len(ARMS) > 1,
                "same_seed_as_reference": expected_pair is not None,
                "matches_reference_pair": reference_hash_match,
                "initial_state_sha256": next(iter({x["state_sha256"] for x in pair_state.values()})),
                "initial_rgb_sha256": next(iter({x["rgb_sha256"] for x in pair_state.values()})),
                "instruction": next(iter({x["instruction"] for x in pair_state.values()})),
                "hidden_cd_025_success": outputs.get("hidden_cd_025", {}).get("success"),
                "hidden_cd_035_success": outputs.get("hidden_cd_035", {}).get("success"),
                "hidden_cd_05_success": outputs.get("hidden_cd_05", {}).get("success"),
                "hidden_cd_010_success": outputs.get("hidden_cd_010", {}).get("success"),
                "hidden_cd_015_success": outputs.get("hidden_cd_015", {}).get("success"),
            })
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
