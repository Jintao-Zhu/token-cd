"""Paired SIMPLER rollouts for pi0-FAST Vanilla and L6-Matched."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(os.environ.get("PI0FAST_WORKSPACE_ROOT", Path.home() / "zjt_ws")).expanduser()
TOKEN_CD = Path(os.environ.get("PI0FAST_REPO_ROOT", Path(__file__).resolve().parents[2])).expanduser()
SIMPLER = Path(os.environ.get("SIMPLER_ENV_ROOT", ROOT / "SimplerEnv")).expanduser()
SPATIAL_SIMPLER = Path(os.environ.get("SPATIALVLA_SIMPLER_ROOT", ROOT / "SpatialVLA-SimplerEnv")).expanduser()
PCD_SOURCE = Path(os.environ.get(
    "PCD_SIMPLER_SOURCE", ROOT / "pcd_openvla_simpler_box_31b027e/source"
)).expanduser()
SPATIAL_MANISKILL = Path(os.environ.get(
    "SPATIALVLA_MANISKILL_ROOT", ROOT / "SpatialVLA/ManiSkill2_real2sim"
)).expanduser()
DEFAULT_OUT = TOKEN_CD / "runs/pi0fast_l6_matched_9task_100seed"
# Logged in every episode row; set from --checkpoint in main() so WidowX (bridge) runs are not
# mislabelled as fractal.
_CHECKPOINT_FOR_LOG = str(Path(os.environ.get(
    "PI0FAST_CHECKPOINT", ROOT / "checkpoints/openpi_fast_finetuned/fractal"
)).expanduser())
# The Fractal checkpoint is the SIMPLER Google Robot fine-tune. The Bridge checkpoint can
# also be evaluated on SIMPLER WidowX through the Bridge state/action adapter, but it is not a
# WidowX-SIMPLER fine-tune; those runs measure transfer from BridgeData V2.
TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
    "google_robot_place_apple_in_closed_top_drawer",
)
# Five arms: official vanilla plus four guidance strengths. The lambda is carried in the arm
# name so a result file identifies its own configuration, and it is sent to the server with
# every request.
LAMBDAS = (0.25, 0.50, 0.75, 1.00, 1.25, 1.50)
ARMS = ("vanilla",) + tuple(f"l6_matched_{int(round(l * 100)):03d}" for l in LAMBDAS)


def arm_lambda(arm: str) -> float | None:
    """Guidance strength for an arm name; None for vanilla."""
    if arm == "vanilla":
        return None
    return int(arm.rsplit("_", 1)[1]) / 100.0


def _digest(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    return hashlib.sha256(value.tobytes()).hexdigest()


def _coarse_rgb_digest(array: np.ndarray) -> str:
    """Hash a low-resolution, quantized RGB view to ignore tiny RT sampling noise."""
    import cv2

    image = np.asarray(array)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    image = cv2.resize(image, (56, 56), interpolation=cv2.INTER_AREA)
    quantized = np.rint(image.astype(np.float32) / 8.0).astype(np.uint8)
    return _digest(quantized)


def _lowres_rgb(array: np.ndarray) -> np.ndarray:
    """Small RGB image used to measure reset equivalence despite RT pixel noise."""
    import cv2

    image = np.asarray(array)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return cv2.resize(image, (56, 56), interpolation=cv2.INTER_AREA).astype(np.uint8)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def _environment(task: str, rt_depth: int = 0):
    import gymnasium as gym
    import simpler_env

    render_config = {
        "rt_samples_per_pixel": 32,
        "rt_max_path_depth": int(rt_depth),
        "rt_use_denoiser": True,
    }
    renderer_kwargs = {"device": "cuda:0", "offscreen_only": True}
    if task == "google_robot_pick_coke_can":
        return gym.make(
            "GraspSingleOpenedCokeCanDistractorInScene-v0",
            obs_mode="rgbd",
            prepackaged_config=True,
            distractor_config="less",
            shader_dir="rt",
            render_config=render_config,
            renderer_kwargs=renderer_kwargs,
        )
    env_id, base_kwargs = simpler_env.ENVIRONMENT_MAP[task]
    kwargs = dict(base_kwargs)
    kwargs.update(
        obs_mode="rgbd",
        prepackaged_config=True,
        shader_dir="rt",
        render_config=render_config,
        renderer_kwargs=renderer_kwargs,
    )
    return gym.make(env_id, **kwargs)


def _eef_pos(agent_obs: dict, env) -> np.ndarray:
    if "eef_pos" in agent_obs:
        return np.asarray(agent_obs["eef_pos"], dtype=np.float32)
    # GoogleRobotStaticBase's public proprioception omits eef_pos. Rebuild the
    # same canonical Fractal/Simpler encoding used by OpenPI's SIMPLER adapter:
    # TCP pose in robot-base coordinates, wxyz quaternion, gripper openness.
    agent = env.unwrapped.agent
    pose = agent.robot.get_pose().inv() * env.unwrapped.tcp.get_pose()
    openness = 1.0 - float(agent.get_gripper_closedness())
    return np.concatenate((np.asarray(pose.p), np.asarray(pose.q), [openness])).astype(np.float32)


# --- WidowX (Bridge) input convention -----------------------------------------------------------------
# The pi0-FAST Bridge checkpoint was trained on BridgeV2 with state layout
# [x, y, z, roll, pitch, yaw, PAD(=0), gripper] -- its own norm_stats prove this: dim 6 has
# std = 0 with q01 = q99 = 0 (a constant pad), dims 0-5 match the published Bridge proprio
# quantiles exactly, and dim 7 matches the Bridge gripper quantiles.  The legacy path below
# instead feeds [pos, quat_xyzw, 1 - gripper], which is the Fractal/LIBERO layout: same 8 dims,
# wrong semantics.  Set PI0FAST_WIDOWX_LAYOUT=bridge to use the trained convention.
# Default is the checkpoint's TRAINED convention for WidowX: state =
# [x, y, z, roll, pitch, yaw, PAD(0), gripper] with R_topdown conjugation.
# The old Fractal/LIBERO layout ([pos, quat_xyzw, 1-gripper]) produced 0 interaction on 26/26
# WidowX episodes; it remains available via PI0FAST_WIDOWX_LAYOUT=legacy for reproducing that
# control. See docs/WIDOWX_PI0FAST_INPUT_CONVENTION_FINDING.md.
WIDOWX_LAYOUT = os.environ.get("PI0FAST_WIDOWX_LAYOUT", "bridge").lower()
# Action clipping is an independent factor from the state layout (2x2 ablation).  The default
# preserves the historical behaviour: clip whenever the trained bridge layout is used.
WIDOWX_CLIP = os.environ.get("PI0FAST_WIDOWX_CLIP", "1" if WIDOWX_LAYOUT == "bridge" else "0") == "1"
BRIDGE_CKPT = Path(os.environ.get(
    "PI0FAST_BRIDGE_CHECKPOINT", ROOT / "checkpoints/openpi_fast_finetuned/bridge"
)).expanduser()
BRIDGE_STATS = BRIDGE_CKPT / "assets/local/bridge_lerobot/norm_stats.json"
# Bridge demonstrations express the EEF orientation relative to this top-down frame
# (INTACT's BridgeSimplerAdapter / open-pi-zero bridge_state use exactly this matrix).
BRIDGE_DEFAULT_ROT = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
_ACTION_BOUNDS: tuple[np.ndarray, np.ndarray] | None = None


def _bridge_action_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Training-time q01/q99 of the Bridge actions (used to clip tail extrapolations)."""
    global _ACTION_BOUNDS
    if _ACTION_BOUNDS is None:
        src = json.loads(BRIDGE_STATS.read_text())["norm_stats"]["actions"]
        _ACTION_BOUNDS = (np.asarray(src["q01"], np.float32), np.asarray(src["q99"], np.float32))
    return _ACTION_BOUNDS


def _bridge_state(eef: np.ndarray) -> np.ndarray:
    from transforms3d.euler import mat2euler, quat2mat

    euler = np.asarray(mat2euler(quat2mat(np.asarray(eef[3:7], np.float64)) @ BRIDGE_DEFAULT_ROT.T),
                       dtype=np.float32)
    return np.concatenate((eef[:3], euler, np.zeros((1,), np.float32), [float(eef[7])])).astype(np.float32)



def _raw_policy_input(image: np.ndarray, instruction: str, agent_obs: dict, env) -> dict:
    eef = _eef_pos(agent_obs, env)
    is_widowx = "google_robot" not in env.unwrapped.robot_uid
    if is_widowx and WIDOWX_LAYOUT == "bridge":
        state = _bridge_state(eef)
    else:
        quat_xyzw = eef[3:7][[1, 2, 3, 0]]
        state = np.concatenate((eef[:3], quat_xyzw, [1.0 - float(eef[7])])).astype(np.float32)
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    import cv2

    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    return {
        "observation/primary_image": image,
        "observation/wrist_image": np.zeros_like(image),
        "observation/state": state,
        "prompt": instruction,
    }


class RolloutPolicy:
    def __init__(self, host: str, port: int):
        os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        for path in (PCD_SOURCE, SPATIAL_SIMPLER, SIMPLER, SPATIAL_MANISKILL, TOKEN_CD):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        # Match the official OpenPI serving path. Keep the environment's
        # synchronous server from timing out while JAX compiles the first request.
        from websockets.sync import client as websocket_client

        original_connect = websocket_client.connect

        def connect_without_keepalive(*args, **kwargs):
            kwargs.setdefault("ping_interval", None)
            kwargs.setdefault("ping_timeout", None)
            # The server's event loop is blocked while a synchronous inference runs, so with
            # several workers per server a new client can wait a long time for the handshake.
            # The default 10 s open_timeout is far too short for that.
            kwargs.setdefault("open_timeout", 600)
            kwargs.setdefault("close_timeout", 60)
            return original_connect(*args, **kwargs)

        websocket_client.connect = connect_without_keepalive
        from openpi_client import websocket_client_policy

        self.client = websocket_client_policy.WebsocketClientPolicy(host, port)
        self.exec_horizon = 4

    def chunk(self, arm: str, raw: dict, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
        request = {**raw, "cd_arm": "vanilla" if arm == "vanilla" else "l6_matched",
                   "cd_seed": int(seed)}
        lam = arm_lambda(arm)
        if lam is not None:
            request["cd_lambda"] = float(lam)
        result = self.client.infer(request)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if arm != "vanilla":
            trace = dict(result["pi0_fast_matched_cd_trace"])
            # A malformed FAST payload uses OpenPI's documented zero-DCT
            # fallback. Accept it only when the direct path numerically agrees
            # with OpenPI's official decoder output.
            if not trace.get("openpi_official_decoder_agrees"):
                # On WidowX this happens on ~2.6% of guided EPISODES (never observed on the
                # google_robot tasks).  Raising here killed the worker, lost the episode and
                # crashed again on retry, so instead this single replan falls back to the
                # unguided action and the fallback is recorded per replan (and counted in the
                # episode row as `guided_fallbacks`) so it can be reported or excluded later.
                fallback = self.client.infer({**raw, "cd_arm": "vanilla", "cd_seed": int(seed)})
                fb_actions = np.asarray(fallback["actions"], dtype=np.float32)
                if fb_actions.shape != (10, 7) or not np.isfinite(fb_actions).all():
                    raise RuntimeError(f"guided fallback produced invalid chunk {fb_actions.shape}")
                self.guided_fallbacks = getattr(self, "guided_fallbacks", 0) + 1
                return fb_actions, {
                    "guided_fallback": True,
                    "guided_trace_rejected": {
                        k: trace.get(k) for k in (
                            "lambda", "selected_token_count", "matched_m_t",
                            "fast_action_stream_valid", "openpi_official_decoder_agrees",
                            "serialized_output_preview")},
                    "fallback_server_meta": fallback.get("pi0_fast_server_meta", {}),
                }
            if trace.get("selector_layer_label") != "L6" or not trace.get("selected_count_equals_matched"):
                raise RuntimeError(f"L6-Matched invariant failed: {trace}")
            meta = result.get("pi0_fast_server_meta", {})
            if meta.get("lambda") != float(lam):
                raise RuntimeError(
                    f"server ran lambda={meta.get('lambda')} but arm {arm} requested {lam}"
                )
        if actions.shape != (10, 7) or not np.isfinite(actions).all():
            raise RuntimeError(f"invalid {arm} action chunk: {actions.shape}")
        trace = dict(result.get("pi0_fast_matched_cd_trace", {
            "official_openpi_policy_infer": True,
            "output_action_shape": list(actions.shape),
            "output_action_finite": bool(np.isfinite(actions).all()),
        }))
        return actions, trace


class ActionChunk:
    def __init__(self, policy_setup: str = "google_robot"):
        from collections import deque

        self.plan = deque()
        self.exec_horizon = 4
        self.previous_gripper_openness = None
        self.sticky_gripper_action = 0.0
        self.gripper_repeat_remaining = 0
        self.policy_setup = policy_setup

    def next(self, chunk: np.ndarray) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        raw = np.asarray(chunk, dtype=np.float32)
        if self.policy_setup == "widowx_bridge" and WIDOWX_CLIP:
            # INTACT's BridgeSimplerAdapter clips the first six dims to the training range
            # (the policy's tail extrapolations exceed p99 by 8-10x); clipping in normalised
            # space to [-1, 1] is the same as clipping the raw values to q01/q99.
            lo, hi = _bridge_action_bounds()
            raw = np.clip(raw, lo, hi)
        action = np.zeros((7,), dtype=np.float32)
        action[:3] = raw[:3]
        action[3:6] = Rotation.from_euler("xyz", raw[3:6]).as_rotvec().astype(np.float32)
        openness = float(raw[6])
        if self.policy_setup == "widowx_bridge":
            # SIMPLER's WidowX convention, taken from the SpatialVLA evaluation code that is
            # verified to drive these four tasks: the gripper is binarised, not emitted as a
            # sticky relative delta. The google_robot sticky rule computes
            # previous_openness - openness, which is ~0 whenever the policy holds a steady
            # openness value, so the gripper command stays 0 and the robot can never grasp
            # anything. That is consistent with moved_wrong_obj=False on 100% of WidowX
            # episodes while the arm itself still moved.
            action[6] = 2.0 * (1.0 if openness > 0.5 else 0.0) - 1.0
            return action
        if self.previous_gripper_openness is None:
            gripper_delta = 0.0
            self.previous_gripper_openness = openness
        else:
            gripper_delta = self.previous_gripper_openness - openness
            if abs(gripper_delta) > 0.5 and self.gripper_repeat_remaining == 0:
                self.sticky_gripper_action = gripper_delta
                self.gripper_repeat_remaining = 10
                self.previous_gripper_openness = openness
        if self.gripper_repeat_remaining > 0:
            gripper_delta = self.sticky_gripper_action
            self.gripper_repeat_remaining -= 1
        action[6] = np.clip(gripper_delta, -1.0, 1.0)
        return action


def _episode(policy: RolloutPolicy, env, task: str, seed: int, arm: str, out: Path, rt_depth: int = 0) -> dict[str, Any]:
    started = time.monotonic()
    fallbacks_before = getattr(policy, "guided_fallbacks", 0)
    obs, reset_info = env.reset(seed=seed)
    instruction = str(env.unwrapped.get_language_instruction())
    camera = "overhead_camera" if "google_robot" in env.unwrapped.robot_uid else "3rd_view_camera"
    policy_setup = "google_robot" if "google_robot" in env.unwrapped.robot_uid else "widowx_bridge"
    initial_image = np.asarray(obs["image"][camera]["rgb"])
    initial_agent = _eef_pos(obs["agent"], env)
    rgb_sha = _digest(initial_image)
    state_sha = _digest(initial_agent)
    chunks: list[dict[str, Any]] = []
    action_chunk = ActionChunk(policy_setup)
    success = False
    terminated = truncated = False
    last_info: dict[str, Any] = {}
    steps = 0
    max_steps = 250
    while not (terminated or truncated) and steps < max_steps:
        image = obs["image"][camera]["rgb"]
        # Staged tasks (e.g. PlaceIntoClosedTopDrawer) change their instruction when the
        # subtask advances: "open top drawer" -> "place apple into top drawer". Reading it once
        # at reset would feed stage-1 language to a stage-2 policy.
        try:
            instruction = str(env.unwrapped.get_language_instruction())
        except Exception:  # noqa: BLE001
            pass
        raw = _raw_policy_input(image, instruction, obs["agent"], env)
        if steps % action_chunk.exec_horizon == 0:
            replan_seed = seed * 1009 + steps // action_chunk.exec_horizon
            chunk, trace = policy.chunk(arm, raw, replan_seed)
            if chunk.shape != (10, 7) or not np.isfinite(chunk).all():
                raise RuntimeError(f"invalid action chunk {chunk.shape} for {task}/{seed}/{arm}")
            action_chunk.plan.extend(chunk[: action_chunk.exec_horizon])
            chunks.append({"step": steps, "trace": trace})
        raw_action = action_chunk.plan.popleft()
        action = action_chunk.next(raw_action)
        obs, reward, terminated, truncated, info = env.step(action)
        last_info = dict(info)
        success = bool(info.get("success", success))
        steps += 1
    if steps >= max_steps and not (terminated or truncated):
        truncated = True
    result = {
        "status": "complete",
        "task": task,
        "seed": int(seed),
        "arm": arm,
        "success": bool(success),
        "steps": steps,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "renderer": {"shader_dir": "rt", "rt_max_path_depth": int(rt_depth), "rt_samples_per_pixel": 32, "rt_use_denoiser": True},
        "guided_fallbacks": getattr(policy, "guided_fallbacks", 0) - fallbacks_before,
        "checkpoint": _CHECKPOINT_FOR_LOG,
        "instruction": instruction,
        "environment_id": getattr(getattr(env, "spec", None), "id", None),
        "initial_rgb_sha256": rgb_sha,
        "initial_rgb_coarse_sha256": _coarse_rgb_digest(initial_image),
        "initial_rgb_lowres_b64": base64.b64encode(_lowres_rgb(initial_image).tobytes()).decode("ascii"),
        "initial_agent_state_sha256": state_sha,
        "reset_info": reset_info,
        "final_info": last_info,
        "replan_count": len(chunks),
        "chunks": chunks,
        "runtime_seconds": round(time.monotonic() - started, 3),
    }
    path = out / "episodes" / task / f"{arm}_seed{seed:03d}.json"
    _atomic_json(path, result)
    print(json.dumps({k: result[k] for k in ("task", "seed", "arm", "success", "steps", "runtime_seconds", "initial_rgb_sha256")}, default=str), flush=True)
    return result


def build_manifest(out: Path, workers: int = 9, seed_start: int = 0, seed_end: int = 99) -> list[list[dict[str, Any]]]:
    """One worker per (task, seed-chunk).

    With only the five google_robot tasks, one worker per task would leave the policy servers
    starved, so each task's seed range is split across several workers. Every (task, seed)
    still appears exactly once, and each worker runs all five arms for the seeds it owns, so
    the five-arm pairing is preserved.
    """
    seeds = list(range(seed_start, seed_end + 1))
    chunks = int(os.environ.get("PI0FAST_SEED_CHUNKS", "3"))
    per = (len(seeds) + chunks - 1) // chunks
    shards: list[list[dict[str, Any]]] = []
    for task in TASKS:
        for c in range(chunks):
            part = seeds[c * per:(c + 1) * per]
            if not part:
                continue
            shards.append([{"task": task, "seed": s, "arms": list(ARMS)} for s in part])
    return shards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--worker-id", type=int)
    parser.add_argument("--shard", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument("--rt-depth", type=int, default=int(os.environ.get("PI0FAST_RT_DEPTH", "8")))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/openpi_fast_finetuned/fractal")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=99)
    parser.add_argument("--manifest-subdir", default="manifests")
    args = parser.parse_args()
    global _CHECKPOINT_FOR_LOG
    _CHECKPOINT_FOR_LOG = str(args.checkpoint)
    if args.manifest_only:
        shards = build_manifest(args.out, seed_start=args.seed_start, seed_end=args.seed_end)
        for i, jobs in enumerate(shards):
            _atomic_json(args.out / args.manifest_subdir / f"worker_{i}.json",
                         {"worker_id": i, "jobs": jobs})
        protocol = {
            "experiment": "pi0fast_simpler_9task_100seed_vanilla_vs_l6_matched_lambda_sweep",
            "total_episodes": len(TASKS) * 100 * len(ARMS),
            "tasks": list(TASKS),
            "arms": list(ARMS),
            "lambdas": list(LAMBDAS),
            "seeds": [0, 99],
            "workers": len(TASKS),
            "worker_task_map": {str(w): t for w, t in enumerate(TASKS)},
            "gpu_worker_map": {str(w): [1, 1, 1, 2, 2, 2, 3, 3, 3][w] for w in range(len(TASKS))},
            "episodes_per_worker": 100 * len(ARMS),
            "renderer": {"shader_dir": "rt", "rt_max_path_depth": 8, "rt_samples_per_pixel": 32,
                         "rt_use_denoiser": True},
            "checkpoint": str(args.checkpoint),
            "selector": ("L6 prompt-to-primary-view attention, zero-based layer index 6; "
                         "dynamic entity-matched budget (sklearn 1.7.2 KMeans K=8, n_init=10); "
                         "harmonic negative; lambdas 0.25/0.50/0.75/1.00"),
            "execution": {"tap_mode": "separate", "decode_mode": "verified",
                          "embed_jit": 0, "batch_branches": 0, "tap_truncation": "disabled"},
            "pairing": ("All five arms reset the same task seed; initial RGB and agent-state "
                        "hashes logged so the pairing can be verified"),
        }
        _atomic_json(args.out / "protocol.json", protocol)
        print(json.dumps({
            "output": str(args.out),
            "episodes": protocol["total_episodes"],
            "shards": [len(s) * len(ARMS) for s in shards],
        }))
        return

    args.out.mkdir(parents=True, exist_ok=True)
    shard_payload = json.loads(args.shard.read_text())
    jobs = shard_payload["jobs"]
    policy = RolloutPolicy(args.host, args.port)
    print(json.dumps({"event": "model_ready", "gpu": args.gpu, "worker": args.worker_id, "jobs": len(jobs)}), flush=True)
    current_task = None
    env = None
    try:
        for job in jobs:
            task, seed = job["task"], int(job["seed"])
            if task != current_task:
                if env is not None:
                    env.close()
                env = _environment(task, args.rt_depth)
                current_task = task
                print(json.dumps({"event": "environment_ready", "task": task, "gpu": args.gpu, "worker": args.worker_id}), flush=True)
            for arm in job["arms"]:
                output = args.out / "episodes" / task / f"{arm}_seed{seed:03d}.json"
                if output.exists():
                    try:
                        if json.loads(output.read_text()).get("status") == "complete":
                            continue
                    except Exception:
                        pass
                try:
                    _episode(policy, env, task, seed, arm, args.out, args.rt_depth)
                except BaseException as exc:
                    failure = {
                        "status": "error", "task": task, "seed": seed, "arm": arm,
                        "gpu": args.gpu, "worker_id": args.worker_id,
                        "error_type": type(exc).__name__, "error": repr(exc),
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    _atomic_json(output, failure)
                    raise
    finally:
        if env is not None:
            env.close()
    print(json.dumps({"event": "worker_complete", "gpu": args.gpu, "worker": args.worker_id}), flush=True)


if __name__ == "__main__":
    main()
