"""Canonical paired Pi0 vanilla/SHR rollout on five SIMPLER tasks."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from research.semantic_token_cd.pi0_shr_policy import Pi0SHRInference


REPO = Path("/home/leju-suzhou/zjt_ws/token-cd")
PI0_ROOT = Path("/home/leju-suzhou/zjt_ws/open-pi-zero")
CHECKPOINT_ROOT = Path("/home/leju-suzhou/zjt_ws/checkpoints/pi0")
FRACTAL_CHECKPOINT = CHECKPOINT_ROOT / "fractal_beta_step29576_2024-12-29_13-10_42.pt"
BRIDGE_CHECKPOINT = CHECKPOINT_ROOT / "bridge_beta_step19296_2024-12-26_22-30_42.pt"
PROTOCOL = "PI0_SHR_5TASK_0_299_V1"
TASKS = (
    "google_robot_close_drawer",
    "google_robot_open_drawer",
    "google_robot_move_near",
    "google_robot_pick_coke_can",
    "widowx_carrot_on_plate",
)
ARMS = ("pi0_vanilla", "pi0_shr_harmonic")
CONFIG_BY_TASK = {
    "google_robot_close_drawer": "fractal_drawer.yaml",
    "google_robot_open_drawer": "fractal_drawer.yaml",
    "google_robot_move_near": "fractal_move.yaml",
    "google_robot_pick_coke_can": "fractal_coke.yaml",
    "widowx_carrot_on_plate": "bridge.yaml",
}


def parse_seeds(spec: str) -> list[int]:
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    seeds = sorted(set(seeds))
    if not seeds or any(seed < 0 or seed > 299 for seed in seeds):
        raise ValueError("seeds must be a non-empty subset of 0..299")
    return seeds


def clone(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return type(value)((key, clone(item)) for key, item in value.items())
    if isinstance(value, list):
        return [clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone(item) for item in value)
    if hasattr(value, "p") and hasattr(value, "q"):
        return type(value)(np.asarray(value.p).copy(), np.asarray(value.q).copy())
    return copy.deepcopy(value)


def array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def wrapped_observation(env):
    observation = env.unwrapped.get_obs()
    wrappers, current = [], env
    while hasattr(current, "env"):
        if hasattr(current, "observation"):
            wrappers.append(current)
        current = current.env
    for wrapper in reversed(wrappers):
        observation = wrapper.observation(observation)
    return observation


def capture_snapshot(env, seed: int) -> dict[str, Any]:
    env.reset(seed=seed)
    inner = env.unwrapped
    return {
        "sim_state": np.asarray(inner.get_state()).copy(),
        "agent_state": clone(inner.agent.get_state()),
        "rng_state": clone(inner._episode_rng.get_state()),
        "instruction": inner.get_language_instruction(),
    }


def restore_snapshot(env, adapter, seed: int, snapshot: dict[str, Any]):
    env.reset(seed=seed)
    inner = env.unwrapped
    inner.set_state(snapshot["sim_state"].copy())
    inner.agent.set_state(clone(snapshot["agent_state"]))
    inner._episode_rng.set_state(clone(snapshot["rng_state"]))
    inner._elapsed_steps = 0
    adapter.reset()
    observation = wrapped_observation(env)
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    state_hash = array_sha256(np.asarray(inner.get_state()))
    rgb_hash = array_sha256(get_image_from_maniskill2_obs_dict(env, observation))
    return observation, state_hash, rgb_hash


def snapshot_sha(snapshot: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(array_sha256(snapshot["sim_state"]).encode())
    digest.update(repr(snapshot["agent_state"]).encode())
    digest.update(repr(snapshot["rng_state"]).encode())
    digest.update(snapshot["instruction"].encode())
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def load_model(task: str, device: torch.device):
    from src.model.vla.pizero import PiZeroInference

    config_path = PI0_ROOT / "config/eval" / CONFIG_BY_TASK[task]
    config = OmegaConf.load(config_path)
    config.env.adapter.dataset_statistics_path = str(
        PI0_ROOT / config.env.adapter.dataset_statistics_path
    )
    config.env.adapter.pretrained_model_path = str(CHECKPOINT_ROOT / "paligemma-3b-pt-224")
    config.use_torch_compile = False
    dtype = torch.float32
    model = PiZeroInference(config, use_ddp=False)
    checkpoint = BRIDGE_CHECKPOINT if task.startswith("widowx_") else FRACTAL_CHECKPOINT
    if not checkpoint.exists() or checkpoint.stat().st_size < 1_000_000_000:
        raise FileNotFoundError(f"missing or incomplete Pi0 checkpoint: {checkpoint}")
    data = torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)
    state = {key.replace("_orig_mod.", ""): value for key, value in data["model"].items()}
    model.load_state_dict(state, strict=True)
    del data, state
    model.freeze_all_weights()
    model.to(dtype=dtype, device=device).eval()
    adapter = hydra.utils.instantiate(config.env.adapter)
    adapter.reset()
    return config, model, adapter, checkpoint


def prepare_inputs(model, adapter, env, observation, instruction, device):
    inputs = adapter.preprocess(env, observation, instruction)
    dtype = next(model.parameters()).dtype
    causal, vlm_pos, proprio_pos, action_pos = model.build_causal_mask_and_position_ids(
        inputs["attention_mask"], dtype=dtype
    )
    condition_mask, action_mask = model.split_full_mask_into_submasks(causal)
    ready = {
        "input_ids": inputs["input_ids"],
        "pixel_values": inputs["pixel_values"].to(dtype),
        "image_text_proprio_mask": condition_mask,
        "action_mask": action_mask,
        "vlm_position_ids": vlm_pos,
        "proprio_position_ids": proprio_pos,
        "action_position_ids": action_pos,
        "proprios": inputs["proprios"].to(dtype),
    }
    return {key: value.to(device) for key, value in ready.items()}


def noise_seed(task: str, seed: int, replan: int) -> int:
    task_index = TASKS.index(task)
    return 2_026_090_300_000 + task_index * 10_000_000 + seed * 10_000 + replan


def write_trace(path: Path, records: list[dict[str, Any]], actions: np.ndarray) -> None:
    payload: dict[str, np.ndarray] = {"executed_actions": actions.astype(np.float32)}
    payload["initial_noise"] = np.stack([
        record["initial_noise"].numpy() for record in records
    ]).astype(np.float16)
    keys = ["positive_velocity", "guided_velocity"]
    if any(flow["negative_velocity"] is not None for record in records for flow in record["flow_steps"]):
        keys.append("negative_velocity")
    for key in keys:
        values = []
        for record in records:
            for flow in record["flow_steps"]:
                value = flow[key]
                values.append(value.numpy())
        payload[key] = np.stack(values).astype(np.float16)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def run_episode(env, adapter, policy, task, seed, arm, observation, instruction, device):
    truncated = False
    success = False
    environment_steps = 0
    replan = 0
    traces: list[dict[str, Any]] = []
    actions: list[np.ndarray] = []
    last_info: dict[str, Any] = {}
    while not truncated:
        inputs = prepare_inputs(policy.model, adapter, env, observation, instruction, device)
        predicted, trace = policy.infer(
            inputs, instruction, arm=arm, noise_seed=noise_seed(task, seed, replan)
        )
        traces.append(trace)
        env_actions = adapter.postprocess(predicted[0].float().cpu().numpy())
        for action in env_actions[: int(policy.model.cfg.act_steps)]:
            action = np.asarray(action, dtype=np.float64)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise FloatingPointError(f"invalid Pi0 action: {action}")
            observation, _reward, success, truncated, last_info = env.step(action)
            actions.append(action.copy())
            environment_steps += 1
            if truncated:
                break
        instruction = env.get_language_instruction()
        replan += 1
    return bool(success), environment_steps, traces, np.asarray(actions), jsonable(last_info)


def ensure_config(artifact: Path) -> None:
    config = {
        "protocol_id": PROTOCOL,
        "tasks": list(TASKS),
        "seeds": "0-299",
        "arms": list(ARMS),
        "pairing": "same simulator snapshot; deterministic shared initial action noise per replan",
        "model": "open-pi-zero Fractal-Beta for Google Robot and Bridge-Beta for WidowX",
        "positive_branch": "unmodified Pi0 flow velocity",
        "negative_branch": "KMeans K=8 semantic entity regions; beta=0 four-neighbor harmonic reconstruction",
        "guidance": "v_pos + 0.5*(v_pos-v_neg), first six dimensions at every flow step",
        "gripper": "positive velocity unchanged at every flow step",
        "flow_steps": 10,
        "torch_compile": False,
    }
    path = artifact / "CONFIG_LOCK.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError(f"CONFIG_LOCK differs: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--gpu", type=int, required=True, help="physical GPU id for provenance")
    parser.add_argument("--worker-id", default="manual")
    args = parser.parse_args()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(args.gpu):
        raise RuntimeError(
            f"launcher must set CUDA_VISIBLE_DEVICES={args.gpu}; found {visible!r}"
        )
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    device = torch.device("cuda:0")
    artifact = args.artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    ensure_config(artifact)

    import simpler_env

    env = simpler_env.make(args.task)
    model_config, model, adapter, checkpoint = load_model(args.task, device)
    policy = Pi0SHRInference(model, adapter.tokenizer, lambd=0.5)
    task_root = artifact / "episodes" / args.task
    snapshot_root = artifact / "snapshots" / args.task
    task_root.mkdir(parents=True, exist_ok=True)
    snapshot_root.mkdir(parents=True, exist_ok=True)

    for seed in parse_seeds(args.seeds):
        snapshot_path = snapshot_root / f"seed_{seed:03d}.pkl"
        if snapshot_path.exists():
            import pickle
            with snapshot_path.open("rb") as handle:
                snapshot = pickle.load(handle)
        else:
            import pickle
            snapshot = capture_snapshot(env, seed)
            temporary = snapshot_path.with_name(f".{snapshot_path.name}.{os.getpid()}.tmp")
            with temporary.open("wb") as handle:
                pickle.dump(snapshot, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary, snapshot_path)
        canonical = snapshot_sha(snapshot)
        arm_hashes = {}
        for arm in ARMS:
            arm_root = task_root / arm
            arm_root.mkdir(parents=True, exist_ok=True)
            summary_path = arm_root / f"episode_{seed:03d}_summary.json"
            arrays_path = arm_root / f"episode_{seed:03d}_arrays.npz"
            if summary_path.exists() and arrays_path.exists():
                summary = json.loads(summary_path.read_text())
                arm_hashes[arm] = (
                    summary["canonical_snapshot_sha256"],
                    summary["initial_state_sha256"],
                    summary["initial_rgb_sha256"],
                )
                continue
            observation, state_hash, rgb_hash = restore_snapshot(
                env, adapter, seed, snapshot
            )
            instruction = env.get_language_instruction()
            started = time.monotonic()
            success, steps, traces, actions, last_info = run_episode(
                env, adapter, policy, args.task, seed, arm, observation,
                instruction, device
            )
            runtime = time.monotonic() - started
            write_trace(arrays_path, traces, actions)
            technical_pass = bool(traces) and all(
                trace["shared_action_state_between_branches"]
                and trace["all_gripper_velocity_positive_unchanged"]
                and trace["reconstruction_finite"]
                for trace in traces
            )
            if not technical_pass:
                raise RuntimeError(f"Pi0-SHR audit failed for {args.task} seed={seed} arm={arm}")
            summary = {
                "protocol_id": PROTOCOL,
                "task": args.task,
                "seed": seed,
                "evaluation_seed": seed,
                "arm": arm,
                "success": success,
                "control_steps": steps,
                "replans": len(traces),
                "runtime_seconds": runtime,
                "gpu_id": args.gpu,
                "worker_id": args.worker_id,
                "checkpoint": str(checkpoint),
                "checkpoint_size": checkpoint.stat().st_size,
                "config": CONFIG_BY_TASK[args.task],
                "act_steps": int(model_config.act_steps),
                "lambda": 0.0 if arm == "pi0_vanilla" else 0.5,
                "canonical_snapshot_sha256": canonical,
                "initial_state_sha256": state_hash,
                "initial_rgb_sha256": rgb_hash,
                "arrays_file": arrays_path.name,
                "noise_seeds": [trace["noise_seed"] for trace in traces],
                "mean_selected_tokens": None if arm == "pi0_vanilla" else float(np.mean([
                    trace["selector"]["num_tokens"] for trace in traces
                ])),
                "mean_residual_norm": float(np.mean([
                    flow["residual_norm"] for trace in traces for flow in trace["flow_steps"]
                ])),
                "selector_trace": [trace["selector"] for trace in traces],
                "flow_residual_norms": [
                    [flow["residual_norm"] for flow in trace["flow_steps"]]
                    for trace in traces
                ],
                "all_gripper_velocity_positive_unchanged": True,
                "shared_action_state_between_branches": True,
                "technical_pass": True,
                "last_info": last_info,
            }
            temporary = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
            temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, summary_path)
            arm_hashes[arm] = (canonical, state_hash, rgb_hash)
            print(json.dumps({
                "task": args.task,
                "seed": seed,
                "arm": arm,
                "success": success,
                "steps": steps,
                "runtime_seconds": round(runtime, 2),
                "technical_pass": True,
            }), flush=True)
        if len(set(arm_hashes.values())) != 1:
            raise RuntimeError(f"Pi0 cross-arm pairing mismatch: {args.task} seed={seed}")


if __name__ == "__main__":
    main()
