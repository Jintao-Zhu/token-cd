"""Pi0 L10 token-PCD paired rollout for the five remaining SIMPLER tasks.

Apple uses the Fractal-Beta checkpoint; the four WidowX tasks use Bridge-Beta.
The PCD implementation is shared with the Fractal Google-task runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/leju-suzhou/zjt_ws")
REPO = ROOT / "token-cd"
OPENPI = ROOT / "open-pi-zero"
ARTIFACT = REPO / "artifacts/pi0_l10_token_pcd_remaining5_rtfull_0_99_v1"
CANONICAL_SNAPSHOTS = REPO / "artifacts/vanilla_recon_shr_canonical_0_299_v2/snapshots"
FRACTAL_CKPT = ROOT / "checkpoints/pi0/fractal_beta_step29576_2024-12-29_13-10_42.pt"
BRIDGE_CKPT = ROOT / "checkpoints/pi0/bridge_beta_step19296_2024-12-26_22-30_42.pt"
TASKS = (
    "google_robot_place_apple_in_closed_top_drawer",
    "widowx_carrot_on_plate",
    "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel",
    "widowx_stack_cube",
)
TASK_CFG = {
    "google_robot_place_apple_in_closed_top_drawer": ("fractal_apple.yaml", FRACTAL_CKPT, "fractal_statistics.json"),
    "google_robot_pick_coke_can": ("fractal_coke.yaml", FRACTAL_CKPT, "fractal_statistics.json"),
    "widowx_carrot_on_plate": ("bridge.yaml", BRIDGE_CKPT, "bridge_statistics.json"),
    "widowx_put_eggplant_in_basket": ("bridge.yaml", BRIDGE_CKPT, "bridge_statistics.json"),
    "widowx_spoon_on_towel": ("bridge.yaml", BRIDGE_CKPT, "bridge_statistics.json"),
    "widowx_stack_cube": ("bridge.yaml", BRIDGE_CKPT, "bridge_statistics.json"),
}
LAYER = 10
CANDIDATES = 24
ALPHA = 0.2
ARMS = ("vanilla", "l10_token_pcd")
PROTOCOL = "PI0_L10_TOKEN_PCD_REMAINING5_RT_FULL_0_99_V1"


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_snapshot(task: str, seed: int) -> dict:
    path = CANONICAL_SNAPSHOTS / task / f"seed_{seed:03d}.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"canonical snapshot missing: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def lock_config(artifact: Path) -> None:
    config = {
        "protocol_id": PROTOCOL,
        "tasks": list(TASKS),
        "seeds_by_task": {task: list(range(100)) for task in TASKS},
        "arms": list(ARMS),
        "pairing": "shared canonical snapshot from vanilla_recon_shr_canonical_0_299_v2",
        "checkpoints_by_task": {task: str(TASK_CFG[task][1]) for task in TASKS},
        "renderer": {"shader": "rt", "depth": 8, "spp": 32, "denoiser": True, "offscreen_only": True},
        "prompt_attention_layer_0based": LAYER,
        "pcd": {"candidate_count": CANDIDATES, "alpha": ALPHA, "action_seed": "seed*1009+replan, shared with Vanilla"},
        "guidance": "KDE action-chunk likelihood-ratio resampling; no flow velocity mixing",
        "flow_steps": 10,
    }
    path = artifact / "CONFIG_LOCK.json"
    encoded = json.dumps(config, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise RuntimeError(f"CONFIG_LOCK differs from requested protocol: {path}")
    else:
        tmp = path.with_suffix(f".json.{os.getpid()}.tmp")
        tmp.write_text(encoded)
        try:
            os.link(tmp, path)
        except FileExistsError:
            if json.loads(path.read_text()) != config:
                raise RuntimeError(f"concurrent CONFIG_LOCK mismatch: {path}")
        finally:
            tmp.unlink(missing_ok=True)


def make_env(task: str):
    import gymnasium as gym
    import simpler_env

    env_id, kwargs = simpler_env.ENVIRONMENT_MAP[task]
    kwargs = dict(kwargs)
    kwargs.update(
        prepackaged_config=True,
        shader_dir="rt",
        renderer_kwargs={"device": "cuda:0", "offscreen_only": True},
        render_config={"rt_samples_per_pixel": 32, "rt_max_path_depth": 8, "rt_use_denoiser": True},
    )
    return gym.make(env_id, obs_mode="rgbd", **kwargs), env_id


def load_model(task: str, device):
    import torch
    import hydra
    from omegaconf import OmegaConf
    from src.model.vla.pizero import PiZeroInference

    config_name, checkpoint_path, stats_name = TASK_CFG[task]
    cfg = OmegaConf.load(OPENPI / "config/eval" / config_name)
    cfg.use_torch_compile = False
    cfg.use_bf16 = False
    cfg.checkpoint_path = str(checkpoint_path)
    cfg.gpu_id = 0
    cfg.env.adapter.dataset_statistics_path = str(OPENPI / "config" / stats_name)
    cfg.env.adapter.pretrained_model_path = str(ROOT / "checkpoints/pi0/paligemma-3b-pt-224")
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size < 1_000_000_000:
        raise FileNotFoundError(f"missing/incomplete Pi0 checkpoint: {checkpoint_path}")
    model = PiZeroInference(cfg, use_ddp=False)
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    weights = {k.replace("_orig_mod.", ""): v for k, v in raw["model"].items()}
    model.load_state_dict(weights, strict=True)
    del raw, weights
    model.freeze_all_weights()
    model.to(torch.float32).to(device).eval()
    adapter = hydra.utils.instantiate(cfg.env.adapter)
    return cfg, model, adapter, checkpoint_path


def atomic_json(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".json.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def run_episode(env, cfg, model, adapter, pcd, obs, instruction, task: str, seed: int, arm: str, device):
    import torch

    infos, trace = [], []
    steps = replans = 0
    terminated = truncated = False
    started = time.monotonic()
    while not (terminated or truncated):
        action_seed = seed * 1009 + replans
        torch.manual_seed(action_seed)
        torch.cuda.manual_seed_all(action_seed)
        if arm == "vanilla":
            inputs = adapter.preprocess(env, obs, instruction)
            inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
            causal, vlm_pos, prop_pos, action_pos = model.build_causal_mask_and_position_ids(
                inputs["attention_mask"], dtype=inputs["pixel_values"].dtype
            )
            causal = causal.to(device)
            vlm_pos, prop_pos, action_pos = vlm_pos.to(device), prop_pos.to(device), action_pos.to(device)
            vlm_prop, action_mask = model.split_full_mask_into_submasks(causal)
            model_inputs = {
                "input_ids": inputs["input_ids"], "pixel_values": inputs["pixel_values"],
                "image_text_proprio_mask": vlm_prop, "action_mask": action_mask,
                "vlm_position_ids": vlm_pos, "proprio_position_ids": prop_pos,
                "action_position_ids": action_pos, "proprios": inputs["proprios"],
            }
            with torch.inference_mode():
                actions = model(**model_inputs)
            env_actions = adapter.postprocess(actions[0].float().cpu().numpy())
        else:
            env_actions, step_trace = pcd.predict(env, obs, instruction, action_seed)
            trace.append(step_trace)

        for action in env_actions[: int(cfg.act_steps)]:
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            success = bool(info.get("success", terminated)) if isinstance(info, dict) else bool(terminated)
            infos.append({"success": success, "reward": float(reward), "terminated": bool(terminated),
                          "truncated": bool(truncated), "info": str(info)})
            if terminated and not env.unwrapped.is_final_subtask():
                terminated = False
                env.advance_to_next_subtask()
            instruction = env.unwrapped.get_language_instruction()
            if truncated or terminated:
                break
        replans += 1

    return {
        "success": bool(infos[-1]["success"]) if infos else False,
        "steps": steps, "replans": replans, "runtime_seconds": time.monotonic() - started,
        "trace": trace, "last_info": infos[-1]["info"] if infos else None,
    }


def parse_seeds(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    out = sorted(set(out))
    if not out or min(out) < 0 or max(out) > 99:
        raise ValueError("seeds must be a non-empty subset of 0..99")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--gpu", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise RuntimeError(f"launch with CUDA_VISIBLE_DEVICES={args.gpu}")

    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / "checkpoints/pi0"))
    sys.path[:0] = [str(OPENPI), str(REPO / "task1/shim_site"), str(REPO)]
    os.chdir(OPENPI)

    import torch
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    from research.semantic_token_cd.pi0_fractal_simpler_l2_pcd_rollout import (
        FractalPi0PCD, restore_snapshot, snapshot_sha,
    )

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    random.seed(0)
    np.random.seed(0)
    device = torch.device("cuda:0")
    artifact = args.artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    lock_config(artifact)
    cfg, model, adapter, checkpoint = load_model(args.task, device)
    checkpoint_sha = sha256_file(checkpoint)
    env, env_id = make_env(args.task)
    task_out = artifact / "episodes" / args.task
    snapshot_dir = artifact / "snapshots" / args.task
    print(json.dumps({"event": "model_ready", "task": args.task, "gpu": args.gpu,
                      "worker": args.worker_id, "seeds": len(parse_seeds(args.seeds)),
                      "checkpoint": str(checkpoint), "layer": LAYER}), flush=True)
    try:
        for seed in parse_seeds(args.seeds):
            snapshot = load_snapshot(args.task, seed)
            canonical_hash = snapshot_sha(snapshot)
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            snapshot_ref = snapshot_dir / f"seed_{seed:03d}.source.txt"
            snapshot_ref.write_text(str(CANONICAL_SNAPSHOTS / args.task / f"seed_{seed:03d}.pkl") + "\n")
            arm_hashes = []
            existing_paths = [task_out / arm / f"episode_{seed:03d}_summary.json" for arm in ARMS]
            existing_rows = [json.loads(path.read_text()) if path.exists() else None for path in existing_paths]
            if all(row is not None for row in existing_rows):
                old_hashes = [(row.get("initial_state_sha256"), row.get("initial_rgb_sha256")) for row in existing_rows]
                if len(set(old_hashes)) == 1:
                    continue
                # If a prior interrupted attempt left a bad pair, rerun both arms
                # from the canonical snapshot rather than preserving half a pair.
                for path in existing_paths:
                    path.unlink(missing_ok=True)
            for arm in ARMS:
                summary_path = task_out / arm / f"episode_{seed:03d}_summary.json"
                if summary_path.exists():
                    summary_path.unlink()
                obs, state_hash, rgb_hash = restore_snapshot(
                    env, seed, snapshot, get_image_from_maniskill2_obs_dict
                )
                arm_hashes.append((state_hash, rgb_hash))
                adapter.reset()
                instruction = env.unwrapped.get_language_instruction()
                pcd = FractalPi0PCD(model, adapter, candidates=CANDIDATES, alpha=ALPHA, layer=LAYER)
                result = run_episode(env, cfg, model, adapter, pcd, obs, instruction,
                                     args.task, seed, arm, device)
                row = {
                    "protocol_id": PROTOCOL, "task": args.task, "environment_id": env_id,
                    "seed": seed, "arm": arm, "success": result["success"],
                    "control_steps": result["steps"], "replans": result["replans"],
                    "runtime_seconds": result["runtime_seconds"], "gpu": args.gpu,
                    "worker_id": args.worker_id, "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checkpoint_sha,
                    "config": TASK_CFG[args.task][0], "renderer": {"shader": "rt", "depth": 8, "spp": 32, "denoiser": True},
                    "canonical_snapshot_sha256": canonical_hash,
                    "initial_state_sha256": state_hash, "initial_rgb_sha256": rgb_hash,
                    "prompt_attention_layer_0based": LAYER,
                    "guidance_method": "KDE candidate reweighting; no flow-velocity mixing",
                    "attention_trace": result["trace"], "last_info": result["last_info"],
                }
                atomic_json(summary_path, row)
                print(json.dumps({"task": args.task, "seed": seed, "arm": arm,
                                  "success": row["success"], "steps": row["control_steps"],
                                  "runtime_s": round(row["runtime_seconds"], 1), "worker": args.worker_id}), flush=True)
            pair_valid = len(set(arm_hashes)) == 1
            for summary_path in existing_paths:
                if summary_path.exists():
                    row = json.loads(summary_path.read_text())
                    row["pairing_valid"] = pair_valid
                    row["pair_initial_hashes"] = arm_hashes
                    atomic_json(summary_path, row)
            if not pair_valid:
                print(json.dumps({"event": "pairing_invalid", "task": args.task,
                                  "seed": seed, "hashes": arm_hashes,
                                  "worker": args.worker_id}), flush=True)
    finally:
        env.close()
    print(json.dumps({"event": "worker_complete", "worker": args.worker_id, "gpu": args.gpu}), flush=True)


if __name__ == "__main__":
    main()
