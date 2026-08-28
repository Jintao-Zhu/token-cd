from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies import make_policy, make_pre_post_processors
from research.coreact_closed_loop.run_pilot import vector_info_value
from research.coreact_closed_loop.runtime import env_config, make_task_env, prepare


STEPS = [5000, 10000, 15000, 70000]
MAX_CONTROL_STEPS = 280
EXECUTED_PREFIX = 10


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def load(checkpoint: Path):
    cfg = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    cfg.pretrained_path, cfg.device, cfg.compile_model = checkpoint, "cuda", False
    rename = {"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}
    policy = make_policy(cfg=cfg, env_cfg=env_config("libero_spatial", 0), rename_map=rename)
    policy.eval().requires_grad_(False)
    policy.to(dtype=torch.float32)
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=checkpoint)
    return cfg, policy, pre, post


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--step", type=int, choices=STEPS, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    checkpoint = artifact / f"public_checkpoints/step_{args.step}"
    manifest = [json.loads(x) for x in (artifact / "public_quality_ladder_manifest.jsonl").read_text().splitlines()]
    rows = [x for x in manifest if x["checkpoint_step"] == args.step]
    cfg, policy, preprocessor, postprocessor = load(checkpoint)
    for ordinal, spec in enumerate(rows, 1):
        output = artifact / "public_quality_ladder_episodes" / f"{spec['episode_id']}.json"
        if output.exists():
            continue
        env, env_preprocessor, env_postprocessor = make_task_env("libero_spatial", spec["task_id"], cfg)
        queue, actions, noise_hashes, latencies = [], [], [], []
        replans, success = 0, False
        try:
            inner = env.envs[0]
            inner.init_state_id = spec["init_state_id"]
            observation, _ = env.reset(seed=spec["reset_seed"])
            initial_state = np.ascontiguousarray(inner._env.get_sim_state())
            initial_state_hash = hashlib.sha256(initial_state.tobytes()).hexdigest()
            for _ in range(MAX_CONTROL_STEPS):
                if not queue:
                    batch = prepare(policy, preprocessor, env_preprocessor, observation, spec["language"])
                    noise = torch.randn(
                        (1, cfg.chunk_size, cfg.max_action_dim),
                        generator=torch.Generator(device=batch["state"].device).manual_seed(spec["noise_seed"] + replans),
                        device=batch["state"].device,
                        dtype=batch["state"].dtype,
                    )
                    noise_hashes.append(tensor_hash(noise))
                    started = time.perf_counter()
                    with torch.inference_mode():
                        chunk = policy.model.sample_actions(batch["images"], batch["image_masks"], batch["lang_tokens"], batch["lang_masks"], batch["state"], noise=noise)
                    torch.cuda.synchronize()
                    latencies.append(time.perf_counter() - started)
                    queue.extend(chunk[:, :EXECUTED_PREFIX, :7].transpose(0, 1))
                    replans += 1
                action = queue.pop(0)
                if not bool(torch.isfinite(action).all()):
                    raise RuntimeError("nonfinite action")
                physical = postprocessor(action)
                legal = env_postprocessor({"action": physical})["action"]
                observation, _, terminated, _, info = env.step(legal.detach().cpu().numpy())
                actions.append(action[0].detach().float().cpu())
                success = bool(vector_info_value(info, "is_success"))
                if success or bool(terminated[0]):
                    break
        finally:
            env.close()
        array = torch.stack(actions)
        record = {**spec, "status": "complete", "success": success, "control_steps": len(actions), "replans": replans, "initial_sim_state_sha256": initial_state_hash, "noise_sha256_by_replan": noise_hashes, "all_actions_finite": bool(torch.isfinite(array).all()), "median_replan_latency_seconds": float(np.median(latencies))}
        if not record["all_actions_finite"] or not math.isfinite(record["median_replan_latency_seconds"]):
            raise RuntimeError("numeric integrity failure")
        atomic_json(output, record)
        print(json.dumps({"step": args.step, "ordinal": ordinal, "task": spec["task_id"], "init": spec["init_state_id"], "success": success, "control_steps": len(actions)}), flush=True)


if __name__ == "__main__":
    main()
