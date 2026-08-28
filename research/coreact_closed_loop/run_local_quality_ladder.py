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


STEPS = (0, 5000, 10000, 15000, 20000)
MAX_CONTROL_STEPS = 280
EXECUTED_PREFIX = 10


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().numpy()
    payload = str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def checkpoint_path(artifact: Path, step: int) -> Path:
    if step == 0:
        return artifact / "training_run/checkpoints/000000/pretrained_model"
    return artifact / f"training_run/trajectory/checkpoints/{step:06d}/pretrained_model"


def load(checkpoint: Path):
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    config.pretrained_path = checkpoint
    config.device = "cuda"
    config.compile_model = False
    rename = {
        "observation.images.image": "observation.images.camera1",
        "observation.images.image2": "observation.images.camera2",
    }
    policy = make_policy(config, env_cfg=env_config("libero_spatial", 0), rename_map=rename)
    policy.eval().requires_grad_(False)
    policy.to(dtype=torch.float32)
    pre_overrides = {"device_processor": {"device": "cuda"}}
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=checkpoint,
        preprocessor_overrides=pre_overrides,
    )
    return config, policy, preprocessor, postprocessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--step", type=int, choices=STEPS, required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--init-state-ids", type=int, nargs="+", default=list(range(10)))
    args = parser.parse_args()
    if not all(0 <= value < 10 for value in (*args.task_ids, *args.init_state_ids)):
        parser.error("task and init-state IDs must be in [0, 9]")
    artifact = args.artifact.resolve()
    checkpoint = checkpoint_path(artifact, args.step)
    output_dir = artifact / "quality_ladder_episodes"
    output_dir.mkdir(exist_ok=True)
    config, policy, preprocessor, postprocessor = load(checkpoint)

    ordinal = 0
    for task_id in args.task_ids:
        for init_state_id in args.init_state_ids:
            ordinal += 1
            episode_id = f"S{args.step:06d}__task{task_id:02d}__init{init_state_id:02d}"
            output = output_dir / f"{episode_id}.json"
            if output.exists():
                continue
            reset_seed = 610_000_000 + task_id * 1000 + init_state_id * 10
            noise_seed = 620_000_000 + task_id * 1000 + init_state_id * 10
            env, env_preprocessor, env_postprocessor = make_task_env(
                "libero_spatial", task_id, config
            )
            queue: list[torch.Tensor] = []
            actions: list[torch.Tensor] = []
            noise_hashes: list[str] = []
            latencies: list[float] = []
            replans = 0
            success = False
            try:
                inner = env.envs[0]
                inner.init_state_id = init_state_id
                observation, _ = env.reset(seed=reset_seed)
                language = inner.task_description
                initial_state = np.ascontiguousarray(inner._env.get_sim_state())
                initial_state_hash = hashlib.sha256(initial_state.tobytes()).hexdigest()
                for _ in range(MAX_CONTROL_STEPS):
                    if not queue:
                        batch = prepare(
                            policy, preprocessor, env_preprocessor, observation, language
                        )
                        noise = torch.randn(
                            (1, config.chunk_size, config.max_action_dim),
                            generator=torch.Generator(device=batch["state"].device).manual_seed(
                                noise_seed + replans
                            ),
                            device=batch["state"].device,
                            dtype=batch["state"].dtype,
                        )
                        noise_hashes.append(tensor_hash(noise))
                        started = time.perf_counter()
                        with torch.inference_mode():
                            chunk = policy.model.sample_actions(
                                batch["images"],
                                batch["image_masks"],
                                batch["lang_tokens"],
                                batch["lang_masks"],
                                batch["state"],
                                noise=noise,
                            )
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
            action_array = torch.stack(actions)
            record = {
                "episode_id": episode_id,
                "checkpoint_step": args.step,
                "task_id": task_id,
                "init_state_id": init_state_id,
                "reset_seed": reset_seed,
                "noise_seed": noise_seed,
                "language": language,
                "status": "complete",
                "success": success,
                "control_steps": len(actions),
                "replans": replans,
                "initial_sim_state_sha256": initial_state_hash,
                "noise_sha256_by_replan": noise_hashes,
                "all_actions_finite": bool(torch.isfinite(action_array).all()),
                "median_replan_latency_seconds": float(np.median(latencies)),
            }
            if not record["all_actions_finite"] or not math.isfinite(
                record["median_replan_latency_seconds"]
            ):
                raise RuntimeError("numeric integrity failure")
            atomic_json(output, record)
            print(
                json.dumps(
                    {
                        "step": args.step,
                        "ordinal": ordinal,
                        "task": task_id,
                        "init": init_state_id,
                        "success": success,
                        "control_steps": len(actions),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
