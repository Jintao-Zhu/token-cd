from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from lerobot.envs.factory import make_env_pre_post_processors

from research.coreact_closed_loop.runtime import env_config, prepare
from research.coreact_closed_loop.run_pilot import vector_info_value
from research.coreact_local_success.prepare_u1_manifest import ARMS
from research.coreact_self_guidance.reference_snapshot_gate import digest, fingerprints
from research.coreact_trained_weak.runtime import load_policy
from research.coreact_w1_slg_rollout.sampler import sample_w1_slg_actions


HORIZON = 520


def atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def scale(direction: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
    return direction * (target_norm / (torch.linalg.vector_norm(direction) + 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-units", type=int)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    os.environ.setdefault("MUJOCO_GL", "egl")
    rows = [json.loads(line) for line in (artifact / "u1_manifest.jsonl").read_text().splitlines()]
    units: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        units[row["unit_id"]][row["arm"]] = row
    selected = [item for index, item in enumerate(sorted(units.items())) if index % args.shard_count == args.shard_index]
    if args.max_units:
        selected = selected[:args.max_units]
    checkpoint = Path(json.loads((artifact / "protocol.lock.json").read_text())["strong_checkpoint"])
    config, policy, preprocessor, postprocessor = load_policy(checkpoint)
    components = torch.as_tensor(np.load(artifact / "pca_basis.npz")["components"], device="cuda", dtype=torch.float32)
    complete = 0
    for unit_id, specs in selected:
        outputs = {arm: artifact / "u1" / f"{specs[arm]['episode_id']}.json" for arm in ARMS}
        if all(path.exists() for path in outputs.values()):
            continue
        if any(path.exists() for path in outputs.values()):
            raise RuntimeError(f"partial unit {unit_id}")
        snapshot_id = specs["Strong"]["snapshot_id"]
        snapshot = torch.load(artifact / "snapshots" / f"{snapshot_id}.pt", weights_only=False, map_location="cpu")
        meta, prefix = snapshot["metadata"], snapshot["action_prefix"].numpy()
        branch, records = {}, {}
        for arm in ARMS:
            spec = specs[arm]
            cfg = env_config("libero_spatial", spec["task_id"])
            env_pre, env_post = make_env_pre_post_processors(env_cfg=cfg, policy_cfg=config)
            env = cfg.create_envs(n_envs=1, use_async_envs=False)["libero_spatial"][spec["task_id"]]
            queue, actions = [], []
            success, reason, replans = False, "horizon", 0
            try:
                inner = env.envs[0]
                inner.init_state_id = spec["init_state_id"]
                observation, _ = env.reset(seed=meta["reset_seed"])
                for prefix_action in prefix:
                    observation, _, terminated, _, _ = env.step(np.asarray(prefix_action, np.float32)[None, :])
                    if bool(terminated[0]):
                        raise RuntimeError("prefix terminated")
                batch = prepare(policy, preprocessor, env_pre, observation, meta["instruction"])
                fingerprint = fingerprints(env, observation, batch, list(prefix))
                generator = torch.Generator(device=batch["state"].device).manual_seed(spec["noise_seed"])
                noise = torch.randn((1, config.chunk_size, config.max_action_dim), generator=generator, device=batch["state"].device, dtype=batch["state"].dtype)
                with torch.inference_mode():
                    strong = policy.model.sample_actions(batch["images"], batch["image_masks"], batch["lang_tokens"], batch["lang_masks"], batch["state"], noise=noise)
                    guided, _ = sample_w1_slg_actions(policy.model, batch["images"], batch["image_masks"], batch["lang_tokens"], batch["lang_masks"], batch["state"], noise, arm="low_w1", lambda_value=0.05)
                base = strong[:, :10, :7]
                w1_delta = guided[:, :10, :7] - base
                target_norm = torch.linalg.vector_norm(w1_delta)
                if arm == "Strong":
                    delta = torch.zeros_like(base)
                    direction_index = None
                else:
                    arm_index = ARMS.index(arm) - 1
                    axis, sign = arm_index // 2, 1.0 if arm_index % 2 == 0 else -1.0
                    delta = scale(sign * components[axis].unsqueeze(0), target_norm)
                    direction_index = arm_index
                intervened = base + delta
                finite = bool(torch.isfinite(intervened).all())
                normalized_abs_gt_one = bool((intervened.abs() > 1.000001).any())
                if not finite:
                    atomic(artifact / "invalid_units" / f"{unit_id}.json", {"unit_id": unit_id, "arm": arm, "finite": finite})
                    raise RuntimeError(f"invalid intervention {unit_id} {arm}")
                chunk = strong.clone()
                chunk[:, :10, :7] = intervened
                branch[arm] = {"fingerprint": fingerprint, "noise": digest(noise), "strong_chunk": digest(strong), "w1_norm": float(target_norm), "delta_norm": float(torch.linalg.vector_norm(delta)), "delta_sha256": digest(delta), "direction_index": direction_index, "normalized_abs_gt_one": normalized_abs_gt_one, "physical_action_processing": "unchanged env postprocessor/controller"}
                queue = [value.detach().cpu() for value in chunk[:, :10, :7].transpose(0, 1)]
                replans = 1
                remaining = HORIZON - meta["resolved_control_step"]
                for _ in range(remaining):
                    if not queue:
                        batch = prepare(policy, preprocessor, env_pre, observation, meta["instruction"])
                        generator = torch.Generator(device=batch["state"].device).manual_seed(spec["noise_seed"] + replans)
                        next_noise = torch.randn((1, config.chunk_size, config.max_action_dim), generator=generator, device=batch["state"].device, dtype=batch["state"].dtype)
                        chunk = policy.model.sample_actions(batch["images"], batch["image_masks"], batch["lang_tokens"], batch["lang_masks"], batch["state"], noise=next_noise)
                        queue = [value.detach().cpu() for value in chunk[:, :10, :7].transpose(0, 1)]
                        replans += 1
                    model_action = queue.pop(0)
                    legal = env_post({"action": postprocessor(model_action)})["action"]
                    observation, _, terminated, _, info = env.step(legal.detach().cpu().numpy())
                    actions.append(model_action[0].float().cpu())
                    success = bool(vector_info_value(info, "is_success"))
                    if success:
                        reason = "success"
                        break
                    if bool(terminated[0]):
                        reason = "terminated"
                        break
            finally:
                env.close()
            records[arm] = {**spec, "status": "complete", "success": success, "termination_reason": reason, "continuation_steps": len(actions), "replans": replans, "all_actions_finite": bool(actions and all(torch.isfinite(value).all() for value in actions)), "branch": branch[arm]}
        for field in ("fingerprint", "noise", "strong_chunk", "w1_norm"):
            if len({json.dumps(branch[arm][field], sort_keys=True) for arm in ARMS}) != 1:
                atomic(artifact / "invalid_units" / f"{unit_id}.json", {"unit_id": unit_id, "field": field, "branch": branch})
                raise RuntimeError(f"branch mismatch {unit_id} {field}")
        for arm in ARMS:
            atomic(outputs[arm], records[arm])
        complete += 1
        print(json.dumps({"worker": args.shard_index, "complete": complete, "unit": unit_id}), flush=True)


if __name__ == "__main__":
    main()
