from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from lerobot.configs import PreTrainedConfig
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.random_utils import set_seed

from research.coreact_closed_loop.runtime import env_config, make_task_env, prepare


ROOT = Path("/data/docker/dev_zjt/data/code")
ART = ROOT / "artifacts/coreact_trained_weak_autoguidance_v1_20260812_000000"
CKPTS = ART / "public_checkpoints"
STEPS = [5000, 10000, 15000, 70000]


def schema(path: Path):
    rows = []
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            rows.append({"key": key, "shape": list(tensor.shape), "dtype": str(tensor.dtype), "numel": tensor.numel()})
    return rows


def weight_differences():
    selected = None
    tensors = {}
    for step in STEPS:
        path = CKPTS / f"step_{step}/model.safetensors"
        with safe_open(path, framework="pt", device="cpu") as f:
            if selected is None:
                keys = list(f.keys())
                selected = [keys[i] for i in np.linspace(0, len(keys) - 1, 16, dtype=int)]
            tensors[step] = {key: f.get_tensor(key).float() for key in selected}
    out = []
    for a, b in zip(STEPS[:-1], STEPS[1:]):
        sq_diff = sq_base = 0.0
        by_module = {}
        for key in selected:
            x, y = tensors[a][key], tensors[b][key]
            diff = float(torch.sum((y - x) ** 2))
            base = float(torch.sum(x**2))
            sq_diff += diff
            sq_base += base
            module = key.split(".")[0]
            item = by_module.setdefault(module, [0.0, 0.0])
            item[0] += diff
            item[1] += base
        out.append({
            "step_a": a,
            "step_b": b,
            "fixed_tensor_keys": selected,
            "relative_l2": (sq_diff / max(sq_base, 1e-30)) ** 0.5,
            "per_top_module_relative_l2": {k: (v[0] / max(v[1], 1e-30)) ** 0.5 for k, v in by_module.items()},
        })
    return out


def smoke(step: int):
    checkpoint = CKPTS / f"step_{step}"
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    config.pretrained_path = checkpoint
    config.device = "cuda"
    config.compile_model = False
    template = env_config("libero_spatial", 4)
    rename_map = {"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}
    policy = make_policy(cfg=config, env_cfg=template, rename_map=rename_map)
    policy.eval().requires_grad_(False)
    policy.to(dtype=torch.float32)
    pre, post = make_pre_post_processors(policy_cfg=config, pretrained_path=checkpoint)
    env, env_pre, _ = make_task_env("libero_spatial", 4, config)
    inner = env.envs[0]
    inner.init_state_id = 0
    observation, _ = env.reset(seed=42000)
    language = inner.task_description
    actions = []
    for _ in range(2):
        set_seed(1729)
        policy.reset()
        batch = prepare(policy, pre, env_pre, observation, language)["batch"]
        with torch.inference_mode():
            action = post(policy.select_action(batch)).detach().float().cpu().numpy()
        actions.append(action)
    env.close()
    delta = float(np.max(np.abs(actions[0] - actions[1])))
    return {"step": step, "shape": list(actions[0].shape), "finite": bool(np.isfinite(actions[0]).all()), "rerun_max_abs": delta, "deterministic": delta == 0.0}


def main():
    schemas = {str(step): schema(CKPTS / f"step_{step}/model.safetensors") for step in STEPS}
    digest = lambda rows: hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    summary = {
        "steps": STEPS,
        "tensor_count": {k: len(v) for k, v in schemas.items()},
        "parameter_count": {k: sum(x["numel"] for x in v) for k, v in schemas.items()},
        "schema_sha256": {k: digest(v) for k, v in schemas.items()},
        "identical": len({digest(v) for v in schemas.values()}) == 1,
        "schema": schemas[str(STEPS[0])],
        "fixed_subset_weight_differences": weight_differences(),
    }
    (ART / "public_checkpoint_tensor_schema.json").write_text(json.dumps(summary, indent=2) + "\n")
    smoke_rows = [smoke(step) for step in STEPS]
    (ART / "inference_smoke.json").write_text(json.dumps({"results": smoke_rows, "pass": all(x["finite"] and x["deterministic"] for x in smoke_rows)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
