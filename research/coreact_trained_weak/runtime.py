from __future__ import annotations

from pathlib import Path
import json

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies import make_policy, make_pre_post_processors
from research.coreact_closed_loop.runtime import env_config


def checkpoint_path(artifact: Path, step: int) -> Path:
    return artifact / f"training_run/trajectory/checkpoints/{step:06d}/pretrained_model"


def load_policy(checkpoint: Path):
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
    if policy.training or any(parameter.requires_grad for parameter in policy.parameters()):
        raise RuntimeError("policy freeze/eval gate failed")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    return config, policy, preprocessor, postprocessor


def load_locked_pair(artifact: Path):
    import yaml

    lock = yaml.safe_load((artifact / "trained_weak_pair.lock.yaml").read_text())
    if lock["strong"]["step"] != 15000 or lock["weak"]["step"] != 10000:
        raise RuntimeError("unexpected trained-weak pair lock")
    strong = load_policy(checkpoint_path(artifact, lock["strong"]["step"]))
    weak = load_policy(checkpoint_path(artifact, lock["weak"]["step"]))
    strong_config = json.loads((checkpoint_path(artifact, lock["strong"]["step"]) / "config.json").read_text())
    weak_config = json.loads((checkpoint_path(artifact, lock["weak"]["step"]) / "config.json").read_text())
    for value in (strong_config, weak_config):
        value.pop("pretrained_path", None)
    if strong_config != weak_config:
        raise RuntimeError("Strong and Weak configs differ")
    return lock, strong, weak
