"""Shared frozen SmolVLA and LIBERO runtime for the closed-loop pilot."""

from __future__ import annotations

from pathlib import Path

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


CHECKPOINT_REVISION = "31d453f7edd78c839a8bbc39744a292686daf0de"


def checkpoint_path(workspace: Path) -> Path:
    return (
        workspace
        / "task1/.hf-cache/hub/models--lerobot--smolvla_libero/snapshots"
        / CHECKPOINT_REVISION
    )


def env_config(suite: str, task_id: int) -> LiberoEnv:
    return LiberoEnv(
        task=suite,
        task_ids=[task_id],
        observation_height=256,
        observation_width=256,
        control_mode="relative",
        episode_length=280,
    )


def load_policy_and_processors(workspace: Path):
    checkpoint = checkpoint_path(workspace)
    template = env_config("libero_spatial", 0)
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    if config.num_steps != 10 or config.chunk_size != 50:
        raise RuntimeError("pilot requires checkpoint-native 10 flow steps and H50")
    config.pretrained_path = checkpoint
    config.device = "cuda"
    config.compile_model = False
    rename_map = {
        "observation.images.image": "observation.images.camera1",
        "observation.images.image2": "observation.images.camera2",
    }
    policy = make_policy(cfg=config, env_cfg=template, rename_map=rename_map)
    policy.eval().requires_grad_(False)
    policy.to(dtype=torch.float32)
    if policy.training or any(parameter.requires_grad for parameter in policy.parameters()):
        raise RuntimeError("policy freeze/eval gate failed")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config, pretrained_path=checkpoint
    )
    return config, policy, preprocessor, postprocessor


def make_task_env(suite: str, task_id: int, policy_config):
    config = env_config(suite, task_id)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=config, policy_cfg=policy_config
    )
    env = config.create_envs(n_envs=1, use_async_envs=False)[suite][task_id]
    return env, env_preprocessor, env_postprocessor


def prepare(policy, preprocessor, env_preprocessor, observation, language: str) -> dict:
    batch = preprocess_observation(observation)
    batch["task"] = [language]
    batch = preprocessor(env_preprocessor(batch))
    images, image_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    return {
        "batch": batch,
        "images": images,
        "image_masks": image_masks,
        "state": state,
        "lang_tokens": batch[OBS_LANGUAGE_TOKENS],
        "lang_masks": batch[OBS_LANGUAGE_ATTENTION_MASK],
    }
