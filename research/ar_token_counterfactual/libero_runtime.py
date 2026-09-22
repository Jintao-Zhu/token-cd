"""Inference-only OpenVLA LIBERO runtime matching the official evaluation path."""

from __future__ import annotations

import io
import json
import os
import random
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
try:
    import tensorflow as tf
except ModuleNotFoundError:  # AR audit environments may not ship TensorFlow.
    tf = None
import torch
from PIL import Image

from .hf_loader import load_and_register_openvla_hf


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def attach_action_statistics(
    model: Any,
    dataset_statistics_path: Path,
    unnorm_key: str,
) -> dict[str, Any]:
    """Attach one explicitly selected action-statistics entry to an OpenVLA model.

    Some OpenVLA checkpoints store LIBERO-90 statistics in the standalone
    ``dataset_statistics.json`` file rather than under ``config.norm_stats``.
    Keeping this as an explicit load-time adaptation avoids silently rewriting
    the downloaded checkpoint config.
    """
    dataset_statistics_path = Path(dataset_statistics_path)
    if not dataset_statistics_path.is_file():
        raise FileNotFoundError(f"dataset statistics file not found: {dataset_statistics_path}")
    payload = json.loads(dataset_statistics_path.read_text())
    if unnorm_key not in payload:
        raise KeyError(
            f"unnorm key {unnorm_key!r} not found in {dataset_statistics_path}; "
            f"available={sorted(payload)}"
        )
    stats = payload[unnorm_key]
    action = stats.get("action")
    if not isinstance(action, dict):
        raise ValueError(f"missing action statistics for unnorm key {unnorm_key!r}")
    required = ("q01", "q99")
    for field in required:
        values = action.get(field)
        if not isinstance(values, list) or len(values) != 7:
            raise ValueError(
                f"invalid {field!r} action statistics for {unnorm_key!r}: {values!r}"
            )
    mask = action.get("mask")
    if mask is not None and (not isinstance(mask, list) or len(mask) != 7):
        raise ValueError(f"invalid action mask for {unnorm_key!r}: {mask!r}")
    model.norm_stats = dict(model.norm_stats or {})
    model.norm_stats[unnorm_key] = stats
    return stats


def load_policy(
    checkpoint: Path,
    code_dir: Path,
    device: str = "cuda:0",
    dataset_statistics_path: Path | None = None,
    unnorm_key: str | None = None,
) -> tuple[Any, Any]:
    classes = load_and_register_openvla_hf(code_dir)
    processor = classes["processor"].from_pretrained(checkpoint, local_files_only=True)
    model = classes["model"].from_pretrained(
        checkpoint,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model = model.to(torch.device(device)).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if dataset_statistics_path is not None:
        if not unnorm_key:
            raise ValueError("unnorm_key is required when dataset_statistics_path is provided")
        attach_action_statistics(model, dataset_statistics_path, unnorm_key)
    if unnorm_key is not None and unnorm_key not in (model.norm_stats or {}):
        raise KeyError(
            f"unnorm key {unnorm_key!r} not found in model.norm_stats; "
            f"available={sorted((model.norm_stats or {}).keys())}"
        )
    return model, processor


def resize_libero_image(image: np.ndarray, size: int = 224) -> np.ndarray:
    """Apply the official RLDS-matching JPEG round trip and Lanczos resize."""
    if tf is None:
        return np.asarray(Image.fromarray(image).resize((size, size), Image.Resampling.LANCZOS), dtype=np.uint8)
    tensor = tf.convert_to_tensor(image, dtype=tf.uint8)
    tensor = tf.image.encode_jpeg(tensor)
    tensor = tf.io.decode_image(tensor, expand_animations=False, dtype=tf.uint8)
    tensor = tf.image.resize(tensor, (size, size), method="lanczos3", antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(tensor), 0, 255), tf.uint8).numpy()


def center_crop_for_aug(image: np.ndarray, crop_scale: float = 0.9) -> Image.Image:
    """Apply the center-crop used by official OpenVLA evaluation for augmented checkpoints."""
    if tf is None:
        height, width = image.shape[:2]
        side = (crop_scale ** 0.5)
        crop_h, crop_w = int(round(height * side)), int(round(width * side))
        top, left = (height - crop_h) // 2, (width - crop_w) // 2
        return Image.fromarray(image[top:top + crop_h, left:left + crop_w]).resize((224, 224), Image.Resampling.BILINEAR).convert("RGB")
    tensor = tf.convert_to_tensor(image)
    original_dtype = tensor.dtype
    tensor = tf.image.convert_image_dtype(tensor, tf.float32)[None]
    side = tf.sqrt(tf.constant(crop_scale, dtype=tf.float32))
    offset = (1.0 - side) / 2.0
    boxes = tf.reshape(tf.stack([offset, offset, offset + side, offset + side]), (1, 4))
    tensor = tf.image.crop_and_resize(tensor, boxes, tf.range(1), (224, 224))[0]
    tensor = tf.image.convert_image_dtype(tf.clip_by_value(tensor, 0, 1), original_dtype, saturate=True)
    return Image.fromarray(tensor.numpy()).convert("RGB")


def prepare_agentview(obs: dict[str, Any]) -> tuple[np.ndarray, Image.Image]:
    rotated = obs["agentview_image"][::-1, ::-1]
    resized = resize_libero_image(rotated, 224)
    return resized, center_crop_for_aug(resized)


def build_prompt(task_description: str) -> str:
    return f"In: What action should the robot take to {task_description.lower()}?\nOut:"


@torch.inference_mode()
def predict_action(model: Any, processor: Any, image: Image.Image, task_description: str,
                   unnorm_key: str = "libero_spatial") -> np.ndarray:
    inputs = processor(build_prompt(task_description), image).to(model.device, dtype=torch.bfloat16)
    if not torch.all(inputs["input_ids"][:, -1] == 29871):
        inputs["input_ids"] = torch.cat(
            [inputs["input_ids"], torch.full((1, 1), 29871, dtype=inputs["input_ids"].dtype, device=model.device)], dim=1
        )
        inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], torch.ones((1, 1), dtype=inputs["attention_mask"].dtype, device=model.device)], dim=1
        )
    action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise RuntimeError(f"Invalid action returned by OpenVLA: shape={action.shape}, action={action}")
    return action


def prepare_env_action(action: np.ndarray) -> np.ndarray:
    result = np.array(action, dtype=np.float64, copy=True)
    result[-1] = np.sign(2.0 * result[-1] - 1.0)
    result[-1] *= -1.0
    return result


def encode_video(frames: list[np.ndarray], path: Path, fps: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=7)


def png_bytes(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()
