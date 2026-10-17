"""Minimal OpenVLA inference base used by the existing L11-Matched policy.

The research policy normally inherits through SIMPLER's OpenVLA wrapper. This
base supplies the same methods without importing a simulator or robot driver.
The second value of ``postprocess_actions`` is deliberately *not* a robot command.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


class OpenVLAContrastInference:
    def __init__(self, checkpoint: str, unnorm_key: str, device: str = "cuda:0") -> None:
        if not unnorm_key:
            raise ValueError("--unnorm-key must name statistics for the robot embodiment")
        if not torch.cuda.is_available() or not device.startswith("cuda"):
            raise RuntimeError("This OpenVLA-7B transfer runner requires a CUDA device")
        self.device = torch.device(device)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The locked source runtime uses bfloat16; this GPU lacks bf16 support")
        self.processor = AutoProcessor.from_pretrained(
            checkpoint, trust_remote_code=True, local_files_only=True
        )
        self.vla = AutoModelForVision2Seq.from_pretrained(
            checkpoint, attn_implementation="eager", torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, trust_remote_code=True, local_files_only=True,
        ).to(self.device).eval()
        self.unnorm_key = unnorm_key
        stats = self.vla.get_action_stats(unnorm_key)
        if len(stats["q01"]) != 7 or len(stats["q99"]) != 7:
            raise ValueError(f"{unnorm_key!r} must describe exactly 7 OpenVLA action dimensions")
        self.task_description: str | None = None
        self.image_size = (224, 224)

    def reset(self, task_description: str, seed: int | None = None) -> None:
        self.task_description = task_description
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("camera image must be RGB uint8 with shape [H,W,3]")
        return cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)

    def process_inputs(self, image: np.ndarray, task_description: str | None = None):
        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)
        if not self.task_description:
            raise ValueError("A nonempty task instruction is required")
        batch = self.processor(self.task_description, Image.fromarray(self._resize_image(image)))
        return batch.to(self.device, dtype=torch.bfloat16)

    def _forward_scores(self, inputs, unnorm_key: str, **kwargs) -> torch.Tensor:
        input_ids = inputs["input_ids"]
        if not torch.all(input_ids[:, -1] == 29871):
            suffix = torch.full((input_ids.shape[0], 1), 29871, device=input_ids.device,
                                dtype=input_ids.dtype)
            input_ids = torch.cat((input_ids, suffix), dim=1)
        result = self.vla.generate(
            input_ids=input_ids, pixel_values=inputs["pixel_values"],
            max_new_tokens=len(self.vla.get_action_stats(unnorm_key)["q01"]),
            output_scores=True, return_dict_in_generate=True, **kwargs,
        )
        return torch.cat(result.scores, dim=0)

    def _decode_actions(self, action_token_ids: torch.Tensor, unnorm_key: str) -> np.ndarray:
        ids = action_token_ids.detach().cpu().numpy()
        bins = np.clip(self.vla.vocab_size - ids - 1, 0, len(self.vla.bin_centers) - 1)
        normalized = np.asarray(self.vla.bin_centers)[bins]
        stats = self.vla.get_action_stats(unnorm_key)
        low = np.asarray(stats["q01"], dtype=np.float64)
        high = np.asarray(stats["q99"], dtype=np.float64)
        mask = np.asarray(stats.get("mask", np.ones(7, dtype=bool)), dtype=bool)
        return np.where(mask, (normalized + 1.0) * (high - low) / 2.0 + low, normalized)

    def postprocess_actions(self, raw_actions: np.ndarray):
        vector = np.asarray(raw_actions, dtype=np.float64).reshape(-1)
        if vector.shape != (7,) or not np.isfinite(vector).all():
            raise ValueError("decoded OpenVLA action must be a finite 7-vector")
        raw_action = {
            "world_vector": vector[:3].copy(),
            "rotation_delta": vector[3:6].copy(),
            "open_gripper": vector[6:7].copy(),
        }
        return raw_action, None
