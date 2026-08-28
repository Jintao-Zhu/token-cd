"""CW-LPCD shared per-state forward engine (STEP 3-6).

For one state this computes, with a single shared Vanilla prefix (teacher-forced):

  clean       P_clean(a_j | image, instruction, a_<j)
  pixel       P_pixelneg(a_j | contrast_image, instruction, a_<j)   [Pixel-PCD image-level]
  persistent  P_persistent(a_j | image, ..., projector row-replacement)  [old Token-PCD]
  latent[W]   P_latentneg(a_j | image, ..., CW window W on object keys)
  random[W]   P_randomneg(a_j | image, ..., CW window W on matched-random keys)

and returns the action-vocab (256-dim) logits plus log_softmax residuals for each.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch

from research.cw_lpcd.core import (
    N_ACTION_TOKENS, N_VISUAL, action_token_slice, ensure_empty_action_token,
    generate_clean_ids, inputs_for, log_softmax_residual, matched_random_ids,
    teacher_forced_logits, teacher_forced_logits_persistent, visual_key_positions,
)


def random_seed_for(state_id: str) -> int:
    """Matched-random deterministic seed = hash(state_uid) (spec STEP 5)."""
    return int(hashlib.sha256(state_id.encode("utf-8")).hexdigest()[:8], 16)


def load_position_mean(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True)["mean"]


def action_query_positions(inputs: dict[str, torch.Tensor]) -> list[int]:
    base_ids, _ = ensure_empty_action_token(inputs["input_ids"], inputs["attention_mask"])
    return [N_VISUAL + base_ids.shape[1] - 1 + offset for offset in range(N_ACTION_TOKENS)]


@torch.inference_mode()
def compute_state(model, processor, pcd_root: Path, row: dict, object_ids: list[int],
                  random_ids: list[int], mean: torch.Tensor, windows: list[dict]) -> dict:
    clean_path = pcd_root / row["clean_path"]
    pixel_path = pcd_root / row["pixel_path"]
    clean_img = cv2.cvtColor(cv2.imread(str(clean_path)), cv2.COLOR_BGR2RGB)
    pixel_img = cv2.cvtColor(cv2.imread(str(pixel_path)), cv2.COLOR_BGR2RGB)
    clean_inputs = inputs_for(processor, model, clean_img, row["instruction"])
    pixel_inputs = inputs_for(processor, model, pixel_img, row["instruction"])
    if not torch.equal(clean_inputs["input_ids"], pixel_inputs["input_ids"]):
        raise RuntimeError(f"clean/pixel language tokens differ for {row['state_id']}")

    clean_ids = generate_clean_ids(model, clean_inputs)  # [1, 7] shared Vanilla prefix
    token_slice = action_token_slice(model)
    qpos = action_query_positions(clean_inputs)
    obj_key = visual_key_positions(object_ids)
    rnd_key = visual_key_positions(random_ids)

    clean = teacher_forced_logits(model, clean_inputs, clean_ids)          # [7, 32064]
    pixel = teacher_forced_logits(model, pixel_inputs, clean_ids)
    persistent, ptrace = teacher_forced_logits_persistent(model, clean_inputs, clean_ids, object_ids, mean)

    latent = {}   # (start, end) -> [7, 32064]
    random = {}
    for w in windows:
        wr = (w["start"], w["end"])
        latent[wr] = teacher_forced_logits(model, clean_inputs, clean_ids,
                                           cw_query_positions=qpos, cw_key_positions=obj_key, cw_window=wr)
        random[wr] = teacher_forced_logits(model, clean_inputs, clean_ids,
                                           cw_query_positions=qpos, cw_key_positions=rnd_key, cw_window=wr)

    # 256-dim action-vocab logits and log_softmax residuals (spec STEP 7).
    clean_a = clean[:, token_slice].numpy()            # [7, 256]
    pixel_a = pixel[:, token_slice].numpy()
    persistent_a = persistent[:, token_slice].numpy()
    latent_a = np.stack([latent[wr][:, token_slice].numpy() for wr in latent], axis=0)    # [n_win, 7, 256]
    random_a = np.stack([random[wr][:, token_slice].numpy() for wr in random], axis=0)

    r_pixel = log_softmax_residual(clean, pixel, token_slice).numpy()
    r_persistent = log_softmax_residual(clean, persistent, token_slice).numpy()
    r_latent = np.stack([log_softmax_residual(clean, latent[wr], token_slice).numpy() for wr in latent], axis=0)
    r_random = np.stack([log_softmax_residual(clean, random[wr], token_slice).numpy() for wr in random], axis=0)

    if tuple(ptrace.changed_indices) != tuple(sorted(object_ids)):
        raise RuntimeError(f"persistent changed indices {ptrace.changed_indices} != object_ids {object_ids}")

    return {
        "state_id": row["state_id"], "task": row["task"], "seed": row["seed"],
        "clean_ids": clean_ids[0].cpu().tolist(),
        "object_ids": object_ids, "random_ids": random_ids,
        "window_order": [(w["center"], w["size"], w["start"], w["end"]) for w in windows],
        "clean_action": clean_a, "pixel_action": pixel_a, "persistent_action": persistent_a,
        "latent_action": latent_a, "random_action": random_a,
        "r_pixel": r_pixel, "r_persistent": r_persistent, "r_latent": r_latent, "r_random": r_random,
    }
