from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
import torch

from contrast_policies.openvla_contrast import OpenVLAContrastInference
from research.ar_token_counterfactual.intervention import projector_intervention
from research.token_pcd_stage_a.core import matched_random_ids, token_ids_from_mask


class FrozenTokenPCDInference(OpenVLAContrastInference):
    """Official OpenVLA PCD decoder with only the negative visual branch changed."""

    def __init__(self, replacement_mean: torch.Tensor, branch: str, **kwargs):
        if branch not in ("object_token_pcd", "random_token_pcd"):
            raise ValueError(branch)
        super().__init__(alpha=0.8, **kwargs)
        self.replacement_mean = replacement_mean
        self.branch = branch
        self._audit_context = None
        self._audit_calls = []

    def set_counterfactual(self, mask: np.ndarray, task: str, seed: int, timestep: int, rgb_sha256: str):
        object_ids, overlaps = token_ids_from_mask(mask, 0.25)
        material = f"{task}|{seed}|{timestep}|{rgb_sha256}|TOKEN_PCD_BRIDGE_R1".encode()
        random_seed = int(hashlib.sha256(material).hexdigest()[:16], 16)
        random_ids = matched_random_ids(object_ids, random_seed)
        selected = object_ids if self.branch == "object_token_pcd" else random_ids
        self._audit_context = {
            "object_token_ids": object_ids, "random_token_ids": random_ids,
            "selected_token_ids": selected, "token_overlap_ratio": overlaps.tolist(),
            "random_seed": random_seed,
            "empty_object_token_noop": not bool(object_ids),
        }

    def step(self, image, _unused_contrast_image, task_description=None, *args, **kwargs):
        if self._audit_context is None:
            raise RuntimeError("set_counterfactual must be called at every replan")
        inputs = self.process_inputs(image, task_description=task_description)
        clean_scores = self._forward_scores(inputs, self.unnorm_key, **kwargs)
        with projector_intervention(self.vla, self._audit_context["selected_token_ids"], self.replacement_mean) as trace:
            negative_scores = self._forward_scores(inputs, self.unnorm_key, **kwargs)
        if trace.before is None or trace.before.shape[1] != 256 or trace.after.shape != trace.before.shape:
            raise RuntimeError("Visual sequence shape changed")
        if list(trace.changed_indices) != self._audit_context["selected_token_ids"]:
            raise RuntimeError("Changed-index mismatch")
        if clean_scores.shape != negative_scores.shape or clean_scores.shape[0] != 7:
            raise RuntimeError(f"Invalid score shapes {clean_scores.shape} {negative_scores.shape}")
        final_scores = clean_scores.clone()
        final_scores[:-1] = 1.8 * clean_scores[:-1] - 0.8 * negative_scores[:-1]
        if not torch.isfinite(final_scores).all():
            raise FloatingPointError("Non-finite Token-PCD logits")
        token_ids = final_scores.argmax(dim=-1)
        raw = self._decode_actions(token_ids, self.unnorm_key)[None]
        raw_action, actions = self.postprocess_actions(raw)
        self._audit_calls.append({**self._audit_context, "changed_indices": list(trace.changed_indices),
                                  "clean_token_ids": clean_scores.argmax(-1).detach().cpu().tolist(),
                                  "negative_token_ids": negative_scores.argmax(-1).detach().cpu().tolist(),
                                  "pcd_token_ids": token_ids.detach().cpu().tolist(), "finite": True})
        self._audit_context = None
        return raw_action, actions, {"final_logits": final_scores}
