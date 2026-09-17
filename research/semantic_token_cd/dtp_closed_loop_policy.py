"""Closed-loop DTP policy used by dtp_closed_loop_rollout.

Each simulator step:
  1. builds processor inputs for the current image/instruction;
  2. runs one clean fixed-length generation to capture projector features;
  3. re-extracts per-layer prompt attention and picks the calibrated layer;
  4. decodes the seven action tokens with the DTP masked-regeneration
     adapter (decode_dtp); the control arm disables masking.

The step returns the same (raw_action, action, aux) contract as the other
closed-loop policies in this repository.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from contrast_policies.openvla_contrast import OpenVLAContrastInference
from research.ar_token_counterfactual.intervention import projector_intervention
from research.semantic_token_cd.dtp_closed_loop_protocol import ARM_CONFIG, ARMS
from research.semantic_token_cd.dtp_openvla_policy import decode_dtp, decode_dtp_fixed
from research.semantic_token_cd.dtp_paper_calibration import spatial, top
from research.semantic_token_cd.prompt_attn_shr_policy import (
    extract_prompt_attention_per_layer,
)

ACTION_DIM = 7


def arm_config(arm: str) -> dict:
    if arm not in ARM_CONFIG:
        raise ValueError(f"unknown DTP arm: {arm}")
    return dict(ARM_CONFIG[arm])


def build_policy(base, task: str, arm: str):
    cfg = arm_config(arm)
    policy = copy.copy(base)
    policy.__class__ = DTPClosedLoopInference
    policy.dtp_layer = int(cfg["layer"])
    policy.dtp_k = int(cfg["k"])
    policy.dtp_tau = float(cfg["tau"])
    policy.dtp_enabled = bool(cfg["enabled"])
    policy.dtp_mode = str(cfg.get("mode", "dynamic"))
    policy.task_index = task
    policy._episode_trace = []
    policy._episode_step = 0
    return policy


class DTPClosedLoopInference(OpenVLAContrastInference):
    """Greedy OpenVLA + DTP paper-based masked visual-key pruning adapter."""

    def step(self, image, contrast_image=None, task_description=None, *args, **kwargs):
        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)
        inputs = self.process_inputs(image, task_description=task_description)
        with projector_intervention(self.vla) as trace:
            clean = self._forward_scores(inputs, self.unnorm_key, do_sample=False)
        if clean.shape[0] != ACTION_DIM or trace.before is None:
            raise RuntimeError(
                "DTP step: clean generation did not yield 7 action scores or visual features"
            )
        visual = trace.before
        prompt, prompt_meta = extract_prompt_attention_per_layer(
            self, inputs, task_description, visual
        )
        decoder = decode_dtp_fixed if self.dtp_mode == "fixed" else decode_dtp
        result = decoder(
            self,
            inputs,
            prompt,
            layer=self.dtp_layer,
            k=self.dtp_k,
            tau=self.dtp_tau,
            enabled=self.dtp_enabled,
        )
        token_ids = np.asarray(result["tokens"], dtype=np.int64)
        dimension_trace = result["trace"]
        tokens_tensor = torch.from_numpy(token_ids)
        raw = self._decode_actions(tokens_tensor, self.unnorm_key)  # [7,]
        raw_action, action = self.postprocess_actions(raw[None])

        pruned = np.asarray([len(d["pruned"]) for d in dimension_trace], dtype=np.int64)
        flips = np.asarray(
            [int(d["refined_token"] != d["probe_token"]) for d in dimension_trace],
            dtype=np.int64,
        )
        self._episode_trace.append({
            "layer": int(self.dtp_layer),
            "k": int(self.dtp_k),
            "tau": float(self.dtp_tau),
            "enabled": bool(self.dtp_enabled),
            "mode": str(self.dtp_mode),
            "pruned_per_dim": pruned.tolist(),
            "token_flips_per_dim": flips.tolist(),
            "any_prune": bool(int(pruned.sum()) > 0),
            "total_pruned": int(pruned.sum()),
            "total_flips": int(flips.sum()),
            "attention_mass_on_visual": prompt_meta["per_layer_attention_mass_on_visual"],
            # mechanism fields per step's first action token (fixed mode D is
            # defined from dim0; dynamic mode records the refined-prefix dim0 probe):
            # |D|, a_m, tau*a_m, max A_unimportant, sum_{v in D} A[v] (per dimension row).
            "dim0_trace": [
                row for row in dimension_trace if int(row["dimension"]) == 0
            ],
        })
        step_index = int(self._episode_step)
        self._episode_step += 1
        aux = {
            "layer": int(self.dtp_layer),
            "k": int(self.dtp_k),
            "tau": float(self.dtp_tau),
            "mode": str(self.dtp_mode),
            "step": step_index,
        }
        del clean, result
        return raw_action, action, aux
