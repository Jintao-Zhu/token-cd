"""Closed-loop VLA-Pruner-only policy for the audited OpenVLA+SIMPLER harness.

Reference: MINT-SJTU/VLA-Pruner commit 84d4b71.
  * predict_action history bookkeeping (modeling_prismatic.py:558)
  * _extract_action_modality_attentions (modeling_prismatic.py:640)
Porting decisions are recorded in
  artifacts/vla_pruner_openvla_reproduction/IMPLEMENTATION_GAP.md.

Arms:
  - vanilla           : fastv_cfg=None -> step() replicates base
                        OpenVLAInference exactly (current-harness vanilla).
  - vla_pruner_r0     : fastv_r=0.0 sanity arm (must equal vanilla exactly;
                        assertions: pruned==0, kept image count==256).
  - vla_pruner_prune25: fastv_r=0.25 (paper-main prefill+temporal)
  - vla_pruner_prune50: fastv_r=0.50

Per env step:
  1. fresh prefill runs inside self.vla.generate(); FastVLlamaModel prunes once
     at layer k (default 3) before layer k is applied.
  2. after the 7-token decode we read layer-15 action->vision attention
     (reconstructed to the original 256 visual columns) and push it into
     av_hist with decay gamma=0.8. The deque is only consulted once full, i.e.
     the first three steps of an episode run with fastv_r=0 (official warm-up).
  3. episode start must call reset(), which clears av_hist.
"""
from __future__ import annotations

import copy
from collections import deque
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image
from simpler_env.policies.openvla.openvla_model import OpenVLAInference
from transforms3d.euler import euler2axangle

from research.semantic_token_cd.vla_pruner_llama import DEFAULT_FASTV_CONFIG, attach_fastv, detach_fastv

ACTION_TOKEN_PREFIX = 29871
HISTORY_LAYER = 15
PAPER_HISTORY_LAYERS = tuple(range(16, 32))
VISION_START, VISION_END = 1, 257

VLA_PRUNER_ARM_CFG: Dict[str, Dict[str, Any]] = {
    "vanilla": {"fastv_r": None, "use_temporal": True, "av_hist_w": 3, "av_decay": 0.8},
    "vla_pruner_r0": {"fastv_r": 0.0, "use_temporal": True, "av_hist_w": 3, "av_decay": 0.8},
    "vla_pruner_prune25": {"fastv_r": 0.25, "use_temporal": True, "av_hist_w": 3, "av_decay": 0.8},
    "vla_pruner_prune50": {"fastv_r": 0.50, "use_temporal": True, "av_hist_w": 3, "av_decay": 0.8},
    # Keep code-faithful and paper-v5-faithful definitions separate.  The
    # current upstream repository hard-codes L15; arXiv v5 Sec. 4.3 says to
    # average action-to-vision attention over the latter half of 32 layers.
    "vla_pruner_prune75_code_l15": {
        "fastv_r": 0.75, "use_temporal": True, "av_hist_w": 3,
        "av_decay": 0.8, "history_layers": (15,),
    },
    "vla_pruner_prune75_paper_l16_31": {
        "fastv_r": 0.75, "use_temporal": True, "av_hist_w": 3,
        "av_decay": 0.8, "history_layers": PAPER_HISTORY_LAYERS,
    },
    # Diagnostic only: isolates prefill token deletion from temporal
    # Combine-then-Filter.  It is not a formal VLA-Pruner arm.
    "vla_pruner_prune75_prefill_only": {
        "fastv_r": 0.75, "use_temporal": False, "av_hist_w": 3,
        "av_decay": 0.8, "history_layers": (15,),
    },
}
VLA_PRUNER_ARMS = tuple(VLA_PRUNER_ARM_CFG)


def aggregate_action_history(action_vision: torch.Tensor, history_layers) -> torch.Tensor:
    """Return the 256-D temporal guide contribution for one control step."""
    layers = tuple(int(x) for x in history_layers)
    if not layers or min(layers) < 0 or max(layers) >= action_vision.shape[0]:
        raise ValueError(f"invalid history layers {layers} for {action_vision.shape[0]}-layer model")
    if len(layers) == 1:
        # Preserve the current upstream L15 reduction order bit-for-bit.
        return action_vision[layers[0]].float().mean(dim=0).mean(dim=0)
    return action_vision[list(layers)].float().mean(dim=(0, 1, 2))


def build_vla_pruner_policy(base, arm: str, task: str):
    cfg = VLA_PRUNER_ARM_CFG[arm]
    policy = copy.copy(base)
    policy.__class__ = VlaPrunerOpenVLAInference
    policy.fastv_cfg = None if arm == "vanilla" else dict(cfg)
    policy.arm = arm
    policy.task = task
    policy.av_hist = deque(maxlen=int(cfg.get("av_hist_w", 3)))
    policy.av_decay = float(cfg.get("av_decay", 0.8))
    policy.history_layers = tuple(int(x) for x in cfg.get("history_layers", (HISTORY_LAYER,)))
    policy._episode_trace = []
    policy._episode_step = 0
    return policy


def extract_action_vision_attentions(attentions, pruning_info: Optional[Dict[str, Any]] = None):
    """Action-token -> vision-token attention across layers, in the *original*
    256 visual layout.

    attentions: list over the 7 model calls of a step (call 0 = multimodal
    prefill, calls 1..6 = cached decode); each element is the per-layer tuple
    of attention tensors [1, heads, q_len, kv_len].

    Geometry after scheme-B pruning:
      * layers <  prune_layer ran on the full sequence: their kv columns 1..256
        are the original visual tokens.
      * layers >= prune_layer only ever saw the kept tokens: their kv columns
        are `kept_indices` in ascending original order, so columns 1..V
        (V = #kept visual tokens) are the kept visuals.
    Pruned visuals carry no key in layers >= prune_layer; their attention is
    reconstructed as 0 (the actual quantity the model computes with).

    Query row: for the prefill (action token 0) we use the final input row
    (the action-prefix token); for each cached decode step (tokens 1..6) the
    single query row.
    """
    if pruning_info is None or pruning_info.get("kept_indices") is None:
        # No pruning happened: every layer maps 1:1 to the original sequence.
        keep = None
        prune_layer = None
    else:
        keep = pruning_info["kept_indices"]
        prune_layer = int(pruning_info.get("pruning_layer", -1))
    num_action_tokens = len(attentions)
    num_layers = len(attentions[0])
    num_heads = attentions[0][0].shape[1]
    vision_start, vision_end = 1, 257
    vision_len = vision_end - vision_start
    device = attentions[0][0].device
    dtype = attentions[0][0].dtype
    action_vision = torch.zeros(
        num_layers, num_heads, num_action_tokens, vision_len, device=device, dtype=dtype
    )
    if keep is not None:
        keep = keep.to(device=device)
        kept_vision_mask = (keep >= vision_start) & (keep < vision_end)
        v_keep = int(kept_vision_mask.sum().item())
        vis_orig = keep[kept_vision_mask] - vision_start  # original 0..255 slots
    else:
        v_keep = vision_len
        vis_orig = torch.arange(vision_len, device=device)
    for action_idx in range(num_action_tokens):
        for layer_idx in range(num_layers):
            cur = attentions[action_idx][layer_idx][0]  # [heads, q, kv]
            if action_idx == 0:
                if cur.shape[1] > 1:
                    row = cur[:, -1, :]           # prefill: action-prefix query row
                else:
                    row = cur[:, 0, :]
            else:
                row = cur[:, 0, :]                # cached decode single query
            if prune_layer is not None and layer_idx >= prune_layer:
                vis_attn = row[:, 1 : 1 + v_keep] if v_keep else row.new_zeros(row.shape[0], 0)
                action_vision[layer_idx, :, action_idx, vis_orig] = vis_attn
            else:
                action_vision[layer_idx, :, action_idx, :] = row[:, vision_start:vision_end]
    return action_vision



class VlaPrunerOpenVLAInference(OpenVLAInference):
    """OpenVLA + optional VLA-Pruner branch. Steps replicate the base
    OpenVLAInference.step exactly; the only differences are (a) the pruner arm
    records attentions during generate, (b) a third aux output is returned."""

    fastv_cfg: Optional[Dict[str, Any]] = None
    arm: str = "vanilla"

    def reset(self, task_description: str, seed=None) -> None:
        super().reset(task_description, seed)
        self.av_hist.clear()
        self._episode_trace = []
        self._episode_step = 0

    def step(self, image, contrast_image=None, task_description=None, *args, **kwargs):
        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)

        assert image.dtype == np.uint8
        image = self._resize_image(image)
        pil_image = Image.fromarray(image)
        prompt = task_description
        inputs = self.processor(prompt, pil_image).to("cuda:0", dtype=torch.bfloat16)
        input_ids, pixel_values = inputs["input_ids"], inputs["pixel_values"]
        if not torch.all(input_ids[:, -1] == ACTION_TOKEN_PREFIX):
            input_ids = torch.cat(
                (
                    input_ids,
                    torch.tensor([[ACTION_TOKEN_PREFIX]], dtype=torch.long, device=input_ids.device),
                ),
                dim=1,
            )
        if "proprio" in kwargs:
            del kwargs["proprio"]

        use_prune = self.fastv_cfg is not None
        step_index = int(getattr(self, "_episode_step", 0))
        trace: Dict[str, Any] = {
            "arm": self.arm,
            "step_index": step_index,
            "history_len": len(self.av_hist),
        }

        if not use_prune:
            # Shared-model hygiene: the model may still carry a FastV config /
            # class from a previous pruner-arm step in the same process.
            detach_fastv(self.vla)
            scores = self.vla.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=self.get_action_dim(self.unnorm_key),
                output_scores=True,
                return_dict_in_generate=True,
                **kwargs,
            )["scores"]
        else:
            scores, prune_meta = self._pruned_generate(input_ids, pixel_values, kwargs)
            trace.update(prune_meta)

        scores = torch.cat(scores, dim=0)
        predicted_action_token_ids = scores.argmax(dim=-1)
        raw_actions = self._decode_actions(predicted_action_token_ids, self.unnorm_key)[None]

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),
        }
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle * self.action_scale

        if self.policy_setup == "google_robot":
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action
            self.previous_gripper_action = current_gripper_action
            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action
            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action
            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0
            action["gripper"] = relative_gripper_action
        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0

        action["terminate_episode"] = np.array([0.0])
        trace["token_ids"] = predicted_action_token_ids.detach().cpu().tolist()
        trace["raw_action"] = [float(x) for x in raw_actions[0].astype(np.float64)]

        if not use_prune:
            self.av_hist.clear()  # vanilla arm keeps no history (unused)
        self._episode_trace.append(trace)
        self._episode_step = step_index + 1
        aux = {"arm": self.arm, "step": step_index}
        return raw_action, action, aux

    # ---- VLA-Pruner branch ----------------------------------------------
    def _pruned_generate(self, input_ids, pixel_values, kwargs):
        fastv_r = float(self.fastv_cfg["fastv_r"])
        historical = None
        use_temporal = bool(self.fastv_cfg.get("use_temporal", True))
        if not use_temporal:
            effective_r = fastv_r
        elif len(self.av_hist) == self.av_hist.maxlen:
            weights = np.array(
                [self.av_decay ** i for i in range(len(self.av_hist))], dtype=np.float32
            )
            guided = np.zeros(256, dtype=np.float32)
            for i in range(len(weights)):
                guided += weights[i] * self.av_hist[-1 - i]
            guided = guided / np.sum(guided)
            historical = torch.tensor(guided, device=input_ids.device, dtype=torch.bfloat16)
            effective_r = fastv_r
        else:
            effective_r = 0.0  # official: history not full -> fastv_r forced to 0

        cfg = dict(DEFAULT_FASTV_CONFIG)
        cfg.update(
            {
                "fastv_k": int(self.fastv_cfg.get("fastv_k", 3)),
                "fastv_r": effective_r,
                "use_temporal": use_temporal,
                "historical_attention": historical,
                "use_prefil_attention": bool(self.fastv_cfg.get("use_prefil_attention", True)),
            }
        )
        attach_fastv(self.vla, cfg)
        out = self.vla.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=self.get_action_dim(self.unnorm_key),
            output_scores=True,
            output_attentions=True,
            return_dict_in_generate=True,
            **kwargs,
        )
        pi = getattr(self.vla.language_model.model, "pruning_info", None)
        prune_meta = {
            "fastv_r_effective": float(effective_r),
            "pruned_indices_empty": bool(
                pi is not None
                and pi.get("pruned_indices") is not None
                and pi["pruned_indices"].numel() == 0
            ),
            "pruned_count": (
                int(pi["pruned_indices"].shape[0])
                if pi and pi.get("pruned_indices") is not None
                else None
            ),
            "kept_count": (
                int(pi["kept_indices"].shape[0])
                if pi and pi.get("kept_indices") is not None
                else None
            ),
            "kept_image_count": (
                int(((pi["kept_indices"] >= 1) & (pi["kept_indices"] < 257)).sum())
                if pi and pi.get("kept_indices") is not None
                else None
            ),
            "original_seq_length": pi.get("original_seq_length") if pi else None,
            "pruning_layer": pi.get("pruning_layer") if pi else None,
            "guide_topk_overlap": pi.get("guide_topk_overlap") if pi else None,
            "used_redundancy": bool(pi.get("used_redundancy", False)) if pi else None,
            "used_fixed_visual_keep": bool(pi.get("used_fixed_visual_keep", False)) if pi else None,
            "kept_visual_token_ids": (
                (pi["kept_indices"][(pi["kept_indices"] >= 1) & (pi["kept_indices"] < 257)] - 1)
                .detach().cpu().tolist()
                if pi and pi.get("kept_indices") is not None
                else None
            ),
        }
        action_vision = extract_action_vision_attentions(out["attentions"], pi)
        detach_fastv(self.vla)
        if action_vision.numel() > 0:
            history_layers = tuple(getattr(self, "history_layers", (HISTORY_LAYER,)))
            # Equal weighting over selected layers, all heads and all seven
            # autoregressive action-query rows.  A singleton (15,) is exactly
            # the current upstream implementation.
            reduced = aggregate_action_history(action_vision, history_layers)
            vec = reduced.detach().cpu().numpy()
            if vec.shape[0] == 256:
                self.av_hist.append(vec)
        prune_meta["history_layers"] = list(getattr(self, "history_layers", (HISTORY_LAYER,)))
        prune_meta["history_after"] = len(self.av_hist)
        return out["scores"], prune_meta
