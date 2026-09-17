"""Shared-clean VLA-Pruner P50 + Prompt-CD policy.

The positive and harmonic-negative branches use the same P50 visual keep set.
Only the positive branch updates the temporal action-attention history.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import torch

from research.ar_token_counterfactual.intervention import ensure_empty_action_token, projector_intervention
from research.semantic_token_cd.distractor_policy import ACTION_VOCAB_SIZE, _action_logits
from research.semantic_token_cd.global_merge_policy import projector_merge_intervention
from research.semantic_token_cd.prompt_attn_shr_policy import (
    N_VISUAL,
    PromptAttentionSHRInference,
    extract_prompt_attention,
    mask_spatial_stats,
    stable_top_m,
)
from research.semantic_token_cd.st_shr_policy import harmonic_reconstruct
from research.semantic_token_cd.vla_pruner_llama import DEFAULT_FASTV_CONFIG, attach_fastv, detach_fastv
from research.semantic_token_cd.vla_pruner_policy import VlaPrunerOpenVLAInference

ACTION_TOKEN_PREFIX = 29871


@torch.inference_mode()
def fixed_keep_guided_scores(
    policy,
    inputs: dict[str, torch.Tensor],
    clean_token_ids: torch.Tensor,
    reconstructed_visual: torch.Tensor,
    visual_keep: list[int],
) -> tuple[torch.Tensor, dict]:
    base_ids, base_mask = ensure_empty_action_token(inputs["input_ids"], inputs["attention_mask"])
    prefix = clean_token_ids[:-1].unsqueeze(0)
    teacher_ids = torch.cat([base_ids, prefix], dim=1)
    teacher_mask = torch.cat(
        [base_mask, torch.ones_like(prefix, dtype=base_mask.dtype, device=base_mask.device)], dim=1
    )
    cfg = dict(DEFAULT_FASTV_CONFIG)
    cfg.update({
        "fastv_k": 3,
        "fastv_r": 0.5,
        "use_temporal": False,
        "historical_attention": None,
        "use_prefil_attention": True,
        "fixed_visual_keep": list(visual_keep),
    })
    attach_fastv(policy.vla, cfg)
    try:
        with projector_merge_intervention(policy.vla, reconstructed_visual) as trace:
            output = policy.vla(
                input_ids=teacher_ids,
                attention_mask=teacher_mask,
                pixel_values=inputs["pixel_values"],
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )
        pruning_info = getattr(policy.vla.language_model.model, "pruning_info", None)
        if trace["before"] is None or trace["after"] is None:
            raise RuntimeError("negative projector intervention was not applied")
        if pruning_info is None or not pruning_info.get("used_fixed_visual_keep", False):
            raise RuntimeError("negative branch did not use the locked visual keep set")
        actual_keep = (
            pruning_info["kept_indices"]
            [(pruning_info["kept_indices"] >= 1) & (pruning_info["kept_indices"] < 257)]
            - 1
        ).detach().cpu().tolist()
        if actual_keep != sorted(visual_keep):
            raise RuntimeError("negative branch visual keep set differs from positive branch")
        scores = output.logits[0, -int(clean_token_ids.shape[0]):].detach().float()
        if scores.shape[0] != clean_token_ids.shape[0]:
            raise RuntimeError("fixed-keep negative branch returned incomplete action logits")
        return scores, {
            "negative_fixed_keep_exact": True,
            "negative_kept_image_count": len(actual_keep),
            "negative_pruning_layer": int(pruning_info["pruning_layer"]),
        }
    finally:
        detach_fastv(policy.vla)


class P50PromptCDInference(PromptAttentionSHRInference):
    """Factorial D arm with explicit switches used by the equivalence gates."""

    pruner_enabled: bool = True
    cd_enabled: bool = True

    def reset(self, task_description: str, seed=None) -> None:
        PromptAttentionSHRInference.reset(self, task_description, seed)
        self.av_hist.clear()
        self._episode_step = 0

    def step(self, image, contrast_image=None, task_description=None, *args, **kwargs):
        if not self.pruner_enabled:
            return PromptAttentionSHRInference.step(
                self, image, contrast_image, task_description, *args, **kwargs
            )
        if not self.cd_enabled:
            return VlaPrunerOpenVLAInference.step(
                self, image, contrast_image, task_description, *args, **kwargs
            )

        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)
        inputs = self.process_inputs(image, task_description=task_description)
        kwargs = dict(kwargs)
        kwargs.pop("proprio", None)

        positive_input_ids = inputs["input_ids"]
        if not torch.all(positive_input_ids[:, -1] == ACTION_TOKEN_PREFIX):
            positive_input_ids = torch.cat([
                positive_input_ids,
                torch.tensor(
                    [[ACTION_TOKEN_PREFIX]], dtype=torch.long, device=positive_input_ids.device
                ),
            ], dim=1)
        history_len_before = len(self.av_hist)

        with projector_intervention(self.vla) as positive_trace:
            positive_score_list, prune_meta = VlaPrunerOpenVLAInference._pruned_generate(
                self, positive_input_ids, inputs["pixel_values"], kwargs
            )
        clean_scores = torch.cat(positive_score_list, dim=0)
        if clean_scores.shape[0] != 7 or positive_trace.before is None:
            raise RuntimeError("P50 positive branch did not return seven action logits")
        visual_keep = prune_meta.get("kept_visual_token_ids")
        if visual_keep is None or len(visual_keep) not in (128, 256):
            raise RuntimeError(f"invalid P50 positive keep set: {visual_keep}")
        expected_history_len = min(history_len_before + 1, self.av_hist.maxlen)
        if len(self.av_hist) != expected_history_len:
            raise RuntimeError("P50 temporal history did not update exactly once")

        visual = positive_trace.before
        features = visual[0].numpy().astype(np.float32)
        attention_scores, attention_meta = extract_prompt_attention(
            self, inputs, task_description, visual, layers=(11,)
        )
        labels, _, per_entity_score = self._semantic_clusters(features)
        entity_groups = self._entity_groups(features, labels)
        reference = sorted(set(
            int(index) for group in entity_groups for index in np.flatnonzero(labels == group)
        ))
        if not reference:
            raise RuntimeError("Prompt-CD reference selector returned an empty region")
        selected = stable_top_m(attention_scores, len(reference))

        reconstructed = features.copy()
        reconstructed[selected] = harmonic_reconstruct(
            features, np.asarray(selected, dtype=np.int64), beta=0.0
        )
        keep_set = set(visual_keep)
        retained = sorted(set(selected) & keep_set)
        removed = sorted(set(selected) - keep_set)
        clean_token_ids = clean_scores.argmax(dim=-1)
        negative_scores, negative_meta = fixed_keep_guided_scores(
            self,
            inputs,
            clean_token_ids,
            torch.from_numpy(reconstructed).unsqueeze(0),
            visual_keep,
        )

        final_scores, guidance_meta = self._combine_action_scores(clean_scores, negative_scores)
        eos_id = int(self.vla.generation_config.eos_token_id)
        invalid = ~torch.isfinite(final_scores)
        if invalid.any():
            unexpected = invalid.clone()
            unexpected[:, eos_id] = False
            if unexpected.any():
                raise FloatingPointError("non-finite non-EOS combined logits")
            final_scores = final_scores.clone()
            final_scores[:, eos_id] = torch.finfo(final_scores.dtype).min
        token_ids = final_scores.argmax(dim=-1)
        raw = self._decode_actions(token_ids, self.unnorm_key)[None]
        raw_action, actions = self.postprocess_actions(raw)

        perturbation = reconstructed - features
        outside = np.asarray([index for index in range(N_VISUAL) if index not in set(selected)])
        action_start = int(self.vla.vocab_size) - ACTION_VOCAB_SIZE
        centered = (
            clean_scores[:6, action_start:action_start + ACTION_VOCAB_SIZE].float()
            - negative_scores[:6, action_start:action_start + ACTION_VOCAB_SIZE].float()
        )
        centered -= centered.mean(dim=-1, keepdim=True)
        clean_decoded = self._decode_actions(clean_token_ids, self.unnorm_key)
        guided_decoded = self._decode_actions(token_ids, self.unnorm_key)
        component_count, isolated_ratio = mask_spatial_stats(selected)
        meta = {
            "arm": "p50_prompt_cd",
            "selection_mode": "prompt_attention",
            "attention_layers": [11],
            "instruction": task_description,
            "selected_entities": list(self._entities),
            "selected_group_ids": entity_groups,
            "per_entity_score": per_entity_score,
            "selected_token_ids": selected,
            "reference_shr_token_ids": reference,
            "m_t": len(selected),
            "coverage_exact": len(selected) == len(reference),
            "coverage_mode": "own_state_standard_shr_matched",
            "mask_component_count": component_count,
            "isolated_token_ratio": isolated_ratio,
            "p50_keep_token_ids": sorted(visual_keep),
            "p50_keep_count": len(visual_keep),
            "g_intersect_k_token_ids": retained,
            "g_minus_k_token_ids": removed,
            "g_intersect_k_count": len(retained),
            "g_minus_k_count": len(removed),
            "g_retention_ratio": len(retained) / len(selected),
            "history_len_before": history_len_before,
            "history_len_after": len(self.av_hist),
            "history_updated_once": len(self.av_hist) == expected_history_len,
            "fastv_r_effective": prune_meta.get("fastv_r_effective"),
            "pruning_layer": prune_meta.get("pruning_layer"),
            "feature_equal": True,
            "non_target_bit_identical": bool(np.array_equal(reconstructed[outside], features[outside])),
            "reconstruction_finite": bool(np.isfinite(reconstructed).all()),
            "feature_perturbation_norm": float(np.linalg.norm(perturbation)),
            "centered_logit_residual_norm": float(torch.linalg.vector_norm(centered).item()),
            "lambda": float(self.lambd),
            "guided_prefix": True,
            "positive_token_ids": clean_token_ids.detach().cpu().tolist(),
            "negative_token_ids": negative_scores.argmax(dim=-1).detach().cpu().tolist(),
            "final_token_ids": token_ids.detach().cpu().tolist(),
            "guided_changed_dims": int((token_ids[:6] != clean_token_ids[:6]).sum().item()),
            "clean_action": np.asarray(clean_decoded).tolist(),
            "guided_action": np.asarray(guided_decoded).tolist(),
            **attention_meta,
            **prune_meta,
            **negative_meta,
            **guidance_meta,
        }
        record = {
            "positive": _action_logits(self, clean_scores),
            "negative": _action_logits(self, negative_scores),
            "selected_mask": np.isin(np.arange(N_VISUAL), selected).astype(np.uint8),
            "p50_keep_mask": np.isin(np.arange(N_VISUAL), visual_keep).astype(np.uint8),
            "g_intersect_k_mask": np.isin(np.arange(N_VISUAL), retained).astype(np.uint8),
            "prompt_attention": attention_scores.astype(np.float32),
        }
        self._episode_logits.append(record)
        self._episode_trace.append(meta)
        self._selector_step += 1
        self._episode_step += 1
        return raw_action, actions, meta


def initialize_p50_prompt_cd(policy, task: str, lambd: float = 0.5) -> None:
    policy.__class__ = P50PromptCDInference
    policy.lambd = float(lambd)
    policy.alpha = float(lambd)
    policy.selector_mode = "prompt_attention"
    policy.attention_layers = (11,)
    policy.beta = 0.0
    policy.kmeans_K = 8
    policy.kmeans_seed = 0
    policy._selector_instr = None
    policy._entities = []
    policy._entity_emb = []
    policy._emb_cache = {}
    policy._episode_seed = 0
    policy._selector_step = 0
    policy.task = task
    policy.fastv_cfg = {
        "fastv_r": 0.5,
        "use_temporal": True,
        "av_hist_w": 3,
        "av_decay": 0.8,
    }
    policy.av_hist = deque(maxlen=3)
    policy.av_decay = 0.8
    policy.pruner_enabled = True
    policy.cd_enabled = True
    policy._episode_trace = []
    policy._episode_logits = []
    policy._episode_step = 0
