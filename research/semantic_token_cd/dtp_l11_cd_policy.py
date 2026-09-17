"""DTP-Fixed positive branch contrasted against an unpruned L11 harmonic negative."""
from __future__ import annotations

import numpy as np
import torch

from research.semantic_token_cd.distractor_policy import ACTION_VOCAB_SIZE, _action_logits
from research.semantic_token_cd.dtp_l11_cd_protocol import DTP_K, DTP_LAYER, DTP_TAU
from research.semantic_token_cd.dtp_openvla_policy import decode_dtp_fixed
from research.semantic_token_cd.global_merge_policy import guided_forward_scores, projector_merge_intervention
from research.semantic_token_cd.prompt_attn_shr_policy import (
    N_VISUAL, PromptAttentionSHRInference, extract_prompt_attention_per_layer,
    mask_spatial_stats, stable_top_m,
)
from research.semantic_token_cd.st_shr_policy import harmonic_reconstruct
from research.ar_token_counterfactual.intervention import projector_intervention


class DTPPositiveL11NegativeInference(PromptAttentionSHRInference):
    """Literal heterogeneous CD requested by the protocol.

    Positive logits come from the paper-style fixed DTP mask.  Negative logits
    come from the full (unpruned) visual stream after L11-Matched harmonic
    reconstruction and are teacher-forced on the DTP-positive prefix.
    """

    def step(self, image, contrast_image=None, task_description=None, *args, **kwargs):
        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)
        inputs = self.process_inputs(image, task_description=task_description)

        # Match the audited DTP harness: capture clean projector features, then
        # extract prompt relevance, then run fixed-mask DTP decoding.
        with projector_intervention(self.vla) as visual_trace:
            reference_clean_scores = self._forward_scores(inputs, self.unnorm_key, do_sample=False)
        if reference_clean_scores.shape[0] != 7 or visual_trace.before is None:
            raise RuntimeError("DTP/L11 step did not capture seven clean scores and visual features")
        visual = visual_trace.before
        features = visual[0].numpy().astype(np.float32)
        prompt_per_layer, prompt_meta = extract_prompt_attention_per_layer(
            self, inputs, task_description, visual
        )
        dtp = decode_dtp_fixed(
            self, inputs, prompt_per_layer, layer=DTP_LAYER, k=DTP_K,
            tau=DTP_TAU, enabled=True,
        )
        positive_scores = dtp["final_full_logits"]
        dtp_clean_scores = dtp["clean_full_logits"]
        if positive_scores.shape != reference_clean_scores.shape or positive_scores.shape[0] != 7:
            raise RuntimeError("DTP positive branch returned incomplete logits")
        reference_ids = reference_clean_scores.argmax(dim=-1)
        dtp_clean_ids = dtp_clean_scores.argmax(dim=-1)
        if not torch.equal(reference_ids, dtp_clean_ids):
            raise RuntimeError("DTP internal clean generation differs from audited clean generation")

        # Historical L11-Matched selection: raw L11 prompt ranking, with the
        # own-state KMeans semantic selector supplying only the count m_t.
        labels, _, per_entity_score = self._semantic_clusters(features)
        entity_groups = self._entity_groups(features, labels)
        reference = sorted(set(
            int(index) for group in entity_groups for index in np.flatnonzero(labels == group)
        ))
        if not reference:
            raise RuntimeError("L11-Matched reference selector returned an empty region")
        selected = stable_top_m(prompt_per_layer[DTP_LAYER], len(reference))
        reconstructed = features.copy()
        reconstructed[selected] = harmonic_reconstruct(
            features, np.asarray(selected, dtype=np.int64), beta=0.0
        )

        positive_token_ids = positive_scores.argmax(dim=-1)
        with projector_merge_intervention(
            self.vla, torch.from_numpy(reconstructed).unsqueeze(0)
        ) as negative_trace:
            negative_scores = guided_forward_scores(
                self.vla, inputs, positive_token_ids, visual.shape[1]
            )
        if negative_trace["before"] is None or not torch.equal(visual, negative_trace["before"]):
            raise RuntimeError("L11 negative projector input differs from the clean visual features")

        final_scores, guidance_meta = self._combine_action_scores(positive_scores, negative_scores)
        eos_id = int(self.vla.generation_config.eos_token_id)
        invalid = ~torch.isfinite(final_scores)
        if invalid.any():
            unexpected = invalid.clone()
            unexpected[:, eos_id] = False
            if unexpected.any():
                raise FloatingPointError("non-finite non-EOS DTP/L11 contrastive logits")
            final_scores = final_scores.clone()
            final_scores[:, eos_id] = torch.finfo(final_scores.dtype).min
        token_ids = final_scores.argmax(dim=-1)
        if int(token_ids[-1]) != int(positive_token_ids[-1]):
            raise RuntimeError("gripper dimension differs from the DTP positive branch")
        raw = self._decode_actions(token_ids, self.unnorm_key)[None]
        raw_action, actions = self.postprocess_actions(raw)

        selected_set = set(selected)
        outside = np.asarray([i for i in range(N_VISUAL) if i not in selected_set])
        perturbation = reconstructed - features
        action_start = int(self.vla.vocab_size) - ACTION_VOCAB_SIZE
        centered = (
            positive_scores[:6, action_start:action_start + ACTION_VOCAB_SIZE]
            - negative_scores[:6, action_start:action_start + ACTION_VOCAB_SIZE]
        ).float()
        centered -= centered.mean(dim=-1, keepdim=True)
        components, isolated_ratio = mask_spatial_stats(selected)
        dtp_pruned = dtp["trace"][0]["pruned"] if dtp["trace"] else []
        dtp_triggered = bool(dtp_pruned)
        clean_decoded = self._decode_actions(positive_token_ids, self.unnorm_key)
        guided_decoded = self._decode_actions(token_ids, self.unnorm_key)
        meta = {
            "arm": f"dtp_l11_lambda_{str(float(self.lambd)).replace('.', 'p')}",
            "lambda": float(self.lambd),
            "positive_branch": "DTP-Fixed L11/k64/tau0.5 clean-dim0 single mask",
            "negative_branch": "unpruned L11-Matched harmonic reconstruction beta0",
            "dtp_layer": DTP_LAYER, "dtp_k": DTP_K, "dtp_tau": DTP_TAU,
            "dtp_fixed_mask": True, "dtp_pruned_token_ids": list(dtp_pruned),
            "dtp_pruned_count": len(dtp_pruned), "dtp_triggered": dtp_triggered,
            "dtp_trace": dtp["trace"],
            "attention_layers": [DTP_LAYER],
            "selected_token_ids": selected, "reference_shr_token_ids": reference,
            "selected_group_ids": entity_groups, "per_entity_score": per_entity_score,
            "m_t": len(selected), "coverage_exact": len(selected) == len(reference),
            "coverage_mode": "own_state_standard_shr_matched",
            "mask_component_count": components, "isolated_token_ratio": isolated_ratio,
            "feature_equal": True,
            "non_target_bit_identical": bool(np.array_equal(reconstructed[outside], features[outside])),
            "reconstruction_finite": bool(np.isfinite(reconstructed).all()),
            "feature_perturbation_norm": float(np.linalg.norm(perturbation)),
            "centered_logit_residual_norm": float(torch.linalg.vector_norm(centered).item()),
            "guided_prefix": True, "prefix_source": "DTP positive greedy tokens",
            "positive_token_ids": positive_token_ids.detach().cpu().tolist(),
            "negative_token_ids": negative_scores.argmax(dim=-1).detach().cpu().tolist(),
            "final_token_ids": token_ids.detach().cpu().tolist(),
            "guided_changed_dims": int((token_ids[:6] != positive_token_ids[:6]).sum().item()),
            "gripper_positive_exact": True,
            "dtp_clean_equivalence": True,
            "dtp_positive_action": np.asarray(clean_decoded).tolist(),
            "guided_action": np.asarray(guided_decoded).tolist(),
            **prompt_meta, **guidance_meta,
        }
        self._episode_logits.append({
            "positive": _action_logits(self, positive_scores),
            "negative": _action_logits(self, negative_scores),
            "final": _action_logits(self, final_scores),
            "selected_mask": np.isin(np.arange(N_VISUAL), selected).astype(np.uint8),
            "dtp_pruned_mask": np.isin(np.arange(N_VISUAL), dtp_pruned).astype(np.uint8),
            "prompt_attention_l11": prompt_per_layer[DTP_LAYER].astype(np.float32),
        })
        self._episode_trace.append(meta)
        self._selector_step += 1
        return raw_action, actions, meta


def initialize_dtp_l11_cd(policy, task: str, lambd: float) -> None:
    from research.semantic_token_cd.semantic_recon_rollout import _init_common
    policy.__class__ = DTPPositiveL11NegativeInference
    _init_common(policy, float(lambd))
    policy.beta = 0.0
    policy.selector_mode = "prompt_attention"
    policy.attention_layers = (DTP_LAYER,)
    policy.task_index = task
    policy._episode_trace = []
    policy._episode_logits = []
