from __future__ import annotations

from contextlib import contextmanager

import torch

from research.token_pcd_bridge.token_policy import FrozenTokenPCDInference


@contextmanager
def early_patch_intervention(model, selected_ids, means):
    handles = []
    modules = [model.vision_backbone.featurizer.patch_embed]
    if model.config.use_fused_vision_backbone:
        modules.append(model.vision_backbone.fused_featurizer.patch_embed)
    for branch, module in enumerate(modules):
        def hook(_module, _inputs, output, branch=branch):
            modified = output.clone()
            if selected_ids:
                replacement = means[branch].to(device=output.device, dtype=output.dtype)
                modified[:, selected_ids, :] = replacement[selected_ids]
            return modified
        handles.append(module.register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


class FrozenEarlyTokenPCDInference(FrozenTokenPCDInference):
    """PCD negative branch intervened before vision Transformer self-attention."""

    def __init__(self, replacement_means, **kwargs):
        super().__init__(replacement_mean=None, **kwargs)
        self.replacement_means = replacement_means

    def step(self, image, _unused_contrast_image, task_description=None, *args, **kwargs):
        if self._audit_context is None:
            raise RuntimeError("set_counterfactual must be called at every replan")
        inputs = self.process_inputs(image, task_description=task_description)
        clean_scores = self._forward_scores(inputs, self.unnorm_key, **kwargs)
        with early_patch_intervention(self.vla, self._audit_context["selected_token_ids"], self.replacement_means):
            negative_scores = self._forward_scores(inputs, self.unnorm_key, **kwargs)
        if clean_scores.shape != negative_scores.shape or clean_scores.shape[0] != 7:
            raise RuntimeError(f"Invalid score shapes {clean_scores.shape} {negative_scores.shape}")
        final_scores = clean_scores.clone()
        final_scores[:-1] = 1.8 * clean_scores[:-1] - 0.8 * negative_scores[:-1]
        if not torch.isfinite(final_scores).all():
            raise FloatingPointError("Non-finite Early Token-PCD logits")
        token_ids = final_scores.argmax(dim=-1)
        raw = self._decode_actions(token_ids, self.unnorm_key)[None]
        raw_action, actions = self.postprocess_actions(raw)
        self._audit_calls.append({**self._audit_context,
                                  "changed_indices": list(self._audit_context["selected_token_ids"]),
                                  "clean_token_ids": clean_scores.argmax(-1).detach().cpu().tolist(),
                                  "negative_token_ids": negative_scores.argmax(-1).detach().cpu().tolist(),
                                  "pcd_token_ids": token_ids.detach().cpu().tolist(),
                                  "early_intervention": True, "finite": True})
        self._audit_context = None
        return raw_action, actions, {"final_logits": final_scores}
