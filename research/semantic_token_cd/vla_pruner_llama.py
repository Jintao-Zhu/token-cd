"""Scheme-B port of the VLA-Pruner / FastV Llama pruning semantics onto HF
transformers 4.40.1 (the version installed in the openvla-ar-h100 venv).

Reference (official): MINT-SJTU/VLA-Pruner commit 84d4b71
  - LlamaModel.fastv_forward in
    src/openvla/transformers/src/transformers/models/llama/modeling_llama.py
  - Pruning happens exactly once, at the prefill forward of an environment
    step, on the layer-`k` boundary, and performs a REAL deletion of the
    hidden states (`hidden_states = hidden_states[:, keep, :]`), never an
    attention-mask blocking.

Semantics kept identical to the official implementation:
  * num_keep = round(256 * (1 - fastv_r)); fastv_r is a REMOVAL ratio.
  * decision is made with the attention of layer k-1:
      - fastv criterion    : last row (disabled here)
      - prefill criterion  : mean over all query rows (paper main method)
      - text-vision        : SparseVLM rows (disabled here)
  * temporal guide: guide_topk U current_topk -> cosine-redundancy filter
    back to num_keep using current-step image embeddings.
  * keep set = [0] + kept image tokens (+1..+256) + all tokens after the
    image block; positions are the ORIGINAL indices (RoPE preserved);
    cache_position is truncated to the new length.
  * decode steps are left untouched (standard LlamaModel.forward); the KV
    cache simply contains fewer keys for layers >= k.

Differences that do not change semantics (recorded in IMPLEMENTATION_GAP.md):
  * official calls LlamaForCausalLM.fastv_forward only for the multimodal
    prefill; we reach the same point by swapping the LlamaModel instance
    class and overriding `forward`. When the adapter is disabled, or on any
    decode (single-token) call, we delegate verbatim to LlamaModel.forward,
    so a disabled adapter is bit-exact vanilla.
  * transformers 4.40.1 `_update_causal_mask` takes an extra cache_position
    argument; we pass the truncated cache position sequence as in official.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import LlamaModel

DEFAULT_FASTV_CONFIG: Dict[str, Any] = {
    "fastv_k": 3,
    "fastv_r": 0.25,
    "image_token_start_index": 1,
    "image_token_length": 256,
    "historical_attention": None,
    "use_temporal": True,
    "use_text_vision_selection": False,
    "use_prefil_attention": True,
    "SparseVLM": False,
}


def _redundancy_minimization(visual_features: torch.Tensor, num_keep: int) -> torch.Tensor:
    """Official `_redundancy_minization` (cosine-dissimilarity greedy pick).

    visual_features: [N, D] embeddings of the candidate image tokens.
    Returns long indices of shape [num_keep] into the candidate set.
    """
    if len(visual_features) <= num_keep:
        return torch.arange(len(visual_features), device=visual_features.device)
    normed = visual_features / visual_features.norm(dim=1, keepdim=True)
    cosine = torch.mm(normed, normed.t())
    cosine_matrix = 1.0 - cosine
    s = torch.empty(num_keep, dtype=torch.long, device=visual_features.device)
    for i in range(num_keep):
        if i == 0:
            m2 = cosine_matrix
            scores = torch.topk(m2, 2, dim=0, largest=False).values[1, :]
        else:
            m2 = torch.index_select(
                cosine_matrix, 0, torch.index_select(s, 0, torch.arange(0, i, device=cosine_matrix.device))
            )
            scores = torch.min(m2, dim=0).values
        phrase_to_add_idx = torch.argmax(scores)
        s[i] = phrase_to_add_idx
    return s


def _select_keep_indices(
    layer_k_minus_1_attention: torch.Tensor,  # [heads, seq, seq] already head-averaged by caller? keep raw heads here
    hidden_inputs_embeds: torch.Tensor,       # [1, seq, D] original prefill embeddings
    cfg: Dict[str, Any],
    seq_length_with_past: int,
) -> Dict[str, Any]:
    """Returns dict {kept_indices, pruned_indices, num_keep, pruning_layer}."""
    cfg = dict(DEFAULT_FASTV_CONFIG, **cfg)
    start = int(cfg["image_token_start_index"])
    length = int(cfg["image_token_length"])
    r = float(cfg["fastv_r"])
    k = int(cfg["fastv_k"])

    last_layer_attention = layer_k_minus_1_attention
    # average over heads
    last_layer_attention_avg = torch.mean(last_layer_attention, dim=0)  # [seq, seq]
    last_layer_attention_fastv = last_layer_attention_avg[-1]
    last_layer_attention_prefill = last_layer_attention_avg.mean(dim=0)

    # current-step selection vector (row dimension), then visual slice
    if cfg.get("SparseVLM"):
        raise NotImplementedError("SparseVLM path is not part of this reproduction.")
    if cfg["use_text_vision_selection"]:
        raise NotImplementedError("use_text_vision_selection is False in the locked config; not ported.")
    if cfg["use_prefil_attention"]:
        last_layer_attention_avg_last_tok = last_layer_attention_prefill
    else:
        last_layer_attention_avg_last_tok = last_layer_attention_fastv

    image_end = start + length
    available = min(image_end, seq_length_with_past)
    image_vec = last_layer_attention_avg_last_tok[start:available]

    fixed_visual_keep = cfg.get("fixed_visual_keep")
    if fixed_visual_keep is not None:
        fixed_visual_keep = torch.as_tensor(
            fixed_visual_keep, dtype=torch.long, device=image_vec.device
        )
        if fixed_visual_keep.ndim != 1:
            raise ValueError("fixed_visual_keep must be one-dimensional")
        if fixed_visual_keep.numel() == 0 or fixed_visual_keep.unique().numel() != fixed_visual_keep.numel():
            raise ValueError("fixed_visual_keep must be non-empty and unique")
        if int(fixed_visual_keep.min()) < 0 or int(fixed_visual_keep.max()) >= length:
            raise ValueError("fixed_visual_keep contains an invalid visual-token index")
        top_attention_rank_index = fixed_visual_keep + start
        keep_indexs = torch.cat(
            (
                torch.arange(start, device=image_vec.device),
                top_attention_rank_index,
                torch.arange(available, seq_length_with_past, device=image_vec.device),
            )
        ).sort().values
        pruned_indices = torch.arange(seq_length_with_past, device=image_vec.device)
        pruned_indices = pruned_indices[~torch.isin(pruned_indices, keep_indexs)]
        return {
            "kept_indices": keep_indexs,
            "pruned_indices": pruned_indices,
            "num_keep": int(fixed_visual_keep.numel()),
            "pruning_layer": k,
            "guide_topk_overlap": None,
            "used_redundancy": False,
            "used_fixed_visual_keep": True,
        }

    num_keep = round(length * (1 - r))
    hist = cfg.get("historical_attention")
    diag = {"guide_topk_overlap": None, "used_redundancy": False}
    if cfg["use_temporal"] and hist is not None:
        guide_topk = hist.topk(num_keep).indices
        current_topk = image_vec.topk(num_keep).indices
        diag["guide_topk_overlap"] = int(
            len(set(guide_topk.tolist()) & set(current_topk.tolist()))
        )
        combined = torch.cat([guide_topk, current_topk])
        unique_indices = torch.unique(combined)
        if len(unique_indices) > num_keep:
            visual_features = hidden_inputs_embeds[0, start : start + length]
            selected_features = visual_features[unique_indices]
            final_indices = _redundancy_minimization(selected_features, num_keep)
            final_topk = unique_indices[final_indices]
            diag["used_redundancy"] = True
        else:
            final_topk = unique_indices
        top_attention_rank_index = final_topk + start
    else:
        top_attention_rank_index = image_vec.topk(num_keep).indices + start

    keep_indexs = torch.cat(
        (
            torch.arange(start, device=image_vec.device),
            top_attention_rank_index,
            torch.arange(available, seq_length_with_past, device=image_vec.device),
        )
    )
    keep_indexs = keep_indexs.sort().values
    pruned_indices = torch.arange(seq_length_with_past, device=image_vec.device)
    pruned_indices = pruned_indices[~torch.isin(pruned_indices, keep_indexs)]
    return {
        "kept_indices": keep_indexs,
        "pruned_indices": pruned_indices,
        "num_keep": num_keep,
        "pruning_layer": k,
        "guide_topk_overlap": diag["guide_topk_overlap"],
        "used_redundancy": diag["used_redundancy"],
        "used_fixed_visual_keep": False,
    }


class FastVLlamaModel(LlamaModel):
    """LlamaModel whose forward optionally prunes vision tokens during prefill.

    Instance attributes (set by vla_pruner_policy):
      fastv_cfg: dict or None. When None/falsy `enabled`, forward delegates to
                 the untouched LlamaModel.forward (bit-exact vanilla).
      pruning_info: last prefill pruning record (kept/pruned/num_keep/layer).
    """

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        cfg = getattr(self, "fastv_cfg", None)
        enabled = bool(cfg is not None and cfg.get("enabled", True))

        # Resolve flags the same way the vanilla forward does so that the
        # delegated decode path is identical.
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # --- Only the multimodal prefill of an env step can prune.
        # generate() may hand LlamaModel either None, an initially empty
        # DynamicCache, or a non-empty cache (decode). A true decode step has
        # past tokens; the fresh prefill does not.
        is_decode = False
        if use_cache:
            if isinstance(past_key_values, Cache):
                is_decode = past_key_values.get_seq_length() > 0
            elif past_key_values is not None:
                is_decode = len(past_key_values) > 0
        if not enabled or not use_cache or is_decode:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # seq_len > 1 check must happen on the *current* call length.
        if inputs_embeds.shape[1] == 1:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )

        if not output_attentions:
            raise RuntimeError(
                "FastV/VLA-Pruner branch requires output_attentions=True so that the "
                "layer-(k-1) attention can be used for the pruning decision."
            )

        # The pruning path is only reached for a fresh multimodal prefill
        # (use_cache=True, past_key_values=None), so past_seen_tokens == 0.
        if use_cache:
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_seen_tokens = past_key_values.get_seq_length()
        else:
            past_seen_tokens = 0
        # cache_position default (mirrors vanilla forward)
        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_seen_tokens)

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        seq_length_with_past = past_seen_tokens + inputs_embeds.shape[1]

        k = int(cfg.get("fastv_k", 3))
        self.pruning_info = {
            "original_seq_length": seq_length_with_past,
            "kept_indices": None,
            "pruned_indices": None,
            "num_keep": None,
            "pruning_layer": None,
        }
        prev_layer_attn = None
        prune_done = False
        pruned_mask = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if not prune_done and layer_idx == k:
                # Decide on layer k-1 attention (recorded by previous iteration)
                if prev_layer_attn is None:
                    raise RuntimeError(f"fastv pruning at layer {k} needs layer {k - 1} attention")
                sel = _select_keep_indices(
                    prev_layer_attn, inputs_embeds, cfg, seq_length_with_past
                )
                keep = sel["kept_indices"]
                new_seq_length = keep.shape[0]
                self.pruning_info = {
                    "original_seq_length": seq_length_with_past,
                    "kept_indices": keep,
                    "pruned_indices": sel["pruned_indices"],
                    "num_keep": sel["num_keep"],
                    "pruning_layer": sel["pruning_layer"],
                    "guide_topk_overlap": sel.get("guide_topk_overlap"),
                    "used_redundancy": sel.get("used_redundancy", False),
                    "used_fixed_visual_keep": sel.get("used_fixed_visual_keep", False),
                }
                hidden_states = hidden_states[:, keep, :]
                position_ids = keep.unsqueeze(0)
                cache_position = cache_position[:new_seq_length]
                seq_length_with_past = new_seq_length
                mask_arg = None if attention_mask is None else attention_mask[:, keep]
                pruned_mask = self._update_causal_mask(mask_arg, hidden_states, cache_position, 0)
                prune_done = True
            elif pruned_mask is not None:
                # after pruning has happened, all remaining layers must keep
                # using the pruned-sequence causal mask (not the full one).
                pass
            else:
                pruned_mask = causal_mask

            layer_mask = pruned_mask if prune_done else causal_mask

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                prev_layer_attn = layer_outputs[1][0]  # [heads, q_len, kv_len]
                all_self_attns += (layer_outputs[1],)
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if isinstance(next_decoder_cache, Cache)
                else next_decoder_cache
            )
        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


def attach_fastv(model_vla, fastv_cfg: Dict[str, Any]) -> None:
    """Enable scheme-B pruning on a loaded OpenVLA model (shared instance)."""
    llm = model_vla.language_model
    if not isinstance(llm.model, FastVLlamaModel):
        llm.model.__class__ = FastVLlamaModel
    merged = dict(DEFAULT_FASTV_CONFIG)
    merged.update(fastv_cfg)
    merged["enabled"] = True
    llm.model.fastv_cfg = merged
    llm.model.pruning_info = None


def detach_fastv(model_vla) -> None:
    """Restore pure-vanilla behavior on the shared instance."""
    llm = model_vla.language_model
    if isinstance(llm.model, FastVLlamaModel):
        llm.model.fastv_cfg = None
        llm.model.pruning_info = None
