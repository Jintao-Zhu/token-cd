"""AMCD Phase-0 shared infrastructure (logit-lens readout, action token mapping).

Reads intermediate hidden states h_l (after transformer block l, 0-index, 0..31)
through the model's OWN final RMSNorm + LM head::

    z_l(j) = lm.lm_head(decoder.norm(h_l(j)))

No trained probe, no weak checkpoint, no attention masking. The only variable
is transformer depth. Bit-exact with the official OpenVLA no-cache forward
(validated by the implementation sentinel in sentinel.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

UNNORM_KEY = "libero_object"
VOCAB_SIZE = 32000            # == model.vocab_size (32064 - pad_to_multiple_of 64)
ACTION_LO = VOCAB_SIZE - 256  # 31744
ACTION_HI = VOCAB_SIZE        # 32000 (exclusive)
N_ACTION_BINS = 256
BINS = np.linspace(-1.0, 1.0, 256)


def build_multimodal(model, input_ids, attention_mask, pixel_values):
    """Replicate Prismatic forward's multimodal embedding build exactly."""
    patch_features = model.vision_backbone(pixel_values)
    projected = model.projector(patch_features)
    input_embeds = model.get_input_embeddings()(input_ids)
    n_visual = projected.shape[1]
    mm_embeds = torch.cat([input_embeds[:, :1], projected, input_embeds[:, 1:]], dim=1)
    mm_mask = torch.cat(
        [attention_mask[:, :1],
         torch.full((1, n_visual), True, device=attention_mask.device, dtype=attention_mask.dtype),
         attention_mask[:, 1:]], dim=1)
    return mm_embeds, mm_mask, n_visual


@torch.inference_mode()
def layer_loop_with_hidden(model, inputs_embeds, attention_mask_2d):
    """Layer-by-layer no-cache forward, collecting h_l after EVERY block (0..31).

    Returns (hiddens, final_logits) where hiddens[l] is the hidden state AFTER
    transformer block l (pre-final-norm), and final_logits = lm_head(norm(h_31)).
    """
    lm = model.language_model
    decoder = lm.model
    seq_len = inputs_embeds.shape[1]
    cache_position = torch.arange(0, seq_len, device=inputs_embeds.device)
    position_ids = cache_position.unsqueeze(0)
    causal_mask = decoder._update_causal_mask(attention_mask_2d, inputs_embeds, cache_position, 0)

    hidden_states = inputs_embeds
    hiddens = []
    for layer in decoder.layers:
        layer_outputs = layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=cache_position,
        )
        hidden_states = layer_outputs[0]
        hiddens.append(hidden_states)
    final_logits = lm.lm_head(decoder.norm(hidden_states))
    return hiddens, final_logits


@torch.inference_mode()
def readout_layer(model, hidden_states):
    """z_l = lm_head(final_norm(h_l)). hidden_states: [1, seq, 4096] pre-final-norm."""
    lm = model.language_model
    decoder = lm.model
    return lm.lm_head(decoder.norm(hidden_states))


@torch.inference_mode()
def scan_layer_readouts(model, inputs_embeds, attention_mask_2d, aq):
    """Full no-cache forward with inline logit-lens readout at every block.

    For each block l (0..31): read h_l through final_norm + lm_head, slice the
    action vocab at the action-query positions `aq`, and record the full-vocab
    log-sum-exp. Returns (z_action, logZ):
      z_action: [32, len(aq), 256] float32 numpy (action-vocab logits)
      logZ:     [32, len(aq)]      float32 numpy (full-vocab logsumexp per position)
    """
    lm = model.language_model
    decoder = lm.model
    seq_len = inputs_embeds.shape[1]
    cache_position = torch.arange(0, seq_len, device=inputs_embeds.device)
    position_ids = cache_position.unsqueeze(0)
    causal_mask = decoder._update_causal_mask(attention_mask_2d, inputs_embeds, cache_position, 0)

    hidden_states = inputs_embeds
    z_list, logz_list = [], []
    for layer in decoder.layers:
        layer_outputs = layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=cache_position,
        )
        hidden_states = layer_outputs[0]
        full = lm.lm_head(decoder.norm(hidden_states))      # [1, seq, 32000]
        faq = full[0, aq].float()                            # [len(aq), 32000]
        z_list.append(faq[:, ACTION_LO:ACTION_HI].cpu().numpy())
        logz_list.append(torch.logsumexp(faq, dim=-1).cpu().numpy())
    return np.stack(z_list), np.stack(logz_list)


@torch.inference_mode()
def readout_specific_layers(model, inputs_embeds, attention_mask_2d, aq, layers):
    """Like scan_layer_readouts but only read out the given block indices (0..31).

    Returns ({l: z_action[l]}, {l: logZ[l]}) as dicts keyed by layer index, where
    z_action[l] is [len(aq), 256] float32 numpy and logZ[l] is [len(aq)] float32.
    """
    lm = model.language_model
    decoder = lm.model
    seq_len = inputs_embeds.shape[1]
    cache_position = torch.arange(0, seq_len, device=inputs_embeds.device)
    position_ids = cache_position.unsqueeze(0)
    causal_mask = decoder._update_causal_mask(attention_mask_2d, inputs_embeds, cache_position, 0)
    want = set(layers)

    hidden_states = inputs_embeds
    z_dict, logz_dict = {}, {}
    for idx, layer in enumerate(decoder.layers):
        layer_outputs = layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=cache_position,
        )
        hidden_states = layer_outputs[0]
        if idx in want:
            full = lm.lm_head(decoder.norm(hidden_states))  # [1, seq, 32000]
            faq = full[0, aq].float()
            z_dict[idx] = faq[:, ACTION_LO:ACTION_HI].cpu().numpy()
            logz_dict[idx] = torch.logsumexp(faq, dim=-1).cpu().numpy()
    return z_dict, logz_dict


def load_action_stats(ckpt_dir: Path):
    stats = json.loads((ckpt_dir / "dataset_statistics.json").read_text())[UNNORM_KEY]["action"]
    low = np.asarray(stats["q01"], dtype=np.float64)
    high = np.asarray(stats["q99"], dtype=np.float64)
    mask = np.asarray(stats["mask"], dtype=bool)
    return low, high, mask


def encode_action_tokens(raw, low, high, mask):
    """raw HDF5 action (7 dims, gripper in [0,1]) -> 7 token ids in [31744, 31999]."""
    proc = np.asarray(raw, dtype=np.float64).copy()
    proc[6] = 1.0 - np.clip(proc[6], 0.0, 1.0)  # LIBERO gripper transform
    norm = np.where(mask, 2.0 * (proc - low) / (high - low + 1e-8) - 1.0, proc)
    norm = np.clip(norm, -1.0, 1.0)
    disc = np.digitize(norm, BINS)
    return VOCAB_SIZE - disc  # [31744, 31999]


def transformed_action(raw):
    """The exact 7-dim action vector that expert tokens encode (pre-normalization)."""
    proc = np.asarray(raw, dtype=np.float64).copy()
    proc[6] = 1.0 - np.clip(proc[6], 0.0, 1.0)
    return proc


def decode_tokens(model, tokens, low, high, mask):
    """tokens [7] in [31744, 31999] -> 7-dim continuous action (gripper dim = transformed)."""
    tokens = np.asarray(tokens, dtype=np.int64)
    disc = model.vocab_size - tokens
    disc = np.clip(disc - 1, 0, model.bin_centers.shape[0] - 1)
    normalized = model.bin_centers[disc]
    if hasattr(normalized, "detach"):
        normalized = normalized.detach().cpu().numpy()
    normalized = np.asarray(normalized, dtype=np.float64)
    return np.where(mask, 0.5 * (normalized + 1.0) * (high - low) + low, normalized)


def logits_to_action(model, z_a, low, high, mask):
    """z_a: [7, 256] action-vocab logits (numpy float) -> 7-dim continuous action."""
    tok = z_a.argmax(-1) + ACTION_LO
    return decode_tokens(model, tok, low, high, mask)


def action_slice(logits, aq):
    """logits: [1, seq, vocab] bfloat16 -> [len(aq), 256] float32 numpy (action vocab)."""
    return logits[0, aq, ACTION_LO:ACTION_HI].float().cpu().numpy()


def log_softmax_256(z):
    """z: [..., 256] -> log-softmax over the 256 action bins (float64)."""
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return z - np.log(e.sum(axis=-1, keepdims=True))
