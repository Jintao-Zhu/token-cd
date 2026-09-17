"""DTP paper-based autoregressive adapter; does not use SHR or harmonic CD.

Each dimension probes attention on its current refined prefix. A second pass
blocks distracting visual keys and regenerates that token. Past cache is
cloned for each branch so the probe never contaminates the refined prefix.
Earlier cached states retain the masks applied when they were produced.
This cache convention is explicit because the paper does not specify it.
"""
from contextlib import contextmanager
import copy
import numpy as np
import torch
from research.semantic_token_cd.dtp_paper_calibration import action_pattern,spatial,top
from research.ar_token_counterfactual.intervention import ensure_empty_action_token


@contextmanager
def attention_access(model,blocked=()):
    rows=[None]*32
    handles=[]
    def read(i):
        def hook(module,args,output):
            rows[i]=output[1][0,:,-1,1:257].detach().float().mean(0).cpu().numpy()
        return hook
    def mask(module,args,kwargs):
        hidden=kwargs.get('hidden_states',args[0] if args else None)
        original=kwargs.get('attention_mask')
        if original is None:
            raise RuntimeError('Expected explicit causal attention mask for key pruning')
        updated=original.clone()
        updated[:,:,:,torch.as_tensor(blocked,device=updated.device)+1]=torch.finfo(hidden.dtype).min
        kwargs['attention_mask']=updated
        return args,kwargs
    for i,layer in enumerate(model.language_model.model.layers):
        if len(blocked):handles.append(layer.self_attn.register_forward_pre_hook(mask,with_kwargs=True))
        handles.append(layer.self_attn.register_forward_hook(read(i)))
    try:yield rows
    finally:
        for h in handles:h.remove()


def clone_cache(cache):
    if cache is None:return None
    if isinstance(cache,(list,tuple)):
        return tuple(tuple(x.clone() for x in layer) for layer in cache)
    return copy.deepcopy(cache)


@torch.inference_mode()

def _prune_trace(q, protected, blocked, probe_token, refined_token, pattern=None, tau=None):
    """Per-dimension trace row plus pruning-mechanism fields (populated when pattern given)."""
    row = dict(dimension=q, protected=protected.tolist(), pruned=list(blocked),
               probe_token=int(probe_token), refined_token=int(refined_token))
    if pattern is not None:
        unprotected = np.ones(pattern.shape[0], dtype=bool)
        unprotected[protected] = False
        row["a_m"] = float(pattern[protected].max())
        row["tau_times_a_m"] = float(tau * row["a_m"]) if tau is not None else float(row["a_m"])
        row["max_A_unimportant"] = float(pattern[unprotected].max()) if unprotected.any() else 0.0
        row["sum_A_pruned"] = float(pattern[blocked].sum()) if blocked else 0.0
    return row


@torch.inference_mode()
def decode_dtp(policy, inputs, prompt_scores, layer=11, k=109, tau=1., enabled=True):
    """Dynamic per-dimension DTP (v1): each dimension probes on its refined prefix."""
    model = policy.vla
    ids, mask = ensure_empty_action_token(inputs['input_ids'], inputs['attention_mask'])
    protected = top(spatial(prompt_scores[layer], 'both'), k)
    past = None
    tokens = []
    traces = []
    probe_logits = []
    final_logits = []
    start = int(model.vocab_size) - 256

    def run(current, cache, blocked=()):
        with attention_access(model, blocked) as attention:
            result = model(
                input_ids=current,
                attention_mask=mask if cache is None else None,
                pixel_values=inputs['pixel_values'] if cache is None else None,
                past_key_values=clone_cache(cache),
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )
        score = result.logits[0, -1].float().clone()
        score[int(model.generation_config.eos_token_id)] = -torch.inf
        return result, score, np.stack(attention)

    current = ids
    for q in range(7):
        probe, score, a = run(current, past)
        mass = a.sum(-1)
        weights = mass / mass.sum()
        pattern = (weights[:, None] * a).sum(0)
        candidates = np.flatnonzero(pattern > tau * pattern[protected].max())
        blocked = sorted(set(candidates.tolist()) - set(protected.tolist())) if enabled else []
        probe_logits.append(score[start:start + 256].cpu().numpy())
        if blocked:
            result, refined, masked_attention = run(current, past, blocked)
            if float(np.abs(masked_attention[:, blocked]).max()) > 1e-7:
                raise RuntimeError('pruned visual keys retain attention')
        else:
            result, refined = probe, score
        token = int(refined.argmax())
        if not start <= token < start + 256:
            raise RuntimeError(f'illegal action token {token}')
        tokens.append(token)
        final_logits.append(refined[start:start + 256].cpu().numpy())
        traces.append(_prune_trace(q, protected, blocked, int(score.argmax()), token,
                                   pattern, tau))
        past = result.past_key_values
        current = torch.tensor([[token]], device=ids.device, dtype=ids.dtype)
        del probe, result
    return dict(tokens=tokens, trace=traces, probe_logits=np.stack(probe_logits),
                final_logits=np.stack(final_logits))


@torch.inference_mode()
def decode_dtp_fixed(policy, inputs, prompt_scores, layer=11, k=109, tau=0.5, enabled=True):
    """Paper Appendix A.2 SpatialVLA-style path (v2).

    Detection happens on the model's ORIGINAL (clean) action generation: the
    pruning mask D is built once from the first action token's weighted visual
    attention pattern and then kept FIXED for the whole 7-dimension step.  The
    step is re-decoded from a masked prefill under that single D (all tokens,
    including cached ones, then see a consistent visual context F(V\\D)).

    Choices not pinned by the paper and used here:
      * D is built from the clean first-token pattern vs the relevance
        protected set (Eq.5, corner+Gaussian spatial bias applied first);
      * the whole step (all 7 dims) is regenerated under D, starting from a
        full masked prefill, rather than keeping clean dims 1..6;
      * relevance still uses a single calibrated layer (prompt_scores[layer]).
    """
    model = policy.vla
    ids, mask = ensure_empty_action_token(inputs["input_ids"], inputs["attention_mask"])
    protected = top(spatial(prompt_scores[layer], "both"), k)
    start = int(model.vocab_size) - 256

    def run(current, cache, blocked=()):
        with attention_access(model, blocked) as attention:
            result = model(
                input_ids=current,
                attention_mask=mask if cache is None else None,
                pixel_values=inputs["pixel_values"] if cache is None else None,
                past_key_values=clone_cache(cache),
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )
        score = result.logits[0, -1].float().clone()
        score[int(model.generation_config.eos_token_id)] = -torch.inf
        return result, score, np.stack(attention)

    # (1) original clean generation; record first-dimension attention pattern.
    current = ids
    past = None
    clean_tokens = []
    clean_full_logits = []
    dim0_attention = None
    for q in range(7):
        result, score, a = run(current, past)
        if q == 0:
            dim0_attention = a
        token = int(score.argmax())
        if not start <= token < start + 256:
            raise RuntimeError(f"illegal action token {token}")
        clean_tokens.append(token)
        clean_full_logits.append(score)
        past = result.past_key_values
        current = torch.tensor([[token]], device=ids.device, dtype=ids.dtype)
        del result

    if not enabled or dim0_attention is None:
        traces = [_prune_trace(q, protected, [], clean_tokens[q], clean_tokens[q])
                  for q in range(7)]
        return dict(tokens=clean_tokens, trace=traces,
                    probe_logits=np.stack([np.zeros(256, dtype=np.float32)] * 7),
                    final_logits=np.stack([np.zeros(256, dtype=np.float32)] * 7),
                    clean_full_logits=torch.stack(clean_full_logits),
                    final_full_logits=torch.stack(clean_full_logits))

    mass = dim0_attention.sum(-1)
    weights = mass / mass.sum()
    pattern = (weights[:, None] * dim0_attention).sum(0)
    candidates = np.flatnonzero(pattern > tau * pattern[protected].max())
    blocked = sorted(set(candidates.tolist()) - set(protected.tolist()))

    if not blocked:
        traces = [_prune_trace(q, protected, [], clean_tokens[q], clean_tokens[q],
                               pattern if q == 0 else None, tau if q == 0 else None)
                  for q in range(7)]
        return dict(tokens=clean_tokens, trace=traces,
                    probe_logits=np.stack([np.zeros(256, dtype=np.float32)] * 7),
                    final_logits=np.stack([np.zeros(256, dtype=np.float32)] * 7),
                    clean_full_logits=torch.stack(clean_full_logits),
                    final_full_logits=torch.stack(clean_full_logits))

    # (2) re-decode the whole step under the single fixed mask D.
    current = ids
    past = None
    refined = []
    final_logits = []
    final_full_logits = []
    for q in range(7):
        result, score, masked_a = run(current, past, blocked)
        if float(np.abs(masked_a[:, blocked]).max()) > 1e-7:
            raise RuntimeError("pruned visual keys retain attention")
        token = int(score.argmax())
        if not start <= token < start + 256:
            raise RuntimeError(f"illegal action token {token}")
        refined.append(token)
        final_logits.append(score[start:start + 256].cpu().numpy())
        final_full_logits.append(score)
        past = result.past_key_values
        current = torch.tensor([[token]], device=ids.device, dtype=ids.dtype)
        del result
    traces = [_prune_trace(q, protected, blocked, clean_tokens[q], refined[q],
                           pattern if q == 0 else None, tau if q == 0 else None)
              for q in range(7)]
    return dict(tokens=refined, trace=traces,
                probe_logits=np.stack([np.zeros(256, dtype=np.float32)] * 7),
                final_logits=np.stack(final_logits),
                clean_full_logits=torch.stack(clean_full_logits),
                final_full_logits=torch.stack(final_full_logits))
