"""Task-conditioned contrast with an optional clean-plausibility constraint.

This module is intentionally self-contained so the frozen Prompt-Attn-SHR
policy remains byte-for-byte unchanged.  It extends that policy with two
control branches while preserving the historical L11 mask and clean-prefix
decoding contract.
"""
from __future__ import annotations

import math
import numpy as np
import torch

from research.ar_token_counterfactual.intervention import projector_intervention
from research.semantic_token_cd.global_merge_policy import (
    guided_forward_scores,
    projector_merge_intervention,
)
from research.semantic_token_cd.prompt_attn_shr_policy import (
    EPS,
    N_VISUAL,
    PromptAttentionSHRInference,
    extract_prompt_attention,
    mask_spatial_stats,
)


CONTROL_INSTRUCTIONS = {
    "google_robot_open_drawer": "open a drawer",
    "google_robot_close_drawer": "close a drawer",
    "google_robot_pick_coke_can": "pick up an object",
    "google_robot_move_near": "move an object near another object",
}

CONTROL_INSTRUCTIONS_ALT = {
    "google_robot_open_drawer": "pull open a drawer",
    "google_robot_close_drawer": "push closed a drawer",
    "google_robot_pick_coke_can": "grasp an object",
    "google_robot_move_near": "bring an object close to another object",
}

ARMS = ("C", "C_APC", "T", "T_APC")


def _action_start(policy) -> tuple[int, int]:
    """Return the verified action-token start and bin count.

    OpenVLA's action tokenizer reserves the final ``n_action_bins`` tokens.
    ``bin_centers`` has one fewer entry by construction, so the model config is
    the authoritative source for the action-token vocabulary size.
    """
    bins = int(getattr(policy.vla.config, "n_action_bins", 0))
    if bins <= 0:
        bins = int(policy.vla.bin_centers.shape[0]) + 1
    start = int(policy.vla.vocab_size) - bins
    if bins <= 0 or start < 0:
        raise RuntimeError(f"invalid action tokenizer range: start={start} bins={bins}")
    return start, bins


def _log_softmax_action(scores: torch.Tensor, policy) -> torch.Tensor:
    start, bins = _action_start(policy)
    if scores.shape[-1] < start + bins:
        raise RuntimeError(f"score vocabulary too small: {scores.shape[-1]} < {start + bins}")
    return torch.log_softmax(scores[:, start : start + bins].detach().float(), dim=-1)


def _decode_local(policy, local_ids: torch.Tensor, unnorm_key: str) -> np.ndarray:
    """Decode local action-vocabulary indices to a 7-D normalized action."""
    start, _bins = _action_start(policy)
    global_ids = local_ids.detach().long() + start
    decoded = np.asarray(policy._decode_actions(global_ids, unnorm_key), dtype=np.float64).reshape(-1)
    if decoded.shape != (7,):
        raise RuntimeError(f"decoded action shape mismatch: {decoded.shape}")
    return decoded


def _rms(values: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(values.float() ** 2)).item())


def _centered_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().flatten()
    right = right.float().flatten()
    left = left - left.mean()
    right = right - right.mean()
    return float(torch.dot(left, right).item() / ((torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)).item() + EPS))


def _apc_apply(scores: torch.Tensor, clean_log_probs: torch.Tensor, beta: float) -> tuple[torch.Tensor, dict]:
    if not 0.0 < float(beta) <= 1.0:
        raise ValueError(f"APC beta must be in (0, 1], got {beta}")
    keep = clean_log_probs >= (clean_log_probs.max(dim=-1, keepdim=True).values + math.log(beta))
    constrained = scores.clone()
    constrained[~keep] = -float("inf")
    if not torch.isfinite(constrained[keep]).all():
        raise FloatingPointError("non-finite retained APC scores")
    if not (constrained[~keep] < 0).all():
        raise FloatingPointError("APC masked candidates were not set to negative infinity")
    raw = scores.argmax(dim=-1)
    final = constrained.argmax(dim=-1)
    blocked = final != raw
    clean_top = clean_log_probs.argmax(dim=-1)
    clean_relative = torch.exp(clean_log_probs - clean_log_probs.max(dim=-1, keepdim=True).values)
    # Rank is 1-based, descending.  Ties follow token index exactly as argmax.
    clean_order = torch.argsort(clean_log_probs, dim=-1, descending=True, stable=True)
    ranks = torch.empty_like(clean_order)
    ranks.scatter_(1, clean_order, torch.arange(clean_log_probs.shape[-1], device=clean_log_probs.device).expand_as(clean_order))
    blocked_rank = ranks.gather(1, raw[:, None]).squeeze(1) + 1
    blocked_relative = clean_relative.gather(1, raw[:, None]).squeeze(1)
    return constrained, {
        "keep": keep,
        "blocked": blocked,
        "raw_winner": raw,
        "final_winner": final,
        "clean_top1": clean_top,
        "blocked_clean_rank": blocked_rank,
        "blocked_clean_relative_prob": blocked_relative,
    }


class TaskConditionedContrastInference(PromptAttentionSHRInference):
    """Four-condition task-conditioned residual with optional APC.

    ``arm`` is one of ``C``, ``C_APC``, ``T``, ``T_APC``.  The control
    instruction never changes mask selection or reconstruction; it only
    changes the text condition used for the C0/D0 forward passes.
    """

    arm: str = "C"
    apc_beta: float = 0.1
    control_instruction: str | None = None
    task_key: str | None = None
    save_condition_logits: bool = True
    debug_identity_reconstruction: bool = False
    debug_zero_control_residual: bool = False

    def reset(self, task_description: str, seed=None) -> None:
        super().reset(task_description, seed)
        if self.control_instruction is None:
            if self.task_key is None:
                for key in CONTROL_INSTRUCTIONS:
                    if key.replace("google_robot_", "") in task_description.replace(" ", "_"):
                        self.task_key = key
                        break
            if self.task_key is None:
                raise RuntimeError(f"no registered control instruction for: {task_description}")
            self.control_instruction = CONTROL_INSTRUCTIONS[self.task_key]

    def step(self, image, contrast_image=None, task_description=None, *args, **kwargs):
        if self.arm not in ARMS:
            raise ValueError(f"unknown task-conditioned arm: {self.arm}")
        inputs = self.process_inputs(image, task_description=task_description)
        control_instruction = self.control_instruction or CONTROL_INSTRUCTIONS[self.task_key]
        control_inputs = self.process_inputs(image, task_description=control_instruction)

        with projector_intervention(self.vla) as positive_trace:
            clean_scores = self._forward_scores(inputs, self.unnorm_key, do_sample=False)
        if clean_scores.shape[0] != 7 or positive_trace.before is None:
            raise RuntimeError("A branch did not produce projector features and 7 scores")
        clean_visual = positive_trace.before

        # Reuse the frozen L11 prompt-attention selector exactly.  The mask and
        # m_t are computed once from the original instruction and are shared by
        # all four model conditions.
        attention_scores, selector_meta = extract_prompt_attention(
            self, inputs, task_description, clean_visual, layers=self.attention_layers
        )
        h = clean_visual[0].detach().cpu().numpy().astype(np.float32)
        labels, selected_groups, per_entity_score = self._semantic_clusters(h)
        matched_groups = []
        for group in self._entity_groups(h, labels):
            if int(group) not in matched_groups:
                matched_groups.append(int(group))
        if not matched_groups:
            raise RuntimeError("matched selector produced an empty mask")
        m = len({int(index) for group in matched_groups for index in np.flatnonzero(labels == group)})
        selected = self._stable_top_m(attention_scores, m)
        reconstructed = h.copy()
        if self.debug_identity_reconstruction:
            reconstructed = h.copy()
        else:
            reconstructed[np.asarray(selected, dtype=np.int64)] = self._harmonic_reconstruct(h, np.asarray(selected, dtype=np.int64))
        if not np.isfinite(reconstructed).all():
            raise FloatingPointError("non-finite harmonic reconstruction")

        action_start, action_bins = _action_start(self)
        clean_global_ids = clean_scores.argmax(dim=-1)
        clean_local_ids = clean_scores[:, action_start : action_start + action_bins].argmax(dim=-1)
        if not torch.equal(clean_local_ids[:6] + action_start, clean_global_ids[:6]):
            raise RuntimeError("clean action token outside verified action vocabulary")
        with projector_merge_intervention(self.vla, torch.from_numpy(reconstructed).unsqueeze(0)) as negative_trace:
            recon_original_scores = guided_forward_scores(self.vla, inputs, clean_global_ids, clean_visual.shape[1])
        if negative_trace["before"] is None or not torch.equal(clean_visual, negative_trace["before"]):
            raise RuntimeError("B branch projector input differs from clean features")

        use_task = self.arm in ("T", "T_APC")
        if use_task:
            with projector_intervention(self.vla) as control_clean_trace:
                clean_control_scores = guided_forward_scores(
                    self.vla, control_inputs, clean_global_ids, clean_visual.shape[1]
                )
            if control_clean_trace.before is None or not torch.equal(clean_visual, control_clean_trace.before):
                raise RuntimeError("C0 branch projector features differ from clean features")
            with projector_merge_intervention(self.vla, torch.from_numpy(reconstructed).unsqueeze(0)) as control_recon_trace:
                recon_control_scores = guided_forward_scores(
                    self.vla, control_inputs, clean_global_ids, clean_visual.shape[1]
                )
            if control_recon_trace["before"] is None or not torch.equal(clean_visual, control_recon_trace["before"]):
                raise RuntimeError("D0 branch projector input differs from clean features")
        else:
            clean_control_scores = None
            recon_control_scores = None

        LA = _log_softmax_action(clean_scores, self)
        LB = _log_softmax_action(recon_original_scores, self)
        if use_task:
            LC = _log_softmax_action(clean_control_scores, self)
            LD = _log_softmax_action(recon_control_scores, self)
            S_visual = LA - LB
            S_control = LC - LD
            S_task = S_visual if self.debug_zero_control_residual else S_visual - S_control
        else:
            S_visual = LA - LB
            S_control = None
            S_task = S_visual

        lam = float(self.lambd)
        guided_scores = LA + lam * S_task
        apc_meta = None
        if self.arm in ("C_APC", "T_APC"):
            guided_scores, apc_meta = _apc_apply(guided_scores, LA, float(self.apc_beta))
        # Gripper always follows clean A exactly.  This must be assigned after
        # local argmax, because replacing the local log-prob row with a global
        # token id would change decoding.
        if apc_meta is None:
            if not torch.isfinite(guided_scores).all():
                raise FloatingPointError("non-finite guided scores")
        final_local_ids = guided_scores.argmax(dim=-1)
        # The frozen policy preserves the clean global gripper token exactly.
        final_global_ids = final_local_ids + action_start
        final_global_ids[6] = clean_global_ids[6]
        final_local_ids[6] = clean_local_ids[6]
        clean_action = np.asarray(self._decode_actions(clean_global_ids, self.unnorm_key), dtype=np.float64).reshape(-1)
        final_action = np.asarray(self._decode_actions(final_global_ids, self.unnorm_key), dtype=np.float64).reshape(-1)

        changed_dims = int((final_local_ids[:6] != clean_local_ids[:6]).sum().item())
        keep_counts = None if apc_meta is None else [int(value) for value in apc_meta["keep"].sum(dim=-1).cpu().tolist()]
        blocked_raw = None if apc_meta is None else [int(value) for value in apc_meta["raw_winner"].cpu().tolist()]
        blocked_rank = None if apc_meta is None else [int(value) for value in apc_meta["blocked_clean_rank"].cpu().tolist()]
        blocked_relative = None if apc_meta is None else [float(value) for value in apc_meta["blocked_clean_relative_prob"].cpu().tolist()]
        blocked_mask = None if apc_meta is None else [bool(value) for value in apc_meta["blocked"].cpu().tolist()]
        clean_top1_logprob = [float(value) for value in LA.gather(1, clean_local_ids[:, None]).squeeze(1).cpu().tolist()]
        final_selected_clean_logprob = [float(value) for value in LA.gather(1, final_local_ids[:, None]).squeeze(1).cpu().tolist()]

        raw = self._decode_actions(final_global_ids, self.unnorm_key)[None]
        raw_action, actions = self.postprocess_actions(raw)
        meta = {
            "arm": self.arm,
            "control_instruction": control_instruction,
            "selected_token_ids": selected,
            "m_t": int(m),
            "matched_cluster_ids": [int(value) for value in selected_groups],
            "per_entity_score": [float(value) for value in per_entity_score],
            "reconstruction_changed_token_ids": selected,
            "reconstruction_delta_norm": float(np.linalg.norm(reconstructed - h)),
            "clean_action_tokens": [int(value) for value in clean_global_ids.cpu().tolist()],
            "final_action_tokens": [int(value) for value in final_global_ids.cpu().tolist()],
            "clean_local_action_bins": [int(value) for value in clean_local_ids.cpu().tolist()],
            "final_local_action_bins": [int(value) for value in final_local_ids.cpu().tolist()],
            "executed_action": final_action.tolist(),
            "changed_dims_vs_clean": changed_dims,
            "S_visual_rms": _rms(S_visual),
            "S_control_rms": None if S_control is None else _rms(S_control),
            "S_task_rms": _rms(S_task),
            "cos_visual_task": _centered_cosine(S_visual, S_task),
            "APC_keep_count_per_dim": keep_counts,
            "APC_blocked_raw_guided_winner": blocked_raw,
            "APC_blocked": blocked_mask,
            "blocked_winner_clean_rank": blocked_rank,
            "blocked_winner_clean_relative_prob": blocked_relative,
            "clean_top1_logprob": clean_top1_logprob,
            "final_selected_clean_logprob": final_selected_clean_logprob,
            "lambda": lam,
            "action_vocab_start": int(action_start),
            "action_vocab_size": int(action_bins),
            "apc_beta": float(self.apc_beta) if self.arm in ("C_APC", "T_APC") else None,
            "clean_prefix": "A_greedy_shared",
            "feature_equal": True,
            "control_clean_feature_equal": bool(use_task),
            "control_recon_feature_equal": bool(use_task),
            "attention_meta": selector_meta,
        }
        self._episode_trace.append(meta)
        if self.save_condition_logits:
            self._episode_logits.append({
                "LA": LA.cpu().numpy().astype(np.float32),
                "LB": LB.cpu().numpy().astype(np.float32),
                "LC": None if S_control is None else LC.cpu().numpy().astype(np.float32),
                "LD": None if S_control is None else LD.cpu().numpy().astype(np.float32),
                "S_visual": S_visual.cpu().numpy().astype(np.float32),
                "S_control": None if S_control is None else S_control.cpu().numpy().astype(np.float32),
                "S_task": S_task.cpu().numpy().astype(np.float32),
            })
        return raw_action, actions, meta

    @staticmethod
    def _stable_top_m(scores: np.ndarray, m: int) -> list[int]:
        values = np.asarray(scores, dtype=np.float64)
        if values.shape != (N_VISUAL,) or not np.isfinite(values).all():
            raise ValueError(f"invalid prompt attention scores: {values.shape}")
        if not 1 <= int(m) <= N_VISUAL:
            raise ValueError(f"invalid coverage m={m}")
        order = np.lexsort((np.arange(N_VISUAL), -values))
        return sorted(int(index) for index in order[: int(m)])

    def _harmonic_reconstruct(self, features: np.ndarray, selected: np.ndarray) -> np.ndarray:
        from research.semantic_token_cd.st_shr_policy import harmonic_reconstruct
        return harmonic_reconstruct(features, selected, beta=0.0)
