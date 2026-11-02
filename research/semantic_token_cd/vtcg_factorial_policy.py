"""Independent nine-arm VTCG factorial policy. Never patches shared modules.

Runner interface: build_policy(base, task, arm, fixed_m=48, top_p=.85,
                              lambd=.5), reset(instruction, seed), step(...).
Only frozen model/processor are shared; every mutable policy buffer is private.
Records are in _episode_logits (float32 action-vocabulary scores) and
_episode_trace (JSON-compatible audit). No model is loaded by this module.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
from pathlib import Path

import numpy as np
import torch

from research.ar_token_counterfactual.intervention import projector_intervention
from research.semantic_token_cd.global_merge_policy import (
    guided_forward_scores, projector_merge_intervention,
)
from research.semantic_token_cd.prompt_attn_shr_policy import (
    PromptAttentionSHRInference, extract_prompt_attention,
)
from research.semantic_token_cd.st_shr_policy import harmonic_reconstruct
from research.semantic_token_cd.vtcg_factorial_math import (
    attention_ranking, deterministic_seed, fixed_count, kmeans_full_ranking,
    matched_count, norm_matched_random_replacement, select_top_count, top_p_count,
)

ARMS = (
    'attn_matched', 'attn_fixed', 'attn_topp', 'kmeans_matched',
    'kmeans_fixed', 'kmeans_topp', 'vanilla', 'random_matched', 'norm_noise',
)
TOP_P_MIN_COUNT = 16
TOP_P_MAX_COUNT = 64
_SOURCE = Path(__file__).with_name('prompt_attn_shr_policy.py')
SOURCE_SHA256 = hashlib.sha256(_SOURCE.read_bytes()).hexdigest()
# Audited revisions of the shared implementation this factory was reviewed against.
# 2026-10-17: another session modified the shared file (see
# artifacts/vtcg_factorial_controls_v1/INCIDENT_20261017_SOURCE_CHANGE.md). The edit
# lives inside PromptAttentionSHRInference.step, which this class overrides and never
# calls, and bitwise equivalence was verified by re-running completed episodes under the
# new revision (SOURCE_PIN_UPDATE.json). Both revisions stay accepted; any third
# revision still fails fast, and each episode records the exact hash it ran under.
INSPECTED_SOURCE_SHA256 = (
    '0ed313033aa2d3a07ce78c3219a3cec2ed84e15695b5bd39aea2912cb012338b',  # original review
    '75eb975ea384f2fe4ad0d23e54a5a23cbe79d093c386ed2adea3c0fbc7799bde',  # audited 2026-10-17
)
if SOURCE_SHA256 not in INSPECTED_SOURCE_SHA256:
    raise RuntimeError('shared source changed since review; re-audit before factorial use')


@contextmanager
def _audited_projector_merge(model, replacement):
    """Capture actual projector output dtype, not a parameter-dtype guess."""
    observed = {}
    def capture_dtype(module, inputs, output):
        observed['dtype'] = output.dtype
    handle = model.projector.register_forward_hook(capture_dtype)
    try:
        with projector_merge_intervention(model, replacement) as trace:
            trace['dtype_observation'] = observed
            yield trace
    finally:
        handle.remove()


def action_logits_float32(policy, scores):
    start = int(policy.vla.vocab_size) - 256
    result = scores[:, start:start + 256].detach().float().cpu().numpy().copy()
    if result.shape != (7, 256) or not np.isfinite(result).all():
        raise FloatingPointError('expected finite float32 action logits [7,256]')
    return result


class VTCGFactorialPolicy(PromptAttentionSHRInference):
    """Own step implementation; inherited only frozen forward/semantic primitives."""

    def reset(self, task_description, seed=None):
        # Parent reset does not clear trace/logits; do so explicitly per episode.
        # It also caches entity embeddings by instruction; our caches are private.
        super().reset(task_description, seed)
        self._episode_trace = []
        self._episode_logits = []
        self._selector_step = 0
        if seed is not None:
            self._episode_seed = int(seed)
        self._mask_audit = None

    def _select_factorial(self, features, attention):
        """CPU selection seam: attention mass never replaced by ranking scores."""
        labels, _, entity_scores = self._semantic_clusters(features)
        # Preserve canonical PromptAttentionSHR matched budget implementation:
        # _entity_groups uses the original feature dtype, unlike _semantic_clusters.
        groups = list(dict.fromkeys(self._entity_groups(features, labels)))
        union = np.asarray(sorted({int(i) for g in groups
                                   for i in np.flatnonzero(labels == g)}), dtype=np.int64)
        matched_m = matched_count(union)
        attn_order = attention_ranking(attention)
        arm = self.factorial_arm
        count_rule = ('fixed' if arm.endswith('_fixed') else
                      'topp' if arm.endswith('_topp') else 'matched')
        raw_top_p = top_p_count(attention, self.factorial_top_p)
        count = (fixed_count(self.factorial_fixed_m) if count_rule == 'fixed' else
                 top_p_count(attention, self.factorial_top_p,
                             TOP_P_MIN_COUNT, TOP_P_MAX_COUNT) if count_rule == 'topp'
                 else matched_m)
        random_seed = None
        if arm.startswith('kmeans_'):
            ranking = kmeans_full_ranking(features, labels, np.stack(self._entity_emb))
            ranking_rule = 'max_entity_centroid_cosine_then_member_centroid_cosine'
        elif arm == 'random_matched':
            random_seed = deterministic_seed('VTCG_RANDOM_V1', self.factorial_task,
                                            self._episode_seed, self._selector_step)
            ranking = np.random.default_rng(random_seed).permutation(256)
            ranking_rule = 'PCG64_full_permutation'
        else:
            ranking = attn_order
            ranking_rule = 'clean_prompt_attention_L11_desc_token_id_asc'
        selected = select_top_count(ranking, count)
        probability = np.asarray(attention, dtype=np.float64)
        probability = probability / probability.sum()
        audit = {
            'arm': arm, 'ranking_rule': ranking_rule,
            'full_ranking_token_ids': ranking.tolist(),
            'attention_full_ranking_token_ids': attn_order.tolist(),
            'selected_token_ids': selected.tolist(), 'num_tokens': count,
            'count_rule': count_rule, 'matched_count': matched_m,
            'reference_shr_token_ids': union.tolist(),
            'matched_entity_group_ids': [int(g) for g in groups],
            'kmeans_labels': np.asarray(labels).tolist(),
            'per_entity_score': [float(x) for x in entity_scores],
            'top_p': self.factorial_top_p, 'top_p_raw_count': raw_top_p,
            'top_p_count_source': 'clean_L11_attention_visual_normalized_mass',
            'top_p_min_count': TOP_P_MIN_COUNT, 'top_p_max_count': TOP_P_MAX_COUNT,
            'selected_attention_mass': float(probability[selected].sum()),
            'random_seed': random_seed, 'coverage_exact': len(selected) == count,
            'kmeans_K': 8, 'kmeans_seed': 0, 'kmeans_n_init': 10,
        }
        return selected, audit

    def step(self, image, contrast_image=None, task_description=None, *args, **kwargs):
        instruction = task_description if task_description is not None else self.task_description
        inputs = self.process_inputs(image, task_description=instruction)
        with projector_intervention(self.vla) as positive_trace:
            clean = self._forward_scores(inputs, self.unnorm_key, do_sample=False).float()
        if clean.shape[0] != 7 or positive_trace.before is None:
            raise RuntimeError('clean branch must capture projector features and seven scores')
        visual = positive_trace.before
        features = visual[0].detach().float().cpu().numpy().copy()
        if features.ndim != 2 or features.shape[0] != 256:
            raise RuntimeError('expected clean projector features [256,D]')
        clean_ids = clean.argmax(dim=-1)
        meta = {
            'arm': self.factorial_arm, 'instruction': instruction,
            'lambda': float(self.lambd), 'beta': 0.0, 'attention_layers': [11],
            'source_policy_sha256': SOURCE_SHA256,
            'inspected_source_policy_sha256': INSPECTED_SOURCE_SHA256,
            'guided_prefix': True, 'prefix_rule': 'shared_clean_greedy_prefix',
            'guided_dimensions': list(range(6)), 'gripper_dimension_clean': 6,
            'step_index': self._selector_step, 'episode_seed': self._episode_seed,
            'intervention_location': 'projector',
        }
        record = {}
        if self.factorial_arm == 'vanilla':
            negative = clean.clone()
            guided = clean.clone()
            selected = np.empty(0, dtype=np.int64)
            perturbation = np.zeros_like(features)
            meta.update({'num_tokens': 0, 'selected_token_ids': [],
                         'full_ranking_token_ids': [], 'count_rule': 'none',
                         'counterfactual_constructor': 'none', 'negative_branch_executed': False,
                         'non_target_bit_identical': True, 'feature_equal': True})
        else:
            attention, attention_meta = extract_prompt_attention(
                self, inputs, instruction, visual, layers=(11,))
            selected, selection_meta = self._select_factorial(features, attention)
            if len(selected) == 256:
                raise RuntimeError('all-grid harmonic reconstruction has no Dirichlet boundary')
            harmonic = harmonic_reconstruct(features, selected, beta=0.0)
            replacement = harmonic
            constructor = 'harmonic'
            constructor_audit = {}
            if self.factorial_arm == 'norm_noise':
                constructor = 'norm_matched_random_direction'
                seed = deterministic_seed('VTCG_NORM_NOISE_V1', self.factorial_task,
                                          self._episode_seed, self._selector_step)
                replacement, constructor_audit = norm_matched_random_replacement(
                    features, selected, harmonic, seed)
                replacement = replacement.astype(np.float32)
                target = np.linalg.norm(harmonic.astype(np.float64) - features[selected], axis=1)
                actual = np.linalg.norm(replacement.astype(np.float64) - features[selected], axis=1)
                constructor_audit.update({
                    'float32_target_residual_norms': target.tolist(),
                    'float32_actual_residual_norms': actual.tolist(),
                    'float32_max_abs_norm_error': float(np.max(np.abs(target - actual))),
                    'float32_norms_match': bool(np.allclose(actual, target, rtol=1e-5, atol=1e-6)),
                })
                if not constructor_audit['float32_norms_match']:
                    raise FloatingPointError('float32 per-token noise norm mismatch')
            negative_features = features.copy()
            negative_features[selected] = replacement
            outside = np.ones(256, dtype=bool)
            outside[selected] = False
            unchanged = bool(np.array_equal(negative_features[outside], features[outside]))
            if not unchanged or not np.isfinite(negative_features).all():
                raise FloatingPointError('counterfactual feature audit failed')
            with _audited_projector_merge(
                    self.vla, torch.from_numpy(negative_features).unsqueeze(0)) as trace:
                negative = guided_forward_scores(self.vla, inputs, clean_ids, visual.shape[1]).float()
            if trace['before'] is None or not torch.equal(visual, trace['before']):
                raise RuntimeError('negative branch projector input differs from clean')
            if negative.shape != clean.shape:
                raise RuntimeError('negative branch score shape differs from clean')
            if trace.get('after') is None:
                raise RuntimeError('negative branch did not capture applied features')
            applied = trace['after'][0].detach().float().cpu().numpy()
            if not np.array_equal(applied[outside], features[outside]):
                raise RuntimeError('model-dtype cast changed unselected clean tokens')
            # Counterfactual norm matching is exact pre-cast, not generally after
            # bf16 quantization. Report the realized discrepancy without hiding it.
            dtype = trace['dtype_observation'].get('dtype')
            if dtype is None:
                raise RuntimeError('actual projector dtype was not observed')
            harmonic_applied = torch.from_numpy(harmonic).to(dtype=dtype).float().numpy()
            target_applied = np.linalg.norm(harmonic_applied.astype(np.float64) - features[selected], axis=1)
            actual_applied = np.linalg.norm(applied[selected].astype(np.float64) - features[selected], axis=1)
            constructor_audit.update({
                'applied_harmonic_residual_norms': target_applied.tolist(),
                'applied_actual_residual_norms': actual_applied.tolist(),
                'applied_max_abs_norm_error': float(np.max(np.abs(target_applied - actual_applied))),
                'applied_max_relative_norm_error': float(np.max(np.abs(target_applied - actual_applied) / np.maximum(target_applied, 1e-12))),
                'applied_model_dtype': str(dtype),
                'applied_non_target_bit_identical': True,
                'applied_norm_matching_claim': 'measured_only; model quantization may change norms',
            })
            guided, _ = self._combine_action_scores(clean, negative)
            # EOS alone is suppressed to -inf. CD of two -inf scores is NaN;
            # sanitize this excluded token, not other invalid logits.
            eos = int(self.vla.generation_config.eos_token_id)
            allowed = torch.zeros_like(guided, dtype=torch.bool)
            allowed[:, eos] = True
            if ((~torch.isfinite(guided)) & ~allowed).any():
                raise FloatingPointError('nonfinite non-EOS guided logits')
            guided = guided.clone()
            guided[:, eos] = torch.finfo(torch.float32).min
            perturbation = negative_features - features
            meta.update(attention_meta)
            meta.update(selection_meta)
            meta.update({'counterfactual_constructor': constructor,
                         'constructor_audit': constructor_audit,
                         'negative_branch_executed': True, 'feature_equal': True,
                         'non_target_bit_identical': unchanged,
                         'reconstruction_finite': True,
                         'feature_perturbation_norm': float(np.linalg.norm(perturbation))})
            record.update({
                'prompt_attention': np.asarray(attention, dtype=np.float32),
                'full_ranking_token_ids': np.asarray(selection_meta['full_ranking_token_ids'], dtype=np.int16),
                'attention_full_ranking_token_ids': np.asarray(selection_meta['attention_full_ranking_token_ids'], dtype=np.int16),
                'kmeans_labels': np.asarray(selection_meta['kmeans_labels'], dtype=np.int16),
            })
            feature_arrays = {
                'selected_source_features': features[selected],
                'selected_harmonic_features': harmonic,
                'selected_replacement_features': np.asarray(replacement, dtype=np.float32),
            }
            meta['feature_sha256'] = {
                key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                for key, value in feature_arrays.items()
            }
            if self.record_features and self._selector_step < self.record_feature_steps:
                record.update({key: value.copy() for key, value in feature_arrays.items()})
        ids = guided.argmax(dim=-1)
        if not torch.equal(ids[6:], clean_ids[6:]):
            raise RuntimeError('gripper changed from clean action')
        raw = self._decode_actions(ids, self.unnorm_key)[None]
        raw_action, actions = self.postprocess_actions(raw)
        meta.update({'positive_token_ids': clean_ids.detach().cpu().tolist(),
                     'negative_token_ids': negative.argmax(dim=-1).detach().cpu().tolist(),
                     'final_token_ids': ids.detach().cpu().tolist()})
        record.update({'positive': action_logits_float32(self, clean),
                       'negative': action_logits_float32(self, negative),
                       'guided': action_logits_float32(self, guided),
                       'selected_token_ids': selected.astype(np.int16),
                       'selected_mask': np.isin(np.arange(256), selected).astype(np.uint8),
                       'per_token_perturbation_norm': np.linalg.norm(perturbation, axis=1).astype(np.float32),
                       'step_index': np.asarray(self._selector_step, dtype=np.int32)})
        self._mask_audit = copy.deepcopy(meta)
        self._episode_logits.append(record)
        self._episode_trace.append(meta)
        self._selector_step += 1
        return raw_action, actions, meta


def build_policy(base, task, arm, fixed_m=48, top_p=.85, lambd=.5,
                 record_features=False, record_feature_steps=2):
    """Build without loading model/resetting task. Call reset(instruction, seed).

    Fresh base per arm is preferred. Even if reused, all instance attributes
    except frozen vla/processor are deep-copied to prevent mutable state leaks.
    """
    if arm not in ARMS:
        raise ValueError(f'unknown factorial arm: {arm}')
    if float(lambd) != .5:
        raise ValueError('factorial protocol locks lambda=.5')
    m = fixed_count(fixed_m)
    if m == 256:
        raise ValueError('fixed_m=256 leaves no harmonic boundary')
    # Validate p immediately using positive dummy attention, without model work.
    top_p_count(np.ones(256), top_p, TOP_P_MIN_COUNT, TOP_P_MAX_COUNT)
    policy = copy.copy(base)
    policy.__dict__ = {k: (v if k in {'vla', 'processor'} else copy.deepcopy(v))
                       for k, v in vars(base).items()}
    policy.__class__ = VTCGFactorialPolicy
    # Explicit copy of _init_common semantics, avoiding rollout/simulator imports.
    policy.alpha = policy.lambd = float(lambd)
    policy.selection_mode = 'semantic'
    policy.kmeans_K = 8
    policy.kmeans_seed = 0
    policy._selector_instr = None
    policy._entities = []
    policy._entity_emb = []
    policy._emb_cache = {}
    policy._episode_trace = []
    policy._episode_logits = []
    policy._episode_seed = 0
    policy._selector_step = 0
    policy._st_history = {}
    policy._previous_residual = None
    policy._mask_audit = None
    policy.previous_gripper_action = None
    policy.sticky_action_is_on = False
    policy.gripper_action_repeat = 0
    policy.sticky_gripper_action = 0.0
    policy.num_image_history = 0
    policy.task_description = None
    policy.beta = 0.0
    policy.attention_layers = (11,)
    policy.use_partial_l11_attention = False
    policy.factorial_arm = arm
    policy.factorial_task = str(task)
    policy.factorial_fixed_m = m
    policy.factorial_top_p = float(top_p)
    policy.record_features = bool(record_features)
    policy.record_feature_steps = int(record_feature_steps)
    if policy.record_feature_steps < 0:
        raise ValueError('record_feature_steps must be nonnegative')
    return policy
