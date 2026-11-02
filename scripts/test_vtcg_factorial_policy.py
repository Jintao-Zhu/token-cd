"""CPU-only reproducible factorial tests; never loads a model or initializes CUDA.
Run: CUDA_VISIBLE_DEVICES='' bash -c 'source scripts/activate_simpler_local.sh; python scripts/test_vtcg_factorial_policy.py'
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from research.semantic_token_cd import vtcg_factorial_policy as m


def run():
    rng = np.random.default_rng(71)
    # A bf16-representable clean tensor matches the real trace.before behavior.
    features = torch.from_numpy(rng.normal(size=(256, 8)).astype(np.float32)).bfloat16().float().numpy()
    attention = np.linspace(.001, 1, 256, dtype=np.float32)
    visual = torch.from_numpy(features).unsqueeze(0)
    class Base: pass
    base = Base()
    base.vla = SimpleNamespace(vocab_size=300, generation_config=SimpleNamespace(eos_token_id=0))
    base.processor = object()
    base.history = {'nested': []}
    base.unnorm_key = 'x'
    applied = {}
    @contextmanager
    def capture(*a, **kw):
        yield SimpleNamespace(before=visual)
    @contextmanager
    def merge(model, replacement):
        applied['features'] = replacement.bfloat16().float().numpy()[0]
        yield {'before': visual, 'after': replacement.bfloat16().float(),
               'dtype_observation': {'dtype': torch.bfloat16}}
    clean = torch.linspace(-3, 3, 2100).reshape(7, 300)
    clean[:, 0] = -float('inf')
    negative = clean.clone()
    negative[:, 44:] += torch.linspace(.02, .1, 256)
    policies = {}
    with patch.object(m, 'projector_intervention', capture), patch.object(m, '_audited_projector_merge', merge), patch.object(m, 'extract_prompt_attention', lambda *a, **kw: (attention.copy(), {'attention_layers': [11]})), patch.object(m, 'guided_forward_scores', lambda *a, **kw: negative.clone()):
        for arm in m.ARMS:
            p = m.build_policy(base, 'task', arm)
            p.task_description = 'move cup'
            p._episode_seed = 12
            p.process_inputs = lambda *a, **kw: {}
            p._forward_scores = lambda *a, **kw: clean.clone()
            p._decode_actions = lambda ids, key: ids.numpy().astype(np.float32) / 300
            p.postprocess_actions = lambda raw: (raw.copy(), raw.copy())
            p._entity_emb = [np.ones(8)]
            p._entities = ['cup']
            p.step(None, task_description='move cup')
            meta, rec = p._episode_trace[0], p._episode_logits[0]
            assert all(rec[k].dtype == np.float32 for k in ('positive', 'negative', 'guided'))
            assert np.array_equal(rec['positive'], clean[:, 44:].numpy())
            assert meta['final_token_ids'][6] == meta['positive_token_ids'][6]
            if arm != 'vanilla':
                assert sorted(meta['full_ranking_token_ids']) == list(range(256))
                assert meta['non_target_bit_identical']
                assert len(meta['selected_token_ids']) == meta['num_tokens']
                assert 'selected_source_features' not in rec
                assert len(meta['feature_sha256']) == 3
                assert meta['constructor_audit']['applied_model_dtype'] == 'torch.bfloat16'
                assert meta['constructor_audit']['applied_non_target_bit_identical']
            if arm == 'norm_noise':
                assert meta['constructor_audit']['float32_norms_match']
                expected = np.linalg.norm(applied['features'][rec['selected_token_ids']].astype(np.float64) - features[rec['selected_token_ids']], axis=1)
                assert np.array_equal(expected, meta['constructor_audit']['applied_actual_residual_norms'])
            policies[arm] = p
            print('CPU_STEP_OK', arm, meta['num_tokens'])
    for rule in ('matched', 'fixed', 'topp'):
        assert policies['attn_' + rule]._episode_trace[0]['num_tokens'] == policies['kmeans_' + rule]._episode_trace[0]['num_tokens']
    for arm in ('norm_noise', 'random_matched'):
        assert policies[arm]._episode_trace[0]['num_tokens'] == policies['attn_matched']._episode_trace[0]['num_tokens']
    assert np.array_equal(policies['norm_noise']._episode_logits[0]['selected_mask'], policies['attn_matched']._episode_logits[0]['selected_mask'])
    assert all(p.history is not base.history for p in policies.values())
    assert len({id(p.history) for p in policies.values()}) == 9
    assert len({id(p._episode_logits) for p in policies.values()}) == 9
    p = policies['attn_matched']
    # Canonical matched budget's _entity_groups path, not float64 winner path.
    labels, _, _ = p._semantic_clusters(features)
    canonical_groups = p._entity_groups(features, labels)
    canonical_union = sorted({int(i) for g in canonical_groups for i in np.flatnonzero(labels == g)})
    assert p._episode_trace[0]['reference_shr_token_ids'] == canonical_union
    p._embed_phrase = lambda s: np.ones(8)
    p.reset('move cup', seed=123)
    assert p._episode_seed == 123 and p._selector_step == 0 and p._episode_logits == []
    assert p.previous_gripper_action is None
    assert not torch.cuda.is_initialized()
    print('BUDGET_MASK_PARITY_RESET_ISOLATION_CAST_AUDIT_OK CUDA_INITIALIZED', torch.cuda.is_initialized())


if __name__ == '__main__':
    run()
