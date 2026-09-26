#!/usr/bin/env python3
"""Compare harmonic vs local-boundary negative logits at trace-valid replay states.

The selector, matched m, clean branch, image, and lambda are kept fixed within
each state. Only the feature replacement on selected projector rows changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path('/home/leju-suzhou/zjt_ws/token-cd')
ARTIFACT = REPO / 'artifacts/libero90_five_task_simpler_config_v1'
CHECKPOINT = Path('/home/leju-suzhou/zjt_ws/checkpoints/libero/vq-vla-openvla-7b-libero90')
CODE_DIR = REPO / 'third_party/openvla/prismatic/extern/hf'
OUT_DEFAULT = REPO / 'artifacts/libero_transfer_diagnostics/negative_variant_probe_v4'
GRID = 16


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def local_mean_reconstruct(features: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Replace each selected token by the original clean 4-neighbor feature mean."""
    h = np.asarray(features, dtype=np.float32)
    selected = sorted(set(int(x) for x in region))
    out = h[np.asarray(selected, dtype=np.int64)].copy()
    for row, token in enumerate(selected):
        r, c = divmod(token, GRID)
        neighbors = []
        if r > 0: neighbors.append(token - GRID)
        if r + 1 < GRID: neighbors.append(token + GRID)
        if c > 0: neighbors.append(token - 1)
        if c + 1 < GRID: neighbors.append(token + 1)
        out[row] = h[np.asarray(neighbors, dtype=np.int64)].mean(axis=0) if neighbors else h.mean(axis=0)
    return out


def feature_knn_reconstruct(features: np.ndarray, region: np.ndarray, k: int = 4) -> np.ndarray:
    """Replace selected rows by the mean of their four nearest unselected feature vectors."""
    h = np.asarray(features, dtype=np.float32)
    selected = sorted(set(int(x) for x in region))
    selected_set = set(selected)
    complement = np.asarray([i for i in range(h.shape[0]) if i not in selected_set], dtype=np.int64)
    if not len(complement):
        return np.repeat(h.mean(axis=0, keepdims=True), len(selected), axis=0)
    norms = np.linalg.norm(h, axis=1, keepdims=True).clip(min=1e-12)
    hn = h / norms
    out = h[np.asarray(selected, dtype=np.int64)].copy()
    for row, token in enumerate(selected):
        similarities = hn[complement] @ hn[token]
        take = min(int(k), len(complement))
        nearest = complement[np.argpartition(similarities, -take)[-take:]]
        out[row] = h[nearest].mean(axis=0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--case-id', required=True)
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--device-index', type=int, default=0)
    ap.add_argument('--out', type=Path, default=OUT_DEFAULT)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    outdir = args.out / args.case_id
    if outdir.exists() and not args.overwrite:
        if (outdir/'metadata.json').is_file() and (outdir/'margin_logits.npz').is_file():
            print(json.dumps({'case': args.case_id, 'status': 'skip-existing'}), flush=True)
            return
        raise FileExistsError(outdir)

    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    sys.path.insert(0, str(REPO))
    import torch
    from research.ar_token_counterfactual.libero_runtime import load_policy, prepare_agentview, set_determinism
    import research.semantic_token_cd.libero_matched_rollout as lmr
    torch.set_num_threads(1)
    model, processor = load_policy(
        CHECKPOINT, CODE_DIR, device=f'cuda:{args.device_index}',
        dataset_statistics_path=CHECKPOINT/'dataset_statistics.json',
        unnorm_key='libero_90_no_noops')

    pair = json.loads((ARTIFACT/'pairs'/f'{args.case_id}.json').read_text())
    episode_path = ARTIFACT/'episodes'/pair['task_name']/'matched'/f"init_{int(pair['init_state_id']):03d}.json"
    episode = json.loads(episode_path.read_text())
    old = json.loads((REPO/'artifacts/libero_transfer_diagnostics/libero_mild_view_logits'/args.case_id/'metadata.json').read_text())
    target_steps = {}
    for frame in old['frames']:
        v = frame['views'][0]
        if v['identity_trace_m_match'] and v['identity_trace_selected_exact']:
            target_steps[int(frame['step'])] = frame['source_rgb_sha256']

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()['libero_90']()
    task = suite.get_task(int(pair['task_id']))
    init_state = suite.get_task_init_states(int(pair['task_id']))[int(pair['init_state_id'])]
    bddl = str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file)
    set_determinism(int(pair['case_seed']))
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    captured: dict[str, np.ndarray] = {}
    orig_generate, orig_forward = lmr.generate_clean_action, lmr.forward_logits

    def capture_generate(*a, **kw):
        result = orig_generate(*a, **kw)
        captured['positive'] = result[1].detach().cpu().numpy().astype(np.float32)
        return result

    def capture_forward(*a, **kw):
        result = orig_forward(*a, **kw)
        captured['negative'] = result.detach().cpu().numpy().astype(np.float32)
        return result

    lmr.generate_clean_action = capture_generate
    lmr.forward_logits = capture_forward
    original_harmonic = lmr.harmonic_reconstruct
    frames = []
    start = time.monotonic()
    try:
        env.seed(int(episode.get('env_seed', 0)))
        obs = env.reset()
        obs = env.set_init_state(init_state)
        for _ in range(int(episode.get('settle_steps', 10))):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        initial_state_hash = sha(np.asarray(env.get_sim_state()))
        for step, action in enumerate(episode['trajectory']):
            if step in target_steps:
                _, rgb = prepare_agentview(obs)
                rgb = np.asarray(rgb, dtype=np.uint8)[..., :3].copy()
                rgb_hash = sha(rgb)
                historical_frame = old['frames'][step//5]['views'][0]
                image = Image.fromarray(rgb, mode='RGB')
                set_determinism(int(pair['case_seed']))
                captured.clear()
                _, meta_h = lmr.predict_matched(
                    model, processor, image, pair['instruction'],
                    entity_mode='source_target_libero90', query_mode='instruction_only',
                    attention_layers=(11,), attention_heads=(), destination_weight=0.0,
                    lambda_scale=1.0, unnorm_key='libero_90_no_noops', position_mode='attention')
                z_h = captured['positive'].copy()
                n_h = captured['negative'].copy()
                old_ids = historical_frame['selected_ids_historical']
                m_match = int(meta_h['m_matched']) == int(historical_frame['m_historical'])
                ids_match = sorted(meta_h['selected_token_ids']) == sorted(old_ids)
                if not (m_match and ids_match):
                    frames.append({'step': step, 'status': 'current_selector_mismatch',
                                   'historical_rgb_hash_match': rgb_hash == target_steps[step],
                                   'historical_m_match': m_match, 'historical_ids_match': ids_match,
                                   'm_current': int(meta_h['m_matched']),
                                   'm_historical': int(historical_frame['m_historical'])})
                else:
                    lmr.harmonic_reconstruct = feature_knn_reconstruct
                    set_determinism(int(pair['case_seed']))
                    captured.clear()
                    _, meta_local = lmr.predict_matched(
                        model, processor, image, pair['instruction'],
                        entity_mode='source_target_libero90', query_mode='instruction_only',
                        attention_layers=(11,), attention_heads=(), destination_weight=0.0,
                        lambda_scale=1.0, unnorm_key='libero_90_no_noops', position_mode='attention')
                    z_local = captured['positive'].copy()
                    n_local = captured['negative'].copy()
                    if meta_local['m_matched'] != meta_h['m_matched'] or meta_local['selected_token_ids'] != meta_h['selected_token_ids']:
                        raise RuntimeError(f"selector changed across negative arms on {args.case_id}:{step}")
                    vocab_start = int(model.vocab_size) - 256
                    finite_pair = np.isfinite(z_h) & np.isfinite(z_local)
                    clean_finite_error = float(np.max(np.abs(z_h[finite_pair]-z_local[finite_pair]))) if finite_pair.any() else None
                    nonfinite_pattern_mismatches = int(np.count_nonzero(np.isfinite(z_h) != np.isfinite(z_local)))
                    frames.append({
                        'step': step, 'status': 'complete', 'rgb_sha256': rgb_hash,
                        'historical_rgb_hash_match': rgb_hash == target_steps[step],
                        'm': int(meta_h['m_matched']), 'selected_ids': meta_h['selected_token_ids'],
                        'selected_ids_historical': old_ids,
                        'clean_positive_arm_max_abs_error_finite': clean_finite_error,
                        'clean_positive_arm_nonfinite_pattern_mismatches': nonfinite_pattern_mismatches,
                        'harmonic_relative_perturbation': float(meta_h['feature_perturbation_relative']),
                        'local_relative_perturbation': float(meta_local['feature_perturbation_relative']),
                        'arrays_index': len(frame_arrays),
                    })
                    pos_h = z_h[:, vocab_start:vocab_start+256]
                    pos_local = z_local[:, vocab_start:vocab_start+256]
                    neg_h = n_h[:, vocab_start:vocab_start+256]
                    neg_local = n_local[:, vocab_start:vocab_start+256]
                    frame_arrays.append((pos_h, pos_local, neg_h, neg_local))
                    print(json.dumps({'gpu': args.gpu, 'case': args.case_id, 'step': step,
                                      'valid_state': len(frame_arrays)}, separators=(',', ':')), flush=True)
            obs, _, done, _ = env.step(np.asarray(action, dtype=float).tolist())
            if done and step + 1 < len(episode['trajectory']):
                raise RuntimeError(f'early replay termination {args.case_id} at step {step}')
        success = bool(env.check_success())
    finally:
        lmr.harmonic_reconstruct = original_harmonic
        env.close()

    complete = [x for x in frames if x.get('status') == 'complete']
    if len(frame_arrays) != len(complete):
        raise RuntimeError('frame metadata / array count mismatch')
    outdir.mkdir(parents=True, exist_ok=True)
    if frame_arrays:
        arrays = [np.stack(x, axis=0).astype(np.float16) for x in zip(*frame_arrays)]
    else:
        arrays = [np.empty((0, 7, 256), dtype=np.float16) for _ in range(4)]
    np.savez_compressed(outdir/'margin_logits.npz', positive_harmonic=arrays[0],
                        positive_local=arrays[1], negative_harmonic=arrays[2],
                        negative_local=arrays[3])
    meta = {
        'protocol': 'TRACE_VALID_HARMONIC_VS_LOCAL_BOUNDARY_NEGATIVE_V1',
        'case_id': args.case_id, 'task_id': int(pair['task_id']), 'task': pair['task_name'],
        'outcome': old['outcome'], 'vanilla_success': bool(pair['vanilla_success']),
        'matched_success': bool(pair['matched_success']), 'initial_state_hash': initial_state_hash,
        'initial_state_matches_history': initial_state_hash == pair['init_state_sha256'],
        'replay_success': success, 'replay_success_matches_history': success == bool(pair['matched_success']),
        'lambda': .5, 'selector': 'L11 instruction-only Matched; same m and token IDs in both negative arms',
        'alternative_negative': 'replace each selected grid feature by mean of its four nearest unselected clean projector features by cosine similarity',
        'historical_rgb_hash_match_count': sum(x.get('historical_rgb_hash_match', False) for x in complete),
        'historical_m_and_ids_match_count': len(complete),
        'frames': frames, 'arrays': 'margin_logits.npz', 'runtime_seconds': time.monotonic()-start,
    }
    (outdir/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps({'gpu': args.gpu, 'case': args.case_id, 'status': 'complete',
                      'valid_states': len(complete),
                      'historical_rgb_hash_mismatches': sum(not x.get('historical_rgb_hash_match', False) for x in complete),
                      'success_match': meta['replay_success_matches_history'],
                      'runtime_seconds': meta['runtime_seconds']}, separators=(',', ':')), flush=True)


if __name__ == '__main__':
    frame_arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    main()
