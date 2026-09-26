#!/usr/bin/env python3
"""Compute mild-view L11-Matched action logits on saved LIBERO trajectories.

This performs offline policy inference on frames from historical matched-arm
videos. It does not step the environment or create new episodes. The same
decoded frame is the base for identity and all six photometric views.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

REPO = Path('/home/leju-suzhou/zjt_ws/token-cd')
ARTIFACT = REPO / 'artifacts/libero90_five_task_simpler_config_v1'
DEFAULT_OUT = REPO / 'artifacts/libero_transfer_diagnostics/libero_mild_view_logits'
CHECKPOINT = Path('/home/leju-suzhou/zjt_ws/checkpoints/libero/vq-vla-openvla-7b-libero90')
CODE_DIR = REPO / 'third_party/openvla/prismatic/extern/hf'
VARIANTS = ('identity', 'brightness_095', 'brightness_105',
            'contrast_095', 'contrast_105', 'gamma_095', 'gamma_105')


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def transform(image: np.ndarray, name: str) -> np.ndarray:
    x = image.astype(np.float32) / 255.0
    if name.startswith('brightness_'):
        x *= float(name.rsplit('_', 1)[1])
    elif name.startswith('contrast_'):
        factor = float(name.rsplit('_', 1)[1])
        mean = x.mean(axis=(0, 1), keepdims=True)
        x = (x - mean) * factor + mean
    elif name.startswith('gamma_'):
        x = np.power(np.clip(x, 0.0, 1.0), float(name.rsplit('_', 1)[1]))
    return np.rint(np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--device-index', type=int, default=0,
                        help='CUDA index inside CUDA_VISIBLE_DEVICES for the model')
    case_source = parser.add_mutually_exclusive_group(required=True)
    case_source.add_argument('--case-ids', help='comma-separated pair IDs')
    case_source.add_argument('--case-list', type=Path, help='newline- or comma-separated pair IDs')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--frame-interval', type=int, default=5)
    parser.add_argument('--max-frames', type=int, default=None, help='smoke cap per case')
    parser.add_argument('--max-cases', type=int, default=None)
    parser.add_argument('--skip-existing', action='store_true',
                        help='skip a case only if both metadata and logits outputs exist')
    parser.add_argument('--rgb-source', choices=('video', 'live-replay'), default='video',
                        help='prefer live-replay for canonical simulator frames; video is lossy')
    args = parser.parse_args()
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    sys.path.insert(0, str(REPO))

    import torch
    from research.ar_token_counterfactual.libero_runtime import load_policy, prepare_agentview, set_determinism
    import research.semantic_token_cd.libero_matched_rollout as lmr

    torch.set_num_threads(1)
    model, processor = load_policy(
        CHECKPOINT, CODE_DIR, device=f'cuda:{args.device_index}',
        dataset_statistics_path=CHECKPOINT / 'dataset_statistics.json',
        unnorm_key='libero_90_no_noops',
    )
    original_generate = lmr.generate_clean_action
    original_forward_logits = lmr.forward_logits
    captured: dict[str, np.ndarray] = {}

    def capture_generate(*call_args, **call_kwargs):
        result = original_generate(*call_args, **call_kwargs)
        captured['positive'] = result[1].detach().cpu().numpy().astype(np.float32)
        return result

    def capture_negative(*call_args, **call_kwargs):
        result = original_forward_logits(*call_args, **call_kwargs)
        captured['negative'] = result.detach().cpu().numpy().astype(np.float32)
        return result

    lmr.generate_clean_action = capture_generate
    lmr.forward_logits = capture_negative
    args.out.mkdir(parents=True, exist_ok=True)
    raw_case_ids = args.case_ids if args.case_ids is not None else args.case_list.read_text()
    case_ids = [s.strip() for s in raw_case_ids.replace('\n', ',').split(',') if s.strip()]
    if args.max_cases is not None:
        case_ids = case_ids[:args.max_cases]
    for case_id in case_ids:
        case_out = args.out / case_id
        if case_out.exists():
            if args.skip_existing and (case_out / 'metadata.json').is_file() and (case_out / 'action_logits.npz').is_file():
                print(json.dumps({'gpu': args.gpu, 'case': case_id, 'status': 'skip-existing'},
                                 separators=(',', ':')), flush=True)
                continue
            raise FileExistsError(f'refusing to overwrite existing case output: {case_out}')
        pair = json.loads((ARTIFACT / 'pairs' / f'{case_id}.json').read_text())
        matched_path = ARTIFACT / 'episodes' / pair['task_name'] / 'matched' / f"init_{int(pair['init_state_id']):03d}.json"
        episode = json.loads(matched_path.read_text())
        video_path = ARTIFACT / 'videos' / pair['task_name'] / 'matched' / f"init_{int(pair['init_state_id']):03d}.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        replay_validation = {'source': 'video'}
        source_frame_count = len(episode['trajectory'])
        if args.rgb_source == 'live-replay':
            from libero.libero import benchmark, get_libero_path
            from libero.libero.envs import OffScreenRenderEnv
            suite = benchmark.get_benchmark_dict()['libero_90']()
            task = suite.get_task(int(pair['task_id']))
            init_state = suite.get_task_init_states(int(pair['task_id']))[int(pair['init_state_id'])]
            bddl = str(Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file)
            set_determinism(int(pair['case_seed']))
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
            frames = []
            try:
                env.seed(int(episode.get('env_seed', 0)))
                obs = env.reset()
                obs = env.set_init_state(init_state)
                for _ in range(int(episode.get('settle_steps', 10))):
                    obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                state_hash = sha(np.asarray(env.get_sim_state()))
                selected_action_steps = list(range(0, len(episode['trajectory']), args.frame_interval))
                selected_action_steps = set(selected_action_steps)
                for step, action in enumerate(episode['trajectory']):
                    if step in selected_action_steps:
                        _, image = prepare_agentview(obs)
                        frames.append((step, np.asarray(image, dtype=np.uint8).copy()))
                    obs, _, done, _ = env.step(np.asarray(action, dtype=float).tolist())
                    if done and step + 1 < len(episode['trajectory']):
                        raise RuntimeError(f'early replay termination {case_id} at step {step}')
                replay_success = bool(env.check_success())
                replay_validation = {
                    'source': 'live-replay',
                    'initial_state_sha256': state_hash,
                    'initial_state_matches_history': state_hash == pair['init_state_sha256'],
                    'historical_matched_success': bool(pair['matched_success']),
                    'replay_success': replay_success,
                    'success_matches_history': replay_success == bool(pair['matched_success']),
                    'action_steps_replayed': len(episode['trajectory']),
                }
            finally:
                env.close()
            source_pairs = [(step, arr) for step, arr in frames]
        else:
            video_frames = imageio.mimread(video_path, memtest=False)
            source_frame_count = len(video_frames)
            if len(video_frames) != len(episode['trace']) or len(video_frames) != len(episode['trajectory']):
                raise RuntimeError(f'frame/trace/action length mismatch for {case_id}: '
                                   f'{len(video_frames)}/{len(episode["trace"])}/{len(episode["trajectory"])}')
            source_pairs = [(step, np.asarray(video_frames[step], dtype=np.uint8)[..., :3])
                            for step in range(0, len(video_frames), args.frame_interval)]
        frames = [arr for _, arr in source_pairs]
        selected_steps = [step for step, _ in source_pairs]
        if args.max_frames is not None:
            selected_steps = selected_steps[:args.max_frames]
            frames = frames[:args.max_frames]
        step_rows = []
        positive_steps, negative_steps = [], []
        started = time.monotonic()
        for step, source_frame in zip(selected_steps, frames):
            source_image = np.asarray(source_frame, dtype=np.uint8)[..., :3]
            source_sha = sha(source_image)
            pos_views, neg_views, view_rows = [], [], []
            for view_name in VARIANTS:
                view_array = transform(source_image, view_name)
                view_image = Image.fromarray(view_array, mode='RGB')
                set_determinism(int(pair['case_seed']))
                captured.clear()
                action, meta = lmr.predict_matched(
                    model, processor, view_image, pair['instruction'],
                    entity_mode='source_target_libero90', query_mode='instruction_only',
                    attention_layers=(11,), attention_heads=(), destination_weight=0.0,
                    lambda_scale=1.0, unnorm_key='libero_90_no_noops', position_mode='attention',
                )
                if 'positive' not in captured or 'negative' not in captured:
                    raise RuntimeError('failed to capture positive/negative action logits')
                vocab_start = int(model.vocab_size) - 256
                pos = captured['positive'][:, vocab_start:vocab_start + 256]
                neg = captured['negative'][:, vocab_start:vocab_start + 256]
                if pos.shape != (7, 256) or neg.shape != pos.shape:
                    raise RuntimeError(f'bad action logits for {case_id} step={step}: {pos.shape}/{neg.shape}')
                pos_views.append(pos.astype(np.float16))
                neg_views.append(neg.astype(np.float16))
                historical = episode['trace'][step]
                ids = [int(x) for x in meta['selected_token_ids']]
                hist_ids = [int(x) for x in historical.get('selected_token_ids', [])]
                view_rows.append({
                    'view': view_name,
                    'view_rgb_sha256': sha(view_array),
                    'm_current': int(meta['m_matched']),
                    'm_historical': int(historical.get('m_effective', -1)),
                    'groups_current': meta.get('kmeans_groups'),
                    'selected_ids_current': ids,
                    'selected_ids_historical': hist_ids,
                    'identity_trace_m_match': int(meta['m_matched']) == int(historical.get('m_effective', -1)),
                    'identity_trace_selected_exact': sorted(ids) == sorted(hist_ids),
                    'attention_sha256': meta.get('attention_sha256'),
                    'positive_token_ids': meta.get('positive_token_ids'),
                    'guided_token_ids': meta.get('final_token_ids'),
                    'guided_action': np.asarray(action, dtype=float).tolist(),
                })
            positive_steps.append(np.stack(pos_views))
            negative_steps.append(np.stack(neg_views))
            step_rows.append({'step': step, 'source_rgb_sha256': source_sha, 'views': view_rows})
            print(json.dumps({'gpu': args.gpu, 'case': case_id, 'step': step,
                              'total_steps': len(selected_steps)}, separators=(',', ':')), flush=True)

        case_out.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(case_out / 'action_logits.npz',
                            positive=np.stack(positive_steps), negative=np.stack(negative_steps))
        paired = {'rescue': bool(pair['matched_success']) and not bool(pair['vanilla_success']),
                  'harm': bool(pair['vanilla_success']) and not bool(pair['matched_success'])}
        metadata = {
            'protocol': 'LIBERO_MILD_RGB_COUNTERFACTUAL_STABILITY_V1',
            'case_id': case_id, 'task_id': int(pair['task_id']), 'task': pair['task_name'],
            'init_state_id': int(pair['init_state_id']), 'case_seed': int(pair['case_seed']),
            'outcome': 'rescue' if paired['rescue'] else 'harm' if paired['harm'] else 'same',
            'vanilla_success': bool(pair['vanilla_success']), 'matched_success': bool(pair['matched_success']),
            'checkpoint_revision': episode.get('checkpoint_revision'),
            'source_video': str(video_path), 'source_frame_count': source_frame_count,
            'trajectory_steps': len(episode['trajectory']), 'frame_interval': args.frame_interval,
            'rgb_source': args.rgb_source, 'replay_validation': replay_validation,
            'views': list(VARIANTS), 'frames': step_rows,
            'arrays': 'action_logits.npz', 'runtime_seconds': time.monotonic() - started,
            'historical_trace_match_counts': {
                'm': sum(x['views'][0]['identity_trace_m_match'] for x in step_rows),
                'selected_ids': sum(x['views'][0]['identity_trace_selected_exact'] for x in step_rows),
                'n': len(step_rows),
            },
        }
        (case_out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(json.dumps({'gpu': args.gpu, 'case': case_id, 'status': 'complete',
                          'outcome': metadata['outcome'], 'sampled_states': len(step_rows),
                          'runtime_seconds': metadata['runtime_seconds']}, separators=(',', ':')), flush=True)
    del model


if __name__ == '__main__':
    main()
