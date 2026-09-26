#!/usr/bin/env python3
"""Replay recorded SIMPLER actions under RT depth-0 and score eight L11 bins."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path('/home/leju-suzhou/zjt_ws/token-cd')
PCD_SOURCE = Path('/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source')
CANONICAL = ROOT / 'artifacts/vanilla_recon_shr_canonical_0_299_v2'
TASKS = (
    'google_robot_open_drawer', 'google_robot_close_drawer',
    'google_robot_pick_coke_can', 'google_robot_move_near',
)


def sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def make_rt_depth0_environment(task: str):
    import gymnasium as gym
    import simpler_env
    render = {'rt_samples_per_pixel': 32, 'rt_max_path_depth': 0, 'rt_use_denoiser': True}
    renderer = {'device': 'cuda:0', 'offscreen_only': True}
    if task == 'google_robot_pick_coke_can':
        return gym.make(
            'GraspSingleOpenedCokeCanDistractorInScene-v0', obs_mode='rgbd',
            prepackaged_config=True, distractor_config='less', shader_dir='rt',
            render_config=render, renderer_kwargs=renderer,
        )
    env_id, base = simpler_env.ENVIRONMENT_MAP[task]
    kwargs = dict(base)
    kwargs.update({
        'prepackaged_config': True, 'shader_dir': 'rt',
        'render_config': render, 'renderer_kwargs': renderer,
    })
    return gym.make(env_id, obs_mode='rgbd', **kwargs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, required=True, help='physical OpenVLA inference GPU')
    ap.add_argument('--worker-id', required=True)
    ap.add_argument('--artifact', type=Path, required=True)
    ap.add_argument('--tasks', default=','.join(TASKS))
    ap.add_argument('--seeds', default='100-129')
    ap.add_argument('--states-per-episode', type=int, default=10)
    ap.add_argument('--max-episodes', type=int, default=0)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    # SIMPLER's known-good RT depth-0 preflights use one visible GPU for both
    # CUDA inference and SAPIEN. LIBERO's failing EGL path is handled separately
    # with OSMesa CPU rendering.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    for path in (ROOT / 'task1/shim_site', ROOT, PCD_SOURCE):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import torch
    torch.set_num_threads(1)
    from parallel_inference import get_image_from_maniskill2_obs_dict
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    from research.semantic_token_cd.distractor_rollout import restore_snapshot, snapshot_sha
    from research.semantic_token_cd.rollout_pilot import array_sha256
    from research.semantic_token_cd.l11_rank_causal_utils import evaluate_simpler_rank_bins
    from research.semantic_token_cd.prompt_attn_shr_policy import PromptAttentionSHRInference
    from research.semantic_token_cd.prompt_attn_shr_rollout import LAMBDA
    from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common

    seeds = [int(x) for x in args.seeds.split(',') if x.strip()] if ',' in args.seeds else list(
        range(int(args.seeds.split('-')[0]), int(args.seeds.split('-')[1]) + 1)
        if '-' in args.seeds else [int(args.seeds)]
    )
    tasks = [x.strip() for x in args.tasks.split(',') if x.strip()]
    if not tasks or any(task not in TASKS for task in tasks):
        raise ValueError(f'unsupported SIMPLER task list: {tasks}')
    jobs = [(task, seed) for task in tasks for seed in seeds]
    if args.smoke:
        jobs = [(task, seeds[0]) for task in tasks]
    if args.max_episodes:
        jobs = jobs[:args.max_episodes]

    artifact = args.artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    checkpoint = str(PCD_SOURCE / 'pretrained/openvla-7b')
    config = get_policy_config('openvla', checkpoint, tasks[0], {}, False)
    base = OpenVLAInference(**config)
    policy = base
    policy.__class__ = PromptAttentionSHRInference
    _init_common(policy, LAMBDA)
    policy.beta = 0.0
    policy.selector_mode = 'prompt_attention'
    policy.attention_layers = (11,)
    policy.selection_count = 32
    policy.selection_rank_bin = None
    policy.save_prompt_attention = False

    log = artifact / 'logs' / f'worker_{args.worker_id}.jsonl'
    log.parent.mkdir(parents=True, exist_ok=True)
    env = None
    current_task = None
    for task, seed in jobs:
        source = CANONICAL / 'episodes' / task / 'vanilla' / f'episode_{seed:03d}_summary.json'
        actions_path = source.parent / f'episode_{seed:03d}_arrays.npz'
        snapshot_path = CANONICAL / 'snapshots' / task / f'seed_{seed:03d}.pkl'
        if not source.is_file() or not actions_path.is_file() or not snapshot_path.is_file():
            raise FileNotFoundError(f'missing canonical action/snapshot for {task} seed={seed}')
        summary = json.loads(source.read_text())
        with np.load(actions_path) as arrays:
            actions = np.asarray(arrays['executed_actions'], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise RuntimeError(f'invalid recorded action array: {actions.shape}')
        out = artifact / 'episodes' / task / f'seed_{seed:03d}_rank_effects.json'
        if out.is_file():
            continue
        if task != current_task:
            if env is not None:
                env.close()
            env = make_rt_depth0_environment(task)
            current_task = task
        with snapshot_path.open('rb') as handle:
            snapshot = pickle.load(handle)
        if snapshot_sha(snapshot) != summary['canonical_snapshot_sha256']:
            raise RuntimeError(f'canonical snapshot mismatch for {task} seed={seed}')
        obs, state_sha, _old_rgb_sha = restore_snapshot(env, seed, snapshot)
        if state_sha != summary['initial_state_sha256']:
            raise RuntimeError(f'initial simulator state mismatch for {task} seed={seed}')
        instruction = env.unwrapped.get_language_instruction()
        if instruction != summary['instruction']:
            raise RuntimeError(f'instruction mismatch for {task} seed={seed}')
        policy.task_index = TASK_INDEX[task]
        policy.reset(instruction, seed=seed)
        max_steps = min(len(actions), int(summary['control_steps']))
        frame_cache = []
        ended_early = False
        # Some historical action arrays continue past a terminal state. Replay
        # the valid canonical prefix and stop at the simulator's terminal flag.
        for t, action in enumerate(actions[:max_steps]):
            image = get_image_from_maniskill2_obs_dict(env, obs)
            rgb = np.asarray(image.convert('RGB'), dtype=np.uint8) if isinstance(image, Image.Image) else np.asarray(image)
            sim_state = np.asarray(env.unwrapped.get_state())
            frame_cache.append((rgb.copy(), array_sha256(sim_state)))
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                ended_early = t + 1 < max_steps
                break
        if not frame_cache:
            raise RuntimeError(f'empty replay prefix for {task}:{seed}')
        sample_steps = sorted(set(int(x) for x in np.linspace(
            0, len(frame_cache) - 1, min(args.states_per_episode, len(frame_cache)),
        )))
        rows = []
        for t in sample_steps:
            rgb, sim_state_sha = frame_cache[t]
            result = evaluate_simpler_rank_bins(policy, rgb, instruction)
            rows.append({
                'step': t, 'sim_state_sha256': sim_state_sha,
                'rgb_sha256': array_sha256(rgb),
                'rank_bins': result['rank_bins'],
                'attention_sha256': result['attention_sha256'],
            })
        payload = {
            'protocol_id': 'L11_RANK_CAUSAL_EFFECT_OFFLINE_V1',
            'benchmark': 'SIMPLER Google Robot', 'shader_dir': 'rt',
            'render_config': {'rt_samples_per_pixel': 32, 'rt_max_path_depth': 0, 'rt_use_denoiser': True},
            'task': task, 'seed': seed, 'source_episode': str(source),
            'source_success': bool(summary['success']), 'source_steps': int(summary['control_steps']),
            'replayed_steps': len(frame_cache), 'terminal_before_recorded_horizon': ended_early,
            'canonical_snapshot_sha256': summary['canonical_snapshot_sha256'],
            'initial_state_sha256': state_sha, 'historical_initial_rgb_sha256': summary['initial_rgb_sha256'],
            'sampled_steps': sample_steps, 'rank_effect_trace': rows,
            'renderer_physical_gpu': args.gpu, 'inference_physical_gpu': args.gpu,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
        os.replace(tmp, out)
        line = {'status': 'complete', 'task': task, 'seed': seed, 'sampled_steps': len(rows), 'success_label': payload['source_success']}
        with log.open('a') as handle:
            handle.write(json.dumps(line, sort_keys=True) + '\n')
        print(json.dumps(line, sort_keys=True), flush=True)
    if env is not None:
        env.close()


if __name__ == '__main__':
    main()
