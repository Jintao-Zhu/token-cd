#!/usr/bin/env python3
"""Persistent worker for paired five-task LIBERO-90 formal evaluation.

Each worker loads the OpenVLA checkpoint once, claims one paired case
(task_id, init_state_id) from the file queue, and runs vanilla then matched
from the same init state.  Queue claims use an exclusive file lock and atomic
renames; results are written to unique paths and never silently overwritten.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.ar_token_counterfactual.libero_runtime import (
    encode_video,
    load_policy,
    predict_action,
    prepare_agentview,
    prepare_env_action,
    set_determinism,
)
from research.semantic_token_cd.libero_matched_rollout import predict_matched

torch.set_num_threads(1)

PROTOCOL = 'LIBERO90_FIVE_TASK_SIMPLER_CONFIG_V1'


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def code_commit() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[2], text=True
        ).strip()
    except Exception:
        return 'unknown'


def claim_case(artifact: Path, worker_id: str) -> tuple[dict, Path] | None:
    pending = artifact / 'cases' / 'pending'
    running = artifact / 'cases' / 'running'
    lock_path = artifact / 'cases' / '.lock'
    running.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        candidates = sorted(pending.glob('*.json'))
        if not candidates:
            return None
        source = candidates[0]
        payload = json.loads(source.read_text())
        target = running / f"{source.stem}__{worker_id}.json"
        if target.exists():
            return None
        os.replace(source, target)
        return payload, target


def complete_case(running_path: Path, artifact: Path) -> None:
    target = artifact / 'cases' / 'completed' / running_path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(running_path, target)


def fail_case(running_path: Path, artifact: Path, error: str) -> None:
    target = artifact / 'cases' / 'error' / running_path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(running_path.read_text())
    payload['error'] = error
    payload['failed_at'] = int(time.time())
    atomic_json(target, payload)
    running_path.unlink(missing_ok=True)


def compact_matched_meta(meta: dict[str, Any], raw_action: np.ndarray, env_action: np.ndarray) -> dict[str, Any]:
    selected = [int(x) for x in meta.get('selected_token_ids', [])]
    return {
        'entities': meta.get('entities'),
        'matched_cluster_ids': meta.get('kmeans_groups'),
        'm_raw': int(meta.get('m_matched', 0)),
        'm_effective': len(selected),
        'unique_mask_count': len(set(selected)),
        'selected_token_ids': selected,
        'guided_changed_dims': int(meta.get('guided_changed_dims', 0)),
        'feature_perturbation_relative': float(meta.get('feature_perturbation_relative', 0.0)),
        'reconstruction_finite': bool(meta.get('reconstruction_finite', False)),
        'lambda': float(meta.get('lambda', 0.0)),
        'raw_action': np.asarray(raw_action, dtype=float).tolist(),
        'executed_action': np.asarray(env_action, dtype=float).tolist(),
        'translation_change': np.asarray(env_action[:3] - raw_action[:3], dtype=float).tolist(),
        'rotation_change': np.asarray(env_action[3:6] - raw_action[3:6], dtype=float).tolist(),
        'gripper_change': float(env_action[6] - raw_action[6]),
    }


def compact_vanilla_meta(raw_action: np.ndarray, env_action: np.ndarray) -> dict[str, Any]:
    return {
        'raw_action': np.asarray(raw_action, dtype=float).tolist(),
        'executed_action': np.asarray(env_action, dtype=float).tolist(),
        'translation_change': [0.0, 0.0, 0.0],
        'rotation_change': [0.0, 0.0, 0.0],
        'gripper_change': 0.0,
    }


def run_arm(
    *,
    arm: str,
    env: Any,
    task: Any,
    init_state: np.ndarray,
    model: Any,
    processor: Any,
    artifact: Path,
    case: dict[str, Any],
    case_seed: int,
    config_hash: str,
    checkpoint_revision: str,
    commit: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    set_determinism(case_seed)
    env.seed(int(args.env_seed))
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(int(args.settle_steps)):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    initial_state_sha = hashlib.sha256(np.asarray(env.get_sim_state()).tobytes()).hexdigest()
    trajectory: list[list[float]] = []
    trace: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    done = False
    started = time.monotonic()
    for control_step in range(int(args.max_steps)):
        _, image = prepare_agentview(obs)
        if args.save_video:
            frames.append(np.asarray(image.copy()))
        if arm == 'vanilla':
            raw_action = predict_action(
                model,
                processor,
                image,
                task.language,
                unnorm_key=args.unnorm_key,
            )
            env_action = prepare_env_action(raw_action)
            trace.append(compact_vanilla_meta(raw_action, env_action))
        else:
            raw_action, meta = predict_matched(
                model,
                processor,
                image,
                task.language,
                entity_mode=args.entity_mode,
                query_mode=args.query_mode,
                attention_layers=tuple(int(x) for x in args.attention_layers.split(',') if x),
                attention_heads=(),
                destination_weight=0.0,
                lambda_scale=args.lambda_scale,
                unnorm_key=args.unnorm_key,
                position_mode='attention',
            )
            env_action = prepare_env_action(raw_action)
            trace.append(compact_matched_meta(meta, raw_action, env_action))
        if env_action.shape != (7,) or not np.isfinite(env_action).all():
            raise FloatingPointError(f'invalid action at task={task.name} arm={arm}: {env_action}')
        obs, _reward, done, _info = env.step(env_action.tolist())
        trajectory.append(np.asarray(env_action, dtype=float).tolist())
        if done:
            break
    success = bool(env.check_success())
    normal_end = bool(success or done or len(trajectory) >= int(args.max_steps))
    failure_reason = 'success' if success else ('environment_done' if done else 'environment_time_limit')
    result = {
        'protocol_id': PROTOCOL,
        'case_id': case['case_id'],
        'task_id': int(case['task_id']),
        'task_name': task.name,
        'instruction': task.language,
        'arm': arm,
        'init_state_id': int(case['init_state_id']),
        'init_state_sha256': initial_state_sha,
        'case_seed': int(case_seed),
        'env_seed': int(args.env_seed),
        'settle_steps': int(args.settle_steps),
        'max_policy_steps': int(args.max_steps),
        'success': success,
        'normal_end': normal_end,
        'done': bool(done),
        'failure_reason': failure_reason,
        'steps': len(trajectory),
        'runtime_seconds': time.monotonic() - started,
        'checkpoint_revision': checkpoint_revision,
        'code_commit': commit,
        'config_sha256': config_hash,
        'worker_id': args.worker_id,
        'physical_gpu': int(args.physical_gpu),
        'render_gpu': int(args.render_gpu if args.render_gpu is not None else args.physical_gpu),
        'trajectory': trajectory,
        'trace': trace,
    }
    out = artifact / 'episodes' / task.name / arm / f"init_{int(case['init_state_id']):03d}.json"
    atomic_json(out, result)
    if args.save_video and frames:
        video = artifact / 'videos' / task.name / arm / f"init_{int(case['init_state_id']):03d}.mp4"
        encode_video(frames, video, fps=30)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-revision', required=True)
    parser.add_argument('--unnorm-key', default='libero_90_no_noops')
    parser.add_argument('--entity-mode', default='source_target_libero90')
    parser.add_argument('--query-mode', default='instruction_only')
    parser.add_argument('--attention-layers', default='11')
    parser.add_argument('--lambda-scale', type=float, default=1.0)
    parser.add_argument('--env-seed', type=int, default=0)
    parser.add_argument('--settle-steps', type=int, default=10)
    parser.add_argument('--max-steps', type=int, default=400)
    parser.add_argument('--physical-gpu', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--render-gpu', type=int, choices=(1, 2, 3), default=None,
                        help='physical GPU used for MuJoCo/EGL; defaults to --physical-gpu')
    parser.add_argument('--worker-id', required=True)
    parser.add_argument('--max-cases', type=int, default=0, help='0 means drain the queue')
    parser.add_argument('--save-video', action='store_true')
    parser.add_argument('--recover-running', action='store_true')
    args = parser.parse_args()
    args.artifact = args.artifact.resolve()
    args.config = args.config.resolve()
    render_gpu = int(args.render_gpu if args.render_gpu is not None else args.physical_gpu)
    visible = [int(value) for value in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if value]
    if args.physical_gpu not in visible:
        raise RuntimeError(f'physical GPU {args.physical_gpu} is not in CUDA_VISIBLE_DEVICES={visible}')
    if render_gpu not in visible:
        raise RuntimeError(f'render GPU {render_gpu} is not in CUDA_VISIBLE_DEVICES={visible}')
    if os.environ.get('MUJOCO_EGL_DEVICE_ID') != str(render_gpu):
        raise RuntimeError(
            f'MUJOCO_EGL_DEVICE_ID must be {render_gpu}, got {os.environ.get("MUJOCO_EGL_DEVICE_ID")!r}'
        )
    model_device_index = visible.index(args.physical_gpu)

    if args.recover_running:
        running = args.artifact / 'cases' / 'running'
        pending = args.artifact / 'cases' / 'pending'
        for path in sorted(running.glob('*.json')):
            os.replace(path, pending / path.name.split('__worker')[0] + '.json')

    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    config_hash = sha256_file(args.config)
    commit = code_commit()
    suite = benchmark.get_benchmark_dict()['libero_90']()
    model, processor = load_policy(
        args.checkpoint,
        Path('/home/leju-suzhou/zjt_ws/token-cd/third_party/openvla/prismatic/extern/hf'),
        device=f'cuda:{model_device_index}',
        dataset_statistics_path=args.checkpoint / 'dataset_statistics.json',
        unnorm_key=args.unnorm_key,
    )
    log_path = args.artifact / 'logs' / f'worker_{args.worker_id}.jsonl'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    while args.max_cases <= 0 or processed < args.max_cases:
        claimed = claim_case(args.artifact, args.worker_id)
        if claimed is None:
            break
        case, running_path = claimed
        started = time.monotonic()
        try:
            task_id = int(case['task_id'])
            task = suite.get_task(task_id)
            if task.name != case['task_name']:
                raise RuntimeError(f"task mismatch for id {task_id}: {task.name} != {case['task_name']}")
            init_states = suite.get_task_init_states(task_id)
            init_id = int(case['init_state_id'])
            init_state = init_states[init_id]
            bddl = str(Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file)
            case_seed = 20260922 + task_id * 10000 + init_id
            results = {}
            for arm in ('vanilla', 'matched'):
                env = OffScreenRenderEnv(
                    bddl_file_name=bddl,
                    camera_heights=256,
                    camera_widths=256,
                )
                try:
                    result = run_arm(
                        arm=arm,
                        env=env,
                        task=task,
                        init_state=init_state,
                        model=model,
                        processor=processor,
                        artifact=args.artifact,
                        case=case,
                        case_seed=case_seed,
                        config_hash=config_hash,
                        checkpoint_revision=args.checkpoint_revision,
                        commit=commit,
                        args=args,
                    )
                finally:
                    env.close()
                results[arm] = result
            if results['vanilla']['init_state_sha256'] != results['matched']['init_state_sha256']:
                raise RuntimeError('paired initial-state hash mismatch')
            pair = {
                'protocol_id': PROTOCOL,
                'case_id': case['case_id'],
                'task_id': task_id,
                'task_name': task.name,
                'instruction': task.language,
                'init_state_id': init_id,
                'case_seed': case_seed,
                'init_state_sha256': results['vanilla']['init_state_sha256'],
                'vanilla_success': results['vanilla']['success'],
                'matched_success': results['matched']['success'],
                'vanilla_steps': results['vanilla']['steps'],
                'matched_steps': results['matched']['steps'],
                'vanilla_runtime_seconds': results['vanilla']['runtime_seconds'],
                'matched_runtime_seconds': results['matched']['runtime_seconds'],
                'matched_mean_m': (
                    float(np.mean([x['m_effective'] for x in results['matched']['trace']]))
                    if results['matched']['trace'] else None
                ),
                'worker_id': args.worker_id,
                'physical_gpu': args.physical_gpu,
                'render_gpu': int(args.render_gpu if args.render_gpu is not None else args.physical_gpu),
            }
            atomic_json(args.artifact / 'pairs' / f"{case['case_id']}.json", pair)
            complete_case(running_path, args.artifact)
            with log_path.open('a') as handle:
                handle.write(json.dumps({
                    'case_id': case['case_id'],
                    'status': 'complete',
                    'seconds': time.monotonic() - started,
                    **pair,
                }, sort_keys=True) + '\n')
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}'
            fail_case(running_path, args.artifact, error)
            with log_path.open('a') as handle:
                handle.write(json.dumps({
                    'case_id': case.get('case_id'),
                    'status': 'error',
                    'seconds': time.monotonic() - started,
                    'error': error,
                }, sort_keys=True) + '\n')
        processed += 1

if __name__ == '__main__':
    main()
