#!/usr/bin/env python3
"""Replay saved OSMesa LIBERO actions and probe all L11 rank-bin negatives."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path('/home/leju-suzhou/zjt_ws/token-cd')
SOURCE_ROOTS = (
    ROOT / 'artifacts/libero_action_value_trace_pilot_v2_osmesa_40pair_20260925',
    ROOT / 'artifacts/libero_action_value_trace_pilot_v2_osmesa_ext40_init8_15_20260925',
    ROOT / 'artifacts/libero_action_value_gate_osmesa_pilot_v1_20260925',
)
CHECKPOINT = Path('/home/leju-suzhou/zjt_ws/checkpoints/libero/vq-vla-openvla-7b-libero90')
CODE_DIR = ROOT / 'third_party/openvla/prismatic/extern/hf'
OSMESA_LIB = ROOT / 'artifacts/libero_osmesa_runtime/root/usr/lib/x86_64-linux-gnu'


def sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def collect_cases() -> list[dict]:
    chosen: dict[tuple[int, int], dict] = {}
    for root in SOURCE_ROOTS:
        for path in sorted(root.glob('episodes/*/vanilla/episode.json')):
            data = json.loads(path.read_text())
            key = (int(data['task_id']), int(data['init_state_id']))
            if key not in chosen:
                chosen[key] = {'path': str(path), 'case': data}
    return [chosen[key] for key in sorted(chosen)]


def sampled_steps(length: int, count: int) -> list[int]:
    if length < 1 or count < 1:
        return []
    return sorted(set(int(x) for x in np.linspace(0, length - 1, min(length, count))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--worker-id', required=True)
    ap.add_argument('--artifact', type=Path, required=True)
    ap.add_argument('--cases-file', type=Path, required=True)
    ap.add_argument('--max-cases', type=int, default=0)
    ap.add_argument('--states-per-episode', type=int, default=10)
    ap.add_argument('--strict-replay', action='store_true')
    args = ap.parse_args()

    os.environ['MUJOCO_GL'] = 'osmesa'
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
    os.environ.pop('MUJOCO_EGL_DEVICE_ID', None)
    if not (OSMESA_LIB / 'libOSMesa.so.8').is_file():
        raise FileNotFoundError(f'OSMesa runtime not found: {OSMESA_LIB}')
    os.environ['LD_LIBRARY_PATH'] = str(OSMESA_LIB) + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    import torch
    torch.set_num_threads(1)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from research.ar_token_counterfactual.libero_runtime import load_policy, prepare_agentview, set_determinism
    from research.semantic_token_cd.l11_rank_causal_utils import evaluate_libero_rank_bins

    artifact = args.artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()['libero_90']()
    model, processor = load_policy(
        CHECKPOINT, CODE_DIR, device='cuda:0',
        dataset_statistics_path=CHECKPOINT / 'dataset_statistics.json',
        unnorm_key='libero_90_no_noops',
    )
    cases = [json.loads(line) for line in args.cases_file.read_text().splitlines() if line.strip()]
    log_path = artifact / 'logs' / f'worker_{args.worker_id}.jsonl'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for number, row in enumerate(cases):
        if args.max_cases and number >= args.max_cases:
            break
        source = Path(row['episode_json'])
        source_episode = json.loads(source.read_text())
        traces = source_episode['trace']
        case_id = source_episode['case_id'] + '__vanilla'
        out_path = artifact / 'episodes' / case_id / 'rank_effects.json'
        if out_path.is_file():
            continue
        task_id = int(source_episode['task_id'])
        task = suite.get_task(task_id)
        init_id = int(source_episode['init_state_id'])
        init = suite.get_task_init_states(task_id)[init_id]
        bddl = str(Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        try:
            set_determinism(int(source_episode['case_seed']))
            env.seed(int(source_episode['env_seed']))
            env.reset()
            obs = env.set_init_state(init)
            for _ in range(int(source_episode['settle_steps'])):
                obs, _, done, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                if done:
                    raise RuntimeError('environment ended during settling')
            steps = sampled_steps(len(traces), args.states_per_episode)
            sample_set = set(steps)
            rows = []
            mismatch_count = 0
            for t, trace in enumerate(traces):
                if t in sample_set:
                    state_hash = sha(np.asarray(env.get_sim_state()))
                    rgb, image = prepare_agentview(obs)
                    rgb_hash = sha(np.asarray(image.convert('RGB'), dtype=np.uint8))
                    state_match = state_hash == trace['sim_state_sha256']
                    rgb_match = rgb_hash == trace['rgb_sha256']
                    mismatch_count += int(not state_match or not rgb_match)
                    if args.strict_replay and not (state_match and rgb_match):
                        raise RuntimeError(
                            f'replay mismatch {case_id} step={t}: state={state_match} rgb={rgb_match}'
                        )
                    if state_match and rgb_match:
                        result = evaluate_libero_rank_bins(
                            model, processor, image, source_episode['instruction'],
                            unnorm_key='libero_90_no_noops',
                        )
                        rows.append({
                            'step': t, 'sim_state_sha256': state_hash,
                            'rgb_sha256': rgb_hash, 'replay_state_match': True,
                            'replay_rgb_match': True, 'rank_bins': result['rank_bins'],
                            'attention_sha256': result['attention_sha256'],
                        })
                    else:
                        rows.append({
                            'step': t, 'sim_state_sha256': state_hash,
                            'rgb_sha256': rgb_hash, 'replay_state_match': state_match,
                            'replay_rgb_match': rgb_match, 'rank_bins': None,
                        })
                obs, _, done, _ = env.step(trace['executed_env_action'])
                if done and t + 1 < len(traces):
                    raise RuntimeError(f'environment ended early at step {t} before trace end')
            payload = {
                'protocol_id': 'L11_RANK_CAUSAL_EFFECT_OFFLINE_V1',
                'benchmark': 'LIBERO-90', 'renderer': 'OSMesa CPU',
                'task_id': task_id, 'task_name': task.name, 'init_state_id': init_id,
                'case_seed': int(source_episode['case_seed']), 'source_episode': str(source),
                'source_arm': 'vanilla', 'source_success': bool(source_episode['success']),
                'source_steps': len(traces), 'sampled_steps': steps,
                'integrity': {
                    'sampled_state_count': len(steps), 'replay_mismatches': mismatch_count,
                    'valid_rank_effect_states': sum(x.get('rank_bins') is not None for x in rows),
                },
                'rank_effect_trace': rows, 'worker_id': args.worker_id,
                'inference_gpu': args.gpu,
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
            os.replace(tmp, out_path)
            line = {'status': 'complete', 'case_id': case_id, **payload['integrity']}
            with log_path.open('a') as handle:
                handle.write(json.dumps(line, sort_keys=True) + '\n')
            print(json.dumps(line, sort_keys=True), flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    main()
