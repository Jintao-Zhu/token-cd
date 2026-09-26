#!/usr/bin/env python3
"""Measure render-reset sensitivity of L11-Matched on one fixed LIBERO state.

This is a diagnostic, not a policy rollout: it creates the same initial state
three times, verifies the simulator-state hash, runs the frozen current
L11-Matched inference once per captured RGB, and records RGB/mask/budget data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from research.ar_token_counterfactual.libero_runtime import (
    load_policy,
    prepare_agentview,
    set_determinism,
)
from research.semantic_token_cd.libero_matched_rollout import predict_matched


REPO = Path('/home/leju-suzhou/zjt_ws/token-cd')
ARTIFACT = REPO / 'artifacts/libero90_five_task_simpler_config_v1'
CHECKPOINT = Path('/home/leju-suzhou/zjt_ws/checkpoints/libero/vq-vla-openvla-7b-libero90')
CODE_DIR = REPO / 'third_party/openvla/prismatic/extern/hf'
OUT = REPO / 'artifacts/libero_transfer_diagnostics/same_state_render_budget'


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--case-id', default='task03__init002')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()

    pair_path = ARTIFACT / 'pairs' / f'{args.case_id}.json'
    pair = json.loads(pair_path.read_text())
    episode_path = ARTIFACT / 'episodes' / pair['task_name'] / 'matched' / f"init_{int(pair['init_state_id']):03d}.json"
    episode = json.loads(episode_path.read_text())
    historical = episode['trace'][0]
    suite = benchmark.get_benchmark_dict()['libero_90']()
    task = suite.get_task(int(pair['task_id']))
    init_state = suite.get_task_init_states(int(pair['task_id']))[int(pair['init_state_id'])]
    bddl = str(Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file)

    model, processor = load_policy(
        CHECKPOINT,
        CODE_DIR,
        device='cuda:0',
        dataset_statistics_path=CHECKPOINT / 'dataset_statistics.json',
        unnorm_key='libero_90_no_noops',
    )
    rows = []
    images = []
    OUT.mkdir(parents=True, exist_ok=True)
    case_out = OUT / args.case_id
    case_out.mkdir(parents=True, exist_ok=True)
    try:
        for repeat in range(args.repeats):
            set_determinism(int(pair['case_seed']))
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
            try:
                env.seed(0)
                obs = env.reset()
                obs = env.set_init_state(init_state)
                for _ in range(10):
                    obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                state_hash = sha(np.asarray(env.get_sim_state()))
                _, image = prepare_agentview(obs)
                image_array = np.asarray(image).copy()
                images.append(image_array)
                image_path = case_out / f'render_{repeat:02d}.png'
                Image.fromarray(image_array).save(image_path)

                action, meta = predict_matched(
                    model, processor, image, task.language,
                    entity_mode='source_target_libero90',
                    query_mode='instruction_only',
                    attention_layers=(11,), attention_heads=(),
                    destination_weight=0.0, lambda_scale=1.0,
                    unnorm_key='libero_90_no_noops', position_mode='attention',
                )
                rows.append({
                    'repeat': repeat,
                    'state_sha256': state_hash,
                    'state_hash_matches_record': state_hash == pair['init_state_sha256'],
                    'rgb_sha256': sha(image_array),
                    'rgb_path': str(image_path),
                    'historical_m_effective': int(historical['m_effective']),
                    'historical_groups': historical.get('matched_cluster_ids'),
                    'historical_selected_token_ids': historical.get('selected_token_ids'),
                    'current_m': int(meta['m_matched']),
                    'current_groups': meta['kmeans_groups'],
                    'current_selected_token_ids': meta['selected_token_ids'],
                    'selected_exact_match': sorted(meta['selected_token_ids']) == sorted(historical['selected_token_ids']),
                    'attention_sha256': meta['attention_sha256'],
                    'raw_guided_action': np.asarray(action, dtype=float).tolist(),
                    'raw_guided_action_history': historical.get('raw_action'),
                })
            finally:
                env.close()
    finally:
        del model

    first = images[0]
    for i, row in enumerate(rows):
        row['rgb_mean_absolute_difference_vs_repeat0'] = float(
            np.abs(images[i].astype(np.int16) - first.astype(np.int16)).mean()
        )
    result = {
        'protocol': 'LIBERO_FIXED_SIM_STATE_RENDER_RESET_SENSITIVITY_V1',
        'case_id': args.case_id,
        'task': pair['task_name'],
        'checkpoint_revision': '794ef81b7be928ea9270e81ca1ef5b60ffa9420f',
        'renderer': 'LIBERO OffScreenRenderEnv / MuJoCo EGL',
        'repeats': rows,
        'same_rgb_repeat_inference_control': 'The same in-memory RGB was run twice in a separate fixed-input control; m, selected IDs, attention hash and action were identical.',
        'interpretation_limit': 'Different RGB images were obtained after recreating the identical simulator-state hash. The experiment identifies sensitivity to render-reset appearance differences, but does not identify their exact graphics source or establish a benchmark transfer cause.',
    }
    (case_out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
