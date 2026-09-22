#!/usr/bin/env python3
"""Prepare the frozen five-task LIBERO-90 paired evaluation queue."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path

TASK_IDS = (3, 10, 49, 72, 73)
TASK_LABELS = {
    3: 'butter_at_back_to_top_drawer_and_close',
    10: 'black_bowl_on_top_of_cabinet',
    49: 'tomato_sauce_to_basket',
    72: 'white_mug_on_plate',
    73: 'book_to_front_caddy_compartment',
}
CHECKPOINT_REPO = 'VQ-VLA/openvla-7b-finetuned-libero-90'
CHECKPOINT_REVISION = '794ef81b7be928ea9270e81ca1ef5b60ffa9420f'
UNNORM_KEY = 'libero_90_no_noops'
INIT_STATES_PER_TASK = 50


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    a = p.parse_args()
    a.artifact = a.artifact.resolve()
    a.config = a.config.resolve()
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    from libero.libero import benchmark, get_libero_path

    config = json.loads(a.config.read_text())
    suite = benchmark.get_benchmark_dict()['libero_90']()
    tasks = []
    all_hashes: dict[str, str] = {}
    duplicate_within = []
    for task_id in TASK_IDS:
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        if len(init_states) < INIT_STATES_PER_TASK:
            raise RuntimeError(f'task {task_id} has only {len(init_states)} init states')
        rows = []
        seen = {}
        for init_id in range(INIT_STATES_PER_TASK):
            digest = sha256_bytes(init_states[init_id].tobytes())
            if digest in seen:
                duplicate_within.append({'task_id': task_id, 'init_state_id': init_id, 'duplicate_of': seen[digest]})
            seen[digest] = init_id
            all_hashes[f'{task_id}:{init_id}'] = digest
            rows.append({'init_state_id': init_id, 'sha256': digest})
            case_id = f'task{task_id:02d}__init{init_id:03d}'
            case = {
                'case_id': case_id,
                'task_id': task_id,
                'task_name': task.name,
                'instruction': task.language,
                'init_state_id': init_id,
                'init_state_sha256': digest,
                'label': TASK_LABELS[task_id],
            }
            pending = a.artifact / 'cases' / 'pending' / f'{case_id}.json'
            if pending.exists():
                old = json.loads(pending.read_text())
                if old != case:
                    raise RuntimeError(f'queue case mismatch: {pending}')
            else:
                atomic_json(pending, case)
        tasks.append({
            'task_id': task_id,
            'task_name': task.name,
            'instruction': task.language,
            'label': TASK_LABELS[task_id],
            'problem_folder': task.problem_folder,
            'bddl_file': task.bddl_file,
            'bddl_path': str(Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file),
            'init_states': rows,
            'init_state_count': len(rows),
        })
    manifest = {
        'protocol_id': 'LIBERO90_FIVE_TASK_SIMPLER_CONFIG_V1',
        'suite': 'libero_90',
        'task_ids': list(TASK_IDS),
        'init_states_per_task': INIT_STATES_PER_TASK,
        'paired_cases': len(TASK_IDS) * INIT_STATES_PER_TASK,
        'arms': ['vanilla', 'matched_simpler_config'],
        'checkpoint_repo': CHECKPOINT_REPO,
        'checkpoint_revision': CHECKPOINT_REVISION,
        'unnorm_key': UNNORM_KEY,
        'frozen_config_path': str(a.config),
        'tasks': tasks,
        'init_state_hash_duplicates': duplicate_within,
        'all_initial_state_hashes': all_hashes,
    }
    atomic_json(a.artifact / 'TASK_MANIFEST.json', manifest)
    experiment = {
        'protocol_id': manifest['protocol_id'],
        'purpose': 'paired five-task LIBERO-90 evaluation of the frozen SIMPLER matched configuration',
        'method': config['method'],
        'environment_adaptation': config['environment_adaptation'],
        'expected_episodes': len(TASK_IDS) * INIT_STATES_PER_TASK * 2,
        'paired_cases': len(TASK_IDS) * INIT_STATES_PER_TASK,
        'cases_pending_dir': str(a.artifact / 'cases' / 'pending'),
        'result_dir': str(a.artifact / 'episodes'),
        'checkpoint_revision': CHECKPOINT_REVISION,
        'unnorm_key': UNNORM_KEY,
    }
    atomic_json(a.artifact / 'EXPERIMENT_MANIFEST.json', experiment)
    print(json.dumps({
        'artifact': str(a.artifact),
        'cases': manifest['paired_cases'],
        'episodes': experiment['expected_episodes'],
        'duplicates': duplicate_within,
    }, indent=2))


if __name__ == '__main__':
    main()
