#!/usr/bin/env python3
"""Single task/seed/arm fresh-process worker, never a launcher.

Initial state/RGB hashes certify one frame, NOT trajectory equivalence.
CONFIG_LOCK is read-only. Debug-capped engineering episodes are never formal.
"""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PCD = ROOT.parent / 'pcd_openvla_simpler_box_31b027e/source'
PROTOCOL = 'VTCG_FACTORIAL_CONTROLS_V1'
TASKS = ('google_robot_open_drawer', 'google_robot_close_drawer',
         'google_robot_pick_coke_can', 'google_robot_move_near')
ARMS = ('attn_matched', 'attn_fixed', 'attn_topp', 'kmeans_matched',
        'kmeans_fixed', 'kmeans_topp', 'vanilla', 'random_matched', 'norm_noise')
_ENTERED = False


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def boolean(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    if value.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError('expected true/false')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--task', choices=TASKS, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--gpu', type=int, choices=(1, 2, 3), required=True)
    p.add_argument('--snapshot-mode', choices=('create', 'run'), required=True)
    p.add_argument('--phase', choices=('engineering', 'formal'), required=True)
    p.add_argument('--run-config-id', required=True)
    p.add_argument('--max-steps', type=int, help='engineering preflightdebug ONLY; never formal')
    p.add_argument('--video', nargs='?', const=True, default=False, type=boolean)
    a = p.parse_args(argv)
    if a.seed < 0:
        p.error('seed must be nonnegative; 300+ supported')
    if a.max_steps is not None and (a.max_steps <= 0 or a.phase != 'engineering'):
        p.error('--max-steps is positive engineering preflightdebug ONLY')
    return a


def validate_lock(a):
    raw = (a.artifact / 'CONFIG_LOCK.json').read_bytes()
    lock = json.loads(raw)
    if lock.get('protocol_id') != PROTOCOL:
        raise ValueError('CONFIG_LOCK protocol mismatch')
    if lock.get('run_config_ids', {}).get(a.phase) != a.run_config_id:
        raise ValueError('run_config_id missing or mismatches episode phase')
    for key, value in (('tasks', a.task), ('arms', a.arm),
                       (f'{a.phase}_seeds', a.seed), ('allowed_gpus', a.gpu)):
        if value not in lock.get(key, []):
            raise ValueError(f'{key}={value!r} not frozen for this phase')
    for key, value in {'fixed_m': 48, 'top_p': .85, 'top_p_min': 16,
                       'top_p_max': 64, 'lambda': .5, 'renderer_spp': 32,
                       'renderer_denoiser': True}.items():
        if lock.get(key) != value:
            raise ValueError(f'unsupported frozen {key}')
    if lock.get('renderer_depth', {}).get(a.task) != (8 if a.task in TASKS[:2] else 0):
        raise ValueError('renderer depth mismatch')
    return lock, raw


def atomic_bytes(path, data, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    if exclusive:
        try:
            os.link(tmp, path)
        finally:
            tmp.unlink()
    else:
        os.replace(tmp, path)


def atomic_json(path, payload, exclusive=False):
    atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True,
                                  allow_nan=False) + '\n').encode(), exclusive)


@contextlib.contextmanager
def claim(path):
    # Never unlink this inode: avoids flock/unlink/recreate races.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'another process owns {path}') from exc
        yield


def source_hashes():
    paths = [Path(__file__), PCD / 'properties.py',
             PCD / 'simpler_env/policies/openvla/openvla_model.py']
    for name in ('vtcg_factorial_policy.py', 'vtcg_factorial_math.py',
                 'prompt_attn_shr_policy.py', 'semantic_recon_rollout.py',
                 'prompt_attn_l11_budget_causal_rollout.py', 'spatial_grid_rollout.py',
                 'rollout_pilot.py', 'distractor_rollout.py'):
        paths.append(Path(__file__).parent / name)
    return {str(p): digest(p.read_bytes()) for p in paths if p.is_file()}


def audit_trace(trace, arm):
    if not trace:
        raise RuntimeError('empty policy trace')
    for row in trace:
        expected = 'none' if arm == 'vanilla' else (
            'norm_matched_random_direction' if arm == 'norm_noise' else 'harmonic')
        checks = [row.get('arm') == arm, row.get('feature_equal') is True,
                  row.get('non_target_bit_identical') is True,
                  row.get('counterfactual_constructor') == expected,
                  row.get('guided_prefix') is True,
                  row.get('attention_layers') == [11],
                  row.get('lambda') == .5, row.get('beta') == 0,
                  row.get('intervention_location') == 'projector',
                  len(row.get('positive_token_ids', [])) == 7,
                  len(row.get('final_token_ids', [])) == 7,
                  row.get('positive_token_ids', [None])[-1:] == row.get('final_token_ids', [])[-1:]]
        if arm != 'vanilla':
            checks += [row.get('coverage_exact') is True,
                       row.get('reconstruction_finite') is True,
                       row.get('negative_branch_executed') is True,
                       row.get('constructor_audit', {}).get('applied_non_target_bit_identical') is True]
        if arm == 'norm_noise':
            checks += [row.get('constructor_audit', {}).get('float32_norms_match') is True]
        if not all(checks):
            raise RuntimeError(f'factorial technical audit failed at step {row.get("step_index")}: {checks}')
    return {'technical_pass': True, 'audited_control_steps': len(trace)}


def serialize_records(records):
    """Strict native fp32 scores; preserve audit dtypes and ragged per-step data."""
    import numpy as np
    if not records:
        raise RuntimeError('missing policy records')
    payload = {}
    for key in ('positive', 'negative', 'guided'):
        values = [np.asarray(row[key]) for row in records]
        if any(v.shape != (7, 256) or v.dtype != np.float32 or not np.isfinite(v).all() for v in values):
            raise RuntimeError(f'{key} must be finite native float32 [7,256]')
        payload[key + '_logits'] = np.stack(values)
    auxiliary = set().union(*(row.keys() for row in records)) - {'positive', 'negative', 'guided'}
    for key in sorted(auxiliary):
        values = [np.asarray(row[key]) for row in records if key in row]
        if len(values) == len(records) and len({(v.shape, v.dtype.str) for v in values}) == 1:
            payload[key] = np.stack(values)
        else:
            # Non-object entries retain ragged features/IDs and sparse feature steps.
            for index, row in enumerate(records):
                if key in row:
                    payload[f'step_{index:04d}__{key}'] = np.asarray(row[key])
    if any(value.dtype.hasobject for value in payload.values()):
        raise RuntimeError('object arrays forbidden')
    return payload


def run_episode(env, policy, instruction, obs, writer, max_steps):
    """Mirror spatial_grid_rollout.run_episode, including existing termination."""
    import numpy as np
    from research.semantic_token_cd.distractor_rollout import flatten_action, get_image_from_maniskill2_obs_dict
    from utils import convert_numpy_or_torch_to_python, stat_final, stat_first, summarize
    predicted_terminated = truncated = False
    timestep = 0
    infos, actions_out = [], []
    image = get_image_from_maniskill2_obs_dict(env, obs)
    initial = np.asarray(image).copy()
    if writer is not None:
        writer.append_data(initial)
    while not (predicted_terminated or truncated):
        _, actions, _ = policy.step(image, None, instruction, proprio=obs['agent']['eef_pos'])
        if not isinstance(actions, list):
            actions = [actions]
        if not actions:
            raise RuntimeError('policy returned no actions')
        for action in actions:
            executed = flatten_action(action)
            if executed.shape != (7,) or not np.isfinite(executed).all():
                raise FloatingPointError(f'Invalid executed action: {executed}')
            actions_out.append(executed.copy())
            obs, _reward, _success, truncated, info = env.step(executed)
            image = get_image_from_maniskill2_obs_dict(env, obs)
            if writer is not None:
                writer.append_data(np.asarray(image))
            timestep += 1
            infos.append(convert_numpy_or_torch_to_python(info))
            predicted_terminated = bool(action['terminate_episode'][0] > 0)
            if max_steps is not None and timestep >= max_steps:
                truncated = True
                break
            if predicted_terminated and not env.unwrapped.is_final_subtask():
                predicted_terminated = False
                env.advance_to_next_subtask()
            instruction = env.unwrapped.get_language_instruction()
    result = summarize(infos)
    result.update(stat_first(infos))
    result.update(stat_final(infos))
    reason = None if result['success'] else ('environment_time_limit' if truncated else 'policy_terminated_without_success')
    actions_array = np.asarray(actions_out, dtype=np.float32)
    jitter = float(np.linalg.norm(np.diff(actions_array, axis=0), axis=1).mean()) if len(actions_array) > 1 else 0.
    return result, timestep, reason, actions_array, jitter, infos, initial, np.asarray(image).copy()


def main(argv=None):
    global _ENTERED
    if _ENTERED:
        raise RuntimeError('execute once per fresh process')
    _ENTERED = True
    a = parse_args(argv)
    a.artifact = a.artifact.resolve()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(a.gpu):
        raise RuntimeError('CUDA_VISIBLE_DEVICES must already be exactly physical --gpu 1/2/3')
    if 'torch' in sys.modules:
        raise RuntimeError('torch imported before physical GPU guard; use fresh process')
    lock, lock_raw = validate_lock(a)
    for path in (ROOT / 'task1/shim_site', ROOT, PCD):
        sys.path.insert(0, str(path))
    os.environ['PCD_SOURCE'] = str(PCD)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    import numpy as np
    from research.semantic_token_cd import prompt_attn_l11_budget_causal_rollout as shared
    from research.semantic_token_cd.distractor_rollout import capture_snapshot, jsonable, restore_snapshot, snapshot_sha
    snapshot_path = a.artifact / 'snapshots' / a.phase / a.task / f'seed_{a.seed:03d}.pkl'
    meta_path = snapshot_path.with_name(snapshot_path.stem + '_meta.json')
    provenance = {'protocol_id': PROTOCOL, 'run_config_id': a.run_config_id, 'phase': a.phase,
                  'task': a.task, 'seed': a.seed, 'physical_gpu': a.gpu,
                  'config_lock_sha256': digest(lock_raw), 'source_sha256': source_hashes(),
                  'renderer_depth': lock['renderer_depth'][a.task], 'renderer_spp': 32,
                  'renderer_denoiser': True, 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
                  'hash_scope': 'single restored initial state and RGB frame ONLY; NOT trajectory equivalence',
                  'reset_sequence': 'fresh_environment; reset(seed); restore_snapshot(reset(seed),set_state,agent,rng,get_obs)'}
    env = None
    try:
        env, environment_id = shared.make_environment(a.task)
        provenance['environment_id'] = environment_id
        if a.snapshot_mode == 'create':
            with claim(snapshot_path.with_suffix('.create.lock')):
                if snapshot_path.exists() or meta_path.exists():
                    raise FileExistsError('create never overwrites snapshot or metadata')
                snapshot = capture_snapshot(env, a.seed)  # first reset
                obs, sh, rh = restore_snapshot(env, a.seed, snapshot)  # second reset
                if env.unwrapped.get_language_instruction() != snapshot['instruction']:
                    raise RuntimeError('snapshot instruction mismatch')
                raw = pickle.dumps(snapshot, protocol=pickle.HIGHEST_PROTOCOL)
                meta = dict(provenance, instruction=snapshot['instruction'],
                            initial_state_sha256=sh, initial_rgb_sha256=rh,
                            canonical_snapshot_sha256=snapshot_sha(snapshot), snapshot_file_sha256=digest(raw))
                if (a.artifact / 'CONFIG_LOCK.json').read_bytes() != lock_raw:
                    raise RuntimeError('CONFIG_LOCK changed during create')
                atomic_bytes(snapshot_path, raw, exclusive=True)
                atomic_json(meta_path, meta, exclusive=True)
                print(json.dumps({'snapshot': str(snapshot_path), 'meta': str(meta_path)}))
            return
        arm_dir = a.artifact / 'episodes' / a.phase / a.task / a.arm
        prefix = arm_dir / f'episode_{a.seed:03d}'
        summary = Path(str(prefix) + '_summary.json')
        with claim(Path(str(prefix) + '.lock')):
            if summary.exists():
                raise FileExistsError(f'completed episode already exists: {summary}')
            raw = snapshot_path.read_bytes()
            meta = json.loads(meta_path.read_bytes())
            for key in ('protocol_id', 'run_config_id', 'phase', 'task', 'seed',
                        'renderer_depth', 'renderer_spp', 'renderer_denoiser', 'environment_id', 'reset_sequence'):
                if meta.get(key) != provenance[key]:
                    raise RuntimeError(f'snapshot metadata mismatch: {key}')
            if digest(raw) != meta['snapshot_file_sha256']:
                raise RuntimeError('snapshot file digest mismatch')
            snapshot = pickle.loads(raw)  # trusted locally generated pickle only
            if snapshot_sha(snapshot) != meta['canonical_snapshot_sha256']:
                raise RuntimeError('canonical hash mismatch')
            env.reset(seed=a.seed)  # match create capture(reset) before restore(reset)
            obs, sh, rh = restore_snapshot(env, a.seed, snapshot)
            if sh != meta['initial_state_sha256'] or rh != meta['initial_rgb_sha256']:
                raise RuntimeError('strict initial state/RGB hash mismatch')
            instruction = env.unwrapped.get_language_instruction()
            if instruction != meta['instruction']:
                raise RuntimeError('instruction mismatch')
            import torch
            torch.set_num_threads(1)
            from properties import get_policy_config
            from simpler_env.policies.openvla.openvla_model import OpenVLAInference
            from research.semantic_token_cd.vtcg_factorial_policy import build_policy
            config = get_policy_config('openvla', str(PCD / 'pretrained/openvla-7b'), a.task, {}, False)
            base = OpenVLAInference(**config)  # one model load, one arm
            policy = build_policy(base, a.task, a.arm, fixed_m=48, top_p=.85, lambd=.5,
                                  record_features=a.phase == 'engineering', record_feature_steps=2)
            policy.reset(instruction, seed=a.seed)
            policy._episode_trace, policy._episode_logits = [], []
            policy._episode_seed, policy._selector_step = a.seed, 0
            video_enabled = a.video or a.arm in ('vanilla', 'attn_matched')
            video_path = Path(str(prefix) + '.mp4')
            started = time.monotonic()
            with contextlib.ExitStack() as stack:
                writer = None
                if video_enabled:
                    import imageio.v2 as imageio
                    writer = stack.enter_context(imageio.get_writer(str(video_path), fps=5, macro_block_size=1))
                result, steps, reason, actions, jitter, infos, initial, final = run_episode(
                    env, policy, instruction, obs, writer, a.max_steps)
            trace, records = policy._episode_trace, policy._episode_logits
            if len(records) != len(trace):
                raise RuntimeError('trace/logit record alignment mismatch')
            technical = audit_trace(trace, a.arm)
            arrays = serialize_records(records)
            arrays.update(executed_actions=actions, initial_rgb=initial, final_rgb=final)
            arrays_path = Path(str(prefix) + '_arrays.npz')
            with arrays_path.open('wb') as f:
                np.savez_compressed(f, **arrays)
            atomic_json(Path(str(prefix) + '_trace.json'), jsonable(trace))
            atomic_json(Path(str(prefix) + '_step_infos.json'), jsonable(infos))
            import imageio.v2 as imageio
            imageio.imwrite(str(prefix) + '_initial.png', initial)
            imageio.imwrite(str(prefix) + '_final.png', final)
            payload = dict(provenance, arm=a.arm, instruction=instruction, success=bool(result['success']),
                           result=jsonable(result), technical_pass=True, technical_audit=technical,
                           steps=steps, failure_reason=reason, action_jitter_index=jitter,
                           runtime_seconds=time.monotonic() - started, policy_config=jsonable(config),
                           preflightdebug=a.max_steps is not None, max_steps_debug=a.max_steps,
                           formal_eligible=a.phase == 'formal' and a.max_steps is None,
                           canonical_snapshot_sha256=meta['canonical_snapshot_sha256'],
                           initial_state_sha256=sh, initial_rgb_sha256=rh,
                           fixed_m=48, top_p=.85, top_p_min=16, top_p_max=64, lambd=.5,
                           video=str(video_path) if video_enabled else None,
                           arrays_file_sha256=digest(arrays_path.read_bytes()))
            if (a.artifact / 'CONFIG_LOCK.json').read_bytes() != lock_raw:
                raise RuntimeError('CONFIG_LOCK changed during run')
            # Final atomic no-clobber completion marker; exceptions never write summary.
            atomic_json(summary, jsonable(payload), exclusive=True)
            print(json.dumps({'summary': str(summary), 'success': payload['success']}))
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
