"""Export exact frozen observations from an existing engineering snapshot.

Independent of the formal runner; writes ONLY a new manifest directory. Each row
is a real first observation (RGB + eef_pos + instruction) captured after the same
reset/restore sequence the formal runner uses, so the profiling worker replays a
real state instead of a guessed one. Physical GPU guard 1/2/3, read-only on
snapshots, never writes inside the artifact except the new manifest folder.
"""
from pathlib import Path
import argparse, hashlib, json, os, pickle, sys

ROOT = Path(__file__).resolve().parents[1]
PCD = ROOT.parent / 'pcd_openvla_simpler_box_31b027e/source'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifact', type=Path, required=True)
    ap.add_argument('--gpu', type=int, choices=(1, 2, 3), required=True)
    ap.add_argument('--phase', choices=('engineering', 'formal'), default='engineering')
    ap.add_argument('--task', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    a = ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(a.gpu):
        raise RuntimeError('CUDA_VISIBLE_DEVICES must already be exactly physical --gpu 1/2/3')
    if 'torch' in sys.modules:
        raise RuntimeError('torch imported before the physical GPU guard; use a fresh process')
    art = a.artifact.resolve()
    lock = json.loads((art / 'CONFIG_LOCK.json').read_text())
    if a.gpu not in lock['allowed_gpus']:
        raise RuntimeError('gpu not authorized by CONFIG_LOCK')
    if a.task not in lock['tasks'] or a.seed not in lock[f'{a.phase}_seeds']:
        raise RuntimeError('task/seed not in frozen manifest')
    snap = art / 'snapshots' / a.phase / a.task / f'seed_{a.seed:03d}.pkl'
    meta = snap.with_name(snap.stem + '_meta.json')
    raw = snap.read_bytes()
    m = json.loads(meta.read_text())
    if digest(raw) != m['snapshot_file_sha256']:
        raise RuntimeError('snapshot file digest mismatch')
    out = a.out_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f'refuse to write into non-empty {out}')
    out.mkdir(parents=True, exist_ok=True)
    for path in (ROOT / 'task1/shim_site', ROOT, PCD):
        sys.path.insert(0, str(path))
    os.environ['PCD_SOURCE'] = str(PCD)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    import numpy as np
    from research.semantic_token_cd import prompt_attn_l11_budget_causal_rollout as shared
    from research.semantic_token_cd.distractor_rollout import (
        capture_snapshot, get_image_from_maniskill2_obs_dict, restore_snapshot, snapshot_sha)
    snapshot = pickle.loads(raw)  # locally generated trusted pickle only
    if snapshot_sha(snapshot) != m['canonical_snapshot_sha256']:
        raise RuntimeError('canonical snapshot hash mismatch')
    env, env_id = shared.make_environment(a.task)
    try:
        env.reset(seed=a.seed)
        obs, sh, rh = restore_snapshot(env, a.seed, snapshot)
        if sh != m['initial_state_sha256'] or rh != m['initial_rgb_sha256']:
            raise RuntimeError('strict initial state/RGB hash mismatch')
        instruction = env.unwrapped.get_language_instruction()
        if instruction != m['instruction']:
            raise RuntimeError('instruction mismatch')
        import imageio.v2 as imageio
        image = np.asarray(get_image_from_maniskill2_obs_dict(env, obs))
        proprio = np.asarray(obs['agent']['eef_pos'], dtype=np.float32)
        if proprio.shape != (3,):
            raise RuntimeError(f'unexpected eef_pos shape {proprio.shape}')
        image_path = out / f'{a.task}_seed{a.seed:03d}_rgb.png'
        imageio.imwrite(str(image_path), image)
        rows = [{'image_path': image_path.name, 'proprio': [float(v) for v in proprio],
                 'instruction': instruction, 'task': a.task}]
        provenance = {'protocol_id': lock['protocol_id'], 'phase': a.phase, 'task': a.task,
                      'seed': a.seed, 'physical_gpu': a.gpu, 'environment_id': env_id,
                      'snapshot_file_sha256': m['snapshot_file_sha256'],
                      'canonical_snapshot_sha256': m['canonical_snapshot_sha256'],
                      'initial_state_sha256': sh, 'initial_rgb_sha256': rh,
                      'rgb_file_sha256': digest(image_path.read_bytes()),
                      'reset_sequence': m['reset_sequence'],
                      'scope': 'single restored first observation; not a trajectory replay',
                      'proprio_source': 'obs["agent"]["eef_pos"] saved verbatim'}
    finally:
        env.close()
    (out / 'states.json').write_text(json.dumps(rows, indent=2) + '\n')
    (out / 'states_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'states': str(out / 'states.json'), 'rows': len(rows),
                      'rgb_sha256': provenance['rgb_file_sha256']}))


if __name__ == '__main__':
    main()
