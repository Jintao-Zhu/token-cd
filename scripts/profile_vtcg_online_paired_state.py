#!/usr/bin/env python3
"""Fresh-process initial-state-only online paired profiling; NOT stage profiling.

One existing factorial snapshot / task / seed / method / trial per process.
Never advances the simulator or modifies a formal rollout. Restores the same
snapshot with explicit reset(seed) then restore_snapshot (which resets again).
Run externally in fresh, sequential processes with CUDA_DEVICE_ORDER=PCI_BUS_ID
and CUDA_VISIBLE_DEVICES exactly --gpu (1/2/3). No automatic launcher.

All methods measure reset + optional GT/LaMa generate + step, synchronized.
VTCG and vanilla use the instrumented factorial audit path, NOT deployment speed.
PCD uses the existing unmodified GT+LaMa/OpenVLAContrastInference path (alpha=.2),
so instrumentation and decoding differences prohibit a pure algorithm-overhead
claim. This is ONE INITIAL FRAME ONLY, not phase/stage/trajectory profiling.

Trusted local pickle snapshots only. No unsafe torch.load global monkeypatch:
if the existing LaMa loader is incompatible with the installed torch version,
this worker fails without producing a report; use a compatible trusted runtime.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import time

# Sibling helper has stdlib-only imports, no CUDA side effects.
from profile_vtcg_factorial import (ROOT, WORKSPACE, PCD, assert_exclusive,
                                    gpu_claim, gpu_identity, memory, sha, summary)

_ENTERED = False
TASKS = ('google_robot_open_drawer', 'google_robot_close_drawer',
         'google_robot_pick_coke_can', 'google_robot_move_near')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--artifact', type=Path, required=True, help='existing factorial artifact with snapshots/<phase>/<task>/seed_NNN.pkl')
    p.add_argument('--phase', choices=('engineering', 'formal'), required=True)
    p.add_argument('--task', choices=TASKS, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--method', choices=('vanilla', 'vtcg', 'pcd'), required=True)
    p.add_argument('--gpu', type=int, choices=(1, 2, 3), required=True)
    p.add_argument('--trial-id', required=True)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--repeats', type=int, default=50)
    p.add_argument('--output-dir', type=Path, required=True, help='new directory; never overwrites')
    p.add_argument('--checkpoint', type=Path, default=PCD / 'pretrained/openvla-7b')
    a = p.parse_args(argv)
    if a.seed < 0 or min(a.warmup, a.repeats) < 1:
        p.error('nonnegative seed and positive warmup/repeats required')
    return a


def load_snapshot(a):
    path = a.artifact.resolve() / 'snapshots' / a.phase / a.task / f'seed_{a.seed:03d}.pkl'
    raw = path.read_bytes()
    meta = json.loads(path.with_name(path.stem + '_meta.json').read_bytes())
    for key, value in [('phase', a.phase), ('task', a.task), ('seed', a.seed),
                       ('protocol_id', 'VTCG_FACTORIAL_CONTROLS_V1'),
                       ('renderer_depth', 8 if a.task in TASKS[:2] else 0),
                       ('renderer_spp', 32), ('renderer_denoiser', True),
                       ('reset_sequence', 'fresh_environment; reset(seed); restore_snapshot(reset(seed),set_state,agent,rng,get_obs)') ]:
        if meta.get(key) != value:
            raise ValueError(f'snapshot metadata mismatch: {key}')
    if sha(raw) != meta.get('snapshot_file_sha256'):
        raise ValueError('snapshot file digest mismatch')
    return path, pickle.loads(raw), meta  # trusted local runner-generated pickle ONLY


def main(argv=None):
    global _ENTERED
    a = parse_args(argv)
    if _ENTERED or 'torch' in sys.modules or 'tensorflow' in sys.modules:
        raise RuntimeError('one method/trial per fresh process, before GPU framework imports')
    _ENTERED = True
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(a.gpu) or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
        raise RuntimeError('require exact CUDA_VISIBLE_DEVICES=physical --gpu and CUDA_DEVICE_ORDER=PCI_BUS_ID')
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    snapshot_path, snapshot, meta = load_snapshot(a)
    identity = gpu_identity(a.gpu)
    with gpu_claim(a.gpu):
        assert_exclusive(identity)
        os.environ.update(PCD_SOURCE=str(PCD), TOKENIZERS_PARALLELISM='false',
                          HF_HUB_OFFLINE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        sys.path[:0] = [str(ROOT / 'task1/shim_site'), str(ROOT), str(PCD)]
        init_started = time.perf_counter()
        import numpy as np
        import torch
        from PIL import Image
        torch.set_num_threads(1)
        if torch.cuda.device_count() != 1:
            raise RuntimeError('exactly one CUDA device must be visible')
        actual_uuid = str(getattr(torch.cuda.get_device_properties(0), 'uuid', '')).lower().removeprefix('gpu-')
        if actual_uuid != identity['uuid'].lower().removeprefix('gpu-'):
            raise RuntimeError('logical CUDA0 UUID does not match requested physical GPU')
        torch.cuda.set_device(0)
        torch.cuda.synchronize(0)
        from research.semantic_token_cd import prompt_attn_l11_budget_causal_rollout as shared
        from research.semantic_token_cd.distractor_rollout import restore_snapshot, snapshot_sha, get_image_from_maniskill2_obs_dict
        from properties import get_policy_config, get_contrast_image_generator_config
        from simpler_env.policies.openvla.openvla_model import OpenVLAInference
        env = None
        try:
            env, env_id = shared.make_environment(a.task)
            if env_id != meta.get('environment_id') or snapshot_sha(snapshot) != meta.get('canonical_snapshot_sha256'):
                raise ValueError('environment/canonical snapshot mismatch')
            env.reset(seed=a.seed)
            obs, state_hash, rgb_hash = restore_snapshot(env, a.seed, snapshot)
            if state_hash != meta.get('initial_state_sha256') or rgb_hash != meta.get('initial_rgb_sha256'):
                raise ValueError('restored initial state/RGB hash mismatch')
            instruction = env.unwrapped.get_language_instruction()
            if instruction != meta.get('instruction') or instruction != snapshot['instruction']:
                raise ValueError('restored instruction mismatch')
            rgb = np.asarray(get_image_from_maniskill2_obs_dict(env, obs)).copy()
            proprio = np.asarray(obs['agent']['eef_pos']).copy()
            if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3
                    or proprio.ndim != 1 or not proprio.size or not np.isfinite(proprio).all()):
                raise ValueError('invalid restored RGB/proprio')
            config = dict(get_policy_config('openvla', str(a.checkpoint.resolve()), a.task, {}, False))
            gen, gen_config = None, None
            if a.method == 'pcd':
                from contrast_policies.openvla_contrast import OpenVLAContrastInference
                from contrast_utils import get_contrast_image_generator
                gen_config = dict(get_contrast_image_generator_config({}))
                if gen_config.get('by') != 'gt' or gen_config.get('inpaint_mode') != 'lama':
                    raise RuntimeError('expected existing PCD default GT + LaMa')
                gen = get_contrast_image_generator(dict(gen_config, env=env))
                policy = OpenVLAContrastInference(alpha=.2, **config)
                scope = 'existing unmodified PCD GT+LaMa + two independently decoded OpenVLA branches; not factorial audit instrumentation'
            else:
                from research.semantic_token_cd.vtcg_factorial_policy import build_policy
                base = OpenVLAInference(**config)
                policy = build_policy(base, a.task, 'vanilla' if a.method == 'vanilla' else 'attn_matched',
                                      fixed_m=48, top_p=.85, lambd=.5, record_features=False)
                scope = 'instrumented factorial audit path, including rankings/hash/audits/CPU copies; NOT deployment-optimized speed'
            policy.vla.eval()
            policy.reset(instruction, seed=a.seed)
            torch.cuda.synchronize(0)
            init_ms, init_memory = (time.perf_counter() - init_started) * 1000, memory(torch)

            def call():
                policy.reset(instruction, seed=a.seed)
                with torch.no_grad():
                    negative = gen.generate(obs, instruction) if gen is not None else None
                    return policy.step(rgb, negative, instruction, proprio=proprio)

            def timed():
                assert_exclusive(identity)
                torch.cuda.synchronize(0)
                started = time.perf_counter()
                result = call()
                torch.cuda.synchronize(0)
                elapsed = (time.perf_counter() - started) * 1000
                del result
                assert_exclusive(identity)
                return elapsed

            torch.cuda.reset_peak_memory_stats(0)
            first_ms = timed()
            first_memory = memory(torch)
            for _ in range(a.warmup):
                timed()
            if gen is None:
                policy._episode_trace, policy._episode_logits, policy._mask_audit = [], [], None
            torch.cuda.synchronize(0)
            torch.cuda.reset_peak_memory_stats(0)
            baseline = memory(torch)
            samples = [timed() for _ in range(a.repeats)]
            steady_memory = memory(torch)
            assert_exclusive(identity)
            # No simulator step has occurred. Export precisely the profiled frame.
            a.output_dir.mkdir(parents=True, exist_ok=False)
            image_path = a.output_dir / 'initial_rgb.png'
            Image.fromarray(rgb).save(image_path)
            manifest = [{'image_path': image_path.name, 'proprio': proprio.tolist(),
                         'instruction': instruction, 'task': a.task,
                         'initial_state_sha256': state_hash, 'initial_rgb_sha256': rgb_hash}]
            if proprio.ndim != 1:
                raise ValueError('eef_pos is not a vector compatible with offline manifest')
            (a.output_dir / 'state_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
            source_files = [Path(__file__), Path(__file__).with_name('profile_vtcg_factorial.py'),
                            ROOT / 'research/semantic_token_cd/vtcg_factorial_policy.py',
                            ROOT / 'research/semantic_token_cd/prompt_attn_shr_policy.py',
                            PCD / 'properties.py', PCD / 'contrast_utils/contrast_image_generator.py',
                            PCD / 'contrast_utils/inpainters.py', PCD / 'contrast_policies/openvla_contrast.py']
            report = {'protocol': 'VTCG_INITIAL_PAIRED_STATE_ONLINE_PROFILE_V1',
                      'method': a.method, 'measurement_scope': scope,
                      'coverage': 'ONE restored INITIAL state only; no phase/stage/trajectory claim',
                      'trial_id': a.trial_id, 'task': a.task, 'seed': a.seed, 'snapshot_phase': a.phase,
                      'snapshot': str(snapshot_path), 'snapshot_file_sha256': meta['snapshot_file_sha256'],
                      'initial_state_sha256': state_hash, 'initial_rgb_sha256': rgb_hash,
                      'canonical_snapshot_sha256': meta['canonical_snapshot_sha256'], 'environment_id': env_id,
                      'reset_sequence': 'fresh env; reset(seed); restore_snapshot(reset(seed),set_state,agent,rng,get_obs)',
                      'gpu': identity, 'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
                      'policy_config': config, 'pcd_generator_config': gen_config,
                      'pcd_alpha': .2 if gen is not None else None,
                      'initialization': {'wall_ms': init_ms, 'memory': init_memory,
                                         'boundary': 'imports + CUDA context + simulator construction/double reset/restore + model(s) load + policy reset, synchronized'},
                      'first_call': {'wall_ms': first_ms, 'memory': first_memory},
                      'warmup': a.warmup, 'repeats': a.repeats, 'steady': summary(samples), 'samples_ms': samples,
                      'steady_baseline_memory': baseline, 'steady_memory': steady_memory,
                      'steady_boundary': 'same restored observation ready -> policy.reset(seed) + optional GT mask/LaMa generate(obs,instruction) + policy.step + action returned + CUDA synchronize',
                      'excluded': ['simulator restore/render/step', 'observation save', 'nvidia-smi guards', 'report writing'],
                      'cache_note': 'Policy reset per call; generator instruction/mask-object cache stays warm after first call. No cached contrast image: GT mask + LaMa rerun every call.',
                      'memory_note': 'Absolute PyTorch process allocator peaks, includes resident model(s); reset_peak_memory_stats does not clear reserved initialization/warmup cache. No empty_cache. Simulator/non-PyTorch allocations excluded; not total GPU memory.',
                      'comparison_limit': 'PCD default alpha=.2, independent negative generation versus factorial lambda=.5 shared clean prefix; audit instrumentation differs. Paired frame alone does not equal matched algorithm implementation.',
                      'components': {'status': 'not_measured', 'reason': 'no trusted component hooks, no fabricated CUDA-event timings'},
                      'exclusivity_note': 'flock + visible compute PID checks before/after each call; unrelated/MPS/invisible clients cannot be atomically excluded; reserve GPU externally',
                      'source_sha256': {str(p.relative_to(WORKSPACE)): sha(p.read_bytes()) for p in source_files}}
            report_path = a.output_dir / 'profile.json'
            with report_path.open('x') as f:
                json.dump(report, f, indent=2, allow_nan=False)
                f.write('\n')
            print(json.dumps({'output': str(report_path.resolve()), 'steady': report['steady']}))
        finally:
            if env is not None:
                env.close()


if __name__ == '__main__':
    main()
