#!/usr/bin/env python3
"""Strict offline VTCG factorial profiling worker (one method/trial/process).

No simulator, rendering, rollout changes, component hooks, or automatic launcher.
CPU-only: python scripts/profile_vtcg_factorial.py --help
Run each method/trial in a NEW process, sequentially on an exclusively idle GPU:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/profile_vtcg_factorial.py \
    --gpu 1 --method vanilla --state-manifest states.json --trial-id 0 --output vanilla_0.json
Repeat externally with --method vtcg (factory arm attn_matched), then other trial IDs.

Manifest: JSON list of {image_path, proprio, instruction, task}. Image paths are
relative to the manifest directory. Proprio must be the actual saved eef_pos,
not invented zeros; OpenVLA currently ignores it. All rows must have one task.
Each observation is independently replayed from reset, not trajectory replay:
policy sticky-gripper/history state is reset; warm instruction caches are kept.
The existing runner's initial PNG/initial_rgb alone is NOT a valid saved state.

PCD is deliberately NOT IMPLEMENTED: its default GT+LaMa generator needs
segmentation, camera identity and simulator name->actor/articulation ID mapping;
RGB/proprio alone cannot reproduce it. Precomputed negatives would exclude mask
and inpainting cost and must never be reported as PCD end-to-end here.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
PCD = WORKSPACE / 'pcd_openvla_simpler_box_31b027e/source'
_ENTERED = False


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--gpu', type=int, choices=(1, 2, 3), required=True, help='physical nvidia-smi index; exact CUDA_VISIBLE_DEVICES required')
    p.add_argument('--method', choices=('vanilla', 'vtcg', 'attn_matched', 'pcd'), required=True)
    p.add_argument('--state-manifest', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, default=PCD / 'pretrained/openvla-7b')
    p.add_argument('--warmup', type=int, default=10, help='per observation, after separately reported first call')
    p.add_argument('--repeats', type=int, default=50, help='measured reset+step calls per observation')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--trial-id', required=True, help='external launcher identifier; this worker never loops trials/methods')
    p.add_argument('--output', type=Path, required=True, help='exclusive-create JSON; never overwrites')
    a = p.parse_args(argv)
    if a.warmup < 1 or a.repeats < 1 or a.seed < 0:
        p.error('warmup/repeats must be positive; seed must be nonnegative')
    return a


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_manifest(path):
    """Stdlib validation runs before any torch/CUDA imports."""
    path = path.resolve(strict=True)
    raw = path.read_bytes()
    rows = json.loads(raw)
    if not isinstance(rows, list) or not rows:
        raise ValueError('state manifest must be a nonempty JSON list')
    result = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or not {'image_path', 'proprio', 'instruction', 'task'} <= row.keys():
            raise ValueError(f'state {i}: required image_path/proprio/instruction/task missing')
        if any(not isinstance(row[k], str) or not row[k].strip() for k in ('image_path', 'instruction', 'task')):
            raise ValueError(f'state {i}: image_path/instruction/task must be nonempty strings')
        proprio = row['proprio']
        if not isinstance(proprio, list) or not proprio or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in proprio
        ):
            raise ValueError(f'state {i}: proprio must be a nonempty finite numeric vector saved from eef_pos')
        image = (path.parent / row['image_path']).resolve(strict=True)
        if not image.is_file():
            raise ValueError(f'state {i}: image is not a regular file')
        result.append(dict(row, image_path=str(image)))
    if len({r['task'] for r in result}) != 1:
        raise ValueError('one task per process required; launch separate workers for different tasks')
    if not result[0]['task'].startswith(('google_robot', 'widowx')):
        raise ValueError('task is not supported by existing OpenVLA configuration')
    return result, sha(raw)


def smi_rows(query):
    proc = subprocess.run(['nvidia-smi', query, '--format=csv,noheader,nounits'],
                          text=True, capture_output=True, check=True, timeout=15)
    return [r for r in csv.reader(io.StringIO(proc.stdout), skipinitialspace=True) if r]


def gpu_identity(index):
    matches = [r for r in smi_rows('--query-gpu=index,uuid,pci.bus_id,name') if int(r[0]) == index]
    if len(matches) != 1:
        raise RuntimeError('physical GPU index could not be uniquely resolved')
    r = matches[0]
    return dict(index=index, uuid=r[1].strip(), pci_bus_id=r[2].strip(), name=r[3].strip())


def assert_exclusive(identity):
    """Fail closed on unsupported/unparseable process accounting; never kill others."""
    others = []
    for row in smi_rows('--query-compute-apps=gpu_uuid,pid,process_name'):
        if len(row) != 3 or not row[0].strip().startswith('GPU-'):
            raise RuntimeError(f'unusable nvidia-smi compute accounting: {row!r}')
        pid = int(row[1])
        if row[0].strip() == identity['uuid'] and pid != os.getpid():
            others.append({'pid': pid, 'name': row[2].strip()})
    if others:
        raise RuntimeError(f'GPU {identity["index"]} has other compute processes: {others}; refusing contaminated profiling')


@contextlib.contextmanager
def gpu_claim(index):
    # Persistent inode: do not unlink on exit (flock unlink/recreate race).
    lock_dir = ROOT / 'artifacts/.vtcg_profiling_locks'
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / f'physical_gpu_{index}.lock').open('a+') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('another cooperating profiling process owns this GPU') from exc
        yield


def summary(values):
    ordered = sorted(values)
    def percentile(q):
        x = (len(ordered) - 1) * q
        lo, hi = math.floor(x), math.ceil(x)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (x - lo)
    return {'n': len(values), 'mean_ms': statistics.fmean(values),
            'p50_ms': percentile(.5), 'p95_ms': percentile(.95)}


def memory(torch):
    return {'allocated_bytes': torch.cuda.memory_allocated(0),
            'reserved_bytes': torch.cuda.memory_reserved(0),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(0),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(0)}


def main(argv=None):
    global _ENTERED
    a = parse_args(argv)
    if _ENTERED:
        raise RuntimeError('one invocation per fresh process; trials require external launcher')
    _ENTERED = True
    if a.method == 'pcd':
        raise NotImplementedError('PCD end-to-end unavailable for this manifest: save segmentation, camera identity, actor/articulation name-ID mapping and generator configuration first. No PCD timings fabricated.')
    if 'torch' in sys.modules or 'tensorflow' in sys.modules:
        raise RuntimeError('GPU framework already imported; use a fresh process')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(a.gpu):
        raise RuntimeError('CUDA_VISIBLE_DEVICES must already equal exactly physical --gpu (1/2/3)')
    if os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
        raise RuntimeError('CUDA_DEVICE_ORDER must already equal PCI_BUS_ID; UUID mapping is additionally verified')
    rows, manifest_sha = load_manifest(a.state_manifest)
    if a.output.exists():
        raise FileExistsError(a.output)
    # Input I/O is not part of policy-ready end-to-end latency.
    import numpy as np
    from PIL import Image
    for row in rows:
        with Image.open(row['image_path']) as image:
            if image.mode != 'RGB':
                raise ValueError('saved observation must be RGB; implicit conversions are forbidden')
            row['_image'] = np.asarray(image).copy()
        row['_proprio'] = np.asarray(row['proprio'], dtype=np.float32)
        if not np.isfinite(row['_proprio']).all():
            raise ValueError('proprio overflows float32')
        row['_image_sha256'] = sha(Path(row['image_path']).read_bytes())
    identity = gpu_identity(a.gpu)
    with gpu_claim(a.gpu):
        assert_exclusive(identity)  # before importing torch or creating a CUDA context
        os.environ['PCD_SOURCE'] = str(PCD)
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        sys.path[:0] = [str(ROOT / 'task1/shim_site'), str(ROOT), str(PCD)]
        initialization_started = time.perf_counter()
        import torch
        torch.set_num_threads(1)
        if torch.cuda.device_count() != 1:
            raise RuntimeError('exactly one visible CUDA device required')
        props = torch.cuda.get_device_properties(0)
        actual_uuid = str(getattr(props, 'uuid', '')).lower().removeprefix('gpu-')
        if actual_uuid != identity['uuid'].lower().removeprefix('gpu-'):
            raise RuntimeError('CUDA logical 0 UUID does not match requested physical GPU (or runtime lacks UUID verification)')
        torch.cuda.set_device(0)
        torch.cuda.synchronize(0)
        assert_exclusive(identity)
        from properties import get_policy_config
        from simpler_env.policies.openvla.openvla_model import OpenVLAInference
        from research.semantic_token_cd.vtcg_factorial_policy import build_policy
        arm = 'vanilla' if a.method == 'vanilla' else 'attn_matched'
        config = dict(get_policy_config('openvla', str(a.checkpoint.resolve()), rows[0]['task'], {}, False))
        base = OpenVLAInference(**config)
        policy = build_policy(base, rows[0]['task'], arm, fixed_m=48, top_p=.85,
                              lambd=.5, record_features=False)
        policy.vla.eval()
        policy.reset(rows[0]['instruction'], seed=a.seed)
        torch.cuda.synchronize(0)
        initialization_ms = (time.perf_counter() - initialization_started) * 1000
        initialization_memory = memory(torch)

        def call(row):
            # Existing factory interfaces only. Keep all audit/recording overhead.
            policy.reset(row['instruction'], seed=a.seed)
            with torch.no_grad():
                return policy.step(row['_image'], None, row['instruction'], proprio=row['_proprio'])

        def timed(row):
            assert_exclusive(identity)
            torch.cuda.synchronize(0)
            t0 = time.perf_counter()
            output = call(row)
            torch.cuda.synchronize(0)
            elapsed = (time.perf_counter() - t0) * 1000
            del output
            assert_exclusive(identity)
            return elapsed

        results, all_samples = [], []
        for i, row in enumerate(rows):
            torch.cuda.reset_peak_memory_stats(0)
            first_ms = timed(row)
            first_memory = memory(torch)
            for _ in range(a.warmup):
                timed(row)
            # Clear last audit references without altering implementation or caches.
            policy._episode_trace, policy._episode_logits = [], []
            policy._mask_audit = None
            torch.cuda.synchronize(0)
            torch.cuda.reset_peak_memory_stats(0)
            baseline = memory(torch)
            samples = [timed(row) for _ in range(a.repeats)]
            peak = memory(torch)
            all_samples.extend(samples)
            results.append({'state_index': i, 'task': row['task'], 'instruction': row['instruction'],
                            'proprio': row['proprio'], 'image_path': row['image_path'],
                            'image_file_sha256': row['_image_sha256'], 'rgb_shape': list(row['_image'].shape),
                            'first_call_ms': first_ms, 'first_call_memory': first_memory,
                            'steady': summary(samples), 'samples_ms': samples,
                            'steady_baseline_memory': baseline, 'steady_memory': peak})
        assert_exclusive(identity)
        sources = [Path(__file__), ROOT / 'research/semantic_token_cd/vtcg_factorial_policy.py',
                   ROOT / 'research/semantic_token_cd/vtcg_factorial_math.py',
                   ROOT / 'research/semantic_token_cd/vtcg_factorial_episode.py',
                   ROOT / 'research/semantic_token_cd/prompt_attn_shr_policy.py',
                   PCD / 'simpler_env/policies/openvla/openvla_model.py', PCD / 'properties.py']
        report = {'protocol': 'VTCG_FACTORIAL_OFFLINE_PROFILE_V1', 'method': a.method, 'factory_arm': arm,
                  'trial_id': a.trial_id, 'pid': os.getpid(), 'seed': a.seed, 'gpu': identity,
                  'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
                  'checkpoint': str(a.checkpoint.resolve()), 'policy_config': config,
                  'state_manifest': str(a.state_manifest.resolve()), 'state_manifest_sha256': manifest_sha,
                  'source_sha256': {str(p.relative_to(WORKSPACE)): sha(p.read_bytes()) for p in sources},
                  'warmup_per_state': a.warmup, 'repeats_per_state': a.repeats,
                  'initialization': {'wall_ms': initialization_ms, 'memory': initialization_memory,
                                     'boundary': 'torch import/context + policy imports + model load + factory + initial reset; synchronized'},
                  'measurement_scope': 'instrumented audit path ONLY; NOT deployment-optimized inference speed. Includes full rankings, hashes, audit checks and CPU copies in unmodified factory step.',
                  'steady_boundary': 'CPU RGB/proprio/instruction ready -> reset(seed) + unmodified factory policy.step + decoded/postprocessed action returned + CUDA synchronize',
                  'excluded': ['input file I/O', 'simulator', 'rendering', 'environment step', 'report I/O', 'nvidia-smi guards'],
                  'components': {'status': 'not_measured', 'reason': 'no trusted component timing hooks; no synthetic CUDA-event attribution'},
                  'memory_note': 'Absolute process memory, model included. reset_peak_memory_stats resets counters NOT cached allocations; reserved includes initialization/warmup/earlier-state cache. No empty_cache between samples. PyTorch allocator only; not total device memory. State peaks are not independent trials.',
                  'replay_note': 'Independent reset per observation, not restored trajectory controller state; instruction embeddings warmed and cached. Factory vanilla includes audit copies/records and is not raw OpenVLA baseline. Both arms keep identical factory instrumentation.',
                  'exclusivity_note': 'Persistent flock excludes cooperating workers; nvidia-smi checks before/after every call reject other visible compute PIDs. Cannot atomically prevent unrelated processes starting or invisible/MPS clients; reserve GPU externally too.',
                  'pcd': {'status': 'not_implemented', 'needed_state_fields': ['camera_name', 'camera RGB and Segmentation array', 'actor/articulation name-to-segmentation-ID mapping', 'GT/LaMa generator configuration and checkpoint provenance'],
                          'note': 'Default generate(obs,instruction) uses env mapping; standalone adapter required. Precomputed contrast_image is policy-only, not comparable end-to-end.'},
                  'aggregate_steady': summary(all_samples), 'states': results}
        a.output.parent.mkdir(parents=True, exist_ok=True)
        with a.output.open('x') as f:
            json.dump(report, f, indent=2, allow_nan=False)
            f.write('\n')
        print(json.dumps({'output': str(a.output.resolve()), 'aggregate_steady': report['aggregate_steady']}))


if __name__ == '__main__':
    main()
