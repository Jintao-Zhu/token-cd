"""Fast media audit for the full formal set using ffprobe metadata (no decoding).

Checks per video: exists, decodable, frame count == steps+1, size, resolution.
ffprobe reads the container/stream header, so this stays cheap on thousands of files.
A separate, smaller decoded-frame check is done on a deterministic sample.
"""
from pathlib import Path
import argparse, json, subprocess, sys

FFPROBE = '/usr/bin/ffprobe'


def probe(path):
    cmd = [FFPROBE, '-v', 'error', '-select_streams', 'v:0', '-count_frames',
           '-show_entries', 'stream=nb_read_frames,width,height,codec_name',
           '-of', 'json', str(path)]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if out.returncode:
        raise RuntimeError(f'ffprobe failed: {out.stderr.strip()[:200]}')
    stream = json.loads(out.stdout)['streams'][0]
    return int(stream['nb_read_frames']), int(stream['width']), int(stream['height']), stream.get('codec_name')


def _label(path, artifact):
    """Path label that works with absolute video fields and a relative artifact root."""
    try:
        return str(Path(path).resolve().relative_to(artifact.resolve()))
    except Exception:
        return str(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifact', type=Path, required=True)
    ap.add_argument('--phase', choices=('engineering', 'formal'), required=True)
    ap.add_argument('--sample', type=int, default=40,
                    help='deterministically sampled videos to fully DECODE as a cross-check')
    a = ap.parse_args()
    lock = json.loads((a.artifact / 'CONFIG_LOCK.json').read_text())
    tasks, seeds, arms = lock['tasks'], lock[a.phase + '_seeds'], lock['arms']
    errors, checked, missing, by_arm = [], 0, [], {}
    expected_episodes = len(tasks) * len(seeds) * len(arms)
    for task in tasks:
        for seed in seeds:
            for arm in arms:
                root = a.artifact / 'episodes' / a.phase / task / arm
                s = root / f'episode_{seed:03d}_summary.json'
                if not s.is_file():
                    missing.append(f'{task}/{arm}/{seed}')
                    continue
                d = json.loads(s.read_text())
                checked += 1
                if not d.get('video'):
                    continue
                v = Path(d['video'])
                if not v.is_file():
                    errors.append(f'{task}/{arm}/{seed}: missing file')
                    continue
                try:
                    frames, w, h, codec = probe(v)
                except Exception as exc:
                    errors.append(f'{task}/{arm}/{seed}: {exc}')
                    continue
                if frames != d['steps'] + 1:
                    errors.append(f'{task}/{arm}/{seed}: frames {frames} != steps+1 {d["steps"]+1}')
                if (w, h) != (640, 512):
                    errors.append(f'{task}/{arm}/{seed}: resolution {w}x{h}')
                by_arm[arm] = by_arm.get(arm, 0) + 1
    # deterministic decoded cross-check (only episodes whose summary already exists:
    # an in-flight MP4 is still open for writing and must not be audited)
    decoded = []
    import imageio.v2 as imageio
    candidates = []
    for task in tasks:
        for seed in seeds:
            for arm in arms:
                s = a.artifact / 'episodes' / a.phase / task / arm / f'episode_{seed:03d}_summary.json'
                if s.is_file():
                    v = json.loads(s.read_text()).get('video')
                    if v and Path(v).is_file():
                        candidates.append(Path(v))
    candidates = sorted(set(candidates))
    step = max(1, len(candidates) // max(1, a.sample)) if candidates else 1
    for v in candidates[::step][:a.sample]:
        try:
            reader = imageio.get_reader(str(v))
            n = reader.count_frames()
            frame = reader.get_data(0)
            reader.close()
            decoded.append({'video': _label(v, a.artifact), 'frames': n,
                            'shape': list(frame.shape)})
            if frame.shape != (512, 640, 3):
                errors.append(f'{v}: decoded shape {frame.shape}')
        except Exception as exc:
            errors.append(f'{v}: decode {exc!r}')
    out = {'phase': a.phase, 'episodes_checked': checked, 'expected_episodes': expected_episodes,
           'videos_probed': by_arm, 'missing_episodes': missing[:50], 'missing_count': len(missing),
           'errors': errors[:50], 'error_count': len(errors),
           'decoded_cross_check': decoded,
           'note': ('ffprobe metadata for every video plus a deterministic decoded sample. '
                    'Only vanilla/attn_matched and seed%10==0 arms save video by design.')}
    (a.artifact / f'{a.phase.upper()}_MEDIA_AUDIT.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k not in ('decoded_cross_check', 'errors')}, indent=2))
    if errors:
        print('ERRORS:', errors[:10], file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
