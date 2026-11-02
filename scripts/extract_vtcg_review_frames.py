"""Extract deterministic frames from an episode video for human review.

Read-only on episode artifacts; writes PNGs into a new review directory. Used to
check that saved video is actually usable as failure-stage evidence (annotation
needs the moment of failure to be visible, not just the final frame).
"""
from pathlib import Path
import argparse, json
import imageio.v2 as imageio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifact', type=Path, required=True)
    ap.add_argument('--phase', default='engineering')
    ap.add_argument('--task', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--arms', nargs='+', required=True)
    ap.add_argument('--frames', default='0,25%,50%,75%,-1')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    made = []
    for arm in a.arms:
        root = a.artifact / 'episodes' / a.phase / a.task / arm
        summary = json.loads((root / f'episode_{a.seed:03d}_summary.json').read_text())
        video = Path(summary['video'])
        reader = imageio.get_reader(str(video))
        n = reader.count_frames()
        picks = []
        for token in a.frames.split(','):
            token = token.strip()
            idx = n - 1 if token == '-1' else (int(round(float(token[:-1]) / 100 * (n - 1)))
                                               if token.endswith('%') else int(token))
            picks.append(max(0, min(n - 1, idx)))
        for idx in sorted(set(picks)):
            frame = reader.get_data(idx)
            path = a.out / f'{a.task}_seed{a.seed}_{arm}_frame{idx:04d}.png'
            imageio.imwrite(str(path), frame)
            made.append(str(path))
        reader.close()
        print(f'{arm}: steps={summary["steps"]} success={summary["success"]} '
              f'failure_reason={summary["failure_reason"]} frames={n}')
    (a.out / 'frames_manifest.json').write_text(json.dumps(
        {'task': a.task, 'seed': a.seed, 'phase': a.phase, 'files': made,
         'note': 'review frames only; not used for any statistic'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
