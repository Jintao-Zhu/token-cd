"""Formal-run throughput / progress monitor. Read-only; never touches episodes."""
from pathlib import Path
import argparse, json, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifact', type=Path, required=True)
    ap.add_argument('--phase', default='formal')
    a = ap.parse_args()
    art = a.artifact
    lock = json.loads((art / 'CONFIG_LOCK.json').read_text())
    total = len(lock['tasks']) * len(lock[a.phase + '_seeds']) * len(lock['arms'])
    root = art / 'episodes' / a.phase
    summaries = sorted(root.glob('*/*/*_summary.json'), key=lambda p: p.stat().st_mtime)
    done = len(summaries)
    cases = sorted((art / 'case_status' / a.phase).glob('*/*.json'))
    complete = sum(1 for c in cases if json.loads(c.read_text()).get('status') == 'complete')
    status = json.loads((art / 'QUEUE_STATUS.json').read_text())
    started = status.get('started', time.time())
    elapsed = time.time() - started
    rate = done / elapsed * 3600 if elapsed > 0 else 0
    # windowed rate: completions in the last `window_min`, ignoring the ramp-up phase
    now = time.time()
    window_min = 15
    recent = [p for p in summaries if now - p.stat().st_mtime <= window_min * 60]
    window_rate = len(recent) / window_min * 60 if recent else 0
    rate_used = window_rate or rate
    eta_h = (total - done) / rate_used if rate_used > 0 else float('inf')
    arms_per_case = len(lock['arms'])
    case_times = [json.loads(c.read_text()).get('elapsed') for c in cases
                  if json.loads(c.read_text()).get('status') == 'complete']
    print(json.dumps({
        'queue_status': status.get('status'), 'pid': status.get('pid'),
        'episodes_done': done, 'episodes_total': total,
        'cases_complete': complete, 'cases_total': len(lock['tasks']) * len(lock[a.phase + '_seeds']),
        'elapsed_min': round(elapsed / 60, 1),
        'episodes_per_hour': round(rate, 1),
        'episodes_per_hour_windowed_15min': round(window_rate, 1),
        'episodes_per_hour_per_gpu': round(rate_used / len(lock['allowed_gpus']), 1),
        'baseline_1worker_per_gpu_ep_per_hour_per_gpu': 55,
        'eta_hours_remaining': round(eta_h if eta_h != float('inf') else -1, 2),
        'median_case_seconds': (sorted(case_times)[len(case_times) // 2] if case_times else None),
        'episodes_per_case': arms_per_case,
    }, indent=2))


if __name__ == '__main__':
    main()
