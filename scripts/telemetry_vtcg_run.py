"""Append-only telemetry for the long formal run: episodes done, cases done, GPU memory.

Writes one JSON line every --interval seconds so the final report can show a real
throughput/memory timeline rather than a single hand-taken reading. Read-only on
episodes; the only file it creates is the telemetry JSONL.
"""
from pathlib import Path
import argparse, json, subprocess, time


def gpu_stats():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20)
        rows = []
        for line in out.stdout.strip().splitlines():
            idx, mem, util = [x.strip() for x in line.split(',')]
            rows.append({'gpu': int(idx), 'mem_mib': int(mem), 'util_pct': int(util)})
        return rows
    except Exception as exc:
        return [{'error': repr(exc)}]


def workers():
    try:
        out = subprocess.run(['pgrep', '-fc', 'vtcg_factorial_episode.py'], capture_output=True, text=True, timeout=20)
        return int(out.stdout.strip() or 0)
    except Exception:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifact', type=Path, required=True)
    ap.add_argument('--phase', default='formal')
    ap.add_argument('--interval', type=int, default=60)
    a = ap.parse_args()
    art = a.artifact
    lock = json.loads((art / 'CONFIG_LOCK.json').read_text())
    total = len(lock['tasks']) * len(lock[a.phase + '_seeds']) * len(lock['arms'])
    out = art / f'{a.phase}_telemetry.jsonl'
    while True:
        try:
            root = art / 'episodes' / a.phase
            episodes = len(list(root.glob('*/*/*_summary.json')))
            cases_done = sum(1 for c in (art / 'case_status' / a.phase).glob('*/*.json')
                             if json.loads(c.read_text()).get('status') == 'complete')
            status = json.loads((art / 'QUEUE_STATUS.json').read_text())
            exempt = [{'gpu': g['gpu'], 'mem_mib': g['mem_mib'], 'util_pct': g['util_pct']}
                      for g in gpu_stats() if g.get('gpu') in lock['allowed_gpus']]
            rec = {'ts': time.time(), 'episodes': episodes, 'episodes_total': total,
                   'cases_complete': cases_done, 'workers': workers(),
                   'queue_status': status.get('status'), 'gpus': exempt}
            with out.open('a') as fh:
                fh.write(json.dumps(rec) + '\n')
            if status.get('status') == 'complete' and workers() == 0:
                break
        except Exception as exc:  # never die on a transient read error
            with out.open('a') as fh:
                fh.write(json.dumps({'ts': time.time(), 'error': repr(exc)}) + '\n')
        time.sleep(a.interval)
    print('telemetry finished')


if __name__ == '__main__':
    main()
