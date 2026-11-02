"""Is one policy server per GPU saturating the card?

If the GPU is saturated, adding workers only lengthens the queue and adding a second model
copy buys nothing. If it is not saturated (bursty, low duty cycle), a second server per card
would nearly double inference throughput.

Measures the duty cycle of each target GPU at high frequency, and counts how many replans the
run actually completes per minute so throughput can be compared against the per-replan cost
measured in isolation (10.4 s for L6-Matched, 1.58 s for vanilla).
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

RUN = Path("/home/leju-suzhou/zjt_ws/token-cd/runs/pi0fast_l6_matched_9task_100seed")
GPUS = [1, 2, 3]
SAMPLES = 60
INTERVAL = 0.5


def gpu_util() -> dict:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader"],
        capture_output=True, text=True).stdout
    d = {}
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 3:
            d[int(p[0])] = (int(p[1].split()[0]), int(p[2].split()[0]))
    return d


def replan_count() -> int:
    n = 0
    for f in RUN.glob("episodes/*/*.json"):
        try:
            d = json.loads(f.read_text())
            n += len(d.get("chunks", []))
        except Exception:
            pass
    return n


def main() -> None:
    samples = {g: [] for g in GPUS}
    mem = {g: [] for g in GPUS}
    r0 = replan_count()
    t0 = time.time()
    for _ in range(SAMPLES):
        u = gpu_util()
        for g in GPUS:
            if g in u:
                samples[g].append(u[g][0])
                mem[g].append(u[g][1])
        time.sleep(INTERVAL)
    t1 = time.time()
    r1 = replan_count()

    dt = t1 - t0
    dreplans = r1 - r0
    print(f"window {dt:.1f}s, replans completed = {dreplans} "
          f"({dreplans / dt * 60:.1f} replans/min across 3 servers)")
    print(f"  -> {dreplans / dt * 60 / 3:.1f} replans/min per server")
    summary = {"window_s": dt, "replans": dreplans,
               "replans_per_min_total": dreplans / dt * 60,
               "replans_per_min_per_server": dreplans / dt * 60 / 3, "gpus": {}}
    for g in GPUS:
        s = samples[g]
        busy = sum(1 for x in s if x > 50)
        mid = sum(1 for x in s if 10 < x <= 50)
        idle = sum(1 for x in s if x <= 10)
        summary["gpus"][g] = {
            "mean_util": sum(s) / len(s), "max_util": max(s),
            "pct_above_50": 100 * busy / len(s), "pct_10_to_50": 100 * mid / len(s),
            "pct_below_10": 100 * idle / len(s),
            "mean_mem_mib": sum(mem[g]) / len(mem[g]),
        }
        print(f"  GPU {g}: mean={sum(s)/len(s):5.1f}% max={max(s):3d}%  "
              f">50%: {100*busy/len(s):5.1f}%  10-50%: {100*mid/len(s):5.1f}%  "
              f"<10%: {100*idle/len(s):5.1f}%  mem={sum(mem[g])/len(mem[g]):.0f} MiB")
    (RUN / "saturation.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
