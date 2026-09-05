#!/usr/bin/env python3
"""Reproducible LIBERO benchmark protocol utilities.

This module is deliberately independent of model inference: it creates the
canonical episode manifests consumed by every arm and aggregates completed
episode artifacts.  Keeping this bookkeeping separate makes adding a new arm
possible without rerunning the baselines.
"""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path

ARMS = ("vanilla", "recon", "shr")
BENCHMARKS = {"spatial": "LIBERO-Spatial", "object": "LIBERO-Object"}

def manifest(root: Path, benchmark: str, tasks: list[str], n: int = 200, seed0: int = 0):
    """Create immutable seed manifests (one per task) and return their paths."""
    out = root / benchmark
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for task in tasks:
        d = out / task; d.mkdir(parents=True, exist_ok=True)
        p = d / "seeds.json"
        payload = [{"task": task, "episode": i, "seed": seed0 + i} for i in range(n)]
        if p.exists() and json.loads(p.read_text()) != payload:
            raise RuntimeError(f"refusing to overwrite existing manifest: {p}")
        if not p.exists(): p.write_text(json.dumps(payload, indent=2) + "\n")
        paths.append(p)
    return paths

def _sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None

def aggregate(root: Path, benchmark: str, tasks: list[str]):
    rows = []
    for task in tasks:
        for seed in range(200):
            rec = {"benchmark": BENCHMARKS[benchmark], "task": task, "episode": seed}
            for arm in ARMS:
                p = root / benchmark / task / f"seed_{seed:03d}" / f"{arm}_result.json"
                if not p.exists(): rec[arm] = None
                else: rec[arm] = int(bool(json.loads(p.read_text()).get("success", False)))
            rows.append(rec)
    complete = [r for r in rows if all(r[a] is not None for a in ARMS)]
    def stats(a, b):
        rescue = sum(r[a] == 1 and r[b] == 0 for r in complete)
        harm = sum(r[a] == 0 and r[b] == 1 for r in complete)
        # Exact two-sided McNemar p-value (binomial conditional test).
        import math
        n = rescue + harm
        tail = sum(math.comb(n, k) for k in range(min(rescue, harm)+1)) / (2 ** n) if n else 1.0
        p = min(1.0, 2 * tail)
        return {"rescue": rescue, "harm": harm, "net": rescue-harm, "mcnemar_p": p, "n": len(complete)}
    result = {"benchmark": BENCHMARKS[benchmark], "n_complete": len(complete),
              "success_rate": {a: (sum(r[a] for r in complete)/len(complete) if complete else None) for a in ARMS},
              "shr_vs_vanilla": stats("shr", "vanilla"), "shr_vs_recon": stats("shr", "recon")}
    out = root / benchmark; out.mkdir(parents=True, exist_ok=True)
    with (out / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["benchmark", "task", "episode"] + list(ARMS)); w.writeheader(); w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result

def main():
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True)
    p.add_argument("--benchmark", choices=BENCHMARKS, required=True); p.add_argument("--task", action="append", required=True)
    p.add_argument("--episodes", type=int, default=200); p.add_argument("--make-manifest", action="store_true"); p.add_argument("--aggregate", action="store_true")
    a = p.parse_args()
    if a.make_manifest: print(json.dumps([str(x) for x in manifest(a.root, a.benchmark, a.task, a.episodes)], indent=2))
    if a.aggregate: print(json.dumps(aggregate(a.root, a.benchmark, a.task), indent=2))
if __name__ == "__main__": main()
