#!/usr/bin/env python3
"""Requeue interrupted fork cases.

Running files are named ``<case_id>__<worker_id>.json``.  Splitting on the last
``__`` recovers the original case id; a naive ``split('__worker')`` silently
misses worker ids like ``fork_g1c`` and pollutes the queue with duplicates.
"""
import os
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else
            "/home/leju-suzhou/zjt_ws/token-cd/artifacts/same_state_fork_v1/cases")
pending = root / "pending"
pending.mkdir(parents=True, exist_ok=True)

moved = 0
for sub in ("running", "error"):
    for path in sorted((root / sub).glob("*.json")):
        case_id = path.name.rsplit("__", 1)[0]
        target = pending / case_id
        if target.exists():
            path.unlink()
            continue
        os.replace(path, target)
        moved += 1

print(f"requeued {moved}; pending={len(list(pending.glob('*.json')))}")
