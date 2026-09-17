"""Aggregate the formal fixed-benchmark run and compare to the vanilla 0-99 baseline.

Formal episodes: artifact formal_400 with frozen unified arm l11_k64_t05 over
canonical seeds 0-99 for the four tasks (400 episodes).  Baseline: existing
vanilla summaries in vanilla_recon_shr_canonical_0_299_v2 for the same seeds.
Outputs paired rescue/harm and success tables, plus per-scene clustering using
recorded initial state/RGB hashes.
"""
from __future__ import annotations

import collections
import csv
import json
from pathlib import Path

from research.semantic_token_cd.dtp_closed_loop_protocol import (
    ARM_CONFIG, ARTIFACT, CANONICAL, PROTOCOL, TASKS,
)

FORMAL_ROOT = ARTIFACT / "formal_400"
FORMAL_ARM = "l11_k64_t05"


def load_episodes(root: Path, tasks=TASKS, arm=None) -> dict:
    out = {}
    for task in tasks:
        pattern = f"{arm}/episode_*_summary.json" if arm else "*/episode_*_summary.json"
        rows = {}
        for p in sorted((root / "episodes" / task).glob(pattern)):
            d = json.loads(p.read_text())
            rows[int(d["seed"])] = d
        out[task] = rows
    return out


def main() -> None:
    formal = load_episodes(FORMAL_ROOT, arm=FORMAL_ARM)
    vanilla = load_episodes(CANONICAL, arm="vanilla")

    issues = []
    per_task = {}
    paired = collections.Counter()
    table = []
    total = {k: 0 for k in ("formal_success", "vanilla_success", "n")}
    for task in TASKS:
        f = formal[task]
        v = vanilla[task]
        f_seeds = set(f); v_seeds = set(v)
        if f_seeds != v_seeds:
            issues.append(f"{task}: seed mismatch formal={sorted(f_seeds - v_seeds)[:5]} vanilla={sorted(v_seeds - f_seeds)[:5]}")
        n = 0; fs = 0; vs = 0
        for seed in sorted(f_seeds & v_seeds):
            n += 1
            fd = f[seed]; vd = v[seed]
            if fd["success"]: fs += 1
            if vd["success"]: vs += 1
            if fd["success"] and not vd["success"]:
                paired[(task, "rescue")] += 1
            elif not fd["success"] and vd["success"]:
                paired[(task, "harm")] += 1
            else:
                paired[(task, "unchanged")] += 1
        per_task[task] = {"n": n, "formal_success": fs, "vanilla_success": vs}
        total["n"] += n; total["formal_success"] += fs; total["vanilla_success"] += vs
        table.append({"task": task, "n": n, "formal_success": fs, "vanilla_success": vs,
                      "delta": fs - vs})

    # scene clustering by (state, rgb) identity within formal arm
    clusters = {}
    for task in TASKS:
        for seed, d in formal[task].items():
            key = (task, d.get("initial_state_sha256"), d.get("initial_rgb_sha256"),
                   d.get("canonical_snapshot_sha256"))
            clusters.setdefault(key, []).append(seed)
    cluster_rows = sorted((len(seeds), task, seeds[0], sorted(seeds)) for (task, *_rest), seeds in clusters.items())
    n_unique = len(clusters)
    n_seeds_total = sum(len(s) for s in clusters.values())

    summary = {
        "protocol_id": PROTOCOL,
        "stage": "formal_fixed_benchmark",
        "frozen_arm": FORMAL_ARM,
        "frozen_config": ARM_CONFIG[FORMAL_ARM],
        "rule": "fixed benchmark on canonical seeds 0-99; formal episodes are NOT an unseen independent test",
        "seeds": {task: sorted(formal[task]) for task in TASKS},
        "totals": total,
        "per_task": per_task,
        "paired": {f"{t}__{k}": v for (t, k), v in paired.items()},
        "scene_cluster": {
            "unique_physical_scenes": n_unique,
            "seed_total": n_seeds_total,
            "largest_clusters": [(n, t, s) for n, t, s, _ in cluster_rows[-5:]][::-1],
        },
        "issues": issues,
    }
    root = FORMAL_ROOT
    csv_path = root / "formal_summary.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    summary["summary_csv"] = str(csv_path.relative_to(root))
    lock_path = root / "FORMAL_RESULTS.json"
    tmp = lock_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    tmp.replace(lock_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
