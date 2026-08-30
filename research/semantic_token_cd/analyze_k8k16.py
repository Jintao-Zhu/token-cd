"""Aggregate the three-arm (vanilla / k8 / k16) narrow-band Attention-CD results.

Reads per-arm per-seed summaries under <artifact>/episodes/<task>/<arm>/ and
reports, per task:
  * per-arm success count (over seeds where all three arms completed),
  * Rescue / Harm cross-tabs for (k8, vanilla), (k16, vanilla), (k8, k16),
  * exact McNemar two-sided p-values (scipy.stats.binomtest on discordant pairs).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy.stats import binomtest

ARMS = ("vanilla", "attn_semantic_k8", "attn_semantic_k16")


def load_arm(task_root: Path, arm: str) -> dict[int, bool]:
    out: dict[int, bool] = {}
    arm_dir = task_root / arm
    for f in sorted(arm_dir.glob("episode_*_summary.json")):
        seed = int(f.name.split("_")[1])
        out[seed] = bool(json.loads(f.read_text())["success"])
    return out


def mcnemar(a_name: str, b_name: str, a: dict[int, bool], b: dict[int, bool], seeds) -> dict:
    rescue = harm = both_ok = both_fail = 0
    for seed in seeds:
        a_ok, b_ok = a[seed], b[seed]
        if not a_ok and b_ok:
            rescue += 1  # a fails, b succeeds
        elif a_ok and not b_ok:
            harm += 1  # a succeeds, b fails
        elif a_ok and b_ok:
            both_ok += 1
        else:
            both_fail += 1
    disc = rescue + harm
    p = binomtest(rescue, disc, 0.5).pvalue if disc > 0 else 1.0
    return {
        "pair": f"{a_name} -> {b_name}",
        "rescue": rescue,  # b succeeds where a failed
        "harm": harm,      # a succeeds where b failed
        "both_ok": both_ok,
        "both_fail": both_fail,
        "discordant": disc,
        "mcnemar_p": float(p),
    }


def report_task(task_root: Path, task: str) -> dict:
    per_arm = {arm: load_arm(task_root, arm) for arm in ARMS}
    complete = sorted(set(per_arm["vanilla"]) & set(per_arm["attn_semantic_k8"]) & set(per_arm["attn_semantic_k16"]))
    n = len(complete)
    if n == 0:
        print(f"=== {task}: no complete seeds yet ===")
        return {"task": task, "n_complete": 0}

    wins = {arm: sum(1 for s in complete if per_arm[arm][s]) for arm in ARMS}
    cross = {
        "k8_vs_vanilla": mcnemar("vanilla", "attn_semantic_k8", per_arm["vanilla"], per_arm["attn_semantic_k8"], complete),
        "k16_vs_vanilla": mcnemar("vanilla", "attn_semantic_k16", per_arm["vanilla"], per_arm["attn_semantic_k16"], complete),
        "k8_vs_k16": mcnemar("attn_semantic_k16", "attn_semantic_k8", per_arm["attn_semantic_k16"], per_arm["attn_semantic_k8"], complete),
    }

    print(f"\n=== {task}  (n={n}/100 complete seeds) ===")
    print(f"  success: vanilla {wins['vanilla']}/{n} | k8 {wins['attn_semantic_k8']}/{n} | k16 {wins['attn_semantic_k16']}/{n}")
    for key, c in cross.items():
        print(f"  {key}: rescue={c['rescue']} harm={c['harm']} (both_ok={c['both_ok']} both_fail={c['both_fail']}) McNemar_p={c['mcnemar_p']:.6g}")
    return {"task": task, "n_complete": n, "wins": wins, "cross": cross}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact.resolve()
    eps = root / "episodes"
    for task_dir in sorted(eps.iterdir()) if eps.exists() else []:
        if task_dir.is_dir():
            report_task(task_dir, task_dir.name)


if __name__ == "__main__":
    main()
