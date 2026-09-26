#!/usr/bin/env python3
"""Analysis for SAME_STATE_FORK_V1.

Answers the three frozen questions:
  Q1 what does the reconstruction branch actually change?
  Q2 does guidance improve or worsen task progress?
  Q3 where do SIMPLER and LIBERO diverge?

Reports per-task results first, then pools with task-equal weighting, and
always shows Rescue and Harm separately (never only the net).
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from research.semantic_token_cd.same_state_fork_common import (
    atomic_json,
    cluster_bootstrap_ci,
    mcnemar_exact,
)

CLEAN_KEY = "clean|repeat"


def load_forks(root: Path) -> list[dict]:
    rows = []
    for path in glob.glob(str(root / "forks" / "*.json")):
        rows.append(json.loads(Path(path).read_text()))
    return rows


def is_clean_control(fork: dict) -> bool:
    return str(fork.get("case_id", "")).startswith("clean_")


def load_clean_repeat(forks: list[dict]) -> dict[tuple[int, int, str], bool]:
    """Load same-state clean controls recorded as clean_* fork files."""
    clean = {}
    for fork in forks:
        if not is_clean_control(fork):
            continue
        result = fork.get("results", {}).get(CLEAN_KEY) or fork.get("results", {}).get("clean|d=1")
        if result is None:
            continue
        key = (int(fork["task_id"]), int(fork["init_state_id"]), str(fork["stage"]))
        clean[key] = bool(result["success"])
    return clean


def paired(fork: dict, branch: str, duration: int, clean_success: bool) -> dict | None:
    key = f"{branch}|d={duration}"
    result = fork["results"].get(key)
    if result is None:
        return None
    return {"success": bool(result["success"]), "clean": bool(clean_success), "steps": result["steps"]}


def all_durations(forks: list[dict]) -> list[int]:
    out = set()
    for fork in forks:
        for key in fork["results"]:
            if "|d=" in key:
                out.add(int(key.split("|d=")[1]))
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--clean-root", type=Path, required=True,
                    help="five-task artifact holding the vanilla episodes used as the Clean branch")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    forks = load_forks(args.artifact)
    out = args.out or (args.artifact / "ANALYSIS.json")
    if not forks:
        print("no forks yet")
        return

    branch_forks = [fork for fork in forks if not is_clean_control(fork)]
    clean_repeat = load_clean_repeat(forks)

    # Clean branch success = the stored vanilla episode outcome for that init state.
    vanilla = {}
    for path in glob.glob(str(args.clean_root / "episodes" / "*" / "vanilla" / "*.json")):
        payload = json.loads(Path(path).read_text())
        vanilla[(int(payload["task_id"]), int(payload["init_state_id"]))] = bool(payload["success"])

    report = {
        "protocol_id": "SAME_STATE_FORK_V1_ANALYSIS",
        "n_forks": len(branch_forks),
        "n_clean_repeat": len(clean_repeat),
        "clean_control": "same_state clean|d=1 when available, otherwise stored vanilla episode",
        "per_task": {},
        "pooled": {},
    }

    # ---- Q1: what does reconstruction change? ----
    q1 = defaultdict(list)
    for fork in branch_forks:
        diag = fork["diagnostics"]
        q1["recon_l2_vs_clean"].append(fork["action_deltas"]["recon_vs_clean"]["l2"])
        q1["guided_l2_vs_clean"].append(fork["action_deltas"]["guided_vs_clean"]["l2"])
        q1["m_matched"].append(diag["m_matched"])
        q1["identity_bin_diff"].append(diag["identity_max_abs_action_bin_diff"])
        q1["recon_tokens_changed"].append(len(diag["reconstruction_changed_token_ids"]))
        q1["recon_only_selected"].append(int(diag["reconstruction_only_modified_selected"]))
    report["q1_reconstruction_effect"] = {
        k: {"mean": float(np.mean(v)), "median": float(np.median(v)), "n": len(v)}
        for k, v in q1.items()
    }

    # ---- Q2/Q3: paired branch outcomes ----
    durations = all_durations(branch_forks)
    tasks = sorted({int(f["task_id"]) for f in branch_forks})
    for branch in ("reconstruction", "guided"):
        for duration in durations:
            per_task = {}
            for task_id in tasks:
                recs = []
                for fork in branch_forks:
                    if int(fork["task_id"]) != task_id:
                        continue
                    clean_key = (task_id, int(fork["init_state_id"]), str(fork["stage"]))
                    clean = clean_repeat.get(clean_key)
                    clean_source = "same_state_repeat"
                    if clean is None:
                        clean = vanilla.get((task_id, int(fork["init_state_id"])))
                        clean_source = "stored_vanilla"
                    if clean is None:
                        continue
                    got = paired(fork, branch, duration, clean)
                    if got is None:
                        continue
                    recs.append({"fork": fork, "got": got, "clean_source": clean_source})
                if not recs:
                    continue
                success = [r["got"]["success"] for r in recs]
                clean_arr = [r["got"]["clean"] for r in recs]
                mc = mcnemar_exact(success, clean_arr)
                per_task[str(task_id)] = {
                    "n": len(recs),
                    "branch_success": int(np.sum(success)),
                    "clean_success": int(np.sum(clean_arr)),
                    "branch_rate": float(np.mean(success)),
                    "clean_rate": float(np.mean(clean_arr)),
                    "clean_source_counts": dict(defaultdict(int, {
                        source: sum(1 for r in recs if r["clean_source"] == source)
                        for source in {r["clean_source"] for r in recs}
                    })),
                    **mc,
                }
            if not per_task:
                continue
            # task-equal weighting
            nets = [v["net"] / v["n"] for v in per_task.values()]
            deltas = [v["branch_rate"] - v["clean_rate"] for v in per_task.values()]
            ci = cluster_bootstrap_ci(deltas, list(per_task.keys()))
            report["per_task"].setdefault(f"{branch}|d={duration}", {
                "tasks": per_task,
                "task_equal_mean_delta": float(np.mean(deltas)),
                "task_equal_net_per_fork": float(np.mean(nets)),
                "delta_ci": ci,
                "rescue_total": int(sum(v["rescue"] for v in per_task.values())),
                "harm_total": int(sum(v["harm"] for v in per_task.values())),
            })

    # ---- duration dose-response ----
    dose = {}
    for branch in ("reconstruction", "guided"):
        curve = []
        for duration in durations:
            key = f"{branch}|d={duration}"
            if key in report["per_task"]:
                entry = report["per_task"][key]
                curve.append({
                    "duration": duration,
                    "task_equal_mean_delta": entry["task_equal_mean_delta"],
                    "rescue": entry["rescue_total"],
                    "harm": entry["harm_total"],
                })
        dose[branch] = curve
    report["dose_response"] = dose

    atomic_json(out, report)
    print(f"forks analyzed: {len(forks)}")
    print("\n=== Q1 reconstruction branch effect ===")
    for k, v in report["q1_reconstruction_effect"].items():
        print(f"  {k:<26} mean={v['mean']:.4f} median={v['median']:.4f}")
    print("\n=== Q2/Q3 dose-response (task-equal mean delta vs clean) ===")
    for branch, curve in dose.items():
        print(f"  {branch}:")
        for point in curve:
            print(f"    d={point['duration']:<4} delta={point['task_equal_mean_delta']:+.4f}  rescue={point['rescue']:<4} harm={point['harm']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
