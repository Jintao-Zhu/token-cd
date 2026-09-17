"""Offline threshold-separation check on the 60 saved L11 diagnostic states."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


DIAGNOSTIC = Path("artifacts/prompt_attn_l11_budget_diagnostic_v1")
SOURCES = (DIAGNOSTIC / "states", Path("artifacts/prompt_attn_layer_selection_v1/states"))
OUTPUT = Path("artifacts/prompt_attn_l11_top_p_v1/preflight")
THRESHOLDS = (.75, .80, .85)


def summarize(values: list[int]) -> dict:
    data = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values), "mean": float(data.mean()), "std": float(data.std()),
        "min": int(data.min()), "p10": float(np.percentile(data, 10)),
        "median": float(np.median(data)), "p90": float(np.percentile(data, 90)),
        "max": int(data.max()),
    }


def main() -> None:
    rows = []
    masks = defaultdict(dict)
    manifest = json.loads((DIAGNOSTIC / "selection_manifest.json").read_text())["tasks"]
    paths = []
    for task, seeds in manifest.items():
        for seed in seeds:
            candidates = []
            for source in SOURCES:
                candidates = sorted((source / task / f"seed_{seed:03d}").glob("*.npz"))
                if candidates:
                    break
            if len(candidates) != 3:
                raise RuntimeError(f"expected three saved states for {task} seed {seed}, found {len(candidates)}")
            paths.extend((task, path) for path in candidates)
    for task, path in paths:
        data = np.load(path)
        scores = np.asarray(data["original"][11], dtype=np.float64)
        probability = scores / scores.sum()
        order = np.lexsort((np.arange(256), -probability))
        cumulative = np.cumsum(probability[order])
        state = f"{task}/{path.parent.name}/{path.name}"
        for threshold in THRESHOLDS:
            raw = int(np.searchsorted(cumulative, threshold) + 1)
            count = min(64, max(16, raw))
            selected = tuple(sorted(order[:count].tolist()))
            masks[state][threshold] = selected
            rows.append({
                "state": state, "task": task, "threshold": threshold,
                "m_raw": raw, "m_t": count,
                "selected_attention_mass": float(probability[list(selected)].sum()),
                "lower_bound_triggered": raw < 16,
                "upper_bound_triggered": raw > 64,
            })

    if len(masks) != 60:
        raise RuntimeError(f"expected 60 states, found {len(masks)}")
    summary = {"state_count": len(masks), "thresholds_locked": list(THRESHOLDS), "overall": {}, "tasks": {}}
    for threshold in THRESHOLDS:
        selected = [row for row in rows if row["threshold"] == threshold]
        summary["overall"][str(threshold)] = {
            "m_raw": summarize([row["m_raw"] for row in selected]),
            "m_t": summarize([row["m_t"] for row in selected]),
            "lower_trigger_rate": float(np.mean([row["lower_bound_triggered"] for row in selected])),
            "upper_trigger_rate": float(np.mean([row["upper_bound_triggered"] for row in selected])),
        }
    for task in sorted({row["task"] for row in rows}):
        summary["tasks"][task] = {}
        for threshold in THRESHOLDS:
            selected = [row for row in rows if row["task"] == task and row["threshold"] == threshold]
            summary["tasks"][task][str(threshold)] = {
                "m_t": summarize([row["m_t"] for row in selected]),
                "lower_trigger_rate": float(np.mean([row["lower_bound_triggered"] for row in selected])),
                "upper_trigger_rate": float(np.mean([row["upper_bound_triggered"] for row in selected])),
            }
    states = list(masks)
    summary["identical_mask_rates"] = {
        "all_three": float(np.mean([len({masks[state][t] for t in THRESHOLDS}) == 1 for state in states])),
        "p75_p80": float(np.mean([masks[state][.75] == masks[state][.80] for state in states])),
        "p80_p85": float(np.mean([masks[state][.80] == masks[state][.85] for state in states])),
    }
    summary["passed"] = (
        summary["identical_mask_rates"]["all_three"] < .10
        and all(summary["overall"][str(t)]["upper_trigger_rate"] < .25 for t in THRESHOLDS)
        and all(summary["overall"][str(t)]["lower_trigger_rate"] < .25 for t in THRESHOLDS)
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "offline_top_p_state_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (OUTPUT / "OFFLINE_THRESHOLD_PRECHECK.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise RuntimeError("Top-p thresholds are insufficiently distinct")


if __name__ == "__main__":
    main()
