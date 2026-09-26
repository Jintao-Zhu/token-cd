#!/usr/bin/env python3
"""Apply the frozen episode-median action-value metric to an EGL trace cohort."""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def auc_rescue_harm(rescue: np.ndarray, harm: np.ndarray) -> float:
    if not len(rescue) or not len(harm):
        return float("nan")
    cmp = rescue[:, None] - harm[None, :]
    return float((np.sum(cmp > 0) + 0.5 * np.sum(cmp == 0)) / cmp.size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact.resolve()
    audit = json.loads((root / "FINAL_AUDIT.json").read_text())
    if audit.get("status") != "PASS":
        raise SystemExit("Refusing analysis: cohort integrity audit is not PASS")
    manifest = json.loads((root / "cases_manifest.json").read_text())
    records = []
    for case in manifest["cases"]:
        cid = case["case_id"]
        pair = json.loads((root / "pairs" / f"{cid}.json").read_text())
        ep_path = root / "episodes" / cid / "matched" / "episode.json"
        ep = json.loads(ep_path.read_text())
        with np.load(ep_path.parent / "step_arrays.npz") as z:
            positive = z["positive_action_logits"].astype(np.float64)
            negative = z["negative_action_logits"].astype(np.float64)
        per_step_dim_g = []
        for t in range(positive.shape[0]):
            dim_g = []
            for q in range(min(6, positive.shape[1])):
                order = np.argsort(positive[t, q])[::-1]
                a1, a2 = int(order[0]), int(order[1])
                delta = positive[t, q] - negative[t, q]
                dim_g.append(float(delta[a1] - delta[a2]))
            per_step_dim_g.append(dim_g)
        values = np.asarray(per_step_dim_g, dtype=np.float64)
        episode_median = float(np.median(values))
        gate_step_score = np.median(values, axis=1)
        records.append({
            "case_id": cid,
            "task_id": int(pair["task_id"]),
            "init_state_id": int(pair["init_state_id"]),
            "outcome": pair["outcome"],
            "vanilla_success": bool(pair["vanilla_success"]),
            "matched_success": bool(pair["matched_success"]),
            "steps": int(pair["matched_steps"]),
            "episode_median_G": episode_median,
            "gate_on_fraction": float(np.mean(gate_step_score > 0)),
            "gate_score_episode_median": float(np.median(gate_step_score)),
        })

    rescue = np.asarray([r["episode_median_G"] for r in records if r["outcome"] == "rescue"])
    harm = np.asarray([r["episode_median_G"] for r in records if r["outcome"] == "harm"])
    auc = auc_rescue_harm(rescue, harm)
    all_values = np.concatenate([rescue, harm])
    nr = len(rescue)
    ge = 0
    total = 0
    for idx in itertools.combinations(range(len(all_values)), nr):
        mask = np.zeros(len(all_values), dtype=bool)
        mask[list(idx)] = True
        candidate = auc_rescue_harm(all_values[mask], all_values[~mask])
        ge += candidate >= auc - 1e-12
        total += 1
    rng = np.random.default_rng(20260925)
    if len(rescue) and len(harm):
        boot = np.empty(50000, dtype=np.float64)
        for i in range(len(boot)):
            boot[i] = auc_rescue_harm(
                rng.choice(rescue, len(rescue), replace=True),
                rng.choice(harm, len(harm), replace=True),
            )
        ci = np.quantile(boot, [0.025, 0.975]).tolist()
    else:
        ci = [None, None]
    outcomes = Counter(r["outcome"] for r in records)
    by_task = defaultdict(list)
    for r in records:
        by_task[r["task_id"]].append(r)
    result = {
        "protocol_id": manifest["protocol_id"],
        "artifact": str(root),
        "renderer": "EGL",
        "n_pairs": len(records),
        "metric": "episode median over all matched-trajectory step x first-six-dimension G values; G=(z+-z-)(clean top1)-(z+-z-)(clean top2)",
        "independent_unit": "paired initial state / episode",
        "outcomes": dict(outcomes),
        "successes": {
            "vanilla": sum(r["vanilla_success"] for r in records),
            "l11_matched": sum(r["matched_success"] for r in records),
        },
        "rescue_episode_median_G": float(np.median(rescue)) if len(rescue) else None,
        "harm_episode_median_G": float(np.median(harm)) if len(harm) else None,
        "auc_higher_G_predicts_rescue": auc,
        "exact_one_sided_permutation_p": float(ge / total) if total else None,
        "permutation_extreme": ge,
        "permutation_total": total,
        "whole_episode_bootstrap_95_percentile_ci_auc": ci,
        "gate_on_fraction_episode_median": {
            label: float(np.median([r["gate_on_fraction"] for r in records if r["outcome"] == label]))
            if any(r["outcome"] == label for r in records) else None
            for label in ("rescue", "harm", "concordant")
        },
        "per_task": {
            str(task): {
                "n": len(rows),
                "outcomes": dict(Counter(r["outcome"] for r in rows)),
                "rescue_median_G": float(np.median([r["episode_median_G"] for r in rows if r["outcome"] == "rescue"])) if any(r["outcome"] == "rescue" for r in rows) else None,
                "harm_median_G": float(np.median([r["episode_median_G"] for r in rows if r["outcome"] == "harm"])) if any(r["outcome"] == "harm" for r in rows) else None,
            }
            for task, rows in sorted(by_task.items())
        },
        "per_episode": records,
        "interpretation": "Observational analysis of the independently completed EGL N=40 cohort; the metric's association is not a causal gate effect. Keep EGL and OSMesa results separate.",
    }
    output = root / "ACTION_VALUE_ANALYSIS.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_episode"}, indent=2))


if __name__ == "__main__":
    main()
