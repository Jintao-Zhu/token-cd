"""Stage 1: rank-only semantic-difference diagnosis on all 918 saved states."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from research.semantic_token_cd.prompt_attn_shr_policy import stable_top_m
from research.semantic_token_cd.xdiff_protocol import ARTIFACT, EPSILON, SOURCE_ARTIFACT, TASKS, atomic_json

CONFIGS = {
    "eta0": ("swapped", 0.0),
    "semantic_0p5": ("swapped", 0.5),
    "semantic_1p0": ("swapped", 1.0),
    "paraphrase_0p5": ("paraphrase", 0.5),
    "reverse_0p5": ("swapped", -0.5),
}


def score(raw_p: np.ndarray, raw_q: np.ndarray, eta: float):
    p = np.asarray(raw_p, dtype=np.float64)
    q = np.asarray(raw_q, dtype=np.float64)
    p /= p.sum()
    q /= q.sum()
    log_p = np.log(p + EPSILON)
    log_q = np.log(q + EPSILON)
    d = log_p - log_q
    return p, q, d, log_p + eta * d


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


def main():
    out_states = ARTIFACT / "stage1_rank" / "states"
    out_stats = ARTIFACT / "stage1_rank" / "statistics"
    out_states.mkdir(parents=True, exist_ok=True)
    out_stats.mkdir(parents=True, exist_ok=True)
    rows = []
    exact_eta0 = True
    for task in TASKS:
        src = SOURCE_ARTIFACT / "runs" / "offline" / task
        for jp in sorted(src.glob("seed_*/step_*.json")):
            if jp.name == "offline_summary.json":
                continue
            meta = json.loads(jp.read_text())
            npz = np.load(jp.with_suffix(".npz"))
            m = int(meta["branches"]["correct"]["m"])
            raw_p = np.asarray(npz["correct"], dtype=np.float64)
            original = [int(x) for x in meta["branches"]["correct"]["selected_token_ids"]]
            payload = {"correct_raw": raw_p}
            masks = {}
            state_metrics = {}
            p_ranks = np.empty(256, dtype=np.int64)
            p_ranks[np.lexsort((np.arange(256), -raw_p))] = np.arange(1, 257)
            for name, (q_name, eta) in CONFIGS.items():
                p, q, d, s = score(raw_p, np.asarray(npz[q_name]), eta)
                selected = stable_top_m(s, m)
                masks[name] = selected
                entered = sorted(set(selected) - set(original))
                exited = sorted(set(original) - set(selected))
                state_metrics[name] = {
                    "jaccard_vs_correct": jaccard(selected, original),
                    "entered_count": len(entered), "exited_count": len(exited),
                    "entered": entered, "exited": exited,
                    "entered_correct_rank_mean": float(np.mean(p_ranks[entered])) if entered else None,
                    "entered_correct_rank_max": int(np.max(p_ranks[entered])) if entered else None,
                    "entered_p_mean": float(np.mean(p[entered])) if entered else None,
                    "entered_q_mean": float(np.mean(q[entered])) if entered else None,
                    "entered_d_mean": float(np.mean(d[entered])) if entered else None,
                }
                payload[f"{name}__p"] = p
                payload[f"{name}__q"] = q
                payload[f"{name}__d"] = d
                payload[f"{name}__s"] = s
                mask = np.zeros(256, dtype=np.uint8); mask[selected] = 1
                payload[f"{name}__mask"] = mask
            eta0_ok = masks["eta0"] == sorted(original)
            exact_eta0 = exact_eta0 and eta0_ok
            rel = jp.relative_to(src).with_suffix("")
            op = out_states / task / rel
            op.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(op.with_suffix(".npz"), **payload)
            record = {
                "task": task, "seed": int(meta["seed"]),
                "control_step": int(meta["control_step"]), "m": m,
                "instruction": meta["instruction"],
                "semantic_contrast": meta["branches"]["swapped"]["selector_instruction"],
                "paraphrase": meta["branches"]["paraphrase"]["selector_instruction"],
                "eta0_exact_mask": eta0_ok, "configs": state_metrics,
                "arrays_file": str(op.with_suffix(".npz").relative_to(ARTIFACT)),
            }
            atomic_json(op.with_suffix(".json"), record)
            for name, vals in state_metrics.items():
                rows.append({"task": task, "seed": meta["seed"], "control_step": meta["control_step"],
                             "m": m, "config": name, **{k: v for k, v in vals.items()
                             if k not in ("entered", "exited")}})
    if not exact_eta0:
        raise RuntimeError("eta=0 did not exactly reproduce Correct masks")
    fields = list(rows[0])
    with (out_stats / "state_rank_metrics.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    summary = {"protocol": "PROMPT_ATTN_SEMANTIC_DIFFERENCE_V1", "n_states": len(rows)//len(CONFIGS),
               "eta0_exact_all_states": exact_eta0, "epsilon": EPSILON, "configs": {}}
    for name in CONFIGS:
        rr = [r for r in rows if r["config"] == name]
        summary["configs"][name] = {
            "mean_jaccard_vs_correct": float(np.mean([r["jaccard_vs_correct"] for r in rr])),
            "mean_entered_count": float(np.mean([r["entered_count"] for r in rr])),
            "changed_state_fraction": float(np.mean([r["entered_count"] > 0 for r in rr])),
            "mean_entered_correct_rank": float(np.nanmean([
                np.nan if r["entered_correct_rank_mean"] is None else r["entered_correct_rank_mean"] for r in rr])),
            "max_entered_correct_rank": int(max([r["entered_correct_rank_max"] or 0 for r in rr])),
        }
    atomic_json(out_stats / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
