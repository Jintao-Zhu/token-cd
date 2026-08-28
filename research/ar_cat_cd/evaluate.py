#!/usr/bin/env python3
"""CAT-CD Phase-0 evaluation: gates A/B/C, attention comparison, STOP RULE verdict.

Reads per-state cat_cd_npz from the compute loop, recomputes all residuals and
cosines offline (log_softmax space, matching r_pixel), aggregates over the
Confirmation split, and writes phase0_decision.json + REPORT_ZH.md.

Run:
  cd /data/docker/dev_zjt/data/code
  task1/.venvs/openvla-ar/bin/python research/ar_cat_cd/evaluate.py \
    --artifact artifacts/cat_cd_phase0_v1 --split confirmation
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from research.cw_lpcd.metrics import per_position_cosine, state_cosine

KS = (4, 8, 16)


def log_softmax_residual(clean: np.ndarray, branch: np.ndarray) -> np.ndarray:
    """r = log_softmax(clean[action]) - log_softmax(branch[action]); [7,256] (float64)."""
    ca = torch.tensor(clean, dtype=torch.float64)
    ba = torch.tensor(branch, dtype=torch.float64)
    return (torch.log_softmax(ca, -1) - torch.log_softmax(ba, -1)).numpy()


def norm(r: np.ndarray) -> float:
    return float(np.linalg.norm(r))


def _med(pool):
    pool = np.asarray(pool, dtype=np.float64)
    pool = pool[np.isfinite(pool)]
    return float(np.median(pool)) if pool.size else float("nan")


def load_states(npz_dir: Path, state_ids: list[str]) -> dict:
    return {sid: np.load(npz_dir / f"{sid}.npz") for sid in state_ids
            if (npz_dir / f"{sid}.npz").exists()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--split", choices=["selection", "confirmation"], default="confirmation")
    a = p.parse_args()
    art = a.artifact.resolve()
    npz_dir = art / "cat_cd_npz"

    split = json.loads((art / "FROZEN_SPLIT.json").read_text())
    key = "selection_state_ids" if a.split == "selection" else "confirmation_state_ids"
    state_ids = split[key]
    states = load_states(npz_dir, state_ids)
    n = len(states)
    missing = [s for s in state_ids if s not in states]
    print(json.dumps({"split": a.split, "n_loaded": n, "n_missing": len(missing), "missing_head": missing[:5]}))

    # per (state) x (method) x (k) metrics
    rows = []  # one row per state per k
    for sid in state_ids:
        if sid not in states:
            continue
        d = states[sid]
        obj = set(int(x) for x in d["object_ids"].tolist())
        n_obj = len(obj)
        clean = d["clean_action"]
        r_pixel = d["r_pixel"].astype(np.float64)
        chance = n_obj / 256.0
        for k in KS:
            for m in ("cat", "attn", "rnd"):
                ids = d[f"{m}_ids_{k}"].tolist()
                branch = d[f"{m}_action_{k}"]
                r = log_softmax_residual(clean, branch)
                pc = per_position_cosine(r, r_pixel)          # [7]
                sc = state_cosine(r, r_pixel)                  # scalar
                D = norm(r)
                prec = len(set(ids) & obj) / k
                rec = len(set(ids) & obj) / n_obj
                rows.append({
                    "state_id": sid, "task": str(d["task"]), "k": k, "method": m,
                    "D": D, "cos_pos": float(np.median(pc)), "cos_pos_mean": float(np.mean(pc)),
                    "cos_state": sc, "precision": prec, "recall": rec, "chance": chance,
                })

    # ---- aggregate: median over states (D, precision, recall), median-pooled cosine over (state x position) ----
    agg = {}
    for m in ("cat", "attn", "rnd"):
        for k in KS:
            rr = [r for r in rows if r["method"] == m and r["k"] == k]
            agg[(m, k)] = {
                "D_median": _med([r["D"] for r in rr]),
                "cos_pos_median": _med([r["cos_pos"] for r in rr]),
                "cos_state_median": _med([r["cos_state"] for r in rr]),
                "precision_median": _med([r["precision"] for r in rr]),
                "recall_median": _med([r["recall"] for r in rr]),
            }

    # pooled per-position cosine (all states x 7 positions -> median), matching cw_lpcd STEP 9
    def pooled_cos_pos(m, k):
        vals = []
        for sid in state_ids:
            if sid not in states:
                continue
            d = states[sid]
            r = log_softmax_residual(d["clean_action"], d[f"{m}_action_{k}"])
            vals.extend(per_position_cosine(r, d["r_pixel"].astype(np.float64)).tolist())
        return _med(vals)

    pooled = {}
    for m in ("cat", "attn", "rnd"):
        for k in KS:
            pooled[(m, k)] = pooled_cos_pos(m, k)

    # ---- gates ----
    gateA = all(agg[("cat", k)]["D_median"] > agg[("rnd", k)]["D_median"] for k in KS)
    gateB = all(pooled[("cat", k)] > pooled[("rnd", k)] for k in KS)
    # Gate C: precision over object, at k=16 (largest, most informative)
    mean_chance = _med([r["chance"] for r in rows if r["k"] == 16 and r["method"] == "cat"])
    gateC = (agg[("cat", 16)]["precision_median"] > agg[("rnd", 16)]["precision_median"]) and \
            (agg[("cat", 16)]["precision_median"] >= 2.0 * mean_chance)

    attn_pass = all(
        agg[("cat", k)]["D_median"] > agg[("attn", k)]["D_median"] and
        pooled[("cat", k)] > pooled[("attn", k)]
        for k in KS
    )

    stop_hit = not (gateA and gateB and gateC)
    verdict = "STOP_CAT_CD_NO_GO"
    if not stop_hit and attn_pass:
        verdict = "PASS_TO_PHASE1"
    elif not stop_hit and not attn_pass:
        verdict = "STOP_CAT_CD_NO_GO"  # beat random but not attention -> still stop per spec core question

    # ---- per-task structure breakdown (precision at k=16) ----
    task_rows = {}
    for r in rows:
        if r["k"] != 16:
            continue
        t = r["task"]
        task_rows.setdefault(t, {}).setdefault(r["method"], []).append(r["precision"])
    task_breakdown = []
    for t in sorted(task_rows):
        task_breakdown.append({
            "task": t,
            "cat_precision_median": _med(task_rows[t].get("cat", [])),
            "attn_precision_median": _med(task_rows[t].get("attn", [])),
            "rnd_precision_median": _med(task_rows[t].get("rnd", [])),
            "n_states": len(task_rows[t].get("cat", [])),
        })

    # ---- write CSVs ----
    with (art / "cat_cd_per_state.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["state_id", "task", "k", "method", "D",
                                           "cos_pos", "cos_pos_mean", "cos_state", "precision", "recall", "chance"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with (art / "cat_cd_task_breakdown.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["task", "cat_precision_median", "attn_precision_median",
                                           "rnd_precision_median", "n_states"])
        w.writeheader()
        for r in task_breakdown:
            w.writerow(r)

    # ---- decision JSON ----
    results = {
        "experiment": "AR_CAT_CD_PHASE0_V1",
        "split": a.split, "n_states": n, "n_missing": len(missing),
        "verdict": verdict,
        "stop_rule_hit": stop_hit,
        "mean_chance_precision": mean_chance,
        "gates": {
            "A_disruption_CAT_vs_RANDOM": {"PASS": bool(gateA), "per_k": {
                str(k): {"cat_D": agg[("cat", k)]["D_median"], "rnd_D": agg[("rnd", k)]["D_median"]} for k in KS}},
            "B_alignment_CAT_vs_RANDOM": {"PASS": bool(gateB), "per_k": {
                str(k): {"cat_cos": pooled[("cat", k)], "rnd_cos": pooled[("rnd", k)]} for k in KS}},
            "C_structure": {"PASS": bool(gateC),
                            "cat_precision_16": agg[("cat", 16)]["precision_median"],
                            "rnd_precision_16": agg[("rnd", 16)]["precision_median"],
                            "cat_recall_16": agg[("cat", 16)]["recall_median"],
                            "chance": mean_chance},
        },
        "attention_pass": bool(attn_pass),
        "aggregate_median": {f"{m}_k{k}": agg[(m, k)] for m in ("cat", "attn", "rnd") for k in KS},
        "pooled_cos_pos_median": {f"{m}_k{k}": pooled[(m, k)] for m in ("cat", "attn", "rnd") for k in KS},
        "task_breakdown": task_breakdown,
        "historical_reference": {
            "persistent_object_pixel_cosine_median": 0.4765,
            "persistent_random_pixel_cosine_median": 0.0721,
            "note": "旧 Persistent Token-PCD 全对象 mask 的对齐上限 0.4765 / random 0.0721（selection split）",
        },
    }
    (art / "evaluation_results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")

    decision = {
        "experiment": "AR_CAT_CD_PHASE0_V1",
        "status": verdict,
        "split": a.split, "n_states": n,
        "date": "2026-08-23",
        "gates": {k: ("PASS" if v["PASS"] else "FAIL") for k, v in results["gates"].items()},
        "attention_pass": bool(attn_pass),
        "core_numbers": {
            "cat_vs_random_disruption": {str(k): round(agg[("cat", k)]["D_median"] - agg[("rnd", k)]["D_median"], 4) for k in KS},
            "cat_vs_random_cos_pos_median": {str(k): round(pooled[("cat", k)] - pooled[("rnd", k)], 4) for k in KS},
            "cat_cos_pos_16": pooled[("cat", 16)],
            "rnd_cos_pos_16": pooled[("rnd", 16)],
            "cat_precision_16": agg[("cat", 16)]["precision_median"],
            "rnd_precision_16": agg[("rnd", 16)]["precision_median"],
            "chance_precision": mean_chance,
        },
        "closed_loop": "NOT_RUN",
    }
    (art / "phase0_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")

    print(json.dumps(results, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
