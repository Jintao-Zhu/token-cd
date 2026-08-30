#!/usr/bin/env python3
"""SEMANTIC_TOKEN_CD Phase-1 selector evaluation.

Reads group_features_npz, runs 4 selectors (oracle / random / language / hybrid),
computes oracle recovery (top-1), residual alignment vs r_PCD, semantic IoU,
task breakdown, gates A/B/C, STOP RULE verdict. Writes selector_results.json,
CSVs, visualization, decision.json, REPORT_ZH.md.

Run:
  cd /home/leju-suzhou/zjt_ws/token-cd
  task1/.venvs/openvla-ar/bin/python research/semantic_token_cd/evaluate_selector.py \
    --artifact artifacts/semantic_token_cd_phase1_selector_v1 --split confirmation
"""
from __future__ import annotations

import argparse
import csv
import json
from math import sqrt
from pathlib import Path

import numpy as np
import torch

from research.cw_lpcd.engine import random_seed_for
from research.cw_lpcd.metrics import per_position_cosine, state_cosine

K = 8


def log_softmax_residual(clean: np.ndarray, branch: np.ndarray) -> np.ndarray:
    ca = torch.tensor(clean, dtype=torch.float64)
    ba = torch.tensor(branch, dtype=torch.float64)
    return (torch.log_softmax(ca, -1) - torch.log_softmax(ba, -1)).numpy()


def _med(pool):
    pool = np.asarray(pool, dtype=np.float64)
    pool = pool[np.isfinite(pool)]
    return float(np.median(pool)) if pool.size else float("nan")


def _mean(pool):
    pool = np.asarray(pool, dtype=np.float64)
    pool = pool[np.isfinite(pool)]
    return float(np.mean(pool)) if pool.size else float("nan")


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-12)


def mcnemar_p(n: int, b: int, c: int) -> float:
    """One-sided McNemar p-value: b = sel-correct&rnd-wrong, c = sel-wrong&rnd-correct.
    Tests H1: selector > random (exact binomial on discordant pairs)."""
    from scipy.stats import binom
    if b + c == 0:
        return 1.0
    return float(binom.sf(b - 1, b + c, 0.5))


def load_states(feat_dir: Path, state_ids: list[str]) -> dict:
    return {sid: np.load(feat_dir / f"{sid}.npz") for sid in state_ids
            if (feat_dir / f"{sid}.npz").exists()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--split", choices=["selection", "confirmation"], default="confirmation")
    p.add_argument("--viz-states", type=int, default=6)
    a = p.parse_args()
    art = a.artifact.resolve()
    feat_dir = art / "group_features_npz"

    split = json.loads((art / "FROZEN_SPLIT.json").read_text())
    key = "selection_state_ids" if a.split == "selection" else "confirmation_state_ids"
    state_ids = split[key]
    states = load_states(feat_dir, state_ids)
    n = len(states)
    missing = [s for s in state_ids if s not in states]
    print(json.dumps({"split": a.split, "n_loaded": n, "n_missing": len(missing)}))

    selectors = ("oracle", "random", "language", "hybrid", "d_only")
    rows = []  # per state per selector
    for sid in state_ids:
        if sid not in states:
            continue
        d = states[sid]
        v = d["v"].astype(np.float64)
        l = d["l"].astype(np.float64)
        D = d["D"].astype(np.float64)
        oracle = int(d["g_obj_idx"][0])
        labels = d["labels"].astype(np.int64)
        obj = set(int(x) for x in d["object_ids"].tolist())
        clean = d["clean_action"]
        r_pixel = d["r_pixel"].astype(np.float64)
        masked = d["masked"]

        # language score
        vn = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
        ln = l / (np.linalg.norm(l) + 1e-8)
        slang = vn @ ln
        # hybrid score
        shyb = 0.5 * minmax(slang) + 0.5 * minmax(D)
        rnd_g = int(np.random.default_rng(random_seed_for(sid) + 2000).integers(0, K))

        chosen = {
            "oracle": oracle,
            "random": rnd_g,
            "language": int(np.argmax(slang)),
            "hybrid": int(np.argmax(shyb)),
            "d_only": int(np.argmax(D)),
        }
        for sel in selectors:
            g = chosen[sel]
            grp = set(int(x) for x in np.flatnonzero(labels == g).tolist())
            iou = len(grp & obj) / len(grp | obj) if (grp | obj) else 0.0
            r = log_softmax_residual(clean, masked[g])
            pc = per_position_cosine(r, r_pixel)  # [7]
            rows.append({
                "state_id": sid, "task": str(d["task"]), "selector": sel,
                "group": g, "hit": int(g == oracle), "iou": iou,
                "cos_pos_mean": float(np.mean(pc)), "cos_pos_median": float(np.median(pc)),
                "cos_state": state_cosine(r, r_pixel),
            })

    # ---------------- aggregate ----------------
    agg = {}
    for sel in selectors:
        rr = [r for r in rows if r["selector"] == sel]
        agg[sel] = {
            "oracle_recovery": _mean([r["hit"] for r in rr]),
            "iou_median": _med([r["iou"] for r in rr]),
            "cos_pos_mean": _mean([r["cos_pos_mean"] for r in rr]),
            "cos_pos_median_pooled": _med([r["cos_pos_mean"] for r in rr]),
            "cos_state_mean": _mean([r["cos_state"] for r in rr]),
        }

    # pooled per-position alignment (mean + median over states x 7)
    pooled_mean, pooled_med = {}, {}
    for sel in selectors:
        vals = []
        for sid in state_ids:
            if sid not in states:
                continue
            d = states[sid]
            v = d["v"].astype(np.float64); l = d["l"].astype(np.float64); D = d["D"].astype(np.float64)
            vn = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
            ln = l / (np.linalg.norm(l) + 1e-8)
            slang = vn @ ln; shyb = 0.5 * minmax(slang) + 0.5 * minmax(D)
            oracle = int(d["g_obj_idx"][0])
            rnd_g = int(np.random.default_rng(random_seed_for(sid) + 2000).integers(0, K))
            g = {"oracle": oracle, "random": rnd_g, "language": int(np.argmax(slang)),
                 "hybrid": int(np.argmax(shyb)), "d_only": int(np.argmax(D))}[sel]
            r = log_softmax_residual(d["clean_action"], d["masked"][g])
            vals.extend(per_position_cosine(r, d["r_pixel"].astype(np.float64)).tolist())
        pooled_mean[sel] = _mean(vals)
        pooled_med[sel] = _med(vals)

    # ---------------- McNemar significance (language/hybrid vs random) ----------------
    def mcnemar(sel):
        b = c = 0
        rr = {r["state_id"]: r for r in rows if r["selector"] == sel}
        rnd = {r["state_id"]: r for r in rows if r["selector"] == "random"}
        for sid in rr:
            if rr[sid]["hit"] == 1 and rnd[sid]["hit"] == 0:
                b += 1
            elif rr[sid]["hit"] == 0 and rnd[sid]["hit"] == 1:
                c += 1
        return mcnemar_p(n, b, c)

    p_lang = mcnemar("language")
    p_hyb = mcnemar("hybrid")

    # ---------------- task breakdown ----------------
    task_rows = {}
    for r in rows:
        task_rows.setdefault(r["task"], {}).setdefault(r["selector"], []).append(r["hit"])
    task_breakdown = []
    for t in sorted(task_rows):
        task_breakdown.append({
            "task": t,
            "n_states": len(task_rows[t]["oracle"]),
            "oracle_recovery": _mean(task_rows[t]["oracle"]),
            "language_recovery": _mean(task_rows[t]["language"]),
            "hybrid_recovery": _mean(task_rows[t]["hybrid"]),
            "random_recovery": _mean(task_rows[t]["random"]),
        })

    # ---------------- gates ----------------
    rnd_rec = agg["random"]["oracle_recovery"]
    lang_rec = agg["language"]["oracle_recovery"]
    hyb_rec = agg["hybrid"]["oracle_recovery"]
    best_rec = max(lang_rec, hyb_rec)
    best_alignment = max(pooled_mean["language"], pooled_mean["hybrid"])

    gateA_lang = (lang_rec > 0.30) and (lang_rec > rnd_rec)
    gateA_hyb = (hyb_rec > 0.30) and (hyb_rec > rnd_rec)
    gateA = gateA_lang or gateA_hyb

    gateB_lang = (pooled_mean["language"] > 0.4) and (pooled_mean["language"] > pooled_mean["random"])
    gateB_hyb = (pooled_mean["hybrid"] > 0.4) and (pooled_mean["hybrid"] > pooled_mean["random"])
    gateB = gateB_lang or gateB_hyb

    n_tasks = len(task_breakdown)
    gateC_tasks = sum(1 for tb in task_breakdown
                      if max(tb["language_recovery"], tb["hybrid_recovery"]) >= tb["random_recovery"])
    gateC = gateC_tasks >= max(7, int(0.7 * n_tasks))

    # stop flags
    stop_lang_random = (lang_rec <= rnd_rec + 0.05)  # language ≈ random
    # hybrid only relies on D: hybrid == d_only almost always, and language adds nothing
    hyb_group = {r["state_id"]: r["group"] for r in rows if r["selector"] == "hybrid"}
    donly_group = {r["state_id"]: r["group"] for r in rows if r["selector"] == "d_only"}
    hyb_donly_agree = _mean([1 if hyb_group[sid] == donly_group[sid] else 0 for sid in hyb_group])
    stop_hybrid_only_D = (hyb_donly_agree > 0.9) and (lang_rec <= rnd_rec + 0.05)
    stop_align_low = best_alignment < 0.3

    stop_hit = stop_lang_random or stop_hybrid_only_D or stop_align_low
    verdict = "PASS_TO_ROLLOUT" if (gateA and gateB and gateC and not stop_hit) else "STOP_SEMANTIC_TOKEN_CD_NO_GO"

    # ---------------- write CSVs ----------------
    with (art / "oracle_vs_pred.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["state_id", "task", "selector", "group", "hit", "iou",
                                           "cos_pos_mean", "cos_pos_median", "cos_state"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with (art / "task_breakdown.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["task", "n_states", "oracle_recovery", "language_recovery",
                                           "hybrid_recovery", "random_recovery"])
        w.writeheader()
        for tb in task_breakdown:
            w.writerow(tb)

    # ---------------- selector_results.json ----------------
    results = {
        "experiment": "SEMANTIC_TOKEN_CD_PHASE1_SELECTOR_V1",
        "split": a.split, "n_states": n, "n_missing": len(missing),
        "verdict": verdict, "stop_rule_hit": stop_hit,
        "aggregate": agg,
        "pooled_alignment": {"mean": pooled_mean, "median": pooled_med},
        "mcnemar_p_vs_random": {"language": p_lang, "hybrid": p_hyb},
        "hybrid_vs_d_only_agreement": hyb_donly_agree,
        "gates": {
            "A_selector_recovery": {"PASS": bool(gateA), "language": lang_rec, "hybrid": hyb_rec,
                                    "random": rnd_rec, "lang_pass": bool(gateA_lang), "hyb_pass": bool(gateA_hyb)},
            "B_alignment": {"PASS": bool(gateB), "language": pooled_mean["language"], "hybrid": pooled_mean["hybrid"],
                            "random": pooled_mean["random"]},
            "C_task_consistency": {"PASS": bool(gateC), "n_tasks_pass": gateC_tasks, "n_tasks": n_tasks},
        },
        "stop_flags": {
            "language_approx_random": bool(stop_lang_random),
            "hybrid_only_D": bool(stop_hybrid_only_D),
            "alignment_low": bool(stop_align_low),
        },
        "task_breakdown": task_breakdown,
        "oracle_reference": {"cos_pos_pooled_mean": pooled_mean["oracle"],
                             "cos_pos_median_pooled": pooled_med["oracle"],
                             "phase0_kmeans8_cos": 0.7029},
    }
    (art / "selector_results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")

    decision = {
        "experiment": "SEMANTIC_TOKEN_CD_PHASE1_SELECTOR_V1",
        "status": verdict, "split": a.split, "n_states": n, "date": "2026-08-23",
        "gates": {k: ("PASS" if v["PASS"] else "FAIL") for k, v in results["gates"].items()},
        "core_numbers": {
            "oracle_recovery_language": lang_rec, "oracle_recovery_hybrid": hyb_rec,
            "oracle_recovery_random": rnd_rec, "oracle_recovery_d_only": agg["d_only"]["oracle_recovery"],
            "alignment_mean_language": pooled_mean["language"], "alignment_mean_hybrid": pooled_mean["hybrid"],
            "alignment_mean_random": pooled_mean["random"], "alignment_mean_oracle": pooled_mean["oracle"],
            "mcnemar_p_language_vs_random": p_lang,
        },
        "closed_loop": "NOT_RUN",
    }
    (art / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")

    print(json.dumps(results, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
