#!/usr/bin/env python3
"""Compare episode-level counterfactual action-margin effects by dimension.

All cohorts use the same clean top-1/top-2 G definition. This analysis is
observational and does not treat replans as independent samples.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
SIMPLER_ROOT = REPO / "artifacts/prompt_attn_l11_matched_full_9task_0_299_v1/run"
SIMPLER_LEDGER = REPO / "artifacts/simpler_action_value_transfer_analysis_v1/SIMPLER_ACTION_VALUE_ANALYSIS.json"
LIBERO_N50 = REPO / "artifacts/libero_action_value_gate_egl_unseen_task_v1_20260925"
LIBERO_EGL40 = REPO / "artifacts/libero_action_value_egl_gpu6_n40_20260925"
LIBERO_OSMESA40 = REPO / "artifacts/libero_action_value_trace_pilot_v2_osmesa_ext40_init8_15_20260925"
OUT = REPO / "artifacts/cross_benchmark_dimwise_action_value_v1"


def episode_g(pos: np.ndarray, neg: np.ndarray) -> tuple[list[float], float]:
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    qn = min(6, pos.shape[1])
    result = []
    all_values = []
    for q in range(qn):
        values = []
        for t in range(pos.shape[0]):
            z = pos[t, q]
            zn = neg[t, q]
            idx = np.argsort(z)[::-1]
            a1, a2 = int(idx[0]), int(idx[1])
            g = float((z[a1] - zn[a1]) - (z[a2] - zn[a2]))
            values.append(g)
            all_values.append(g)
        result.append(float(np.median(values)))
    return result, float(np.median(all_values))


def summarize(rows: list[dict]) -> dict:
    outcomes = Counter(r["outcome"] for r in rows)
    dims = {}
    for q in range(6):
        rescue = [r["G_by_dim"][q] for r in rows if r["outcome"] == "rescue"]
        harm = [r["G_by_dim"][q] for r in rows if r["outcome"] == "harm"]
        if rescue and harm:
            auc = sum((a > b) + 0.5 * (a == b) for a in rescue for b in harm) / (len(rescue) * len(harm))
        else:
            auc = None
        dims[str(q)] = {
            "n_rescue": len(rescue), "n_harm": len(harm),
            "rescue_median_G": float(np.median(rescue)) if rescue else None,
            "harm_median_G": float(np.median(harm)) if harm else None,
            "rescue_minus_harm_median": float(np.median(rescue) - np.median(harm)) if rescue and harm else None,
            "auc_rescue_higher": float(auc) if auc is not None else None,
            "rescue_values": rescue, "harm_values": harm,
        }
    overall_rescue = [r["G_overall"] for r in rows if r["outcome"] == "rescue"]
    overall_harm = [r["G_overall"] for r in rows if r["outcome"] == "harm"]
    overall_auc = (sum((a > b) + 0.5 * (a == b) for a in overall_rescue for b in overall_harm)
                   / (len(overall_rescue) * len(overall_harm))) if overall_rescue and overall_harm else None
    return {
        "n_episodes": len(rows), "outcome_counts": dict(outcomes),
        "overall_median_G": {
            "rescue": float(np.median(overall_rescue)) if overall_rescue else None,
            "harm": float(np.median(overall_harm)) if overall_harm else None,
            "auc_rescue_higher": float(overall_auc) if overall_auc is not None else None,
        },
        "per_dimension": dims,
    }


def load_simpler() -> tuple[list[dict], dict]:
    ledger = json.loads(SIMPLER_LEDGER.read_text())
    selected = [r for r in ledger["episode_values"]
                if r["task"] in {"google_robot_open_drawer", "google_robot_close_drawer",
                                 "google_robot_pick_coke_can", "google_robot_move_near"}
                and 100 <= int(r["seed"]) <= 199 and r["outcome"] in {"rescue", "harm"}]
    rows = []
    missing = []
    for r in selected:
        path = SIMPLER_ROOT / "episodes" / r["task"] / "prompt_single" / f"episode_{int(r['seed'])}_arrays.npz"
        if not path.is_file():
            missing.append({"task": r["task"], "seed": r["seed"], "outcome": r["outcome"]})
            continue
        z = np.load(path)
        by_dim, pooled = episode_g(z["positive"], z["negative"])
        rows.append({"case_id": f"{r['task']}__seed{r['seed']:03d}", "task": r["task"],
                     "outcome": r["outcome"], "G_by_dim": by_dim,
                     "G_overall": pooled})
    expected = ledger["primary_four_task_seed100_199"]
    counts = Counter(r["outcome"] for r in rows)
    if missing or counts["rescue"] != expected["rescue_n"] or counts["harm"] != expected["harm_n"]:
        raise RuntimeError(f"SIMPLER coverage mismatch: {dict(counts)}, missing={len(missing)}")
    return rows, {"source": str(SIMPLER_LEDGER), "array_coverage": len(rows), "missing_arrays": missing,
                  "expected_rescue_harm": {"rescue": expected["rescue_n"], "harm": expected["harm_n"]}}


def load_libero(root: Path, name: str, arms_layout: str = "gate") -> list[dict]:
    rows = []
    if arms_layout == "gate":
        pairs = [json.loads(p.read_text()) for p in sorted((root / "pairs").glob("*.json"))]
        for pair in pairs:
            outcome = "rescue" if pair["matched_success"] and not pair["vanilla_success"] else \
                      "harm" if pair["vanilla_success"] and not pair["matched_success"] else "concordant"
            if outcome not in {"rescue", "harm"}:
                continue
            p = root / "episodes" / pair["case_id"] / "matched" / "step_arrays.npz"
            a = np.load(p)
            by_dim, pooled = episode_g(a["positive_action_logits"], a["negative_action_logits"])
            rows.append({"case_id": pair["case_id"], "task": pair["task_name"], "outcome": outcome,
                         "G_by_dim": by_dim, "G_overall": pooled})
    else:
        audit = json.loads((root / "FINAL_AUDIT.json").read_text())
        if audit.get("status") != "PASS":
            raise RuntimeError(f"{name}: integrity audit not PASS")
        pairs = [json.loads(p.read_text()) for p in sorted((root / "pairs").glob("*.json"))]
        for pair in pairs:
            if pair["outcome"] not in {"rescue", "harm"}:
                continue
            p = root / "episodes" / pair["case_id"] / "matched" / "step_arrays.npz"
            a = np.load(p)
            by_dim, pooled = episode_g(a["positive_action_logits"], a["negative_action_logits"])
            rows.append({"case_id": pair["case_id"], "task": pair["task_name"], "outcome": pair["outcome"],
                         "G_by_dim": by_dim, "G_overall": pooled})
    return rows


def main():
    simpler_rows, simpler_meta = load_simpler()
    cohorts = {
        "simpler_four_google_seed100_199": (simpler_rows, simpler_meta),
        "libero_egl_unseen_task_n50": (load_libero(LIBERO_N50, "n50", "gate"), {"source": str(LIBERO_N50), "renderer": "EGL"}),
        "libero_fresh_egl_n40": (load_libero(LIBERO_EGL40, "egl40", "pairs"), {"source": str(LIBERO_EGL40), "renderer": "EGL"}),
        "libero_osmesa_heldout_n40": (load_libero(LIBERO_OSMESA40, "osmesa40", "pairs"), {"source": str(LIBERO_OSMESA40), "renderer": "OSMesa"}),
    }
    payload = {
        "protocol": "CROSS_BENCHMARK_DIMWISE_ACTION_VALUE_V1",
        "metric": "G=(z+-z-)[clean top1]-(z+-z-)[clean top2]. Overall episode G is the median over replans and action dimensions 0..5; each per-dimension value is the median over replans for that dimension.",
        "cohorts": {name: {**meta, **summarize(rows), "episodes": rows} for name, (rows, meta) in cohorts.items()},
        "limits": [
            "The SIMPLER and LIBERO cohorts differ in benchmark, tasks, renderer, and/or task split; dimension-wise contrast is observational, not a causal benchmark effect.",
            "LIBERO discordant counts are small in each cohort; confidence intervals and task-stratified tests are not used to avoid overclaiming.",
            "Dimensions share the same model output format but their physical semantics are not guaranteed to be identical across task suites.",
            "Results are exploratory and use only episodes with retained logits; the SIMPLER subset includes 101 Rescue and 28 Harm arrays from the declared 400-pair cohort.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "CROSS_BENCHMARK_DIMWISE_ACTION_VALUE.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Cross-benchmark action guidance by action dimension",
        "",
        "Same clean top-1/top-2 `G` definition in every cohort. Each episode contributes one overall median across replans × action dimensions, plus one median per action dimension; episodes, not replans, are the unit. This is exploratory and not a causal benchmark comparison.",
        "",
        "## Cohort outcomes",
        "",
        "| Cohort | N discordant | Rescue / Harm | Overall Rescue-higher AUC |",
        "|---|---:|---:|---:|",
    ]
    for name, (rows, meta) in cohorts.items():
        s = summarize(rows); c = s["outcome_counts"]
        lines.append(f"| {name} | {len(rows)} | {c.get('rescue',0)} / {c.get('harm',0)} | {s['overall_median_G']['auc_rescue_higher']:.3f} |")
    lines += ["", "## Per-dimension episode-level association", ""]
    for name, (rows, _meta) in cohorts.items():
        s = summarize(rows)
        lines += [f"### {name}", "", "| Action dim | Rescue n, median G | Harm n, median G | Rescue-higher AUC |", "|---:|---:|---:|---:|"]
        for q, d in s["per_dimension"].items():
            rv = "—" if d["rescue_median_G"] is None else f"{d['rescue_median_G']:.3f}"
            hv = "—" if d["harm_median_G"] is None else f"{d['harm_median_G']:.3f}"
            auc = "—" if d["auc_rescue_higher"] is None else f"{d['auc_rescue_higher']:.3f}"
            lines.append(f"| {q} | {d['n_rescue']}, {rv} | {d['n_harm']}, {hv} | {auc} |")
        lines.append("")
    lines += [
        "## Interpretation rule",
        "",
        "Do not infer a dimension mechanism from the small LIBERO AUCs alone. A candidate dimension is only worth a confirmatory test if the per-dimension Rescue/Harm direction is consistent across independent LIBERO cohorts and is different from the same dimension's SIMPLER direction; these cohorts are small and not task/checkpoint/renderer matched.",
        "",
        "Machine-readable per-episode values: [CROSS_BENCHMARK_DIMWISE_ACTION_VALUE.json](CROSS_BENCHMARK_DIMWISE_ACTION_VALUE.json). Reproducer: [analyzer](../../research/semantic_token_cd/analyze_cross_benchmark_dimwise_action_value.py).",
    ]
    (OUT / "CROSS_BENCHMARK_DIMWISE_ACTION_VALUE.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: {"n": summarize(v[0])["n_episodes"], "counts": summarize(v[0])["outcome_counts"],
                        "overall_auc": summarize(v[0])["overall_median_G"]["auc_rescue_higher"],
                        "dimension_auc": {q: d["auc_rescue_higher"] for q, d in summarize(v[0])["per_dimension"].items()}}
                      for k, v in cohorts.items()}, indent=2))


if __name__ == "__main__":
    main()
