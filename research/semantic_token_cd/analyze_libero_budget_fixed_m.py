#!/usr/bin/env python3
"""Audit and compare the held-out fixed-task-m budget arm."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

FIXED_M = {3: 16, 10: 25, 49: 47, 72: 42, 73: 46}


def read(path):
    return json.loads(path.read_text())


def summarize(y):
    return {"successes": int(sum(y)), "n": len(y), "rate": float(np.mean(y)) if y else None}


def paired(a, b):
    wins = sum(bool(x) and not bool(y) for x, y in zip(a, b))
    losses = sum(not bool(x) and bool(y) for x, y in zip(a, b))
    ties = len(a) - wins - losses
    p = float(binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else 1.0
    rng = np.random.default_rng(20260926)
    indexes = rng.integers(0, len(a), size=(20000, len(a)))
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    ci = np.quantile(delta[indexes].mean(axis=1), [0.025, 0.975])
    return {"wins_a": int(wins), "losses_a": int(losses), "ties": int(ties),
            "risk_difference_a_minus_b": float(delta.mean()),
            "bootstrap_95_ci": [float(x) for x in ci],
            "exact_mcnemar_p_two_sided": p}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--treatment", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    cases = read(a.baseline / "cases_manifest.json")["cases"]
    rows, errors = [], []
    for case in cases:
        cid, tid = case["case_id"], int(case["task_id"])
        expected_m = FIXED_M[tid]
        base_pair = read(a.baseline / "pairs" / f"{cid}.json")
        fixed_pair = read(a.treatment / "pairs" / f"{cid}.json")
        base_dir = a.baseline / "episodes" / cid / "matched"
        fixed_dir = a.treatment / "episodes" / cid / "matched_fixed_m"
        base_ep, fixed_ep = read(base_dir / "episode.json"), read(fixed_dir / "episode.json")
        base0, fixed0 = base_ep["trace"][0], fixed_ep["trace"][0]
        fixed_arr = np.load(fixed_dir / "step_arrays.npz")
        m_used = [int(step["m_used"]) for step in fixed_ep["trace"]]
        checks = {
            "case_identity": base_ep["case_id"] == cid == fixed_ep["case_id"],
            "task_init_identity": int(fixed_ep["task_id"]) == tid and int(fixed_ep["init_state_id"]) == int(case["init_state_id"]),
            "initial_state_hash": base_pair["initial_state_sha256"] == fixed_pair["initial_state_sha256"] == fixed_ep["initial_state_sha256"],
            "first_rgb_hash": base0["rgb_sha256"] == fixed0["rgb_sha256"],
            "first_attention_hash": base0["attention_sha256"] == fixed0["attention_sha256"],
            "first_clean_logits_bitwise": bool(np.array_equal(np.load(base_dir / "step_arrays.npz")["positive_action_logits"][0], fixed_arr["positive_action_logits"][0])),
            "fixed_m_recorded": all(v == expected_m for v in m_used) and int(fixed_pair["fixed_m"]) == expected_m,
            "finite_arrays": all(np.isfinite(fixed_arr[k]).all() for k in fixed_arr.files),
        }
        for key, okay in checks.items():
            if not okay:
                errors.append({"case_id": cid, "check": key})
        rows.append({"case_id": cid, "task_id": tid, "init_state_id": int(case["init_state_id"]),
                     "fixed_m": expected_m, "vanilla_success": bool(base_pair["vanilla_success"]),
                     "matched_success": bool(base_pair["matched_success"]),
                     "fixed_m_success": bool(fixed_pair["fixed_m_success"]), "checks": checks,
                     "canonical_steps": int(base_ep["steps"]), "fixed_m_steps": int(fixed_ep["steps"])})
    v = [r["vanilla_success"] for r in rows]
    dyn = [r["matched_success"] for r in rows]
    fix = [r["fixed_m_success"] for r in rows]
    result = {"protocol_id": "LIBERO90_L11_MATCHED_FIXED_TASK_M_V1", "n": len(rows),
              "integrity_pass": len(rows) == 40 and not errors, "integrity_errors": errors,
              "fixed_m_by_task_id": FIXED_M,
              "success": {"vanilla": summarize(v), "matched_dynamic": summarize(dyn), "matched_fixed_task_m": summarize(fix)},
              "paired": {"fixed_m_vs_dynamic": paired(fix, dyn), "dynamic_vs_vanilla": paired(dyn, v), "fixed_m_vs_vanilla": paired(fix, v)},
              "discordance_vs_vanilla": {
                  "dynamic_rescue": int(sum(x and not y for x, y in zip(dyn, v))),
                  "dynamic_harm": int(sum(y and not x for x, y in zip(dyn, v))),
                  "fixed_m_rescue": int(sum(x and not y for x, y in zip(fix, v))),
                  "fixed_m_harm": int(sum(y and not x for x, y in zip(fix, v))),
              },
              "by_task": {}, "cases": rows,
              "interpretation_limit": "One held-out OSMesa cohort and one fixed task-wise budget estimated from separate discovery init IDs. A neutral result with 40 cases does not establish equivalence or EGL generalization."}
    for tid in sorted({r["task_id"] for r in rows}):
        subset = [r for r in rows if r["task_id"] == tid]
        result["by_task"][str(tid)] = {"n": len(subset),
             "vanilla": summarize([r["vanilla_success"] for r in subset]),
             "matched_dynamic": summarize([r["matched_success"] for r in subset]),
             "matched_fixed_task_m": summarize([r["fixed_m_success"] for r in subset])}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("n", "integrity_pass", "success", "paired", "discordance_vs_vanilla")}, indent=2))
    if not result["integrity_pass"]:
        raise SystemExit("fixed-m audit failed; do not interpret outcomes")


if __name__ == "__main__":
    main()
