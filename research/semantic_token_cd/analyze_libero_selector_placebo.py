#!/usr/bin/env python3
"""Audit and compare the 40-case L11 spatial-placebo rollout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def read_json(path: Path):
    return json.loads(path.read_text())


def summarize(y):
    return {"successes": int(sum(y)), "n": len(y), "rate": float(np.mean(y)) if y else None}


def paired(a, b):
    wins = sum(bool(x) and not bool(y) for x, y in zip(a, b))
    losses = sum(not bool(x) and bool(y) for x, y in zip(a, b))
    ties = len(a) - wins - losses
    p = float(binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else 1.0
    rng = np.random.default_rng(20260925)
    indexes = rng.integers(0, len(a), size=(20000, len(a)))
    deltas = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    boot = deltas[indexes].mean(axis=1)
    return {
        "wins_a": int(wins), "losses_a": int(losses), "ties": int(ties),
        "risk_difference_a_minus_b": float(deltas.mean()),
        "bootstrap_95_ci": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "exact_mcnemar_p_two_sided": p,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.baseline / "cases_manifest.json")
    cases = manifest["cases"]
    rows = []
    integrity_errors = []
    for case in cases:
        cid = case["case_id"]
        base_pair = read_json(args.baseline / "pairs" / f"{cid}.json")
        ctrl_pair = read_json(args.treatment / "pairs" / f"{cid}.json")
        base_ep = args.baseline / "episodes" / cid / "matched"
        rot_ep = args.treatment / "episodes" / cid / "matched_rot180"
        base_json = read_json(base_ep / "episode.json")
        rot_json = read_json(rot_ep / "episode.json")
        base_trace0, rot_trace0 = base_json["trace"][0], rot_json["trace"][0]
        base_arrays = np.load(base_ep / "step_arrays.npz")
        rot_arrays = np.load(rot_ep / "step_arrays.npz")
        expected_ids = sorted(255 - int(token) for token in base_trace0["selected_token_ids"])
        mapped_ids = sorted(int(token) for token in rot_trace0["selected_token_ids"])
        checks = {
            "task_init_identity": base_json["case_id"] == cid == rot_json["case_id"],
            "initial_state_hash": base_pair["initial_state_sha256"] == ctrl_pair["initial_state_sha256"] == rot_json["initial_state_sha256"],
            "first_rgb_hash": base_trace0["rgb_sha256"] == rot_trace0["rgb_sha256"],
            "first_attention_hash": base_trace0["attention_sha256"] == rot_trace0["attention_sha256"],
            "first_clean_logits_bitwise": bool(np.array_equal(base_arrays["positive_action_logits"][0], rot_arrays["positive_action_logits"][0])),
            "same_first_m": int(base_trace0["matched_m"]) == int(rot_trace0["matched_m"]),
            "rotated_ids_match_graph_mapping": expected_ids == mapped_ids,
            "selector_tag": rot_trace0.get("selector_transform") == "rot180",
            "finite_arrays": all(np.isfinite(rot_arrays[k]).all() for k in rot_arrays.files),
        }
        for key, okay in checks.items():
            if not okay:
                integrity_errors.append({"case_id": cid, "check": key})
        rows.append({
            "case_id": cid,
            "task_id": int(case["task_id"]),
            "init_state_id": int(case["init_state_id"]),
            "vanilla_success": bool(base_pair["vanilla_success"]),
            "matched_success": bool(base_pair["matched_success"]),
            "rot180_success": bool(ctrl_pair["rot180_success"]),
            "checks": checks,
            "canonical_steps": int(base_json["steps"]),
            "rot180_steps": int(rot_json["steps"]),
        })
    vanilla = [r["vanilla_success"] for r in rows]
    matched = [r["matched_success"] for r in rows]
    rot = [r["rot180_success"] for r in rows]
    result = {
        "protocol_id": "LIBERO90_L11_MATCHED_SELECTOR_ROT180_V1",
        "n": len(rows),
        "integrity_pass": len(rows) == 40 and not integrity_errors,
        "integrity_errors": integrity_errors,
        "success": {"vanilla": summarize(vanilla), "matched_l11": summarize(matched), "matched_l11_rot180": summarize(rot)},
        "paired": {
            "rot180_vs_matched_l11": paired(rot, matched),
            "matched_l11_vs_vanilla": paired(matched, vanilla),
            "rot180_vs_vanilla": paired(rot, vanilla),
        },
        "discordance_vs_vanilla": {
            "matched_l11_rescue": int(sum(m and not v for m, v in zip(matched, vanilla))),
            "matched_l11_harm": int(sum(v and not m for m, v in zip(matched, vanilla))),
            "rot180_rescue": int(sum(r and not v for r, v in zip(rot, vanilla))),
            "rot180_harm": int(sum(v and not r for r, v in zip(rot, vanilla))),
        },
        "by_task": {},
        "cases": rows,
        "interpretation_limit": "One fixed spatial placebo on one task-finetuned checkpoint and OSMesa condition. The 40 episodes can localize selector spatial correspondence within this cohort, but do not establish EGL or cross-benchmark generalization.",
    }
    for task_id in sorted({r["task_id"] for r in rows}):
        subset = [r for r in rows if r["task_id"] == task_id]
        result["by_task"][str(task_id)] = {
            "n": len(subset),
            "vanilla": summarize([r["vanilla_success"] for r in subset]),
            "matched_l11": summarize([r["matched_success"] for r in subset]),
            "matched_l11_rot180": summarize([r["rot180_success"] for r in subset]),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("n", "integrity_pass", "success", "paired", "discordance_vs_vanilla")}, indent=2))
    if not result["integrity_pass"]:
        raise SystemExit("selector placebo audit failed; do not interpret outcomes")


if __name__ == "__main__":
    main()
