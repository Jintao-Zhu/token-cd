#!/usr/bin/env python3
"""Auditable paired analysis for the preregistered positive-G gate pilot."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ARMS = ("vanilla", "matched", "positive_gated")
ARRAY_KEYS = (
    "positive_action_logits",
    "negative_action_logits",
    "guided_action_logits",
    "l11_attention_scores",
)


def exact_mcnemar_two_sided(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def paired_bootstrap_ci(differences: np.ndarray, draws: int, seed: int):
    if len(differences) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        sample = rng.choice(differences, size=len(differences), replace=True)
        estimates[i] = np.mean(sample)
    return [float(x) for x in np.quantile(estimates, [0.025, 0.975])]


def audit_case(root: Path, case_id: str, pair: dict, errors: list[str]):
    episodes = {}
    for arm in ARMS:
        ep_path = root / "episodes" / case_id / arm / "episode.json"
        arrays_path = ep_path.parent / "step_arrays.npz"
        if not ep_path.is_file() or not arrays_path.is_file():
            errors.append(f"{case_id}: missing {arm} episode or arrays")
            continue
        ep = json.loads(ep_path.read_text())
        episodes[arm] = ep
        arrays = np.load(arrays_path)
        for key in ARRAY_KEYS:
            if key not in arrays:
                errors.append(f"{case_id}/{arm}: missing array {key}")
                continue
            values = arrays[key]
            if values.ndim < 2 or not np.isfinite(values).all():
                errors.append(f"{case_id}/{arm}: invalid/nonfinite {key}")
        if int(ep.get("steps", -1)) != len(ep.get("trace", [])):
            errors.append(f"{case_id}/{arm}: trace length disagrees with steps")
        if "positive_action_logits" in arrays and arrays["positive_action_logits"].shape[0] != len(ep.get("trace", [])):
            errors.append(f"{case_id}/{arm}: logits/trace length mismatch")
        if ep.get("initial_state_sha256") != pair.get("initial_state_sha256"):
            errors.append(f"{case_id}/{arm}: initial simulator-state hash mismatch")
        trace = ep.get("trace", [])
        first_rgb = trace[0].get("rgb_sha256") if trace else None
        if first_rgb != pair.get("initial_rgb_sha256"):
            errors.append(f"{case_id}/{arm}: initial RGB hash mismatch")
        if bool(ep.get("success")) != bool(pair.get({
            "vanilla": "vanilla_success",
            "matched": "matched_success",
            "positive_gated": "positive_gated_success",
        }[arm])):
            errors.append(f"{case_id}/{arm}: episode success disagrees with pair")
        if arm == "positive_gated":
            for step, row in enumerate(trace):
                should_apply = float(row.get("guidance_G_median", float("nan"))) > 0.0
                if not math.isfinite(float(row.get("guidance_G_median", float("nan")))):
                    errors.append(f"{case_id}/{arm}/{step}: nonfinite gate score")
                    continue
                if bool(row.get("positive_G_gate_on")) != should_apply:
                    errors.append(f"{case_id}/{arm}/{step}: gate sign rule mismatch")
                if bool(row.get("guidance_applied")) != should_apply:
                    errors.append(f"{case_id}/{arm}/{step}: applied flag mismatch")
                executed = np.asarray(row.get("executed_raw_action", []), dtype=float)
                expected = np.asarray(row.get(
                    "guided_action" if should_apply else "clean_action_from_positive_logits", []
                ), dtype=float)
                if executed.shape != expected.shape or not np.allclose(executed, expected, atol=1e-7, rtol=0):
                    errors.append(f"{case_id}/{arm}/{step}: executed action disagrees with gate")
    if set(episodes) == set(ARMS):
        states = {ep["initial_state_sha256"] for ep in episodes.values()}
        rgbs = {ep["trace"][0]["rgb_sha256"] for ep in episodes.values() if ep.get("trace")}
        if len(states) != 1 or len(rgbs) != 1:
            errors.append(f"{case_id}: three-arm initial state/RGB hashes differ")
    return episodes


def outcome(candidate: bool, baseline: bool) -> str:
    if candidate and not baseline:
        return "rescue"
    if baseline and not candidate:
        return "harm"
    return "concordant"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=50000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260925)
    args = parser.parse_args()
    root = args.artifact.resolve()
    manifest = json.loads((root / "cases_manifest.json").read_text())
    cases = manifest["cases"]
    expected_ids = [c["case_id"] for c in cases]
    errors = []
    if len(expected_ids) != 30 or len(set(expected_ids)) != len(expected_ids):
        errors.append("manifest must contain 30 unique case IDs")

    rows = []
    audited_arm_count = 0
    for case in cases:
        case_id = case["case_id"]
        pair_path = root / "pairs" / f"{case_id}.json"
        if not pair_path.is_file():
            continue
        pair = json.loads(pair_path.read_text())
        if pair.get("case_id") != case_id:
            errors.append(f"{case_id}: pair case ID mismatch")
        if not pair.get("paired_initial_state_match"):
            errors.append(f"{case_id}: pair state/RGB audit is not PASS")
        arms = audit_case(root, case_id, pair, errors)
        audited_arm_count += len(arms)
        if set(arms) != set(ARMS):
            continue
        expected_outcome = outcome(bool(pair["matched_success"]), bool(pair["vanilla_success"]))
        expected_gated = outcome(bool(pair["positive_gated_success"]), bool(pair["vanilla_success"]))
        if pair.get("outcome") != expected_outcome:
            errors.append(f"{case_id}: Matched-vs-Vanilla outcome mismatch")
        if pair.get("positive_gated_outcome_vs_vanilla") != expected_gated:
            errors.append(f"{case_id}: gated-vs-Vanilla outcome mismatch")
        rows.append(pair)

    extra_pair_ids = {p.stem for p in (root / "pairs").glob("*.json")} - set(expected_ids)
    if extra_pair_ids:
        errors.append(f"unexpected pair files: {sorted(extra_pair_ids)}")
    n = len(rows)
    wins = sum(bool(r["positive_gated_success"]) and not bool(r["matched_success"]) for r in rows)
    losses = sum(bool(r["matched_success"]) and not bool(r["positive_gated_success"]) for r in rows)
    differences = np.asarray([int(r["positive_gated_success"]) - int(r["matched_success"]) for r in rows], dtype=np.int8)
    matched_outcomes = Counter(r["outcome"] for r in rows)
    gated_outcomes = Counter(r["positive_gated_outcome_vs_vanilla"] for r in rows)
    harm_fewer = gated_outcomes["harm"] < matched_outcomes["harm"]
    rescue_retention = None
    rescue_retained = None
    if matched_outcomes["rescue"] > 0:
        rescue_retained = gated_outcomes["rescue"] / matched_outcomes["rescue"]
        rescue_retention = rescue_retained >= 0.75
    else:
        rescue_retention = True
    support = (
        n == 30 and wins - losses >= 2 and harm_fewer and rescue_retention
    )
    technical_pass = (
        n == 30 and audited_arm_count == 90 and not errors
    )
    tasks = defaultdict(list)
    for row in rows:
        tasks[str(row["task_id"])].append(row)
    per_task = {}
    for task_id, task_rows in sorted(tasks.items(), key=lambda item: int(item[0])):
        per_task[task_id] = {
            "n": len(task_rows),
            "vanilla_successes": sum(bool(x["vanilla_success"]) for x in task_rows),
            "matched_successes": sum(bool(x["matched_success"]) for x in task_rows),
            "gated_successes": sum(bool(x["positive_gated_success"]) for x in task_rows),
            "matched_vs_vanilla": dict(Counter(x["outcome"] for x in task_rows)),
            "gated_vs_vanilla": dict(Counter(x["positive_gated_outcome_vs_vanilla"] for x in task_rows)),
            "gate_vs_matched_wins": sum(bool(x["positive_gated_success"]) and not bool(x["matched_success"]) for x in task_rows),
            "gate_vs_matched_losses": sum(bool(x["matched_success"]) and not bool(x["positive_gated_success"]) for x in task_rows),
        }
    result = {
        "protocol_id": "LIBERO90_ACTION_VALUE_GPOSITIVE_GATE_PILOT_V1",
        "artifact": str(root),
        "n_complete_pairs": n,
        "n_audited_arm_traces": audited_arm_count,
        "technical_audit": {"status": "PASS" if technical_pass else "INCOMPLETE_OR_FAIL", "errors": errors},
        "vanilla_successes": sum(bool(r["vanilla_success"]) for r in rows),
        "matched_successes": sum(bool(r["matched_success"]) for r in rows),
        "gated_successes": sum(bool(r["positive_gated_success"]) for r in rows),
        "matched_vs_vanilla": dict(matched_outcomes),
        "gated_vs_vanilla": dict(gated_outcomes),
        "primary_gate_vs_matched": {
            "wins": wins,
            "losses": losses,
            "net_wins": wins - losses,
            "paired_risk_difference": float(np.mean(differences)) if n else None,
            "exact_mcnemar_two_sided_p": exact_mcnemar_two_sided(wins, losses),
            "episode_pair_bootstrap_95_percentile_ci": paired_bootstrap_ci(differences, args.bootstrap_draws, args.bootstrap_seed),
            "bootstrap_draws": args.bootstrap_draws,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "preregistered_support_rule": {
            "net_gate_wins_at_least_2": wins - losses >= 2,
            "fewer_harms_than_matched": harm_fewer,
            "rescue_retention_fraction": rescue_retained,
            "retains_at_least_75_percent_rescues": rescue_retention,
            "overall_status": "SUPPORT_PILOT_CONTINUE_TO_CONFIRMATORY" if support and technical_pass else ("REJECT_SIMPLE_SIGN_GATE" if n == 30 and technical_pass else "INCOMPLETE"),
        },
        "per_task": per_task,
        "case_rows": rows,
        "interpretation_limit": "A supportive 30-pair pilot is exploratory causal evidence for this frozen OSMesa-LIBERO setting only; confirmatory validation and benchmark-transfer isolation remain required.",
    }
    out = root / ("GATE_PILOT_FINAL_ANALYSIS.json" if n == 30 else "GATE_PILOT_INTERIM_ANALYSIS.json")
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in (
        "n_complete_pairs", "n_audited_arm_traces", "technical_audit",
        "matched_vs_vanilla", "gated_vs_vanilla", "primary_gate_vs_matched",
        "preregistered_support_rule",
    )}, indent=2))


if __name__ == "__main__":
    main()
