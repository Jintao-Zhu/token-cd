#!/usr/bin/env python3
"""Episode-level phase diagnostic for the frozen N=50 positive-G cohort.

This is descriptive only. It reports per-episode median matched-arm guidance G
within logged task phases, comparing Rescue and Harm episodes. It deliberately
does not treat individual steps as independent samples or derive a new gate.
"""
from __future__ import annotations

import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
ARTIFACT = REPO / "artifacts/libero_action_value_gate_egl_unseen_task_v1_20260925"
OUT_JSON = ARTIFACT / "N50_PHASE_GUIDANCE_DIAGNOSTIC.json"
OUT_MD = ARTIFACT / "N50_PHASE_GUIDANCE_DIAGNOSTIC.md"


def exact_auc_and_p(rescue: list[float], harm: list[float]) -> tuple[float | None, float | None]:
    if not rescue or not harm:
        return None, None
    wins = sum((r > h) + 0.5 * (r == h) for r in rescue for h in harm)
    auc = float(wins / (len(rescue) * len(harm)))
    values = np.asarray(rescue + harm, dtype=float)
    nr = len(rescue)
    observed = auc
    permuted = []
    for idx in itertools.combinations(range(len(values)), nr):
        mask = np.zeros(len(values), dtype=bool)
        mask[list(idx)] = True
        r = values[mask]
        h = values[~mask]
        w = sum((x > y) + 0.5 * (x == y) for x in r for y in h)
        permuted.append(w / (len(r) * len(h)))
    p = sum(abs(x - 0.5) >= abs(observed - 0.5) - 1e-12 for x in permuted) / len(permuted)
    return auc, float(p)


def main() -> None:
    cases: dict[str, dict] = {}
    for p in sorted((ARTIFACT / "episodes").glob("*/matched/episode.json")):
        matched = json.loads(p.read_text())
        vanilla = json.loads((p.parents[1] / "vanilla" / "episode.json").read_text())
        if matched["initial_state_sha256"] != vanilla["initial_state_sha256"]:
            raise RuntimeError(f"initial state mismatch in {p}")
        outcome = (
            "rescue" if matched["success"] and not vanilla["success"] else
            "harm" if vanilla["success"] and not matched["success"] else "concordant"
        )
        by_phase: dict[str, list[float]] = defaultdict(list)
        gate_on: Counter[str] = Counter()
        step_count: Counter[str] = Counter()
        for step in matched["trace"]:
            phase = str(step.get("task_phase", "unknown"))
            g = step.get("guidance_G_median")
            if g is not None and np.isfinite(float(g)):
                by_phase[phase].append(float(g))
            step_count[phase] += 1
            gate_on[phase] += int(bool(step.get("positive_G_gate_on", False)))
        cases[matched["case_id"]] = {
            "outcome": outcome,
            "task_id": matched["task_id"],
            "success_vanilla": bool(vanilla["success"]),
            "success_matched": bool(matched["success"]),
            "phase_median_G": {k: float(np.median(v)) for k, v in by_phase.items() if v},
            "phase_steps": dict(step_count),
            "phase_positive_gate_fraction": {
                k: float(gate_on[k] / n) for k, n in step_count.items() if n
            },
        }

    phases = sorted({phase for row in cases.values() for phase in row["phase_median_G"]})
    stats = {}
    for phase in phases:
        grouped = {
            outcome: [row["phase_median_G"][phase] for row in cases.values()
                      if row["outcome"] == outcome and phase in row["phase_median_G"]]
            for outcome in ("rescue", "harm", "concordant")
        }
        auc, p = exact_auc_and_p(grouped["rescue"], grouped["harm"])
        stats[phase] = {
            "n_rescue": len(grouped["rescue"]),
            "n_harm": len(grouped["harm"]),
            "n_concordant": len(grouped["concordant"]),
            "rescue_median_G_median": float(np.median(grouped["rescue"])) if grouped["rescue"] else None,
            "harm_median_G_median": float(np.median(grouped["harm"])) if grouped["harm"] else None,
            "rescue_values": grouped["rescue"],
            "harm_values": grouped["harm"],
            "rescue_higher_auc": auc,
            "two_sided_exact_permutation_p": p,
            "rescue_total_phase_steps": sum(row["phase_steps"].get(phase, 0) for row in cases.values() if row["outcome"] == "rescue"),
            "harm_total_phase_steps": sum(row["phase_steps"].get(phase, 0) for row in cases.values() if row["outcome"] == "harm"),
        }

    payload = {
        "protocol": "N50_PHASE_GUIDANCE_DIAGNOSTIC_V1",
        "cohort": str(ARTIFACT),
        "n_cases": len(cases),
        "outcome_counts": dict(Counter(row["outcome"] for row in cases.values())),
        "unit": "episode; per-phase median of per-step guidance_G_median",
        "phase_statistics": stats,
        "cases": cases,
        "limits": [
            "Only 4 Rescue and 5 Harm episodes are present in this N=50 cohort.",
            "Phase-specific counts are smaller because not all episodes reach every phase.",
            "Conditioning on a phase reached is post-treatment selection and is descriptive, not causal.",
            "Tasks are heterogeneous and the analysis is not task-stratified due to sparse discordant counts.",
            "Do not tune or validate a gate on this held-out cohort.",
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# N=50 phase-stratified guidance diagnostic",
        "",
        "Exploratory episode-level analysis of the frozen, held-out positive-G cohort. The unit is the episode; each value is the median logged `guidance_G_median` within a task phase.",
        "",
        f"Cases: {len(cases)}; outcomes: {payload['outcome_counts']}",
        "",
        "| Phase | Rescue n / median G | Harm n / median G | Rescue-higher AUC | Exact two-sided p |",
        "|---|---:|---:|---:|---:|",
    ]
    for phase, s in stats.items():
        auc = "—" if s["rescue_higher_auc"] is None else f"{s['rescue_higher_auc']:.3f}"
        p = "—" if s["two_sided_exact_permutation_p"] is None else f"{s['two_sided_exact_permutation_p']:.3f}"
        rv = "—" if s["rescue_median_G_median"] is None else f"{s['rescue_median_G_median']:.3f}"
        hv = "—" if s["harm_median_G_median"] is None else f"{s['harm_median_G_median']:.3f}"
        lines.append(f"| {phase} | {s['n_rescue']} / {rv} | {s['n_harm']} / {hv} | {auc} | {p} |")
    lines += [
        "",
        "## Reading",
        "",
        "The small-sample point estimates are phase-dependent: approach medians trend higher in Rescue than Harm, while grasp and transport point in the opposite direction. This is compatible with a phase interaction and may explain why a single all-step `G > 0` rule was neutral, but the counts are too small to establish that mechanism. It is not evidence that a phase-gated policy will improve success.",
        "",
        "## Hypothesis update",
        "",
        "**Hypothesis:** the action-value proxy may have different physical meaning across interaction phases; aggregating its sign across approach, grasp, transport, and placement can cancel useful and harmful choices.",
        "",
        "**Prediction:** on a fresh, task-balanced cohort, a predeclared phase-by-G interaction should reproduce on episode-level outcomes; then an event-conditioned intervention policy should improve paired success without losing most approach/grasp Rescues.",
        "",
        "**Evidence for:** approach Rescue/Harm episode-median G points in the predicted direction in this held-out N=50; grasp/transport medians point oppositely.",
        "",
        "**Evidence against / limits:** only 4 Rescue and 5 Harm total; phase reached is post-treatment selected; mixed tasks; no causal intervention tested. The earlier RT time-window ablation did not support a generic early-vs-late account.",
        "",
        "**Confidence:** low; hypothesis-generating only.",
        "",
        "**Decision:** REFINE. Do not change the method or tune a gate on these data. If pursued, pre-register a fresh phase-stratified validation and then a minimal event-conditioned paired pilot.",
        "",
        "## Integrity and scope",
        "",
        "All episode pairings were checked for identical initial-state hashes. The script uses episode-level summaries rather than treating steps as independent. Phase-specific AUC/permutation values are descriptive because of the very small discordant sample and survivor selection.",
        "",
        "Machine-readable records: [N50_PHASE_GUIDANCE_DIAGNOSTIC.json](N50_PHASE_GUIDANCE_DIAGNOSTIC.json). Reproducer: [analyze_n50_phase_guidance.py](../../../../research/semantic_token_cd/analyze_n50_phase_guidance.py).",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(json.dumps({"n_cases": len(cases), "outcomes": payload["outcome_counts"], "phase_statistics": stats}, indent=2))


if __name__ == "__main__":
    main()
