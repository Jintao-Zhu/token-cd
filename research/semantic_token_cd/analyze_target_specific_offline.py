"""Audit the 60-state generic-frame experiment and issue the closed-loop gate."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from research.semantic_token_cd.target_specific_protocol import ARTIFACT, TASKS, atomic_json


def main() -> None:
    files = sorted((ARTIFACT / "offline").glob("*/seed_*_step_*.json"))
    if len(files) != 60:
        raise RuntimeError(f"offline audit incomplete: {len(files)}/60")
    rows = [json.loads(path.read_text()) for path in files]
    old_root = Path("/home/leju-suzhou/zjt_ws/token-cd/artifacts/prompt_attn_instr_swap_v1/runs/offline")
    equivalent = []
    for row in rows:
        old = old_root / row["task"] / f"seed_{int(row['seed']):03d}" / f"step_{int(row['control_step']):04d}.json"
        if not old.exists():
            equivalent.append(False); continue
        previous = json.loads(old.read_text())["branches"]["correct"]
        current = row["branches"]["correct"]
        equivalent.append(previous["selected_token_ids"] == current["selected"] and
                          np.allclose(previous["clean_action"], current["clean_action"], atol=1e-6) and
                          np.allclose(previous["guided_action"], current["guided_action"], atol=1e-6))
    by_task = {}
    for task in TASKS:
        subset = [x for x in rows if x["task"] == task]
        by_task[task] = {
            "states": len(subset),
            "mean_m": float(np.mean([x["branches"]["correct"]["m"] for x in subset])),
            "mean_generic_paraphrase_mask_jaccard": float(np.mean([x["generic_paraphrase_mask_jaccard"] for x in subset])),
            "target_diff_mean_jaccard_vs_correct": float(np.mean([x["branches"]["target_diff"]["jaccard_vs_correct"] for x in subset])),
            "target_boost_mean_jaccard_vs_correct": float(np.mean([x["branches"]["target_boost"]["jaccard_vs_correct"] for x in subset])),
            "reverse_mean_jaccard_vs_correct": float(np.mean([x["branches"]["reverse_diff"]["jaccard_vs_correct"] for x in subset])),
            "target_diff_min_selected_positive_fraction": float(min(x["branches"]["target_diff"]["selected_positive_fraction"] for x in subset)),
            "mean_target_diff_selected_correct_rank": float(np.mean([x["branches"]["target_diff"]["mean_selected_correct_rank"] for x in subset])),
        }
    alt_mean = float(np.mean([x["generic_paraphrase_mask_jaccard"] for x in rows]))
    identity = all(x["clean_equal"] and x["m_equal"] for x in rows) and all(equivalent)
    positive = min(x["branches"]["target_diff"]["selected_positive_fraction"] for x in rows) >= .999
    finite_mass = all(x["branches"]["target_diff"]["visual_attention_mass_correct"] > 0 and
                      x["branches"]["target_diff"]["visual_attention_mass_generic"] > 0 for x in rows)
    # This is only a gross wording-instability stop, not a selector-quality claim.
    wording_stable = alt_mean >= .20
    passed = identity and positive and finite_mass and wording_stable
    result = {"protocol": "TARGET_SPECIFIC_ATTENTION_SHR_V1", "states": 60,
              "identity_and_historical_correct_equivalence": identity,
              "all_target_diff_selected_scores_positive": positive,
              "attention_mass_finite_positive": finite_mass,
              "mean_primary_vs_alternate_generic_mask_jaccard": alt_mean,
              "wording_stability_floor": .20, "technical_pass": passed, "tasks": by_task,
              "boundary": "Offline mask/action plausibility gate; no success-rate claim."}
    atomic_json(ARTIFACT / "OFFLINE_RESULTS.json", result)
    report = ["# Target-specific Attention Offline Gate", "", f"**{'PASS' if passed else 'STOP'}**", "",
              f"- 60/60 same-state cases complete.",
              f"- Historical Correct mask/action equivalence: {'PASS' if identity else 'FAIL'}.",
              f"- Target-Diff selected-score positivity: {'PASS' if positive else 'FAIL'}.",
              f"- Primary/alternate generic wording mask Jaccard: {alt_mean:.3f}.", "",
              "| Task | mean m | Diff∩Correct Jaccard | Boost∩Correct | Reverse∩Correct | Generic wording Jaccard |",
              "|---|---:|---:|---:|---:|---:|"]
    for task, value in by_task.items():
        report.append(f"| {task.removeprefix('google_robot_')} | {value['mean_m']:.1f} | "
                      f"{value['target_diff_mean_jaccard_vs_correct']:.3f} | {value['target_boost_mean_jaccard_vs_correct']:.3f} | "
                      f"{value['reverse_mean_jaccard_vs_correct']:.3f} | {value['mean_generic_paraphrase_mask_jaccard']:.3f} |")
    (ARTIFACT / "OFFLINE_REPORT.md").write_text("\n".join(report) + "\n")
    marker = ARTIFACT / ("OFFLINE_PASS" if passed else "OFFLINE_STOP")
    marker.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"technical_pass": passed, "states": 60, "wording_jaccard": alt_mean}))


if __name__ == "__main__": main()
