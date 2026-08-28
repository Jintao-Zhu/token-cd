"""Audit and summarize the corrected cascade secondary output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    rows = [json.loads(line) for line in (artifact / "cascade_effects_v2.jsonl").read_text().splitlines() if line]
    keys = {(row["snapshot_id"], row["visual_token_idx"]) for row in rows}
    if len(rows) != 2400 or len(keys) != 2400 or not all(row["all_finite"] for row in rows):
        raise RuntimeError(f"Cascade secondary integrity failure: rows={len(rows)}, keys={len(keys)}")
    teacher = np.asarray([row["teacher_forced_argmax_action_l2"] for row in rows])
    free = np.asarray([row["free_running_action_l2"] for row in rows])
    ratios = np.asarray([row["cascade_amplification"] for row in rows])
    positive_teacher = teacher > 0
    mismatch_states = {
        row["snapshot_id"] for row in rows if row["clean_teacher_vs_generated_mismatch"]
    }
    summary = {
        "status": "PASS",
        "rows": len(rows),
        "unique_token_interventions": len(keys),
        "nonfinite": 0,
        "clean_teacher_vs_cached_generation_mismatch_states": len(mismatch_states),
        "clean_teacher_vs_cached_generation_mismatch_state_fraction": len(mismatch_states) / 150,
        "teacher_forced_argmax_nonzero_fraction": float(np.mean(teacher > 0)),
        "free_running_nonzero_fraction": float(np.mean(free > 0)),
        "teacher_vs_free_effect_spearman": float(spearmanr(teacher, free).statistic),
        "cascade_amplification_median_all": float(np.median(ratios)),
        "cascade_amplification_median_teacher_nonzero": float(np.median(ratios[positive_teacher])) if positive_teacher.any() else None,
        "cascade_amplification_gt_1_fraction_teacher_nonzero": float(np.mean(ratios[positive_teacher] > 1)) if positive_teacher.any() else None,
        "interpretation": "Secondary descriptive result; primary decision is unchanged.",
    }
    (artifact / "cascade_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
