"""Re-score the saved 60-state P/Q audit without any new model forward."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from research.semantic_token_cd.prompt_attn_shr_policy import stable_top_m
from research.semantic_token_cd.target_positive_boost_protocol import ARTIFACT, SOURCE_OFFLINE, TASKS, atomic_json, write_lock


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


def main() -> None:
    write_lock(); output = ARTIFACT / "offline"; output.mkdir(parents=True, exist_ok=True)
    source_files = sorted(SOURCE_OFFLINE.glob("*/seed_*_step_*.json"))
    if len(source_files) != 60: raise RuntimeError(f"source audit incomplete: {len(source_files)}/60")
    rows = []
    for source_json in source_files:
        old = json.loads(source_json.read_text()); source_npz = np.load(source_json.with_suffix(".npz"), allow_pickle=False)
        task, m = old["task"], int(old["branches"]["correct"]["m"])
        correct = old["branches"]["correct"]["selected"]
        p = np.asarray(source_npz["target_diff_p"], dtype=np.float64)
        q = np.asarray(source_npz["target_diff_q"], dtype=np.float64)
        p_alt = np.asarray(source_npz["target_diff_alt_p"], dtype=np.float64)
        q_alt = np.asarray(source_npz["target_diff_alt_q"], dtype=np.float64)
        d = p - q; d_alt = p_alt - q_alt
        scores = {
            "positive_boost_0p5": p + .5 * np.maximum(d, 0),
            "positive_boost_1p0": p + np.maximum(d, 0),
            "reverse_boost_1p0": p + np.maximum(-d, 0),
            "positive_boost_alt_1p0": p_alt + np.maximum(d_alt, 0),
        }
        selected = {name: stable_top_m(score, m) for name, score in scores.items()}
        eta0 = stable_top_m(p, m)
        branches = {}
        for name in ("positive_boost_0p5", "positive_boost_1p0", "reverse_boost_1p0"):
            entered = sorted(set(selected[name]) - set(correct)); direction = d if name.startswith("positive") else -d
            branches[name] = {"selected": selected[name], "entered": entered, "exited": sorted(set(correct)-set(selected[name])),
                "replacement_count": len(entered), "jaccard_vs_correct": jaccard(selected[name], correct),
                "entered_correct_direction": all(direction[index] > 0 for index in entered),
                "mean_entered_direction": float(np.mean(direction[entered])) if entered else 0.0}
        row = {"task": task, "seed": old["seed"], "control_step": old["control_step"], "m": m,
               "instruction": old["instruction"], "generic_primary": old["generic_primary"],
               "generic_alternate": old["generic_alternate"], "eta0_exact_correct": eta0 == correct,
               "positive_boost_generic_paraphrase_jaccard": jaccard(selected["positive_boost_1p0"], selected["positive_boost_alt_1p0"]),
               "branches": branches}
        destination = output / task / source_json.name; destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(destination, row); np.savez_compressed(destination.with_suffix(".npz"), p=p, q=q, difference=d,
            **{f"score_{name}": value for name, value in scores.items()},
            **{f"mask_{name}": np.isin(np.arange(256), ids).astype(np.uint8) for name, ids in selected.items()})
        rows.append(row)
    identity = all(x["eta0_exact_correct"] for x in rows)
    direction = all(branch["entered_correct_direction"] for x in rows for branch in x["branches"].values())
    wording = float(np.mean([x["positive_boost_generic_paraphrase_jaccard"] for x in rows]))
    changed = {arm: float(np.mean([x["branches"][arm]["replacement_count"] for x in rows]))
               for arm in ("positive_boost_0p5", "positive_boost_1p0", "reverse_boost_1p0")}
    passed = identity and direction and wording >= .2 and changed["positive_boost_1p0"] > 0
    by_task = {}
    for task in TASKS:
        subset = [x for x in rows if x["task"] == task]
        by_task[task] = {"states": len(subset), "mean_m": float(np.mean([x["m"] for x in subset])),
            "wording_jaccard": float(np.mean([x["positive_boost_generic_paraphrase_jaccard"] for x in subset])),
            "mean_replacements": {arm: float(np.mean([x["branches"][arm]["replacement_count"] for x in subset])) for arm in changed}}
    result = {"protocol": "TARGET_POSITIVE_BOOST_SHR_V1", "states": 60, "eta0_exact_correct": identity,
              "all_entered_tokens_have_intended_difference_sign": direction, "mean_generic_wording_jaccard": wording,
              "mean_replacements": changed, "tasks": by_task, "technical_pass": passed}
    atomic_json(ARTIFACT / "OFFLINE_RESULTS.json", result)
    lines = ["# Target Positive-Boost Offline Gate", "", f"**{'PASS' if passed else 'STOP'}**", "",
             f"- eta=0 exact Correct: {'PASS' if identity else 'FAIL'}.",
             f"- Every entered token has the intended difference sign: {'PASS' if direction else 'FAIL'}.",
             f"- Primary/alternate generic wording Jaccard: {wording:.3f}.", "",
             "| Task | mean m | Boost .5 replacements | Boost 1.0 replacements | Reverse replacements | wording Jaccard |",
             "|---|---:|---:|---:|---:|---:|"]
    for task, value in by_task.items():
        r = value["mean_replacements"]; lines.append(f"| {task.removeprefix('google_robot_')} | {value['mean_m']:.1f} | "
            f"{r['positive_boost_0p5']:.1f} | {r['positive_boost_1p0']:.1f} | {r['reverse_boost_1p0']:.1f} | {value['wording_jaccard']:.3f} |")
    (ARTIFACT / "OFFLINE_REPORT.md").write_text("\n".join(lines) + "\n")
    (ARTIFACT / ("OFFLINE_PASS" if passed else "OFFLINE_STOP")).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
