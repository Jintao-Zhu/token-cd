"""Final paired analysis for target-specific attention SHR."""
from __future__ import annotations

import glob
import json
import argparse
from collections import defaultdict

from scipy.stats import binomtest

from research.semantic_token_cd.target_specific_protocol import ARTIFACT as DIFFERENCE_ARTIFACT, NEW_ARMS as DIFFERENCE_ARMS, TASKS, atomic_json
from research.semantic_token_cd.target_positive_boost_protocol import ARTIFACT as BOOST_ARTIFACT, NEW_ARMS as BOOST_ARMS

BASE = {
    "google_robot_open_drawer": "artifacts/prompt_attn_layer_selection_v1/closed_loop/episodes/google_robot_open_drawer/prompt_single",
    "google_robot_close_drawer": "artifacts/prompt_attn_layer_selection_v1/closed_loop_remaining6/episodes/google_robot_close_drawer/prompt_single",
    "google_robot_pick_coke_can": "artifacts/prompt_attn_layer_selection_v1/closed_loop/episodes/google_robot_pick_coke_can/prompt_single",
    "google_robot_move_near": "artifacts/prompt_attn_layer_selection_v1/closed_loop/episodes/google_robot_move_near/prompt_single",
}


def load(path, root_success=False):
    output = {}
    for filename in glob.glob(path + "/episode_*_summary.json"):
        data = json.load(open(filename)); success = data["success"] if root_success else data["result"]["success"]
        output[int(data["seed"])] = (int(success), data)
    return output


def paired(a, b):
    rescue = sum(not a[s][0] and b[s][0] for s in a); harm = sum(a[s][0] and not b[s][0] for s in a)
    return {"rescue": rescue, "harm": harm, "net": rescue - harm,
            "mcnemar_exact_p": float(binomtest(min(rescue, harm), rescue + harm, .5).pvalue) if rescue + harm else 1.0}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--variant", choices=("difference", "positive_boost"), default="difference")
    args = parser.parse_args(); boost = args.variant == "positive_boost"
    artifact = BOOST_ARTIFACT if boost else DIFFERENCE_ARTIFACT; arms = BOOST_ARMS if boost else DIFFERENCE_ARMS
    protocol = "TARGET_POSITIVE_BOOST_SHR_V1" if boost else "TARGET_SPECIFIC_ATTENTION_SHR_V1"
    results = {"protocol": protocol, "complete": True, "tasks": {}}
    pooled = defaultdict(dict)
    for task in TASKS:
        correct = load(BASE[task]); arm_data = {arm: load(str(artifact / "closed_loop/episodes" / task / arm), True) for arm in arms}
        if len(correct) < 100 or any(len(value) != 100 for value in arm_data.values()):
            raise RuntimeError(f"incomplete task: {task}, correct={len(correct)}, new={[len(x) for x in arm_data.values()]}")
        correct = {s: correct[s] for s in range(100)}
        identity = all(all(arm_data[arm][s][1][key] == correct[s][1][key] for key in
                           ("canonical_snapshot_sha256", "initial_state_sha256", "initial_rgb_sha256"))
                       for arm in arms for s in range(100))
        record = {"n": 100, "correct_success": sum(x[0] for x in correct.values()), "identity_pass": identity, "arms": {}}
        for arm, data in arm_data.items():
            data = {s: data[s] for s in range(100)}; record["arms"][arm] = {"success": sum(x[0] for x in data.values()),
                "vs_correct": paired(correct, data)}
            for seed in range(100): pooled["correct"][f"{task}/{seed}"] = correct[seed]; pooled[arm][f"{task}/{seed}"] = data[seed]
        if boost:
            record["positive_0p5_vs_reverse"] = paired(arm_data["reverse_boost_1p0"], arm_data["positive_boost_0p5"])
            record["positive_1p0_vs_reverse"] = paired(arm_data["reverse_boost_1p0"], arm_data["positive_boost_1p0"])
        else:
            record["target_diff_vs_reverse"] = paired(arm_data["reverse_diff"], arm_data["target_diff"])
            record["target_boost_vs_reverse"] = paired(arm_data["reverse_diff"], arm_data["target_boost"])
        results["tasks"][task] = record
    results["overall"] = {"n": 400, "correct_success": sum(x[0] for x in pooled["correct"].values()), "arms": {}}
    for arm in arms:
        results["overall"]["arms"][arm] = {"success": sum(x[0] for x in pooled[arm].values()),
            "vs_correct": paired(pooled["correct"], pooled[arm])}
    if boost:
        results["overall"]["positive_0p5_vs_reverse"] = paired(pooled["reverse_boost_1p0"], pooled["positive_boost_0p5"])
        results["overall"]["positive_1p0_vs_reverse"] = paired(pooled["reverse_boost_1p0"], pooled["positive_boost_1p0"])
    else:
        results["overall"]["target_diff_vs_reverse"] = paired(pooled["reverse_diff"], pooled["target_diff"])
        results["overall"]["target_boost_vs_reverse"] = paired(pooled["reverse_diff"], pooled["target_boost"])
    results["identity_pass"] = all(x["identity_pass"] for x in results["tasks"].values())
    atomic_json(artifact / "FINAL_RESULTS.json", results)
    labels = ({"positive_boost_0p5": "PositiveBoost-0.5", "positive_boost_1p0": "PositiveBoost-1.0", "reverse_boost_1p0": "ReverseBoost"}
              if boost else {"target_diff": "Target-Diff", "target_boost": "Target-Boost", "reverse_diff": "Reverse"})
    lines = ["# Target Positive-Boost SHR" if boost else "# Target-specific Attention SHR", "",
             f"Snapshot/hash identity: **{'PASS' if results['identity_pass'] else 'FAIL'}**", "",
             "| Task | Correct | " + " | ".join(labels[arm] for arm in arms) + " |",
             "|---|---:|" + "---:|" * len(arms)]
    for task, value in results["tasks"].items():
        lines.append(f"| {task.removeprefix('google_robot_')} | {value['correct_success']}/100 | " +
                     " | ".join(f"{value['arms'][arm]['success']}/100" for arm in arms) + " |")
    overall = results["overall"]
    lines.append(f"| Overall | {overall['correct_success']}/400 | " +
                 " | ".join(f"{overall['arms'][arm]['success']}/400" for arm in arms) + " |")
    lines += ["", "## Paired against Correct", "", "| Arm | Rescue | Harm | Net | exact p |", "|---|---:|---:|---:|---:|"]
    for arm in arms:
        value = overall["arms"][arm]["vs_correct"]
        lines.append(f"| {labels[arm]} | {value['rescue']} | {value['harm']} | {value['net']:+d} | {value['mcnemar_exact_p']:.4g} |")
    (artifact / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(results["overall"], indent=2))


if __name__ == "__main__": main()
