"""Final paired report for PA-A10 and PA-Joint10 against L11-Matched."""
from __future__ import annotations

import json
import math
import pickle

import numpy as np

from research.semantic_token_cd.distractor_rollout import snapshot_sha
from research.semantic_token_cd.prompt_action_joint_protocol import (
    ARMS, ARTIFACT, CANONICAL, MATCHED_ROOT, SEEDS, TASKS, atomic_json,
)


def exact_p(rescue, harm):
    n = rescue + harm
    if not n:
        return 1.0
    k = min(rescue, harm)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def load(task, arm, seed):
    name = f"episode_{seed:03d}_summary.json"
    path = (MATCHED_ROOT[task] / name if arm == "l11_matched"
            else ARTIFACT / "closed_loop/episodes" / task / arm / name)
    row = json.loads(path.read_text())
    with (CANONICAL / "snapshots" / task / f"seed_{seed:03d}.pkl").open("rb") as handle:
        expected = snapshot_sha(pickle.load(handle))
    if row.get("canonical_snapshot_sha256") != expected:
        raise RuntimeError(f"snapshot mismatch: {path}")
    if arm != "l11_matched" and row.get("technical_pass") is not True:
        raise RuntimeError(f"technical audit failure: {path}")
    return row


def comparison(a, b):
    rescue = sum(x and not y for x, y in zip(a, b))
    harm = sum(not x and y for x, y in zip(a, b))
    return {"n": len(a), "rescue": rescue, "harm": harm, "net": rescue-harm,
            "delta": (sum(a)-sum(b))/len(a), "mcnemar_exact_p": exact_p(rescue, harm)}


def main():
    arms = ("l11_matched", *ARMS)
    data = {t: {a: [load(t, a, s) for s in SEEDS] for a in arms} for t in TASKS}
    success = {t: {a: sum(x["success"] for x in data[t][a]) for a in arms} for t in TASKS}
    success["overall"] = {a: sum(success[t][a] for t in TASKS) for a in arms}
    paired = {}
    for candidate, baseline in (("pa_a10", "l11_matched"),
                                ("pa_joint10", "l11_matched"),
                                ("pa_joint10", "pa_a10")):
        key = f"{candidate}_vs_{baseline}"
        paired[key] = {}
        for task in (*TASKS, "overall"):
            use = TASKS if task == "overall" else (task,)
            a = [bool(x["success"]) for t in use for x in data[t][candidate]]
            b = [bool(x["success"]) for t in use for x in data[t][baseline]]
            paired[key][task] = comparison(a, b)
    diagnostics = {}
    for arm in ARMS:
        rows = [x for t in TASKS for episode in data[t][arm] for x in episode["selector_trace"]]
        diagnostics[arm] = {
            "control_steps": len(rows),
            "mean_m": float(np.mean([x["m_t"] for x in rows])),
            "mean_overlap": float(np.mean([x["bounded_overlap_ratio"] for x in rows])),
            "mean_jaccard": float(np.mean([x["bounded_jaccard"] for x in rows])),
            "mean_components": float(np.mean([x["mask_component_count"] for x in rows])),
            "mean_isolated_ratio": float(np.mean([x["isolated_token_ratio"] for x in rows])),
            "mean_feature_perturbation": float(np.mean([x["feature_perturbation_norm"] for x in rows])),
            "mean_centered_residual": float(np.mean([x["centered_logit_residual_norm"] for x in rows])),
            "mean_action_flips": float(np.mean([x["guided_changed_dims"] for x in rows])),
        }
    result = {"complete": True, "new_episodes": 800, "success": success,
              "paired": paired, "diagnostics": diagnostics}
    atomic_json(ARTIFACT / "FINAL_RESULTS.json", result)
    lines = ["# PA-Constrained Joint L11 v1", "", "| Task | L11-Matched | PA-A10 | PA-Joint10 |",
             "|---|---:|---:|---:|"]
    for task in (*TASKS, "overall"):
        n = 400 if task == "overall" else 100
        lines.append("| " + task.removeprefix("google_robot_") + " | " + " | ".join(
            f"{success[task][a]}/{n} ({100*success[task][a]/n:.1f}%)" for a in arms) + " |")
    lines += ["", "| Comparison | Rescue | Harm | Net | Delta | exact p |", "|---|---:|---:|---:|---:|---:|"]
    for key, tasks in paired.items():
        row = tasks["overall"]
        lines.append(f"| {key} | {row['rescue']} | {row['harm']} | {row['net']:+d} | {100*row['delta']:+.1f} pp | {row['mcnemar_exact_p']:.4g} |")
    (ARTIFACT / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
