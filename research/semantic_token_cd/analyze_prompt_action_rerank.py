"""Final paired report for PA-Rerank-L11-Matched v1."""
from __future__ import annotations

import json
import math
import pickle

import numpy as np

from research.semantic_token_cd.distractor_rollout import snapshot_sha
from research.semantic_token_cd.prompt_action_rerank_protocol import (
    ARM, ARTIFACT, CANONICAL, HH_ROOT, MATCHED_ROOT, SEEDS, TASKS, atomic_json,
)

ARMS = ("vanilla", "shr_harmonic", "l11_matched", "hh", ARM)
COMPARISONS = (
    (ARM, "l11_matched"),
    (ARM, "hh"),
    (ARM, "shr_harmonic"),
    (ARM, "vanilla"),
)


def path_for(task: str, arm: str, seed: int):
    name = f"episode_{seed:03d}_summary.json"
    if arm == ARM:
        return ARTIFACT / "closed_loop/episodes" / task / arm / name
    if arm == "l11_matched":
        return MATCHED_ROOT[task] / name
    if arm == "hh":
        return HH_ROOT / task / "prompt_high_action_high" / name
    return CANONICAL / "episodes" / task / arm / name


def load(task: str, arm: str, seed: int):
    path = path_for(task, arm, seed)
    if not path.exists():
        raise RuntimeError(f"missing result: {path}")
    row = json.loads(path.read_text())
    with (CANONICAL / "snapshots" / task / f"seed_{seed:03d}.pkl").open("rb") as handle:
        expected = snapshot_sha(pickle.load(handle))
    if row.get("canonical_snapshot_sha256") != expected:
        raise RuntimeError(f"snapshot mismatch: {path}")
    if arm == ARM and row.get("technical_pass") is not True:
        raise RuntimeError(f"PA-Rerank technical audit failed: {path}")
    return row


def exact_p(rescue: int, harm: int) -> float:
    n = rescue + harm
    if n == 0:
        return 1.0
    k = min(rescue, harm)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def bootstrap_delta(candidate, baseline, seed=2711):
    delta = np.asarray(candidate, dtype=float) - np.asarray(baseline, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = np.empty(20000, dtype=float)
    for index in range(sampled.size):
        sampled[index] = delta[rng.integers(0, delta.size, delta.size)].mean()
    return [float(x) for x in np.quantile(sampled, (0.025, 0.975))]


def paired(candidate, baseline):
    rescue = sum(a and not b for a, b in zip(candidate, baseline))
    harm = sum(not a and b for a, b in zip(candidate, baseline))
    return {
        "n": len(candidate), "rescue": rescue, "harm": harm,
        "net": rescue - harm,
        "delta": (sum(candidate) - sum(baseline)) / len(candidate),
        "ci95": bootstrap_delta(candidate, baseline),
        "mcnemar_exact_p": exact_p(rescue, harm),
    }


def mechanism(rows):
    traces = [step for row in rows for step in row["selector_trace"]]
    fields = (
        "m_t", "candidate_pool_size", "rerank_overlap_ratio", "rerank_jaccard",
        "mean_selected_prompt_attention", "mean_selected_action_attention",
        "mean_original_l11_action_attention", "feature_perturbation_norm",
        "centered_logit_residual_norm", "guided_changed_dims",
    )
    result = {"control_steps": len(traces)}
    for field in fields:
        values = np.asarray([step[field] for step in traces], dtype=float)
        result[f"mean_{field}"] = float(values.mean())
        result[f"median_{field}"] = float(np.median(values))
    result["candidate_saturated_steps"] = sum(
        bool(step["candidate_pool_saturated"]) for step in traces
    )
    result["candidate_saturation_rate"] = (
        result["candidate_saturated_steps"] / len(traces)
    )
    result["action_changed_step_rate"] = float(np.mean([
        step["guided_changed_dims"] > 0 for step in traces
    ]))
    return result


def main() -> None:
    data = {
        task: {arm: [load(task, arm, seed) for seed in SEEDS] for arm in ARMS}
        for task in TASKS
    }
    success = {
        task: {arm: sum(bool(row["success"]) for row in data[task][arm]) for arm in ARMS}
        for task in TASKS
    }
    success["overall"] = {
        arm: sum(success[task][arm] for task in TASKS) for arm in ARMS
    }
    comparisons = {}
    for candidate, baseline in COMPARISONS:
        comparisons[f"{candidate}_vs_{baseline}"] = {}
        for task in (*TASKS, "overall"):
            tasks = TASKS if task == "overall" else (task,)
            a = [bool(row["success"]) for item in tasks for row in data[item][candidate]]
            b = [bool(row["success"]) for item in tasks for row in data[item][baseline]]
            comparisons[f"{candidate}_vs_{baseline}"][task] = paired(a, b)
    diagnostics = {
        task: mechanism(data[task][ARM]) for task in TASKS
    }
    diagnostics["overall"] = mechanism([
        row for task in TASKS for row in data[task][ARM]
    ])
    result = {
        "complete": True, "new_episodes": 400, "reused_baselines": list(ARMS[:-1]),
        "success": success, "paired": comparisons, "mechanism": diagnostics,
    }
    atomic_json(ARTIFACT / "FINAL_RESULTS.json", result)

    labels = {
        "vanilla": "Vanilla", "shr_harmonic": "SHR", "l11_matched": "L11-Matched",
        "hh": "HH", ARM: "PA-Rerank",
    }
    lines = [
        "# PA-Rerank-L11-Matched v1", "", "## Success rate", "",
        "| Task | Vanilla | SHR | L11-Matched | HH | PA-Rerank |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task in (*TASKS, "overall"):
        n = 400 if task == "overall" else 100
        lines.append("| " + task.removeprefix("google_robot_") + " | " + " | ".join(
            f"{success[task][arm]}/{n} ({100*success[task][arm]/n:.1f}%)" for arm in ARMS
        ) + " |")
    lines += ["", "## Primary paired comparison", "",
              "| Task | Rescue | Harm | Net | Delta | 95% CI | exact p |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    primary = comparisons[f"{ARM}_vs_l11_matched"]
    for task in (*TASKS, "overall"):
        row = primary[task]
        lines.append(
            f"| {task.removeprefix('google_robot_')} | {row['rescue']} | {row['harm']} | "
            f"{row['net']:+d} | {100*row['delta']:+.1f} pp | "
            f"[{100*row['ci95'][0]:+.1f}, {100*row['ci95'][1]:+.1f}] | "
            f"{row['mcnemar_exact_p']:.4g} |"
        )
    lines += ["", "## Selector diagnostics", "",
              "| Task | mean K | mean pool | saturation | overlap | Jaccard | action-changed steps |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for task in (*TASKS, "overall"):
        row = diagnostics[task]
        lines.append(
            f"| {task.removeprefix('google_robot_')} | {row['mean_m_t']:.2f} | "
            f"{row['mean_candidate_pool_size']:.2f} | {100*row['candidate_saturation_rate']:.2f}% | "
            f"{100*row['mean_rerank_overlap_ratio']:.1f}% | {row['mean_rerank_jaccard']:.3f} | "
            f"{100*row['action_changed_step_rate']:.1f}% |"
        )
    (ARTIFACT / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
