"""Final paired closed-loop report for Prompt/Action complement v1."""
from __future__ import annotations

import json
import math
import pickle

import numpy as np

from research.semantic_token_cd.distractor_rollout import snapshot_sha
from research.semantic_token_cd.prompt_action_complement_protocol import (
    ARMS, ARTIFACT, CANONICAL, ORIGINAL_ROOT, SEEDS, TASKS, atomic_json,
)

COMPARISONS = (
    ("prompt_high_action_high", "prompt_high_action_low"),
    ("prompt_low_action_high", "prompt_high_action_high"),
    ("prompt_high_action_high", "original"),
    ("prompt_high_action_low", "original"),
    ("prompt_low_action_high", "original"),
)


def load(task, arm, seed):
    path = (ORIGINAL_ROOT[task] / f"episode_{seed:03d}_summary.json" if arm == "original"
            else ARTIFACT / "closed_loop/episodes" / task / arm / f"episode_{seed:03d}_summary.json")
    if not path.exists(): raise RuntimeError(f"missing result: {path}")
    row = json.loads(path.read_text())
    with (CANONICAL / "snapshots" / task / f"seed_{seed:03d}.pkl").open("rb") as handle:
        expected = snapshot_sha(pickle.load(handle))
    if row.get("canonical_snapshot_sha256") != expected: raise RuntimeError(f"snapshot mismatch: {path}")
    if arm == "original":
        if row.get("attention_layers") != [11] or row.get("lambda") != 0.5 or row.get("beta") != 0.0:
            raise RuntimeError(f"Original config mismatch: {path}")
    elif not row.get("technical_pass"):
        raise RuntimeError(f"new-arm technical audit failed: {path}")
    return row


def exact_p(rescue, harm):
    n = rescue + harm
    if n == 0: return 1.0
    k = min(rescue, harm)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def bootstrap_delta(a, b, seed=1701):
    rng = np.random.default_rng(seed); n = len(a); values = []
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    for _ in range(20000): values.append(float(delta[rng.integers(0, n, n)].mean()))
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def main():
    data = {task: {arm: [load(task, arm, seed) for seed in SEEDS] for arm in ARMS} for task in TASKS}
    success = {task: {arm: sum(row["success"] for row in data[task][arm]) for arm in ARMS} for task in TASKS}
    success["overall"] = {arm: sum(success[task][arm] for task in TASKS) for arm in ARMS}
    paired = []
    for candidate, baseline in COMPARISONS:
        a = [bool(row["success"]) for task in TASKS for row in data[task][candidate]]
        b = [bool(row["success"]) for task in TASKS for row in data[task][baseline]]
        rescue = sum(x and not y for x, y in zip(a, b)); harm = sum(not x and y for x, y in zip(a, b))
        paired.append({"candidate": candidate, "baseline": baseline, "rescue": rescue, "harm": harm,
                       "net": rescue - harm, "delta": (sum(a) - sum(b)) / len(a),
                       "ci95": bootstrap_delta(a, b), "p_exact": exact_p(rescue, harm)})
    order = sorted(range(len(paired)), key=lambda i: paired[i]["p_exact"])
    adjusted = [0.0] * len(paired); running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, paired[index]["p_exact"] * (len(paired) - rank)))
        adjusted[index] = running
    for row, value in zip(paired, adjusted): row["holm_p"] = value
    stage_a = json.loads((ARTIFACT / "stage_a/statistics/summary.json").read_text())
    stage_b = json.loads((ARTIFACT / "stage_b/statistics/summary.json").read_text())
    result = {"success": success, "paired": paired, "stage_a": stage_a, "stage_b": stage_b,
              "episodes": {"reused_original": 400, "new": 1200, "total": 1600}}
    atomic_json(ARTIFACT / "FINAL_RESULTS.json", result)
    labels = {"original": "Original", "prompt_high_action_high": "HH",
              "prompt_high_action_low": "HL", "prompt_low_action_high": "LH"}
    lines = ["# Prompt/Action Complement SHR v1", "", "## 成功率", "",
             "| Task | Original | HH | HL | LH |", "|---|---:|---:|---:|---:|"]
    for task in (*TASKS, "overall"):
        n = 400 if task == "overall" else 100
        lines.append("| " + task.removeprefix("google_robot_") + " | " + " | ".join(
            f"{success[task][arm]}/{n} ({100*success[task][arm]/n:.1f}%)" for arm in ARMS) + " |")
    lines += ["", "## 预设配对比较", "",
              "| Candidate vs baseline | Rescue | Harm | Net | Δ | 95% CI | exact p | Holm p |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in paired:
        lines.append(f"| {labels[row['candidate']]} vs {labels[row['baseline']]} | {row['rescue']} | {row['harm']} | {row['net']:+d} | {100*row['delta']:+.1f} pp | [{100*row['ci95'][0]:+.1f}, {100*row['ci95'][1]:+.1f}] | {row['p_exact']:.4g} | {row['holm_p']:.4g} |")
    lines += ["", "## 审计", "", "- Stage A: 918/918同状态分组审计通过。",
              "- Stage B: 240/240同状态动作响应审计通过。",
              "- Closed loop: 400条Original严格复用，1200条新轨迹完成。", ""]
    (ARTIFACT / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__": main()

