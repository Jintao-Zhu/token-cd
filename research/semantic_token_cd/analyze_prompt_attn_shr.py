"""Aggregate Prompt-Attn-SHR v1 closed-loop and mechanism results."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from research.semantic_token_cd.prompt_attn_shr_rollout import ARMS, TASKS


def binomial_two_sided(rescue: int, harm: int) -> float:
    n = rescue + harm
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(rescue, harm) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def paired(left: dict, right: dict) -> dict:
    keys = sorted(set(left) & set(right))
    rescue = sum(not left[key]["success"] and right[key]["success"] for key in keys)
    harm = sum(left[key]["success"] and not right[key]["success"] for key in keys)
    return {
        "pairs": len(keys),
        "left_success": sum(left[key]["success"] for key in keys),
        "right_success": sum(right[key]["success"] for key in keys),
        "rescue": rescue,
        "harm": harm,
        "net": rescue - harm,
        "mcnemar_exact_p": binomial_two_sided(rescue, harm),
    }


def mean(rows: list[dict], key: str) -> float | None:
    values = np.asarray([row[key] for row in rows if row.get(key) is not None], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else None


def load(artifact: Path) -> dict[str, dict[tuple[str, int], dict]]:
    data = {arm: {} for arm in ARMS}
    for task in TASKS:
        for arm in ARMS:
            for path in (artifact / "episodes" / task / arm).glob("episode_*_summary.json"):
                row = json.loads(path.read_text())
                key = (task, int(row["seed"]))
                if key in data[arm]:
                    raise RuntimeError(f"duplicate result: {arm} {key}")
                if not row.get("technical_pass"):
                    raise RuntimeError(f"technical audit failed: {path}")
                data[arm][key] = row
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    data = load(artifact)
    result = {"complete": all(len(data[arm]) == 300 for arm in ARMS), "by_task": {}, "overall": {}}
    for task in TASKS:
        task_result = {"arms": {}, "paired": {}}
        for arm in ARMS:
            rows = [row for (row_task, _), row in data[arm].items() if row_task == task]
            wins = sum(row["success"] for row in rows)
            task_result["arms"][arm] = {
                "episodes": len(rows),
                "success": wins,
                "success_rate": wins / len(rows) if rows else None,
                "mean_m_t": mean(rows, "mean_m_t"),
                "mean_prompt_shr_overlap_ratio": mean(rows, "mean_prompt_shr_overlap_ratio"),
                "mean_feature_perturbation_norm": mean(rows, "mean_feature_perturbation_norm"),
                "mean_centered_logit_residual_norm": mean(rows, "mean_centered_logit_residual_norm"),
                "mean_guided_change_ratio": mean(rows, "mean_guided_change_ratio"),
            }
        for left, right in (
            ("standard_shr", "prompt_attn_shr"),
            ("random_shr", "prompt_attn_shr"),
            ("standard_shr", "random_shr"),
        ):
            lv = {key: row for key, row in data[left].items() if key[0] == task}
            rv = {key: row for key, row in data[right].items() if key[0] == task}
            task_result["paired"][f"{right}_vs_{left}"] = paired(lv, rv)
        result["by_task"][task] = task_result

    result["overall"]["arms"] = {}
    for arm in ARMS:
        rows = list(data[arm].values())
        wins = sum(row["success"] for row in rows)
        result["overall"]["arms"][arm] = {
            "episodes": len(rows), "success": wins,
            "success_rate": wins / len(rows) if rows else None,
            "mean_m_t": mean(rows, "mean_m_t"),
            "mean_prompt_shr_overlap_ratio": mean(rows, "mean_prompt_shr_overlap_ratio"),
            "mean_feature_perturbation_norm": mean(rows, "mean_feature_perturbation_norm"),
            "mean_centered_logit_residual_norm": mean(rows, "mean_centered_logit_residual_norm"),
            "mean_guided_change_ratio": mean(rows, "mean_guided_change_ratio"),
        }
    result["overall"]["paired"] = {}
    for left, right in (
        ("standard_shr", "prompt_attn_shr"),
        ("random_shr", "prompt_attn_shr"),
        ("standard_shr", "random_shr"),
    ):
        result["overall"]["paired"][f"{right}_vs_{left}"] = paired(data[left], data[right])
    output = artifact / "FINAL_RESULTS.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["overall"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
