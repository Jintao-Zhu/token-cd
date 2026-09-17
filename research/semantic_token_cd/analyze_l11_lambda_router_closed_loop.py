"""Summarize paired held-out closed-loop L11 lambda routing."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from research.semantic_token_cd.l11_lambda_router_protocol import ROUTER_PROTOCOL, SEEDS, SHORT, TASKS, atomic_json


HALF_ROOT = Path("artifacts/prompt_attn_l11_matched_full_9task_0_299_v1/run/episodes")


def pair(router: np.ndarray, fixed: np.ndarray) -> dict:
    rescue = int(np.sum((router == 1) & (fixed == 0)))
    harm = int(np.sum((router == 0) & (fixed == 1)))
    return {
        "rescue": rescue, "harm": harm, "net": rescue - harm,
        "exact_p": float(binomtest(min(rescue, harm), rescue + harm, 0.5).pvalue) if rescue + harm else 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    confirmation = args.confirmation.resolve()
    predicted = {
        (row["task"], int(row["seed"])): row
        for row in csv.DictReader((confirmation / "analysis" / "predictions.csv").open())
    }
    rows = []
    all_router = []
    all_fixed = []
    by_task = {}
    for task in TASKS:
        task_router = []
        task_fixed = []
        for seed in SEEDS:
            router = json.loads((artifact / "episodes" / task / "router" / f"episode_{seed:03d}_summary.json").read_text())
            fixed = json.loads((HALF_ROOT / task / "prompt_single" / f"episode_{seed:03d}_summary.json").read_text())
            if router["protocol_id"] != ROUTER_PROTOCOL or not router["technical_pass"]:
                raise RuntimeError(f"router technical audit failure: {task} seed={seed}")
            prediction = predicted[(SHORT[task], seed)]
            if float(prediction["chosen_lambda"]) != float(router["chosen_lambda"]):
                raise RuntimeError("online choice differs from frozen confirmation prediction")
            if not router["replay_actions_equal"] or not router["replay_success_equal"]:
                raise RuntimeError("online route did not replay selected fixed arm")
            task_router.append(int(router["success"]))
            task_fixed.append(int(fixed["success"]))
            rows.append({
                "task": SHORT[task], "seed": seed, "chosen_lambda": router["chosen_lambda"],
                "router_success": int(router["success"]), "fixed_050_success": int(fixed["success"]),
            })
        router_array = np.asarray(task_router)
        fixed_array = np.asarray(task_fixed)
        result = pair(router_array, fixed_array)
        result.update({"router_successes": int(router_array.sum()), "fixed_050_successes": int(fixed_array.sum())})
        by_task[SHORT[task]] = result
        all_router.extend(task_router)
        all_fixed.extend(task_fixed)
    router = np.asarray(all_router)
    fixed = np.asarray(all_fixed)
    payload = {
        "protocol_id": ROUTER_PROTOCOL,
        "seeds": [100, 199],
        "router_successes": int(router.sum()),
        "fixed_050_successes": int(fixed.sum()),
        "paired_vs_fixed_050": pair(router, fixed),
        "by_task": by_task,
        "all_online_choices_match_frozen_predictions": True,
        "all_online_actions_match_selected_fixed_arm": True,
    }
    output = artifact / "analysis"
    atomic_json(output / "CLOSED_LOOP_RESULTS.json", payload)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "paired_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# L11 Episode-Start Lambda Router Closed Loop", "",
        "| Task | Router | Fixed .5 | Rescue | Harm | Net | p |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        result = by_task[SHORT[task]]
        lines.append(f"| {SHORT[task]} | {result['router_successes']}/100 | {result['fixed_050_successes']}/100 | {result['rescue']} | {result['harm']} | {result['net']:+d} | {result['exact_p']:.4g} |")
    pooled = payload["paired_vs_fixed_050"]
    lines.extend(["", f"- Router: **{int(router.sum())}/200**", f"- Fixed λ=.5: **{int(fixed.sum())}/200**", f"- Rescue/Harm/Net: **{pooled['rescue']}/{pooled['harm']}/{pooled['net']:+d}**", f"- Exact paired p: **{pooled['exact_p']:.6g}**"])
    (output / "CLOSED_LOOP_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
