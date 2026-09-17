"""Summarize and visualize same-state L11 budget differences."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ARMS = ("l11_matched", "l11_top_p80", "l11_top_p85")
SHORT = {
    "google_robot_open_drawer": "open_drawer", "google_robot_close_drawer": "close_drawer",
    "google_robot_pick_coke_can": "pick_coke_can", "google_robot_move_near": "move_near",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def overlay(image: np.ndarray, mask: np.ndarray, color=(1.0, 0.1, 0.1), alpha=.48) -> np.ndarray:
    result = image.astype(np.float32) / 255.0
    expanded = np.repeat(np.repeat(mask.reshape(16, 16), image.shape[0] // 16 + 1, axis=0),
                         image.shape[1] // 16 + 1, axis=1)[:image.shape[0], :image.shape[1]]
    tint = np.asarray(color, dtype=np.float32)
    result[expanded > 0] = (1 - alpha) * result[expanded > 0] + alpha * tint
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--top-p-artifact", type=Path, required=True)
    parser.add_argument("--matched-artifact", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact.resolve(); rows = []; payloads = []

    for path in sorted((root / "same_state_eval").rglob("step_*.json")):
        data = json.loads(path.read_text()); arrays = np.load(path.with_suffix(".npz"))
        item = {"path": path, "data": data, "arrays": arrays}; payloads.append(item)
        metrics = data["metrics"]
        row = {"state_id": data["state_id"], "task": SHORT[data["task"]],
               "seed": data["seed"], "step": data["source_step"], "phase": data["phase"]}
        for arm in ARMS:
            meta = metrics[arm]
            row[f"{arm}_m"] = meta["actual_selected_count"]
            row[f"{arm}_residual"] = meta["centered_logit_residual_norm"]
            row[f"{arm}_perturbation"] = meta["feature_perturbation_relative"]
            row[f"{arm}_clean_flips"] = meta["guided_changed_dims"]
        for arm in ARMS[1:]:
            row[f"{arm}_count_minus_matched"] = row[f"{arm}_m"] - row["l11_matched_m"]
            row[f"{arm}_action_dims_vs_matched"] = sum(
                a != b for a, b in zip(metrics[arm]["final_token_ids"][:6],
                                       metrics["l11_matched"]["final_token_ids"][:6])
            )
            row[f"{arm}_residual_minus_matched"] = row[f"{arm}_residual"] - row["l11_matched_residual"]
            diff = data["differences"][arm]
            row[f"{arm}_matched_only_tokens"] = len(diff["matched_minus_top_p"])
            row[f"{arm}_top_p_only_tokens"] = len(diff["top_p_minus_matched"])
        rows.append(row)

    if len(rows) != 60:
        raise RuntimeError(f"expected 60 completed same-state evaluations, found {len(rows)}")
    stats = []
    for task in [*SHORT.values(), "Overall"]:
        chosen = [row for row in rows if task == "Overall" or row["task"] == task]
        for arm in ARMS[1:]:
            delta = np.asarray([row[f"{arm}_count_minus_matched"] for row in chosen], dtype=float)
            action = np.asarray([row[f"{arm}_action_dims_vs_matched"] for row in chosen], dtype=float)
            residual = np.asarray([row[f"{arm}_residual_minus_matched"] for row in chosen], dtype=float)
            stats.append({
                "task": task, "arm": arm, "states": len(chosen),
                "mean_count_minus_matched": float(delta.mean()), "median_count_minus_matched": float(np.median(delta)),
                "top_p_selects_less_rate": float(np.mean(delta < 0)), "same_count_rate": float(np.mean(delta == 0)),
                "top_p_selects_more_rate": float(np.mean(delta > 0)), "mean_abs_count_difference": float(np.abs(delta).mean()),
                "action_diff_state_rate": float(np.mean(action > 0)), "mean_action_dims_vs_matched": float(action.mean()),
                "mean_residual_minus_matched": float(residual.mean()),
                "count_delta_action_dims_correlation": float(np.corrcoef(delta, action)[0, 1]) if delta.std() else 0.0,
            })

    statistics = root / "statistics"; figures = root / "state_figures"
    statistics.mkdir(exist_ok=True); figures.mkdir(exist_ok=True)
    write_csv(statistics / "same_state_metrics.csv", rows)
    write_csv(statistics / "task_budget_summary.csv", stats)

    # Episode outcomes locate cases, while remaining explicitly own-trajectory diagnostics.
    outcomes = []; case_manifest = defaultdict(lambda: defaultdict(list))
    top_root = args.top_p_artifact.resolve(); matched_root = args.matched_artifact.resolve()
    for task, short in SHORT.items():
        for seed in range(100, 200):
            matched = json.loads((matched_root / "episodes" / task / "l11_matched" / f"episode_{seed:03d}_summary.json").read_text())
            p80 = json.loads((top_root / "episodes" / task / "l11_top_p80" / f"episode_{seed:03d}_summary.json").read_text())
            p85 = json.loads((top_root / "episodes" / task / "l11_top_p85" / f"episode_{seed:03d}_summary.json").read_text())
            category = ("matched_only" if matched["success"] and not p85["success"] else
                        "top_p85_only" if p85["success"] and not matched["success"] else
                        "both_success" if matched["success"] else "both_fail")
            row = {"task": short, "seed": seed, "category": category,
                   "matched_success": matched["success"], "top_p80_success": p80["success"], "top_p85_success": p85["success"],
                   "matched_own_trajectory_mean_m": matched["mean_selected_token_count"],
                   "top_p80_own_trajectory_mean_m": p80["mean_m_t"], "top_p85_own_trajectory_mean_m": p85["mean_m_t"]}
            outcomes.append(row); case_manifest[short][category].append(seed)
    write_csv(statistics / "closed_loop_outcome_budget_locator.csv", outcomes)
    selected_cases = {}
    desired = {"matched_only": 4, "top_p85_only": 4, "both_success": 2, "both_fail": 2}
    for task, buckets in case_manifest.items():
        selected_cases[task] = {}
        for category, count in desired.items():
            values = buckets[category]
            if not values:
                selected_cases[task][category] = []
                continue
            indices = np.linspace(0, len(values) - 1, min(count, len(values))).round().astype(int)
            selected_cases[task][category] = [values[index] for index in indices]
    (statistics / "outcome_case_manifest.json").write_text(json.dumps(selected_cases, indent=2) + "\n")

    # Render every state, plus a compact gallery index ordered by action impact and count gap.
    for item, row in zip(payloads, rows):
        data=item["data"]; arrays=item["arrays"]; image=arrays["image"]
        attention=arrays["l11_matched__prompt_attention"].reshape(16, 16)
        fig, axes=plt.subplots(2, 3, figsize=(13, 8))
        axes[0,0].imshow(image); axes[0,0].set_title(f'{row["task"]} seed {row["seed"]} {row["phase"]}')
        axes[0,1].imshow(image); axes[0,1].imshow(attention, cmap="magma", alpha=.62,
                                                extent=(0,image.shape[1],image.shape[0],0)); axes[0,1].set_title("L11 attention")
        for ax,arm,title in ((axes[0,2],"l11_matched","Matched"),(axes[1,0],"l11_top_p80","TopP80"),(axes[1,1],"l11_top_p85","TopP85")):
            mask=arrays[f"{arm}__selected_mask"]; ax.imshow(overlay(image,mask)); ax.set_title(f'{title}: m={row[f"{arm}_m"]}')
        matched=arrays["l11_matched__selected_mask"].astype(bool); p85=arrays["l11_top_p85__selected_mask"].astype(bool)
        diff=np.zeros((256,3),dtype=float); diff[matched & ~p85]=(.1,.3,1.0); diff[p85 & ~matched]=(1.0,.55,.05)
        mask=(matched ^ p85).astype(np.uint8); base=image.astype(float)/255
        expanded=np.repeat(np.repeat(diff.reshape(16,16,3),image.shape[0]//16+1,0),image.shape[1]//16+1,1)[:image.shape[0],:image.shape[1]]
        emask=np.repeat(np.repeat(mask.reshape(16,16),image.shape[0]//16+1,0),image.shape[1]//16+1,1)[:image.shape[0],:image.shape[1]]
        base[emask>0]=.45*base[emask>0]+.55*expanded[emask>0]; axes[1,2].imshow(base)
        axes[1,2].set_title("Difference: blue=Matched only, orange=TopP85 only")
        for ax in axes.ravel(): ax.axis("off")
        fig.tight_layout(); fig.savefig(figures/f'{data["state_id"]}.png',dpi=150); plt.close(fig)

    ranked = sorted(rows, key=lambda row: (row["l11_top_p85_action_dims_vs_matched"],
                                           abs(row["l11_top_p85_count_minus_matched"])), reverse=True)
    summary = {"state_count": len(rows), "task_summary": stats,
               "most_informative_states": [row["state_id"] for row in ranked[:12]],
               "closed_loop_case_manifest": selected_cases,
               "causal_boundary": "same-state one-step action diagnostics; no environment continuation performed"}
    (statistics / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
