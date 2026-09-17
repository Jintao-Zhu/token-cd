"""Summarize the drawer phase x guidance-strength x token-group causal audit."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STRENGTH_ARMS = ("dynamic_l11_lambda_0p5", "dynamic_l11_lambda_0p25", "clean_lambda_0")


def mean(values):
    values = [float(x) for x in values if x is not None and np.isfinite(float(x))]
    return float(np.mean(values)) if values else None


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def aggregate(rows: list[dict], value: str, keys: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for labels, subset in sorted(groups.items()):
        record = {key: label for key, label in zip(keys, labels)}
        vals = [x[value] for x in subset]
        record.update(n=len(subset), episodes=len({(x["task"], x["seed"]) for x in subset}),
                      mean=mean(vals), median=float(np.median(vals)),
                      positive=sum(x > 0.002 for x in vals), negative=sum(x < -0.002 for x in vals))
        output.append(record)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args(); root = args.artifact.resolve()
    manifest = json.loads((root / "MANIFEST.json").read_text())["states"]
    files = sorted((root / "results").rglob("*.json"))
    if len(files) != len(manifest):
        raise RuntimeError(f"incomplete: {len(files)}/{len(manifest)} states")
    strength_rows, token_rows, integrity = [], [], []
    for path in files:
        data = json.loads(path.read_text()); results = data["results"]
        audit = data["restore_audit"]
        integrity.append(audit["state_max_abs_difference"] <= 1e-6 and
                         audit["rgb_mean_abs_difference"] <= .1 and audit["rgb_gt16_fraction"] <= 1e-3)
        base = results["dynamic_l11_lambda_0p5"]["primary_progress_gain"]
        common = {"task": data["task"].removeprefix("google_robot_"), "seed": data["seed"],
                  "category": data["category"], "phase": data["phase"], "control_step": data["control_step"]}
        for arm in STRENGTH_ARMS:
            branch = results[arm]
            strength_rows.append({**common, "arm": arm,
                "progress": branch["primary_progress_gain"], "delta_vs_lambda_0p5": branch["primary_progress_gain"]-base,
                "mean_action_l2": branch["mean_guided_clean_action_l2"],
                "mean_residual_norm": branch["mean_residual_norm"], "final_success": int(branch["after"]["success"])})
        locked = results["locked_full_then_l11_0p5"]["primary_progress_gain"]
        for arm, branch in results.items():
            if not arm.startswith("without_"):
                continue
            token_rows.append({**common, "sector": arm.removeprefix("without_"),
                "removed_count": len(branch["removed_token_ids"]), "progress": branch["primary_progress_gain"],
                "delta_vs_locked_full": branch["primary_progress_gain"]-locked,
                "mean_action_l2": branch["mean_guided_clean_action_l2"],
                "mean_residual_norm": branch["mean_residual_norm"], "final_success": int(branch["after"]["success"])})
    write_csv(root / "strength_rows.csv", strength_rows); write_csv(root / "token_rows.csv", token_rows)
    strength_summary = aggregate(strength_rows, "delta_vs_lambda_0p5", ("task", "category", "phase", "arm"))
    token_summary = aggregate(token_rows, "delta_vs_locked_full", ("task", "category", "phase", "sector"))

    def lookup(rows, task, category, phase, arm_key, arm):
        subset = [x for x in rows if x["task"] == task and x["category"] == category and
                  x["phase"] == phase and x[arm_key] == arm]
        return mean(x["delta_vs_lambda_0p5" if arm_key == "arm" else "delta_vs_locked_full"] for x in subset)

    tasks = sorted({x["task"] for x in strength_rows}); phases = ("approach", "contact_or_motion", "late")
    # A strength mechanism requires attenuation to help Harm without producing the same-sized loss on Rescue.
    phase_evidence = []
    for task in tasks:
        for phase in phases:
            for arm in ("dynamic_l11_lambda_0p25", "clean_lambda_0"):
                h = lookup(strength_rows, task, "harm", phase, "arm", arm)
                r = lookup(strength_rows, task, "rescue", phase, "arm", arm)
                phase_evidence.append({"task": task, "phase": phase, "arm": arm,
                                       "harm_delta": h, "rescue_delta": r,
                                       "desired_direction": bool(h is not None and r is not None and h > .002 and r >= -.002)})
    strength_hits = [x for x in phase_evidence if x["desired_direction"]]
    # A token rule must have the desired sign in both drawer tasks at the same phase.
    sectors = sorted({x["sector"] for x in token_rows}); token_candidates = []
    for phase in phases:
        for sector in sectors:
            values = []
            for task in tasks:
                h = lookup(token_rows, task, "harm", phase, "sector", sector)
                r = lookup(token_rows, task, "rescue", phase, "sector", sector)
                values.append({"task": task, "harm_delta": h, "rescue_delta": r})
            qualifies = all(x["harm_delta"] is not None and x["rescue_delta"] is not None and
                            x["harm_delta"] > .002 and x["rescue_delta"] >= -.002 for x in values)
            token_candidates.append({"phase": phase, "sector": sector, "tasks": values,
                                     "desired_direction_both_tasks": qualifies})
    stable_tokens = [x for x in token_candidates if x["desired_direction_both_tasks"]]
    if strength_hits and stable_tokens:
        decision = "MIXED_PHASE_STRENGTH_AND_TOKEN_EVIDENCE"
    elif strength_hits:
        decision = "PHASE_DEPENDENT_GUIDANCE_STRENGTH_EVIDENCE"
    elif stable_tokens:
        decision = "PHASE_SPECIFIC_TOKEN_GROUP_EVIDENCE"
    else:
        decision = "NO_STABLE_LOCAL_RULE_FOUND"

    # Compact heatmap: lambda .25 effect relative to .5.
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    vmax = .03
    for axis, category in zip(axes, ("harm", "rescue")):
        matrix = np.array([[lookup(strength_rows, task, category, phase, "arm", "dynamic_l11_lambda_0p25")
                            for phase in phases] for task in tasks], dtype=float)
        image = axis.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axis.set_title(f"lambda .25 - .5: {category}"); axis.set_xticks(range(3), phases, rotation=25, ha="right")
        axis.set_yticks(range(len(tasks)), tasks)
        for y in range(len(tasks)):
            for x in range(3): axis.text(x, y, f"{1000*matrix[y,x]:+.1f}", ha="center", va="center", fontsize=9)
    figure.colorbar(image, ax=axes, label="10-action progress delta (mm)")
    figure.savefig(root / "phase_strength_heatmap.png", dpi=180); plt.close(figure)
    final = {"protocol_id": "DRAWER_PHASE_STRENGTH_TOKEN_CAUSAL_V1", "decision": decision,
             "integrity_pass": all(integrity), "states": len(files), "strength_branches": len(strength_rows),
             "token_ablations": len(token_rows), "phase_strength_evidence": phase_evidence,
             "stable_strength_hits": strength_hits, "token_candidates": token_candidates,
             "stable_token_candidates": stable_tokens,
             "interpretation_boundary": "Same-state 10-action physical progress; not full-episode success."}
    (root / "FINAL_RESULTS.json").write_text(json.dumps(final, indent=2) + "\n")
    lines = ["# Drawer Phase × Strength × Token Causal Audit", "", "## Decision", "", f"**{decision}**", "",
             "## Design and integrity", "", f"- Same-state snapshots: {len(files)}/{len(manifest)}; restore gate: {'PASS' if all(integrity) else 'FAIL'}.",
             "- Strength branches apply dynamic L11-Matched for 10 actions at lambda 0.5, 0.25, or 0.",
             "- Token branches alter only the first decision's locked mask, then all return to dynamic L11-Matched lambda 0.5.",
             "- Positive delta means more task-native physical progress than the corresponding baseline.", "",
             "## Phase-specific strength result", "", "| Task | Outcome | Phase | λ=.25 minus .5 (mm) | clean minus .5 (mm) |",
             "|---|---|---|---:|---:|"]
    for task in tasks:
        for category in ("harm", "rescue"):
            for phase in phases:
                q = lookup(strength_rows, task, category, phase, "arm", "dynamic_l11_lambda_0p25")
                c = lookup(strength_rows, task, category, phase, "arm", "clean_lambda_0")
                lines.append(f"| {task} | {category} | {phase} | {1000*q:+.2f} | {1000*c:+.2f} |")
    lines += ["", "## Mechanism boundary", "",
              "The branch outcomes are causal for the restored state and next 10 actions. They identify whether attenuation or one token subset changes immediate physical progress; they do not by themselves prove a full rollout would flip success.",
              "", "![Phase strength heatmap](phase_strength_heatmap.png)", "", "## Files", "",
              "- `FINAL_RESULTS.json`: decision and machine-readable evidence.",
              "- `strength_rows.csv`: every state × lambda branch.",
              "- `token_rows.csv`: every state × removed sector."]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"decision": decision, "states": len(files), "token_ablations": len(token_rows)}))


if __name__ == "__main__":
    main()
