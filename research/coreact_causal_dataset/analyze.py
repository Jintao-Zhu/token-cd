#!/usr/bin/env python3
"""Analyze label existence only after the locked rollout manifest is complete."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from research.coreact_causal_dataset.common import paired_effect
from research.coreact_closed_loop.run_pilot import read_jsonl, write_json


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    manifest = read_jsonl(artifact / "rollout_manifest.jsonl")
    completed, failed, missing = {}, [], []
    for spec in manifest:
        directory = artifact / "episodes" / spec["episode_id"]
        result, failure = directory / "episode.json", directory / "failure.json"
        if result.exists():
            completed[spec["episode_id"]] = json.loads(result.read_text())
        elif failure.exists():
            failed.append(json.loads(failure.read_text()))
        else:
            missing.append(spec["episode_id"])
    manifest_ids = [spec["episode_id"] for spec in manifest]
    duplicate_manifest_ids = len(manifest_ids) - len(set(manifest_ids))
    completeness = {
        "expected": len(manifest), "complete": len(completed), "failed": len(failed),
        "missing": len(missing), "duplicates": duplicate_manifest_ids,
        "complete_snapshots": 0, "paired_identity_failures": [],
    }
    specs_by_snapshot = defaultdict(list)
    for spec in manifest:
        specs_by_snapshot[spec["snapshot_id"]].append(spec)
    for snapshot_id, specs in specs_by_snapshot.items():
        if all(spec["episode_id"] in completed for spec in specs):
            completeness["complete_snapshots"] += 1
        for replicate in range(3):
            clean_spec = next(s for s in specs if s["condition"] == "clean" and s["replicate"] == replicate)
            if clean_spec["episode_id"] not in completed:
                continue
            clean = completed[clean_spec["episode_id"]]
            for masked_spec in (s for s in specs if s["condition"] == "masked" and s["replicate"] == replicate):
                if masked_spec["episode_id"] not in completed:
                    continue
                masked = completed[masked_spec["episode_id"]]
                shared = min(len(clean["noise_sha256_by_replan"]), len(masked["noise_sha256_by_replan"]))
                if not (clean["initial_sim_state_sha256"] == masked["initial_sim_state_sha256"]
                        and clean["initial_prefix_sha256"] == masked["initial_prefix_sha256"]
                        and clean["noise_sha256_by_replan"][:shared] == masked["noise_sha256_by_replan"][:shared]
                        and clean["rollout_seed"] == masked["rollout_seed"]):
                    completeness["paired_identity_failures"].append(masked_spec["episode_id"])
    write_json(artifact / "completeness.json", completeness)
    if len(completed) != len(manifest) or failed or missing or completeness["paired_identity_failures"]:
        write_json(artifact / "decision.json", {
            "status": "INCONCLUSIVE_DATA_OR_RESOURCES", "reason": "rollout completeness or paired identity failed",
            "completeness": completeness,
        })
        raise SystemExit("analysis embargo: manifest is incomplete or integrity failed")

    rows = []
    for snapshot_id, specs in sorted(specs_by_snapshot.items()):
        clean = {s["replicate"]: completed[s["episode_id"]] for s in specs if s["condition"] == "clean"}
        groups = defaultdict(list)
        for spec in specs:
            if spec["condition"] == "masked":
                groups[spec["group_id"]].append(spec)
        for group_id, group_specs in sorted(groups.items()):
            group_specs.sort(key=lambda x: x["replicate"])
            clean_success = [int(clean[s["replicate"]]["success"]) for s in group_specs]
            masked_success = [int(completed[s["episode_id"]]["success"]) for s in group_specs]
            effect = paired_effect(clean_success, masked_success)
            first = group_specs[0]
            rows.append({
                "snapshot_id": snapshot_id, "suite": first["suite"], "task_id": first["task_id"],
                "phase": first["assigned_phase"], "phase_fallback": first["phase_fallback"],
                "group_id": group_id, "stratum": first["stratum"],
                "token_index": first["token_indices"][0], "attention_score": first["attention_score"],
                "clean_successes": sum(clean_success), "masked_successes": sum(masked_success), **effect,
            })
    fields = list(rows[0])
    with (artifact / "causal_group_labels.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    strong = [row for row in rows if row["strong_anchor"] or row["strong_nuisance"]]
    anchors = [row for row in rows if row["strong_anchor"]]
    nuisances = [row for row in rows if row["strong_nuisance"]]
    rho = spearmanr([row["attention_score"] for row in rows], [abs(row["tau"]) for row in rows])
    by_task = {}
    for task_id in sorted({row["task_id"] for row in rows}):
        subset = [row for row in rows if row["task_id"] == task_id]
        by_task[str(task_id)] = {
            "groups": len(subset), "strong_fraction": sum(abs(r["tau"]) > 0.4 for r in subset) / len(subset),
            "anchors": sum(r["strong_anchor"] for r in subset),
            "nuisances": sum(r["strong_nuisance"] for r in subset),
        }
    contradictory = sum(
        (-1 in row["paired_differences"] and 1 in row["paired_differences"]) for row in rows
    )
    summary = {
        "groups": len(rows), "snapshots": len(specs_by_snapshot),
        "tau_distribution": dict(sorted(Counter(row["tau"] for row in rows).items())),
        "strong_effect_groups": len(strong), "strong_effect_fraction": len(strong) / len(rows),
        "strong_anchors": len(anchors), "strong_anchor_fraction": len(anchors) / len(rows),
        "strong_nuisances": len(nuisances), "strong_nuisance_fraction": len(nuisances) / len(rows),
        "contradictory_seed_sign_groups": contradictory,
        "attention_vs_abs_tau_spearman": float(rho.statistic), "attention_p_value": float(rho.pvalue),
        "by_task": by_task,
        "by_phase": {phase: {
            "groups": sum(r["phase"] == phase for r in rows),
            "strong": sum(r["phase"] == phase and abs(r["tau"]) > 0.4 for r in rows),
            "anchors": sum(r["phase"] == phase and r["strong_anchor"] for r in rows),
            "nuisances": sum(r["phase"] == phase and r["strong_nuisance"] for r in rows),
        } for phase in sorted({row["phase"] for row in rows})},
    }
    go = len(strong) / len(rows) >= 0.15 and len(anchors) > 0 and len(nuisances) > 0
    status = "TOKEN_CAUSAL_LABELS_QUALIFIED_FOR_PREDICTOR" if go else "TOKEN_GRANULARITY_NO_GO"
    write_json(artifact / "summary.json", summary)
    write_json(artifact / "decision.json", {
        "status": status, "existence_gate_pass": go,
        "predictor_trained": False,
        "interpretation": "Qualification concerns stable paired closed-loop labels, not a trained selector or guidance benefit.",
    })
    report = f"""# Closed-Loop Token Causal Effect Dataset Qualification

## Locked design

- 3 LIBERO-Spatial development tasks, 15 simulator snapshots per task.
- 8 disjoint singleton visual-token groups per snapshot, spanning attention strata.
- 3 matched Gaussian-noise seeds; one masked replan followed by vanilla control.
- Total: {len(manifest)}/{len(manifest)} rollouts, {len(rows)} causal group labels.
- Attention was proposal metadata only. No predictor was trained.

## Integrity

- Complete snapshots: {completeness['complete_snapshots']}/45.
- Missing/failed/duplicate: 0/0/0.
- Paired state, prefix, rollout seed, and overlapping per-replan noise hashes: PASS.

## Label existence

- Strong effects (`|tau| > 0.4`): {len(strong)}/{len(rows)} ({len(strong)/len(rows):.1%}).
- Strong anchors (`tau < -0.4`): {len(anchors)} ({len(anchors)/len(rows):.1%}).
- Strong nuisances (`tau > 0.4`): {len(nuisances)} ({len(nuisances)/len(rows):.1%}).
- Groups with contradictory nonzero signs across seeds: {contradictory}/{len(rows)}.
- Attention versus real closed-loop `|tau|`: Spearman rho={float(rho.statistic):.3f}, p={float(rho.pvalue):.4g}.

## Task stratification

```json
{json.dumps(by_task, indent=2, sort_keys=True)}
```

## Decision

`{status}`

This result only decides whether singleton token causal labels have enough closed-loop dynamic range to justify a later task-held-out predictor. It does not validate attention as causal attribution and does not show that token guidance improves success.
"""
    (artifact / "report.md").write_text(report)
    audit_paths = [path for path in artifact.rglob("*") if path.is_file() and path.name != "sha256_audit.json"]
    write_json(artifact / "sha256_audit.json", {str(path.relative_to(artifact)): sha256_file(path)
                                                for path in sorted(audit_paths)})
    print(json.dumps({"status": status, **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
