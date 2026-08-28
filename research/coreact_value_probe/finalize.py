#!/usr/bin/env python3
"""Render the qualified value-probe report and final SHA-256 audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def fmt(value) -> str:
    return "NA" if value is None else f"{value:.3f}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--artifact", type=Path, required=True); args = parser.parse_args()
    artifact = args.artifact.resolve()
    integrity = json.loads((artifact / "integrity_report.json").read_text())
    if not integrity["pass"]:
        raise RuntimeError("cannot finalize a failed capture integrity gate")
    summary = json.loads((artifact / "probe_summary.json").read_text())
    decision = json.loads((artifact / "decision.json").read_text())
    lines = [
        "# Success-Value Token Causal Probe: Single-State Qualification",
        "",
        "## Question",
        "",
        "Can a lightweight probe over frozen SmolVLA state representations predict the final success of a vanilla episode on a task excluded from fitting? This stage does not perform token masking or guidance.",
        "",
        "## Locked design",
        "",
        "- Train tasks: 0, 1, 2, 5, 6, 7, 9; validation task: 3; held-out test task: 4.",
        "- Representation: final-normalized contextual prefix hidden states, separately mean-pooled over camera1, camera2, valid language, and state, then concatenated.",
        "- Primary model: linear logistic probe. The 64-unit MLP is a secondary capacity diagnostic and cannot override the primary gate.",
        "- Statistical unit for uncertainty: episode, not replan state.",
        "",
        "## Capture integrity",
        "",
        f"- Complete deterministic action replays: {integrity['actual']}/{integrity['expected']}.",
        f"- Prepared-input byte-hash mismatches: {integrity['initial_prepared_hash_mismatches']}/450. Simulator hashes and terminal success remained mandatory exact matches.",
        f"- Captured replan states: {summary['state_count']}.",
        "",
        "## Results",
        "",
        "| Probe | Split | AUROC | AP | Prevalence | Brier | ECE-10 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ("linear", "mlp"):
        for split in ("train", "validation", "heldout_test"):
            metric = summary["models"][model][split]
            lines.append(f"| {model} | {split} | {fmt(metric['auroc'])} | {fmt(metric['ap'])} | {fmt(metric['prevalence'])} | {fmt(metric['brier'])} | {fmt(metric['ece10'])} |")
    primary = summary["models"]["linear"]["heldout_test"]
    prediction_rows = list(csv.DictReader((artifact / "probe_predictions.csv").open()))
    heldout_rows = [row for row in prediction_rows if row["split"] == "heldout_test"]
    support = {}
    for phase in sorted({row["phase"] for row in heldout_rows}):
        episode_labels = {
            row["capture_id"]: int(row["label"])
            for row in heldout_rows
            if row["phase"] == phase
        }
        positive = sum(episode_labels.values())
        support[phase] = (len(episode_labels), positive, len(episode_labels) - positive)
    lines += [
        "",
        f"Primary held-out linear AUROC episode-bootstrap 95% CI: [{fmt(primary['bootstrap']['ci95'][0])}, {fmt(primary['bootstrap']['ci95'][1])}].",
        f"Language-only control AUROC: {fmt(primary['language_only_control']['auroc'])}.",
        f"Contextual language-token-position pool diagnostic AUROC: {fmt(primary['contextual_language_position_pool_diagnostic']['auroc'])}; this is not language-only because it has attended to images/state.",
        "",
        "## Phase results",
        "",
        "Phase metrics are descriptive when episode support does not meet the locked 10-positive/10-negative rule.",
        "",
        "| Phase | AUROC | AP | States | Episodes (+/-) | Qualified support |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for phase, metric in summary["models"]["linear"]["heldout_test"]["phase_metrics"].items():
        episodes, positive, negative = support[phase]
        qualified = positive >= 10 and negative >= 10
        lines.append(f"| {phase} | {fmt(metric['auroc'])} | {fmt(metric['ap'])} | {metric['n']} | {episodes} ({positive}/{negative}) | {'yes' if qualified else 'no'} |")
    lines += [
        "",
        "## Gate",
        "",
    ]
    for check, passed in summary["qualification"]["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{check}`")
    lines += [
        "",
        f"**Decision: {decision['decision']}**",
        "",
        "No closed-loop token-utility claim is supported by this probe qualification alone. Even on PASS, the next required step is a separately locked new-state causal calibration with strong-positive, strong-negative, near-zero, and vanilla conditions. Guidance remains prohibited.",
        "",
        "## Limitations",
        "",
        "- This is a development replication over prior vanilla outcomes, not an independent confirmation.",
        "- LIBERO-Spatial task success rates differ sharply; task 5 supplies most training failures. The task-4 test prevents direct task memorization but does not establish cross-suite generalization.",
        "- A small number of exact simulator-state replays may have non-bit-identical rendered/preprocessed inputs; these are explicitly counted above.",
        f"- Held-out linear calibration is imperfect (ECE-10 {fmt(primary['ece10'])}); signed token value changes require real rollout calibration and must not be interpreted as calibrated success-probability deltas yet.",
    ]
    (artifact / "report.md").write_text("\n".join(lines) + "\n")
    paths = sorted(path for path in artifact.rglob("*") if path.is_file() and path.name != "sha256.audit")
    with (artifact / "sha256.audit").open("w") as stream:
        for path in paths:
            stream.write(f"{sha256(path)}  {path.relative_to(artifact)}\n")


if __name__ == "__main__":
    main()
