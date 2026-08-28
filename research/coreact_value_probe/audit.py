#!/usr/bin/env python3
"""Mandatory completeness and replay-integrity audit for value-probe captures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    manifest = read_jsonl(artifact / "capture_manifest.jsonl")
    failures, rows = [], []
    if len(manifest) != 450 or len({row["capture_id"] for row in manifest}) != 450:
        failures.append("manifest must contain 450 unique captures")
    for spec in manifest:
        stem = artifact / "captures" / spec["capture_id"]
        if not stem.with_suffix(".json").exists() or not stem.with_suffix(".npz").exists():
            failures.append(f"missing capture {spec['capture_id']}")
            continue
        audit = json.loads(stem.with_suffix(".json").read_text())
        data = np.load(stem.with_suffix(".npz"))
        checks = {
            "complete": audit.get("status") == "complete",
            "success_match": audit.get("success") == spec["expected_success"],
            "sim_hash_match": audit.get("initial_sim_state_sha256") == spec["expected_initial_sim_state_sha256"],
            "input_hash_recorded": bool(audit.get("initial_prepared_sha256")),
            "state_count_match": audit.get("states") == len(data["representation"]),
            "finite": bool(np.isfinite(data["representation"]).all()),
            "label_match": bool(np.all(data["label"] == spec["expected_success"])),
            "two_real_camera_pools": bool(audit.get("pool_indices", {}).get("camera1")) and bool(audit.get("pool_indices", {}).get("camera2")),
        }
        if not all(checks.values()):
            failures.append(f"{spec['capture_id']}: {[key for key, value in checks.items() if not value]}")
        rows.append({"capture_id": spec["capture_id"], "task_id": spec["task_id"], "split": spec["split"], "states": audit["states"], "success": int(audit["success"]), "input_hash_match": audit.get("initial_prepared_sha256") == spec["expected_initial_prepared_sha256"], "npz_sha256": file_sha256(stem.with_suffix(".npz")), **checks})
    with (artifact / "capture_audit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["capture_id"])
        writer.writeheader(); writer.writerows(rows)
    input_mismatches = sum(not row.get("input_hash_match", False) for row in rows)
    if input_mismatches / 450 > 0.01:
        failures.append(f"initial prepared byte-hash mismatch exceeds 1%: {input_mismatches}/450")
    passed = not failures
    report = {"gate": "value_probe_capture_integrity", "pass": passed, "expected": 450, "actual": len(rows), "initial_prepared_hash_mismatches": input_mismatches, "failures": failures}
    (artifact / "integrity_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (artifact / "integrity_report.md").write_text(
        "# Success-Value Probe Capture Integrity\n\n"
        f"- Status: **{'PASS' if passed else 'FAIL'}**\n"
        f"- Complete captures: {len(rows)}/450\n"
        "- Unit: one deterministic vanilla episode replay\n"
        "- Checks: initial simulator hash, released preprocessor hash, terminal success, finite latent, labels, and two real camera pools\n"
        f"- Failures: {json.dumps(failures, indent=2)}\n"
    )
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
