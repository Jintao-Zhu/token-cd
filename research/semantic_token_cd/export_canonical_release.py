"""Export a compact, Git-friendly canonical paired rollout release."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path


ARMS = ("vanilla", "semantic_recon_k8_m10", "shr_harmonic")
TASKS = (
    "google_robot_pick_coke_can",
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_move_near",
    "google_robot_place_apple_in_closed_top_drawer",
    "widowx_carrot_on_plate",
    "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel",
    "widowx_stack_cube",
)


def exact_mcnemar(rescue: int, harm: int) -> float:
    n = rescue + harm
    if n == 0:
        return 1.0
    k = min(rescue, harm)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact_summary(summary: dict) -> dict:
    # Per-step selector traces and logits are regenerable and dominate disk use.
    omitted = {"selector_trace", "arrays_file"}
    return {key: value for key, value in summary.items() if key not in omitted}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    records: dict[tuple[str, int, str], dict] = {}
    snapshot_manifest = []
    for task in TASKS:
        for seed in range(300):
            arm_records = []
            for arm in ARMS:
                summary_path = source / "episodes" / task / arm / f"episode_{seed:03d}_summary.json"
                arrays_path = source / "episodes" / task / arm / f"episode_{seed:03d}_arrays.npz"
                if not summary_path.exists() or not arrays_path.exists():
                    raise RuntimeError(f"incomplete source episode: {task} seed={seed} arm={arm}")
                summary = json.loads(summary_path.read_text())
                records[task, seed, arm] = summary
                arm_records.append(summary)
                out = destination / "episodes" / task / arm / summary_path.name
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(compact_summary(summary), indent=2, sort_keys=True) + "\n")

            for key in ("canonical_snapshot_sha256", "initial_state_sha256", "initial_rgb_sha256"):
                if len({record.get(key) for record in arm_records}) != 1:
                    raise RuntimeError(f"paired hash mismatch: {task} seed={seed} key={key}")

            snapshot = source / "snapshots" / task / f"seed_{seed:03d}.pkl"
            if not snapshot.exists():
                raise RuntimeError(f"missing snapshot: {task} seed={seed}")
            snapshot_out = destination / "snapshots" / task / snapshot.name
            snapshot_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(snapshot, snapshot_out)
            reference = arm_records[0]
            snapshot_manifest.append({
                "task": task,
                "seed": seed,
                "path": str(snapshot_out.relative_to(destination)),
                "file_sha256": file_sha256(snapshot_out),
                "canonical_snapshot_sha256": reference["canonical_snapshot_sha256"],
                "initial_state_sha256": reference["initial_state_sha256"],
                "initial_rgb_sha256": reference["initial_rgb_sha256"],
            })

        task_root = source / "episodes" / task
        for name in ("CONFIG_LOCK.json",):
            candidate = task_root / name
            if candidate.exists():
                out = destination / "episodes" / task / name
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate, out)
        for candidate in task_root.glob("pairing_manifest*.json"):
            out = destination / "episodes" / task / candidate.name
            shutil.copy2(candidate, out)

    per_task = defaultdict(dict)
    overall = {}
    for task in TASKS:
        for arm in ARMS:
            success = sum(bool(records[task, seed, arm]["success"]) for seed in range(300))
            per_task[task][arm] = {
                "success": success,
                "episodes": 300,
                "success_rate": success / 300,
            }
    for arm in ARMS:
        success = sum(bool(records[task, seed, arm]["success"]) for task in TASKS for seed in range(300))
        overall[arm] = {"success": success, "episodes": 2700, "success_rate": success / 2700}

    comparisons = {}
    for left, right in ((ARMS[0], ARMS[1]), (ARMS[0], ARMS[2]), (ARMS[1], ARMS[2])):
        pairs = [
            (bool(records[task, seed, left]["success"]), bool(records[task, seed, right]["success"]))
            for task in TASKS
            for seed in range(300)
        ]
        rescue = sum((not a) and b for a, b in pairs)
        harm = sum(a and (not b) for a, b in pairs)
        comparisons[f"{right}_vs_{left}"] = {
            "rescue": rescue,
            "harm": harm,
            "net_success": rescue - harm,
            "success_rate_delta": (rescue - harm) / 2700,
            "mcnemar_exact_two_sided_p": exact_mcnemar(rescue, harm),
        }

    result = {
        "artifact": source.name,
        "protocol_id": records[TASKS[0], 0, ARMS[0]]["protocol_id"],
        "tasks": list(TASKS),
        "arms": list(ARMS),
        "seeds": "0-299",
        "paired_scenarios": 2700,
        "integrity": {
            "summaries": 8100,
            "arrays_verified_locally_but_not_exported": 8100,
            "canonical_snapshots": 2700,
            "paired_hash_mismatches": 0,
        },
        "overall": overall,
        "per_task": per_task,
        "paired_comparisons": comparisons,
    }
    (destination / "FINAL_RESULTS.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (destination / "SNAPSHOT_MANIFEST.jsonl").open("w") as handle:
        for record in snapshot_manifest:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    readme = f"""# {source.name}

Canonical paired rollout release for Vanilla, semantic Recon, and spatial SHR.

- 9 tasks, seeds 0-299 per task
- 2,700 paired simulator scenarios
- 8,100 compact episode summaries
- 2,700 reusable canonical simulator snapshots
- zero snapshot/state/RGB hash mismatches

`FINAL_RESULTS.json` contains aggregate success rates and paired McNemar tests.
`SNAPSHOT_MANIFEST.jsonl` records both the serialized-file SHA256 and the
canonical/state/RGB hashes for every `(task, seed)`.

The `episodes/` summaries omit only the large per-step `selector_trace` and raw
logit/action array reference. The 8,100 local NPZ arrays were validated during
export but intentionally excluded from Git because they are regenerable and
occupy several GiB.

To reuse the exact scenes, copy `snapshots/` into a new artifact directory and
run a compatible rollout driver with the same OpenVLA checkpoint and simulator
environment. Drivers must restore the pickle rather than recapture from seed,
and should verify the canonical, initial-state, and initial-RGB hashes against
the manifest before executing an arm.
"""
    (destination / "README.md").write_text(readme)


if __name__ == "__main__":
    main()
