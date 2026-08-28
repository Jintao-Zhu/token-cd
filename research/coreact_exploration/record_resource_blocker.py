#!/usr/bin/env python3
"""Record the verified second-suite resource blocker without touching held-out data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import socket
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    dataset = workspace / "counterfactual-flow-vla/pi05_svcpd_p0/dataset_cache/HuggingFaceVLA_libero"
    rows_per_task = Counter()
    frames_per_episode = defaultdict(list)
    task_per_episode = {}
    shards = sorted((dataset / "data/chunk-000").glob("*.parquet"))
    for shard in shards:
        table = pq.read_table(shard, columns=["task_index", "episode_index", "frame_index"])
        for task, episode, frame in zip(
            table["task_index"].to_pylist(),
            table["episode_index"].to_pylist(),
            table["frame_index"].to_pylist(),
            strict=True,
        ):
            rows_per_task[task] += 1
            frames_per_episode[episode].append(frame)
            task_per_episode[episode] = task
    complete_per_task = Counter()
    for episode, frames in frames_per_episode.items():
        if sorted(frames) == list(range(max(frames) + 1)):
            complete_per_task[task_per_episode[episode]] += 1
    object_episodes = sum(complete_per_task[task] for task in range(20, 30))
    non_object_episodes = sum(count for task, count in complete_per_task.items() if task not in range(20, 30))
    audit = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "dataset_repo": "HuggingFaceVLA/libero",
        "evaluation_snapshot_revision": "86958911c0f959db2bbbdb107eb3e17c5f9c798e",
        "local_shard_count": len(shards),
        "local_shard_indices": [int(path.stem.split("-")[1]) for path in shards],
        "rows_per_task": dict(sorted(rows_per_task.items())),
        "complete_episodes_per_task": dict(sorted(complete_per_task.items())),
        "eligible_libero_object_complete_episodes": object_episodes,
        "non_object_complete_episodes": non_object_episodes,
        "non_object_assessment": "boundary fragments only; insufficient for a second suite",
        "local_cache_search": {
            "dataset_cache_found": False,
            "model_caches_only": [
                "models--lerobot--pi05_libero_finetuned",
                "models--lerobot--smolvla_libero",
            ],
        },
        "network_attempts": [
            {
                "operation": "HfApi.list_repo_tree recursive at fixed revision",
                "result": "timeout without file listing",
            },
            {
                "operation": "curl fixed-revision file-210.parquet",
                "url": "https://huggingface.co/datasets/HuggingFaceVLA/libero/resolve/86958911c0f959db2bbbdb107eb3e17c5f9c798e/data/chunk-000/file-210.parquet",
                "result": "curl exit 28: failed to connect to huggingface.co:443 after 5205 ms",
            },
        ],
        "second_verified_suite_available": False,
        "heldout_created_or_viewed": False,
    }
    write_json(artifact / "resource_audit.json", audit)
    with (artifact / "missingness.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scope", "expected_states", "actual_states", "missing_states", "missing_fraction", "reason"])
        writer.writerow(["two-suite heldout", 120, 0, 120, 1.0, "second verified suite unavailable"])
    decision = {
        "status": "INCONCLUSIVE_DATA_OR_RESOURCES",
        "development_integrity_status": "PASS",
        "development_states_passed": 24,
        "heldout_created": False,
        "heldout_viewed": False,
        "heldout_missing_fraction": 1.0,
        "cross_suite_proceed_eligible": False,
        "reason": "A second formally verified task suite at the fixed evaluation revision is unavailable locally and Hugging Face is unreachable.",
        "strongest_supported_conclusion": "The frozen fp32 probe, prefix map, attention recorder, intervention builder, and calibration means passed single-suite development integrity checks.",
        "ranking_hypothesis_evaluated_on_heldout": False,
    }
    write_json(artifact / "decision.json", decision)
    environment = json.loads((artifact / "environment.json").read_text())
    checkpoint = (
        workspace
        / "task1/.hf-cache/hub/models--lerobot--smolvla_libero/snapshots"
        / "31d453f7edd78c839a8bbc39744a292686daf0de"
    )
    environment.update(
        {
            "date": datetime.now().astimezone().date().isoformat(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "transformers": "5.5.4",
            "pyarrow": "24.0.0",
            "lerobot": "0.6.1",
            "uv_available": shutil.which("uv") is not None,
            "checkpoint": {
                "repo": "lerobot/smolvla_libero",
                "revision": "31d453f7edd78c839a8bbc39744a292686daf0de",
                "config_sha256": sha256(checkpoint / "config.json"),
                "weights_sha256": sha256(checkpoint / "model.safetensors"),
                "preprocessor_sha256": sha256(checkpoint / "policy_preprocessor.json"),
            },
            "dataset": {
                "repo": "HuggingFaceVLA/libero",
                "evaluation_snapshot_revision": "86958911c0f959db2bbbdb107eb3e17c5f9c798e",
                "claimed_as_training_revision": False,
                "download_manifest_sha256": sha256(
                    workspace
                    / "counterfactual-flow-vla/pi05_svcpd_p0/verified_calibration_download_manifest.json"
                ),
            },
        }
    )
    write_json(artifact / "environment.json", environment)
    write_json(
        artifact / "test_report.json",
        {
            "commands": [
                {
                    "command": "python -m pytest tests/coreact_exploration -q",
                    "exit_code": 0,
                    "result": "12 passed",
                },
                {
                    "command": "python -m pytest tests/coreact_exploration lerobot/tests/policies/smolvla/test_smolvla_rtc.py -q",
                    "exit_code": 0,
                    "result": "14 passed, 3 skipped",
                },
                {"command": "git -C lerobot diff --check", "exit_code": 0, "result": "clean"},
                {
                    "command": "sha256sum -c artifact_sha256.txt",
                    "exit_code": 0,
                    "result": "all files passed before final environment/test-report refresh",
                },
            ],
            "ruff": "not available; not installed",
        },
    )
    report = f"""# CoreAct-SC Offline Exploration Report

## 1. Research Question And Locked Protocol

The preregistered question is whether cheap token ranking proxies predict grouped intervention magnitude and distinguish anchor-like from nuisance-candidate signs. The single-suite development protocol is in `protocol.yaml`; implementation and thresholds are frozen in `protocol.lock.yaml`.

## 2. Checkpoint, Dataset, And Split

- Checkpoint: `lerobot/smolvla_libero@31d453f7edd78c839a8bbc39744a292686daf0de`.
- Evaluation snapshot: `HuggingFaceVLA/libero@86958911c0f959db2bbbdb107eb3e17c5f9c798e`; this is not claimed as the training revision.
- Formal local data: 98 verified source Parquet shards, indices 211-308.
- Development: 24 states from 24 episodes across 10 LIBERO-Object tasks.
- Mean calibration: 512 states from 256 disjoint episodes.
- Held-out: not created because a second verified suite is unavailable.

## 3. Implementation And Parity Evidence

The probe reuses SmolVLA `embed_prefix`, `embed_suffix`, flow construction, masks, position IDs, expert forward, and `action_out_proj`. The policy stayed frozen in `eval()` at fp32. The attention hook is default-off and immediately detaches action-query-to-prefix probabilities to CPU.

## 4. Integrity Gates

All 24 development states passed. Determinism and same-shape no-op discrepancies were 0; maximum forward loss discrepancy was `1.862645149230957e-09`; attention row-sum discrepancy was `3.5762786865234375e-07`; masked probability was 0. Serial/batched velocity discrepancy was `1.1831521987915039e-05`, below the development-locked `1e-4` batch tolerance. See `integrity_report.md` and raw `development_integrity_rows.jsonl`.

## 5. Missingness

The required two-suite held-out split has 120/120 states missing (100%). This exceeds the 5% gate. See `missingness.csv` and `resource_audit.json`.

## 6. Primary Result

Not computed. No held-out visual groups were selected or evaluated.

## 7. Secondary And Sensitivity Results

Not computed. Development effects were used only to lock numerical/effect thresholds and cannot be reported as held-out evidence.

## 8. Representative Cases And Counterexamples

Not selected because held-out results do not exist. Diagnostic controls changed velocity when all language or one camera was zeroed and when a real alternative instruction was used; these only validate probe placement.

## 9. Strongest Supported Conclusion

The frozen fp32 instrumentation and single-suite development pipeline pass their integrity gates. No conclusion about ranking predictive power is supported yet.

## 10. Unsupported Conclusions

This artifact does not show that attention is causal attribution, that any token is a real causal core, that CoreAct improves success rate, or that vision shortcuts are mitigated.

## 11. Decision And Next Step

`INCONCLUSIVE_DATA_OR_RESOURCES`. Obtain and checksum a complete second LIBERO suite from the same fixed evaluation revision, then create a new confirmation artifact and held-out split. Do not reuse development rows in held-out.

## 12. Exact Reproduction Commands

```bash
PYTHONPATH=/data/docker/dev_zjt/data/code/lerobot/src:/data/docker/dev_zjt/data/code \\
  /data/docker/dev_zjt/data/code/task1/.conda-envs/flow-vla/bin/python -m pytest tests/coreact_exploration -q

HF_HOME=/data/docker/dev_zjt/data/code/task1/.hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
PYTHONPATH=/data/docker/dev_zjt/data/code/lerobot/src:/data/docker/dev_zjt/data/code \\
  /data/docker/dev_zjt/data/code/task1/.conda-envs/flow-vla/bin/python \\
  research/coreact_exploration/run_development_gates.py \\
  --workspace /data/docker/dev_zjt/data/code --artifact {artifact}
```

`uv` was unavailable and no dependency was installed or upgraded. The completed commands and environment are recorded in this artifact.
"""
    (artifact / "exploratory_report.md").write_text(report)
    checksum_path = artifact / "artifact_sha256.txt"
    files = sorted(path for path in artifact.rglob("*") if path.is_file() and path != checksum_path)
    checksum_path.write_text("".join(f"{sha256(path)}  {path.relative_to(artifact)}\n" for path in files))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
