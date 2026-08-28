#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import h5py
import yaml


PHASES = (("early", 0.2), ("middle", 0.5), ("late", 0.8))
FLOW_TIMES = tuple(round(1.0 - 0.1 * i, 1) for i in range(10))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output = (args.output or workspace / "artifacts" / f"coreact_flow_timestep_compatibility_f0_v1_{datetime.now():%Y%m%d_%H%M%S}").resolve()
    if output.exists():
        raise FileExistsError(output)
    for name in ("selection", "confirmation", "logs", "status", "figures"):
        (output / name).mkdir(parents=True, exist_ok=True)

    import sys
    sys.path.insert(0, str(workspace / "LIBERO"))
    from libero.libero import benchmark

    source_artifact = workspace / "artifacts/coreact_trained_weak_local_finetune_v1_20260812_121522"
    pair_lock = yaml.safe_load((source_artifact / "trained_weak_pair.lock.yaml").read_text())
    if pair_lock["strong"]["step"] != 15000 or pair_lock["weak"]["step"] != 10000:
        raise RuntimeError("the locked pair is not Strong=15k / Weak=10k")

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    demo_root = workspace / "LIBERO/libero/datasets/libero_spatial"
    states = []
    sources = []
    for task_id in range(10):
        language = suite.get_task(task_id).language
        demo_path = demo_root / (language.replace(" ", "_") + "_demo.hdf5")
        if not demo_path.exists():
            raise FileNotFoundError(demo_path)
        with h5py.File(demo_path, "r") as handle:
            demos = sorted(handle["data"], key=lambda value: int(value.split("_")[-1]))
            if len(demos) != 50:
                raise RuntimeError(f"task {task_id}: expected 50 demonstrations, got {len(demos)}")
            lengths = []
            for ordinal, demo_id in enumerate(demos):
                episode_length = len(handle["data"][demo_id]["actions"])
                if episode_length < 2:
                    raise RuntimeError(f"task {task_id}/{demo_id}: too short")
                split = "selection" if ordinal % 2 == 0 else "confirmation"
                split_rank = ordinal // 2
                phase, progress = PHASES[split_rank % len(PHASES)]
                frame = math.floor((episode_length - 1) * progress)
                state_id = f"task{task_id:02d}__demo{ordinal:02d}__frame{frame:04d}"
                states.append({
                    "state_id": state_id,
                    "split": split,
                    "task_id": task_id,
                    "language": language,
                    "demo_id": demo_id,
                    "demo_ordinal": ordinal,
                    "episode_length": episode_length,
                    "phase": phase,
                    "target_progress": progress,
                    "resolved_frame": frame,
                    "demo_path": str(demo_path),
                    "noise_seeds": [202608150000 + task_id * 10000 + ordinal * 10 + i for i in range(3)],
                })
                lengths.append(episode_length)
        sources.append({
            "task_id": task_id,
            "path": str(demo_path),
            "sha256": sha256(demo_path),
            "bytes": demo_path.stat().st_size,
            "demo_count": 50,
            "min_length": min(lengths),
            "max_length": max(lengths),
        })

    split_counts = {
        split: sum(row["split"] == split for row in states)
        for split in ("selection", "confirmation")
    }
    if split_counts != {"selection": 250, "confirmation": 250}:
        raise RuntimeError(split_counts)

    protocol = {
        "experiment_name": "coreact_flow_timestep_compatibility_f0_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "stage": "selection_locked_confirmation_sealed_no_closed_loop",
        "scientific_question": "Does vStrong-vWeak have flow-timestep-dependent supervised compatibility?",
        "suite": "libero_spatial",
        "tasks": list(range(10)),
        "states_per_task": 50,
        "split_rule": "even demo ordinal=selection; odd demo ordinal=confirmation; each split rank cycles progress 0.2/0.5/0.8",
        "split_counts": split_counts,
        "noise_seeds_per_state": 3,
        "flow_times": list(FLOW_TIMES),
        "noise_order": "step 0/t=1.0 highest noise; step 9/t=0.1 lowest noise",
        "flow_pair": {"x_t": "t*epsilon + (1-t)*x0", "target": "epsilon-x0"},
        "pair": {
            "source_artifact": str(source_artifact),
            "pair_lock_sha256": sha256(source_artifact / "trained_weak_pair.lock.yaml"),
            "strong_step": 15000,
            "weak_step": 10000,
        },
        "guidance": {
            "lambda": 0.5,
            "trust_region_kappa": 0.25,
            "raw": "vS + 0.5*(vS-vW)",
            "applied": "vS + clip(0.5*(vS-vW), norm_limit=0.25*norm(vS))",
        },
        "selection_gate": {
            "candidate": "contiguous suffix adjacent to lowest-noise end, length >=2",
            "pooled_g_positive_min": 0.60,
            "state_cluster_ci_lower_min": 0.50,
            "tasks_at_or_above_half_min": 7,
            "gain_over_full_min_pp": 10.0,
            "gain_over_matched_high_noise_min_pp": 10.0,
            "tie_rule": "highest pooled P(G>0), then shorter segment, then deterministic start index",
        },
        "confirmation_gate": {
            "pooled_g_positive_min": 0.60,
            "state_cluster_ci_lower_min": 0.50,
            "tasks_above_half_min": 7,
            "must_exceed_matched_high_noise": True,
            "applied_delta_mse_positive_min": 0.55,
        },
        "bootstrap": {"unit": "state", "task_stratified": True, "replicates": 10000, "seed": 20260815},
        "prohibited": [
            "closed-loop rollout", "lambda tuning", "Weak reselection", "entropy gating",
            "token guidance", "structural Weak", "task-specific windows", "dimension-specific gating",
            "noncontiguous timestep subsets",
        ],
        "sources": sources,
    }
    (output / "protocol.lock.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False, allow_unicode=False))
    (output / "state_manifest.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in states))
    (output / "source_audit.json").write_text(json.dumps({"sources": sources}, indent=2, sort_keys=True) + "\n")
    (output / "status/current.json").write_text(json.dumps({"stage": "PREPARED", **split_counts}, indent=2) + "\n")
    (output / "decision.json").write_text(json.dumps({"decision": "PENDING_SELECTION"}, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
