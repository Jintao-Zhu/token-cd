#!/usr/bin/env python3
"""Resume the frozen OpenVLA OOD study on GPUs 1-3 with nine workers.

Complete task/condition/seed triplets remain untouched. Any incomplete triplet
is archived and rerun as a whole on one new GPU so its three arms retain exact
RGB pairing after the GPU reassignment.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
ART = ROOT / "artifacts/openvla_ood_robustness_v3"
ARCHIVE = ART / "rebalance_archive_20261008"
SNAP = ROOT / "artifacts/vanilla_recon_shr_canonical_0_299_v2"
RUNNER = ROOT / "research/semantic_token_cd/openvla_ood_robustness_rollout.py"
TASKS = (
    "google_robot_open_drawer", "google_robot_close_drawer",
    "google_robot_pick_coke_can", "google_robot_move_near",
)
ARMS = ("vanilla", "pcd", "ours")
CONDITIONS = ("clean", "distractor", "texture", "illumination", "viewpoint", "composite")


def main() -> None:
    sessions = subprocess.run(["tmux", "ls"], text=True, capture_output=True).stdout
    if any(line.startswith("ood_formal_") for line in sessions.splitlines()):
        raise SystemExit("Old OOD tmux workers are still alive; refusing to rebalance")
    for i in range(9):
        if subprocess.run(["tmux", "has-session", "-t", f"ood_r123_{i:02d}"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            raise SystemExit(f"Rebalance tmux session already exists: ood_r123_{i:02d}")

    grouped: dict[tuple[str, str, int], dict[str, Path]] = defaultdict(dict)
    for path in (ART / "episodes").rglob("episode_*_summary.json"):
        row = json.loads(path.read_text())
        if row.get("protocol_id") != "OPENVLA_OOD_ROBUSTNESS_200_299_V3_ASSET_DISTRACTORS":
            raise RuntimeError(f"foreign summary in formal tree: {path}")
        grouped[(row["task"], row["condition"], int(row["seed"]))][row["arm"]] = path

    partial = [(key, arms) for key, arms in grouped.items() if set(arms) != set(ARMS)]
    # Preserve the old one- or two-arm records outside the formal episodes tree,
    # then rerun all three arms for that state/condition on one physical GPU.
    archived = 0
    for (task, condition, seed), arm_paths in partial:
        for arm, source in arm_paths.items():
            destination = ARCHIVE / "episodes" / task / condition / arm / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise RuntimeError(f"archive destination already exists: {destination}")
            shutil.move(str(source), str(destination))
            archived += 1

    # Nine workers, three per GPU. Seed subsets are greedily balanced by the
    # number of missing arm-episodes, while each task/seed belongs to one worker.
    plan = [
        ("google_robot_open_drawer", 1, 1),
        ("google_robot_open_drawer", 2, 1),
        ("google_robot_open_drawer", 3, 1),
        ("google_robot_close_drawer", 1, 1),
        ("google_robot_close_drawer", 2, 1),
        ("google_robot_close_drawer", 3, 1),
        ("google_robot_pick_coke_can", 2, 1),
        ("google_robot_move_near", 1, 1),
        ("google_robot_move_near", 3, 1),
    ]
    workers = []
    for task in TASKS:
        indices = [i for i, (t, _, _) in enumerate(plan) if t == task]
        loads = [0] * len(indices)
        assignments = [[] for _ in indices]
        weights = {}
        for seed in range(200, 300):
            existing = grouped.get((task, "clean", seed), {})
            # Use all conditions, not just clean, to estimate remaining work.
            missing = 0
            for condition in CONDITIONS:
                present = grouped.get((task, condition, seed), {})
                if (task, condition, seed) in {k for k, _ in partial}:
                    present = {}
                missing += len(set(ARMS) - set(present))
            weights[seed] = missing
        for seed in sorted(weights, key=lambda s: (-weights[s], s)):
            slot = min(range(len(indices)), key=lambda j: (loads[j], j))
            assignments[slot].append(seed)
            loads[slot] += weights[seed]
        for slot, idx in enumerate(indices):
            t, gpu, _ = plan[idx]
            seeds = sorted(assignments[slot])
            worker = len(workers)
            workers.append({"session": f"ood_r123_{worker:02d}", "worker_id": f"resume_r123_{worker:02d}_gpu{gpu}",
                            "gpu": gpu, "task": t, "seeds": seeds,
                            "estimated_remaining_episode_arms": loads[slot]})

    # Assert each task's seeds are assigned exactly once and each card gets 3 jobs.
    for task in TASKS:
        assigned = [s for w in workers if w["task"] == task for s in w["seeds"]]
        if sorted(assigned) != list(range(200, 300)):
            raise RuntimeError(f"bad seed partition for {task}")
    gpu_counts = {gpu: sum(w["gpu"] == gpu for w in workers) for gpu in (1, 2, 3)}
    if gpu_counts != {1: 3, 2: 3, 3: 3}:
        raise RuntimeError(f"bad GPU worker counts: {gpu_counts}")

    manifest = {"protocol_id": "OPENVLA_OOD_ROBUSTNESS_200_299_V3_ASSET_DISTRACTORS",
                "rebalanced_at": "2026-10-08", "gpus": [1, 2, 3], "workers": workers,
                "archived_partial_summary_rows": archived,
                "note": "Complete arm triplets kept; partial triplets archived and rerun together for exact RGB pairing."}
    manifest_path = ART / "REBALANCE_MANIFEST_20261008.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (ART / "logs").mkdir(exist_ok=True)
    for w in workers:
        seeds_arg = ",".join(str(s) for s in w["seeds"])
        log = ART / "logs" / f"{w['worker_id']}.log"
        command = (f"source scripts/activate_simpler_local.sh && "
                   f"export CUDA_VISIBLE_DEVICES={w['gpu']} && "
                   f"exec \"$SIMPLER_ENV/bin/python\" research/semantic_token_cd/openvla_ood_robustness_rollout.py "
                   f"--artifact {ART} --snapshot-artifact {SNAP} --task {w['task']} "
                   f"--seeds {seeds_arg} --gpu {w['gpu']} --worker-id {w['worker_id']} "
                   f"> {log} 2>&1")
        subprocess.run(["tmux", "new-session", "-d", "-s", w["session"], "-c", str(ROOT),
                        "bash", "-lc", command], check=True)
        print(f"started {w['session']} GPU{w['gpu']} {w['task']} seeds={len(w['seeds'])} "
              f"pending_episode_arms~{w['estimated_remaining_episode_arms']}")
    print(f"archived {archived} partial rows; manifest={manifest_path}")


if __name__ == "__main__":
    main()
