#!/usr/bin/env python3
"""Create the SQLite job queue for smoke or formal rollout."""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from research.semantic_token_cd.task_conditioned_contrast_protocol import (
    ARTIFACT,
    FORMAL_SEEDS,
    TASKS,
)


def create(path: Path, jobs: list[tuple[str, int, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    try:
        con.execute(
            "CREATE TABLE IF NOT EXISTS jobs ("
            "job_id TEXT PRIMARY KEY, task TEXT NOT NULL, seed INTEGER NOT NULL, arm TEXT NOT NULL, "
            "status TEXT NOT NULL, worker TEXT, attempt INTEGER NOT NULL DEFAULT 0, heartbeat INTEGER, "
            "start_time INTEGER, finish_time INTEGER, error TEXT, config_hash TEXT, "
            "UNIQUE(task, seed, arm))"
        )
        for task, seed, arm in jobs:
            job_id = f"{task}__seed_{seed:03d}__{arm}"
            con.execute(
                "INSERT OR IGNORE INTO jobs(job_id, task, seed, arm, status) VALUES(?,?,?,?, 'pending')",
                (job_id, task, seed, arm),
            )
        con.commit()
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--mode", choices=["smoke", "formal"], required=True)
    args = parser.parse_args()
    if args.mode == "smoke":
        jobs = [(task, seed, arm) for task in TASKS for seed in (0, 1) for arm in ("C", "C_APC", "T", "T_APC")]
        path = args.artifact / "smoke_jobs.sqlite"
    else:
        jobs = [(task, seed, arm) for task in TASKS for seed in FORMAL_SEEDS for arm in ("C", "C_APC", "T", "T_APC")]
        path = args.artifact / "jobs.sqlite"
    create(path, jobs)
    print(f"{args.mode}: {len(jobs)} jobs -> {path}")


if __name__ == "__main__":
    main()
