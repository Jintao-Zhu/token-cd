#!/usr/bin/env python3
"""Requeue stale or failed task-conditioned jobs after a worker crash."""
from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--stale-seconds", type=int, default=900)
    parser.add_argument("--max-attempts", type=int, default=2)
    args = parser.parse_args()
    now = int(time.time())
    con = sqlite3.connect(str(args.db), timeout=60)
    try:
        stale = con.execute(
            "SELECT job_id, attempt FROM jobs WHERE task=? AND status='running' "
            "AND COALESCE(heartbeat,0) < ?",
            (args.task, now - args.stale_seconds),
        ).fetchall()
        failed = con.execute(
            "SELECT job_id, attempt FROM jobs WHERE task=? AND status='error'",
            (args.task,),
        ).fetchall()
        requeued = 0
        for job_id, attempt in stale + failed:
            if int(attempt) < args.max_attempts:
                con.execute(
                    "UPDATE jobs SET status='pending', worker=NULL, heartbeat=NULL, "
                    "start_time=NULL, finish_time=NULL, error=NULL WHERE job_id=?",
                    (job_id,),
                )
                requeued += 1
        con.commit()
        status = dict(con.execute(
            "SELECT status, count(*) FROM jobs WHERE task=? GROUP BY status",
            (args.task,),
        ).fetchall())
        print({"task": args.task, "requeued": requeued, "status": status})
    finally:
        con.close()


if __name__ == "__main__":
    main()
