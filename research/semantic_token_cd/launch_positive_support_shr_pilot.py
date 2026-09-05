"""Wait for IC-SHR, then run the 400-episode PSC K10/K20 drawer pilot."""
from __future__ import annotations

import collections
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Job:
    task: str
    k: int
    lo: int
    hi: int
    attempts: int = 0

    @property
    def arm(self): return f"psc_shr_k{self.k}"
    @property
    def name(self): return f"{self.task}_{self.arm}_{self.lo:03d}_{self.hi:03d}"


def complete(root: Path, job: Job) -> bool:
    return all((root / "episodes" / job.task / job.arm / f"episode_{s:03d}_summary.json").exists()
               and (root / "episode_logits" / job.task / job.arm / f"episode_{s:03d}_arrays.npz").exists()
               for s in range(job.lo, job.hi + 1))


def analyze(root: Path) -> None:
    tasks = ("google_robot_close_drawer", "google_robot_open_drawer")
    rows, result = [], {}
    for task in tasks:
        result[task] = {}
        for k in (10, 20):
            records = [json.loads((root / "episodes" / task / f"psc_shr_k{k}" /
                                   f"episode_{s:03d}_summary.json").read_text()) for s in range(100)]
            rescue = sum(x["success"] and not x["shr_success"] for x in records)
            harm = sum(not x["success"] and x["shr_success"] for x in records)
            result[task][f"k{k}"] = {
                "n": 100, "success": sum(x["success"] for x in records),
                "shr_success": sum(x["shr_success"] for x in records),
                "vanilla_success": sum(x["vanilla_success"] for x in records),
                "rescue_vs_shr": rescue, "harm_vs_shr": harm, "net_vs_shr": rescue - harm,
                "episodes_action_changed": sum(x["changed_dimensions"] > 0 for x in records),
                "changed_dimensions": sum(x["changed_dimensions"] for x in records),
                "technical_pass": sum(x["technical_pass"] for x in records),
            }
            rows.extend({"task": task, "k": k, "seed": x["seed"],
                         "vanilla_success": x["vanilla_success"], "shr_success": x["shr_success"],
                         "psc_success": x["success"], "changed_dimensions": x["changed_dimensions"]}
                        for x in records)
    (root / "statistics").mkdir(exist_ok=True); (root / "paired_results").mkdir(exist_ok=True)
    (root / "statistics" / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (root / "paired_results" / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ["# PSC-SHR K10/K20 paired drawer pilot", "",
             "| Task | K | Vanilla | SHR | PSC | Delta pp | Rescue | Harm | Net | Changed episodes |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for task in tasks:
        for k in (10, 20):
            x = result[task][f"k{k}"]
            lines.append(f"| {task.replace('google_robot_', '')} | {k} | {x['vanilla_success']}% | "
                         f"{x['shr_success']}% | {x['success']}% | {x['success']-x['shr_success']:+.1f} | "
                         f"{x['rescue_vs_shr']} | {x['harm_vs_shr']} | {x['net_vs_shr']:+d} | "
                         f"{x['episodes_action_changed']}/100 |")
    (root / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--snapshot-artifact", type=Path, required=True)
    ap.add_argument("--wait-for", type=Path, required=True)
    args = ap.parse_args()
    root, source, marker = args.artifact.resolve(), args.snapshot_artifact.resolve(), args.wait_for.resolve()
    log_dir = root / "rollout_logs"; log_dir.mkdir(parents=True, exist_ok=True)
    while not marker.exists():
        print(json.dumps({"waiting_for": str(marker)}), flush=True); time.sleep(60)
    jobs = collections.deque(Job(task, k, lo, lo + 19)
                             for task in ("google_robot_close_drawer", "google_robot_open_drawer")
                             for k in (10, 20) for lo in range(0, 100, 20))
    slots = [(gpu, slot) for gpu in (0, 1, 2, 3, 4) for slot in range(4)]
    active, failed = {}, []
    repo = Path(__file__).resolve().parents[2]
    rollout = repo / "research/semantic_token_cd/positive_support_shr_rollout.py"
    env = os.environ.copy(); env.update({"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    env["PYTHONPATH"] = f"{repo}:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source:{env.get('PYTHONPATH','')}"
    while jobs or active:
        for gpu, slot in slots:
            key = (gpu, slot)
            if key in active or not jobs: continue
            job = jobs.popleft()
            if complete(root, job): continue
            job.attempts += 1
            handle = (log_dir / f"{job.name}_gpu{gpu}_attempt{job.attempts}.log").open("a")
            cmd = [sys.executable, str(rollout), "--artifact", str(root), "--snapshot-artifact", str(source),
                   "--task", job.task, "--k", str(job.k), "--seeds", f"{job.lo}-{job.hi}",
                   "--gpu", str(gpu), "--worker-id", f"gpu{gpu}_slot{slot}_{job.name}"]
            proc = subprocess.Popen(cmd, cwd=repo, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            active[key] = (proc, job, handle)
            print(json.dumps({"started": job.name, "gpu": gpu, "pid": proc.pid}), flush=True)
        time.sleep(5)
        for key, (proc, job, handle) in list(active.items()):
            code = proc.poll()
            if code is None: continue
            handle.close(); del active[key]
            if code == 0 and complete(root, job):
                print(json.dumps({"completed": job.name}), flush=True)
            elif job.attempts < 3:
                jobs.append(job); print(json.dumps({"retry": job.name, "status": code}), flush=True)
            else:
                failed.append({"job": job.name, "status": code})
    if failed:
        (log_dir / "FAILED.json").write_text(json.dumps(failed, indent=2) + "\n")
        raise RuntimeError(f"PSC pilot failed jobs: {failed}")
    analyze(root)
    (log_dir / "COMPLETE").write_text("400/400 complete and analyzed\n")
    print(json.dumps({"complete": True, "episodes": 400}), flush=True)


if __name__ == "__main__":
    main()
