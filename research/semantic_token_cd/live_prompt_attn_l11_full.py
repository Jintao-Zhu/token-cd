"""Continuously publish the live results of the running L11 full rollout."""
from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "artifacts/prompt_attn_l11_matched_full_9task_0_299_v1"
RUN = ROOT / "run/episodes"
CANON = REPO / "artifacts/vanilla_recon_shr_canonical_0_299_v2/episodes"
STATE = ROOT / "coordinator_state.json"
OUT_MD = ROOT / "LIVE_RESULTS.md"
OUT_JSON = ROOT / "LIVE_RESULTS.json"
OUT_CSV = ROOT / "LIVE_EPISODES.csv"
HASH_KEYS = ("canonical_snapshot_sha256", "initial_state_sha256", "initial_rgb_sha256")


def atomic_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def exact_p(rescue: int, harm: int) -> float:
    n = rescue + harm
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(rescue, harm) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def rollout_running() -> bool:
    result = subprocess.run(
        ["pgrep", "-f", r"launch_prompt_attn_l11_full_0_299\.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def collect() -> tuple[dict, list[dict]]:
    rows: list[dict] = []
    for path in sorted(RUN.glob("*/*/episode_*_summary.json")):
        item = json.loads(path.read_text())
        task, seed = item["task"], int(item["seed"])
        row = {"task": task, "seed": seed, "l11": bool(item["success"])}
        okay = True
        for arm, subdir in (("vanilla", "vanilla"), ("shr", "shr_harmonic")):
            ref_path = CANON / task / subdir / f"episode_{seed:03d}_summary.json"
            if not ref_path.exists():
                okay = False
                break
            ref = json.loads(ref_path.read_text())
            if any(item.get(key) != ref.get(key) for key in HASH_KEYS):
                raise RuntimeError(f"hash mismatch: {task} seed={seed} arm={arm}")
            row[arm] = bool(ref["success"])
        if okay:
            rows.append(row)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["task"]].append(row)

    payload = {
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "completed_episodes": len(rows),
        "target_episodes": 1800,
        "l11_success": sum(row["l11"] for row in rows),
        "tasks": {},
    }
    for task, values in sorted(grouped.items()):
        comparison = {}
        for baseline in ("vanilla", "shr"):
            rescue = sum((not row[baseline]) and row["l11"] for row in values)
            harm = sum(row[baseline] and (not row["l11"]) for row in values)
            comparison[baseline] = {
                "success": sum(row[baseline] for row in values),
                "rescue": rescue,
                "harm": harm,
                "net": rescue - harm,
                "exact_p": exact_p(rescue, harm),
            }
        payload["tasks"][task] = {
            "n": len(values),
            "l11_success": sum(row["l11"] for row in values),
            "vanilla_success": comparison["vanilla"]["success"],
            "shr_success": comparison["shr"]["success"],
            "paired_vs_vanilla": comparison["vanilla"],
            "paired_vs_shr": comparison["shr"],
        }

    if STATE.exists():
        state = json.loads(STATE.read_text())
        payload["active_jobs"] = state.get("active", [])
        payload["completed_jobs"] = state.get("completed_jobs", 0)
        payload["queued_or_active_jobs"] = state.get("queued_or_active_jobs", 0)
    return payload, rows


def render_markdown(payload: dict) -> str:
    lines = [
        "# L11-Matched full run: live results",
        "",
        f"Updated: **{payload['last_updated']}**",
        f"Completed: **{payload['completed_episodes']}/1800**",
        f"L11 success: **{payload['l11_success']}**",
        "",
        "| Task | n | Vanilla | SHR | L11 | Net vs Vanilla | Net vs SHR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task, value in payload["tasks"].items():
        lines.append(
            f"| {task} | {value['n']} | {value['vanilla_success']} | "
            f"{value['shr_success']} | {value['l11_success']} | "
            f"{value['paired_vs_vanilla']['net']:+d} | "
            f"{value['paired_vs_shr']['net']:+d} |"
        )
    lines += [
        "",
        "## Paired same-seed comparisons",
        "",
        "| Task | Baseline | Rescue | Harm | Net | exact p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for task, value in payload["tasks"].items():
        for baseline, label in (("paired_vs_vanilla", "Vanilla"), ("paired_vs_shr", "SHR")):
            paired = value[baseline]
            lines.append(
                f"| {task} | {label} | {paired['rescue']} | {paired['harm']} | "
                f"{paired['net']:+d} | {paired['exact_p']:.6g} |"
            )
    lines += ["", "## Active jobs", ""]
    for job in payload.get("active_jobs", []):
        lines.append(f"- `{job}`")
    return "\n".join(lines) + "\n"


def write_csv(rows: list[dict]) -> None:
    tmp = OUT_CSV.with_name(f".{OUT_CSV.name}.{os.getpid()}.tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "seed", "vanilla", "shr", "l11"])
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, OUT_CSV)


def main() -> None:
    while True:
        payload, rows = collect()
        atomic_text(OUT_JSON, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        atomic_text(OUT_MD, render_markdown(payload))
        write_csv(rows)
        if (ROOT / "COMPLETE").exists() or not rollout_running():
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
