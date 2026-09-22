#!/usr/bin/env python3
"""Compare official-protocol matched and vanilla LIBERO-Spatial runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(root: Path) -> dict[tuple[str, int], dict]:
    rows = {}
    for path in sorted(root.glob("*/episode_*.json")):
        row = json.loads(path.read_text())
        rows[(row["task"], int(row["init_state_index"]))] = row
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--matched-root", type=Path, required=True)
    p.add_argument("--vanilla-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    a = p.parse_args()
    matched, vanilla = load(a.matched_root.resolve()), load(a.vanilla_root.resolve())
    keys = sorted(set(matched) | set(vanilla))
    tasks = sorted({key[0] for key in keys})
    out = {"matched_total": len(matched), "vanilla_total": len(vanilla), "tasks": {}}
    lines = [
        "# LIBERO-Spatial official protocol: matched vs vanilla",
        "",
        "| Task | n | matched | vanilla | Δ | rescue | harm | net |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    total = {"n": 0, "m": 0, "v": 0, "rescue": 0, "harm": 0}
    for task in tasks:
        task_keys = [key for key in keys if key[0] == task]
        paired = [key for key in task_keys if key in matched and key in vanilla]
        m_success = sum(bool(matched[key]["success"]) for key in paired)
        v_success = sum(bool(vanilla[key]["success"]) for key in paired)
        rescue = sum(
            (not bool(matched[key]["success"])) and bool(vanilla[key]["success"])
            for key in paired
        )
        harm = sum(
            bool(matched[key]["success"]) and (not bool(vanilla[key]["success"]))
            for key in paired
        )
        rec = {
            "paired_n": len(paired),
            "matched_available": sum(1 for key in task_keys if key in matched),
            "vanilla_available": sum(1 for key in task_keys if key in vanilla),
            "matched_success": int(m_success),
            "vanilla_success": int(v_success),
            "delta": int(m_success - v_success),
            "rescue": int(rescue),
            "harm": int(harm),
            "net": int(rescue - harm),
        }
        matched_rows = [matched[key] for key in paired]
        if matched_rows:
            rec["matched_m_mean"] = float(np.mean([r["selected_token_count_mean"] for r in matched_rows]))
            rec["matched_m_std"] = float(np.std([r["selected_token_count_mean"] for r in matched_rows]))
            rec["matched_steps_mean"] = float(np.mean([r["steps"] for r in matched_rows]))
        out["tasks"][task] = rec
        lines.append(
            f"| {task} | {len(paired)} | {m_success} | {v_success} | "
            f"{m_success - v_success:+d} | {rescue} | {harm} | {rescue - harm:+d} |"
        )
        total["n"] += len(paired)
        total["m"] += m_success
        total["v"] += v_success
        total["rescue"] += rescue
        total["harm"] += harm
    out["paired_total"] = {
        "n": total["n"],
        "matched_success": total["m"],
        "vanilla_success": total["v"],
        "delta": total["m"] - total["v"],
        "rescue": total["rescue"],
        "harm": total["harm"],
        "net": total["rescue"] - total["harm"],
    }
    lines += [
        "",
        f"**Paired total**: n={total['n']}, matched={total['m']}, "
        f"vanilla={total['v']}, delta={total['m'] - total['v']:+d}, "
        f"rescue={total['rescue']}, harm={total['harm']}, "
        f"net={total['rescue'] - total['harm']:+d}",
    ]
    root = a.out_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "COMPARE_SUMMARY.json").write_text(json.dumps(out, indent=2) + "\n")
    (root / "COMPARE_REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
