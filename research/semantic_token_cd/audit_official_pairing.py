#!/usr/bin/env python3
"""Audit paired LIBERO Spatial Matched/Vanilla raw episode JSON outcomes."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts/libero_official_compare_500_v1"
MATCHED = ROOT / "artifacts/libero_official_matched_500_v1"
VANILLA = ROOT / "artifacts/libero_official_vanilla_500_v1"


def load_episode_map(directory: Path) -> dict[str, dict]:
    return {p.name: json.loads(p.read_text()) for p in directory.glob("episode_*.json")}


def main() -> None:
    task_rows = []
    for matched_dir in sorted(p for p in MATCHED.iterdir() if p.is_dir() and p.name != "logs"):
        task = matched_dir.name
        matched = load_episode_map(matched_dir)
        vanilla = load_episode_map(VANILLA / task)
        if set(matched) != set(vanilla):
            raise RuntimeError(f"unpaired episode IDs in {task}")
        episodes = sorted(matched)
        rescue = sum(bool(matched[k]["success"]) and not bool(vanilla[k]["success"]) for k in episodes)
        harm = sum(bool(vanilla[k]["success"]) and not bool(matched[k]["success"]) for k in episodes)
        m_success = sum(bool(matched[k]["success"]) for k in episodes)
        v_success = sum(bool(vanilla[k]["success"]) for k in episodes)
        if rescue - harm != m_success - v_success:
            raise AssertionError(f"paired arithmetic mismatch in {task}")
        task_rows.append({"task": task, "n": len(episodes), "matched": m_success,
                          "vanilla": v_success, "rescue": rescue, "harm": harm,
                          "net": m_success - v_success})

    total = {key: sum(row[key] for row in task_rows)
             for key in ("n", "matched", "vanilla", "rescue", "harm", "net")}
    if total["n"] != 500 or total["rescue"] - total["harm"] != total["net"]:
        raise AssertionError(f"unexpected totals: {total}")
    payload = {"protocol": "LIBERO_SPATIAL_RAW_PAIRED_OUTCOME_AUDIT_V1",
               "pairing": "task directory + episode_NNN filename",
               "task_rows": task_rows, "total": total}
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    (ARTIFACT / "AUDITED_COMPARE_SUMMARY.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = ["# Audited LIBERO Spatial Matched vs Vanilla comparison", "",
             "This report recomputes paired outcomes from the raw episode JSON files in the Matched and Vanilla result directories. Pairing key is `(task directory, episode_NNN filename)`.", "",
             "| Task | n | Matched | Vanilla | Rescue | Harm | Net |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in task_rows:
        lines.append(f"| {row['task']} | {row['n']} | {row['matched']} | {row['vanilla']} | {row['rescue']} | {row['harm']} | {row['net']:+d} |")
    lines += [f"| **Total** | **{total['n']}** | **{total['matched']}** | **{total['vanilla']}** | **{total['rescue']}** | **{total['harm']}** | **{total['net']:+d}** |", "",
              "Rescue is Matched success / Vanilla failure. Harm is Vanilla success / Matched failure. The identity `Rescue - Harm = Matched successes - Vanilla successes` was asserted for every task and the total.", "",
              "The earlier `COMPARE_REPORT.md` reports total Matched 410, Vanilla 423, delta −13, Rescue 57, Harm 44, net +13, and reverses Rescue/Harm in each task row. Raw pairing yields Rescue 44, Harm 57, net −13. See `AUDITED_COMPARE_SUMMARY.json` for machine-readable counts.", "",
              "Reproduce with: `python3 research/semantic_token_cd/audit_official_pairing.py`."]
    (ARTIFACT / "AUDITED_COMPARE_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
