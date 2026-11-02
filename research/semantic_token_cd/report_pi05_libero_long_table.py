"""Cross-configuration table for the pi0.5 LIBERO-Long paired runs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


def load(artifact: Path) -> dict:
    summary = json.loads((artifact / "paired_summary.json").read_text())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for root in args.roots:
        candidates = [root] if (root / "paired_summary.json").exists() else sorted(
            path for path in root.iterdir() if (path / "paired_summary.json").exists()
        )
        for artifact in candidates:
            summary = load(artifact)
            test = summary.get("paired_test", {})
            rows.append(
                {
                    "artifact": artifact.name,
                    "pairs": summary["pairs_complete"],
                    "vanilla_rate": summary["per_arm"]["vanilla"]["success_rate"],
                    "matched_rate": summary["per_arm"]["matched"]["success_rate"],
                    "delta": test.get("delta_success_rate"),
                    "v_only": test.get("vanilla_only_success"),
                    "m_only": test.get("matched_only_success"),
                    "p": test.get("mcnemar_exact_p"),
                }
            )

    lines = [
        "| run | pairs | vanilla | matched | delta | v-only / m-only | McNemar p |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['artifact']} | {row['pairs']} | {row['vanilla_rate']:.3f} | "
            f"{row['matched_rate']:.3f} | {row['delta']:+.3f} | "
            f"{row['v_only']} / {row['m_only']} | {row['p']:.3f} |"
        )
    table = "\n".join(lines)
    print(table)
    if args.out:
        args.out.write_text(table + "\n")


if __name__ == "__main__":
    main()
