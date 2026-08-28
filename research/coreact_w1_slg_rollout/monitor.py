from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    while True:
        episodes = len(list((artifact / "episodes").glob("*.json")))
        invalid = len(list((artifact / "invalid_pairs").glob("*.json")))
        decision = artifact / "decision.json"
        if decision.exists():
            return
        status = "running_background" if invalid == 0 else "rollout_failed"
        (artifact / "status.json").write_text(json.dumps({
            "status": status, "episodes_complete": episodes,
            "episodes_planned": 2000, "invalid_pairs": invalid,
        }, indent=2) + "\n")
        if episodes >= 2000 or invalid:
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
