"""Run final Top-p analysis once the persistent rollout coordinator completes."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--matched-artifact", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact.resolve()
    complete = root / "rollout_logs/COMPLETE.json"
    failed = root / "rollout_logs/FAILED.json"
    while not complete.exists():
        if failed.exists():
            raise RuntimeError(f"rollout coordinator failed: {failed}")
        time.sleep(30)
    repo = Path(__file__).resolve().parents[2]
    subprocess.run([
        sys.executable, str(repo / "research/semantic_token_cd/analyze_prompt_attn_l11_top_p.py"),
        "--artifact", str(root), "--matched-artifact", str(args.matched_artifact.resolve()),
    ], cwd=repo, check=True)


if __name__ == "__main__":
    main()
