#!/usr/bin/env bash
# Download candidate pi0-FAST bridge checkpoints for a widowx interaction test.
# Only the params + assets of one step are needed; opt_state is large and unused.
set -uo pipefail
export HF_ENDPOINT=https://hf-mirror.com
DEST=/home/leju-suzhou/zjt_ws/checkpoints/pi0fast_widowx_candidates
mkdir -p "$DEST"
PY=/home/leju-suzhou/miniconda3/envs/pi05/bin/python
"$PY" - <<'PY'
import os
from huggingface_hub import snapshot_download
tok = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
dest = "/home/leju-suzhou/zjt_ws/checkpoints/pi0fast_widowx_candidates"
jobs = [
    ("akuramshin/pi0_fast_bridge_mix", "10000"),
    ("akuramshin/pi0_fast_bridge_original", "10000"),
]
for repo, step in jobs:
    name = repo.split("/")[-1]
    print(f"=== {repo} step {step} ===", flush=True)
    try:
        p = snapshot_download(
            repo_id=repo, token=tok,
            local_dir=f"{dest}/{name}_{step}",
            allow_patterns=[f"{step}/params/**", f"{step}/assets/**", f"{step}/_CHECKPOINT_METADATA"],
        )
        print("  ->", p, flush=True)
    except Exception as e:
        print(f"  FAIL {type(e).__name__}: {str(e)[:200]}", flush=True)
print("DONE", flush=True)
PY
