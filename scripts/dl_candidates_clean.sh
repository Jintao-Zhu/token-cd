#!/usr/bin/env bash
# Single clean downloader for the candidate pi0-FAST bridge checkpoints.
#
# Earlier attempts failed for two reasons worth recording: huggingface_hub aborts the whole
# batch on a CDN read timeout, and a hand-rolled curl loop with `set -u` died on an empty
# array expansion while a stray snapshot_download was still writing the same files. So: one
# process only, generous per-request timeout, unlimited retries, and resume.
set -uo pipefail
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_ENABLE_HF_TRANSFER=0
DEST=/home/leju-suzhou/zjt_ws/checkpoints/pi0fast_widowx_candidates
mkdir -p "$DEST"
exec /home/leju-suzhou/miniconda3/envs/pi05/bin/python - <<'PY'
import os, time
from huggingface_hub import snapshot_download
tok = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
dest = "/home/leju-suzhou/zjt_ws/checkpoints/pi0fast_widowx_candidates"
jobs = [("akuramshin/pi0_fast_bridge_mix", "10000"),
        ("akuramshin/pi0_fast_bridge_original", "10000")]
for repo, step in jobs:
    name = repo.split("/")[-1]
    target = f"{dest}/{name}_{step}"
    for attempt in range(1, 11):
        try:
            p = snapshot_download(
                repo_id=repo, token=tok, local_dir=target,
                allow_patterns=[f"{step}/params/**", f"{step}/assets/**",
                                f"{step}/_CHECKPOINT_METADATA"],
                max_workers=4,
            )
            print(f"[dl] OK {repo} -> {p}", flush=True)
            break
        except Exception as e:
            print(f"[dl] attempt {attempt} failed for {repo}: {type(e).__name__}: {str(e)[:160]}",
                  flush=True)
            time.sleep(min(60, attempt * 10))
    else:
        print(f"[dl] GAVE UP on {repo}", flush=True)
print("[dl] DONE", flush=True)
PY
