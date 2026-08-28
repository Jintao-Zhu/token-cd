#!/usr/bin/env python3
"""Download and verify a contiguous suite shard range at a fixed HF revision."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


REPO = "HuggingFaceVLA/libero"
REVISION = "86958911c0f959db2bbbdb107eb3e17c5f9c798e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--first-shard", type=int, required=True)
    parser.add_argument("--last-shard", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.manifest.exists():
        raise FileExistsError(f"refusing to overwrite {args.manifest}")

    tree = list(
        HfApi().list_repo_tree(
            REPO, repo_type="dataset", revision=REVISION, recursive=True, expand=True
        )
    )
    by_path = {item.path: item for item in tree}
    records = []
    for index in range(args.first_shard, args.last_shard + 1):
        relative = f"data/chunk-000/file-{index:03d}.parquet"
        item = by_path[relative]
        records.append(
            {
                "index": index,
                "path": relative,
                "bytes": item.size,
                "sha256": item.lfs.sha256,
            }
        )

    def download(record):
        target = args.local_dir / record["path"]
        if not (
            target.exists()
            and target.stat().st_size == record["bytes"]
            and sha256(target) == record["sha256"]
        ):
            hf_hub_download(
                REPO,
                record["path"],
                repo_type="dataset",
                revision=REVISION,
                local_dir=args.local_dir,
            )
        actual_size = target.stat().st_size
        actual_hash = sha256(target)
        return {
            **record,
            "actual_bytes": actual_size,
            "actual_sha256": actual_hash,
            "verified": actual_size == record["bytes"] and actual_hash == record["sha256"],
        }

    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download, record): record for record in records}
        for future in as_completed(futures):
            result = future.result()
            completed.append(result)
            print(
                f"{len(completed)}/{len(records)} shard {result['index']:03d} "
                f"verified={result['verified']}",
                flush=True,
            )
    completed.sort(key=lambda row: row["index"])
    payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "repo": REPO,
        "revision": REVISION,
        "suite": "LIBERO-Spatial",
        "suite_global_index_interval": [101469, 153511],
        "boundary_filter_required": True,
        "records": completed,
        "shard_count": len(completed),
        "total_bytes": sum(row["actual_bytes"] for row in completed),
        "all_verified": all(row["verified"] for row in completed),
    }
    args.manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.manifest)
    return 0 if payload["all_verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
