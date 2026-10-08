"""Redownload the five large OpenPI pi0-FAST shards into clean sparse files.

The interrupted gsutil ``*.gstmp`` files failed object-level CRC32C validation,
so this script deliberately does not trust or reuse their allocated extents.
It resumes only its own clean sparse files, and replaces the old shards only
after every clean download passes GCS CRC32C verification.
"""
from __future__ import annotations

import base64
import concurrent.futures
import os
import re
import subprocess
from pathlib import Path

import google_crc32c

ROOT = Path("/home/leju-suzhou/zjt_ws")
CHECKPOINT_DIR = ROOT / "checkpoints/openpi-assets/openpi-assets/checkpoints/pi0_fast_base"
PARTIAL_DIR = CHECKPOINT_DIR / "params.partial"
CLEAN_DIR = CHECKPOINT_DIR / "params.clean_download"
RANGE_DIR = ROOT / "checkpoints/pi0fast_clean_range_parts"
BAD_BACKUP_DIR = CHECKPOINT_DIR / "params.badcrc_backup"
BASE_URL = "https://storage.googleapis.com/openpi-assets/checkpoints/pi0_fast_base/params"
WORKERS = 24
CHUNK_SIZE = 16 * 1024 * 1024


def run_curl(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=True, capture_output=True, text=True)


def remote_crc32c(url: str) -> str:
    result = run_curl(["curl", "--fail", "--silent", "--show-error", "--head", "--location", url])
    match = re.search(r"(?im)^x-goog-hash:.*?crc32c=([^;\s]+)", result.stdout)
    if not match:
        raise RuntimeError(f"No GCS CRC32C header for {url}")
    return match.group(1)


def local_crc32c(path: Path) -> str:
    checksum = google_crc32c.Checksum()
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            checksum.update(block)
    return base64.b64encode(checksum.digest()).decode("ascii")


def sparse_holes(path: Path) -> list[tuple[int, int]]:
    fd = os.open(path, os.O_RDONLY)
    size = os.fstat(fd).st_size
    holes: list[tuple[int, int]] = []
    offset = 0
    try:
        while offset < size:
            try:
                data = os.lseek(fd, offset, os.SEEK_DATA)
            except OSError:
                holes.append((offset, size - offset))
                break
            if data > offset:
                holes.append((offset, data - offset))
            try:
                hole = os.lseek(fd, data, os.SEEK_HOLE)
            except OSError:
                hole = size
            offset = max(hole, data + 1)
    finally:
        os.close(fd)
    return [(start, length) for start, length in holes if length]


def absorb_partials() -> int:
    absorbed = 0
    for tmp in sorted(RANGE_DIR.glob("*.part")):
        match = re.search(r"\.(\d+)\.(\d+)\.part$", tmp.name)
        if not match:
            continue
        start, requested = map(int, match.groups())
        size = tmp.stat().st_size
        if size > requested:
            raise RuntimeError(f"Range temp exceeds request: {tmp}")
        base = tmp.name[: match.start()]
        target = next((p for p in CLEAN_DIR.rglob(base) if p.is_file()), None)
        if target is None:
            raise FileNotFoundError(f"No clean shard found for {tmp}")
        fd = os.open(target, os.O_WRONLY)
        try:
            with tmp.open("rb") as source:
                offset = start
                while block := source.read(8 * 1024 * 1024):
                    written = 0
                    while written < len(block):
                        n = os.pwrite(fd, block[written:], offset + written)
                        if n <= 0:
                            raise OSError("pwrite made no progress")
                        written += n
                    offset += len(block)
        finally:
            os.close(fd)
        absorbed += size
        tmp.unlink()
    return absorbed


def fetch_range(path: Path, start: int, length: int) -> None:
    end = start + length - 1
    relative = path.relative_to(CLEAN_DIR).as_posix()
    url = f"{BASE_URL}/{relative}"
    tmp = RANGE_DIR / f"{path.name}.{start}.{length}.part"
    run_curl([
        "curl", "--fail", "--silent", "--show-error", "--location",
        "--retry", "4", "--connect-timeout", "20", "--range", f"{start}-{end}",
        "--output", str(tmp), url,
    ])
    if tmp.stat().st_size != length:
        raise RuntimeError(f"Bad range length for {relative} [{start},{end}]")
    fd = os.open(path, os.O_WRONLY)
    try:
        with tmp.open("rb") as source:
            offset = start
            while block := source.read(8 * 1024 * 1024):
                written = 0
                while written < len(block):
                    n = os.pwrite(fd, block[written:], offset + written)
                    if n <= 0:
                        raise OSError("pwrite made no progress")
                    written += n
                offset += len(block)
    finally:
        os.close(fd)
        tmp.unlink(missing_ok=True)


def main() -> None:
    if not PARTIAL_DIR.is_dir():
        raise FileNotFoundError(PARTIAL_DIR)
    RANGE_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    BAD_BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    old_shards = sorted(PARTIAL_DIR.rglob("*.gstmp"))
    if len(old_shards) != 5:
        raise RuntimeError(f"Expected five failed large shards, found {len(old_shards)}")

    clean_shards: list[tuple[Path, Path]] = []
    for old in old_shards:
        rel = old.relative_to(PARTIAL_DIR).as_posix().removesuffix("_.gstmp")
        size = old.stat().st_size
        clean = CLEAN_DIR / rel
        clean.parent.mkdir(parents=True, exist_ok=True)
        if clean.exists() and clean.stat().st_size != size:
            raise RuntimeError(f"Unexpected clean shard size at {clean}")
        if not clean.exists():
            with clean.open("wb") as stream:
                stream.truncate(size)
        clean_shards.append((old, clean))

    print(f"absorbed {absorb_partials()} bytes from interrupted clean ranges", flush=True)

    tasks: list[tuple[Path, int, int]] = []
    for _, clean in clean_shards:
        holes = sparse_holes(clean)
        missing = sum(n for _, n in holes)
        print(f"clean shard {clean.name}: size={clean.stat().st_size}, missing={missing}, ranges={len(holes)}", flush=True)
        for start, length in holes:
            cursor = start
            while cursor < start + length:
                n = min(CHUNK_SIZE, start + length - cursor)
                tasks.append((clean, cursor, n))
                cursor += n

    print(f"fetching {len(tasks)} fresh ranges with {WORKERS} workers", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(fetch_range, path, start, length) for path, start, length in tasks]
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            print(f"completed fresh range {done}/{len(futures)}", flush=True)

    for old, clean in clean_shards:
        rel = clean.relative_to(CLEAN_DIR).as_posix()
        url = f"{BASE_URL}/{rel}"
        expected = remote_crc32c(url)
        actual = local_crc32c(clean)
        if actual != expected:
            raise RuntimeError(f"Fresh shard CRC32C mismatch for {rel}: local={actual}, remote={expected}")
        print(f"CRC32C PASS {rel}: {actual}", flush=True)

    # Also verify the already complete, small files before publishing the tree.
    small_files = sorted(p for p in PARTIAL_DIR.rglob("*") if p.is_file() and not p.name.endswith("_.gstmp"))
    for path in small_files:
        rel = path.relative_to(PARTIAL_DIR).as_posix()
        expected = remote_crc32c(f"{BASE_URL}/{rel}")
        actual = local_crc32c(path)
        if actual != expected:
            raise RuntimeError(f"Existing small object CRC32C mismatch for {rel}: local={actual}, remote={expected}")
    print(f"CRC32C PASS for all {len(small_files)} existing small files", flush=True)

    for old, clean in clean_shards:
        rel = old.relative_to(PARTIAL_DIR)
        backup = BAD_BACKUP_DIR / rel
        backup.parent.mkdir(parents=True, exist_ok=True)
        if backup.exists():
            raise FileExistsError(backup)
        old.rename(backup)
        clean.rename(old.with_name(old.name.removesuffix("_.gstmp")))
        print(f"replaced corrupt shard {old.name}", flush=True)

    leftovers = list(PARTIAL_DIR.rglob("*.gstmp"))
    if leftovers:
        raise RuntimeError(f"Unverified temporary shards remain: {leftovers}")
    final_dir = CHECKPOINT_DIR / "params"
    if final_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {final_dir}")
    os.replace(PARTIAL_DIR, final_dir)
    print(f"All checkpoint objects verified; checkpoint ready at {final_dir}", flush=True)


if __name__ == "__main__":
    main()
