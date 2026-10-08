"""Fill holes in an interrupted gsutil download of the public OpenPI checkpoint."""
from __future__ import annotations

import base64
import concurrent.futures
import os
import re
import subprocess
import time
from pathlib import Path

import google_crc32c

ROOT = Path("/home/leju-suzhou/zjt_ws")
CHECKPOINT_DIR = ROOT / "checkpoints/openpi-assets/openpi-assets/checkpoints/pi0_fast_base"
PARTIAL_DIR = CHECKPOINT_DIR / "params.partial"
FINAL_DIR = CHECKPOINT_DIR / "params"
TMP_DIR = ROOT / "checkpoints/pi0fast_http_range_parts"
BASE_URL = "https://storage.googleapis.com/openpi-assets/checkpoints/pi0_fast_base/params"
WORKERS = 24
CHUNK_SIZE = 32 * 1024 * 1024


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


def fetch_range(path: Path, start: int, length: int) -> None:
    end = start + length - 1
    relative = path.relative_to(PARTIAL_DIR).as_posix().removesuffix("_.gstmp")
    url = f"{BASE_URL}/{relative}"
    tmp = TMP_DIR / f"{path.name}.{start}.{length}.part"
    cmd = [
        "curl", "--fail", "--silent", "--show-error", "--location",
        "--retry", "4", "--range", f"{start}-{end}",
        "--output", str(tmp), url,
    ]
    subprocess.run(cmd, check=True)
    if tmp.stat().st_size != length:
        raise RuntimeError(f"Range length mismatch for {relative} [{start},{end}]: {tmp.stat().st_size} != {length}")
    fd = os.open(path, os.O_WRONLY)
    try:
        with tmp.open("rb") as src:
            offset = start
            while block := src.read(8 * 1024 * 1024):
                written = 0
                while written < len(block):
                    count = os.pwrite(fd, block[written:], offset + written)
                    if count <= 0:
                        raise OSError("pwrite returned no progress")
                    written += count
                offset += len(block)
    finally:
        os.close(fd)
        tmp.unlink(missing_ok=True)


def remote_crc32c(url: str) -> str:
    result = subprocess.run(["curl", "--fail", "--silent", "--show-error", "--head", "--location", url], check=True, capture_output=True, text=True)
    match = re.search(r"(?im)^x-goog-hash:.*?crc32c=([^;\s]+)", result.stdout)
    if not match:
        raise RuntimeError(f"GCS CRC32C header missing for {url}: {result.stdout[-500:]}")
    return match.group(1).strip()


def local_crc32c(path: Path) -> str:
    checksum = google_crc32c.Checksum()
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            checksum.update(block)
    return base64.b64encode(checksum.digest()).decode("ascii")


def absorb_partial_ranges() -> int:
    """Preserve prefixes already fetched into range temp files before retrying."""
    absorbed = 0
    for tmp in sorted(TMP_DIR.glob("*.part")):
        match = re.search(r"\.(\d+)\.(\d+)\.part$", tmp.name)
        if not match:
            continue
        start, requested = map(int, match.groups())
        prefix_size = tmp.stat().st_size
        if not prefix_size:
            tmp.unlink(missing_ok=True)
            continue
        if prefix_size > requested:
            raise RuntimeError(f"Temporary range exceeds request size: {tmp}")
        base_name = tmp.name[: match.start()]
        target = next((p for p in PARTIAL_DIR.rglob(base_name) if p.is_file()), None)
        if target is None:
            raise FileNotFoundError(f"No sparse checkpoint object for range file {tmp}")
        fd = os.open(target, os.O_WRONLY)
        try:
            with tmp.open("rb") as src:
                offset = start
                while block := src.read(8 * 1024 * 1024):
                    written = 0
                    while written < len(block):
                        count = os.pwrite(fd, block[written:], offset + written)
                        if count <= 0:
                            raise OSError("pwrite returned no progress")
                        written += count
                    offset += len(block)
        finally:
            os.close(fd)
        absorbed += prefix_size
        tmp.unlink()
    return absorbed


def main() -> None:
    if FINAL_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite completed checkpoint: {FINAL_DIR}")
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    partials = sorted(PARTIAL_DIR.rglob("*.gstmp"))
    if not partials:
        raise RuntimeError(f"No gsutil sparse temporary files under {PARTIAL_DIR}")

    absorbed = absorb_partial_ranges()
    print(f"preserved {absorbed} additional bytes from interrupted HTTPS ranges", flush=True)

    tasks: list[tuple[Path, int, int]] = []
    for path in partials:
        size = path.stat().st_size
        holes = sparse_holes(path)
        missing = sum(length for _, length in holes)
        print(f"sparse {path.name}: size={size} missing={missing} ranges={len(holes)}", flush=True)
        for start, length in holes:
            cursor = start
            limit = start + length
            while cursor < limit:
                chunk_length = min(CHUNK_SIZE, limit - cursor)
                tasks.append((path, cursor, chunk_length))
                cursor += chunk_length

    started = time.monotonic()
    print(f"filling {len(tasks)} byte ranges with {WORKERS} HTTPS workers", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(fetch_range, path, start, length) for path, start, length in tasks]
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            print(f"completed range {done}/{len(futures)}", flush=True)

    for partial in partials:
        expected_size = partial.stat().st_size
        relative = partial.relative_to(PARTIAL_DIR).as_posix().removesuffix("_.gstmp")
        target = partial.with_name(partial.name.removesuffix("_.gstmp"))
        url = f"{BASE_URL}/{relative}"
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing object: {target}")
        if partial.stat().st_size != expected_size:
            raise RuntimeError(f"Size changed for {partial}")
        expected_crc = remote_crc32c(url)
        actual_crc = local_crc32c(partial)
        if actual_crc != expected_crc:
            raise RuntimeError(f"CRC32C mismatch for {relative}: local={actual_crc}, remote={expected_crc}")
        partial.rename(target)
        print(f"verified and finalized {relative} ({expected_size} bytes, crc32c={actual_crc})", flush=True)

    leftovers = list(PARTIAL_DIR.rglob("*.gstmp"))
    if leftovers:
        raise RuntimeError(f"Unexpected sparse files remain: {leftovers}")
    os.replace(PARTIAL_DIR, FINAL_DIR)
    print(f"checkpoint complete at {FINAL_DIR}; elapsed={time.monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
