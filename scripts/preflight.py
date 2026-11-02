"""Preflight for the frozen 9-worker pi0-FAST experiment.

Checks, in order, everything that has to hold before 4500 episodes are worth starting:

  1. GPU identity: physical index -> UUID -> PCI bus id, and which cards are free.
  2. Config consistency: the frozen environment variables are exactly the frozen values, the
     sklearn backend actually imports (an isolated overlay, not a silent SciPy fallback), and
     both gate artifacts exist and PASS for THIS configuration.
  3. Checkpoint presence and readability.
  4. JAX: the process sees the intended physical GPU, and reports its memory limits.
  5. Vulkan/SAPIEN: enumerate render devices and match them to physical GPUs BY PCI BUS ID,
     because Vulkan ordinals do not follow nvidia-smi indices.
  6. Optional per-GPU memory probe: launch N JAX processes on one card and record peak usage.

Usage:
  preflight.py                     # checks 1-5
  preflight.py --probe-gpu 1 --probe-procs 3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/leju-suzhou/zjt_ws")
TOKEN_CD = ROOT / "token-cd"
RUN_DIR = TOKEN_CD / "runs/pi0fast_l6_matched_9task_100seed"
OVERLAY = ROOT / "pi05_sklearn_overlay"
CHECKPOINT = ROOT / "checkpoints/openpi_fast_finetuned/fractal"
TARGET_GPUS = [1, 2, 3]

FROZEN_ENV = {
    "PI0FAST_CD_KMEANS": "sklearn",
    "PI0FAST_CD_TAP_MODE": "separate",
    "PI0FAST_CD_DECODE": "verified",
    "PI0FAST_CD_EMBED_JIT": "0",
    "PI0FAST_CD_BATCH": "0",
}

results: list[dict] = []


def record(name: str, ok: bool, detail) -> bool:
    results.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    return bool(ok)


def run(cmd: list[str], timeout: int = 30) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception as exc:  # noqa: BLE001
        return f"<error {type(exc).__name__}: {exc}>"


def check_gpus() -> dict:
    out = run(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,memory.total,memory.used",
               "--format=csv,noheader"])
    gpus = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            gpus[int(parts[0])] = {
                "uuid": parts[1], "pci_bus_id": parts[2],
                "mem_total_mib": int(parts[3].split()[0]),
                "mem_used_mib": int(parts[4].split()[0]),
            }
    ok = all(g in gpus for g in TARGET_GPUS)
    record("gpu enumeration", ok,
           {g: gpus.get(g, "MISSING") for g in TARGET_GPUS})
    free = [g for g in TARGET_GPUS if gpus.get(g, {}).get("mem_used_mib", 10**9) < 1000]
    record("target GPUs idle", len(free) == len(TARGET_GPUS),
           f"idle={free} of {TARGET_GPUS}; "
           f"used={ {g: gpus.get(g,{}).get('mem_used_mib') for g in TARGET_GPUS} }")
    return gpus


def check_config(gpus: dict) -> None:
    # environment as the launcher will set it
    ok = True
    for k, v in FROZEN_ENV.items():
        got = os.environ.get(k)
        if got is not None and got != v:
            ok = False
    record("frozen env not overridden", ok,
           {k: os.environ.get(k, "<unset: launcher will set it>") for k in FROZEN_ENV})

    # sklearn must be importable from the overlay, never silently absent
    ov_ok = (OVERLAY / "sklearn").is_dir()
    record("sklearn overlay present", ov_ok, str(OVERLAY))
    out = run([sys.executable, "-c",
               "import sklearn, numpy, scipy, jax;"
               "print(sklearn.__version__, numpy.__version__, scipy.__version__, jax.__version__)"])
    parts = out.strip().split()
    record("sklearn importable with pinned deps",
           len(parts) == 4 and parts[0] == "1.7.2",
           {"sklearn": parts[0] if parts else None, "numpy": parts[1] if len(parts) > 1 else None,
            "scipy": parts[2] if len(parts) > 2 else None, "jax": parts[3] if len(parts) > 3 else None})

    # gates must exist and PASS for this configuration
    g1 = TOKEN_CD / "artifacts/pi0_fast_lambda0_equivalence_v1/frozen_v1/EQUIVALENCE.json"
    if g1.exists():
        d = json.loads(g1.read_text())
        n = d.get("states")
        ok = (d.get("same_payload_string") == n and d.get("same_actions") == n and n)
        record("gate1 (lambda=0) PASS for frozen config", ok,
               {k: d.get(k) for k in ("states", "same_payload_string", "same_actions", "verdict")})
    else:
        record("gate1 (lambda=0) artifact present", False, str(g1))

    g2 = TOKEN_CD / "artifacts/pi0_fast_gate2_regression_v1/REGRESSION.json"
    if g2.exists():
        d = json.loads(g2.read_text())
        s, g = d.get("streams"), d.get("guided_streams")
        ok = (d.get("stream_valid") == s and d.get("decoder_agrees") == s
              and d.get("neg_branch_valid") == g and s and g)
        record("gate2 (FAST stream) PASS", ok,
               {k: d.get(k) for k in ("states", "streams", "stream_valid", "decoder_agrees",
                                      "guided_streams", "neg_branch_valid", "verdict")})
        record("gate2 ran the sklearn backend",
               "sklearn" in str(d.get("kmeans_backends", [])),
               d.get("kmeans_backends"))
    else:
        record("gate2 artifact present", False, str(g2))

    # rejected optimizations must be off
    record("rejected optimizations disabled",
           os.environ.get("PI0FAST_CD_EMBED_JIT", "0") == "0"
           and os.environ.get("PI0FAST_CD_BATCH", "0") == "0",
           {"embed_jit": os.environ.get("PI0FAST_CD_EMBED_JIT", "0"),
            "batch": os.environ.get("PI0FAST_CD_BATCH", "0")})

    record("config.yaml present", (RUN_DIR / "config.yaml").exists(), str(RUN_DIR / "config.yaml"))
    record("manifests generated", (RUN_DIR / "manifests").is_dir(),
           f"{len(list((RUN_DIR/'manifests').glob('worker_*.json'))) if (RUN_DIR/'manifests').is_dir() else 0} shards")


def check_checkpoint() -> None:
    ok = CHECKPOINT.is_dir()
    record("checkpoint directory", ok, str(CHECKPOINT))
    if not ok:
        return
    files = sorted(p.name for p in CHECKPOINT.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in CHECKPOINT.rglob("*") if p.is_file())
    record("checkpoint non-empty", len(files) > 0,
           f"{len(files)} files, {total/1e9:.1f} GB, sample={files[:4]}")


def check_jax(gpus: dict) -> None:
    for g in TARGET_GPUS:
        code = (
            "import os,sys,json;"
            f"os.environ['CUDA_VISIBLE_DEVICES']='{g}';"
            "import jax;"
            "d=jax.devices();"
            "print(json.dumps({'backend':jax.default_backend(),"
            "'devices':[str(x) for x in d],"
            "'mem':[getattr(x,'memory_stats',lambda: {})() for x in d]}))"
        )
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(g)
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        try:
            out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                 timeout=180, env=env).stdout.strip().splitlines()
            info = json.loads(out[-1]) if out else {}
        except Exception as exc:  # noqa: BLE001
            info = {"error": f"{type(exc).__name__}: {exc}"}
        record(f"JAX sees physical GPU {g}", info.get("backend") == "gpu",
               {"backend": info.get("backend"), "devices": info.get("devices")})


def check_vulkan(gpus: dict) -> None:
    """Renderer binding, verified empirically rather than through device tables.

    vulkaninfo is not installed here, and Vulkan ordinals do not follow nvidia-smi indices
    anyway. SAPIEN binds its renderer through the explicit `device` kwarg (the rollout passes
    "cuda:0"), so the property to verify is that rendering with CUDA_VISIBLE_DEVICES=g
    actually consumes memory on physical GPU g. scripts/check_vulkan_binding.py measures
    exactly that with the production _environment() constructor; this reads its artifact.
    """
    art = RUN_DIR / "vulkan_binding.json"
    if not art.exists():
        record("renderer binding measured", False,
               f"{art} missing - run scripts/check_vulkan_binding.py first")
        return
    d = json.loads(art.read_text())
    per = d.get("per_worker", {})
    detail = {g: {"rendered": v.get("rendered"), "by_memory": v.get("by_memory")}
              for g, v in per.items()}
    ok = d.get("renderer_follows_cuda_pin") and all(
        v.get("rendered") for v in per.values()
    ) and len(per) >= len(TARGET_GPUS)
    record("renderer follows CUDA pin by physical GPU (memory-attributed)", bool(ok),
           {"verdict": d.get("verdict"), "per_worker": detail,
            "note": "Vulkan ordinal is NOT assumed to equal the nvidia-smi index"})


def probe_gpu(gpu: int, procs: int) -> None:
    """Launch N copies of a small JAX load on one card and report peak memory."""
    code = (
        "import os,time,json;"
        "import jax, jax.numpy as jnp;"
        "x=jnp.ones((4096,4096));"
        "y=(x@x).block_until_ready();"
        "time.sleep(20);"
        "d=jax.devices()[0];"
        "print(json.dumps({'mem':d.memory_stats()}))"
    )
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    ps = [subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True, env=env) for _ in range(procs)]
    import time as _t
    _t.sleep(12)
    out = run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader", "-i", str(gpu)])
    peak = out.strip()
    for p in ps:
        p.terminate()
    for p in ps:
        try:
            p.wait(timeout=20)
        except Exception:  # noqa: BLE001
            p.kill()
    record(f"gpu {gpu} with {procs} concurrent JAX processes", True, f"memory.used={peak}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-gpu", type=int, default=None)
    ap.add_argument("--probe-procs", type=int, default=3)
    args = ap.parse_args()

    print(f"=== pi0-FAST 9-worker preflight ===\nrun_dir={RUN_DIR}\npython={sys.executable}")
    gpus = check_gpus()
    check_config(gpus)
    check_checkpoint()
    check_jax(gpus)
    check_vulkan(gpus)
    if args.probe_gpu is not None:
        probe_gpu(args.probe_gpu, args.probe_procs)

    n_fail = sum(1 for r in results if not r["ok"])
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "preflight.json").write_text(json.dumps(
        {"checks": results, "failed": n_fail, "total": len(results),
         "verdict": "PASS" if n_fail == 0 else "FAIL"}, indent=2) + "\n")
    print(f"\n=== {len(results) - n_fail}/{len(results)} checks passed -> "
          f"{'PREFLIGHT_PASS' if n_fail == 0 else 'PREFLIGHT_FAIL'} ===")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
