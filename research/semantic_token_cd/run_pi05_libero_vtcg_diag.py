from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


REPO = Path("/home/leju-suzhou/zjt_ws/token-cd")
ROLLOUT = REPO / "research/semantic_token_cd/pi05_libero_plus_paired_rollout.py"
SOURCE_ARTIFACT = REPO / "artifacts/pi05_libero_pcd_full_v1_20261024_01"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_status(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--port", type=int, default=8106)
    args = parser.parse_args()

    artifact = args.artifact.resolve()
    manifest_path = artifact / "diagnostic_manifest.json"
    run_path = artifact / "diagnostic_run.json"
    if run_path.exists():
        raise FileExistsError(run_path)
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest["source_hashes_sha256"].items():
        actual = _sha256(SOURCE_ARTIFACT / relative)
        if actual != expected:
            raise RuntimeError(f"source artifact changed after freeze: {relative}")
    for relative, expected in manifest["source_code_hashes_sha256"].items():
        actual = _sha256(REPO / relative)
        if actual != expected:
            raise RuntimeError(f"trace source changed after manifest freeze: {relative}")

    env = os.environ.copy()
    env.update(
        {
            "LIBERO_CONFIG_PATH": "/tmp/libero_pcd_smoke_config",
            "MAGICK_HOME": "/home/leju-suzhou/zjt_ws/imagemagick_lib",
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "MUJOCO_EGL_DEVICE_ID": str(args.gpu),
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
    env["LD_LIBRARY_PATH"] = (
        "/home/leju-suzhou/zjt_ws/imagemagick_lib/lib:"
        + env.get("LD_LIBRARY_PATH", "")
    )
    env["PYTHONPATH"] = ":".join(
        (
            "/home/leju-suzhou/zjt_ws/libero_plus_src",
            "/home/leju-suzhou/zjt_ws/openpi/packages/openpi-client/src",
            str(REPO),
            env.get("PYTHONPATH", ""),
        )
    )

    log_dir = artifact / "logs"
    log_dir.mkdir(exist_ok=False)
    status = {
        "protocol": "PI05_LIBERO_VTCG_TRACE_DIAG_RUN_V1",
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gpu": args.gpu,
        "port": args.port,
        "plans": [],
        "status": "running",
    }
    _write_status(run_path, status)

    selected_by_plan: dict[str, list[int]] = {}
    for row in manifest["selection"]["strata"]:
        selected_by_plan.setdefault(row["plan"], []).append(int(row["task_index"]))
    plan_by_name = {
        row["name"]: row for row in json.loads((SOURCE_ARTIFACT / "task_manifest.json").read_text())["plans"]
    }

    for plan_name, task_indices in selected_by_plan.items():
        suite = plan_by_name[plan_name]["suite"]
        plan_artifact = artifact / plan_name
        trace_dir = artifact / "traces" / plan_name
        command = [
            sys.executable,
            "-u",
            str(ROLLOUT),
            "--artifact",
            str(plan_artifact),
            "--suite",
            suite,
            "--task-ids",
            ",".join(str(index) for index in sorted(task_indices)),
            "--trial",
            "0",
            "--arms",
            "vanilla,pcd",
            "--env-seed",
            "7",
            "--noise-seed",
            "20261021",
            "--settle-steps",
            "10",
            "--replan-steps",
            "5",
            "--num-steps",
            "10",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--trace-dir",
            str(trace_dir),
        ]
        log_path = log_dir / f"{plan_name}.log"
        with log_path.open("x", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=REPO,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        status["plans"].append(
            {
                "plan": plan_name,
                "suite": suite,
                "task_indices": sorted(task_indices),
                "returncode": completed.returncode,
                "log": str(log_path),
            }
        )
        _write_status(run_path, status)
        if completed.returncode != 0:
            status["status"] = "failed"
            status["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            _write_status(run_path, status)
            raise SystemExit(completed.returncode)

    status["status"] = "completed"
    status["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    _write_status(run_path, status)
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
