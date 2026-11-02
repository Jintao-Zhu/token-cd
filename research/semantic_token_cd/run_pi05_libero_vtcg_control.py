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
CONTROL_CODE = (
    "research/semantic_token_cd/pi05_matched_flow_cd.py",
    "research/semantic_token_cd/pi05_matched_flow_server.py",
    "research/semantic_token_cd/pi05_libero_plus_paired_rollout.py",
    "research/semantic_token_cd/run_pi05_libero_vtcg_control.py",
)


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
    parser.add_argument("--arm", default="uniform")
    args = parser.parse_args()

    artifact = args.artifact.resolve()
    manifest_path = artifact / "diagnostic_manifest.json"
    control_path = artifact / "control_manifest.json"
    status_path = artifact / "control_run.json"
    for path in (control_path, status_path):
        if path.exists():
            raise FileExistsError(path)

    manifest = json.loads(manifest_path.read_text())
    source = Path(manifest["source_artifact"])
    for relative, expected in manifest["source_hashes_sha256"].items():
        if _sha256(source / relative) != expected:
            raise RuntimeError(f"source artifact changed since freeze: {relative}")

    control = {
        "protocol": "PI05_LIBERO_VTCG_CONTROL_V1",
        "purpose": "alpha=0 random candidate selection over the identical 8-candidate pool",
        "arm": args.arm,
        "alpha": 0.0,
        "n_candidates": 8,
        "bandwidth_scale": 0.5,
        "num_flow_steps": 10,
        "replan_steps": 5,
        "env_seed": 7,
        "base_noise_seed": 20261021,
        "trial": 0,
        "pairs": manifest["selection"]["strata"],
        "pair_count": len(manifest["selection"]["strata"]),
        "code_hashes_sha256": {path: _sha256(REPO / path) for path in CONTROL_CODE},
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    with control_path.open("x", encoding="utf-8") as handle:
        json.dump(control, handle, indent=2, sort_keys=True)
        handle.write("\n")

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
        "/home/leju-suzhou/zjt_ws/imagemagick_lib/lib:" + env.get("LD_LIBRARY_PATH", "")
    )
    env["PYTHONPATH"] = ":".join(
        (
            "/home/leju-suzhou/zjt_ws/libero_plus_src",
            "/home/leju-suzhou/zjt_ws/openpi/packages/openpi-client/src",
            str(REPO),
            env.get("PYTHONPATH", ""),
        )
    )

    plan_by_name = {
        row["name"]: row
        for row in json.loads((source / "task_manifest.json").read_text())["plans"]
    }
    selected: dict[str, list[int]] = {}
    for row in manifest["selection"]["strata"]:
        selected.setdefault(row["plan"], []).append(int(row["task_index"]))

    log_dir = artifact / "control_logs"
    log_dir.mkdir(exist_ok=False)
    status = {
        "protocol": "PI05_LIBERO_VTCG_CONTROL_RUN_V1",
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gpu": args.gpu,
        "port": args.port,
        "arm": args.arm,
        "plans": [],
        "status": "running",
    }
    _write_status(status_path, status)

    for plan_name, task_indices in selected.items():
        suite = plan_by_name[plan_name]["suite"]
        command = [
            sys.executable,
            "-u",
            str(ROLLOUT),
            "--artifact",
            str(artifact / plan_name),
            "--suite",
            suite,
            "--task-ids",
            ",".join(str(index) for index in sorted(task_indices)),
            "--trial",
            "0",
            "--arms",
            args.arm,
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
        _write_status(status_path, status)
        if completed.returncode != 0:
            status["status"] = "failed"
            status["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            _write_status(status_path, status)
            raise SystemExit(completed.returncode)

    status["status"] = "completed"
    status["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    _write_status(status_path, status)
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
