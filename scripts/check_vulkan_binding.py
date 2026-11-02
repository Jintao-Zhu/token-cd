"""Verify which PHYSICAL GPU renders, using the production environment constructor.

SAPIEN binds its renderer through the explicit `device` kwarg (the rollout passes
"cuda:0"), so with CUDA_VISIBLE_DEVICES=g the renderer should follow the pin. That is the
claim being tested here - not assumed.

vulkaninfo is not installed, so rather than parsing Vulkan device tables this measures the
binding directly: build the real SIMPLER environment for a task, step it, and observe which
physical GPU's utilization rises. The previous attempt hand-rolled a SAPIEN scene and failed
before rendering anything ("renderer is not added to simulation"), which made its output
meaningless; this version uses the same constructor the experiment uses.

The Ray training job occupies GPUs 4-7, so their utilization is noisy; the check reports the
deltas rather than only the argmax, so a busy neighbour cannot masquerade as the renderer.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/leju-suzhou/zjt_ws")
TOKEN_CD = ROOT / "token-cd"
OUT = TOKEN_CD / "runs/pi0fast_l6_matched_9task_100seed"
GPUS = list(range(8))
TASK = "google_robot_open_drawer"

RENDER_SNIPPET = r"""
import json, os, sys
sys.path.insert(0, "/home/leju-suzhou/zjt_ws/token-cd")
sys.path.insert(0, "/home/leju-suzhou/zjt_ws/openpi/packages/openpi-client/src")
sys.path.insert(0, "/home/leju-suzhou/zjt_ws/openpi/src")
from research.semantic_token_cd.pi0_fast_simpler_l6_matched_rollout import _environment
import numpy as np
env = _environment("google_robot_open_drawer", int(os.environ.get("RT_DEPTH", "8")))
obs, info = env.reset(seed=0)
cam = "overhead_camera" if "google_robot" in env.unwrapped.robot_uid else "3rd_view_camera"
for _ in range(20):
    env.step(np.zeros(7, dtype=np.float32))
    _ = np.asarray(obs["image"][cam]["rgb"])
print(json.dumps({"rendered": True}), flush=True)
env.close()
"""


def util() -> dict:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader"],
        capture_output=True, text=True).stdout
    d = {}
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 3:
            d[int(p[0])] = {"util": int(p[1].split()[0]), "mem": int(p[2].split()[0])}
    return d


def sample_while(proc: subprocess.Popen, seconds: float) -> dict:
    peak = {g: {"util": 0, "mem": 0} for g in GPUS}
    t0 = time.time()
    while time.time() - t0 < seconds:
        if proc.poll() is not None:
            break
        cur = util()
        for g in GPUS:
            if g in cur:
                peak[g]["util"] = max(peak[g]["util"], cur[g]["util"])
                peak[g]["mem"] = max(peak[g]["mem"], cur[g]["mem"])
        time.sleep(0.4)
    return peak


def main() -> None:
    baseline = util()
    print(f"[vk] baseline util: { {g: baseline.get(g, {}).get('util') for g in GPUS} }", flush=True)
    env = dict(os.environ)
    env["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"
    env["RT_DEPTH"] = os.environ.get("RT_DEPTH", "8")
    results = {}
    for gpu in (1, 2, 3):
        e = dict(env)
        e["CUDA_VISIBLE_DEVICES"] = str(gpu)
        proc = subprocess.Popen([sys.executable, "-c", RENDER_SNIPPET],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=e)
        peak = sample_while(proc, 40)
        try:
            out, err = proc.communicate(timeout=90)
        except Exception:  # noqa: BLE001
            proc.kill()
            out, err = "", "timeout"
        ok = "rendered" in (out or "")
        # memory growth is the more reliable signal than utilization under a busy neighbour
        mem_delta = {g: peak[g]["mem"] - baseline.get(g, {}).get("mem", 0) for g in GPUS}
        util_delta = {g: peak[g]["util"] - baseline.get(g, {}).get("util", 0) for g in GPUS}
        by_mem = max(mem_delta, key=lambda g: mem_delta[g])
        by_util = max(util_delta, key=lambda g: util_delta[g])
        results[gpu] = {
            "rendered": ok, "by_memory": by_mem, "by_utilization": by_util,
            "mem_delta_mib": mem_delta, "util_delta": util_delta,
            "stderr_tail": (err or "")[-400:],
        }
        print(f"[vk] CUDA_VISIBLE_DEVICES={gpu}: rendered={ok} "
              f"by_memory={by_mem} (Δ{mem_delta[by_mem]} MiB) by_util={by_util} "
              f"mem_deltas={ {g: mem_delta[g] for g in GPUS if mem_delta[g] > 50} }", flush=True)
        time.sleep(3)

    rendered = [g for g in (1, 2, 3) if results[g]["rendered"]]
    follows = all(results[g]["by_memory"] == g for g in rendered) if rendered else False
    summary = {
        "protocol": "PI0_FAST_VULKAN_BINDING_V2",
        "task_used": TASK,
        "method": "production _environment() constructor + real env.step, then attribute the "
                  "physical GPU by memory growth (utilization is unreliable while the Ray job "
                  "occupies GPUs 4-7)",
        "vulkaninfo_available": False,
        "per_worker": results,
        "rendered_ok": rendered,
        "renderer_follows_cuda_pin": follows,
        "verdict": ("PASS: SAPIEN renderer follows CUDA_VISIBLE_DEVICES via renderer_kwargs "
                    "device='cuda:0'" if follows else
                    "INCONCLUSIVE_OR_MISMATCH: see per_worker; renderer may not follow the pin"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "vulkan_binding.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[vk] verdict: {summary['verdict']}", flush=True)


if __name__ == "__main__":
    main()
