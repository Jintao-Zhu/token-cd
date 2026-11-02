"""Record a side-by-side comparison of the same task/seed under two arms.

Uses the exact production pipeline (same _environment constructor, same policy input
construction, same ActionChunk execution and replan cadence) against the running policy
server, so what the video shows is what the experiment actually ran. Frames are captured once
per control step from the task camera.

Outputs per-arm frame dumps plus a side-by-side mp4 with a step counter and outcome banner.
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
sys.path.insert(0, str(TOKEN_CD))
sys.path.insert(0, str(ROOT / "openpi/packages/openpi-client/src"))
sys.path.insert(0, str(ROOT / "openpi/src"))

import numpy as np  # noqa: E402


def run_arm(task: str, seed: int, arm: str, port: int, frames_dir: Path, rt_depth: int = 8):
    from research.semantic_token_cd.pi0_fast_simpler_l6_matched_rollout import (
        ActionChunk, RolloutPolicy, _environment, _raw_policy_input,
    )
    import cv2

    policy = RolloutPolicy("127.0.0.1", port)
    env = _environment(task, rt_depth)
    obs, reset_info = env.reset(seed=seed)
    instruction = str(env.unwrapped.get_language_instruction())
    camera = "overhead_camera" if "google_robot" in env.unwrapped.robot_uid else "3rd_view_camera"

    frames_dir.mkdir(parents=True, exist_ok=True)
    chunk = ActionChunk("google_robot" if "google_robot" in env.unwrapped.robot_uid else "widowx_bridge")
    success = terminated = truncated = False
    steps = 0
    max_steps = 250
    replans = 0
    t0 = time.time()
    while not (terminated or truncated) and steps < max_steps:
        image = obs["image"][camera]["rgb"]
        cv2.imwrite(str(frames_dir / f"{steps:04d}.png"),
                    cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
        raw = _raw_policy_input(image, instruction, obs["agent"], env)
        if steps % chunk.exec_horizon == 0:
            replan_seed = seed * 1009 + steps // chunk.exec_horizon
            act_chunk, trace = policy.chunk(arm, raw, replan_seed)
            replans += 1
            chunk.plan.extend(act_chunk[: chunk.exec_horizon])
        obs, reward, terminated, truncated, info = env.step(chunk.next(chunk.plan.popleft()))
        steps += 1
    success = bool(info.get("success")) if info else False
    # final frame a few times so the video holds on the outcome
    final = np.asarray(obs["image"][camera]["rgb"])
    for k in range(12):
        cv2.imwrite(str(frames_dir / f"{steps + k:04d}.png"),
                    cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
    env.close()
    return {"arm": arm, "task": task, "seed": seed, "success": success,
            "steps": steps, "replans": replans, "seconds": time.time() - t0,
            "instruction": instruction}


def main() -> None:
    task = os.environ.get("REC_TASK", "google_robot_pick_coke_can")
    seed = int(os.environ.get("REC_SEED", "34"))
    port = int(os.environ.get("REC_PORT", "8101"))
    arms = os.environ.get("REC_ARMS", "vanilla,l6_matched_050").split(",")
    out = TOKEN_CD / "runs/comparison_videos" / f"{task}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)

    metas = []
    for arm in arms:
        d = out / f"frames_{arm}"
        print(f"[rec] running {arm} ...", flush=True)
        m = run_arm(task, seed, arm, port, d)
        metas.append(m)
        print(f"[rec]   {arm}: success={m['success']} steps={m['steps']} "
              f"replans={m['replans']} {m['seconds']:.0f}s", flush=True)

    # side-by-side: pad the shorter arm by repeating its last frame
    import cv2
    seqs = []
    for arm in arms:
        fs = sorted((out / f"frames_{arm}").glob("*.png"))
        seqs.append([cv2.imread(str(f)) for f in fs])
    n = max(len(s) for s in seqs)
    for i, s in enumerate(seqs):
        while len(s) < n:
            s.append(s[-1])
    h, w = seqs[0][0].shape[:2]
    # scale so the two panels plus the banner fit
    scale = min(1.0, 1200 / (2 * w))
    w2, h2 = int(w * scale), int(h * scale)
    seqs = [[cv2.resize(f, (w2, h2)) for f in s] for s in seqs]

    banner_h = 64
    vid = out / f"compare_{'_vs_'.join(arms)}.mp4"
    vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 10, (2 * w2, h2 + banner_h))
    for i in range(n):
        canvas = np.zeros((h2 + banner_h, 2 * w2, 3), dtype=np.uint8)
        for j, s in enumerate(seqs):
            canvas[banner_h:, j * w2:(j + 1) * w2] = s[i]
        for j, (arm, m) in enumerate(zip(arms, metas)):
            label = arm if arm != "vanilla" else "Vanilla (official)"
            if arm != "vanilla":
                label += "  lambda=" + arm.rsplit("_", 1)[1]
            col = (80, 220, 80) if m["success"] else (80, 80, 220)
            cv2.putText(canvas, label, (j * w2 + 12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)
            cv2.putText(canvas, f"success={m['success']}  final_steps={m['steps']}",
                        (j * w2 + 12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        cv2.putText(canvas, f"step {min(i, max(m['steps'] for m in metas)):3d}",
                    (2 * w2 - 150, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        vw.write(canvas)
    vw.release()

    # re-encode with ffmpeg so the file plays everywhere
    final = out / f"compare_{'_vs_'.join(arms)}_h264.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(vid),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(final)],
                   check=False)
    (out / "meta.json").write_text(json.dumps({"task": task, "seed": seed, "arms": metas,
                                               "frames": n}, indent=2) + "\n")
    print(f"[rec] wrote {final}  ({n} frames, {2*w2}x{h2+banner_h})", flush=True)
    for p in (vid, final):
        if p.exists():
            print(f"[rec]   {p.name}: {p.stat().st_size/1e6:.2f} MB", flush=True)


if __name__ == "__main__":
    main()
