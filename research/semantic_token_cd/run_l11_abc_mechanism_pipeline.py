#!/usr/bin/env python3
"""Sequential, audited A/B/C OpenVLA SIMPLER experiment pipeline.

The orchestrator uses CPU only. All renderer/model work is launched as tmux
workers pinned to physical GPUs 1, 2, and 3, three workers per GPU.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
ENV = Path("/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python")
ART = ROOT / "artifacts/l11_abc_mechanism_v1"
STATE = ART / "state_library"
SNAPS = ROOT / "artifacts/vanilla_recon_shr_canonical_0_299_v2"
FULL = ROOT / "artifacts/l11_table2_main_ablation_v1/formal/full"
LOG = ART / "orchestrator.log"
TASKS = ["google_robot_open_drawer", "google_robot_close_drawer",
         "google_robot_pick_coke_can", "google_robot_move_near"]
REQUIRED = {"rgb", "segmentation_instance", "sim_state", "eef_pos",
            "agent_state_blob", "episode_rng_blob", "elapsed_steps"}


def log(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def launch(session: str, gpu: int, body: str, log_path: Path, exit_path: Path) -> None:
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        raise RuntimeError(f"tmux session already exists: {session}")
    inner = (
        f"cd {shlex.quote(str(ROOT))}; source scripts/activate_simpler_local.sh; set +e; "
        f"export CUDA_VISIBLE_DEVICES={int(gpu)}; {body} > {shlex.quote(str(log_path))} 2>&1; "
        f"rc=$?; echo $rc > {shlex.quote(str(exit_path))}; "
        "if [[ $rc -ne 0 ]]; then sleep 3600; fi"
    )
    subprocess.run(["tmux", "new-session", "-d", "-s", session,
                    "bash -lc " + shlex.quote(inner)], check=True)


def wait_workers(stage: str, sessions: list[str], exit_paths: list[Path]) -> None:
    log(f"WAIT {stage}: {len(sessions)} workers")
    while True:
        pending = []
        for session, exit_path in zip(sessions, exit_paths):
            if exit_path.exists():
                code = int(exit_path.read_text().strip())
                if code != 0:
                    raise RuntimeError(f"worker failed: {session}, exit={code}, log={exit_path.with_suffix('.log')}")
            elif subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode != 0:
                raise RuntimeError(f"worker disappeared without exit record: {session}")
            else:
                pending.append(session)
        if not pending:
            log(f"DONE {stage}")
            return
        log(f"RUNNING {stage}: {len(pending)}/{len(sessions)} workers remain")
        time.sleep(30)


def run_command(command: list[str], env: dict | None = None) -> None:
    log("RUN " + " ".join(map(str, command)))
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def validate_state_library() -> int:
    from research.semantic_token_cd.build_l11_ab_state_library import sample_quartile_indices
    import numpy as np
    expected_count = 0
    for task in TASKS:
        for seed in range(100, 200):
            summary_path = FULL / "episodes" / task / "full" / f"episode_{seed:03d}_summary.json"
            arrays_path = summary_path.with_name(f"episode_{seed:03d}_arrays.npz")
            summary = json.loads(summary_path.read_text())
            with np.load(arrays_path, allow_pickle=True) as arrays:
                actions = arrays["executed_actions"]
            steps = sample_quartile_indices(summary["selector_trace"])
            expected_count += len(steps)
            if len(actions) != int(summary["steps"]):
                raise RuntimeError(f"source action count mismatch {task}:{seed}")
            for step in steps:
                stem = f"{task}__seed{seed:03d}__step{step:04d}"
                mp = STATE / "states" / task / f"{stem}.json"
                ap = mp.with_suffix(".npz")
                if not mp.is_file() or not ap.is_file():
                    raise RuntimeError(f"missing sampled state {task}:{seed}:{step}")
                meta = json.loads(mp.read_text())
                with np.load(ap, allow_pickle=False) as data:
                    if not REQUIRED.issubset(data.files):
                        raise RuntimeError(f"incomplete saved state fields {ap}")
                    if data["rgb"].shape != (512, 640, 3) or data["segmentation_instance"].shape != (512, 640):
                        raise RuntimeError(f"bad image/segmentation shape {ap}")
                    if int(data["elapsed_steps"]) != step:
                        raise RuntimeError(f"state timestep mismatch {ap}")
                if meta["initial_state_sha256"] != summary["initial_state_sha256"] or meta["initial_rgb_sha256"] != summary["initial_rgb_sha256"]:
                    raise RuntimeError(f"initial state/RGB hash mismatch {mp}")
                if int(meta["m_matched"]) != int(summary["selector_trace"][step]["m_t"]):
                    raise RuntimeError(f"Matched budget mismatch {mp}")
    log(f"STATE_LIBRARY_PASS: {expected_count} valid sampled frames")
    return expected_count


def wait_state_library() -> None:
    names = [f"l11abc_final_gpu{gpu}_w{worker}" for gpu in (1, 2, 3) for worker in (1, 2, 3)]
    exits = [ART / "logs" / f"{name}.exit" for name in names]
    wait_workers("A-state-library", names, exits)


def run_a1() -> int:
    count = validate_state_library()
    out = ART / "A_spatial_support"
    run_command([str(ENV), "research/semantic_token_cd/analyze_l11_abc_state_library.py",
                 "--library", str(STATE), "--out", str(out), "--bootstrap", "1000"])
    return count


def run_a2() -> None:
    states_file = ART / "A_budget_dose" / "preflight_states.txt"
    states_file.parent.mkdir(parents=True, exist_ok=True)
    chosen = []
    for task in TASKS:
        found = sorted((STATE / "states" / task).glob("*seed100*.json"))
        if not found:
            raise RuntimeError(f"no preflight state for {task}:100")
        chosen.append(str(found[0].resolve()))
    states_file.write_text("\n".join(chosen) + "\n")
    preflight_out = ART / "A_budget_dose" / "preflight"
    run_command([str(ENV), "research/semantic_token_cd/offline_l11_budget_state_scan.py",
                 "--gpu", "1", "--shard", "0", "--shards", "1", "--library", str(STATE),
                 "--output", str(preflight_out), "--states-file", str(states_file)])
    rows = [json.loads(x) for x in (preflight_out / "shard_00.jsonl").read_text().splitlines() if x]
    if len(rows) != 20 or len({(r["task"], r["seed"], r["step"]) for r in rows}) != 4:
        raise RuntimeError(f"A2 preflight expected four states x five doses, got {len(rows)} rows")
    for row in rows:
        expected = max(1, min(256, int(round(row["m_matched"] * row["multiplier"]))))
        if row["m_used"] != expected or not all(npfinite(row[k]) for k in ("D_feat", "D_action_scaled_l2", "mean_action_logit_js")):
            raise RuntimeError(f"A2 preflight technical mismatch: {row}")
    for key in {(r["task"], r["seed"], r["step"]) for r in rows}:
        group = [r for r in rows if (r["task"], r["seed"], r["step"]) == key]
        if len(group) != 5 or len({r["l11_score_sha256"] for r in group}) != 1 or len({r["positive_action_logits_sha256"] for r in group}) != 1 or len({r["m_matched"] for r in group}) != 1:
            raise RuntimeError(f"A2 changed ranking/clean branch across budget doses at {key}")
    (preflight_out / "PREFLIGHT_PASS.json").write_text(json.dumps({"status": "PASS", "states": 4, "rows": 20}, indent=2) + "\n")

    output = ART / "A_budget_dose" / "formal"
    sessions, exits = [], []
    for shard in range(9):
        gpu = shard // 3 + 1
        session = f"l11abc_a2_gpu{gpu}_w{shard % 3 + 1}"
        logfile = ART / "logs" / f"{session}.log"
        exitfile = ART / "logs" / f"{session}.exit"
        cmd = " ".join(map(shlex.quote, [str(ENV), "research/semantic_token_cd/offline_l11_budget_state_scan.py",
                   "--gpu", str(gpu), "--shard", str(shard), "--shards", "9", "--library", str(STATE), "--output", str(output)]))
        launch(session, gpu, cmd, logfile, exitfile)
        sessions.append(session); exits.append(exitfile)
    wait_workers("A2-offline-budget-scan", sessions, exits)
    run_command([str(ENV), "research/semantic_token_cd/analyze_l11_budget_state_scan.py",
                 "--input", str(output), "--output", str(ART / "A_budget_dose" / "analysis_summary.json")])


def npfinite(value) -> bool:
    import math
    return math.isfinite(float(value))


def b_preflight() -> None:
    artifact = ART / "B_target_region"
    snap = ROOT / "artifacts/vanilla_recon_shr_canonical_0_299_v2"
    first = "google_robot_open_drawer"
    sessions, exits = [], []
    # The first call creates the shared immutable CONFIG_LOCK; subsequent task
    # preflights run only after it is fully written.
    jobs = [(1, first, "open"), (2, "google_robot_close_drawer", "close"),
            (3, "google_robot_pick_coke_can", "pick"), (1, "google_robot_move_near", "move")]
    for idx, (gpu, task, suffix) in enumerate(jobs):
        session = f"l11abc_b_pre_{suffix}"
        if idx > 0:
            # The first worker establishes the same protocol lock before peers start.
            wait_workers("B-preflight-lock", [sessions[0]], [exits[0]])
        logpath = ART / "logs" / f"{session}.log"; exitpath = ART / "logs" / f"{session}.exit"
        body = " ".join(map(shlex.quote, [str(ROOT / "scripts/run_l11_abc_region_worker.sh"), str(gpu),
                    str(artifact), str(snap), session, f"{task}=100-104"]))
        launch(session, gpu, body, logpath, exitpath)
        sessions.append(session); exits.append(exitpath)
    wait_workers("B-preflight", sessions, exits)
    validate_b(artifact, seeds=range(100, 105))
    (artifact / "PREFLIGHT_PASS.json").write_text(json.dumps({"status": "PASS", "episodes": 40}, indent=2) + "\n")


def validate_b(artifact: Path, seeds) -> dict:
    records = {}
    for task in TASKS:
        for seed in seeds:
            base = json.loads((FULL / "episodes" / task / "full" / f"episode_{seed:03d}_summary.json").read_text())
            pair = []
            for arm in ("target_enriched", "target_sparing"):
                path = artifact / "episodes" / task / arm / f"episode_{seed:03d}_summary.json"
                data = json.loads(path.read_text())
                if data.get("technical_pass") is not True:
                    raise RuntimeError(f"B technical audit failed {path}")
                for field in ("canonical_snapshot_sha256", "initial_state_sha256", "initial_rgb_sha256"):
                    if data[field] != base[field]:
                        raise RuntimeError(f"B baseline mismatch {path} field={field}")
                if data.get("rendering", {}).get("rt_max_path_depth") != (8 if "drawer" in task else 0):
                    raise RuntimeError(f"B renderer mismatch {path}")
                pair.append(data)
            if len({x["initial_state_sha256"] for x in pair}) != 1 or len({x["initial_rgb_sha256"] for x in pair}) != 1:
                raise RuntimeError(f"B arm pair mismatch {task}:{seed}")
            records[(task, seed)] = pair
    return {"pairs": len(records)}


def paired_report(reference: dict, arms: dict[str, dict], tasks: list[str], seeds: range,
                  output: Path, comparison_name: str) -> dict:
    from scipy.stats import binomtest
    rng = __import__("numpy").random.default_rng(20261009)
    rows = []
    for task in tasks:
        for seed in seeds:
            ref = reference[(task, seed)]
            row = {"task": task, "seed": seed, "reference_success": bool(ref["success"])}
            for arm, records in arms.items():
                row[arm + "_success"] = bool(records[(task, seed)]["success"])
            rows.append(row)
    report = {"comparison": comparison_name, "pairs": len(rows), "tasks": {}, "overall": {}}

    def summarize(group, arm):
        n = len(group)
        a = sum(int(x[arm + "_success"]) for x in group)
        b = sum(int(x["reference_success"]) for x in group)
        rescue = sum(not x["reference_success"] and x[arm + "_success"] for x in group)
        harm = sum(x["reference_success"] and not x[arm + "_success"] for x in group)
        p = float(binomtest(min(rescue, harm), rescue + harm, 0.5).pvalue) if rescue + harm else 1.0
        return {"reference_successes": b, "arm_successes": a, "n": n,
                "reference_rate": b / n if n else None, "arm_rate": a / n if n else None,
                "rescue": int(rescue), "harm": int(harm), "net": int(rescue - harm),
                "delta_rate": (a - b) / n if n else None, "mcnemar_exact_two_sided_p": p}

    for arm in arms:
        report["overall"][arm] = summarize(rows, arm)
        report["tasks"][arm] = {task: summarize([x for x in rows if x["task"] == task], arm)
                                for task in tasks}
        # Paired bootstrap CI, resampling seeds within each task.
        boot = []
        for _ in range(2000):
            sampled = []
            for task in tasks:
                sub = [x for x in rows if x["task"] == task]
                ix = rng.integers(0, len(sub), len(sub))
                sampled.extend(sub[j] for j in ix)
            boot.append(npmean([int(x[arm + "_success"]) - int(x["reference_success"]) for x in sampled]))
        report["overall"][arm]["paired_bootstrap_95ci_delta_rate"] = [
            float(__import__("numpy").quantile(boot, 0.025)),
            float(__import__("numpy").quantile(boot, 0.975)),
        ]
    ordered = sorted(((report["overall"][arm]["mcnemar_exact_two_sided_p"], arm) for arm in arms), key=lambda x: x[0])
    previous = 0.0
    m = len(ordered)
    for rank, (p, arm) in enumerate(ordered):
        adjusted = min(1.0, max(previous, (m - rank) * p))
        report["overall"][arm]["mcnemar_holm_adjusted_p"] = adjusted
        previous = adjusted
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def npmean(values):
    import numpy as np
    return float(np.mean(values))


def launch_b_formal() -> None:
    artifact = ART / "B_target_region"
    if not (artifact / "PREFLIGHT_PASS.json").is_file():
        raise RuntimeError("B preflight has not passed")
    workers = {f"l11abc_b_gpu{g}_w{w}": {"gpu": g, "jobs": []}
               for g in (1, 2, 3) for w in (1, 2, 3)}
    names = list(workers); idx = 0
    for task in TASKS:
        for lo in range(105, 200, 10):
            hi = min(lo + 9, 199)
            workers[names[idx % 9]]["jobs"].append(f"{task}={lo}-{hi}")
            idx += 1
    sessions, exits = [], []
    for name, cfg in workers.items():
        gpu = cfg["gpu"]
        body = " ".join(map(shlex.quote, [str(ROOT / "scripts/run_l11_abc_region_worker.sh"), str(gpu),
                    str(artifact), str(SNAPS), name, ",".join(cfg["jobs"])]))
        lp = ART / "logs" / f"{name}.log"; ep = ART / "logs" / f"{name}.exit"
        launch(name, gpu, body, lp, ep); sessions.append(name); exits.append(ep)
    wait_workers("B-target-region-formal", sessions, exits)
    validate_b(artifact, seeds=range(100, 200))
    reference, arms = {}, {"target_enriched": {}, "target_sparing": {}}
    for task in TASKS:
        for seed in range(100, 200):
            reference[(task, seed)] = json.loads((FULL / "episodes" / task / "full" / f"episode_{seed:03d}_summary.json").read_text())
            for arm in arms:
                arms[arm][(task, seed)] = json.loads((artifact / "episodes" / task / arm / f"episode_{seed:03d}_summary.json").read_text())
    report = paired_report(reference, arms, TASKS, range(100, 200), artifact / "paired_results_vs_full.json", "GT target controls vs canonical Full L11-Matched")
    log("B_RESULTS " + json.dumps(report["overall"], sort_keys=True))
    log("B_FINAL_PASS: 4 tasks x 100 seeds x 2 new arms; paired with existing Full")


def c_preflight() -> None:
    artifact = ART / "C_reconstruction"
    snaproot = ART / "C_fresh_snapshots"
    # First call creates the common lock; then run the remaining task pilots in parallel.
    jobs = [(1, TASKS[0], "open"), (2, TASKS[1], "close"),
            (3, TASKS[2], "pick"), (1, TASKS[3], "move")]
    sessions, exits = [], []
    gpu, task, suffix = jobs[0]
    session = f"l11abc_c_pre_{suffix}"
    body = " ".join(map(shlex.quote, [str(ROOT / "scripts/run_l11_abc_reconstruction_worker.sh"), str(gpu),
                str(artifact), str(snaproot), session, "preflight", f"{task}=300-304"]))
    lp = ART / "logs" / f"{session}.log"; ep = ART / "logs" / f"{session}.exit"
    launch(session, gpu, body, lp, ep); sessions.append(session); exits.append(ep)
    wait_workers("C-preflight-lock", sessions, exits)
    for gpu, task, suffix in jobs[1:]:
        session = f"l11abc_c_pre_{suffix}"
        body = " ".join(map(shlex.quote, [str(ROOT / "scripts/run_l11_abc_reconstruction_worker.sh"), str(gpu),
                    str(artifact), str(snaproot), session, "preflight", f"{task}=300-304"]))
        lp = ART / "logs" / f"{session}.log"; ep = ART / "logs" / f"{session}.exit"
        launch(session, gpu, body, lp, ep); sessions.append(session); exits.append(ep)
    wait_workers("C-preflight", sessions, exits)
    validate_c(artifact, seeds=range(300, 305))
    (artifact / "PREFLIGHT_PASS.json").write_text(json.dumps({"status": "PASS", "episodes": 60}, indent=2) + "\n")


def validate_c(artifact: Path, seeds) -> dict:
    n = 0
    for task in TASKS:
        for seed in seeds:
            rows = []
            for arm in ("full", "zero", "clean_global_mean"):
                path = artifact / "episodes" / task / arm / f"episode_{seed:03d}_summary.json"
                row = json.loads(path.read_text())
                if row.get("technical_pass") is not True:
                    raise RuntimeError(f"C technical audit failed {path}")
                expected_depth = 8 if "drawer" in task else 0
                if row.get("rendering", {}).get("depth") != expected_depth:
                    raise RuntimeError(f"C renderer mismatch {path}")
                rows.append(row)
            for field in ("canonical_snapshot_sha256", "initial_state_sha256", "initial_rgb_sha256", "instruction"):
                if len({r[field] for r in rows}) != 1:
                    raise RuntimeError(f"C paired mismatch {task}:{seed}:{field}")
            n += 1
    return {"paired_scenes": n}


def launch_c_formal() -> None:
    artifact = ART / "C_reconstruction"
    snaproot = ART / "C_fresh_snapshots"
    if not (artifact / "PREFLIGHT_PASS.json").is_file():
        raise RuntimeError("C preflight has not passed")
    workers = {f"l11abc_c_gpu{g}_w{w}": {"gpu": g, "jobs": []}
               for g in (1, 2, 3) for w in (1, 2, 3)}
    names = list(workers); idx = 0
    for task in TASKS:
        for lo in range(305, 500, 20):
            hi = min(lo + 19, 499)
            workers[names[idx % 9]]["jobs"].append(f"{task}={lo}-{hi}")
            idx += 1
    sessions, exits = [], []
    for name, cfg in workers.items():
        gpu = cfg["gpu"]
        body = " ".join(map(shlex.quote, [str(ROOT / "scripts/run_l11_abc_reconstruction_worker.sh"), str(gpu),
                    str(artifact), str(snaproot), name, "formal", ",".join(cfg["jobs"])]))
        lp = ART / "logs" / f"{name}.log"; ep = ART / "logs" / f"{name}.exit"
        launch(name, gpu, body, lp, ep); sessions.append(name); exits.append(ep)
    wait_workers("C-reconstruction-formal", sessions, exits)
    validate_c(artifact, seeds=range(300, 500))
    reference, arms = {}, {"zero": {}, "clean_global_mean": {}}
    for task in TASKS:
        for seed in range(300, 500):
            reference[(task, seed)] = json.loads((artifact / "episodes" / task / "full" / f"episode_{seed:03d}_summary.json").read_text())
            for arm in arms:
                arms[arm][(task, seed)] = json.loads((artifact / "episodes" / task / arm / f"episode_{seed:03d}_summary.json").read_text())
    report = paired_report(reference, arms, TASKS, range(300, 500), artifact / "paired_reconstruction_results.json", "reconstruction ablations vs harmonic Full")
    log("C_RESULTS " + json.dumps(report["overall"], sort_keys=True))
    log("C_FINAL_PASS: 4 tasks x 200 seeds x 3 arms = 2400 paired arm episodes")


def main() -> None:
    os.chdir(ROOT)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log("PIPELINE START; GPU allowlist fixed to 1,2,3; 3 workers per card")
    wait_state_library()
    state_count = run_a1()
    log(f"A1_DONE sampled_states={state_count}")
    run_a2()
    log("A_DONE spatial support and same-state budget response")
    b_preflight()
    launch_b_formal()
    log("B_DONE target-enriched versus target-sparing")
    c_preflight()
    launch_c_formal()
    log("PIPELINE_COMPLETE A/B/C; see artifacts/l11_abc_mechanism_v1")


if __name__ == "__main__":
    try:
        import numpy as np  # noqa: F401
        main()
    except BaseException as exc:
        ART.mkdir(parents=True, exist_ok=True)
        log(f"PIPELINE_FAILED: {type(exc).__name__}: {exc}")
        raise
