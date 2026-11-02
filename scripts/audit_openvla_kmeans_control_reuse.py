#!/usr/bin/env python3
"""Fail-closed reuse audit for the OpenVLA KMeans-granularity study controls.

Scans the two PRE-EXISTING artifacts and certifies, per (artifact, arm, task) shard:

  * complete seed coverage 200-299, no duplicates, no extra episodes
  * top-level success is a bool, technical_pass is true, steps > 0
  * protocol id, rendering, lambda, attention layer, constructor, execution identity
  * non-empty canonical-snapshot / initial-state / initial-rgb SHA and arrays file
  * for a deterministic seed sample: the in-trace matched-budget invariants

Then cross-checks that each (task, seed) cell shares identical initial-state,
initial-rgb, canonical-snapshot SHA and instruction across every reused arm.

Shards are cached under <artifact>/_audit_shards/, so an interrupted run resumes
without rescanning. CONTROL_REUSE_AUDIT.json is written only with --write, and
technical_pass is strict: any gap keeps it false.

Usage:
    python3 scripts/audit_openvla_kmeans_control_reuse.py [--jobs 8] [--write]
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KMEANS_ROOT = REPO / "artifacts/openvla_kmeans_budget_9task_200_299_v1"
REFERENCE = REPO / "artifacts/openvla_lambda1_extension_200_299_v1"
CONTROLS = REPO / "artifacts/openvla_budget_rule_9task_200_299_v1"
SEEDS = list(range(200, 300))
DEEP_SEEDS = (200, 250, 299)
EXPECTED_CONSTRUCTOR = "harmonic"
EXPECTED_EXECUTION = "contrast"
EXPECTED_LOCATION = "projector"
EXPECTED_K = 8
RT_DEPTH = {
    "google_robot_open_drawer": 8,
    "google_robot_close_drawer": 8,
    "google_robot_pick_coke_can": 0,
    "google_robot_move_near": 0,
    "google_robot_place_apple_in_closed_top_drawer": 0,
    "widowx_carrot_on_plate": 0,
    "widowx_put_eggplant_in_basket": 0,
    "widowx_spoon_on_towel": 0,
    "widowx_stack_cube": 0,
}
TASKS = list(RT_DEPTH)
SHARD_ROOT = KMEANS_ROOT / "_audit_shards"


def expected_rendering(task: str) -> dict:
    return {"shader_dir": "rt", "rt_max_path_depth": RT_DEPTH[task],
            "rt_samples_per_pixel": 32, "rt_use_denoiser": True}


def parse_control_arm(arm: str) -> tuple[str, float | int | None]:
    """Same names the frozen runner accepts: the control sets are fixed* and top_p*."""
    if arm == "lambda_100":
        return "matched", None
    head = arm.split("_l")[0]
    if head.startswith("fixed"):
        return "fixed", int(head[5:])
    if head.startswith("top_p"):
        return "top_p", int(head[5:]) / 100.0
    if head.startswith("random_fixed"):
        return "random_fixed", int(head[12:])
    raise ValueError(f"unknown control arm: {arm}")


def deep_trace_checks(trace: list, steps: int, arm: str) -> dict:
    """Rule-aware: only the matched rule has clusters, unions and a dynamic count."""
    rule, param = parse_control_arm(arm)
    trace = trace or []
    counts = [int(x.get("actual_selected_count", -1)) for x in trace]
    checks = {
        "trace_len_matches_steps": len(trace) == steps and steps > 0,
        "kmeans_K": all(x.get("kmeans_K") == EXPECTED_K for x in trace),
        "kmeans_seed": all(x.get("kmeans_seed") == 0 for x in trace),
        "mt_is_selected_count": all(c == int(x.get("m_t", -2)) for c, x in zip(counts, trace)),
        "selected_ids_exact": all(
            len(set(x.get("selected_token_ids", []))) == int(x.get("m_t", -1))
            and all(isinstance(i, int) and 0 <= i < 256 for i in x.get("selected_token_ids", []))
            for x in trace),
        "feature_equal": all(x.get("feature_equal") is True for x in trace),
        "non_target_bit_identical": all(x.get("non_target_bit_identical") is True for x in trace),
        "reconstruction_finite": all(x.get("reconstruction_finite") is True for x in trace),
        "intervention_location": all(x.get("intervention_location") == EXPECTED_LOCATION for x in trace),
        "lambda_exact": all(abs(float(x.get("lambda", -1)) - 1.0) < 1e-12 for x in trace),
        "seven_action_tokens": all(
            len(x.get("positive_token_ids", [])) == 7 and len(x.get("negative_token_ids", [])) == 7
            and len(x.get("final_token_ids", [])) == 7 for x in trace),
    }
    if not trace:
        return checks
    if rule == "matched":
        checks["coverage_mode"] = all(x.get("coverage_mode") == "own_state_matched_budget" for x in trace)
        checks["reference_union_size"] = all(
            len(set(x.get("reference_shr_token_ids", []))) == int(x.get("m_t", -1)) for x in trace)
        checks["matched_groups_exact"] = all(
            x.get("selected_group_ids") == x.get("matched_budget_group_ids")
            and bool(x.get("selected_group_ids"))
            and all(isinstance(i, int) and 0 <= i < EXPECTED_K for i in x.get("selected_group_ids", []))
            for x in trace)
    elif rule in ("fixed", "random_fixed"):
        checks["coverage_mode"] = all(x.get("coverage_mode") == f"fixed_top_{int(param)}" for x in trace)
        checks["fixed_budget_exact"] = all(int(x.get("m_t", -1)) == int(param) for x in trace)
        checks["no_cluster_metadata"] = all(
            not x.get("selected_group_ids") and not x.get("matched_budget_group_ids")
            and not x.get("reference_shr_token_ids") for x in trace)
    elif rule == "top_p":
        checks["coverage_mode"] = all(
            x.get("coverage_mode") == f"visual_top_p_{float(param):.2f}_clip_1_256" for x in trace)
        checks["top_p_threshold_exact"] = all(
            abs(float(x.get("top_p_threshold") or -1.0) - float(param)) < 1e-12 for x in trace)
        checks["top_p_unclipped_bounds"] = all(
            int(x.get("selection_min_count", -1)) == 1 and int(x.get("selection_max_count", -1)) == 256
            for x in trace)
        checks["top_p_no_clip_triggered"] = all(
            not x.get("lower_bound_triggered") and not x.get("upper_bound_triggered") for x in trace)
        checks["top_p_within_grid"] = all(1 <= int(x.get("m_t", 0)) <= 256 for x in trace)
        checks["no_cluster_metadata"] = all(
            not x.get("selected_group_ids") and not x.get("matched_budget_group_ids")
            and not x.get("reference_shr_token_ids") for x in trace)
    return checks


def audit_shard(job: tuple) -> dict:
    artifact_name, artifact, task, arm, expected_protocol = job
    out_dir = artifact / "episodes" / task / arm
    shard_path = SHARD_ROOT / f"{artifact_name}__{task}__{arm}.json"
    if shard_path.is_file():
        cached = json.loads(shard_path.read_text())
        cached["cached"] = True
        return cached
    result = {"artifact": artifact_name, "task": task, "arm": arm,
              "expected_protocol": expected_protocol, "n_found": 0, "n_success": 0,
              "seeds_found": [], "cell_errors": [], "deep_checked": [],
              "sha": {}, "instruction": {}, "sum_success_steps": 0, "sum_mt": 0.0,
              "sum_runtime": 0.0, "mt_n": 0, "technical_pass": False, "cached": False}
    for seed in SEEDS:
        summary = out_dir / f"episode_{seed:03d}_summary.json"
        if not summary.is_file():
            continue
        try:
            data = json.loads(summary.read_text())
        except Exception as exc:                                   # noqa: BLE001
            result["cell_errors"].append({"seed": seed, "error": f"unreadable: {exc!r}"})
            continue
        failed = []
        arrays = out_dir / f"episode_{seed:03d}_arrays.npz"

        def need(condition: bool, name: str) -> None:
            if not condition:
                failed.append(name)

        need(data.get("task") == task, "task")
        need(data.get("arm") == arm, "arm")
        need(data.get("seed") == seed, "seed")
        need(data.get("protocol_id") == expected_protocol, "protocol_id")
        need(type(data.get("success")) is bool, "success_bool")
        need(data.get("technical_pass") is True, "technical_pass")
        need(isinstance(data.get("steps"), int) and data["steps"] > 0, "steps_positive")
        need(data.get("failure_reason") is None or isinstance(data.get("failure_reason"), str), "failure_reason")
        need(float(data.get("lambda", -1)) == 1.0, "lambda")
        need(data.get("attention_layers") == [11], "attention_layers")
        need(data.get("counterfactual_constructor") == EXPECTED_CONSTRUCTOR, "constructor")
        need(data.get("guidance_execution") == EXPECTED_EXECUTION, "execution")
        need(data.get("rendering") == expected_rendering(task), "rendering")
        for key in ("canonical_snapshot_sha256", "initial_state_sha256", "initial_rgb_sha256"):
            need(isinstance(data.get(key), str) and len(data[key]) == 64, key)
        need(isinstance(data.get("instruction"), str) and bool(data["instruction"]), "instruction")
        need(arrays.is_file() and arrays.stat().st_size > 0, "arrays_file")
        mt = data.get("mean_selected_token_count")
        need(isinstance(mt, (int, float)) and math.isfinite(float(mt)) and float(mt) > 0, "mean_m_t")
        if seed in DEEP_SEEDS:
            deep = deep_trace_checks(data.get("selector_trace"), int(data.get("steps", 0)), arm)
            result["deep_checked"].append({"seed": seed, "checks": deep})
            failed.extend(f"deep:{k}" for k, v in deep.items() if not v)
        result["seeds_found"].append(seed)
        result["n_found"] += 1
        result["n_success"] += int(bool(data["success"]))
        result["sum_success_steps"] += int(data.get("steps", 0))
        if isinstance(mt, (int, float)) and math.isfinite(float(mt)):
            result["sum_mt"] += float(mt)
            result["mt_n"] += 1
        if isinstance(data.get("runtime_seconds"), (int, float)):
            result["sum_runtime"] += float(data["runtime_seconds"])
        result["sha"][str(seed)] = [data.get("canonical_snapshot_sha256"),
                                    data.get("initial_state_sha256"),
                                    data.get("initial_rgb_sha256")]
        result["instruction"][str(seed)] = data.get("instruction")
        if failed:
            result["cell_errors"].append({"seed": seed, "failed": sorted(set(failed))})
    missing = sorted(set(SEEDS) - set(result["seeds_found"]))
    duplicates = len(result["seeds_found"]) - len(set(result["seeds_found"]))
    result["missing_seeds"] = missing
    result["duplicates"] = duplicates
    result["technical_pass"] = (not missing and not duplicates and not result["cell_errors"]
                                and result["n_found"] == len(SEEDS))
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = shard_path.with_name(shard_path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result))
    os.replace(temporary, shard_path)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    lock = json.loads((KMEANS_ROOT / "EXPERIMENT_LOCK.json").read_text())
    control_arms = sorted(a for a in lock["control_arms"] if a != "lambda_100")
    reference_protocol = "OPENVLA_LAMBDA1_EXTENSION_200_299_V1"
    control_protocol = "OPENVLA_BUDGET_RULE_9TASK_200_299_V1"
    jobs = [("reference", REFERENCE, task, "lambda_100", reference_protocol) for task in TASKS]
    jobs += [("controls", CONTROLS, task, arm, control_protocol) for arm in control_arms for task in TASKS]

    started = time.time()
    with mp.Pool(args.jobs) as pool:
        shards = []
        for index, shard in enumerate(pool.imap_unordered(audit_shard, jobs), start=1):
            shards.append(shard)
            print(f"[{index}/{len(jobs)}] {shard['artifact']:9s} {shard['task'][:28]:28s} "
                  f"{shard['arm']:16s} n={shard['n_found']:3d} ok={shard['n_success']:3d} "
                  f"pass={shard['technical_pass']} cached={shard['cached']}", flush=True)
    policy = lock
    shards.sort(key=lambda s: (s["artifact"], s["task"], s["arm"]))

    # ---- pairing: identical initial identity for every (task, seed) across arms ----
    pairing_mismatches, cells_checked = [], 0
    by_task: dict[str, dict[int, list[tuple[str, str, list, str]]]] = {}
    for shard in shards:
        tag = "reference" if shard["artifact"] == "reference" else shard["arm"]
        for seed_key, sha in shard["sha"].items():
            seed = int(seed_key)
            by_task.setdefault(shard["task"], {}).setdefault(seed, []).append(
                (tag, shard["artifact"], sha, shard["instruction"].get(seed_key)))
    for task, seeds in sorted(by_task.items()):
        for seed, entries in sorted(seeds.items()):
            cells_checked += 1
            if len(entries) < len(control_arms) + 1:
                pairing_mismatches.append({"task": task, "seed": seed,
                                           "error": "incomplete arm set", "arms": len(entries)})
                continue
            shapes = {(tuple(e[2]), e[3]) for e in entries}
            if len(shapes) != 1:
                pairing_mismatches.append({"task": task, "seed": seed, "error": "SHA/instruction mismatch",
                                           "arms": sorted(e[0] for e in entries)})

    # ---- excluded diagnostics inside the control artifact ----
    diagnostics = {}
    diag_root = CONTROLS / "episodes"
    for task in TASKS:
        arm_dir = diag_root / task / "lambda_100"
        if arm_dir.is_dir():
            seeds = sorted(int(p.name.split("_")[1]) for p in arm_dir.glob("episode_*_summary.json"))
            diagnostics[task] = {"n": len(seeds), "seeds": seeds,
                                 "reason": "diagnostic arm; excluded from every comparison"}

    budget = {}
    for shard in shards:
        key = "reference_lambda_100" if shard["artifact"] == "reference" else shard["arm"]
        budget.setdefault(shard["task"], {})[key] = {
            "n": shard["n_found"], "successes": shard["n_success"],
            "success_rate": round(shard["n_success"] / shard["n_found"], 6) if shard["n_found"] else None,
            "mean_m_t": round(shard["sum_mt"] / shard["mt_n"], 4) if shard["mt_n"] else None,
            "mean_runtime_seconds": round(shard["sum_runtime"] / shard["n_found"], 2) if shard["n_found"] else None,
        }

    failures = [{"artifact": s["artifact"], "task": s["task"], "arm": s["arm"],
                 "missing_seeds": s["missing_seeds"], "duplicates": s["duplicates"],
                 "cell_errors": s["cell_errors"][:20], "n_cell_errors": len(s["cell_errors"]),
                 "n_found": s["n_found"]}
                for s in shards if not s["technical_pass"]]
    reference_ok = all(s["technical_pass"] for s in shards if s["artifact"] == "reference")
    controls_ok = all(s["technical_pass"] for s in shards if s["artifact"] == "controls")
    n_reference = sum(s["n_found"] for s in shards if s["artifact"] == "reference")
    n_controls = sum(s["n_found"] for s in shards if s["artifact"] == "controls")
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generator": "scripts/audit_openvla_kmeans_control_reuse.py",
        "elapsed_seconds": round(time.time() - started, 1),
        "scope": {
            "reference": str(REFERENCE.relative_to(REPO)) + " (Matched-K8, lambda=1.0)",
            "controls": str(CONTROLS.relative_to(REPO)) + " (Fixed16-64 and unclipped TopP75-90)",
            "tasks": TASKS, "seeds": [SEEDS[0], SEEDS[-1]], "deep_sample_seeds": list(DEEP_SEEDS),
            "expected_reference_episodes": 900, "expected_control_episodes": 9900,
            "expected_total": 10800,
        },
        "counts": {"reference_episodes": n_reference, "control_episodes": n_controls,
                   "total_episodes": n_reference + n_controls},
        "reference_technical_pass": reference_ok,
        "controls_technical_pass": controls_ok,
        "pairing": {"cells_checked": cells_checked, "expected_cells": 900,
                    "all_arms_share_initial_sha_and_instruction": not pairing_mismatches,
                    "mismatches": pairing_mismatches[:50], "n_mismatches": len(pairing_mismatches)},
        "excluded_diagnostics": diagnostics,
        "budget_diagnostics": budget,
        "shard_summary": [{"artifact": s["artifact"], "task": s["task"], "arm": s["arm"],
                           "n": s["n_found"], "successes": s["n_success"],
                           "technical_pass": s["technical_pass"]} for s in shards],
        "failures": failures,
        "technical_pass": bool(reference_ok and controls_ok and not pairing_mismatches
                               and n_reference == 900 and n_controls == 9900
                               and cells_checked == 900),
        "notes": [
            "Reuse gate only: it certifies that historical controls are intact, complete and "
            "identity-paired. It is not an efficacy result.",
            "The control artifact's lambda_100 arm is diagnostic and excluded everywhere.",
            "Shards in _audit_shards/ are cached; delete them to force a full rescan.",
        ],
    }
    print(json.dumps({k: payload[k] for k in ("counts", "reference_technical_pass",
                                              "controls_technical_pass", "pairing",
                                              "technical_pass", "elapsed_seconds")
                      if k != "pairing"}, indent=2))
    print(f"pairing: {payload['pairing']['cells_checked']} cells, "
          f"{payload['pairing']['n_mismatches']} mismatches")
    print(f"failures: {len(failures)}")
    if args.write:
        out = KMEANS_ROOT / "CONTROL_REUSE_AUDIT.json"
        if out.exists():
            out.replace(out.with_suffix(".json.prev"))
        temporary = out.with_name(out.name + f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, out)
        print(f"wrote {out}  technical_pass={payload['technical_pass']}")
    return 0 if payload["technical_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
