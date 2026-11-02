from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _max_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))


def _recompute(clean: np.ndarray, cf: np.ndarray, alpha: float, bandwidth_scale: float):
    clean = np.asarray(clean[:, :, :6], dtype=np.float32)
    cf = np.asarray(cf[:, :, :6], dtype=np.float32)
    n = clean.shape[0]
    scale = np.std(np.concatenate([clean, cf], axis=0), axis=0, ddof=1)
    valid = np.isfinite(scale) & (scale >= 1e-6)
    bandwidth = np.maximum(scale * bandwidth_scale, 1e-6)
    norm = np.log(n * bandwidth * np.sqrt(np.float32(2.0 * np.pi)))
    clean_diff = (clean[:, None] - clean[None, :]) / bandwidth[None, None]
    cf_diff = (clean[:, None] - cf[None, :]) / bandwidth[None, None]
    log_pc_terms = logsumexp(-0.5 * clean_diff * clean_diff, axis=1) - norm
    log_pn_terms = logsumexp(-0.5 * cf_diff * cf_diff, axis=1) - norm
    log_pc = np.sum(np.where(valid[None], log_pc_terms, 0.0), axis=(1, 2))
    log_pn = np.sum(np.where(valid[None], log_pn_terms, 0.0), axis=(1, 2))
    score = log_pc - log_pn
    score = score - np.mean(score)
    score_std = np.std(score, ddof=1)
    if score_std > 1e-12:
        score = score / score_std
    if not np.any(valid):
        score = np.zeros_like(score)
    logits = alpha * score
    weights = np.exp(logits - np.max(logits))
    weights = weights / np.sum(weights)
    return log_pc, log_pn, score, weights


def _stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    artifact = args.artifact.resolve()
    out_path = artifact / "mechanism_summary.json"
    if out_path.exists():
        raise FileExistsError(out_path)
    manifest = json.loads((artifact / "diagnostic_manifest.json").read_text())
    replay = json.loads((artifact / "replay_report.json").read_text())
    replay_by_name = {row["trace_file"]: row for row in replay["traces"]}
    tasks = {(row["plan"], int(row["task_index"])): row for row in manifest["selection"]["strata"]}
    episodes = {}
    for plan in {row["plan"] for row in tasks.values()}:
        path = artifact / plan / "episodes.jsonl"
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                episodes[(plan, int(row["task_index"]), row["arm"])] = row

    formula_errors = {key: [] for key in ("log_pc", "log_pn", "normalized_score", "weights")}
    vtcg_metrics: dict[tuple[str, int], dict[str, list[float]]] = {}
    image_pairs = {}
    finite_failures = 0
    for path in sorted((artifact / "traces").rglob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            plan = path.parent.name
            task_index = int(np.asarray(data["task_index"]).item())
            arm = str(np.asarray(data["arm"]).item())
            replan_index = int(np.asarray(data["replan_index"]).item())
            if replan_index == 0:
                image_pairs[(plan, task_index, arm)] = (
                    str(np.asarray(data["image_sha256"]).item()),
                    str(np.asarray(data["wrist_image_sha256"]).item()),
                    str(np.asarray(data["state_sha256"]).item()),
                )
            if arm != "pcd":
                continue

            clean = np.asarray(data["vtcg_actions_clean"], dtype=np.float32)
            cf = np.asarray(data["vtcg_actions_cf"], dtype=np.float32)
            noise = np.asarray(data["vtcg_noise"], dtype=np.float32)
            weights = np.asarray(data["vtcg_weights"], dtype=np.float32)
            finite_failures += not (
                np.isfinite(clean).all()
                and np.isfinite(cf).all()
                and np.isfinite(noise).all()
                and np.isfinite(weights).all()
            )
            trace = json.loads(str(np.asarray(data["trace_json"]).item()))
            expected = {
                "log_pc": np.asarray(data["vtcg_log_pc"], dtype=np.float32),
                "log_pn": np.asarray(data["vtcg_log_pn"], dtype=np.float32),
                "normalized_score": np.asarray(data["vtcg_normalized_score"], dtype=np.float32),
                "weights": weights,
            }
            actual = dict(
                zip(
                    ("log_pc", "log_pn", "normalized_score", "weights"),
                    _recompute(
                        clean,
                        cf,
                        float(trace["alpha"]),
                        float(trace["bandwidth_scale"]),
                    ),
                )
            )
            for key in formula_errors:
                formula_errors[key].append(_max_error(actual[key], expected[key]))

            key = (plan, task_index)
            metric = vtcg_metrics.setdefault(
                key,
                {name: [] for name in ("ess", "max_weight", "candidate0_probability", "chosen0", "candidate_spread", "cf_separation")},
            )
            ess = 1.0 / float(np.sum(weights.astype(np.float64) ** 2))
            selected = int(np.asarray(data["vtcg_chosen_index"]).item())
            clean6 = clean[:, :, :6].astype(np.float64)
            cf6 = cf[:, :, :6].astype(np.float64)
            spread = np.linalg.norm(clean6 - clean6[0], axis=(1, 2))
            separation = np.linalg.norm(clean6 - cf6, axis=(1, 2))
            metric["ess"].append(ess)
            metric["max_weight"].append(float(weights.max()))
            metric["candidate0_probability"].append(float(weights[0]))
            metric["chosen0"].append(float(selected == 0))
            metric["candidate_spread"].append(float(spread.max()))
            metric["cf_separation"].append(float(separation.mean()))

    pair_states_match = []
    for plan, task_index in tasks:
        vanilla = image_pairs[(plan, task_index, "vanilla")]
        vtcg = image_pairs[(plan, task_index, "pcd")]
        pair_states_match.append(vanilla == vtcg)

    source = Path(manifest["source_artifact"])
    changed_source_files = []
    for relative, expected in manifest["source_hashes_sha256"].items():
        if _sha256(source / relative) != expected:
            changed_source_files.append(relative)

    replay_rows = replay["traces"]
    pcd_rows = [row for row in replay_rows if row["arm"] == "pcd"]
    vanilla_rows = [row for row in replay_rows if row["arm"] == "vanilla"]
    pcd_index_matches = sum(bool(row["chosen_index_matches"]) for row in pcd_rows)
    action_errors = [float(row["action_max_abs_error"]) for row in pcd_rows]
    summary = {
        "protocol": "PI05_LIBERO_VTCG_MECHANISM_SUMMARY_V1",
        "artifact": str(artifact),
        "sample_pairs": len(tasks),
        "rollouts": len(episodes),
        "errors": sum(bool(row.get("error")) for row in episodes.values()),
        "nonfinite_trace_replans": finite_failures,
        "first_observation_pair_hash_matches": sum(pair_states_match),
        "first_observation_pairs": len(pair_states_match),
        "source_artifact_changed_files": changed_source_files,
        "fresh_process_replay": {
            "traces": len(replay_rows),
            "strict_passes": sum(bool(row["passed"]) for row in replay_rows),
            "vanilla_passes": sum(bool(row["passed"]) for row in vanilla_rows),
            "vanilla_traces": len(vanilla_rows),
            "vtcg_chosen_index_matches": pcd_index_matches,
            "vtcg_traces": len(pcd_rows),
            "vtcg_action_max_abs_error": _stats(action_errors),
            "tolerance": {"atol": replay["atol"], "rtol": replay["rtol"]},
        },
        "cached_candidate_score_replay": {
            "tolerance": args.tolerance,
            "max_abs_error": {key: max(values, default=0.0) for key, values in formula_errors.items()},
            "all_within_tolerance": all(
                max(values, default=0.0) <= args.tolerance
                for values in formula_errors.values()
            ),
        },
        "vtcg_selection_metrics": {
            "replans": sum(len(metrics["ess"]) for metrics in vtcg_metrics.values()),
            "overall": {
                name: _stats([value for metrics in vtcg_metrics.values() for value in metrics[name]])
                for name in next(iter(vtcg_metrics.values()))
            },
            "by_task": [
                {
                    "plan": plan,
                    "task_index": task_index,
                    "stratum": tasks[(plan, task_index)]["stratum"],
                    "vanilla_success": bool(episodes[(plan, task_index, "vanilla")]["success"]),
                    "vtcg_success": bool(episodes[(plan, task_index, "pcd")]["success"]),
                    "replans": len(metrics["ess"]),
                    "metrics": {name: _stats(values) for name, values in metrics.items()},
                }
                for (plan, task_index), metrics in sorted(vtcg_metrics.items())
            ],
        },
        "caveat": "Small outcome-blind mechanism sample; not an effect estimate. Fresh-process VTCG model outputs may differ numerically even when request/noise hashes match; cached-candidate score replay isolates the KDE/weight calculation.",
    }
    with out_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
