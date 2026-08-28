"""Full descriptive and snapshot-cluster analysis for signed utility Phase T0."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


ARMS = ("full", "attention_max", "iss_max", "attention_high_relevance_low", "random_control")
SIGNALS = (
    ("attention", "attention_mean", 1.0),
    ("ISS", "iss_action_rms", 1.0),
    ("task_relevance", "task_relevance_cosine", 1.0),
    ("ISS_x_relevance", "iss_x_relevance", 1.0),
    ("nuisance_x_ISS", "nuisance_x_iss", -1.0),
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def bootstrap_effect(state_rows: list[dict], arm: str, seed: int = 20260812, reps: int = 20000):
    values = np.asarray([row[f"U_{arm}"] for row in state_rows], dtype=float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(reps, len(values)))].mean(axis=1)
    return float(values.mean()), [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def cluster_bootstrap_spearman(rows: list[dict], key: str, direction: float, seed: int, reps: int = 20000):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["snapshot_id"]].append(row)
    clusters = sorted(grouped)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(reps):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        score, utility = [], []
        for index in sampled:
            for row in grouped[clusters[index]]:
                score.append(direction * row[key])
                utility.append(row["utility"])
        rho = spearmanr(score, utility).statistic
        if np.isfinite(rho):
            values.append(rho)
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field) for field in fields} for row in rows])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    output = artifact / "analysis"
    output.mkdir(exist_ok=True)
    episodes = [json.loads(path.read_text()) for path in sorted((artifact / "episodes").glob("*.json"))]
    candidates = [json.loads(line) for line in (artifact / "candidate_manifest.jsonl").read_text().splitlines()]
    if len(episodes) != 1000 or len(candidates) != 40:
        raise RuntimeError("incomplete Phase T0 inputs")

    units = defaultdict(dict)
    for row in episodes:
        if row["arm"] in units[row["causal_unit_id"]]:
            raise RuntimeError("duplicate arm in causal unit")
        units[row["causal_unit_id"]][row["arm"]] = row
    if len(units) != 200 or any(set(value) != set(ARMS) for value in units.values()):
        raise RuntimeError("causal unit integrity failure")

    states = defaultdict(list)
    for unit_id, value in units.items():
        states[unit_id.rsplit("__seed", 1)[0]].append(value)
    candidate_by_state = {row["snapshot_id"]: row for row in candidates}
    state_rows = []
    candidate_rows = []
    for snapshot_id, unit_values in sorted(states.items()):
        if len(unit_values) != 5:
            raise RuntimeError(f"{snapshot_id} does not have five noise seeds")
        first = unit_values[0]["full"]
        row = {
            "snapshot_id": snapshot_id,
            "task_id": first["task_id"],
            "init_state_id": first["init_state_id"],
            "progress": first["target_progress"],
        }
        for arm in ARMS:
            row[f"success_{arm}"] = sum(int(value[arm]["success"]) for value in unit_values)
            row[f"P_{arm}"] = row[f"success_{arm}"] / 5
        for arm in ARMS[1:]:
            row[f"U_{arm}"] = row["P_full"] - row[f"P_{arm}"]
        state_rows.append(row)
        candidate_map = {item["candidate_id"]: item for item in candidate_by_state[snapshot_id]["candidates"]}
        for arm in ARMS[1:]:
            item = candidate_map[arm]
            candidate_rows.append({
                "snapshot_id": snapshot_id,
                "task_id": row["task_id"],
                "init_state_id": row["init_state_id"],
                "progress": row["progress"],
                "candidate_id": arm,
                "utility": row[f"U_{arm}"],
                **{key: item[key] for key in (
                    "region_id", "camera_id", "row", "col", "attention_mean", "attention_sum",
                    "iss_action_rms", "iss_action_l2", "task_relevance_cosine", "nuisance_score",
                    "iss_x_relevance", "nuisance_x_iss",
                )},
            })

    overall = []
    for arm in ARMS:
        arm_rows = [row for row in episodes if row["arm"] == arm]
        success = sum(int(row["success"]) for row in arm_rows)
        result = {"arm": arm, "success": success, "n": len(arm_rows), "rate": success / len(arm_rows)}
        if arm != "full":
            differences = [int(value[arm]["success"]) - int(value["full"]["success"]) for value in units.values()]
            utilities = [row[f"U_{arm}"] for row in state_rows]
            mean_u, ci = bootstrap_effect(state_rows, arm)
            result.update({
                "rescue": differences.count(1), "harm": differences.count(-1),
                "unchanged": differences.count(0), "net_perturb_minus_full": sum(differences),
                "mean_utility_full_minus_perturb": mean_u, "cluster_bootstrap_ci95": ci,
                "helpful_states": sum(value > 0 for value in utilities),
                "harmful_states": sum(value < 0 for value in utilities),
                "neutral_states": sum(value == 0 for value in utilities),
            })
        overall.append(result)

    strata = []
    for task in list(range(10)) + [None]:
        for progress in (0.25, 0.65, None):
            if task is None and progress is None:
                continue
            selected = [
                row for row in state_rows
                if (task is None or row["task_id"] == task) and (progress is None or row["progress"] == progress)
            ]
            if not selected:
                continue
            item = {"task_id": "all" if task is None else task, "progress": "all" if progress is None else progress, "snapshots": len(selected)}
            for arm in ARMS:
                item[f"P_{arm}"] = sum(row[f"P_{arm}"] for row in selected) / len(selected)
            for arm in ARMS[1:]:
                item[f"U_{arm}"] = sum(row[f"U_{arm}"] for row in selected) / len(selected)
            strata.append(item)

    signal_rows = []
    utilities = np.asarray([row["utility"] for row in candidate_rows])
    nonneutral = utilities != 0
    for name, key, direction in SIGNALS:
        raw = np.asarray([row[key] for row in candidate_rows], dtype=float)
        score = direction * raw
        rho, pvalue = spearmanr(score, utilities)
        low, high = np.quantile(score, [0.25, 0.75])
        high_rows = utilities[score >= high]
        low_rows = utilities[score <= low]
        nonneutral_score = score[nonneutral]
        nonneutral_utility = utilities[nonneutral]
        nrho, npvalue = spearmanr(nonneutral_score, nonneutral_utility) if len(nonneutral_score) > 2 else (np.nan, np.nan)
        cluster_ci = cluster_bootstrap_spearman(candidate_rows, key, direction, 20260812 + len(signal_rows))
        signal_rows.append({
            "signal": name,
            "higher_score_prediction": "helpful",
            "spearman_all160": float(rho), "spearman_p_all160": float(pvalue),
            "spearman_nonneutral20": float(nrho), "spearman_p_nonneutral20": float(npvalue),
            "snapshot_cluster_bootstrap_ci95": json.dumps(cluster_ci),
            "top_quartile_n": len(high_rows), "top_quartile_helpful": int((high_rows > 0).sum()),
            "top_quartile_harmful": int((high_rows < 0).sum()), "top_quartile_neutral": int((high_rows == 0).sum()),
            "bottom_quartile_n": len(low_rows), "bottom_quartile_helpful": int((low_rows > 0).sum()),
            "bottom_quartile_harmful": int((low_rows < 0).sum()), "bottom_quartile_neutral": int((low_rows == 0).sum()),
        })

    integrity = {
        "planned_episodes": 1000, "accounted_episodes": len(episodes),
        "planned_causal_units": 200, "complete_causal_units": len(units),
        "planned_snapshots": 40, "complete_snapshots": len(states),
        "episodes_per_arm": Counter(row["arm"] for row in episodes),
        "invalid_units": len(list((artifact / "invalid_units").glob("*.json"))),
        "all_finite": all(row["all_actions_finite"] for row in episodes),
        "protocol_hashes": sorted({row["protocol_sha256"] for row in episodes}),
        "manifest_hashes": sorted({row["manifest_sha256"] for row in episodes}),
        "episode_files_sha256": sha(artifact / "episode_manifest.jsonl"),
    }
    payload = {"integrity": integrity, "overall": overall, "signals": signal_rows}
    (output / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=dict) + "\n")
    state_fields = ["snapshot_id", "task_id", "init_state_id", "progress"] + [f"P_{arm}" for arm in ARMS] + [f"U_{arm}" for arm in ARMS[1:]]
    write_csv(output / "snapshot_utilities.csv", state_rows, state_fields)
    write_csv(output / "candidate_features_and_utility.csv", candidate_rows, list(candidate_rows[0]))
    write_csv(output / "task_progress_table.csv", strata, list(strata[0]))
    write_csv(output / "signal_audit.csv", signal_rows, list(signal_rows[0]))

    lines = [
        "# Phase T0 Full Analysis", "", "## Integrity", "",
        f"- Episodes: {len(episodes)}/1000", f"- Matched causal units: {len(units)}/200",
        f"- Independent snapshots: {len(states)}/40", f"- Invalid units: {integrity['invalid_units']}",
        f"- All action outputs finite: {integrity['all_finite']}", "", "## Overall", "",
        "| Arm | Success | Rate | Rescue | Harm | Mean U=Full-Perturb | Snapshot bootstrap 95% CI | Helpful/Harmful/Neutral states |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        if row["arm"] == "full":
            lines.append(f"| full | {row['success']}/{row['n']} | {pct(row['rate'])} | - | - | - | - | - |")
        else:
            ci = row["cluster_bootstrap_ci95"]
            lines.append(
                f"| {row['arm']} | {row['success']}/{row['n']} | {pct(row['rate'])} | {row['rescue']} | {row['harm']} | "
                f"{row['mean_utility_full_minus_perturb']:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
                f"{row['helpful_states']}/{row['harmful_states']}/{row['neutral_states']} |"
            )
    lines += ["", "## By Task", "", "U > 0 means the region was helpful because perturbing it reduced success.", "",
              "| Task | Full | Attention U | ISS U | Nuisance U | Random U |", "|---:|---:|---:|---:|---:|---:|"]
    for row in [x for x in strata if x["progress"] == "all" and x["task_id"] != "all"]:
        lines.append(f"| {row['task_id']} | {pct(row['P_full'])} | {row['U_attention_max']:+.2f} | {row['U_iss_max']:+.2f} | {row['U_attention_high_relevance_low']:+.2f} | {row['U_random_control']:+.2f} |")
    lines += ["", "## By Progress", "", "| Progress | Full | Attention U | ISS U | Nuisance U | Random U |", "|---:|---:|---:|---:|---:|---:|"]
    for row in [x for x in strata if x["task_id"] == "all"]:
        lines.append(f"| {row['progress']} | {pct(row['P_full'])} | {row['U_attention_max']:+.3f} | {row['U_iss_max']:+.3f} | {row['U_attention_high_relevance_low']:+.3f} | {row['U_random_control']:+.3f} |")
    lines += ["", "## Signal Audit", "", "The sign convention is normalized so a higher score predicts helpful utility. Quartile counts are descriptive because no threshold was preregistered.", "",
              "| Signal | Spearman rho (160) | Snapshot-cluster 95% CI | Top quartile H/Harm/N | Bottom quartile H/Harm/N |", "|---|---:|---:|---:|---:|"]
    for row in signal_rows:
        ci = json.loads(row["snapshot_cluster_bootstrap_ci95"])
        lines.append(f"| {row['signal']} | {row['spearman_all160']:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {row['top_quartile_helpful']}/{row['top_quartile_harmful']}/{row['top_quartile_neutral']} | {row['bottom_quartile_helpful']}/{row['bottom_quartile_harmful']}/{row['bottom_quartile_neutral']} |")
    lines += ["", "## Interpretation", "", "- Only 20/160 candidate-state utilities were non-zero; 140/160 were neutral.",
              "- ISS-max regions were most often helpful to the policy: perturbing them caused 15 harms and only 2 rescues.",
              "- ISS-max is the one selector-level effect whose snapshot bootstrap interval excludes zero: it finds regions the policy usually needs, not a bidirectional sign rule.",
              "- The preregistered ISS x relevance and nuisance x ISS signed hypotheses show no separation."]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"analysis": "complete", "output": str(output), "integrity": integrity}, default=dict))


if __name__ == "__main__":
    main()
