"""Evaluate whether frozen initial-state features can route L11 vs baselines."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from research.semantic_token_cd.l11_state_probe_protocol import (
    PROTOCOL,
    SEEDS,
    SHORT_TASKS,
    TASKS,
    atomic_json,
    feature_path,
    load_outcomes,
    metadata_path,
)


FEATURE_SETS = {
    "proprio": ("proprio",),
    "visual_global": ("visual_mean", "visual_std"),
    "visual_l11": ("visual_l11_weighted",),
    "prompt_action": ("prompt_hidden_mean", "action_context_hidden"),
    "all_state": (
        "proprio",
        "visual_mean",
        "visual_std",
        "visual_l11_weighted",
        "prompt_hidden_mean",
        "action_context_hidden",
    ),
}


def load_records(artifact: Path, paired_results: Path) -> list[dict]:
    outcomes = load_outcomes(paired_results)
    records = []
    for task in TASKS:
        for seed in SEEDS:
            path = feature_path(artifact, task, seed)
            meta_path = metadata_path(artifact, task, seed)
            if not path.exists():
                raise FileNotFoundError(path)
            if not meta_path.exists():
                raise FileNotFoundError(meta_path)
            metadata = json.loads(meta_path.read_text())
            if metadata.get("protocol_id") != PROTOCOL:
                raise RuntimeError(f"protocol mismatch: {meta_path}")
            with np.load(path) as arrays:
                features = {
                    name: np.asarray(arrays[name], dtype=np.float32).reshape(-1)
                    for names in FEATURE_SETS.values() for name in names
                }
            records.append({
                "task": task,
                "task_short": SHORT_TASKS[task],
                "seed": seed,
                "features": features,
                **outcomes[(task, seed)],
            })
    return records


def matrix(records: list[dict], feature_set: str) -> np.ndarray:
    names = FEATURE_SETS[feature_set]
    payload = np.stack([
        np.concatenate([record["features"][name] for name in names])
        for record in records
    ])
    if not np.isfinite(payload).all():
        raise FloatingPointError(f"non-finite matrix for {feature_set}")
    return payload


def fit_predict_state(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    varying = np.std(train_x, axis=0) > 1e-6
    if not np.any(varying):
        return np.full(test_x.shape[0], float(np.mean(train_y)), dtype=np.float64)
    train_x = train_x[:, varying]
    test_x = test_x[:, varying]
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x)
    test_scaled = scaler.transform(test_x)
    components = min(32, train_scaled.shape[0] - 2, train_scaled.shape[1])
    if components < 2:
        raise RuntimeError("not enough training examples for PCA")
    pca = PCA(n_components=components, svd_solver="randomized", random_state=20260913)
    train_reduced = pca.fit_transform(train_scaled)
    test_reduced = pca.transform(test_scaled)
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=3000,
        random_state=20260913,
    )
    classifier.fit(train_reduced, train_y)
    return classifier.predict_proba(test_reduced)[:, 1]


def fit_predict_task(train_tasks: list[str], train_y: np.ndarray, test_tasks: list[str]) -> np.ndarray:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_x = encoder.fit_transform(np.asarray(train_tasks, dtype=object)[:, None])
    test_x = encoder.transform(np.asarray(test_tasks, dtype=object)[:, None])
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=20260913,
    )
    classifier.fit(train_x, train_y)
    return classifier.predict_proba(test_x)[:, 1]


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, scores >= 0.5)),
        "count": int(labels.size),
        "positive_count": int(labels.sum()),
        "negative_count": int(labels.size - labels.sum()),
    }


def discordant(records: list[dict], baseline: str) -> tuple[list[dict], np.ndarray]:
    selected = [record for record in records if record["l11_matched"] != record[baseline]]
    labels = np.asarray([record["l11_matched"] for record in selected], dtype=np.int64)
    if len(np.unique(labels)) != 2:
        raise RuntimeError(f"{baseline} discordant set has only one class")
    return selected, labels


def pooled_oof(records: list[dict], labels: np.ndarray, feature_set: str | None) -> np.ndarray:
    groups = np.asarray([record["seed"] for record in records])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260913)
    scores = np.full(labels.shape, np.nan, dtype=np.float64)
    features = None if feature_set is None else matrix(records, feature_set)
    placeholder = np.zeros((len(records), 1), dtype=np.float32)
    for train, test in splitter.split(placeholder, labels, groups):
        if feature_set is None:
            fold_scores = fit_predict_task(
                [records[index]["task"] for index in train], labels[train],
                [records[index]["task"] for index in test],
            )
        else:
            fold_scores = fit_predict_state(features[train], labels[train], features[test])
        scores[test] = fold_scores
    if not np.isfinite(scores).all():
        raise RuntimeError("incomplete pooled OOF predictions")
    return scores


def within_task_oof(records: list[dict], labels: np.ndarray, feature_set: str) -> dict:
    features = matrix(records, feature_set)
    result = {}
    for task in TASKS:
        indices = np.asarray([index for index, record in enumerate(records) if record["task"] == task])
        task_labels = labels[indices]
        minority = int(min(task_labels.sum(), task_labels.size - task_labels.sum()))
        if minority < 2:
            result[SHORT_TASKS[task]] = None
            continue
        splits = min(5, minority)
        splitter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=20260913)
        scores = np.full(task_labels.shape, np.nan, dtype=np.float64)
        for train_local, test_local in splitter.split(np.zeros((len(indices), 1)), task_labels):
            scores[test_local] = fit_predict_state(
                features[indices[train_local]], task_labels[train_local], features[indices[test_local]]
            )
        result[SHORT_TASKS[task]] = metrics(task_labels, scores)
    return result


def leave_one_task_out(records: list[dict], labels: np.ndarray, feature_set: str) -> dict:
    features = matrix(records, feature_set)
    result = {}
    for task in TASKS:
        test = np.asarray([index for index, record in enumerate(records) if record["task"] == task])
        train = np.asarray([index for index, record in enumerate(records) if record["task"] != task])
        if len(np.unique(labels[test])) < 2 or len(np.unique(labels[train])) < 2:
            result[SHORT_TASKS[task]] = None
            continue
        scores = fit_predict_state(features[train], labels[train], features[test])
        result[SHORT_TASKS[task]] = metrics(labels[test], scores)
    return result


def router_success(records: list[dict], baseline: str, discordant_records: list[dict], scores: np.ndarray) -> dict:
    score_by_key = {
        (record["task"], record["seed"]): float(score)
        for record, score in zip(discordant_records, scores)
    }
    successes = 0
    choices = Counter()
    for record in records:
        key = (record["task"], record["seed"])
        use_l11 = score_by_key.get(key, 0.5) >= 0.5
        if record["l11_matched"] == record[baseline]:
            use_l11 = True
        chosen = "l11_matched" if use_l11 else baseline
        successes += int(record[chosen])
        choices[chosen] += 1
    return {
        "successes": successes,
        "episodes": len(records),
        "success_rate": successes / len(records),
        "choices": dict(choices),
        "l11_fixed_successes": sum(int(record["l11_matched"]) for record in records),
        "baseline_fixed_successes": sum(int(record[baseline]) for record in records),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--paired-results", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    records = load_records(artifact, args.paired_results.resolve())

    report = {"protocol_id": PROTOCOL, "state_count": len(records), "baselines": {}}
    prediction_rows = []
    for baseline in ("shr", "vanilla"):
        comparison, labels = discordant(records, baseline)
        task_scores = pooled_oof(comparison, labels, None)
        task_metrics = metrics(labels, task_scores)
        feature_results = {}
        for feature_set in FEATURE_SETS:
            state_scores = pooled_oof(comparison, labels, feature_set)
            pooled = metrics(labels, state_scores)
            within = within_task_oof(comparison, labels, feature_set)
            cross = leave_one_task_out(comparison, labels, feature_set)
            cross_aucs = [item["auc"] for item in cross.values() if item is not None]
            go = {
                "auc_at_least_0_65": pooled["auc"] >= 0.65,
                "beats_task_only_by_0_05": pooled["auc"] >= task_metrics["auc"] + 0.05,
                "cross_task_median_at_least_0_60": bool(cross_aucs) and float(np.median(cross_aucs)) >= 0.60,
            }
            go["passed"] = all(go.values())
            feature_results[feature_set] = {
                "pooled_seed_grouped_oof": pooled,
                "within_task_oof": within,
                "leave_one_task_out": cross,
                "leave_one_task_out_median_auc": float(np.median(cross_aucs)) if cross_aucs else None,
                "router_replay": router_success(records, baseline, comparison, state_scores),
                "go_no_go": go,
            }
            for record, label, score in zip(comparison, labels, state_scores):
                prediction_rows.append({
                    "baseline": baseline,
                    "feature_set": feature_set,
                    "task": record["task_short"],
                    "seed": record["seed"],
                    "label_prefer_l11": int(label),
                    "score_prefer_l11": float(score),
                })
        report["baselines"][baseline] = {
            "discordant_count": len(comparison),
            "label_counts": {
                "prefer_l11": int(labels.sum()),
                f"prefer_{baseline}": int(labels.size - labels.sum()),
            },
            "task_only_seed_grouped_oof": task_metrics,
            "task_only_router_replay": router_success(records, baseline, comparison, task_scores),
            "feature_sets": feature_results,
        }

    output = artifact / "analysis"
    atomic_json(output / "STATE_PROBE_RESULTS.json", report)
    write_csv(output / "oof_predictions.csv", prediction_rows)
    lines = [
        "# L11 Initial-State Representation Probe",
        "",
        "Episode outcomes are used only as episode-level routing labels. No timestep receives a copied Harm/Rescue label.",
        "",
    ]
    for baseline, baseline_result in report["baselines"].items():
        task_auc = baseline_result["task_only_seed_grouped_oof"]["auc"]
        lines.extend([
            f"## L11 vs {baseline.upper()}",
            "",
            f"- Discordant episodes: **{baseline_result['discordant_count']}**",
            f"- Task-only grouped OOF AUC: **{task_auc:.3f}**",
            "",
            "| Feature set | Grouped OOF AUC | Balanced acc. | LOTO median AUC | Replay success | GO |",
            "|---|---:|---:|---:|---:|:---:|",
        ])
        for feature_set, result in baseline_result["feature_sets"].items():
            pooled = result["pooled_seed_grouped_oof"]
            loto = result["leave_one_task_out_median_auc"]
            replay = result["router_replay"]
            lines.append(
                f"| {feature_set} | {pooled['auc']:.3f} | {pooled['balanced_accuracy']:.3f} | "
                f"{loto:.3f} | {replay['successes']}/{replay['episodes']} | "
                f"{'PASS' if result['go_no_go']['passed'] else 'STOP'} |"
            )
        lines.append("")
    (output / "STATE_PROBE_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
